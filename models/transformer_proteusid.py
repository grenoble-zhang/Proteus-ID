import glob
import json
import math
import os
from typing import Any, List, Dict, Optional, Tuple, Union
from collections import OrderedDict

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PromptSelfAttention(nn.Module):
    """
    Self-attention module for processing prompt tokens with layer normalization.
    
    This module applies self-attention over input latent tokens, enabling 
    the model to capture dependencies between different positions in the sequence.
    
    Args:
        dim: The input/output feature dimension.
        dim_head: The dimension of each attention head.
        heads: The number of attention heads.
        kv_dim: Optional dimension for key/value projections, defaults to dim.
    """
    
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 64, kv_dim: Optional[int] = None):
        super().__init__()

        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            latents: Input tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim) after self-attention.
        """
        latents = self.norm(latents)

        batch_size, seq_len, _ = latents.shape

        query = self.to_q(latents)
        key = self.to_k(latents)
        value = self.to_v(latents)

        # Reshape to multi-head format: (batch, heads, seq_len, dim_head)
        query = query.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Scaled dot-product attention with split scaling for numerical stability in fp16
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (query * scale) @ (key * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        output = weight @ value

        # Reshape back: (batch, seq_len, heads * dim_head)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.to_out(output)

class PerceiverAttention(nn.Module):
    """
    Perceiver-style cross-attention module that attends from latent queries to image embeddings.
    
    The key-value pairs are computed from the concatenation of image embeddings and latents,
    allowing the model to capture both cross-modal and self-attention information.
    
    Args:
        dim: The latent/output feature dimension.
        dim_head: The dimension of each attention head.
        heads: The number of attention heads.
        kv_dim: Optional dimension for key/value inputs (image embeddings), defaults to dim.
    """
    
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, kv_dim: Optional[int] = None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, image_embeds: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for perceiver cross-attention.
        
        Args:
            image_embeds: Image embedding tensor of shape (batch_size, img_seq_len, kv_dim).
            latents: Latent query tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        image_embeds = self.norm1(image_embeds)
        latents = self.norm2(latents)

        batch_size, seq_len, _ = latents.shape

        query = self.to_q(latents)
        # Concatenate image embeddings and latents for key-value computation
        kv_input = torch.cat((image_embeds, latents), dim=-2)
        key, value = self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape to multi-head format
        query = query.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Scaled dot-product attention with split scaling for numerical stability
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (query * scale) @ (key * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        output = weight @ value

        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.to_out(output)

class FuseToken(nn.Module):
    """
    Fuses identity prompt embeddings with visual features from CLIP and ViT encoders.
    
    This module uses learnable query tokens that attend to multi-scale visual features
    through a stack of self-attention and cross-attention layers to produce fused tokens
    that capture both textual and visual identity information.
    
    Args:
        text_dim: Dimension of text/prompt embeddings.
        clip_dim: Dimension of CLIP image embeddings.
        vit_dim: Dimension of ViT hidden states.
        num_queries: Number of learnable query tokens.
        depth: Total depth of attention layers (must be divisible by num_scale).
        dim_head: Dimension of each attention head.
        sa_heads: Number of self-attention heads.
        output_dim: Output dimension of fused tokens.
        num_scale: Number of multi-scale ViT features to process.
        ff_mult: Feed-forward expansion multiplier.
    """
    
    def __init__(
        self,
        text_dim: int = 4096,
        clip_dim: int = 1280,
        vit_dim: int = 1024,
        num_queries: int = 32,
        depth: int = 10,
        dim_head: int = 64,
        sa_heads: int = 64,
        output_dim: int = 4096,
        num_scale: int = 5,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.clip_dim = clip_dim
        self.num_queries = num_queries
        
        if depth % num_scale != 0:
            raise ValueError(f"depth ({depth}) must be divisible by num_scale ({num_scale})")
        self.depth = depth // num_scale
        self.num_scale = num_scale
        self.ff_mult = ff_mult
        scale = text_dim**-0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, text_dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(text_dim, output_dim))

        # Attention and feed-forward layer stack
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PromptSelfAttention(dim=text_dim, dim_head=dim_head, heads=sa_heads),
                        PerceiverAttention(dim=text_dim, dim_head=dim_head, heads=sa_heads),
                        nn.Sequential(
                            nn.LayerNorm(text_dim),
                            nn.Linear(text_dim, text_dim, bias=False),
                            nn.GELU(),
                            nn.Linear(text_dim, text_dim, bias=False),
                        ),
                    ]
                )
            )
            
        # Mappings for each of the multi-scale ViT features
        for i in range(num_scale):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim * ff_mult),
                ),
            )
            
        # Map CLIP feature dimension to ViT feature space
        self.img_embedding_mapping = nn.Sequential(
            nn.Linear(clip_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim * ff_mult),
        )

    def forward(
        self, 
        identity_prompt_embeds: torch.Tensor, 
        id_embeds: torch.Tensor, 
        vit_hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass to fuse identity and visual features.
        
        Args:
            identity_prompt_embeds: Identity prompt embeddings of shape (batch, prompt_len, text_dim).
            id_embeds: CLIP identity embeddings of shape (batch, clip_dim).
            vit_hidden_states: List of ViT hidden states at different scales.
            
        Returns:
            Fused token tensor of shape (batch, num_queries, output_dim).
        """
        batch_size = id_embeds.size(0)
        latents = self.latents.repeat(batch_size, 1, 1)
        latents = torch.cat((latents, identity_prompt_embeds), dim=1)

        id_embeds = self.img_embedding_mapping(id_embeds)
        
        for i in range(self.num_scale):
            vit_feature = getattr(self, f"mapping_{i}")(vit_hidden_states[i])
            ctx_feature = torch.cat((id_embeds, vit_feature), dim=1)
            
            for sa, ca, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                latents = sa(latents) + latents
                latents = ca(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        # Extract only the query tokens (exclude concatenated prompt embeddings)
        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents

class FuseAdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with timestep conditioning for fusion modules.
    
    Applies layer normalization with learnable scale and shift parameters
    that are conditioned on timestep embeddings, enabling time-aware feature modulation.
    
    Args:
        embedding_dim: Dimension of the input features.
        time_embedding_dim: Dimension of timestep embeddings, defaults to embedding_dim.
    """
    
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim).
            timestep_embedding: Timestep embedding of shape (batch, 1, time_embedding_dim).
            
        Returns:
            Normalized and modulated tensor of shape (batch, seq_len, embedding_dim).
        """
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class SquaredReLU(nn.Module):
    """Squared ReLU activation function: f(x) = ReLU(x)^2."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.relu(x))

class FuseFaceFeature(nn.Module):
    """
    Fuses CLIP identity embeddings with ViT hidden states to produce face features.
    
    This module maps CLIP embeddings to ViT dimension, concatenates with ViT features,
    and projects to the output dimension for downstream face-related tasks.
    
    Args:
        clip_dim: Dimension of CLIP identity embeddings.
        vit_dim: Dimension of ViT hidden states.
        out_dim: Output dimension of fused face features.
    """
    
    def __init__(
        self,
        clip_dim: int = 1280,
        vit_dim: int = 1024,
        out_dim: int = 2048,
    ):
        super().__init__()
        
        # Map CLIP features to ViT dimension
        self.clip_embedding_mapping = nn.Sequential(
            nn.Linear(clip_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
        )

        # Map concatenated features to output dimension
        self.vit_embedding_mapping = nn.Sequential(
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, out_dim),
        )

    def forward(self, id_embeds: torch.Tensor, vit_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to fuse identity and ViT features.
        
        Args:
            id_embeds: CLIP identity embeddings of shape (batch, seq_len, clip_dim).
            vit_hidden_states: ViT hidden states of shape (batch, vit_seq_len, vit_dim).
            
        Returns:
            Fused face features of shape (batch, total_seq_len, out_dim).
        """
        id_embeds = self.clip_embedding_mapping(id_embeds)
        ctx_feature = torch.cat((id_embeds, vit_hidden_states), dim=1)
        img_feature = self.vit_embedding_mapping(ctx_feature)
        return img_feature

class PerceiverAttentionBlock(nn.Module):
    """
    A perceiver-style attention block with timestep conditioning.
    
    This block applies two sequential cross-attention operations:
    1. Fuse attention: attends to both self and fuse tokens
    2. Image attention: attends to both self and image embeddings
    Followed by a feed-forward network with squared ReLU activation.
    
    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        time_embedding_dim: Dimension of timestep embeddings for adaptive layer norm.
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 32,
        time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn_fuse = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_img = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 2)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 2, d_model)),
                ]
            )
        )

        self.ln_1 = FuseAdaLayerNorm(d_model, time_embedding_dim)
        self.ln_fuse = FuseAdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = FuseAdaLayerNorm(d_model, time_embedding_dim)
        self.ln_img = FuseAdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = FuseAdaLayerNorm(d_model, time_embedding_dim)

    def fuse_attention(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention for fuse tokens."""
        attn_output, _ = self.attn_fuse(q, kv, kv, need_weights=False)
        return attn_output
    
    def img_attention(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention for image embeddings."""
        attn_output, _ = self.attn_img(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        fuse_token: torch.Tensor,
        img_embedding: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the perceiver attention block.
        
        Args:
            fuse_token: Fused identity tokens of shape (batch, fuse_len, d_model).
            img_embedding: Image embeddings of shape (batch, img_len, d_model).
            latents: Query latents of shape (batch, query_len, d_model).
            timestep_embedding: Timestep embeddings for adaptive normalization.
            
        Returns:
            Updated latents of shape (batch, query_len, d_model).
        """
        # Fuse attention: self + fuse tokens
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.fuse_attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_fuse(fuse_token, timestep_embedding)], dim=1),
        )
        
        # Image attention: self + image embeddings
        normed_latents = self.ln_2(latents, timestep_embedding)
        latents = latents + self.img_attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_img(img_embedding, timestep_embedding)], dim=1),
        )

        # Feed-forward
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents

class PerceiverResampler(nn.Module):
    """
    Perceiver-based resampler that produces fixed-length query outputs from variable inputs.
    
    Uses learnable query tokens that attend to both fused identity tokens and image embeddings
    through a stack of perceiver attention blocks with timestep conditioning.
    
    Args:
        num_queries: Number of output query tokens.
        query_dim: Dimension of query tokens and internal representations.
        input_img_dim: Dimension of input image embeddings.
        input_fuse_dim: Dimension of input fused tokens.
        output_dim: Output dimension after final projection.
        layers: Number of perceiver attention blocks.
        heads: Number of attention heads per block.
        time_embedding_dim: Dimension of timestep embeddings.
    """
    
    def __init__(
        self,
        num_queries: int = 32,
        query_dim: int = 2048,
        input_img_dim: int = 2048,
        input_fuse_dim: int = 4096,
        output_dim: int = 2048,
        layers: int = 10,
        heads: int = 32,
        time_embedding_dim: int = 2048,
    ):
        super().__init__()
        scale = query_dim**-0.5
        
        # Learnable latent query embeddings
        self.latents = nn.Parameter(torch.randn(1, num_queries, query_dim) * scale)
        self.time_aware_linear = nn.Linear(time_embedding_dim or query_dim, query_dim, bias=True)

        self.proj_fuse = nn.Linear(input_fuse_dim, query_dim)
        self.proj_img = nn.Linear(input_img_dim, query_dim)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    query_dim, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        self.proj_out = nn.Sequential(
            nn.Linear(query_dim, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(
        self, 
        fuse_token: torch.Tensor, 
        img_embedding: torch.Tensor, 
        timestep_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the perceiver resampler.
        
        Args:
            fuse_token: Fused identity tokens of shape (batch, fuse_len, input_fuse_dim).
            img_embedding: Image embeddings of shape (batch, img_len, input_img_dim).
            timestep_embedding: Timestep embeddings for time-aware processing.
            
        Returns:
            Resampled output of shape (batch, num_queries, output_dim).
        """
        batch_size = fuse_token.size(0)
        latents = self.latents.repeat(batch_size, 1, 1)
        
        # Add time-aware conditioning to latents
        latents = latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        
        # Project inputs to query dimension
        fuse_token = self.proj_fuse(fuse_token)
        img_embedding = self.proj_img(img_embedding)
        
        # Process through perceiver blocks
        for p_block in self.perceiver_blocks:
            latents = p_block(fuse_token, img_embedding, latents, timestep_embedding)

        latents = self.proj_out(latents)
        return latents

class FuseFacitalExtractor(nn.Module):
    """
    Extracts facial features by fusing identity embeddings with visual features using timestep conditioning.
    
    This module combines fused tokens with face features extracted from CLIP and ViT embeddings,
    processing them through a perceiver resampler with timestep-aware conditioning for
    diffusion model integration.
    
    Args:
        num_queries: Number of output query tokens.
        query_dim: Dimension of internal query representations.
        input_img_dim: Dimension of image features from face feature fusion.
        input_fuse_dim: Dimension of input fused tokens.
        output_dim: Output dimension of extracted features.
        layers: Number of perceiver attention layers.
        heads: Number of attention heads.
        clip_dim: Dimension of CLIP identity embeddings.
        vit_dim: Dimension of ViT hidden states.
        time_channel: Number of channels for timestep sinusoidal encoding.
        time_embed_dim: Dimension of timestep embeddings after projection.
        timestep_activation_fn: Activation function for timestep embedding MLP.
    """
    
    def __init__(
        self,
        num_queries: int = 32,
        query_dim: int = 2048,
        input_img_dim: int = 2048,
        input_fuse_dim: int = 4096,
        output_dim: int = 2048,
        layers: int = 10,
        heads: int = 32,
        clip_dim: int = 1280,
        vit_dim: int = 1024,
        time_channel: int = 3072,
        time_embed_dim: int = 2048,
        timestep_activation_fn: str = "silu",
    ):
        super().__init__()

        scale = query_dim**-0.5
        self.layers = layers
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, query_dim) * scale)
        self.proj_out = nn.Sequential(nn.Linear(query_dim, output_dim), nn.LayerNorm(output_dim))
        
        # Timestep encoding components
        self.position = Timesteps(time_channel, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel, 
            time_embed_dim=time_embed_dim, 
            act_fn=timestep_activation_fn
        )
        
        # Face feature fusion and perceiver resampler
        self.fuse_face_feature = FuseFaceFeature(clip_dim=clip_dim, vit_dim=vit_dim, out_dim=input_img_dim)
        self.connector = PerceiverResampler(
            num_queries=num_queries,
            query_dim=query_dim,
            input_img_dim=input_img_dim,
            input_fuse_dim=input_fuse_dim,
            output_dim=output_dim,
            layers=layers,
            heads=heads,
            time_embedding_dim=time_embed_dim,
        )

    def forward(
        self, 
        fuse_token: torch.Tensor, 
        id_embeds: torch.Tensor, 
        vit_hidden_states: List[torch.Tensor], 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract facial features with timestep conditioning.
        
        Args:
            fuse_token: Fused identity tokens of shape (batch, fuse_len, input_fuse_dim).
            id_embeds: CLIP identity embeddings of shape (batch, clip_seq_len, clip_dim).
            vit_hidden_states: List of ViT hidden states at different scales.
            timesteps: Diffusion timesteps tensor.
            
        Returns:
            Extracted facial features of shape (batch, num_queries, output_dim).
        """
        device = fuse_token.device
        dtype = fuse_token.dtype
        batch_size = len(fuse_token)

        # Compute timestep embeddings
        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        if ori_time_feature.ndim == 2:
            ori_time_feature = ori_time_feature.unsqueeze(dim=1)
        ori_time_feature = ori_time_feature.expand(batch_size, -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        # Extract and fuse face features (using the last scale ViT features)
        face_feature = self.fuse_face_feature(id_embeds, vit_hidden_states[-1])
        
        # Process through perceiver resampler
        latents = self.connector(
            fuse_token=fuse_token, 
            img_embedding=face_feature, 
            timestep_embedding=time_embedding
        )

        return latents

class LocalFacialExtractor(nn.Module):
    """
    Extracts local facial features by fusing identity embeddings with multi-scale ViT features.
    
    This module uses learnable query tokens that attend to multi-scale visual features
    through perceiver-style cross-attention, producing high-frequency facial detail embeddings.
    
    Args:
        id_dim: Dimension of identity (CLIP) embeddings.
        vit_dim: Dimension of ViT hidden states.
        depth: Total depth of attention layers (must be divisible by num_scale).
        dim_head: Dimension of each attention head.
        heads: Number of attention heads.
        num_id_token: Number of identity tokens to generate from id embedding.
        num_queries: Number of learnable query tokens.
        output_dim: Output dimension of extracted features.
        ff_mult: Feed-forward expansion multiplier.
        num_scale: Number of multi-scale ViT features to process.
    """
    
    def __init__(
        self,
        id_dim: int = 1280,
        vit_dim: int = 1024,
        depth: int = 10,
        dim_head: int = 64,
        heads: int = 16,
        num_id_token: int = 5,
        num_queries: int = 32,
        output_dim: int = 2048,
        ff_mult: int = 4,
        num_scale: int = 5,
    ):
        super().__init__()

        self.num_id_token = num_id_token
        self.vit_dim = vit_dim
        self.num_queries = num_queries
        
        if depth % num_scale != 0:
            raise ValueError(f"depth ({depth}) must be divisible by num_scale ({num_scale})")
        self.depth = depth // num_scale
        self.num_scale = num_scale
        scale = vit_dim**-0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, vit_dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(vit_dim, output_dim))

        # Attention and ProteusIDFeedForward layer stack
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=vit_dim, dim_head=dim_head, heads=heads),
                        nn.Sequential(
                            nn.LayerNorm(vit_dim),
                            nn.Linear(vit_dim, vit_dim * ff_mult, bias=False),
                            nn.GELU(),
                            nn.Linear(vit_dim * ff_mult, vit_dim, bias=False),
                        ),
                    ]
                )
            )

        # Mappings for each of the 5 different ViT features
        for i in range(num_scale):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.LeakyReLU(),
                    nn.Linear(vit_dim, vit_dim),
                ),
            )

        # Mapping for identity embedding vectors
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(id_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.LeakyReLU(),
            nn.Linear(vit_dim, vit_dim * num_id_token),
        )

    def forward(self, id_embeds: torch.Tensor, vit_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract local facial features from identity embeddings and multi-scale ViT features.
        
        Args:
            id_embeds: Identity embeddings of shape (batch, id_dim).
            vit_hidden_states: List of ViT hidden states at different scales.
            
        Returns:
            Extracted facial features of shape (batch, num_queries, output_dim).
        """
        batch_size = id_embeds.size(0)
        latents = self.latents.repeat(batch_size, 1, 1)

        # Map identity embeddings to multiple tokens
        id_embeds = self.id_embedding_mapping(id_embeds)
        id_embeds = id_embeds.reshape(-1, self.num_id_token, self.vit_dim)

        latents = torch.cat((latents, id_embeds), dim=1)

        # Process through multi-scale attention layers
        for i in range(self.num_scale):
            vit_feature = getattr(self, f"mapping_{i}")(vit_hidden_states[i])
            ctx_feature = torch.cat((id_embeds, vit_feature), dim=1)

            for attn, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        # Extract only query tokens and project to output dimension
        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents

class FusedTokenAdapter(nn.Module):
    """
    Adapter module to transform fused tokens through a multi-layer MLP.
    
    This module applies a series of linear transformations with layer normalization
    and LeakyReLU activations to adapt fused identity tokens for downstream processing.
    
    Args:
        in_dim: Input dimension of fused tokens.
        hidden_dim: Hidden dimension of the MLP.
        out_dim: Output dimension of adapted tokens.
        use_residual: Whether to add residual connection from input to output.
    """
    
    def __init__(
        self,
        in_dim: int = 4096,
        hidden_dim: int = 4096,
        out_dim: int = 4096,
        use_residual: bool = False,
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Multi-layer MLP to adapt fused features
        self.feature_mapping = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, fused_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            fused_token: Input fused tokens of shape (batch, seq_len, in_dim).
            
        Returns:
            Adapted tokens of shape (batch, seq_len, out_dim).
        """
        adapted_token = self.feature_mapping(fused_token)
        
        if self.use_residual:
            adapted_token = fused_token + adapted_token
        
        return adapted_token

class PerceiverCrossAttention(nn.Module):
    def __init__(self, dim: int = 3072, dim_head: int = 128, heads: int = 16, kv_dim: int = 2048):
        super().__init__()

        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, image_embeds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply layer normalization to the input image and latent features
        image_embeds = self.norm1(image_embeds)
        hidden_states = self.norm2(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        # Compute queries, keys, and values
        query = self.to_q(hidden_states)
        key, value = self.to_kv(image_embeds).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        query = query.reshape(query.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(key.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.reshape(value.size(0), -1, self.heads, self.dim_head).transpose(1, 2)

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (query * scale) @ (key * scale).transpose(-2, -1)  # More stable scaling than post-division
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ value

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.to_out(out)

@maybe_allow_in_graph
class ProteusIDBlock(nn.Module):
    r"""
    Transformer block used in [ProteusID](https://github.com/PKU-YuanGroup/ProteusID) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class ProteusIDTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [ProteusID](https://github.com/PKU-YuanGroup/ProteusID).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2V in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because ProteusID processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
            
        is_kps (`bool`, defaults to `False`):
            Whether to enable keypoint for global facial extractor. If `True`, keypoints will be in the model.
        cross_attn_interval (`int`, defaults to `2`):
            The interval between cross-attention layers in the Transformer architecture. A larger value may reduce the
            frequency of cross-attention computations, which can help reduce computational overhead.
        cross_attn_dim_head (`int`, optional, defaults to `128`):
            The dimensionality of each attention head in the cross-attention layers of the Transformer architecture. A
            larger value increases the capacity to attend to more complex patterns, but also increases memory and
            computation costs.
        cross_attn_num_heads (`int`, optional, defaults to `16`):
            The number of attention heads in the cross-attention layers. More heads allow for more parallel attention
            mechanisms, capturing diverse relationships between different components of the input, but can also
            increase computational requirements.
        LFE_id_dim (`int`, optional, defaults to `1280`):
            The dimensionality of the identity vector used in the Local Facial Extractor (LFE). This vector represents
            the identity features of a face, which are important for tasks like face recognition and identity
            preservation across different frames.
        LFE_vit_dim (`int`, optional, defaults to `1024`):
            The dimension of the vision transformer (ViT) output used in the Local Facial Extractor (LFE). This value
            dictates the size of the transformer-generated feature vectors that will be processed for facial feature
            extraction.
        LFE_depth (`int`, optional, defaults to `10`):
            The number of layers in the Local Facial Extractor (LFE). Increasing the depth allows the model to capture
            more complex representations of facial features, but also increases the computational load.
        LFE_dim_head (`int`, optional, defaults to `64`):
            The dimensionality of each attention head in the Local Facial Extractor (LFE). This parameter affects how
            finely the model can process and focus on different parts of the facial features during the extraction
            process.
        LFE_num_heads (`int`, optional, defaults to `16`):
            The number of attention heads in the Local Facial Extractor (LFE). More heads can improve the model's
            ability to capture diverse facial features, but at the cost of increased computational complexity.
        LFE_num_id_token (`int`, optional, defaults to `5`):
            The number of identity tokens used in the Local Facial Extractor (LFE). This defines how many
            identity-related tokens the model will process to ensure face identity preservation during feature
            extraction.
        LFE_num_querie (`int`, optional, defaults to `32`):
            The number of query tokens used in the Local Facial Extractor (LFE). These tokens are used to capture
            high-frequency face-related information that aids in accurate facial feature extraction.
        LFE_output_dim (`int`, optional, defaults to `2048`):
            The output dimension of the Local Facial Extractor (LFE). This dimension determines the size of the feature
            vectors produced by the LFE module, which will be used for subsequent tasks such as face recognition or
            tracking.
        LFE_ff_mult (`int`, optional, defaults to `4`):
            The multiplication factor applied to the feed-forward network's hidden layer size in the Local Facial
            Extractor (LFE). A higher value increases the model's capacity to learn more complex facial feature
            transformations, but also increases the computation and memory requirements.
        LFE_num_scale (`int`, optional, defaults to `5`):
            The number of different scales visual feature. A higher value increases the model's capacity to learn more
            complex facial feature transformations, but also increases the computation and memory requirements.
        local_face_scale (`float`, defaults to `1.0`):
            A scaling factor used to adjust the importance of local facial features in the model. This can influence
            how strongly the model focuses on high frequency face-related content.
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ProteusIDBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        is_kps: bool = False,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        LFE_id_dim: int = 1280,
        LFE_vit_dim: int = 1024,
        LFE_depth: int = 10,
        LFE_dim_head: int = 64,
        LFE_num_heads: int = 16,
        LFE_num_id_token: int = 5,
        LFE_num_querie: int = 32,
        LFE_output_dim: int = 2048,
        LFE_ff_mult: int = 4,
        LFE_num_scale: int = 5,
        local_face_scale: float = 1.0,
        FT_txt_dim: int = 4096,
        FT_clip_dim: int = 1280,
        FT_vit_dim: int = 1024,
        FT_num_querie: int = 32,
        FT_depth: int = 10,
        FT_dim_head: int = 64,
        FT_num_heads: int = 64,
        FT_output_dim: int = 4096,
        FT_num_scale: int = 5,
        FT_ff_mult: int = 4,
        FFE_num_queries: int = 32,
        FFE_query_dim: int = 2048,
        FFE_input_img_dim: int = 2048,
        FFE_input_fuse_dim: int = 4096,
        FFE_output_dim: int = 2048,
        FFE_layers: int = 10,
        FFE_heads: int = 32,
        FFE_clip_dim: int = 1280,
        FFE_vit_dim: int = 1024,
        FFE_time_channel: int = 3072,
        FFE_time_embed_dim: int = 2048,
        FFE_timestep_activation_fn: str = "silu",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no ProteusID checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                ProteusIDBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For Version 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For Version 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.is_kps = is_kps

        self.FT_txt_dim = FT_txt_dim
        self.FT_clip_dim = FT_clip_dim
        self.FT_vit_dim = FT_vit_dim
        self.FT_num_querie = FT_num_querie
        self.FT_depth = FT_depth
        self.FT_dim_head = FT_dim_head
        self.FT_num_heads = FT_num_heads
        self.FT_output_dim = FT_output_dim
        self.FT_num_scale = FT_num_scale
        self.FT_ff_mult = FT_ff_mult
        self._init_fuse_token()

        self.LFE_id_dim = LFE_id_dim
        self.LFE_vit_dim = LFE_vit_dim
        self.LFE_depth = LFE_depth
        self.LFE_dim_head = LFE_dim_head
        self.LFE_num_heads = LFE_num_heads
        self.LFE_num_id_token = LFE_num_id_token
        self.LFE_num_querie = LFE_num_querie
        self.LFE_output_dim = LFE_output_dim
        self.LFE_ff_mult = LFE_ff_mult
        self.LFE_num_scale = LFE_num_scale
        self.inner_dim = inner_dim
        self.cross_attn_interval = cross_attn_interval
        self.num_cross_attn = num_layers // cross_attn_interval
        self.cross_attn_dim_head = cross_attn_dim_head
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_kv_dim = int(self.inner_dim / 3 * 2)
        self.local_face_scale = local_face_scale
        
        self.FFE_num_queries = FFE_num_queries
        self.FFE_query_dim = FFE_query_dim
        self.FFE_input_img_dim = FFE_input_img_dim
        self.FFE_input_fuse_dim = FFE_input_fuse_dim
        self.FFE_output_dim = FFE_output_dim
        self.FFE_layers = FFE_layers
        self.FFE_heads = FFE_heads
        self.FFE_clip_dim = FFE_clip_dim
        self.FFE_vit_dim = FFE_vit_dim
        self.FFE_time_channel = FFE_time_channel
        self.FFE_time_embed_dim = FFE_time_embed_dim
        self.FFE_timestep_activation_fn = FFE_timestep_activation_fn
        self._init_facial_attention()

        self.gradient_checkpointing = False

    def _init_fuse_token(self):
        """Initialize the fuse token former and adapter modules."""
        device = self.device
        weight_dtype = self.dtype
        self.fuse_token_former = FuseToken(
            text_dim=self.FT_txt_dim,
            clip_dim=self.FT_clip_dim,
            vit_dim=self.FT_vit_dim,
            num_queries=self.FT_num_querie,
            depth=self.FT_depth,
            dim_head=self.FT_dim_head,
            sa_heads=self.FT_num_heads,
            output_dim=self.FT_output_dim,
            num_scale=self.FT_num_scale,
            ff_mult=self.FT_ff_mult,
        ).to(device, dtype=weight_dtype)
        self.fuse_token_adapter = FusedTokenAdapter(
            in_dim=self.FT_output_dim,
            hidden_dim=self.FT_output_dim,
            out_dim=self.FT_output_dim,
            use_residual=False
        ).to(device, dtype=weight_dtype)

    def _init_perceiver_cross_attention(self):
        """Initialize perceiver cross-attention modules for face feature injection."""
        device = self.device
        weight_dtype = self.dtype
        self.perceiver_cross_attention = nn.ModuleList(
            [
                PerceiverCrossAttention(
                    dim=self.inner_dim,
                    dim_head=self.cross_attn_dim_head,
                    heads=self.cross_attn_num_heads,
                    kv_dim=self.cross_attn_kv_dim,
                ).to(device, dtype=weight_dtype)
                for _ in range(self.num_cross_attn)
            ]
        )

    def _init_face_inputs(self):
        """Initialize local facial extractor (alternative to fuse facial attention)."""
        device = self.device
        weight_dtype = self.dtype
        self.local_facial_extractor = LocalFacialExtractor(
            id_dim=self.LFE_id_dim,
            vit_dim=self.LFE_vit_dim,
            depth=self.LFE_depth,
            dim_head=self.LFE_dim_head,
            heads=self.LFE_num_heads,
            num_id_token=self.LFE_num_id_token,
            num_queries=self.LFE_num_querie,
            output_dim=self.LFE_output_dim,
            ff_mult=self.LFE_ff_mult,
            num_scale=self.LFE_num_scale,
        ).to(device, dtype=weight_dtype)
        self._init_perceiver_cross_attention()

    def _init_facial_attention(self):
        """Initialize the fuse facial attention extractor and perceiver cross-attention modules."""
        device = self.device
        weight_dtype = self.dtype
        self.fuse_facital_attention = FuseFacitalExtractor(
            num_queries=self.FFE_num_queries,
            query_dim=self.FFE_query_dim,
            input_img_dim=self.FFE_input_img_dim,
            input_fuse_dim=self.FFE_input_fuse_dim,
            output_dim=self.FFE_output_dim,
            layers=self.FFE_layers,
            heads=self.FFE_heads,
            clip_dim=self.FFE_clip_dim,
            vit_dim=self.FFE_vit_dim,
            time_channel=self.FFE_time_channel,
            time_embed_dim=self.FFE_time_embed_dim,
            timestep_activation_fn=self.FFE_timestep_activation_fn
        ).to(device, dtype=weight_dtype)
        self._init_perceiver_cross_attention()
    def _load_perceiver_cross_attention(self, checkpoint):
        for ca, state_dict in zip(self.perceiver_cross_attention, checkpoint["perceiver_cross_attention"]):
            ca.load_state_dict(state_dict)

    def save_face_modules(self, path: str):
        """
        Save face-related modules (fuse token former, facial attention, cross-attention) to file.
        
        Args:
            path: File path to save the checkpoint.
        """
        save_dict = {
            "fuse_token_former": self.fuse_token_former.state_dict(),
            "fuse_facital_attention": self.fuse_facital_attention.state_dict(),
            "perceiver_cross_attention": [ca.state_dict() for ca in self.perceiver_cross_attention],
        }
        torch.save(save_dict, path)

    def load_face_modules(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.fuse_token_former.load_state_dict(checkpoint["fuse_token_former"])
        self.fuse_facital_attention.load_state_dict(checkpoint["fuse_facital_attention"])
        self._load_perceiver_cross_attention(checkpoint)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def combine_with_fused_tokens(
        self, 
        encoder_hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor, 
        fused_token: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepend fused tokens to encoder hidden states while preserving sequence length.
        
        This method inserts fused_token at the beginning of encoder_hidden_states,
        keeping only the valid tokens (indicated by attention_mask) that fit after
        the fused tokens, maintaining the original sequence length.
        
        Args:
            encoder_hidden_states: Original token representations of shape (B, L, D).
            attention_mask: Validity mask for original tokens of shape (B, L), where 1 indicates valid.
            fused_token: Fused tokens to prepend of shape (B, F, D).
            
        Returns:
            Combined hidden states of shape (B, L, D) with fused tokens at the front.
        """
        B, L, D = encoder_hidden_states.shape
        F = fused_token.shape[1]
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        
        # Calculate how many valid tokens we can keep (L - F positions available after fused tokens)
        max_valid_tokens = L - F
        
        # Initialize output tensor with fused tokens at the front
        new_hidden_states = torch.zeros((B, L, D), device=device, dtype=dtype)
        new_hidden_states[:, :F] = fused_token
        
        # Process each batch item to copy valid tokens
        # Note: This loop is necessary due to variable-length valid sequences per batch item
        for i in range(B):
            valid_mask = attention_mask[i] == 1
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            # Limit to max_valid_tokens to fit within remaining sequence length
            num_tokens_to_copy = min(valid_indices.size(0), max_valid_tokens)
            if num_tokens_to_copy > 0:
                valid_indices = valid_indices[:num_tokens_to_copy]
                new_hidden_states[i, F:F + num_tokens_to_copy] = encoder_hidden_states[i, valid_indices]
        
        return new_hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        identity_prompt_embeds: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        id_cond: Optional[torch.Tensor] = None,
        id_vit_hidden: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        is_infer: bool = False,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        timesteps = timestep

        id_cond = id_cond.to(device=hidden_states.device, dtype=hidden_states.dtype)
        id_vit_hidden = [
            tensor.to(device=hidden_states.device, dtype=hidden_states.dtype) for tensor in id_vit_hidden
        ]
        encoder_hidden_states = encoder_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
        identity_prompt_embeds = identity_prompt_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        fused_token = self.fuse_token_former(identity_prompt_embeds, id_cond, id_vit_hidden)
        fused_token = self.fuse_token_adapter(fused_token)
        new_hidden_states = self.combine_with_fused_tokens(encoder_hidden_states, encoder_hidden_mask, fused_token)
        
        valid_face_emb = self.fuse_facital_attention(
            fused_token, id_cond, id_vit_hidden, timesteps[0] if is_infer else timesteps
        )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        hidden_states = self.patch_embed(new_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = new_hidden_states.shape[1]
        new_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, new_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    new_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, new_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=new_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                hidden_states = hidden_states + self.local_face_scale * self.perceiver_cross_attention[ca_idx](
                    valid_face_emb, hidden_states
                )
                ca_idx += 1

        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_cus(cls, pretrained_model_path, subfolder=None, config_path=None, transformer_additional_kwargs={}):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, 'config.json')
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(config_path or pretrained_model_path, 'config.json')

        print(f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ...")

        if not os.path.isfile(config_file):
            raise RuntimeError(f"Configuration file '{config_file}' does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if model.state_dict()['patch_embed.proj.weight'].size() != state_dict['patch_embed.proj.weight'].size():
            new_shape   = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            elif len(new_shape) == 2:
                if model.state_dict()['patch_embed.proj.weight'].size()[1] > state_dict['patch_embed.proj.weight'].size()[1]:
                    model.state_dict()['patch_embed.proj.weight'][:, :state_dict['patch_embed.proj.weight'].size()[1]] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][:, state_dict['patch_embed.proj.weight'].size()[1]:] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']
                else:
                    model.state_dict()['patch_embed.proj.weight'][:, :] = state_dict['patch_embed.proj.weight'][:, :model.state_dict()['patch_embed.proj.weight'].size()[1]]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']
            else:
                if model.state_dict()['patch_embed.proj.weight'].size()[1] > state_dict['patch_embed.proj.weight'].size()[1]:
                    model.state_dict()['patch_embed.proj.weight'][:, :state_dict['patch_embed.proj.weight'].size()[1], :, :] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][:, state_dict['patch_embed.proj.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']
                else:
                    model.state_dict()['patch_embed.proj.weight'][:, :, :, :] = state_dict['patch_embed.proj.weight'][:, :model.state_dict()['patch_embed.proj.weight'].size()[1], :, :]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']

        tmp_state_dict = {}
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model