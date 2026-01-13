from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from models.local_facial_extractor import PerceiverCrossAttention
from torch import nn

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.normalization import CogVideoXLayerNormZero


# 1. The first difference is the way of obtaining local features
def process_face_embeddings(face_imgs, clip_face_pre_values, face_main_model, face_clip_model):
    # face_clip_model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # face_main_model = get_model('IR_101')([112, 112])
    # face_imgs              [B, self.max_frames, C=3, H=112, W=112]
    # clip_face_pre_values   [B, self.max_frames, H=224, W=224]
    B, T = face_imgs.shape[:2]

    valid_face_imgs = []
    valid_pixel_values = []
    B_valid_num = torch.zeros(B, dtype=torch.int32)

    for i in range(B):
        valid_mask = torch.any(torch.any(torch.any(face_imgs[i] != 0, dim=1), dim=1), dim=1)
        B_valid_num[i] = valid_mask.sum()
        valid_face_imgs.extend(face_imgs[i][valid_mask])
        valid_pixel_values.extend(clip_face_pre_values[i][valid_mask])

    valid_face_imgs_tensor = torch.stack(valid_face_imgs)
    valid_pixel_values_tensor = torch.stack(valid_pixel_values)

    _, mid_embedding = face_main_model(valid_face_imgs_tensor,
                                       return_mid_feats=True)  # mid_embedding: torch.Size([1, 64, 56, 56]) torch.Size([1, 128, 28, 28]) torch.Size([1, 256, 14, 14]) torch.Size([1, 512, 7, 7])

    intrinsic_id_embeds = mid_embedding[-1]  # intrinsic_id_embeds is (b, d=512, l=7*7)
    intrinsic_id_embeds = intrinsic_id_embeds.reshape(intrinsic_id_embeds.shape[0], intrinsic_id_embeds.shape[1],
                                                      -1).permute(0, 2, 1)  # intrinsic_id_embeds is (b, l=7*7, d=512)

    image_embeds = face_clip_model(valid_pixel_values_tensor, output_hidden_states=True).hidden_states[
        -2]  # torch.Size([6, 257, 1280])

    temp_structure_embeds = image_embeds[:, 1:, :]  # structure_embeds is (b, l=16*16, d=1280)

    target_shape = (16, 16)

    # Step 1: (x1, x2) -> (16, 16)
    mid_embedding_64_resized = F.interpolate(mid_embedding[0], size=target_shape, mode='bilinear', align_corners=False)
    mid_embedding_128_resized = F.interpolate(mid_embedding[1], size=target_shape, mode='bilinear', align_corners=False)
    mid_embedding_256_resized = F.interpolate(mid_embedding[2], size=target_shape, mode='bilinear', align_corners=False)

    # Step 2: (16, 16) -> (16*16=256)
    mid_embedding_64_flat = mid_embedding_64_resized.reshape(mid_embedding_64_resized.size(0),
                                                             mid_embedding_64_resized.size(1), -1)  # [b, c, 256]
    mid_embedding_128_flat = mid_embedding_128_resized.reshape(mid_embedding_128_resized.size(0),
                                                               mid_embedding_128_resized.size(1), -1)  # [b, c, 256]
    mid_embedding_256_flat = mid_embedding_256_resized.reshape(mid_embedding_256_resized.size(0),
                                                               mid_embedding_256_resized.size(1), -1)  # [b, c, 256]

    # Step 3: [b, l, 256] -> [b, 256, l]
    mid_embedding_64_swapped = mid_embedding_64_flat.permute(0, 2, 1)  # [b, 256, 64]
    mid_embedding_128_swapped = mid_embedding_128_flat.permute(0, 2, 1)  # [b, 256, 128]
    mid_embedding_256_swapped = mid_embedding_256_flat.permute(0, 2, 1)  # [b, 256, 256]

    # Step 4: concatenate
    # structure_embeds is (b, l=16*16, d=64+128+256+1280)
    structure_embeds = torch.cat(
        [mid_embedding_64_swapped, mid_embedding_128_swapped, mid_embedding_256_swapped, temp_structure_embeds],
        dim=2)  # [b, 256, 64+128+256+1280]

    return structure_embeds, intrinsic_id_embeds, B_valid_num

# 2. The second difference is the injection form of local features
class ConsisIDBlock(nn.Module):
    r"""

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
        cross_LFE_final_output_dim: int = None,
        cross_inner_dim: int = None,
        cross_num_ca: int = 2,
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

        # 3. Injection Module
        device = self.device
        weight_dtype = next(self.ff.parameters()).dtype
        self.perceiver_cross_attention = nn.ModuleList([
            PerceiverCrossAttention(dim=cross_inner_dim, dim_head=128, heads=16, kv_dim=cross_LFE_final_output_dim).to(device, dtype=weight_dtype) for _ in range(cross_num_ca)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_idx: Optional[int] = None,
        local_face_scale: Optional[float] = None,
        valid_face_emb: Optional[torch.Tensor]= None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # inject local facial feature
        hidden_states = hidden_states + local_face_scale * self.perceiver_cross_attention[ca_idx](valid_face_emb, hidden_states)

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

# 3. The third difference is the composition of low-frequency facial features. The current codes does not use kps.
