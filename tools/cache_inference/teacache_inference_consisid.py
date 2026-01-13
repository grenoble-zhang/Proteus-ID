import os
import argparse
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    id_cond: Optional[torch.Tensor] = None,
    id_vit_hidden: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # fuse clip and insightface
    if self.is_train_face:
        assert id_cond is not None and id_vit_hidden is not None
        id_cond = id_cond.to(device=hidden_states.device, dtype=hidden_states.dtype)
        id_vit_hidden = [
            tensor.to(device=hidden_states.device, dtype=hidden_states.dtype) for tensor in id_vit_hidden
        ]
        valid_face_emb = self.local_facial_extractor(
            id_cond, id_vit_hidden
        )  # torch.Size([1, 1280]), list[5](torch.Size([1, 577, 1024]))  ->  torch.Size([1, 32, 2048])

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # 2. Patch embedding
    # torch.Size([1, 226, 4096])   torch.Size([1, 13, 32, 60, 90])
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)  # torch.Size([1, 17776, 3072])
    hidden_states = self.embedding_dropout(hidden_states)  # torch.Size([1, 17776, 3072])

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]  # torch.Size([1, 226, 3072])
    hidden_states = hidden_states[:, text_seq_length:]  # torch.Size([1, 17550, 3072])

    if self.enable_teacache:
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = emb
        self.cnt += 1
        if self.cnt == self.num_steps-1:
            self.cnt = 0   

    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
            encoder_hidden_states += self.previous_residual_encoder
        else:
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()
            # 3. Transformer blocks
            ca_idx = 0
            for i, block in enumerate(self.transformer_blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

                if self.is_train_face:
                    if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                        hidden_states = hidden_states + self.local_face_scale * self.perceiver_cross_attention[ca_idx](
                            valid_face_emb, hidden_states
                        )  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])
                        ca_idx += 1
                        
            self.previous_residual = hidden_states - ori_hidden_states
            self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
    else:
        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if self.is_train_face:
                if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                    hidden_states = hidden_states + self.local_face_scale * self.perceiver_cross_attention[ca_idx](
                        valid_face_emb, hidden_states
                    )  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])
                    ca_idx += 1

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    hidden_states = self.norm_final(hidden_states)
    hidden_states = hidden_states[:, text_seq_length:]

    # 4. Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    # Note: we use `-1` instead of `channels`:
    #   - It is okay to `channels` use for ConsisID (number of input channels is equal to output channels)
    p = self.config.patch_size
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def main(args):
    seed = args.seed
    num_infer_steps = args.num_infer_steps
    output_path = args.output_path
    ckpts_path = args.ckpts_path
    # higher speedup will cause to worse quality -- 0.1 for 1.6x speedup -- 0.15 for 2.1x speedup -- 0.2 for 2.5x speedup
    rel_l1_thresh = args.rel_l1_thresh
    # ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
    prompt = args.prompt
    image = args.image
    
    if not os.path.exists(ckpts_path):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=ckpts_path)
    else:
        print(f"Base Model already exists in {ckpts_path}, skipping download.")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = (
        prepare_face_models(ckpts_path, device="cuda", dtype=torch.bfloat16)
    )
    pipe = ConsisIDPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
        face_helper_1,
        face_clip_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        face_main_model,
        "cuda",
        torch.bfloat16,
        image,
        is_align_face=True,
    )

    # TeaCache Config
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_infer_steps
    pipe.transformer.__class__.rel_l1_thresh = rel_l1_thresh  # 0.1 for 1.6x speedup -- 0.15 for 2.1x speedup -- 0.2 for 2.5x speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None
    pipe.transformer.__class__.forward = teacache_forward

    video = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=num_infer_steps,
        guidance_scale=6.0,
        use_dynamic_cfg=False,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    video_path = f"{output_path}/{seed}_{rel_l1_thresh}_{file_count:04d}.mp4"
    export_to_video(video.frames[0], video_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConsisID with given parameters")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_infer_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument("--output_path", type=str, default="./teacache_results", help="The path where the generated video will be saved")
    parser.add_argument('--ckpts_path', type=str, default="BestWishYsh/ConsisID-preview", help='Path to checkpoint')
    # higher speedup will cause to worse quality -- 0.1 for 1.6x speedup -- 0.15 for 2.1x speedup -- 0.2 for 2.5x speedup
    parser.add_argument('--rel_l1_thresh', type=float, default=0.1, help='Higher speedup will cause to worse quality -- 0.1 for 1.6x speedup -- 0.15 for 2.1x speedup -- 0.2 for 2.5x speedup')
    # ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
    parser.add_argument('--prompt', type=str, default="The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy\'s path, adding depth to the scene. The lighting highlights the boy\'s subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.", help='Description of the video for the model to generate')
    parser.add_argument('--image', type=str, default="https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true", help='URL or path to input image')
    args = parser.parse_args()

    main(args)