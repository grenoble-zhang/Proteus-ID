import argparse
import json
import os
import re
import uuid

import torch
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video

from models.proteusid_utils import prepare_face_models, process_face_embeddings_infer
from models.pipeline_proteusid import ProteusIDPipeline

from models.transformer_proteusid import ProteusIDTransformer3DModel

def generate_video(
    prompt: str,
    identity_prompt: str,
    model_path: str,
    model_transformer_path: str,
    img_file_path: str,
    negative_prompt: str = None,
    output_path: str = "./output",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - identity_prompt (str): The identity prompt for the video.
    - model_path (str): The path of the pre-trained model to be used.
    - model_transformer_path (str): The path of the transformer model.
    - img_file_path (str): The path of the face image (required).
    - negative_prompt (str): The description of the negative prompt.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    device = "cuda"
    
    # Validate input image path
    if not img_file_path or not os.path.exists(img_file_path):
        raise ValueError(f"Invalid image path: {img_file_path}. Please provide a valid image file.")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"

    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models(model_path, device, dtype)

    transformer = ProteusIDTransformer3DModel.from_pretrained(model_transformer_path, subfolder=subfolder, torch_dtype=dtype)
    pipe = ProteusIDPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)
    transformer.to(device, dtype=dtype)
    pipe.to(device)

    print(f"Processing image: {img_file_path}")
    print(f"Prompt: {prompt}")
    print(f"identity_prompt: {identity_prompt}")
    
    filename = os.path.basename(img_file_path)
    match = re.match(r'(\d+-\d+-\w+_\w+_\w+)', filename)
    if match:
        base_filename = match.group(1)
    else:
        base_filename = os.path.splitext(filename)[0]

    folder_path = os.path.join(output_path, base_filename)
    os.makedirs(folder_path, exist_ok=True)

    video_path = os.path.join(folder_path, f"{base_filename}_{uuid.uuid4().hex}.mp4")
    
    try:
        id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2,
                                                                                eva_transform_mean, eva_transform_std,
                                                                                face_main_model, device, dtype,
                                                                                img_file_path, is_align_face=True)
    except Exception as e:
        raise RuntimeError(f"Failed to process face embeddings from {img_file_path}. "
                          f"Please ensure the image contains a clear, detectable face. Error: {e}")

    valid_id_conds = []
    valid_id_conds.append(id_cond)
    valid_id_vit_hiddens = []
    valid_id_vit_hiddens.append(id_vit_hidden)
    valid_id_conds = torch.stack(valid_id_conds)
    valid_id_vit_hiddens = [torch.cat(tensor_group, dim=0) for tensor_group in zip(*valid_id_vit_hiddens)]
    
    prompt = prompt.strip('"')
    identity_prompt = identity_prompt.strip('"')
    if negative_prompt:
        negative_prompt = negative_prompt.strip('"')

    generator = torch.Generator(device).manual_seed(seed) if seed else None
    video_pt = pipe(
        prompt=prompt,
        identity_prompt=identity_prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=valid_id_vit_hiddens,
        id_cond=valid_id_conds,
        kps_cond=face_kps,
        output_type="pt",
        is_infer=True,
    ).frames
    torch.cuda.empty_cache()

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    # Save all generated videos
    for idx, video_frames in enumerate(batch_video_frames):
        if idx == 0:
            current_video_path = video_path
        else:
            current_video_path = os.path.join(folder_path, f"{base_filename}_{uuid.uuid4().hex}.mp4")
        export_to_video(video_frames, current_video_path, fps=8)
        print(f"Video {idx + 1}/{batch_size} saved to: {current_video_path}")
    
    del pipe
    del transformer
    free_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt")
    parser.add_argument("--model_path", type=str, default="fateforward/Proteus-ID", help="The path of the pre-trained model to be used")
    parser.add_argument("--model_transformer_path", type=str, default="fateforward/Proteus-ID", help="The path of the pre-trained model to be used")
    parser.add_argument("--img_file_path", type=str, required=True, help="Image path containing clear face, preferably half-body or full-body image")
    parser.add_argument("--json_file_path", type=str, default=None, help="JSON file path containing prompt and identity_prompt. If not provided, will try to find a JSON file with the same name as the image.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for video generation (overrides JSON if provided)")
    parser.add_argument("--identity_prompt", type=str, default=None, help="Identity prompt for video generation (overrides JSON if provided)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Specify a negative prompt to guide the generation model away from certain undesired features or content.")
    parser.add_argument("--output_path", type=str, default="./output", help="The path where the generated video will be saved")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()

    # Read prompt and identity_prompt from JSON file
    prompt = args.prompt
    identity_prompt = args.identity_prompt
    
    # If prompt or identity_prompt is not provided via command line, try to read from JSON file
    if prompt is None or identity_prompt is None:
        json_path = args.json_file_path
        
        # If JSON file path is not specified, try to infer from image path
        if json_path is None:
            img_dir = os.path.dirname(args.img_file_path)
            img_name = os.path.splitext(os.path.basename(args.img_file_path))[0]
            json_path = os.path.join(img_dir, f"{img_name}.json")
        
        if os.path.exists(json_path):
            print(f"Loading prompts from JSON file: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Only read from JSON if not provided via command line
            if prompt is None:
                prompt = json_data.get("prompt")
            if identity_prompt is None:
                identity_prompt = json_data.get("identity_prompt")
        else:
            print(f"Warning: JSON file not found at {json_path}")
    
    # Validate that prompt and identity_prompt exist
    if prompt is None:
        raise ValueError("prompt is required. Please provide it via --prompt or in a JSON file.")
    if identity_prompt is None:
        raise ValueError("identity_prompt is required. Please provide it via --identity_prompt or in a JSON file.")

    if not os.path.exists(args.model_path):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="fateforward/Proteus-ID", local_dir=args.model_path)
    else:
        print(f"Base Model already exists in {args.model_path}, skipping download.")

    generate_video(
        prompt=prompt,
        identity_prompt=identity_prompt,
        model_path=args.model_path,
        model_transformer_path=args.model_transformer_path,
        img_file_path=args.img_file_path,
        negative_prompt=args.negative_prompt,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        seed=args.seed,
    )
    # Usage examples:
    # 1. Specify JSON file path:
    #    CUDA_VISIBLE_DEVICES= 0 python inference.py --img_file_path assets/example_images/1.png --json_file_path assets/example_images/1.json
    # 2. Specify prompt via command line (overrides JSON values):
    #    python inference.py --img_file_path assets/example_images/1.png --prompt "In a tech-filled gaming room illuminated by the glow of monitors, a youthful man with freckled skin and coppery red hair, wearing a black hoodie over a graphic t-shirt, sits in a gaming chair with concentration. His fingers move rapidly over a keyboard and mouse, occasionally punctuated by small smiles in response to on-screen events. He shifts forward during moments of heightened gameplay, then relaxes back. The blue-tinged lighting from screens illuminates his features dramatically. His immersion in the virtual competition creates an atmosphere of authentic engagement, revealing the genuine enthusiasm behind his public gaming persona." --identity_prompt "A youthful man with freckled skin and coppery red hair, wearing a simple black hoodie over a graphic t-shirt"