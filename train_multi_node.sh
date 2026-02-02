export WANDB_MODE="offline"
# For training from scratch
export MODEL_PATH="THUDM/CogVideoX-5b-I2V"    # "THUDM/CogVideoX1.5-5b-I2V"
export CONFIG_PATH="THUDM/CogVideoX-5b-I2V"   # "THUDM/CogVideoX1.5-5b-I2V"
export TYPE="i2v"
export DATASET_PATH="total_train_data.txt"
export OUTPUT_PATH="proteusid_finetune_multi_rank"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo $MASTER_ADDR
echo $NODE_RANK
echo $DISTRIBUTED_NODE_RANK
echo $PET_MASTER_ADDR
echo $PET_MASTER_PORT

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO



TORCH_DISTRIBUTED_DEBUG=DETAIL ACCELERATE_DEBUG_VERBOSITY="debug" accelerate launch --main_process_port=${PET_MASTER_PORT} --main_process_ip=$PET_MASTER_ADDR --machine_rank=$DISTRIBUTED_NODE_RANK \
  --config_file util/deepspeed_configs/accelerate_config_multi_node.yaml \
  train.py \
  --config_path $CONFIG_PATH \
  --dataloader_num_workers 8 \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "In an indoor setting that exudes a historical or cultural ambiance, illuminated by soft, warm lighting and characterized by blurred background elements, a young woman with long dark hair adorned with red ribbons, wearing dangling earrings and traditional attire with a red collar, remains mostly stationary. With only subtle head movements and slight shifts in her gaze, she maintains a close-up presence, capturing the viewer's attention through her evolving expressions. Her demeanor transitions from a neutral look to a gentle smile, hinting at a positive interaction. The overall atmosphere is calm and intimate, emphasizing her facial expressions and the intricate details of her attire, creating a serene and engaging visual experience." \
  --validation_identity_prompt "A young woman with long dark hair adorned with red ribbons, wearing dangling earrings and traditional attire with a red collar" \
  --validation_images "/nfs/dataset-ofs-voyager-research/shichen/project/video_diffusion/RigFace/77yu/video/proteus-evaluation/final/evaluation_count/105-2-stars_woman_Yifei_Liu_1.png" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1000 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 250 \
  --checkpoints_total_limit 50 \
  --max_train_steps 8000 \
  --learning_rate 3e-6 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --resume_from_checkpoint="latest" \
  --report_to wandb \
  --sample_stride 3 \
  --skip_frames_start 7 \
  --skip_frames_end 7 \
  --miss_tolerance 6 \
  --min_distance 3 \
  --min_frames 1 \
  --max_frames 1 \
  --cross_attn_interval 2 \
  --cross_attn_dim_head 128 \
  --cross_attn_num_heads 16 \
  --is_single_face \
  --enable_mask_loss \
  --is_align_face \
  --train_type $TYPE \
  --is_shuffle_data \
  --is_validation \
  --guidance_scale 6.0 \
  --is_cross_face \
  --LFE_id_dim 1280 \
  --LFE_vit_dim 1024 \
  --LFE_depth 10 \
  --LFE_dim_head 64 \
  --LFE_num_heads 16 \
  --LFE_num_id_token 5 \
  --LFE_num_querie 32 \
  --LFE_output_dim 2048 \
  --LFE_ff_mult 4 \
  --LFE_num_scale 5 \
  --local_face_scale 1.0 \
  --FT_txt_dim 4096 \
  --FT_clip_dim 1280 \
  --FT_vit_dim 1024 \
  --FT_num_querie 32 \
  --FT_depth 10 \
  --FT_dim_head 64 \
  --FT_num_heads 64 \
  --FT_output_dim 4096 \
  --FT_num_scale 5 \
  --FT_ff_mult 4 \
  --FFE_num_queries 32 \
  --FFE_query_dim 2048 \
  --FFE_input_img_dim 2048 \
  --FFE_input_fuse_dim 4096 \
  --FFE_output_dim 2048 \
  --FFE_layers 10 \
  --FFE_heads 32 \
  --FFE_clip_dim 1280 \
  --FFE_vit_dim 1024 \
  --FFE_time_channel 3072 \
  --FFE_time_embed_dim 2048 \
  --FFE_timestep_activation_fn silu