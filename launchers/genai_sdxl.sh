export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="GenAI800"
export OUTPUT_DIR="./outputs/sdxl"
export DATASET_DIR="/data3/yixinf/dpo_datasets_800"
# export HF_HOME="/data3/kewenwu/huggingface"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

CUDA_VISIBLE_DEVICES=2 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 200 \
  --beta_dpo 5000 \
  --sdxl  \
  --output_dir=$OUTPUT_DIR \
  --train_data_dir=$DATASET_DIR \
  --choice_model='vqascore' \
  