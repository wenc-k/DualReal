#!/bin/bash
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=
export TRAIN_STEP=1000
export MODEL_PATH='CogVideoX-5b'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

export ID_PATH='train/test_data/identity/dog'
export REF_PATH='train/test_data/identity/dog/images/00.png'
export MOTION_PATH='train/test_data/motion/playingGuitar'
export OUTPUT_PATH='train/output/dog_guitar'

accelerate launch --config_file train/config/finetune_adapter_single.yaml --multi_gpu \
  train/train_adapter.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir "~/.cache" \
  --enable_tiling \
  --enable_slicing \
  --caption_column_id prompts.txt \
  --caption_column_motion prompts.txt \
  --video_column_id videos.txt \
  --video_column_motion videos.txt \
  --seed 42 \
  --mixed_precision bf16 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 3 \
  --train_batch_size_other 1 \
  --max_train_steps $TRAIN_STEP \
  --checkpointing_steps 100000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --clip_pretrained 'train/pretrained/open_clip_pytorch_model.bin' \
  --allow_tf32 \
  --use_8bit_adam \
  --instance_data_root_id $ID_PATH \
  --instance_data_root_motion $MOTION_PATH \
  --output_dir $OUTPUT_PATH \
  --ref_image_path $REF_PATH \
  --training_parameters 'id_attn_adapter,id_ff_adapter,motion_attn_adapter,motion_ff_adapter,controller' \
  --use_IdAdapter true \
  --use_MotionAdapter true \
  --use_motion_condition true \
  --use_controller true \
  --hypernet_outputdim 7 \