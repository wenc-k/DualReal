cd ../

CUDA_VISIBLE_DEVICES=

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CLIP_PATH=''
export ADAPTER_PATH=''
export OUTPUT_PATH='output'
export PROMPT_PATH='dog_guitar/test_prompt.txt'
export REF_IMG_PATH='dog_guitar/reference.png'

python -m inference.inference \
  --pretrained_model_name_or_path "./CogVideoX-5b" \
  --dtype "bfloat16" \
  --use_IdAdapter true \
  --use_MotionAdapter true \
  --use_motion_condition true \
  --use_controller true \
  --hypernet_outputdim 7 \
  --use_gpu_accelerate true \
  --output_path $OUTPUT_PATH \
  --adapter_path $ADAPTER_PATH \
  --prompt $PROMPT_PATH \
  --ref_image_path $REF_IMG_PATH \
  --clip_pretrained $CLIP_PATH \