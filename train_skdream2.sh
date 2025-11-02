# conda activate StableDiffusion
# accelerate launch train_skdream.py \
#  --mixed_precision="fp16" \
#  --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
#  --data_root_dir='/data04/xyy/3D/objaverse/objaverse_sk' \
#  --output_dir="output/origin" \
#  --cond_channels=4 \
#  --cond_module="conv_norm" \
#  --resolution=256 \
#  --lr_scheduler='constant_with_warmup' \
#  --lr_warmup_steps=500 \
#  --learning_rate=1e-5 \
#  --train_batch_size=4 \
#  --max_train_steps=8000 \
#  --dataloader_num_workers=16 \
#  --tracker_project_name="skdream" \
#  --checkpointing_steps=1000 \
#  --validation_steps=5000000 \
#  --gradient_accumulation_steps=2 \
#  --report_to tensorboard 

accelerate launch train_skdream.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
 --data_root_dir='/home/zrs/mix_data' \
 --output_dir="output/sft3" \
 --cond_channels=4 \
 --cond_module="conv_norm" \
 --resolution=256 \
 --lr_scheduler='constant_with_warmup' \
 --lr_warmup_steps=500 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --max_train_steps=16000 \
 --dataloader_num_workers=16 \
 --tracker_project_name="skdream" \
 --checkpointing_steps=1000 \
 --validation_steps=5000000 \
 --gradient_accumulation_steps=2 \
 --report_to tensorboard 
