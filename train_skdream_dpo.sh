# accelerate launch train_skdream_dpo.py \
#  --mixed_precision="fp16" \
#  --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
#  --controlnet_model_name_or_path="/home/zrs/skdream_model/ckpt/skdream/" \
#  --data_root_dir='/home/zrs/mix_data/dpo_dataset' \
#  --output_dir="output/dpo1" \
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

# 对于sft模型 + 自己的数据
# accelerate launch train_skdream_dpo.py \
#  --mixed_precision="fp16" \
#  --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
#  --controlnet_model_name_or_path="/home/zrs/skdream_model/tmp_v3/checkpoint-9000/skdream" \
#  --data_root_dir='/home/zrs/mix_data/dpo_dataset3/sft_model_output_0.05' \
#  --output_dir="output/dpo_sft_dpo_1k_0.05" \
#  --cond_channels=4 \
#  --cond_module="conv_norm" \
#  --resolution=256 \
#  --lr_scheduler='constant_with_warmup' \
#  --lr_warmup_steps=100 \
#  --learning_rate=5e-6 \
#  --train_batch_size=4 \
#  --max_train_steps=1000 \
#  --dataloader_num_workers=16 \
#  --tracker_project_name="skdream" \
#  --checkpointing_steps=500 \
#  --validation_steps=5000000 \
#  --gradient_accumulation_steps=2 \
#  --report_to tensorboard 

 accelerate launch train_skdream_dpo.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
 --controlnet_model_name_or_path="/home/zrs/skdream_model/tmp_v3/checkpoint-9000/skdream" \
 --data_root_dir='/home/zrs/mix_data/dpo_dataset3/sft_model_output_0.1' \
 --output_dir="output/dpo_sft_dpo_1k_0.1" \
 --cond_channels=4 \
 --cond_module="conv_norm" \
 --resolution=256 \
 --lr_scheduler='constant_with_warmup' \
 --lr_warmup_steps=100 \
 --learning_rate=5e-6 \
 --train_batch_size=4 \
 --max_train_steps=1000 \
 --dataloader_num_workers=16 \
 --tracker_project_name="skdream" \
 --checkpointing_steps=500 \
 --validation_steps=5000000 \
 --gradient_accumulation_steps=2 \
 --report_to tensorboard 

accelerate launch train_skdream_dpo.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
 --controlnet_model_name_or_path="/home/zrs/skdream_model/tmp_v3/checkpoint-9000/skdream" \
 --data_root_dir='/home/zrs/mix_data/dpo_dataset3/sft_model_output_0.15' \
 --output_dir="output/dpo_sft_dpo_1k_0.15" \
 --cond_channels=4 \
 --cond_module="conv_norm" \
 --resolution=256 \
 --lr_scheduler='constant_with_warmup' \
 --lr_warmup_steps=100 \
 --learning_rate=5e-6 \
 --train_batch_size=4 \
 --max_train_steps=1000 \
 --dataloader_num_workers=16 \
 --tracker_project_name="skdream" \
 --checkpointing_steps=500 \
 --validation_steps=5000000 \
 --gradient_accumulation_steps=2 \
 --report_to tensorboard \


accelerate launch train_skdream_dpo.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
 --controlnet_model_name_or_path="/home/zrs/skdream_model/tmp_v3/checkpoint-9000/skdream" \
 --data_root_dir='/home/zrs/mix_data/dpo_dataset3/sft_model_output_0.2' \
 --output_dir="output/dpo_sft_dpo_1k_0.2" \
 --cond_channels=4 \
 --cond_module="conv_norm" \
 --resolution=256 \
 --lr_scheduler='constant_with_warmup' \
 --lr_warmup_steps=100 \
 --learning_rate=5e-6 \
 --train_batch_size=4 \
 --max_train_steps=1000 \
 --dataloader_num_workers=16 \
 --tracker_project_name="skdream" \
 --checkpointing_steps=500 \
 --validation_steps=5000000 \
 --gradient_accumulation_steps=2 \
 --report_to tensorboard 
# 
# accelerate launch train_skdream_dpo.py \
#  --mixed_precision="fp16" \
#  --pretrained_model_name_or_path="/data03/xyy/diffusion/mvdream_ckpt/mvdream-sd21-diffusers-lzq" \
#  --controlnet_model_name_or_path="/data05/xyy/3D/skdream_model/ckpt/skdream/" \
#  --data_root_dir='/home/zrs/mix_data/dpo_dataset3/origin_model_output' \
#  --output_dir="output/dpo_origin_origin_data" \
#  --cond_channels=4 \
#  --cond_module="conv_norm" \
#  --resolution=256 \
#  --lr_scheduler='constant_with_warmup' \
#  --lr_warmup_steps=100 \
#  --learning_rate=5e-6 \
#  --train_batch_size=4 \
#  --max_train_steps=1000 \
#  --dataloader_num_workers=16 \
#  --tracker_project_name="skdream" \
#  --checkpointing_steps=200 \
#  --validation_steps=5000000 \
#  --gradient_accumulation_steps=2 \
#  --report_to tensorboard 