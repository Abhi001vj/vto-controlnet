export MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./circle_model/"
export OUTPUT_DIR="fashion_model/"
#  --dataset_name=fusing/fill50k \
#  /home/abhi/.cache/huggingface/accelerate/default_config.yaml 
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=Abhilashvj/vto_hd_train \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./00000_00.jpg" "./00001_00.jpg" \
 --validation_prompt "a woman wearing a white t - shirt with a levi 's logo on it . " "a woman in a black shirt and black skirt . " \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --mixed_precision fp16 \
 --tracker_project_name=fashion_controlnet \
 --resume_from_checkpoint="checkpoint-26000" \
 --checkpoints_total_limit=5 \
 --num_train_epochs=10 \
 --checkpointing_steps=3000 \
 --validation_steps=1000 \
 --report_to wandb \
#  --push_to_hub