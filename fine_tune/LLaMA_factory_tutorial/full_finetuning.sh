export TRANSFORMERS_CACHE="./cache"

# mkdir -p ./output/model/Qwen1.5-1.8B-deita_10k_full_single
# # single card training
# CUDA_VISIBLE_DEVICES=0 python ./LLaMA-Factory/src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path Qwen/Qwen1.5-1.8B \
#     --dataset_dir ./LLaMA-Factory/data \
#     --dataset deita_10k \
#     --template qwen \
#     --finetuning_type full \
#     --output_dir ./output/model/Qwen1.5-1.8B-deita_10k_full_single \
#     --overwrite_cache \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 5.0 \
#     --plot_loss \
#     --use_fast_tokenizer True \
#     --fp16

mkdir -p ./output/model/Qwen1.5-1.8B-deita_10k_deepspeed
# deepspeed distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 --master_port=1023 ./LLaMA-Factory/src/train_bash.py \
    --deepspeed ./configs/deepspeed_stage2.json \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen1.5-1.8B \
    --dataset_dir ./LLaMA-Factory/data \
    --dataset deita_10k \
    --template qwen \
    --finetuning_type full \
    --output_dir ./output/model/Qwen1.5-1.8B-deita_10k_deepspeed \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --use_fast_tokenizer True \
    --fp16

# mkdir -p ./output/model/Qwen1.5-1.8B-deita_10k_accelerate
# # accelerate distributed training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file ./configs/accelerator.yaml \
#                     ./LLaMA-Factory/src/train_bash.py \
#                     --stage sft \
#                     --do_train \
#                     --model_name_or_path Qwen/Qwen1.5-1.8B \
#                     --dataset_dir ./LLaMA-Factory/data \
#                     --dataset deita_10k \
#                     --template qwen \
#                     --finetuning_type full \
#                     --output_dir ./output/model/Qwen1.5-1.8B-deita_10k_accelerate \
#                     --overwrite_cache \
#                     --per_device_train_batch_size 4 \
#                     --gradient_accumulation_steps 8 \
#                     --lr_scheduler_type cosine \
#                     --logging_steps 10 \
#                     --save_steps 1000 \
#                     --learning_rate 5e-5 \
#                     --num_train_epochs 5.0 \
#                     --plot_loss \
#                     --use_fast_tokenizer True \
#                     --fp16
