# 全量微调的demo
base_model="./output/model/Qwen1.5-1.8B-deita_10k_deepspeed" # 模型参数
python ./LLaMA-Factory/src/cli_demo.py \
    --model_name_or_path $base_model \
    --template qwen \
    --finetuning_type full

# # lora微调的demo
# base_model="Qwen/Qwen1.5-1.8B"                                    # 模型参数
# lora_model="./output/model/Qwen1.5-1.8B-deita_10k_deepspeed_lora" # lora参数
# python ./LLaMA-Factory/src/cli_demo.py \
#     --model_name_or_path $base_model \
#     --adapter_name_or_path $lora_model \
#     --template qwen \
#     --finetuning_type lora
