from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import argparse

# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--model_name_or_path",
                    type=str,
                    required=True,
                    help="模型的名称或路径")

# 解析参数
args = parser.parse_args()

model_name_or_path = args.model_name_or_path

quant_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

text = "Merry Christmas! I'm glad to"
inputs = tokenizer(text, return_tensors="pt").to(0)  # 将输入张量放在设备0上

out = quant_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
