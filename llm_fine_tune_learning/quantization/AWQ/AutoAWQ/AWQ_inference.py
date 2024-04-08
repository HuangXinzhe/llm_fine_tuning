from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--model_name_or_path",
                    type=str,
                    required=True,
                    help="模型的名称或路径")
parser.add_argument("--input_text",
                    type=str,
                    required=True,
                    help="输入文本")

# 解析参数
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="cuda").to(0)


def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(0)

    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)


result = generate_text(args.input_text)
print(result)
