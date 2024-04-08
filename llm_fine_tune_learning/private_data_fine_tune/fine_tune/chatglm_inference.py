"""
模型推理：
    使用QLoRA微调后的模型
"""

from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# 模型ID或本地路径
model_name_or_path = 'THUDM/chatglm3-6b'

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

# 加载量化后模型(与微调的 revision 保持一致）
base_model = AutoModel.from_pretrained(model_name_or_path,
                                       quantization_config=q_config,
                                       device_map='auto',
                                       trust_remote_code=True,
                                       revision='b098244')

base_model.requires_grad_(False)  # 冻结模型参数，不进行梯度更新，只进行推理，节省内存，加快速度，提高效率，减少内存泄漏，防止内存溢出
base_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          trust_remote_code=True,  # 信任远程代码
                                          revision='b098244')  # revision 与微调的 revision 保持一致

# 使用微调后的模型
epochs = 3
# timestamp = "20240118_164514"
timestamp = "20240225_222843"

peft_model_path = f"models/{model_name_or_path}-epoch{epochs}-{timestamp}"

config = PeftConfig.from_pretrained(peft_model_path)
qlora_model = PeftModel.from_pretrained(base_model, peft_model_path)
training_tag = f"ChatGLM3-6B(Epoch=3, automade-dataset(fixed))-{timestamp}"


def compare_chatglm_results(query, base_model, qlora_model, training_tag):
    base_response, base_history = base_model.chat(tokenizer, query)

    inputs = tokenizer(query, return_tensors="pt").to(0)
    ft_out = qlora_model.generate(**inputs, max_new_tokens=512)
    ft_response = tokenizer.decode(ft_out[0], skip_special_tokens=True)

    print(
        f"问题：{query}\n\n原始输出：\n{base_response}\n\n\n微调后（{training_tag}）：\n{ft_response}")
    return base_response, ft_response


base_response, ft_response = compare_chatglm_results("解释下乾卦是什么？", base_model, qlora_model, training_tag)
print(base_response, ft_response)