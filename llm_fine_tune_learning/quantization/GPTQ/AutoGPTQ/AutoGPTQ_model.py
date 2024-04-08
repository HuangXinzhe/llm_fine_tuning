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
parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    help="模型的输出路径")

# 解析参数
args = parser.parse_args()


# ===================================1、量化配置文件===================================
model_name_or_path = args.model_name_or_path

quantization_config = GPTQConfig(
    bits=4,  # 量化精度
    group_size=128,  # 将权重矩阵分组，每组的大小（分组进行量化，-1表示每列进行量化）
    dataset="wikitext2",
    desc_act=False,  # 是否按激活大小递减的顺序量化列。将其设置为False可以显著加快推理速度，但困惑可能会变得更糟。
)

# ===================================2、模型量化===================================
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map='auto')  # device_map='auto'是使用了accelerate库，可以自动选择设备

# ===================================3、检车模型量化的正确性===================================
print("检查模型量化的正确性：")
print(quant_model.model.decoder.layers[0].self_attn.q_proj.__dict__)

# ===================================4、保存量化模型===================================
quant_model.save_pretrained(args.output_dir)

print("模型量化完成！")