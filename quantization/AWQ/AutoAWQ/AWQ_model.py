from transformers import AwqConfig, AutoConfig
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
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

model_name_or_path = args.model_name_or_path
quant_model_dir = args.output_dir

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True)

# 量化模型
model.quantize(tokenizer, quant_config=quant_config)

# 修改配置文件以使其与transformers集成兼容
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
).to_dict()

# 预训练的transformers模型存储在model属性中，我们需要传递一个字典
model.model.config.quantization_config = quantization_config

# 保存模型权重
model.save_quantized(quant_model_dir)
# 保存分词器
tokenizer.save_pretrained(quant_model_dir)
