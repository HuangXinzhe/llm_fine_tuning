from transformers import AutoModel, AutoTokenizer
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
                    help="需要验证的文本")

# 解析参数
args = parser.parse_args()

# 模型ID或本地路径
# model_name_or_path = '/Volumes/WD_BLACK/models/chatglm3-6b'
model_name_or_path = args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# base_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
base_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)


base_model = base_model.eval()


# input_text = "解释下乾卦是什么？"
response, history = base_model.chat(tokenizer, query=args.input_text)
print(response)
"""
乾卦是八卦之一，也是八宫图、易经六十四卦中的第一卦。乾卦是由两个阴爻夹一个阳爻构成，象征着天、强、积极、行动、力量、刚健、积极、进取等含义。

乾卦的卦辞是：“元、亨、利、贞。”这四个字代表了乾卦的基本特征和品质。其中，“元”表示万物本源，创造一切；“亨”表示事物发展顺利，通行无阻；“利”表示有利可图，收益无穷；“贞”表示正译，表示事物的坚定和稳固。

乾卦的六爻分别有不同的含义。初爻表示事物刚刚开始，充满希望和机遇；二爻表示事物正在发展，需要努力奋斗；三爻表示事物已经发展成熟，取得了成果；四爻表示事物处于顶点，面临重要的抉择；五爻表示事物已经达到了最高峰，取得了辉煌的成就；六爻表示事物已经到达了极点，需要保持警惕，防止滑落。

乾卦在易经中的作用非常重要，它不仅代表了天、强、积极、行动、力量等含义，而且也反映了人们在面对事物时的态度和行为方式。乾卦告诉我们，要在生活中充满信心和勇气，不断努力，积极进取，才能达到自己的目标。
"""