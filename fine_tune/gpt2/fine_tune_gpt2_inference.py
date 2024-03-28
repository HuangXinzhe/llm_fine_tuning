from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载训练后的 checkpoint
model = AutoModelForCausalLM.from_pretrained("output/checkpoint-1000")

# 模型设为推理模式
model.eval()

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/Volumes/WD_BLACK/models/gpt2", padding_side="left")

# 待分类文本
text = "This is a good movie!"

# 文本转 token ids - 记得以 eos 标识输入结束，与训练时一样
inputs = tokenizer(text+tokenizer.eos_token, return_tensors="pt")

# 推理：预测标签
output = model.generate(**inputs, do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)  # pad_token_id为eos token id，表示输入结束

# label token 转标签文本
print(tokenizer.decode(output[0][-1]))
