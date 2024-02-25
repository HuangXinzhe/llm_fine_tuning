from trl import SFTTrainer
import transformers
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from datasets import load_dataset

# 加载微调数据
data = load_dataset("Abirate/english_quotes",
                    cache_dir="/content/drive/MyDrive/llm/gemma/data")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# 加载模型
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir="/content/drive/MyDrive/llm/gemma/model", token="")
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={
                                             "": 0}, cache_dir="/content/drive/MyDrive/llm/gemma/model", token="")

# 微调模型
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj",
                    "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()

# 保存微调后的模型
trainer.save_model("/content/drive/MyDrive/llm/gemma/peft_model")

# 测试微调后的模型
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
