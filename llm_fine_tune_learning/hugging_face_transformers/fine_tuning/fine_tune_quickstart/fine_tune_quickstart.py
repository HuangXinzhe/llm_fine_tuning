from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import logger

# ===================================1、加载数据集===================================
# 此处为huggingface的datasets库，加载yelp_review数据集
# 可以自制数据集
logger.info("开始加载数据集")
# dataset = load_dataset("yelp_review_full", 
#                        cache_dir="./data",
                    #    )
dataset = load_dataset(
    "/Users/huangxinzhe/code/llm_fine_tuning/hugging_face_transformers/fine_tuning/fine_tune_quickstart/datasets/yelp_review_full")
logger.info("数据加载完成")

# ===================================2、预处理数据===================================
logger.info("开始预处理数据")
tokenizer = AutoTokenizer.from_pretrained("/Volumes/WD_BLACK/models/bert-base-cased")

# 以最大长度进行padding，并截断
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
logger.info("数据预处理完成")

# ===================================3、数据抽样===================================
logger.info("开始抽样")
# 抽取1000条数据作为训练集，100条数据作为测试集
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
logger.info("抽样完成")

# ===================================4、微调训练配置===================================
logger.info("开始微调训练配置")
# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("/Volumes/WD_BLACK/models/bert-base-cased", 
                                                           num_labels=5)

# 训练超参数
model_dir = "models/bert-base-cased"
# 训练过程指标监控
# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=f"{model_dir}/test_trainer",
                                  evaluation_strategy="epoch",
                                  logging_dir=f"{model_dir}/test_trainer/runs",
                                  logging_steps=100)

# 训练过程中的指标评估
metric = evaluate.load("./accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

logger.info("微调训练配置完成")

# ===================================5、微调训练===================================
logger.info("开始微调训练")
# 实例化训练器
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=small_train_dataset,
                  eval_dataset=small_eval_dataset,
                  compute_metrics=compute_metrics,
                  )

trainer.train()
small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))
print(trainer.evaluate(small_test_dataset))
logger.info(trainer.evaluate(small_test_dataset))
logger.info("微调训练完成")

# ===================================6、保存微调模型和训练状态===================================
logger.info("开始保存微调模型和训练状态")
trainer.save_model(f"{model_dir}/finetuned-trainer")
trainer.save_state()
logger.info("保存微调模型和训练状态完成")
