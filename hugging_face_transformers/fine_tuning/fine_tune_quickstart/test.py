from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "/Volumes/WD_BLACK/models/bert-base-cased", num_labels=5)
