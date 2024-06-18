from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("t5-3b")
t5_tok = AutoTokenizer.from_pretrained("t5-3b")

input_ids = t5_tok("When was Franklin D. Roosevelt born?",
                   return_tensors="pt").input_ids
gen_output = t5_qa_model.generate(input_ids)[0]

print(t5_tok.decode(gen_output, skip_special_tokens=True))


dataset = load_dataset("wmt16", "ro-en")
print(dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 610320
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 1999
    })
    test: Dataset({
        features: ['translation'],
        num_rows: 1999
    })
})
"""
