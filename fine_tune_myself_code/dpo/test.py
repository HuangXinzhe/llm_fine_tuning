# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

# model_name = "teknium/OpenHermes-2.5-Mistral-7B"


# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
# )
# toeknizer = AutoTokenizer.from_pretrained(model_name)
# # print("ok")

# from datasets import load_dataset
# dataset = load_dataset("Intel/orca_dpo_pairs")



from transformers import AutoTokenizer
import transformers
new_model = "NeuralHermes-2.5-Mistral-7B"
# Format prompt
message = [
    {"role": "system", "content": "You are a helpful assistant chatbot."},
    {"role": "user", "content": "What is a Large Language Model?"}
]
tokenizer = AutoTokenizer.from_pretrained(new_model)
prompt = tokenizer.apply_chat_template(
    message, add_generation_prompt=True, tokenize=False)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=new_model,
    tokenizer=tokenizer
)

# Generate text
sequences = pipeline(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    max_length=200,
)
print(sequences[0]['generated_text'])