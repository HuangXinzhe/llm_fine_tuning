import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig

device_map = "cuda:0" if torch.cuda.is_available() else "auto"

finetune_model_path="/github_instruction/sft_result"  #微调模型参数保存路径
config = PeftConfig.from_pretrained(finetune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,
                                          use_fast=False)
tokenizer.pad_token = tokenizer.eos_tokendevice_map = device_map
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map=device_map,
                                             torch_dtype=torch.float16,
                                             load_in_8bit=True,
                                             trust_remote_code=True,
                                             use_flash_attention_2=True)
model = PeftModel.from_pretrained(model, 
                                  finetune_model_path, 
                                  device_map=device_map)
model =model.eval()
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant: '], 
                      return_tensors="pt",
                      add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
