from openxlab.model import download
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import torch
from huggingface_hub import hf_hub_download  # Load model directly
from huggingface_hub import snapshot_download
import os

# 方法一
snapshot_download(repo_id="bert-base-cased",
                  local_dir="./model/bert-base-cased")

# 方法二
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')

# 方法三
hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json")

# 方法四
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='your path', revision='master')

# 方法五
download(model_repo='OpenLMLab/InternLM-7b', model_name='InternLM-7b', output='your local path')
