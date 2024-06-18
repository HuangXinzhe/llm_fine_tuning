


# from transformers import AutoTokenizer, AutoModel

# model_name_or_path = "merged_model_directory"
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, trust_remote_code=True, device='cuda')
# model = model.eval()
# input_text = '类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领'
# response, history = model.chat(tokenizer=tokenizer, query=input_text, history=[])
# print(response)



from huggingface_hub import HfApi, notebook_login

# 登录到 Hugging Face（在 Jupyter 笔记本中使用）
# notebook_login()

# 或者，如果你已经通过命令行登录，你可以跳过登录步骤

# 创建 HfApi 实例
api = HfApi()

# 指定要上传的文件夹路径、仓库ID和仓库类型
# folder_path = "path_to_your_folder"  # 你的模型文件夹路径
folder_path = "merged_model_directory"
# repo_id = "your_username/your_repo_name"  # 例如 "snowfly/hihi"
repo_id = "snowfly/glm3-QLoRA-adgen"
repo_type = "model"  # 仓库类型

# 如果仓库不存在，先创建仓库
api.create_repo(repo_id=repo_id, repo_type=repo_type,
                private=False)  # 根据需要设置 private 参数

# 上传文件夹到 Hugging Face Hub
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Folder {folder_path} has been uploaded to Hugging Face Hub successfully!")