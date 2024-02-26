"""
使用gradio部署模型
1、克隆ChatGLM3代码：https://github.com/THUDM/ChatGLM3
2、指定MODEL_PATH为ChatGLM3的模型路径: export MODEL_PATH=模型地址（如果不指定模型地址，则会下载模型）
3、找到web_demo_gradio.py位置，并通过命令行运行：Python web_demo_gradio.py

注意：
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=trust_remote_code, device_map='auto', offload_folder="/path/to/offload_folder"
)
当GPU或CPU内存不足时，可以使用offload_folder参数将模型的一部分放到磁盘上，以减少内存占用。
"""