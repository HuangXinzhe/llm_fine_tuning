# LLM部署
参考资料：https://github.com/LlamaFamily/Llama-Chinese/tree/main/inference-speed/GPU/vllm_example

## LLM部署步骤：
1. 下载需要部署的模型
2. 单卡或多卡推理
3. 启动client测试

### 单卡推理
编辑single_gpus_api_server.sh里面model为上面模型的下载路径  

multi_gpus_api_server.sh 里面的CUDA_VISIBLE_DEVICES指定了要使用的GPU卡  

bash single_gpus_api_server.sh

### 多卡推理
编辑multi_gpus_api_server.sh里面model为上面模型的下载路径  

multi_gpus_api_server.sh 里面的CUDA_VISIBLE_DEVICES指定了要使用的GPU卡  

tensor-parallel-size 指定了卡的个数  

bash multi_gpus_api_server.sh

### 启动client测试
python client_test.py