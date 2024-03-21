# Atom微调
使用Atom作为基座模型进行微调
## 数据集
- data目录下包含训练与测试数据
- 如果只是做简单的微调可以截取部分数据进行微调
## 微调
- finetune_lora.sh是微调脚本
- finetune_clm_lora.py是微调代码
- 单机多卡的微调可以通过修改脚本中的--include localhost:0来实现。
- 微调脚本中需要确认的配置
    - 微调模型保存路径
    - 微调数据集路径
    - 基座模型路径
    - 单机多卡、单机单卡
- 进入微调脚本所在的目录下，执行脚本即可开始微调。脚本：bash finetune_lora.sh
## 使用微调后模型
- atom_fine_tune_test.py是微调后模型的测试代码