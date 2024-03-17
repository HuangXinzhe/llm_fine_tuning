# set env
export TRANSFORMERS_CACHE="./cache"

GPUIDX=0,1,2,3,4,5,6,7 # 显卡id
NUMPROCESS=8
DATAPATH="./sg_52k.json" # 需筛选的数据路径
BSZ=1                    # 每张显卡的批次大小

if [ ! -f $DATAPATH ]; then
    wget https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/old/sg_52k.json
fi

OUTPUTPATH="./cache/sharegpt_embedding/embed.pkl" # 向量输出路径
mkdir -p $(dirname $OUTPUTPATH)

# # 如果你的设备是a100/h100等新显卡，可以开启bf16混合精度训练和flash attention
# CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
#     --mixed_precision bf16 \
#     --num_processes $NUMPROCESS \
#     --num_machines 1 \
#     ./embed_datasets.py \
#     --use_flash_attention true \
#     --data_path $DATAPATH \
#     --output_path $OUTPUTPATH \
#     --batch_size_per_device $BSZ

# 否则只能用fp16并关闭flash attention
CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
    --mixed_precision fp16 \
    --num_processes $NUMPROCESS \
    --num_machines 1 \
    ./embed_datasets.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --batch_size_per_device $BSZ
