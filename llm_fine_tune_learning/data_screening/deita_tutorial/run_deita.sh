# set env
export TRANSFORMERS_CACHE="./cache"

# 预测每个数据样本的复杂度
SCORETYPE="complexity"
DATAPATH="./sg_52k.json"
OUTPUTPATH="./output/dieta/complexity_sg_52k.json"
MODELPATH="hkust-nlp/deita-complexity-scorer"
SCORER="llama"
ISVLLM=false

python ./score_dataset.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --score_type $SCORETYPE \
    --scorer $SCORER \
    --scorer_name_or_path $MODELPATH

# 预测每个数据样本的质量
SCORETYPE="quality"
DATAPATH="./output/dieta/complexity_sg_52k.json"
OUTPUTPATH="./output/dieta/complexity_quality_sg_52k.json"
MODELPATH="hkust-nlp/deita-quality-scorer"
SCORER="llama"
ISVLLM=false

python ./score_dataset.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --score_type $SCORETYPE \
    --scorer $SCORER \
    --scorer_name_or_path $MODELPATH

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

# 最终的数据筛选
GPUIDX="0,1,2,3,4,5,6,7"
NUMGPUS=$(echo $GPUIDX | awk -F',' '{print NF}')
DATAPATH="./output/dieta/complexity_quality_sg_52k.json"
OTHERDATA="./cache/sharegpt_embedding/embed.pkl" # PATH/TO/EMBEDDING_FILE
OUTPUTPATH="./output/dieta/complexity_quality_sg_52k_filtered.json"                  # PATH/TO/OUTPUTS
THETA=0.9
DATASIZE=10000
BSZ=4

CUDA_VISIBLE_DEVICES=$GPUIDX python ./combined_filter.py \
    --data_path $DATAPATH \
    --other_data_path $OTHERDATA \
    --output_path $OUTPUTPATH \
    --threshold $THETA \
    --data_size $DATASIZE \
    --is_compression true \
    --device 0
