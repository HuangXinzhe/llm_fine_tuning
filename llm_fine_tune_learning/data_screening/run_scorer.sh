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

# 预测每个数据样本的复杂度
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