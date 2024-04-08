# set env
export TRANSFORMERS_CACHE="./cache"

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
