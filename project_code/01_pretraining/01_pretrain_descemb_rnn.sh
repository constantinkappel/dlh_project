#!/bin/bash
# Run pretraining  with descemb-RNN from scratch
# run this from project root folder using
# ./project_code/01_pretraining/01_pretrain_descemb_rnn_mlm.sh $(pwd)/project_code

if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
else
    root='../DescEmb/'
fi 

INPUT_PATH=/home/data/output # /data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')
MODEL='descemb_rnn'
TASK='mlm'
for src_data in "${SRC_DATA}"; do
    echo "Processing dataset: $src_data"
    # Pre-train Descemb-BERT initialized with BERT params using MLM task 
    CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
        --distributed_world_size 1 \
        --input_path "$INPUT_PATH" \
        --model "$MODEL" \
        --src_data "$src_data" \
        --ratio 100 \
        --value_mode NV \
        --save_prefix "checkpoint_${src_data}_${MODEL}_ft-${TASK}" \
        --patience 200 \
        --task "$TASK" ;
done