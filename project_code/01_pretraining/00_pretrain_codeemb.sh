#!/bin/bash
# Run pre-training with codeemb and W2V
# run this from project root folder using
# ./project_code/01_pretraining/03_pretrain_descemb_bert_ft-mlm.sh $(pwd)/project_code

if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
else
    root='../DescEmb/'
fi 

INPUT_PATH=/data/DescEmb/output #/home/data/output # /data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')
MODEL='codeemb'
TASK='w2v'
for src_data in "${SRC_DATA}"; do
    echo "Processing dataset: $src_data"
    # Pre-train Descemb-BERT initialized with BERT params using MLM task 
    CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
        --distributed_world_size 1 \
        --input_path "$INPUT_PATH" \
        --model "$MODEL" \
        --init_bert_params \
        --src_data "$src_data" \
        --ratio 100 \
        --value_mode NV \
        --save_prefix "checkpoint_${src_data}_${MODEL}_${TASK}" \
        --patience 20 \
        --task "$TASK" ;
done

