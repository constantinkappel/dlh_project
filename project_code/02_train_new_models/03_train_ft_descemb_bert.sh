#!/bin/bash
# Run training new model using FT with descemb
# run this from project root folder using
# ./project_code/02_single_domain_learning/00_single_domain_learning_descemb.sh $(pwd)/project_code

if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
    MODEL_PATH=("${1%/}/DescEmb/outputs/2024-04-19/06-18-04/checkpoints/checkpoint_mimiciii_descemb_bert_mlm_last.pt")
else
    root='../DescEmb/'
    MODEL_PATH=('../../../outputs/2024-04-19/06-18-04/checkpoints/checkpoint_mimiciii_descemb_bert_mlm_last.pt')
fi 

INPUT_PATH=/data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')

embed_models=('descemb_bert')
tasks=('readmission' 'mortality' 'los_3day' 'los_7day' 'diagnosis')
value_modes=('VA' 'DSVA' 'DSVA_DPE' 'VC')


for data in "${SRC_DATA[@]}"; do
    for index in "${!embed_models[@]}"; do
        emb_model=${embed_models[$index]}
        model_path=${MODEL_PATH[$index]}
        echo "Processing model: $emb_model with data $data"
        for task in "${tasks[@]}"; do
            for value_mode in "${value_modes[@]}"; do
                echo "Current directory: $(pwd)" && \
                CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
                    --distributed_world_size 1 \
                    --input_path "$INPUT_PATH" \
                    --init_bert_params \
                    --model ehr_model \
                    --embed_model "$emb_model" \
                    --pred_model rnn \
                    --src_data $data \
                    --ratio 100 \
                    --value_mode "$value_mode" \
                    --task "$task" ;
            done
        done
    done
done