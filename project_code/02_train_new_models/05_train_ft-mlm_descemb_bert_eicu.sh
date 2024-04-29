#!/bin/bash
# Run training new model using FT with descemb
# run this from project root folder using
# ./project_code/02_single_domain_learning/00_single_domain_learning_descemb.sh $(pwd)/project_code

if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
    MODEL_PATH=("${1%/}/DescEmb/outputs/2024-04-29/06-45-17/checkpoints/checkpoint_eicu_descemb_bert_ft-mlm_last.pt")
else
    root='../DescEmb/'
    MODEL_PATH=('../../../outputs/2024-04-29/06-45-17/checkpoints/checkpoint_eicu_descemb_bert_ft-mlm_last.pt')
fi 

INPUT_PATH=/home/data/output #/data/DescEmb/output
SRC_DATA=('eicu')

embed_models=('descemb_bert')
tasks=('diagnosis' 'los_3day' 'los_7day' 'readmission' 'mortality')
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
                    --model_path "$model_path" \
                    --load_pretrained_weights \
                    --model ehr_model \
                    --embed_model "$emb_model" \
                    --pred_model rnn \
                    --src_data $data \
                    --ratio 100 \
                    --patience 90 \
                    --value_mode "$value_mode" \
                    --task "$task" ;
            done
        done
    done
done