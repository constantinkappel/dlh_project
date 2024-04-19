#!/bin/bash
# Run single domain learning with descemb
INPUT_PATH=/data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')
root='../DescEmb/'
embed_models=('descemb_rnn' 'descemb_bert')
MODEL_PATH=('../DescEmb/outputs/2024-04-19/15-55-40' '../DescEmb/outputs/2024-04-19/06-18-04')
tasks=('readmission' 'mortality' 'los_3day' 'los_7day' 'diagnosis')
value_modes=('NV' 'VA' 'DSVA' 'DSVA_DPE' 'VC')


for data in "${SRC_DATA[@]}"; do
    for index in "${!embed_models[@]}"; do
        emb_model=${embed_models[$index]}
        model_path=${MODEL_PATH[$index]}
        echo "Processing model: $emb_model with data $data"
        for task in "${tasks[@]}"; do
            for value_mode in "${value_modes[@]}"; do
                CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
                    --distributed_world_size 1 \
                    --input_path "$INPUT_PATH" \
                    --model_path "$model_path" \
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
