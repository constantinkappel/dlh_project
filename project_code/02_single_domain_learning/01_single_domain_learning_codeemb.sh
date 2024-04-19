#!/bin/bash
# Run single domain learning with codeemb
root='../DescEmb/'
INPUT_PATH=/data/DescEmb/output
embed_models=('codeemb')
tasks=('readmission' 'mortality' 'los_3day' 'los_7day' 'diagnosis')
value_modes=('NV' 'VA' 'DSVA' 'VC')
SRC_DATA=('mimiciii' 'eicu')
MODEL_PATH=('/../DescEmb/outputs/') # PLEASE MODIFY


for data in "${SRC_DATA[@]}"; do
    for index in "${!embed_models[@]}"; do
        echo "Processing model: $emb_model with data $data"
        for task in "${tasks[@]}"; do
            for value_mode in "${value_modes[@]}"; do
                CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
                    --distributed_world_size 1 \
                    --input_path "$INPUT_PATH" \
                    --model_path "${MODEL_PATH[$index]}" \
                    --load_pretrained_weights \
                    --model ehr_model \
                    --embed_model "${emb_models[$index]}" \
                    --pred_model rnn \
                    --src_data $data \
                    --ratio 100 \
                    --value_mode "$value_mode" \
                    --task "$task" ;
            done
        done
    done
done