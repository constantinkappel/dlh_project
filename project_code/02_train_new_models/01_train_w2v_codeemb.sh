#!/bin/bash
# Run training new model using W2V with codeemb
# run this from project root folder using
# ./project_code/02_train_new_models/01_train_w2v_codeemb.sh $(pwd)/project_code


if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
    MODEL_PATH=("${1%/}/DescEmb/outputs/2024-05-04/07-33-59/checkpoints/checkpoint_mimiciii_codeemb_ft-w2v_best.pt")
else
    root='../DescEmb/'
    MODEL_PATH=('../../../outputs/2024-05-04/07-33-59/checkpoints/checkpoint_mimiciii_codeemb_ft-w2v_best.pt')
fi 

INPUT_PATH=/data/DescEmb/output #/home/data/output #/data/DescEmb/output
SRC_DATA=('mimiciii') # 'eicu')

embed_models=('codeemb')
tasks=('diagnosis' 'los_3day' 'los_7day' 'readmission' 'mortality')
value_modes=('VA' 'DSVA' 'VC')


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
                    --model ehr_model \
                    --embed_model "$emb_model" \
                    --pred_model rnn \
                    --src_data $data \
                    --ratio 100 \
                    --patience 45 \
                    --value_mode "$value_mode" \
                    --save_prefix "checkpoint_${data}_${emb_model}_${task}" \
                    --task "$task" ;
            done
        done
    done
done