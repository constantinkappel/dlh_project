#!/bin/bash
# Run transfer learning with descemb
# run this from project root folder using
# ./project_code/05_transfer_learning/01_transfer_descemb-bert.sh $(pwd)/project_code

# These model paths must list the models trained on all tasks below in same order
# The order of datasets the models were trained on must be reversed compared to the order of SRC_DATA
if [ -n "$1" ]; then
    root="${1%/}/DescEmb/"
    MODEL_PATH=("${1%/}/DescEmb/outputs/2024-05-01/14-15-52/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-01/17-38-21/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-01/21-13-52/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/01-48-46/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/05-41-38/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/09-46-47/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/14-27-50/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/17-23-39/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-02/21-41-30/checkpoints/checkpoint_best.pt" \
                "${1%/}/DescEmb/outputs/2024-05-03/00-56-25/checkpoints/checkpoint_best.pt")
else
    root='../DescEmb/'
    MODEL_PATH=("../../../outputs/2024-05-01/14-15-52/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-01/17-38-21/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-01/21-13-52/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/01-48-46/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/05-41-38/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/09-46-47/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/14-27-50/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/17-23-39/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-02/21-41-30/checkpoints/checkpoint_best.pt" \
                "../../../outputs/2024-05-03/00-56-25/checkpoints/checkpoint_best.pt")
fi 


INPUT_PATH=/home/data/output #/data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')

embed_models=('descemb_bert')
tasks=('diagnosis' 'los_3day' 'los_7day' 'readmission' 'mortality')
value_modes=('DSVA_DPE')


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
                    --model_path $model_path \
                    --transfer \
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