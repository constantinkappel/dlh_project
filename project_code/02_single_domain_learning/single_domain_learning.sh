# Run single domain learning

#!/bin/bash

embed_models=('descemb_rnn' 'descemb_bert')
tasks=('readmission' 'mortality' 'los_3day' 'los_7day' 'diagnosis')
value_modes=('NV' 'VA' 'DSVA' 'DSVA_DPE' 'VC')

for emb_model in "${embed_models[@]}"; do
    echo "Processing model: $emb_model"
    for task in "${tasks[@]}"; do
        for value_mode in "${value_modes[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --distributed_world_size 1 \
                --input_path '/data/DescEmb/output_pretrain'\
                --model ehr_model \
                --embed_model "$emb_model" \
                --pred_model rnn \
                --src_data mimiciii \
                --ratio 100 \
                --value_mode "$value_mode" \
                --task "$task" ;
        done
    done
done

embed_models=('codeemb')
tasks=('readmission' 'mortality' 'los_3day' 'los_7day' 'diagnosis')
value_modes=('NV' 'VA' 'DSVA' 'VC')

for emb_model in "${embed_models[@]}"; do
    echo "Processing model: $emb_model"
    for task in "${tasks[@]}"; do
        for value_mode in "${value_modes[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --distributed_world_size 1 \
                --input_path '/data/DescEmb/output_pretrain'\
                --model ehr_model \
                --embed_model "$emb_model" \
                --pred_model rnn \
                --src_data mimiciii \
                --ratio 100 \
                --value_mode "$value_mode" \
                --task "$task" ;
        done
    done
done

