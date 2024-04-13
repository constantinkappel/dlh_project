root='../../'
# W2V training 
CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
    --distributed_world_size 1 \
    --input_path '/data/DescEmb/output_pretrain' \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task w2v ;


# W2V training 
CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
    --distributed_world_size 1 \
    --input_path '/data/DescEmb/output_pretrain' \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data eicu \
    --ratio 100 \
    --value_mode NV \
    --task w2v ;
