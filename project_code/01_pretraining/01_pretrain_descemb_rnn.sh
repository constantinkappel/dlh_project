root='../../'
# MLM training value mode should be set on NV
CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
    --distributed_world_size 1 \
    --input_path '/data/DescEmb/output_pretrain' \
    --model descemb_rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task mlm ;

# MLM training value mode should be set on NV
CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
    --distributed_world_size 1 \
    --input_path '/data/DescEmb/output_pretrain' \
    --model descemb_rnn \
    --src_data eicu \
    --ratio 100 \
    --value_mode NV \
    --task mlm ;