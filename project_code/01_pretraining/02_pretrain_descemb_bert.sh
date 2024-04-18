root='../DescEmb/'
# MLM training value mode should be set on NV

INPUT_PATH=/data/DescEmb/output
SRC_DATA=('mimiciii' 'eicu')
MODEL='descemb_bert'
TASK='mlm'
for src_data in "${SRC_DATA}"; do
    echo "Processing dataset: $src_data"
    CUDA_VISIBLE_DEVICES=0 python ${root}main.py \
        --distributed_world_size 1 \
        --input_path "$INPUT_PATH" \
        --model "$MODEL" \
        --src_data "$src_data" \
        --ratio 100 \
        --value_mode NV \
        --save_prefix "checkpoint_${src_data}_${MODEL}_${TASK}" \
        --task "$TASK" ;
done

