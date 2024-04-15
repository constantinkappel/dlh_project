INPUT_PATH=/home/data
OUTPUT_PATH=/home/data/output
DX_PATH=$INPUT_PATH/ccs_multi_dx_tool_2015.csv

python ../preprocess/preprocess_main.py \
    --src_data mimiciii \
    --dataset_path $INPUT_PATH/mimic \
    --ccs_dx_tool_path $DX_PATH \
    --dest_path $OUTPUT_PATH ;

python ../preprocess/preprocess_main.py \
    --src_data eicu \
    --dataset_path $INPUT_PATH/eicu \
    --ccs_dx_tool_path $DX_PATH \
    --dest_path $OUTPUT_PATH ;

python ../preprocess/preprocess_main.py \
    --src_data pooled \
    --ccs_dx_tool_path $DX_PATH \
    --dest_path $OUTPUT_PATH ;


python ../preprocess/preprocess_main.py \
    --src_data mimiciii \
    --dataset_path $INPUT_PATH/mimic \
    --dest_path $OUTPUT_PATH \
    --ccs_dx_tool_path $DX_PATH \
    --data_type pretrain ;

python ../preprocess/preprocess_main.py \
    --src_data eicu \
    --dataset_path $INPUT_PATH/eicu \
    --dest_path $OUTPUT_PATH \
    --ccs_dx_tool_path $DX_PATH \
    --data_type pretrain ;

python ../preprocess/preprocess_main.py \
    --src_data pooled \
    --dest_path $OUTPUT_PATH \
    --ccs_dx_tool_path $DX_PATH \
    --data_type pretrain ;