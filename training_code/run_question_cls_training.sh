#!/bin/bash

#set -e

export PYTHONPATH="${PYTHONPATH}:/home/somov/ehrsql"

seed=42
batch_size=16
epoch=15
lr='0.00001'
max_input_length=514
training_type="only_question"

eval_steps=50
log_steps=4

CUDA_DEVICE_NUMBER='1'

statics_dir='/home/somov/ehrsql_statics'
data_dir="${statics_dir}/data/tmp_train_data"

train_file_path="$data_dir/ood_train.json"
val_file_path="$data_dir/ood_val.json"

save_model_dir="training_trials"

model_name='FacebookAI/xlm-roberta-base'
dir_model_name="qs_cls_finetune_v5"
run_name="qs_cls_finetune_v5"

output_dir="$statics_dir/$save_model_dir/${dir_model_name}_s${seed}"


tmux new-session -d -s $run_name

tmux send-keys -t $run_name "PYTHONPATH=/home/somov/ehrsql CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u finetune_cls.py \
                              --training_type $training_type \
                              --model_path $model_name \
                              --device cuda \
                              --batch_size $batch_size \
                              --lr $lr \
                              --train_file_path $train_file_path \
                              --eval_file_path $val_file_path \
                              --max_seq_len $max_input_length \
                              --upsample_ratio 5 \
                              --seed $seed \
                              --valid_steps $eval_steps \
                              --log_steps $log_steps \
                              --gradient_accumulation_steps 2 \
                              --epochs $epoch \
                              --output_path $output_dir" ENTER

tmux a -t $run_name
