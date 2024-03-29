#!/bin/bash

#set -e

seed=42

train_batch_size=2
gradient_accumulation_steps=8

eval_batch_size=2

epoch=16
lr='5e-5'
max_input_length=800
max_output_length=514

eval_steps=500
save_steps=500
log_steps=250

CUDA_DEVICE_NUMBER='0'

statics_dir='/home/somov/ehrsql_statics'
data_dir='data/tmp_train_data'
train_file_name='full_train_for_t5.tsv'
val_file_name='eval_for_t5.tsv'

train_file_path="${statics_dir}/${data_dir}/${train_file_name}"
val_file_path="${statics_dir}/${data_dir}/${val_file_name}"

save_model_dir="training_trials"

model_name='google-t5/t5-3b'
dir_model_name="t5_3b_final_v1_run"
run_name="t5_3b_final_v1_run"

output_dir="$statics_dir/$save_model_dir/${dir_model_name}_s${seed}"
logging_dir="${output_dir}/logs"

export PYTHONPATH="${PYTHONPATH}:/home/somov/ehrsql"

tmux new-session -d -s $run_name
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u finetune_hf_t5.py \
                              --model_name_or_path $model_name \
                              --train_file $train_file_path \
                              --validation_file $val_file_path \
                              --ignore_data_skip \
                              --do_train \
                              --do_eval \
                              --seed $seed \
                              --predict_with_generate \
                              --per_device_train_batch_size $train_batch_size \
                              --per_device_eval_batch_size $eval_batch_size \
                              --learning_rate $lr \
                              --num_train_epochs $epoch \
                              --gradient_accumulation_steps $gradient_accumulation_steps \
                              --max_grad_norm 1.0 \
                              --max_seq_length $max_input_length  \
                              --max_output_length $max_output_length \
                              --save_strategy 'steps' \
                              --metric_for_best_model 'eval_exact_match' \
                              --load_best_model_at_end \
                              --evaluation_strategy 'steps' \
                              --eval_steps $eval_steps \
                              --save_steps $save_steps \
                              --logging_steps $log_steps \
                              --generation_max_length $max_output_length \
                              --save_total_limit 3 \
                              --report_to 'tensorboard' \
                              --output_dir $output_dir \
                              --logging_dir $logging_dir" ENTER

tmux a -t $run_name