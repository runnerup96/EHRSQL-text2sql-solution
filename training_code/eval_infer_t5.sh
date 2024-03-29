#!/bin/bash

#set -e

seed=42

eval_batch_size=6
max_input_length=800
max_output_length=514
num_beams=4


CUDA_DEVICE_NUMBER='1'

statics_dir='/home/somov/ehrsql_statics'
data_dir='data/tmp_train_data'
test_file_name='ehrsql_test_for_t5.tsv'

test_file_path="${statics_dir}/${data_dir}/${test_file_name}"

save_model_dir="training_trials"

trained_model_name='t5_3b_final_v1_run_s42'
run_name="test_ehrsql_inference"

trained_model_path="$statics_dir/$save_model_dir/$trained_model_name"


tmux new-session -d -s $run_name
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/irm_env/bin/python -u infer_t5.py \
                              --model_name_or_path $trained_model_path \
                              --test_file $test_file_path \
                              --seed $seed \
                              --max_seq_length $max_input_length \
                              --max_output_length $max_output_length \
                              --per_device_eval_batch_size $eval_batch_size \
                              --generation_max_length $max_output_length \
                              --num_beams $num_beams \
                              --output_dir $trained_model_path" ENTER

tmux a -t $run_name