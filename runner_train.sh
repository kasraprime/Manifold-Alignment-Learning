#!/bin/bash
script_file=train.py
wandb_track=0
experiment_name=UWDeepCCA
task_name=uw
num_epochs=100
mkdir -p ./results/${experiment_name}/
python $script_file --wandb_track $wandb_track --experiment_name $experiment_name --epochs $num_epochs --task $task_name | tee results/${experiment_name}/terminal_train.txt