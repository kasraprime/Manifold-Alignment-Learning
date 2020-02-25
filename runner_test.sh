#!/bin/bash
script_file=test.py
wandb_track=0
experiment_name=UWDeepCCA
task_name=uw

python $script_file --experiment_name $experiment_name --task $task_name | tee results/${experiment_name}/terminal_test.txt