#!/bin/bash
#SBATCH -J eval                           
#SBATCH -o ./logs/eval_%J.out                       
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 4:00:00   
#SBATCH --mem 50G 

#export all_proxy=

# model=gpt4o_mini
# model=deepseek-chat
# model=qwen3-235b-a22b-instruct-2507
models=(gpt-4o-mini)
method=LDAgent

for model in ${models[@]}; do
    task_id=${method}_${model}
    python main.py full --task-id $task_id --task-types irrelevance_easy,irrelevance_hard,sycophancy,diversity --profile $task_id
done
