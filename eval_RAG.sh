#!/bin/bash
#SBATCH -J eval                           
#SBATCH -o ./logs/eval_%J.out                       
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 4:00:00   
#SBATCH --mem 50G 

# vLLM 模型服务启动脚本
# 用于启动 vLLM 模型服务，支持多种模型和配置选项
# conda init
# source activate
# conda activate MABench

export all_proxy=

# sleep 100

models=(gpt-4o-mini)
method=RAG

for model in ${models[@]}; do
    task_id=${method}_${model}
    python main.py full --task-id $task_id --task-types irrelevance_easy,irrelevance_hard,sycophancy,diversity --profile $task_id 
done
