#!/bin/bash

#SBATCH --job-name=anymal-c-soccer-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=terry
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --output=./slurm_logs/%x-%j.log
#SBATCH --error=./slurm_logs/%x-%j.log

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

python ../train.py \
    --task "Anymal-C-Dreamer-Soccer-v0" \
    --num_envs 100 \
    --headless \
    --enable_cameras \
    --logdir "/scratch/isaacp/dreamerv3_logs/anymal_c_soccer/{timestamp}" \
    --num_env_steps 10000000000

