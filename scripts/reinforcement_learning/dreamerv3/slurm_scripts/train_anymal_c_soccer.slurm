#!/bin/bash

#SBATCH --job-name=anymal-c-soccer-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=frutiger
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run the Dreamer Anymal C play script
python ../play.py \
    --task "Anymal-C-Dreamer-Soccer-v0" \
    --num_envs 10 \
    --headless \
    --enable_cameras

