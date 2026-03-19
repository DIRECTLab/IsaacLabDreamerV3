#!/bin/bash

#SBATCH --job-name=dreamer-anymal-c-play
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run the Dreamer Anymal C play script
python ../play.py \
    --task "Isaac-Velocity-Flat-Anymal-C-Direct-v0" \
    --num_envs 500 \
    --headless
