#!/bin/bash
#SBATCH --job-name=oasis
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

# Same activation as the cuda-check job.sh that is known to work on this account.
# The pip torch wheels bundle their own CUDA, so no `module load cuda` is needed.
source $HOME/miniconda3/bin/activate
conda activate torch

# Strict mode only AFTER activation: conda's activate hooks reference unset
# variables, and `set -u` turns that into a hard failure.
set -euo pipefail

export OASIS_ROOT=/home/groups/comp3710/OASIS

nvidia-smi
# Pick the task to run:
python -m part4_recognition.task1_vae_oasis.train
# python -m part4_recognition.task2_unet_oasis.train
# python -m part4_recognition.task3_gan_oasis.train
