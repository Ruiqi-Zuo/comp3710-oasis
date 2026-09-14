#!/bin/bash
#SBATCH --job-name=oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
# --account is required: the comp3710 partition only admits jobs charged to
# the comp3710 account, and without it SLURM holds the job forever with
# reason (PartitionConfig). No --mem request: the partition registers ~1 MB
# per node, so memory is not a schedulable resource here and a request can
# only make the job unsatisfiable.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
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

# Choose the task on the command line instead of editing this file on the
# cluster:   sbatch scripts/oasis.slurm vae    |    sbatch scripts/oasis.slurm unet
# Each trains and then runs its predict script on the checkpoint it just wrote,
# so the figures exist as soon as the job finishes.
case "${1:-}" in
    vae)
        python -m part4_recognition.task1_vae_oasis.train
        python -m part4_recognition.task1_vae_oasis.predict
        ;;
    unet)
        python -m part4_recognition.task2_unet_oasis.train
        python -m part4_recognition.task2_unet_oasis.predict
        ;;
    *)
        echo "usage: sbatch scripts/oasis.slurm {vae|unet}" >&2
        exit 2
        ;;
esac
