#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --output=slurm/test/slurm-%j.out

# === Cache Configuration ===
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export TORCH_HOME="$HOME/scratch/.cache/torch"


python -m utils.test_dit_extractor --layers 15