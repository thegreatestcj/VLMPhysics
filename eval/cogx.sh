#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --output=tests/slurm-%j.out

# === HuggingFace / Transformers Cache ===
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$HOME/scratch/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$HOME/scratch/.cache/huggingface/transformers"

# === PyTorch Cache (for torch.hub models) ===
export TORCH_HOME="$HOME/scratch/.cache/torch"

# === General XDG Cache (catches other tools) ===
export XDG_CACHE_HOME="$HOME/scratch/.cache"

python tests/cogx.py --prompt "A timelapse of a open toothpaste tube being squeezed by hand, with the pressure intensifying rapidly over time."

