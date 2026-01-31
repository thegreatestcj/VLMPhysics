#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C geforce3090
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --output=slurm/phygenbench/baseline/slurm-%j.out

# === Cache Configuration ===
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export TORCH_HOME="$HOME/scratch/.cache/torch"

# === Create output directory ===
mkdir -p outputs/phygenbench/baseline

# === Run 2 GPUs in parallel ===
echo "Starting generation at $(date)"

# python scripts/generate_phygenbench.py \
#     --model 2b \
#     --prompts-file data/phygenbench/prompts.json \
#     --output-dir outputs/phygenbench/baseline \
#     --start 0 --end 80 \
#     --skip-existing \

python scripts/generate_phygenbench.py \
    --model 2b \
    --prompts-file data/phygenbench/prompts.json \
    --output-dir outputs/phygenbench/baseline \
    --start 80 --end 160 \
    --skip-existing \

wait

echo "All done at $(date)"