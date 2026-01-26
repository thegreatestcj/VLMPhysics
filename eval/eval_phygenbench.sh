#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -n 4
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --output=slurm/phygenbench/baseline/slurm-%j.out

source /users/ctang33/.conda/envs/physics/bin/activate

# === Cache Configuration ===
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export TORCH_HOME="$HOME/scratch/.cache/torch"

# === Create output directory ===

python scripts/eval_phygenbench.py \
    --videos-dir outputs/phygenbench/baseline \
    --prompts-file data/phygenbench/prompts.json \
    --questions-dir data/phygenbench \
    --stage 2,3 \
    --indices 0,1,2,3,4 \
    --verbose \
    --model gpt-4.1-mini \
    --output results/phygenbench/baseline.json
    > results/eval.log 2>&1 &

wait

echo "All done at $(date)"