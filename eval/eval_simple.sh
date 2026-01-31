#!/bin/bash
#SBATCH --mem=16G
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --output=slurm/eval/eval-%j.out


# GPT-4o evaluation
python eval/eval_simple.py \
    --videos_dir outputs/phygenbench/physics \
    --output results/eval/physics_physion_simple.json \
    --vlm gpt4o