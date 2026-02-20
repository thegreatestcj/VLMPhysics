#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --output=slurm/analysis/vqascore-%j.out

# ============================================================
# Compute VQAScore for VideoPhy CogVideoX Training Videos
#
# Loads CLIP-FlanT5-XXL (~11B params) and computes
# VQAScore(frame, caption) for each video.
#
# Time: ~1-2 hours for 343 videos on single GPU
# ============================================================

echo "========================================"
echo "Compute VQAScore for Training Videos"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi -L
echo "Start: $(date)"
echo "========================================"

mkdir -p slurm/analysis

module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export TORCH_HOME="$HOME/scratch/.cache/torch"
mkdir -p $HF_HOME $TORCH_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

DATA_DIR="$HOME/scratch/physics/videophy_cogx"
OUTPUT="$DATA_DIR/vqascores.json"
NUM_FRAMES=4

# ============================================================
# Run
# ============================================================

python utils/compute_vqascore.py \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT" \
    --num-frames $NUM_FRAMES

echo "========================================"
echo "Done: $(date)"
echo "Output: $OUTPUT"
echo "========================================"