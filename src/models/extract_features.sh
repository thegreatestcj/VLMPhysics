#!/bin/bash
#SBATCH --job-name=extract_feat
#SBATCH --output=slurm/extraction/extract_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# ============================================================
# Feature Extraction for VLM-Physics
# 
# Run this ONCE before training. Extracts DiT features and
# saves to disk for fast MLP training.
#
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

# Create output directory for slurm logs
mkdir -p slurm/extraction

# Environment setup
module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

# Set cache directories
export HF_HOME=/users/$USER/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

DATA_DIR="data/Physion"
OUTPUT_DIR="/users/$USER/scratch/physics/physion_features"

# Layers to extract (for ablation study)
LAYERS="5 10 15 20 25"

# Timesteps (for trajectory pruning checkpoints)
TIMESTEPS="400"

MAX_VIDEOS=9999

# ============================================================
# Run extraction
# ============================================================

# echo ""
# echo "Configuration:"
# echo "  Data dir: $DATA_DIR"
# echo "  Output dir: $OUTPUT_DIR"
# echo "  Layers: $LAYERS"
# echo "  Timesteps: $TIMESTEPS"
# echo "  Max videos: $MAX_VIDEOS"
# echo ""


# echo "Extracting..."
# # Run extraction for shard 0
# python -m src.models.extract_features \
#     --data_dir /users/$USER/scratch/physics/videophy_data \
#     --output_dir /users/$USER/scratch/physics/videophy_features \
#     --layers 10 15 20 25 \
#     --dataset videophy \
#     --timesteps 200 400 600 \
#     --use-8bit \
#     --shard 0 \
#     --num_shards 2


# echo "=========================================="
# echo "Shard 0 completed: $(date)"
# echo "=========================================="

Run extraction for shard 1
python -m src.models.extract_features \
    --data_dir /users/$USER/scratch/physics/videophy_data \
    --output_dir /users/$USER/scratch/physics/videophy_features \
    --layers 10 15 20 25 \
    --dataset videophy \
    --timesteps 200 400 600 \
    --use-8bit \
    --shard 1 \
    --num_shards 2

echo "=========================================="
echo "Shard 1 completed: $(date)"
echo "=========================================="

echo ""
echo "Output directory contents:"
ls -la $OUTPUT_DIR

echo ""
echo "Disk usage:"
du -sh $OUTPUT_DIR
