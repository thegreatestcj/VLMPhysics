#!/bin/bash
#SBATCH --job-name=extract_feat
#SBATCH --output=slurm/extraction/extract_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

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

MAX_VIDEOS=200

# ============================================================
# Run extraction
# ============================================================

echo ""
echo "Configuration:"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Layers: $LAYERS"
echo "  Timesteps: $TIMESTEPS"
echo "  Max videos: $MAX_VIDEOS"
echo ""

# Extract train split
echo "Extracting..."
python -m src.models.extract_features \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --layers $LAYERS \
    --timesteps $TIMESTEPS \
    --use-8bit \
    --max_videos $MAX_VIDEOS


# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================"
echo "Extraction complete!"
echo "End: $(date)"
echo "========================================"

echo ""
echo "Output directory contents:"
ls -la $OUTPUT_DIR

echo ""
echo "Disk usage:"
du -sh $OUTPUT_DIR

echo "Phase 1, testing w/ $(MAX_VIDEOS) videos complete."