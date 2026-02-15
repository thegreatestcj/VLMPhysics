#!/bin/bash
#SBATCH --job-name=extract_feat_full
#SBATCH --output=slurm/extraction/extract_full_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
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

DATA_DIR="/users/$USER/scratch/physics/videophy_data"
OUTPUT_DIR="/users/$USER/scratch/physics/videophy_features_2"

# Layers to extract (for ablation study)
LAYERS="0 5 29"

# Timesteps (for trajectory pruning checkpoints)
TIMESTEPS="200 400 600"

MAX_VIDEOS=99999

# ============================================================
# Run extraction
# ============================================================

if [ ! -f "$DATA_DIR/metadata.json" ]; then
    echo "ERROR: metadata.json not found!"
    exit 1
fi

TOTAL=$(python3 -c "import json; print(len(json.load(open('$DATA_DIR/metadata.json'))))")
MP4S=$(find $DATA_DIR/videos -name '*.mp4' 2>/dev/null | wc -l)
echo "Metadata: $TOTAL entries  |  mp4 on disk: $MP4S"
echo ""

echo ""
echo "Configuration:"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Layers: $LAYERS"
echo "  Timesteps: $TIMESTEPS"
echo "  Max videos: $MAX_VIDEOS"
echo ""


# echo "Extracting..."
# Run extraction for shard 0
# python -m src.models.extract_features \
#     --data_dir $DATA_DIR \
#     --output_dir $OUTPUT_DIR \
#     --layers $LAYERS \
#     --dataset videophy \
#     --timesteps 200 400 600 \
#     --use-8bit \
#     --use-text \
#     --pool \
#     --shard 0 \
#     --num_shards 2


# echo "=========================================="
# echo "Shard 0 completed: $(date)"
# echo "=========================================="

# Run extraction for shard 1
python -m src.models.extract_features \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --layers $LAYERS \
    --dataset videophy \
    --timesteps 200 400 600 \
    --use-8bit \
    --use-text \
    --pool \
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

