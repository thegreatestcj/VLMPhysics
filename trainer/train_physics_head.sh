#!/bin/bash
#SBATCH --job-name=train_head
#SBATCH --output=slurm/training/train_full_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================
# Physics Head Training & Ablation Studies
#
# Trains physics discriminator heads on pre-extracted DiT features.
# Training is very fast (~10 min for full ablation).
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

# Create output directory for slurm logs
mkdir -p slurm/training

# Environment setup
module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

# Set cache directories
export HF_HOME=/users/$USER/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

FEATURE_DIR="/users/$USER/scratch/physics/videophy_features_2"
METADATA_FILE="/users/$USER/scratch/physics/videophy_data/metadata.json"
OUTPUT_DIR="results"

# Training hyperparameters
BATCH_SIZE=32
NUM_EPOCHS=100
LR=0.001
WEIGHT_DECAY=0.01
EARLY_STOPPING=20
NUM_WORKERS=4

# Best layer from previous ablation
DEFAULT_LAYER=15

# Multi-task SA training
SA_WEIGHT=0.3

# ============================================================
# Print configuration
# ============================================================

echo ""
echo "Configuration:"
echo "  Feature dir:    $FEATURE_DIR"
echo "  Metadata file:  $METADATA_FILE"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Batch size:     $BATCH_SIZE"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Learning rate:  $LR"
echo "  Early stopping: $EARLY_STOPPING epochs"
echo "  SA weight:      $SA_WEIGHT"
echo ""

# Validate paths
if [ ! -d "$FEATURE_DIR" ]; then
    echo "ERROR: Feature directory not found: $FEATURE_DIR"
    exit 1
fi

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    exit 1
fi

# ============================================================
# Head Architecture Ablation (with multi-task SA)
# ============================================================

# echo "========================================"
# echo "Running HEAD ABLATION (PC + SA)..."
# echo "========================================"

# python -m trainer.train_physics_head \
#     --feature_dir $FEATURE_DIR \
#     --metadata_file $METADATA_FILE \
#     --ablation heads \
#     --heads mean causal_simple temporal_simple multiview_simple \
#     --layer $DEFAULT_LAYER \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --weight_decay $WEIGHT_DECAY \
#     --early_stopping $EARLY_STOPPING \
#     --num_workers $NUM_WORKERS \
#     --is_pooled \
#     --train-sa \
#     --sa-weight $SA_WEIGHT \
#     --exp-name heads_sa

# echo ""
# echo "Head ablation completed: $(date)"
# echo ""

# ============================================================
# Layer Ablation
#
# Compares features from different DiT layers:
#   - Layer 5:  Early features (low-level)
#   - Layer 10: Mid-early features (previous best)
#   - Layer 15: Middle features
#   - Layer 20: Mid-late features
#   - Layer 25: Late features (high-level)
#
# Uses best head from head ablation. Expected time: ~10 minutes
# ============================================================

echo "========================================"
echo "Running LAYER ABLATION (PC + SA)..."
echo "========================================"

python -m trainer.train_physics_head \
    --feature_dir $FEATURE_DIR \
    --metadata_file $METADATA_FILE \
    --ablation layers \
    --layers 10 15 20 25 \
    --head_type mean \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --early_stopping $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --is_pooled \
    --train-sa \
    --sa-weight $SA_WEIGHT

echo ""
echo "Layer ablation completed: $(date)"
echo ""

# ============================================================
# Seed Ablation
#
# Tests model stability across different random seeds.
# Expected time: ~15 minutes (5 seeds)
# ============================================================

# echo "========================================"
# echo "Running SEED ABLATION..."
# echo "========================================"

# python -m trainer.train_physics_head \
#     --feature_dir $FEATURE_DIR \
#     --metadata_file $METADATA_FILE \
#     --ablation seeds \
#     --seeds 42 123 456 789 1024 \
#     --layer $DEFAULT_LAYER \
#     --head_type multiview_simple \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --weight_decay $WEIGHT_DECAY \
#     --early_stopping $EARLY_STOPPING \
#     --num_workers $NUM_WORKERS \
#     --is_pooled \
#     --train-sa \
#     --sa-weight $SA_WEIGHT

# echo ""
# echo "Seed ablation completed: $(date)"
# echo ""

# ============================================================
# Summary
# ============================================================

echo "========================================"
echo "All ablation studies completed!"
echo "End: $(date)"
echo "========================================"

echo ""
echo "Results summary files:"
ls -la ${OUTPUT_DIR}/*/summary.json 2>/dev/null || echo "  (check individual result folders)"

echo ""
echo "Disk usage:"
du -sh ${OUTPUT_DIR}/* 2>/dev/null