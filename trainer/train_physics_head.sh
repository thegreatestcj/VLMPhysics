#!/bin/bash
#SBATCH --job-name=train_head
#SBATCH --output=slurm/training/train_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C geforce3090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================
# Physics Head Training & Ablation Studies
# 
# This script trains physics discriminator heads on pre-extracted
# DiT features. Training is very fast (~10 min for full ablation).
#
#
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

# Set cache directories (in case any HF models needed)
export HF_HOME=/users/$USER/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

# Feature directory (from extract_features.py)
FEATURE_DIR="/users/$USER/scratch/physics/physion_features"
LABEL_FILE="${FEATURE_DIR}/labels.json"

# Output directory (small files, can be in project folder)
OUTPUT_DIR="results"

# Training hyperparameters
BATCH_SIZE=32
NUM_EPOCHS=100
LR=0.001
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=5
VAL_RATIO=0.15
EARLY_STOPPING=20
NUM_WORKERS=4
SEED=42

# Default layer for head ablation
DEFAULT_LAYER=15

# Default head for layer ablation
DEFAULT_HEAD="causal_adaln"

# ============================================================
# Print configuration
# ============================================================

echo ""
echo "Configuration:"
echo "  Feature dir: $FEATURE_DIR"
echo "  Label file: $LABEL_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LR"
echo "  Val ratio: $VAL_RATIO"
echo "  Early stopping: $EARLY_STOPPING epochs"
echo ""

# Check if feature directory exists
if [ ! -d "$FEATURE_DIR" ]; then
    echo "ERROR: Feature directory not found: $FEATURE_DIR"
    echo "Please run extract_features.py first!"
    exit 1
fi

# Check if label file exists
if [ ! -f "$LABEL_FILE" ]; then
    echo "ERROR: Label file not found: $LABEL_FILE"
    exit 1
fi

# ============================================================
# Ablation Study 1: Head Architecture Comparison
# 
# Compares 5 head types:
#   - mean:           Baseline (no timestep)
#   - mean_adaln:     Mean pooling + AdaLN
#   - temporal_adaln: Bidirectional attention + AdaLN
#   - causal_adaln:   Causal attention + AdaLN (recommended)
#   - multiview_adaln: Multi-view pooling + AdaLN
#
# Expected time: ~10 minutes
# ============================================================

echo "========================================"
echo "Running HEAD ABLATION..."
echo "========================================"

python -m trainer.train_physics_head \
    --feature_dir $FEATURE_DIR \
    --label_file $LABEL_FILE \
    --ablation heads \
    --layer $DEFAULT_LAYER \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --val_ratio $VAL_RATIO \
    --early_stopping $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir ${OUTPUT_DIR}/head_ablation

echo ""
echo "Head ablation completed: $(date)"
echo ""


# ============================================================
# Ablation Study 2: DiT Layer Comparison
# 
# Compares features from different DiT layers:
#   - Layer 5:  Early features (low-level)
#   - Layer 10: Mid-early features
#   - Layer 15: Middle features (usually best)
#   - Layer 20: Mid-late features
#   - Layer 25: Late features (high-level)
#
# Expected time: ~10 minutes
# ============================================================

# echo "========================================"
# echo "Running LAYER ABLATION..."
# echo "========================================"

# python -m trainer.train_physics_head \
#     --feature_dir $FEATURE_DIR \
#     --label_file $LABEL_FILE \
#     --ablation layers \
#     --layers 5 10 15 20 25 \
#     --head_type $DEFAULT_HEAD \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --weight_decay $WEIGHT_DECAY \
#     --warmup_epochs $WARMUP_EPOCHS \
#     --val_ratio $VAL_RATIO \
#     --early_stopping $EARLY_STOPPING \
#     --num_workers $NUM_WORKERS \
#     --seed $SEED \
#     --output_dir ${OUTPUT_DIR}/layer_ablation

# echo ""
# echo "Layer ablation completed: $(date)"
# echo ""


# ============================================================
# Ablation Study 3: Timestep Configuration (Optional)
# 
# Compares training with different noise levels:
#   - t=200: Low noise (clearer features)
#   - t=400: Medium-low noise
#   - t=600: Medium-high noise
#   - t=800: High noise (noisier features)
#   - All: Combined training
#
# Expected time: ~10 minutes
# ============================================================

# echo "========================================"
# echo "Running TIMESTEP ABLATION..."
# echo "========================================"

# python -m trainer.train_physics_head \
#     --feature_dir $FEATURE_DIR \
#     --label_file $LABEL_FILE \
#     --ablation timesteps \
#     --layer $DEFAULT_LAYER \
#     --head_type $DEFAULT_HEAD \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --weight_decay $WEIGHT_DECAY \
#     --warmup_epochs $WARMUP_EPOCHS \
#     --val_ratio $VAL_RATIO \
#     --early_stopping $EARLY_STOPPING \
#     --num_workers $NUM_WORKERS \
#     --seed $SEED \
#     --output_dir ${OUTPUT_DIR}/timestep_ablation

# echo ""
# echo "Timestep ablation completed: $(date)"
# echo ""


# ============================================================
# Summary
# ============================================================

echo "========================================"
echo "All ablation studies completed!"
echo "End: $(date)"
echo "========================================"

echo ""
echo "Output directory structure:"
find ${OUTPUT_DIR} -name "*.json" -o -name "*.pt" | head -30

echo ""
echo "Results summary files:"
ls -la ${OUTPUT_DIR}/*/summary.json 2>/dev/null || echo "  (check individual result folders)"

echo ""
echo "Disk usage:"
du -sh ${OUTPUT_DIR}/*

echo ""
echo "========================================"
echo "Next steps:"
echo "  1. Check results: cat ${OUTPUT_DIR}/head_ablation/head_ablation_summary.json"
echo "  2. Compare AUC scores in the summary tables above"
echo "  3. Select best head and layer for final model"
echo "========================================"