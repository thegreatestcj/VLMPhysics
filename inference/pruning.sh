#!/bin/bash
#SBATCH --job-name=traj_dual
#SBATCH --output=slurm/inference/dual_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# ============================================================
# Dual-GPU Trajectory Pruning - Maximum Parallelism
#
# Uses BOTH GPUs to maximize trajectory count:
#   - 16 trajectories total (8 per GPU)
#   - Mini-batch processing (batch_size=4)
#   - ~4x faster than sequential single-GPU
#
# Memory per GPU (~18GB used / 24GB available):
#   - DiT (bfloat16):     ~10GB
#   - VAE (tiling):        ~1GB
#   - 4 traj latents:     ~0.6GB
#   - Activations:         ~6GB
#   - Buffers:             ~1GB
#
# ============================================================

echo "========================================"
echo "Dual-GPU Trajectory Pruning"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi -L
echo "Start: $(date)"
echo "========================================"

# Directories
mkdir -p slurm/inference
mkdir -p results/inference/pruning

# Environment
module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

# Cache
export HF_HOME=/users/ctang33/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/users/ctang33/scratch/torch_cache
mkdir -p $HF_HOME $TORCH_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

# Physics head (set to empty for test mode)
PHYSICS_HEAD="results/inference/temporal_simple_layer10.pt"
HEAD_TYPE="temporal_simple"

# MAXIMIZED for dual-GPU
NUM_TRAJECTORIES=4   # 8 per GPU
BATCH_SIZE=2          # Mini-batch per GPU

# Checkpoints: 16 → 8 → 4 → 2 → 1
CHECKPOINTS="800 600 400 200"
KEEP_RATIO=0.5

# Generation
STEPS=50
FRAMES=49
GUIDANCE=6.0

# Test prompts
PROMPTS=(
    "A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area"
    "A cup of oil is slowly poured out in the space station, releasing the liquid into the surrounding area"
)

OUTPUT_BASE="results/inference/pruning"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Check physics head
# ============================================================

if [ ! -f "$PHYSICS_HEAD" ]; then
    echo "Physics head not found: $PHYSICS_HEAD"
    echo "Running in TEST MODE (random scores)"
    HEAD_FLAG="--test_mode"
else
    echo "Using physics head: $PHYSICS_HEAD"
    HEAD_FLAG="--physics_head $PHYSICS_HEAD --head_type $HEAD_TYPE"
fi

# ============================================================
# Run inference
# ============================================================

echo ""
echo "Configuration:"
echo "  Trajectories: $NUM_TRAJECTORIES (8 per GPU)"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoints: $CHECKPOINTS"
echo "  GPUs: cuda:0, cuda:1"
echo ""

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    
    echo ""
    echo "========================================"
    echo "[$((i+1))/${#PROMPTS[@]}] $PROMPT"
    echo "========================================"
    
    python inference/pruning.py \
        --prompt "$PROMPT" \
        $HEAD_FLAG \
        --num_trajectories $NUM_TRAJECTORIES \
        --batch_size $BATCH_SIZE \
        --checkpoints $CHECKPOINTS \
        --keep_ratio $KEEP_RATIO \
        --steps $STEPS \
        --frames $FRAMES \
        --guidance $GUIDANCE \
        --devices cuda:0 cuda:1 \
        --exp_name "pruning_$i" \
        --output_dir "$OUTPUT_BASE" \
        --seed $((42 + i))
        
done

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo "Output: $OUTPUT_BASE"
ls -la $OUTPUT_BASE/
echo ""
echo "End: $(date)"