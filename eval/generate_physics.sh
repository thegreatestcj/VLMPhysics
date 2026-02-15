#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C geforce3090
#SBATCH --mem=48G
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --output=slurm/phygenbench/physics/traj8-%j.out

# ============================================================
# PhyGenBench Generation with Physics Head Trajectory Pruning
#
# Uses mean head (layer 15) with trajectory pruning.
#
# Time estimate (based on baseline ~204s/video):
#   - 4 trajectories (4->2->1): ~500s/video (~2.5x baseline)
#   - 8 trajectories (8->4->2->1): ~1000s/video (~5x baseline)
#
#   160 prompts @ 4 traj: ~22 hours (single GPU)
#   80 prompts @ 4 traj:  ~11 hours (split job)
#
# Usage:
#   # Full run (all 160 prompts)
#   sbatch eval/generate_physics.sh
#
#   # Split into 2 jobs (run both)
#   sbatch eval/generate_physics.sh 0 80
#   sbatch eval/generate_physics.sh 80 160
#
# ============================================================

echo "========================================"
echo "PhyGenBench Physics Head Generation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi -L
echo "Start: $(date)"
echo "========================================"

# Create directories
mkdir -p slurm/phygenbench/physics

# Environment
module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

# Cache
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HOME/scratch/.cache/huggingface/hub"
export TORCH_HOME="$HOME/scratch/.cache/torch"
mkdir -p $HF_HOME $TORCH_HOME

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

# Best config from ablation: mean head, layer 15, with SA multi-task
# UPDATE this path to your actual best checkpoint
PHYSICS_HEAD="results/training/final_head/mean_l15_full.pt"
HEAD_TYPE="mean"
EXTRACT_LAYER=15

# Mode: "prune" (ours), "best_of_n" (baseline), "random" (ablation)
MODE="prune"

OUTPUT_DIR="outputs/phygenbench/videophy_mean_l15_full_traj8"

# Check if physics head exists
if [ ! -f "$PHYSICS_HEAD" ]; then
    echo "ERROR: Physics head not found: $PHYSICS_HEAD"
    echo ""
    echo "Please copy your best checkpoint:"
    echo "  mkdir -p results/training/final_head"
    echo "  cp results/training/physics_head/<your_run>/mean/best.pt $PHYSICS_HEAD"
    echo ""
    echo "Or update PHYSICS_HEAD path in this script."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Physics head:   $PHYSICS_HEAD"
echo "  Head type:      $HEAD_TYPE"
echo "  Extract layer:  $EXTRACT_LAYER"
echo "  Mode:           $MODE"
echo "  Trajectories:   4"
echo "  Output:         $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Run generation
#
# Comment/uncomment ONE of the two blocks below, then:
#   sbatch eval/generate_physics.sh   (submit twice, one per block)
#
# To run best_of_n baseline:
#   Change MODE="best_of_n" above and OUTPUT_DIR accordingly
# ============================================================

# --- Block A: prompts 0-79 (~11 hours for prune, ~22 hours for best_of_n) ---
python eval/generate_physics.py \
    --prompts-file data/phygenbench/prompts.json \
    --output-dir "$OUTPUT_DIR" \
    --physics-head "$PHYSICS_HEAD" \
    --head-type "$HEAD_TYPE" \
    --extract-layer $EXTRACT_LAYER \
    --mode "$MODE" \
    --num-trajectories 8 \
    --checkpoints 600 400 200 \
    --steps 50 \
    --frames 49 \
    --guidance 6.0 \
    --seed 42 \
    --start 0 \
    --end 80 \
    --skip-existing
    --force-head-type

# --- Block B: prompts 80-159 ---
# python eval/generate_physics.py \
#     --prompts-file data/phygenbench/prompts.json \
#     --output-dir "$OUTPUT_DIR" \
#     --physics-head "$PHYSICS_HEAD" \
#     --head-type "$HEAD_TYPE" \
#     --extract-layer $EXTRACT_LAYER \
#     --mode "$MODE" \
#     --num-trajectories 8 \
#     --checkpoints 600 400 200 \
#     --steps 50 \
#     --frames 49 \
#     --guidance 6.0 \
#     --seed 42 \
#     --start 80 \
#     --end 160 \
#     --skip-existing \
#     --force-head-type

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================"
echo "Generation Complete"
echo "========================================"
echo "End: $(date)"
echo ""
echo "Output directory:"
ls -la "$OUTPUT_DIR"/ | head -20
echo ""
echo "Video count:"
ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l
echo ""

LOG_FILE="$OUTPUT_DIR/generation_log.json"
if [ -f "$LOG_FILE" ]; then
    echo "Log summary:"
    cat "$LOG_FILE" | python -c "
import json, sys
data = json.load(sys.stdin)
success = sum(1 for r in data if r.get('status', '').startswith('success'))
total_time = sum(r.get('time', 0) for r in data)
print(f'  Successful: {success}/{len(data)}')
print(f'  Total time: {total_time/60:.1f} min')
if success > 0:
    print(f'  Avg time/video: {total_time/success:.1f}s')
"
else
    echo "No log file found yet."
fi