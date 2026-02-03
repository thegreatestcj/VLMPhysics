#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C geforce3090
#SBATCH --mem=48G
#SBATCH -n 4
#SBATCH -t 14:00:00
#SBATCH -N 1
#SBATCH --output=slurm/phygenbench/physics/slurm-%j.out

# ============================================================
# PhyGenBench Generation with Physics Head Trajectory Pruning
#
# Uses trajectory pruning guided by trained physics head.
#
# Time estimate (based on baseline ~204s/video):
#   - 4 trajectories (4->2->1): ~500s/video (~2.5x baseline)
#   - 8 trajectories (8->4->2->1): ~1000s/video (~5x baseline)
#
#   160 prompts @ 4 traj: ~22 hours (single GPU)
#   80 prompts @ 4 traj:  ~11 hours (split job)
#
# Usage:
#   # Full run (all 160 prompts) - ~22 hours
#   sbatch eval/generate_physics.sh
#
#   # Split into 2 jobs (run both) - ~11 hours each
#   sbatch eval/generate_physics.sh 0 80
#   sbatch eval/generate_physics.sh 80 160
#
# ============================================================

# Parse arguments (optional start/end)
START_IDX=${1:-0}
END_IDX=${2:-160}

echo "========================================"
echo "PhyGenBench Physics Head Generation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Range: $START_IDX - $END_IDX"
nvidia-smi -L
echo "Start: $(date)"
echo "========================================"

# Create directories
mkdir -p slurm/phygenbench/physics
mkdir -p outputs/phygenbench/physics

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

# Physics head (layer 10 is best based on ablation)
PHYSICS_HEAD="results/training/physics_head/heads_ablation_20260131_184635/causal_simple/best.pt"

# Check if physics head exists
if [ ! -f "$PHYSICS_HEAD" ]; then
    echo "ERROR: Physics head not found: $PHYSICS_HEAD"
    echo "Please train physics head first or update path."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Physics head: $PHYSICS_HEAD"
echo "  Trajectories: 4 -> 2 -> 1"
echo "  Checkpoints: t=600, 400"
echo "  Output: outputs/phygenbench/physics/"
echo ""

# ============================================================
# Run generation
# ============================================================

python eval/generate_physics.py \
    --prompts-file data/phygenbench/prompts.json \
    --output-dir outputs/phygenbench/videophy \
    --physics-head "$PHYSICS_HEAD" \
    --head-type temporal_simple \
    --extract-layer 10 \
    --num-trajectories 4 \
    --checkpoints 600 400 \
    --steps 50 \
    --frames 49 \
    --guidance 6.0 \
    --seed 42 \
    --start 80 \
    --end 160 \
    --skip-existing

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
ls -la outputs/phygenbench/physics/ | head -20
echo ""
echo "Video count:"
ls outputs/phygenbench/physics/*.mp4 2>/dev/null | wc -l
echo ""
echo "Log file:"
cat outputs/phygenbench/physics/generation_log.json | python -c "
import json, sys
data = json.load(sys.stdin)
success = sum(1 for r in data if r.get('status') == 'success')
total_time = sum(r.get('time', 0) for r in data)
print(f'  Successful: {success}/{len(data)}')
print(f'  Total time: {total_time/60:.1f} min')
if success > 0:
    print(f'  Avg time/video: {total_time/success:.1f}s')
"