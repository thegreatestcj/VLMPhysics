#!/bin/bash
#SBATCH --job-name=phygenbench_eval
#SBATCH --output=slurm/eval/phygenbench_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -N 1

# ============================================================
# PhyGenBench 3-Stage Evaluation
# ============================================================
#
# USAGE:
#   export OPENAI_API_KEY="sk-xxx"
#   sbatch eval/eval_phygenbench.sh
#
# DEPENDENCIES:
#   Python packages:
#     - t2v_metrics (for VQAScore in Stage 1)
#     - openai (for GPT-4o in Stages 2 & 3)
#     - opencv-python, numpy, tqdm
#
#   Data files (in data/phygenbench/):
#     - single_question.json  (Stage 1 questions)
#     - multi_question.json   (Stage 2 questions)
#     - video_question.json   (Stage 3 questions)
#
#   Input videos:
#     - outputs/phygenbench/baseline/output_video_*.mp4 (160 videos)
#
# OUTPUT:
#   results/evaluation/phygenbench/eval_YYYYMMDD_HHMMSS/
#     - results.json   (per-video detailed results)
#     - summary.json   (aggregated statistics)
#
# COST: ~$3-5 for 160 videos
# TIME: ~1-2 hours
# ============================================================

set -e

echo "========================================"
echo "PhyGenBench Evaluation"
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "========================================"

# ============================================================
# Environment Setup
# ============================================================

# Create log directory
mkdir -p slurm/eval

# Load CUDA
module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

# Initialize conda
for CONDA_BASE in \
    "/users/$USER/.conda" \
    "/users/$USER/anaconda3" \
    "/users/$USER/miniconda3" \
    "/oscar/runtime/software/external/miniconda3/24.3.0"
do
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        break
    fi
done

# Activate environment
conda activate physics 2>/dev/null || conda activate base 2>/dev/null || true
echo "Python:    $(which python)"
echo "Version:   $(python --version 2>&1)"

# Set HuggingFace cache
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME" 2>/dev/null || true

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/repos/VLMPhysics}"
cd "$PROJECT_ROOT"

# Input/Output paths
VIDEOS_DIR="outputs/phygenbench/videophy"
OUTPUT_DIR="results/eval/phygenbench"

# Stages to run (1=VQAScore, 2=MultiFrame GPT-4o, 3=Video GPT-4o)
STAGES="1 2 3"

echo ""
echo "Configuration:"
echo "  Project:   $PROJECT_ROOT"
echo "  Videos:    $VIDEOS_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Stages:    $STAGES"

# ============================================================
# Verify Prerequisites
# ============================================================

echo ""
echo "Checking prerequisites..."

# Check videos
NUM_VIDEOS=$(ls "$VIDEOS_DIR"/output_video_*.mp4 2>/dev/null | wc -l)
echo "  Videos found: $NUM_VIDEOS"
if [ "$NUM_VIDEOS" -eq 0 ]; then
    echo "ERROR: No videos found in $VIDEOS_DIR"
    exit 1
fi

# Check question files
for FILE in "data/phygenbench/single_question.json" \
            "data/phygenbench/multi_question.json" \
            "data/phygenbench/video_question.json"; do
    if [ -f "$FILE" ]; then
        echo "  ✓ $FILE"
    else
        echo "  ✗ $FILE (MISSING)"
        exit 1
    fi
done

# Check API key (required for Stages 2 & 3)
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY not set!"
    echo ""
    echo "Set it before submitting:"
    echo "  export OPENAI_API_KEY='sk-xxx'"
    echo "  sbatch eval/run_eval_phygenbench.sh"
    exit 1
fi
echo "  ✓ OPENAI_API_KEY is set"

# Check t2v_metrics (for Stage 1)
if python -c "import t2v_metrics" 2>/dev/null; then
    echo "  ✓ t2v_metrics installed"
else
    echo "  ⚠ t2v_metrics not found (Stage 1 will fail)"
    echo "    Install with: pip install t2v-metrics"
fi

# ============================================================
# Run Evaluation
# ============================================================

echo ""
echo "========================================"
echo "Running Evaluation"
echo "========================================"

python eval/eval_phygenbench.py \
    --videos-dir "$VIDEOS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --stage $STAGES

# ============================================================
# Show Results
# ============================================================

echo ""
echo "========================================"
echo "Evaluation Complete"
echo "========================================"
echo "End: $(date)"

# Find latest results directory
LATEST=$(ls -td "$OUTPUT_DIR"/eval_* 2>/dev/null | head -1)

if [ -n "$LATEST" ] && [ -f "$LATEST/summary.json" ]; then
    echo ""
    echo "Results saved to: $LATEST"
    echo ""
    echo "Summary:"
    python -c "
import json
with open('$LATEST/summary.json') as f:
    s = json.load(f)
print(f\"  Total videos:  {s.get('total_videos', 'N/A')}\")
print(f\"  Valid videos:  {s.get('valid_videos', 'N/A')}\")
print()
print('  Per-Stage Averages:')
if s.get('stage1_avg') is not None:
    print(f\"    Stage 1 (VQAScore):    {s['stage1_avg']:.4f}\")
if s.get('stage2_avg') is not None:
    print(f\"    Stage 2 (Multi-frame): {s['stage2_avg']:.4f}\")
if s.get('stage3_avg') is not None:
    print(f\"    Stage 3 (Naturalness): {s['stage3_avg']:.4f}\")
if s.get('overall_avg') is not None:
    print()
    print(f\"    Overall Score:         {s['overall_avg']:.4f}\")
print()
if s.get('per_category'):
    print('  Per-Category:')
    for cat, score in sorted(s['per_category'].items()):
        print(f\"    {cat}: {score:.4f}\")
"
fi