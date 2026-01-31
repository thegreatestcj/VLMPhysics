#!/bin/bash
#SBATCH --job-name=phygenbench_eval
#SBATCH --output=slurm/eval/phygenbench_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -N 1

# ============================================================
# PhyGenBench Evaluation - SLURM Version
#
# Usage:
#   export OPENAI_API_KEY="sk-xxx"
#   sbatch scripts/eval_phygenbench_slurm.sh baseline
#   sbatch scripts/eval_phygenbench_slurm.sh physics
#
# Cost: ~$3.20 per 160 videos
# Time: ~20-30 minutes
# ============================================================

# NOTE: Do NOT use "set -e" here - we want the script to continue
# even if some stages fail

# Get method from command line argument
METHOD="${1:-baseline}"

echo "========================================"
echo "PhyGenBench Evaluation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Method: $METHOD"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# ============================================================
# Environment Setup (robust version)
# ============================================================

# Create log directory
mkdir -p slurm/eval

# Load modules (ignore errors)
module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

# --- Robust Conda Initialization ---
# Method 1: Try conda.sh from common locations
for CONDA_BASE in \
    "/users/$USER/.conda" \
    "/users/$USER/anaconda3" \
    "/users/$USER/miniconda3" \
    "/oscar/runtime/software/external/miniconda3/24.3.0" \
    "$CONDA_PREFIX" \
    "$HOME/.conda" \
    "$HOME/anaconda3" \
    "$HOME/miniconda3"
do
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "Initializing conda from: $CONDA_BASE"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        break
    fi
done

# Method 2: If conda command exists, use it
if ! command -v conda &> /dev/null; then
    # Try to find conda executable
    for CONDA_BIN in \
        "/users/$USER/.conda/bin/conda" \
        "/users/$USER/anaconda3/bin/conda" \
        "/users/$USER/miniconda3/bin/conda"
    do
        if [ -x "$CONDA_BIN" ]; then
            eval "$($CONDA_BIN shell.bash hook)"
            break
        fi
    done
fi

# Now try to activate the environment
if command -v conda &> /dev/null; then
    echo "Conda found: $(which conda)"
    
    # Try to activate physics environment
    if conda activate physics 2>/dev/null; then
        echo "✓ Activated conda env: physics"
    elif conda activate base 2>/dev/null; then
        echo "⚠ Activated conda env: base (physics not found)"
    else
        echo "⚠ Could not activate conda environment"
    fi
else
    echo "⚠ Conda not found, using system Python"
fi

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Set cache directories
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" 2>/dev/null || true

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Set it before submitting:"
    echo "  export OPENAI_API_KEY='sk-xxx'"
    echo "  sbatch scripts/eval_phygenbench_slurm.sh $METHOD"
    exit 1
fi
echo "✓ OPENAI_API_KEY is set"

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/repos/VLMPhysics}"
PHYGENBENCH_DIR="$PROJECT_ROOT/PhyGenBench"
VIDEOS_DIR="$PROJECT_ROOT/outputs/phygenbench/$METHOD"
RESULTS_DIR="$PROJECT_ROOT/results/phygenbench/$METHOD"

echo ""
echo "Paths:"
echo "  Project:     $PROJECT_ROOT"
echo "  PhyGenBench: $PHYGENBENCH_DIR"
echo "  Videos:      $VIDEOS_DIR"
echo "  Results:     $RESULTS_DIR"

# Verify directories exist
if [ ! -d "$PHYGENBENCH_DIR" ]; then
    echo ""
    echo "ERROR: PhyGenBench not found at: $PHYGENBENCH_DIR"
    echo "Run: cd $PROJECT_ROOT && git clone https://github.com/OpenGVLab/PhyGenBench.git"
    exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
    echo ""
    echo "ERROR: Videos directory not found: $VIDEOS_DIR"
    echo "Make sure you have generated videos for method: $METHOD"
    exit 1
fi

# Count videos
NUM_VIDEOS=$(ls "$VIDEOS_DIR"/output_video_*.mp4 2>/dev/null | wc -l)
echo "  Videos found: $NUM_VIDEOS"

if [ "$NUM_VIDEOS" -eq 0 ]; then
    echo "ERROR: No videos found matching pattern: output_video_*.mp4"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# ============================================================
# Setup Symbolic Link
# ============================================================

echo ""
echo "Setting up PhyVideos symbolic link..."
cd "$PHYGENBENCH_DIR"

# Remove existing PhyVideos (file, link, or directory)
rm -rf PhyVideos 2>/dev/null || true

# Create new symbolic link
ln -s "$VIDEOS_DIR" PhyVideos

# Verify
if [ -L "PhyVideos" ] && [ -d "PhyVideos" ]; then
    echo "✓ PhyVideos -> $VIDEOS_DIR"
else
    echo "ERROR: Failed to create symbolic link"
    exit 1
fi

# ============================================================
# Stage 1: VQAScore (requires GPU)
# ============================================================

echo ""
echo "========================================"
echo "Stage 1: VQAScore (Key Physical Phenomena)"
echo "========================================"

cd "$PHYGENBENCH_DIR"

if [ -f "PhyGenEval/single/vqascore.py" ]; then
    echo "Running vqascore.py..."
    python PhyGenEval/single/vqascore.py
    if [ $? -eq 0 ]; then
        echo "✓ Stage 1 complete"
    else
        echo "⚠ Stage 1 failed, continuing..."
    fi
else
    echo "⚠ vqascore.py not found, skipping Stage 1"
fi

# ============================================================
# Stage 2: Multi-frame QA (GPT-4o)
# ============================================================

echo ""
echo "========================================"
echo "Stage 2: Physics Order Verification"
echo "========================================"

# Step 2a: CLIP keyframe retrieval
echo "Step 2a: CLIP keyframe retrieval..."
if [ -f "PhyGenEval/multi/multiimage_clip.py" ]; then
    python PhyGenEval/multi/multiimage_clip.py
    if [ $? -eq 0 ]; then
        echo "✓ Step 2a complete"
    else
        echo "⚠ Step 2a failed, continuing..."
    fi
else
    echo "⚠ multiimage_clip.py not found, skipping"
fi

# Step 2b: GPT-4o multi-frame QA
echo ""
echo "Step 2b: GPT-4o multi-frame QA..."
if [ -f "PhyGenEval/multi/GPT4o.py" ]; then
    python PhyGenEval/multi/GPT4o.py
    if [ $? -eq 0 ]; then
        echo "✓ Step 2b complete"
    else
        echo "⚠ Step 2b failed, continuing..."
    fi
else
    echo "⚠ GPT4o.py (multi) not found, skipping"
fi

echo "✓ Stage 2 complete"

# ============================================================
# Stage 3: Video QA (GPT-4o)
# ============================================================

echo ""
echo "========================================"
echo "Stage 3: Overall Naturalness"
echo "========================================"

if [ -f "PhyGenEval/video/GPT4o.py" ]; then
    python PhyGenEval/video/GPT4o.py
    if [ $? -eq 0 ]; then
        echo "✓ Stage 3 complete"
    else
        echo "⚠ Stage 3 failed, continuing..."
    fi
else
    echo "⚠ GPT4o.py (video) not found, skipping"
fi

# ============================================================
# Overall Score
# ============================================================

echo ""
echo "========================================"
echo "Computing Overall Score"
echo "========================================"

if [ -f "PhyGenEval/overall.py" ]; then
    python PhyGenEval/overall.py
    if [ $? -eq 0 ]; then
        echo "✓ Overall score computed"
    else
        echo "⚠ overall.py failed"
    fi
else
    echo "⚠ overall.py not found, skipping"
fi

# ============================================================
# Save Results
# ============================================================

echo ""
echo "========================================"
echo "Saving Results"
echo "========================================"

cd "$PHYGENBENCH_DIR"

# Copy result directory if exists
if [ -d "result" ]; then
    cp -r result/* "$RESULTS_DIR/" 2>/dev/null
    echo "Copied result/ contents"
fi

# Copy any recently modified JSON files
find . -maxdepth 3 -name "*.json" -mmin -120 -exec cp {} "$RESULTS_DIR/" \; 2>/dev/null

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Contents:"
ls -la "$RESULTS_DIR/" 2>/dev/null || echo "  (empty)"

# ============================================================
# Done
# ============================================================

echo ""
echo "========================================"
echo "EVALUATION COMPLETE"
echo "========================================"
echo "Method:  $METHOD"
echo "Results: $RESULTS_DIR"
echo "End:     $(date)"
echo "========================================"