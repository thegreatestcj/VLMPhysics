#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=16G
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --output=slurm/utils/videophy_cogx-%j.out

# ============================================================
# Download CogVideoX-2B videos from all VideoPhy datasets
# ============================================================

echo "========================================"
echo "Download VideoPhy CogVideoX Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================"

module purge
module load python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

cd ~/repos/VLMPhysics

OUTPUT_DIR="$HOME/scratch/physics/videophy_cogx"

# =============================================================
# Step 1: Install deps (if needed)
# =============================================================
pip install datasets requests --break-system-packages -q 2>/dev/null || true

# =============================================================
# Step 2: Dry run first to see what's available
# =============================================================
# echo ""
# echo "=== DRY RUN: Scanning all datasets ==="
# python utils/download_videophy_cogx.py \
#     --output-dir "$OUTPUT_DIR" \
#     --dry-run

# echo ""
# echo "=== Check metadata preview at ${OUTPUT_DIR}/metadata.json ==="
# echo ""

# =============================================================
# Step 3: Actual download
# Uncomment the block below after verifying dry run output
# =============================================================
echo "=== DOWNLOADING VIDEOS ==="
python utils/download_videophy_cogx.py \
    --output-dir "$OUTPUT_DIR" \
    --workers 8 \
    --include-5b \

echo "========================================"
echo "Done: $(date)"
echo "========================================"