#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --output=slurm/analysis/quality_probe-%j.out

# ============================================================
# Quality-Controlled Physics Probing
#
# Prerequisite: Run compute_vqascore.sh first to generate
#   ~/scratch/physics/videophy_cogx/vqascores.json
#
# This script is CPU-only and fast (~5 minutes).
# ============================================================

echo "========================================"
echo "Quality-Controlled Physics Probing"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================"

mkdir -p slurm/analysis figures/quality_control

module purge
module load python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

cd ~/repos/VLMPhysics

# ============================================================
# Configuration
# ============================================================

FEATURE_DIR="$HOME/scratch/physics/videophy_cogx_features"
METADATA="$HOME/scratch/physics/videophy_cogx/metadata.json"
VQASCORES="$HOME/scratch/physics/videophy_cogx/vqascores.json"
OUTPUT="figures/quality_control/qc"
LAYER=10

# ============================================================
# Verify prerequisites
# ============================================================

if [ ! -f "$VQASCORES" ]; then
    echo "ERROR: VQAScore results not found: $VQASCORES"
    echo "Run compute_vqascore.sh first!"
    exit 1
fi

echo "VQAScore file: $VQASCORES"
echo "  Entries: $(python -c "import json; print(len(json.load(open('$VQASCORES'))))")"

# ============================================================
# Run analysis
# ============================================================

python utils/quality_controlled_probe.py \
    --feature-dir "$FEATURE_DIR" \
    --metadata "$METADATA" \
    --vqascores "$VQASCORES" \
    --output "$OUTPUT" \
    --layer $LAYER \
    --timesteps 200 400 600

echo "========================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh figures/quality_control/qc_*.png figures/quality_control/qc_*.json 2>/dev/null
echo "========================================"