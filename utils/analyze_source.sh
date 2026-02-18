#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --output=slurm/analysis/source/slurm-%j.out

# ============================================================
# Source Confounding Analysis (Full Suite)
#
# 1. Conditional Fisher
# 2. Per-Source AUC (LogReg + MLP)
# 3. Cross-Source Generalization Matrix
# 4. Permutation Test (statistical significance)
# 5. Per-Category Probing (caption confound control)
# 6. Timestep Progression (physics emergence)
# ============================================================

echo "========================================"
echo "Source Confounding Analysis (Full)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================"

mkdir -p slurm/analysis/source figures/source

module purge
source /users/ctang33/.conda/envs/physics/bin/activate

cd ~/repos/VLMPhysics

# =============================================================
# >>>  CONFIGURE HERE  <<<
# =============================================================

FEATURE_DIR="$HOME/scratch/physics/videophy_features_2"
METADATA="$HOME/scratch/physics/videophy_data/metadata.json"
OUTPUT="figures/source/source"

LAYER=20
TIMESTEP=200

N_PERMUTATIONS=200

# Timesteps for progression analysis (auto-detected if not set)
# TIMESTEPS="--timesteps 200 400 600 800"
TIMESTEPS=""

# Uncomment to skip specific analyses
SKIP_CROSS="--skip-cross"
SKIP_PERM="--skip-permutation"
SKIP_CAT="--skip-category"
SKIP_TS="--skip-timestep"

# =============================================================

python utils/analyze_source.py \
    --feature-dir "$FEATURE_DIR" \
    --metadata "$METADATA" \
    --output "$OUTPUT" \
    --layer $LAYER \
    --timestep $TIMESTEP \
    --n-permutations $N_PERMUTATIONS \
    ${TIMESTEPS:-} \
    ${SKIP_CROSS:-} \
    ${SKIP_PERM:-} \
    ${SKIP_CAT:-} \
    ${SKIP_TS:-}

echo "========================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh ${OUTPUT}_*.png ${OUTPUT}_*.json 2>/dev/null
echo "========================================"