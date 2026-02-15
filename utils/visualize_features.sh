#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --output=slurm/viz/slurm-%j.out

# ============================================================
# Feature Space Visualization (CPU only, no GPU needed)
#
# Generates:
#   1. Separation heatmap (Fisher ratio + cosine distance)
#   2. t-SNE/UMAP grid (layers × timesteps)
#   3. Category scatter plots (colored by any metadata field)
#
# Supported --color-by fields:
#   physics           Physics plausibility (0/1)
#   sa                Semantic alignment (0/1)
#   source            Generation model (lavie, pika, ...)
#   states_of_matter  Material type (fluid_fluid, ...)
# ============================================================

echo "========================================"
echo "Feature Visualization"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================"

mkdir -p slurm/viz figures

module purge
module load python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

cd ~/repos/VLMPhysics

# Install umap if needed
pip install umap-learn --quiet --break-system-packages 2>/dev/null || true

# =============================================================
# >>>  CONFIGURE HERE  <<<
# =============================================================

FEATURE_DIR="/users/$USER/scratch/physics/videophy_features_2"
METADATA="/users/$USER/scratch/physics/videophy_data/metadata.json"
OUTPUT="figures/features"

# Heatmap + grid: all layer × timestep combos
LAYERS="10 15 20 25"
TIMESTEPS="200 400 600"

# Category plots: which (layer, timestep) pairs + which fields
COLOR_BY="physics sa source states_of_matter"
CAT_LAYERS="15 20"
CAT_TIMESTEPS="200"

# Options: umap, tsne, both
METHOD="umap"

# Skip flags (uncomment to skip)
SKIP_HEATMAP="--skip-heatmap"
SKIP_GRID="--skip-grid"
# SKIP_CATEGORY="--skip-category"

# =============================================================

python utils/visualize_features.py \
    --feature-dir "$FEATURE_DIR" \
    --metadata "$METADATA" \
    --output "$OUTPUT" \
    --layers $LAYERS \
    --timesteps $TIMESTEPS \
    --color-by $COLOR_BY \
    --category-layers $CAT_LAYERS \
    --category-timesteps $CAT_TIMESTEPS \
    --method "$METHOD" \
    --seed 42 \
    ${SKIP_HEATMAP:-} \
    ${SKIP_GRID:-} \
    ${SKIP_CATEGORY:-}

echo "========================================"
echo "Done: $(date)"
echo "Outputs in: figures/"
echo "========================================"