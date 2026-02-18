#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --output=slurm/analysis/source/cross_source_dit-%j.out

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

module purge
source activate physics 2>/dev/null || conda activate physics 2>/dev/null || true
cd ~/repos/VLMPhysics

mkdir -p slurm/analysis

FEAT_DIR=~/scratch/physics/videophy_cogx_features
META=~/scratch/physics/videophy_cogx/metadata.json
VAE_LATENTS=~/scratch/physics/vae_latents_cogx_2b5b.npz

mkdir -p figures

echo "=========================================="
echo "1. Cross-source: VAE vs DiT (L10, t=600)"
echo "=========================================="
python utils/cross_source_dit.py \
    --vae-latents $VAE_LATENTS \
    --dit-feature-dir $FEAT_DIR \
    --metadata $META \
    --dit-layer 10 --dit-timestep 600 \
    --output figures/cross_source_dit_vs_vae.png

echo ""
echo "=========================================="
echo "2. Combined training test (L10, t=600)"
echo "=========================================="
python utils/combined_training_test.py \
    --dit-feature-dir $FEAT_DIR \
    --metadata $META \
    --dit-layer 10 --dit-timestep 600 \
    --output figures/combined_training_test.json

echo ""
echo "=========================================="
echo "3. Combined training test (L10, t=400)"
echo "=========================================="
python utils/combined_training_test.py \
    --dit-feature-dir $FEAT_DIR \
    --metadata $META \
    --dit-layer 10 --dit-timestep 400 \
    --output figures/combined_training_test_t400.json

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="