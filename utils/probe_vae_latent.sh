#!/bin/bash
#SBATCH --job-name=probe_vae
#SBATCH --output=slurm/analysis/vae/probe_vae_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

mkdir -p slurm

module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

export HF_HOME=/users/$USER/scratch/hf_cache
cd ~/repos/VLMPhysics

# ============================================================
# Multi-source (videophy_data, ~3500 videos)
# Compare VAE latent AUC vs DiT feature AUC across all sources
# ============================================================

for layer in 5 10 15 20 25; do
  for t in 200 400 600; do
    python utils/probe_vae_latent.py \
      --load-latents ~/scratch/physics/vae_latents_cogx_2b5b.npz \
      --dit-feature-dir ~/scratch/physics/videophy_cogx_features \
      --metadata ~/scratch/physics/videophy_cogx/metadata.json \
      --dit-layer $layer --dit-timestep $t \
      --skip-umap --skip-cross-source \
      --output figures/grid_l${layer}_t${t}.png
    echo "=== layer=$layer t=$t done ==="
  done
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="