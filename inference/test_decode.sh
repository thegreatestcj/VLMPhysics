#!/bin/bash
#SBATCH --job-name=traj_dual
#SBATCH --output=slurm/inference/dual_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# ============================================================
# Dual-GPU Trajectory Pruning - Maximum Parallelism
#
# Uses BOTH GPUs to maximize trajectory count:
#   - 16 trajectories total (8 per GPU)
#   - Mini-batch processing (batch_size=4)
#   - ~4x faster than sequential single-GPU
#
# Memory per GPU (~18GB used / 24GB available):
#   - DiT (bfloat16):     ~10GB
#   - VAE (tiling):        ~1GB
#   - 4 traj latents:     ~0.6GB
#   - Activations:         ~6GB
#   - Buffers:             ~1GB
#
# ============================================================

echo "========================================"
echo "Dual-GPU Trajectory Pruning"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi -L
echo "Start: $(date)"
echo "========================================"

# Directories
mkdir -p slurm/inference
mkdir -p results/inference/pruning

# Environment
module purge
module load cuda/12.1 python/3.10
source /users/ctang33/.conda/envs/physics/bin/activate

# Cache
export HF_HOME=/users/ctang33/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/users/ctang33/scratch/torch_cache
mkdir -p $HF_HOME $TORCH_HOME

cd ~/repos/VLMPhysics

python inference/test_decode.py