#!/bin/bash
#SBATCH --job-name=phygenbench_gpu
#SBATCH --output=slurm/eval/phygen_gpu_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -N 1

# =============================================================
# PhyGenBench Evaluation — GPU Stages (no network access needed)
#
# Stage 1:    VQAScore + CLIP retrieval for ALL 160 videos
# Stage 2clip: CLIP + VQAScore for 24 non-Middle videos
#
# VRAM: ~16-18GB (VQAScore ~12-15GB + CLIP ~1GB)
# Time: ~2.5h total
# =============================================================

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load cuda/12.1.1 2>/dev/null || true

# Reduce CUDA memory fragmentation on 24GB GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VIDEOS_DIR="${1:-outputs/phygenbench/videophy_mean_l15_cogx}"

echo "Evaluating: $VIDEOS_DIR"
echo "Stages: 1 (VQAScore) + 2clip (CLIP preprocessing)"
echo ""

python eval/eval_phygenbench.py \
    --videos-dir "$VIDEOS_DIR" \
    --stage 2clip

echo ""
echo "GPU stages complete: $(date)"
echo "Results: $VIDEOS_DIR/eval/"
echo "  stage1_results.json       — Stage 1 scores"
echo "  stage2_clip_results.json  — CLIP data for non-Middle videos"
echo ""
echo "Next: run eval_phygenbench_api.sh for Stage 2 GPT + Stage 3"







