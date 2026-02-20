#!/bin/bash
# =============================================================
# PhyGenBench Evaluation — API Stages (needs network, no GPU)
#
# Stage 2gpt: GPT-4o multi-frame verification (all 160 videos)
# Stage 3:    GPT-4o video naturalness (all 160 videos)
# Overall:    Aggregate final scores
#
# Run on login node or CPU partition WITH network access.
# Prerequisite: eval_phygenbench_gpu.sh must complete first.
#
# API cost: ~$8 total | Time: ~1.5h
# =============================================================

set -euo pipefail

echo "Start: $(date)"

VIDEOS_DIR="${1:-outputs/phygenbench/videophy_mean_l15_cogx}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "  export OPENAI_API_KEY='sk-xxx'"
    exit 1
fi

# Check that GPU stages completed
EVAL_DIR="$VIDEOS_DIR/eval"
if [ ! -f "$EVAL_DIR/stage1_results.json" ]; then
    echo "ERROR: stage1_results.json not found in $EVAL_DIR"
    echo "  Run eval_phygenbench_gpu.sh first!"
    exit 1
fi

if [ ! -f "$EVAL_DIR/stage2_clip_results.json" ]; then
    echo "WARNING: stage2_clip_results.json not found."
    echo "  24 non-Middle videos will be skipped in Stage 2."
    echo "  Run eval_phygenbench_gpu.sh with '2clip' stage to fix."
fi

echo "Evaluating: $VIDEOS_DIR"
echo "Stages: 2gpt + 3 + overall"
echo ""

python eval/eval_phygenbench.py \
    --videos-dir "$VIDEOS_DIR" \
    --stage 2gpt 3 overall

echo ""
echo "All stages complete: $(date)"
echo "Results: $EVAL_DIR/final_results.json"