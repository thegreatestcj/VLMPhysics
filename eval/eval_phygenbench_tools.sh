#!/bin/bash
# =============================================================
# PhyGenBench Evaluation — Setup Data + Overall + Status
# =============================================================

set -euo pipefail

CMD="${1:-status}"

case "$CMD" in
setup)
    # Copy question files from official repo
    echo "Setting up PhyGenBench data files..."
    mkdir -p data/phygenbench

    REPO="PhyGenBench/PhyGenBench"  # Official repo path
    if [ ! -d "$REPO" ]; then
        echo "ERROR: Official repo not found at $REPO"
        echo "  git clone https://github.com/OpenGVLab/PhyGenBench.git"
        exit 1
    fi

    for f in single_question.json multi_question.json video_question.json prompts.json; do
        if [ -f "$REPO/$f" ]; then
            cp "$REPO/$f" "data/phygenbench/$f"
            echo "  Copied: $f"
        else
            echo "  MISSING: $REPO/$f"
        fi
    done

    # CRITICAL: explicit_prompts.json (needed for Stage 2 Q3)
    EP="PhyGenBench/PhyGenEval/multi/explicit_prompts.json"
    if [ -f "$EP" ]; then
        cp "$EP" "data/phygenbench/explicit_prompts.json"
        echo "  Copied: explicit_prompts.json (Stage 2 Q3)"
    else
        echo "  MISSING: $EP — Stage 2 Q3 will be empty!"
    fi

    echo "Done. Files in data/phygenbench/:"
    ls -la data/phygenbench/
    ;;

overall)
    # Run overall aggregation for all methods
    source activate physics
    for dir in outputs/phygenbench/*/; do
        name=$(basename "$dir")
        evdir="$dir/eval"
        if [ -f "$evdir/stage1_results.json" ] && \
           [ -f "$evdir/stage2_results.json" ] && \
           [ -f "$evdir/stage3_results.json" ]; then
            echo "=== $name ==="
            python eval/eval_phygenbench.py --videos-dir "$dir" --stage overall
            echo ""
        fi
    done
    ;;

status)
    # Show completion status for all methods
    echo "PhyGenBench Evaluation Status"
    echo "============================="
    echo ""
    printf "%-40s %6s %6s %6s %6s %8s\n" "Method" "S1" "S2clip" "S2gpt" "S3" "Score"
    printf "%-40s %6s %6s %6s %6s %8s\n" "------" "--" "------" "-----" "--" "-----"

    for dir in outputs/phygenbench/*/; do
        name=$(basename "$dir")
        evdir="$dir/eval"

        s1="—"; s2c="—"; s2="—"; s3="—"; score="—"

        if [ -f "$evdir/stage1_results.json" ]; then
            s1=$(python3 -c "
import json
with open('$evdir/stage1_results.json') as f:
    d = json.load(f)
print(f'{len(d)}/160')
" 2>/dev/null || echo "err")
        fi

        if [ -f "$evdir/stage2_clip_results.json" ]; then
            s2c=$(python3 -c "
import json
with open('$evdir/stage2_clip_results.json') as f:
    d = json.load(f)
print(f'{len(d)}/24')
" 2>/dev/null || echo "err")
        fi

        if [ -f "$evdir/stage2_results.json" ]; then
            s2=$(python3 -c "
import json
with open('$evdir/stage2_results.json') as f:
    d = json.load(f)
print(f'{len(d)}/160')
" 2>/dev/null || echo "err")
        fi

        if [ -f "$evdir/stage3_results.json" ]; then
            s3=$(python3 -c "
import json
with open('$evdir/stage3_results.json') as f:
    d = json.load(f)
print(f'{len(d)}/160')
" 2>/dev/null || echo "err")
        fi

        if [ -f "$evdir/final_results.json" ]; then
            score=$(python3 -c "
import json
with open('$evdir/final_results.json') as f:
    d = json.load(f)
s = d['summary']
cats = s.get('per_category', {})
parts = []
for c in ['Mechanics','Optics','Thermal','Material']:
    if c in cats:
        parts.append(f'{cats[c]:.3f}')
parts.append(f'Avg={s[\"final_score\"]:.3f}')
print(' '.join(parts))
" 2>/dev/null || echo "err")
        fi

        printf "%-40s %6s %6s %6s %6s %s\n" "$name" "$s1" "$s2c" "$s2" "$s3" "$score"
    done

    echo ""
    echo "Ref (CogVideoX-2B, paper ENSEMBLE): Mech=0.38 Opti=0.43 Ther=0.34 Mate=0.39 Avg=0.37"
    echo ""
    echo "Workflow:"
    echo "  1. sbatch eval/eval_phygenbench_gpu.sh <videos_dir>  (GPU: S1 + S2clip)"
    echo "  2. bash eval/eval_phygenbench_api.sh <videos_dir>    (API: S2gpt + S3 + overall)"
    ;;

*)
    echo "Usage: $0 {setup|overall|status}"
    ;;
esac