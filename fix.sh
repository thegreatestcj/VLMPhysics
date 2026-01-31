#!/bin/bash
# Fix PhyGenBench dependencies and hardcoded paths
# Run this on a login node (not in SLURM job)

set -e

echo "=========================================="
echo "Fixing PhyGenBench Dependencies"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate physics

# 1. Fix pytorchvideo + torchvision compatibility
echo ""
echo "[1/4] Fixing pytorchvideo compatibility..."
echo "The issue: pytorchvideo uses deprecated torchvision.transforms.functional_tensor"
echo "Solution: Patch the pytorchvideo source file directly"

PYTORCHVIDEO_PATH=$(python -c "import pytorchvideo; print(pytorchvideo.__path__[0])")
AUGMENT_FILE="$PYTORCHVIDEO_PATH/transforms/augmentations.py"

if [ -f "$AUGMENT_FILE" ]; then
    # Check if already patched
    if grep -q "functional_tensor" "$AUGMENT_FILE"; then
        echo "Patching $AUGMENT_FILE..."
        # Replace the deprecated import with the new one
        sed -i 's/import torchvision.transforms.functional_tensor as F_t/import torchvision.transforms.functional as F_t/' "$AUGMENT_FILE"
        echo "✓ Patched pytorchvideo"
    else
        echo "✓ pytorchvideo already patched or using different import"
    fi
else
    echo "⚠ Could not find pytorchvideo augmentations.py"
fi

# 2. Install missing dependencies
echo ""
echo "[2/4] Installing missing dependencies..."
pip install moviepy --break-system-packages --quiet
echo "✓ Installed moviepy"

# 3. Fix hardcoded paths in PhyGenBench
echo ""
echo "[3/4] Fixing hardcoded paths in PhyGenBench..."

PHYGENBENCH_DIR="$HOME/repos/VLMPhysics/PhyGenBench"

if [ -d "$PHYGENBENCH_DIR" ]; then
    # Fix GPT4o.py in multi/
    MULTI_GPT4O="$PHYGENBENCH_DIR/PhyGenEval/multi/GPT4o.py"
    if [ -f "$MULTI_GPT4O" ]; then
        sed -i "s|/PhyGenBench/PhyGenBench/|$PHYGENBENCH_DIR/PhyGenBench/|g" "$MULTI_GPT4O"
        sed -i "s|/PhyGenBench/PhyGenEval/|$PHYGENBENCH_DIR/PhyGenEval/|g" "$MULTI_GPT4O"
        echo "✓ Fixed $MULTI_GPT4O"
    fi
    
    # Fix GPT4o.py in video/
    VIDEO_GPT4O="$PHYGENBENCH_DIR/PhyGenEval/video/GPT4o.py"
    if [ -f "$VIDEO_GPT4O" ]; then
        sed -i "s|/PhyGenBench/PhyGenBench/|$PHYGENBENCH_DIR/PhyGenBench/|g" "$VIDEO_GPT4O"
        sed -i "s|/PhyGenBench/PhyGenEval/|$PHYGENBENCH_DIR/PhyGenEval/|g" "$VIDEO_GPT4O"
        sed -i "s|/PhyGenBench/PhyVideos/|$PHYGENBENCH_DIR/PhyVideos/|g" "$VIDEO_GPT4O"
        echo "✓ Fixed $VIDEO_GPT4O"
    fi
    
    # Fix vqascore.py
    VQASCORE="$PHYGENBENCH_DIR/PhyGenEval/single/vqascore.py"
    if [ -f "$VQASCORE" ]; then
        sed -i "s|/PhyGenBench/PhyGenBench/|$PHYGENBENCH_DIR/PhyGenBench/|g" "$VQASCORE"
        sed -i "s|/PhyGenBench/PhyGenEval/|$PHYGENBENCH_DIR/PhyGenEval/|g" "$VQASCORE"
        sed -i "s|/PhyGenBench/PhyVideos/|$PHYGENBENCH_DIR/PhyVideos/|g" "$VQASCORE"
        echo "✓ Fixed $VQASCORE"
    fi
    
    # Fix multiimage_clip.py
    MULTIIMAGE="$PHYGENBENCH_DIR/PhyGenEval/multi/multiimage_clip.py"
    if [ -f "$MULTIIMAGE" ]; then
        sed -i "s|/PhyGenBench/PhyGenBench/|$PHYGENBENCH_DIR/PhyGenBench/|g" "$MULTIIMAGE"
        sed -i "s|/PhyGenBench/PhyGenEval/|$PHYGENBENCH_DIR/PhyGenEval/|g" "$MULTIIMAGE"
        sed -i "s|/PhyGenBench/PhyVideos/|$PHYGENBENCH_DIR/PhyVideos/|g" "$MULTIIMAGE"
        echo "✓ Fixed $MULTIIMAGE"
    fi
    
    # Fix overall.py
    OVERALL="$PHYGENBENCH_DIR/PhyGenEval/overall.py"
    if [ -f "$OVERALL" ]; then
        sed -i "s|/PhyGenBench/PhyGenBench/|$PHYGENBENCH_DIR/PhyGenBench/|g" "$OVERALL"
        sed -i "s|/PhyGenBench/PhyGenEval/|$PHYGENBENCH_DIR/PhyGenEval/|g" "$OVERALL"
        echo "✓ Fixed $OVERALL"
    fi
else
    echo "⚠ PhyGenBench directory not found at $PHYGENBENCH_DIR"
fi

# 4. Verify fixes
echo ""
echo "[4/4] Verifying fixes..."

echo "Testing pytorchvideo import..."
python -c "from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample; print('✓ pytorchvideo imports work')" 2>/dev/null || echo "⚠ pytorchvideo still has issues"

echo "Testing moviepy import..."
python -c "from moviepy.editor import VideoFileClip; print('✓ moviepy imports work')" 2>/dev/null || echo "⚠ moviepy still has issues"

echo "Testing t2v_metrics import..."
python -c "import t2v_metrics; print('✓ t2v_metrics imports work')" 2>/dev/null || echo "⚠ t2v_metrics still has issues"

echo ""
echo "=========================================="
echo "Fix complete! Now re-run the evaluation:"
echo "  sbatch scripts/eval_phygenbench_slurm.sh baseline"
echo "=========================================="