#!/bin/bash
#SBATCH --job-name=extract_feat
#SBATCH --output=slurm/extraction/pool_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# ============================================================
# Pool DiT Features for Fast Training
#
# Converts ~67MB files to ~100KB files (700x compression)
# Total time: ~10-20 minutes for ~1200 videos
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================"

# Environment
source ~/.bashrc
conda activate physics

cd /users/$USER/repos/VLMPhysics

# Paths
INPUT_DIR="/users/$USER/scratch/physics/videophy_features"
OUTPUT_DIR="/users/$USER/scratch/physics/videophy_features_pooled"

mkdir -p "$OUTPUT_DIR"

echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Show input size
echo "Input directory size:"
du -sh "$INPUT_DIR"
echo ""

# Run pooling
python src/data/pool_features.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"

# Show output size
echo ""
echo "Output directory size:"
du -sh "$OUTPUT_DIR"

echo ""
echo "Sample files:"
ls -la "$OUTPUT_DIR"/*/t200/layer_15.pt 2>/dev/null | head -5