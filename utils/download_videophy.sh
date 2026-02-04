#!/bin/bash
#SBATCH --job-name=dl_videophy
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/utils/download_videophy_%j.out
#SBATCH --partition=batch

# Proper conda activation for SLURM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate physics

cd /oscar/home/ctang33/repos/VLMPhysics
mkdir -p slurm/utils

DATA_DIR=~/scratch/physics/videophy_data

echo "========================================"
echo "VideoPhy v1+v2 Full Download"
echo "Date: $(date)"
echo "========================================"

# Clean old data
echo "[0] Cleaning $DATA_DIR ..."
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR/videos"

# Download all
echo "[1] Downloading metadata + videos (v1 + v2)..."
python utils/download_videophy.py --data-dir "$DATA_DIR" --workers 8

# Verify
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo "mp4 files:  $(find $DATA_DIR/videos -name '*.mp4' 2>/dev/null | wc -l)"
python3 -c "
import json
from collections import Counter
m = json.load(open('$DATA_DIR/metadata.json'))
splits = Counter(x['dataset_split'] for x in m)
pos = sum(1 for x in m if x['physics']==1)
neg = len(m) - pos
print(f'metadata:   {len(m)} entries')
for k, v in sorted(splits.items()):
    print(f'  {k}: {v}')
print(f'physics:    {pos} pos / {neg} neg (pw={neg/max(pos,1):.3f})')
"
echo "disk:       $(du -sh $DATA_DIR | cut -f1)"
echo "Done at $(date)"