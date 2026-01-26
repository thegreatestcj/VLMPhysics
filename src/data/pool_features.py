#!/usr/bin/env python3
"""
Pool Features Script - Correctly pools spatial dimensions while preserving temporal.

Input:  [17550, 1920] where 17550 = 13 × 30 × 45 (T × H × W)
Output: [13, 1920] where we pool over H × W, keeping T
"""

import torch
from pathlib import Path
from tqdm import tqdm
import shutil
import json

# Constants from CogVideoX
NUM_FRAMES = 13
SPATIAL_H = 30
SPATIAL_W = 45
EXPECTED_SEQ = NUM_FRAMES * SPATIAL_H * SPATIAL_W  # 17550


def pool_features(input_dir: Path, output_dir: Path, layers: list = [15]):
    """Pool spatial dimensions of extracted features."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Copy labels.json
    if (input_dir / "labels.json").exists():
        shutil.copy(input_dir / "labels.json", output_dir / "labels.json")
        print(f"Copied labels.json")

    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} video directories")

    success = 0
    errors = []

    for video_dir in tqdm(video_dirs, desc="Pooling"):
        video_id = video_dir.name
        out_video_dir = output_dir / video_id

        for t_dir in video_dir.iterdir():
            if not t_dir.is_dir() or not t_dir.name.startswith("t"):
                continue

            out_t_dir = out_video_dir / t_dir.name
            out_t_dir.mkdir(parents=True, exist_ok=True)

            for layer in layers:
                in_file = t_dir / f"layer_{layer}.pt"
                out_file = out_t_dir / f"layer_{layer}.pt"

                if not in_file.exists():
                    continue

                try:
                    feat = torch.load(in_file, map_location="cpu", weights_only=True)

                    # Handle batch dimension if present
                    if feat.dim() == 3 and feat.shape[0] == 1:
                        feat = feat.squeeze(0)  # [1, 17550, 1920] -> [17550, 1920]

                    if feat.dim() != 2:
                        errors.append(
                            f"{video_id}/{t_dir.name}: unexpected dim {feat.dim()}"
                        )
                        continue

                    seq_len, hidden_dim = feat.shape

                    if seq_len != EXPECTED_SEQ:
                        errors.append(
                            f"{video_id}/{t_dir.name}: seq_len={seq_len}, expected {EXPECTED_SEQ}"
                        )
                        continue

                    # Reshape: [17550, 1920] -> [13, 30, 45, 1920]
                    feat = feat.view(NUM_FRAMES, SPATIAL_H, SPATIAL_W, hidden_dim)

                    # Pool spatial only: [13, 30, 45, 1920] -> [13, 1920]
                    pooled = feat.mean(dim=(1, 2))  # Mean over H and W

                    # Verify shape
                    assert pooled.shape == (NUM_FRAMES, hidden_dim), (
                        f"Wrong shape: {pooled.shape}"
                    )

                    # Save
                    torch.save(pooled, out_file)
                    success += 1

                except Exception as e:
                    errors.append(f"{video_id}/{t_dir.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Pooling complete!")
    print(f"  Success: {success}")
    print(f"  Errors: {len(errors)}")

    if errors[:10]:
        print(f"\nSample errors:")
        for e in errors[:10]:
            print(f"  {e}")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "source": str(input_dir),
                "pooled_shape": [NUM_FRAMES, 1920],
                "layers": layers,
                "success_count": success,
                "error_count": len(errors),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    args = parser.parse_args()

    pool_features(
        Path(args.input_dir),
        Path(args.output_dir),
        args.layers,
    )
