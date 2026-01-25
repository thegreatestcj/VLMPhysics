#!/usr/bin/env python
"""
Quick test script to verify Physion dataset loading.
Run: python scripts/test_physion_data.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.physion_dataset import PhysionDataset, get_dataloaders


def main():
    data_root = "data/Physion"

    print("=" * 60)
    print("Physion Dataset Verification")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists(data_root):
        print(f"ERROR: {data_root} not found!")
        print("Run: unzip Physion.zip -d data/")
        return

    labels_path = os.path.join(data_root, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"ERROR: {labels_path} not found!")
        return

    # Count labels
    print("\n[1] Checking labels.csv...")
    with open(labels_path, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        total = len(lines)
        true_count = sum(1 for line in lines if "True" in line)
        false_count = total - true_count

    print(f"    Total entries: {total}")
    print(f"    True (contact): {true_count} ({true_count / total * 100:.1f}%)")
    print(f"    False (no contact): {false_count} ({false_count / total * 100:.1f}%)")

    # Count videos per scenario
    print("\n[2] Checking video files...")
    scenarios = [
        "Collide",
        "Contain",
        "Dominoes",
        "Drape",
        "Drop",
        "Link",
        "Roll",
        "Support",
    ]
    total_videos = 0

    for scenario in scenarios:
        video_dir = os.path.join(data_root, scenario, "mp4s-redyellow")
        if os.path.exists(video_dir):
            count = len([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
            total_videos += count
            print(f"    {scenario}: {count} videos")
        else:
            print(f"    {scenario}: NOT FOUND")

    print(f"    Total videos: {total_videos}")

    # Test dataset loading
    print("\n[3] Testing dataset loading...")
    try:
        # Use all data (no train/val split - evaluate on PhyGenBench instead)
        dataset = PhysionDataset(
            data_root=data_root,
            split="all",  # Use all 1200 samples
            num_frames=13,  # Use fewer frames for quick test
        )
        print(f"    Dataset size: {len(dataset)}")

        # Load one sample
        print("\n[4] Loading sample video...")
        sample = dataset[0]
        print(f"    Video shape: {sample['video'].shape}")  # [T, C, H, W]
        print(
            f"    Label: {sample['label']} ({'contact' if sample['label'] else 'no contact'})"
        )
        print(f"    Scenario: {sample['scenario']}")
        print(f"    Video name: {sample['video_name'][:50]}...")

        # Check video stats
        video = sample["video"]
        print(f"    Video value range: [{video.min():.3f}, {video.max():.3f}]")

        print("\n" + "=" * 60)
        print("SUCCESS! Dataset is ready for training.")
        print("=" * 60)

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
