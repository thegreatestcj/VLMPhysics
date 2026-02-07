#!/usr/bin/env python3
"""
Patch extract_features.py: add --pool flag for inline spatial pooling.

Shape flow (CogVideoX-2B, layer 15):
    DiT hook output:      [1, 17550, 1920]   (1=batch, 17550=13*30*45, 1920=hidden)
    pool_spatial_inline:  squeeze -> [17550, 1920]
                          view   -> [13, 1350, 1920]   (13=T, 1350=30*45=H*W)
                          mean(1)-> [13, 1920]
    .half():              [13, 1920] fp16
    saved:                ~100 KB per file

Without --pool (original):
    saved:                [1, 17550, 1920] fp32 -> ~67 MB per file

Usage:
    cd ~/repos/VLMPhysics
    python utils/patch_extract_pool.py
    # verify:
    grep -n "pool_spatial_inline\|--pool" src/models/extract_features.py
"""

import sys
from pathlib import Path


def patch(filepath: str) -> bool:
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: {filepath} not found")
        return False

    text = path.read_text()

    if "pool_spatial_inline" in text:
        print("Already patched — skipping.")
        return True

    # ==================================================================
    # 1) Add pool_spatial_inline() right before extract_single_video()
    # ==================================================================
    ANCHOR1 = "def extract_single_video("
    if ANCHOR1 not in text:
        print(f"ERROR: cannot find '{ANCHOR1}'")
        return False

    POOL_FUNC = '''\
def pool_spatial_inline(features: torch.Tensor, num_frames: int = 13) -> torch.Tensor:
    """
    Pool spatial dims inline during extraction.

    CogVideoX-2B layer 15 shapes:
        Input:  [1, 17550, 1920]   (batch=1, patches=13*30*45, hidden=1920)
        Output: [13, 1920]         (T=13 latent frames, D=1920)

    Args:
        features: Raw DiT hidden states [1, T*H*W, D]
        num_frames: Temporal frames in latent space (13 for CogVideoX)

    Returns:
        [T, D] pooled tensor
    """
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)  # [1, 17550, 1920] -> [17550, 1920]

    if features.dim() == 2:
        num_patches, hidden_dim = features.shape
        spatial = num_patches // num_frames
        # [T*H*W, D] -> [T, H*W, D] -> mean(H*W) -> [T, D]
        return features.view(num_frames, spatial, hidden_dim).mean(dim=1)
    elif features.dim() == 3:
        return features.mean(dim=1)  # [T, H*W, D] -> [T, D]
    else:
        raise ValueError(f"pool_spatial_inline: unexpected shape {features.shape}")


'''

    text = text.replace(ANCHOR1, POOL_FUNC + ANCHOR1, 1)
    print("  [1/5] Added pool_spatial_inline()")

    # ==================================================================
    # 2) Add pool param to extract_single_video() signature
    # ==================================================================
    OLD_SIG = "    text_embeds=None,\n) -> tuple:"
    NEW_SIG = "    text_embeds=None,\n    pool: bool = False,\n) -> tuple:"
    if OLD_SIG in text:
        text = text.replace(OLD_SIG, NEW_SIG, 1)
        print("  [2/5] Added pool param to extract_single_video()")
    else:
        print("  ERROR: cannot patch extract_single_video signature")
        return False

    # ==================================================================
    # 3) Replace save block with pooling logic + debug print
    # ==================================================================
    OLD_SAVE = """\
        # Save features for each layer
        layer_shapes = {}
        for layer_idx, feat in features.items():
            feat_path = t_dir / f"layer_{layer_idx}.pt"
            torch.save(feat.cpu(), feat_path)
            layer_shapes[layer_idx] = list(feat.shape)

        info["timesteps"][t] = {"layer_shapes": layer_shapes}"""

    NEW_SAVE = """\
        # Save features for each layer (with optional inline pooling)
        layer_shapes = {}
        for layer_idx, feat in features.items():
            feat_path = t_dir / f"layer_{layer_idx}.pt"
            raw_shape = list(feat.shape)
            if pool:
                feat = pool_spatial_inline(feat.cpu(), num_frames=13).half()
            else:
                feat = feat.cpu()
            torch.save(feat, feat_path)
            layer_shapes[layer_idx] = list(feat.shape)

        # Debug: log shapes for the very first timestep of this video
        if not info["timesteps"]:
            for layer_idx in layer_shapes:
                raw = features[layer_idx].shape
                saved = layer_shapes[layer_idx]
                size_kb = (t_dir / f"layer_{layer_idx}.pt").stat().st_size / 1024
                logger.debug(
                    f"  [{video_id}] layer {layer_idx}: "
                    f"{list(raw)} -> {saved} "
                    f"({'pooled fp16' if pool else 'raw'}, {size_kb:.0f} KB)"
                )

        info["timesteps"][t] = {"layer_shapes": layer_shapes}"""

    if OLD_SAVE in text:
        text = text.replace(OLD_SAVE, NEW_SAVE, 1)
        print("  [3/5] Replaced save block with pooling + debug logging")
    else:
        print("  ERROR: cannot find save block to patch")
        print("  Check indentation around '# Save features for each layer'")
        return False

    # ==================================================================
    # 4) Add pool param to extract_dataset() and pass through
    # ==================================================================
    # 4a: signature
    OLD_DS_SIG = "    shard: int = 0,\n    num_shards: int = 1,\n):"
    NEW_DS_SIG = (
        "    shard: int = 0,\n    num_shards: int = 1,\n    pool: bool = False,\n):"
    )
    if OLD_DS_SIG in text:
        text = text.replace(OLD_DS_SIG, NEW_DS_SIG, 1)
        print("  [4a/5] Added pool param to extract_dataset()")
    else:
        print("  WARNING: cannot find extract_dataset signature")

    # 4b: pass pool= to extract_single_video call
    OLD_CALL = "                text_embeds=text_embeds,\n            )"
    NEW_CALL = "                text_embeds=text_embeds,\n                pool=pool,\n            )"
    if OLD_CALL in text:
        text = text.replace(OLD_CALL, NEW_CALL, 1)
        print("  [4b/5] Passing pool= to extract_single_video()")
    else:
        print("  WARNING: cannot find extract_single_video() call site")

    # 4c: add log line about pool mode
    OLD_LOG = '    logger.info(f"  Output: {output_dir}")'
    NEW_LOG = (
        '    logger.info(f"  Output: {output_dir}")\n'
        '    logger.info(f"  Pool inline: {pool}  '
        '(will save [13, 1920] fp16 ~100KB/file)")'
    )
    if OLD_LOG in text:
        text = text.replace(OLD_LOG, NEW_LOG, 1)

    # ==================================================================
    # 5) Add --pool CLI argument and pass to extract_dataset()
    # ==================================================================
    # 5a: CLI argument (insert before args = parser.parse_args())
    OLD_PARSE = "    args = parser.parse_args()\n\n    extract_dataset("
    NEW_PARSE = (
        "    parser.add_argument(\n"
        '        "--pool",\n'
        '        action="store_true",\n'
        "        default=False,\n"
        '        help="Pool features inline during extraction. "\n'
        '        "Saves [13, 1920] fp16 (~100KB) instead of "\n'
        '        "[1, 17550, 1920] fp32 (~67MB). "\n'
        '        "Output is directly usable with --is_pooled in training.",\n'
        "    )\n\n"
        "    args = parser.parse_args()\n\n    extract_dataset("
    )
    if OLD_PARSE in text:
        text = text.replace(OLD_PARSE, NEW_PARSE, 1)
        print("  [5a/5] Added --pool CLI argument")
    else:
        print("  WARNING: cannot find parse_args block")

    # 5b: pass pool=args.pool
    OLD_MAIN_CALL = (
        "        shard=args.shard,\n        num_shards=args.num_shards,\n    )"
    )
    NEW_MAIN_CALL = "        shard=args.shard,\n        num_shards=args.num_shards,\n        pool=args.pool,\n    )"
    if OLD_MAIN_CALL in text:
        text = text.replace(OLD_MAIN_CALL, NEW_MAIN_CALL, 1)
        print("  [5b/5] Passing pool=args.pool to extract_dataset()")
    else:
        print("  WARNING: cannot find extract_dataset() call in main()")

    path.write_text(text)
    print(f"\nDone! Patched {filepath}")
    print(f"\nVerify:\n  grep -n 'pool_spatial_inline\\|--pool\\|pool=' {filepath}")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "src/models/extract_features.py"
    ok = patch(target)
    sys.exit(0 if ok else 1)
