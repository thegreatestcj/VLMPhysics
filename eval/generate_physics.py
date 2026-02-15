#!/usr/bin/env python3
"""
PhyGenBench Video Generation with Physics Head Trajectory Pruning

Generates videos for PhyGenBench prompts using CogVideoX-2B with
trajectory pruning guided by the trained physics discriminator head.

Supports resuming - loads existing log and appends without overwriting.

Output Structure:
    outputs/phygenbench/{run_name}/
    ├── config.json
    ├── generation_log.json
    ├── output_video_1.mp4
    ├── output_video_2.mp4
    └── ...

Usage:
    # Generate all prompts (single GPU)
    python eval/generate_physics.py --start 0 --end 160

    # Generate specific range
    python eval/generate_physics.py --start 0 --end 80

    # Skip existing videos (RECOMMENDED for resume)
    python eval/generate_physics.py --start 0 --end 160 --skip-existing
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PhysicsGenConfig:
    """Configuration for physics-guided generation."""

    # Model
    model_id: str = "THUDM/CogVideoX-2b"

    # Physics head
    physics_head_path: str = "results/training/final_head/mean_l15.pt"
    head_type: str = "mean"
    extract_layer: int = 15
    force_head_type: bool = False

    # Mode: "prune" = progressive trajectory pruning (ours)
    #        "best_of_n" = generate all N, score at end, pick best (baseline)
    #        "random" = prune with random scores (ablation)
    mode: str = "prune"

    # Trajectory pruning (used in "prune" and "random" modes)
    # 4 trajectories: 4 -> 2 -> 1 (faster, ~2.5x baseline time)
    # 8 trajectories: 8 -> 4 -> 2 -> 1 (slower, ~5x baseline time)
    num_trajectories: int = 4
    checkpoints: List[int] = field(default_factory=lambda: [600, 400])  # 2 prune points
    keep_ratio: float = 0.5

    # Best-of-N: timestep at which to score (should match training data)
    # t=200 is the cleanest timestep the physics head was trained on
    scoring_timestep: int = 200

    # Generation
    num_inference_steps: int = 50
    num_frames: int = 49
    guidance_scale: float = 6.0
    height: int = 480
    width: int = 720

    # Output
    output_dir: str = "outputs/phygenbench/physics"
    seed: int = 42


# =============================================================================
# Constants
# =============================================================================

# CogVideoX-2B architecture constants
TEXT_SEQ_LEN = 226  # T5 text token sequence length
HIDDEN_DIM = 1920  # DiT hidden dimension


# =============================================================================
# Physics Head Wrapper
# =============================================================================


class PhysicsHeadEvaluator:
    """Evaluates trajectory physics scores using trained head."""

    def __init__(
        self,
        head_path: str,
        head_type: str,
        extract_layer: int,
        device: str,
        force_head_type: bool = False,
    ):
        self.device = device
        self.extract_layer = extract_layer
        self.head_type = head_type
        self.captured_features = None
        self._hook_handle = None

        # Load physics head
        from src.models.physics_head import create_physics_head

        # Read head_type from checkpoint if available, to prevent mismatch
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        ckpt_head_type = ckpt.get("head_type", None)
        if ckpt_head_type and ckpt_head_type != head_type:
            if force_head_type:
                logger.warning(
                    f"FORCING head_type='{head_type}' (checkpoint has '{ckpt_head_type}')"
                )
            else:
                logger.warning(
                    f"head_type mismatch: arg='{head_type}', checkpoint='{ckpt_head_type}'. "
                    f"Using checkpoint's '{ckpt_head_type}'."
                )
                head_type = ckpt_head_type
            self.head_type = head_type

        # Create model
        self.head = create_physics_head(head_type, hidden_dim=HIDDEN_DIM)

        # Wrap with MultiTaskWrapper if checkpoint was trained with SA
        has_sa = any("sa_classifier" in k for k in ckpt["model_state_dict"].keys())
        if has_sa:
            from src.models.physics_head import MultiTaskWrapper

            self.head = MultiTaskWrapper(self.head)
            logger.info("Checkpoint has SA head, wrapped with MultiTaskWrapper")
        self.has_sa = has_sa

        # Load weights
        self.head.load_state_dict(ckpt["model_state_dict"])
        self.head = self.head.to(device).eval()

        num_params = sum(p.numel() for p in self.head.parameters())
        logger.info(
            f"Physics head loaded: {head_path} "
            f"(type={head_type}, layer={extract_layer}, params={num_params:,}, sa={has_sa})"
        )

    def setup_hook(self, transformer) -> None:
        """Register feature extraction hook on the target DiT layer."""

        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured_features = out.detach()

        block = transformer.transformer_blocks[self.extract_layer]
        self._hook_handle = block.register_forward_hook(hook_fn)
        logger.info(f"Hook registered on transformer_blocks[{self.extract_layer}]")

    def remove_hook(self) -> None:
        """Remove hook."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def score(self, timestep: int, num_frames: int = 49) -> float:
        """
        Score current captured features.

        The hook captures raw DiT intermediate features [B, text+video, D].
        We need to:
          1. Strip text tokens (first 226 tokens)
          2. Reshape video tokens and pool spatial dims -> [B, T, D]
          3. Run through physics head
        """
        if self.captured_features is None:
            logger.warning("No captured features, returning neutral score")
            return 0.5

        feat = self.captured_features
        B, seq_len, D = feat.shape

        # Step 1: Strip text tokens if present
        # CogVideoX intermediate layers output [B, 226 + T*H*W, D]
        num_latent_frames = (num_frames - 1) // 4 + 1  # 13 for 49 frames
        video_seq_len = num_latent_frames * 30 * 45  # 13 * 30 * 45 = 17550

        if seq_len == TEXT_SEQ_LEN + video_seq_len:
            # Has text tokens, strip them
            feat = feat[:, TEXT_SEQ_LEN:, :]
        elif seq_len == video_seq_len:
            # Video only, no stripping needed
            pass
        else:
            logger.warning(
                f"Unexpected seq_len={seq_len} "
                f"(expected {video_seq_len} or {TEXT_SEQ_LEN + video_seq_len}). "
                f"Attempting to strip first {TEXT_SEQ_LEN} tokens."
            )
            feat = feat[:, TEXT_SEQ_LEN:, :]

        # Step 2: Reshape and pool spatial dims
        # [B, T*H*W, D] -> [B, T, H*W, D] -> mean over H*W -> [B, T, D]
        B_new, remaining, D = feat.shape
        spatial_size = remaining // num_latent_frames
        feat = feat.view(B_new, num_latent_frames, spatial_size, D).mean(dim=2)

        # Step 3: Run through physics head
        # mean head: forward(features) — no timestep
        # other heads: forward(features, timestep)
        feat = feat.float()

        if self.has_sa:
            # MultiTaskWrapper.forward(features, timestep=None)
            # It internally calls base_head with appropriate args
            if self.head_type == "mean":
                output = self.head(feat, None)
            else:
                t = torch.tensor([timestep], device=self.device, dtype=torch.float)
                output = self.head(feat, t)
            # MultiTaskWrapper returns (physics_logits, sa_logits)
            logits = output[0]
        else:
            # Raw head without wrapper
            if self.head_type == "mean":
                logits = self.head(feat)
            else:
                t = torch.tensor([timestep], device=self.device, dtype=torch.float)
                logits = self.head(feat, t)

        score = torch.sigmoid(logits).item()

        # Clear captured features
        self.captured_features = None
        return score


# =============================================================================
# Trajectory Pruning Generator
# =============================================================================


class TrajectoryPruningGenerator:
    """
    Generates videos using trajectory pruning with physics head guidance.

    Strategy:
    - Maintain multiple trajectories (e.g., 4)
    - At checkpoints, evaluate all with physics head
    - Keep top fraction based on physics scores
    - Final: 4 -> 2 -> 1
    """

    def __init__(self, config: PhysicsGenConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.pipe = None
        self.evaluator = None

    def load(self) -> None:
        """Load pipeline and physics head."""
        from diffusers import CogVideoXPipeline

        logger.info(f"Loading {self.config.model_id}...")

        self.pipe = CogVideoXPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        # Load physics head
        if Path(self.config.physics_head_path).exists():
            self.evaluator = PhysicsHeadEvaluator(
                head_path=self.config.physics_head_path,
                head_type=self.config.head_type,
                extract_layer=self.config.extract_layer,
                device=self.device,
                force_head_type=self.config.force_head_type,
            )
            self.evaluator.setup_hook(self.pipe.transformer)
        else:
            logger.warning(f"Physics head not found: {self.config.physics_head_path}")
            logger.warning("Running without physics guidance (random selection)")

        logger.info("Pipeline ready!")

    def _get_checkpoint_steps(self, timesteps: torch.Tensor) -> List[int]:
        """Map timestep values to step indices."""
        ts_list = timesteps.tolist()
        steps = []
        for cp in self.config.checkpoints:
            idx = min(range(len(ts_list)), key=lambda i: abs(ts_list[i] - cp))
            steps.append(idx)
        return sorted(set(steps))

    def _get_scoring_step(self, timesteps: torch.Tensor) -> int:
        """Find step index closest to scoring_timestep (for best_of_n mode)."""
        ts_list = timesteps.tolist()
        return min(
            range(len(ts_list)),
            key=lambda i: abs(ts_list[i] - self.config.scoring_timestep),
        )

    def _score_trajectory(self, tr, t, prompt_embeds) -> float:
        """Score a single trajectory using physics head (extra forward pass)."""
        if not self.evaluator:
            return torch.rand(1).item()

        latent_input = self.pipe.scheduler.scale_model_input(tr["latents"], t)
        t_input = t.expand(latent_input.shape[0])
        _ = self.pipe.transformer(
            hidden_states=latent_input,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds[:1],  # No CFG for scoring
            return_dict=False,
        )
        t_val = int(t.item()) if hasattr(t, "item") else int(t)
        return self.evaluator.score(t_val, self.config.num_frames)

    @torch.no_grad()
    def generate(self, prompt: str, seed: int) -> Tuple[torch.Tensor, Dict]:
        """
        Generate video using the configured mode.

        Modes:
            prune:     Progressive trajectory pruning at checkpoints (ours)
            best_of_n: Run all N trajectories to completion, score at t=200, pick best
            random:    Same as prune but with random scores (ablation baseline)

        Returns:
            video: Generated video (list of numpy arrays)
            info: Dictionary with scoring/pruning history and timing
        """
        mode = self.config.mode
        timing = {"encode": 0, "sampling": 0, "decode": 0}

        # Encode prompt once
        t0 = time.time()
        prompt_embeds, neg_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=self.config.guidance_scale > 1.0,
            device=self.device,
        )
        timing["encode"] = time.time() - t0

        # For CFG, prepare combined prompt embeds: [neg, pos]
        if self.config.guidance_scale > 1.0 and neg_embeds is not None:
            cfg_prompt_embeds = torch.cat(
                [neg_embeds, prompt_embeds], dim=0
            )  # [2, seq, D]
        else:
            cfg_prompt_embeds = prompt_embeds

        # Initialize multiple trajectories with different seeds
        num_traj = self.config.num_trajectories
        trajectories = []

        # Calculate latent shape
        num_latent_frames = (self.config.num_frames - 1) // 4 + 1  # 13 for 49 frames
        latent_channels = self.pipe.transformer.config.in_channels  # 16
        latent_height = self.config.height // 8  # 60
        latent_width = self.config.width // 8  # 90
        latent_shape = (
            1,
            num_latent_frames,
            latent_channels,
            latent_height,
            latent_width,
        )

        for i in range(num_traj):
            generator = torch.Generator(device=self.device).manual_seed(seed + i)

            latents = (
                torch.randn(
                    latent_shape,
                    generator=generator,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                * self.pipe.scheduler.init_noise_sigma
            )

            trajectories.append(
                {
                    "idx": i,
                    "latents": latents,
                    "active": True,
                    "score": 0.5,
                    "scores_history": {},  # timestep -> score
                    "seed": seed + i,
                }
            )

        # Setup scheduler
        self.pipe.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps

        # Determine which steps need scoring
        if mode == "prune" or mode == "random":
            checkpoint_steps = self._get_checkpoint_steps(timesteps)
        elif mode == "best_of_n":
            # Score at the designated timestep, no pruning
            scoring_step = self._get_scoring_step(timesteps)
            checkpoint_steps = [scoring_step]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        timesteps = timesteps.to(self.device)

        logger.info(
            f"Mode: {mode}, Trajectories: {num_traj}, "
            f"Checkpoint steps: {checkpoint_steps}"
        )
        logger.info(
            f"Latent shape: {latent_shape}, timesteps device: {timesteps.device}"
        )

        # Sampling loop
        scoring_history = []
        sampling_start = time.time()

        for step, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            active_trajs = [tr for tr in trajectories if tr["active"]]
            num_active = len(active_trajs)

            if num_active == 0:
                break

            # Denoise each active trajectory
            for tr in active_trajs:
                # Prepare latents for CFG
                if self.config.guidance_scale > 1.0:
                    latent_model_input = torch.cat(
                        [tr["latents"], tr["latents"]], dim=0
                    )
                    prompt_input = cfg_prompt_embeds  # [2, seq, D]: [neg, pos]
                else:
                    latent_model_input = tr["latents"]
                    prompt_input = prompt_embeds

                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Timestep needs to be 1D array with batch size
                t_input = t.expand(latent_model_input.shape[0])

                # DiT forward pass
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_input,
                    encoder_hidden_states=prompt_input,
                    return_dict=False,
                )[0]

                # CFG
                if self.config.guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Scheduler step
                tr["latents"] = self.pipe.scheduler.step(
                    noise_pred, t, tr["latents"], return_dict=False
                )[0]

            # Checkpoint: score (and optionally prune)
            if step in checkpoint_steps and num_active > 1:
                t_val = int(t.item()) if hasattr(t, "item") else int(t)

                # Score each active trajectory
                for tr in active_trajs:
                    if mode == "random":
                        tr["score"] = torch.rand(1).item()
                    else:
                        tr["score"] = self._score_trajectory(tr, t, prompt_embeds)
                    tr["scores_history"][t_val] = tr["score"]

                # Sort by score
                active_trajs.sort(key=lambda x: x["score"], reverse=True)
                scores_summary = [
                    (tr["idx"], f"{tr['score']:.3f}") for tr in active_trajs
                ]

                if mode == "best_of_n":
                    # Score only, NO pruning — all trajectories keep running
                    logger.info(
                        f"  Step {step} (t={t_val}) [best_of_n]: Scores {scores_summary}"
                    )
                    scoring_history.append(
                        {
                            "step": step,
                            "timestep": t_val,
                            "scores": scores_summary,
                            "action": "score_only",
                        }
                    )
                else:
                    # Prune mode (or random): keep top fraction
                    num_keep = max(1, int(num_active * self.config.keep_ratio))

                    kept = []
                    pruned = []
                    for i, tr in enumerate(active_trajs):
                        if i < num_keep:
                            kept.append((tr["idx"], f"{tr['score']:.3f}"))
                        else:
                            tr["active"] = False
                            tr["latents"] = None  # Free memory
                            pruned.append((tr["idx"], f"{tr['score']:.3f}"))

                    logger.info(
                        f"  Step {step} (t={t_val}): Keep {kept}, Prune {pruned}"
                    )
                    scoring_history.append(
                        {
                            "step": step,
                            "timestep": t_val,
                            "kept": kept,
                            "pruned": pruned,
                            "action": "prune",
                        }
                    )

                    # Clear memory from pruned trajectories
                    gc.collect()
                    torch.cuda.empty_cache()

        # Select best trajectory
        active_trajs = [tr for tr in trajectories if tr["active"]]
        best = max(active_trajs, key=lambda x: x["score"])

        timing["sampling"] = time.time() - sampling_start
        logger.info(
            f"Best trajectory: idx={best['idx']}, score={best['score']:.3f}, "
            f"sampling_time={timing['sampling']:.1f}s"
        )

        # Decode only the best trajectory
        decode_start = time.time()
        video = self._decode(best["latents"])
        timing["decode"] = time.time() - decode_start

        info = {
            "mode": mode,
            "best_trajectory": best["idx"],
            "best_score": best["score"],
            "all_scores": {tr["idx"]: tr["scores_history"] for tr in trajectories},
            "scoring_history": scoring_history,
            "sampling_time": timing["sampling"],
            "decode_time": timing["decode"],
            "encode_time": timing["encode"],
        }

        return video, info

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video."""
        # Move transformer to CPU to free memory for VAE
        self.pipe.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare latents: [B, T, C, H, W] -> [B, C, T, H, W]
        latents = latents.float()
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        latents = latents / self.pipe.vae.config.scaling_factor

        # VAE decode
        self.pipe.vae = self.pipe.vae.float().to(self.device)
        with torch.no_grad():
            video = self.pipe.vae.decode(latents, return_dict=False)[0]

        # Post-process to numpy
        video = self.pipe.video_processor.postprocess_video(video, output_type="np")[0]

        # Move transformer back for next generation
        self.pipe.transformer.to(self.device)

        return video  # List of numpy arrays

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.evaluator:
            self.evaluator.remove_hook()


# =============================================================================
# Main Generation Loop
# =============================================================================


def load_prompts(prompts_file: str) -> List[Dict]:
    """Load PhyGenBench prompts."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def load_existing_log(log_path: Path) -> Tuple[List[Dict], set]:
    """
    Load existing generation log and extract completed indices.

    Returns:
        results: List of existing results
        completed_indices: Set of indices that have been successfully generated
    """
    if not log_path.exists():
        return [], set()

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Extract indices of successful generations
        completed_indices = {
            r["index"]
            for r in results
            if r.get("status") == "success" or r.get("status", "").startswith("success")
        }

        logger.info(
            f"Loaded existing log with {len(results)} entries, "
            f"{len(completed_indices)} successful"
        )
        return results, completed_indices

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not parse existing log: {e}")
        backup_path = log_path.with_suffix(f".backup_{int(time.time())}.json")
        log_path.rename(backup_path)
        logger.warning(f"Backed up corrupted log to {backup_path}")
        return [], set()


def save_log_sorted(log_path: Path, results: List[Dict]) -> None:
    """Save results to log, sorted by index."""
    sorted_results = sorted(results, key=lambda x: x.get("index", 0))
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PhyGenBench videos with physics head pruning",
    )

    # Data
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/phygenbench/prompts.json",
        help="Path to prompts file",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phygenbench/physics",
        help="Output directory",
    )

    # Physics head
    parser.add_argument(
        "--physics-head",
        type=str,
        default="results/training/final_head/mean_l15.pt",
        help="Path to physics head checkpoint",
    )
    parser.add_argument("--head-type", type=str, default="mean")
    parser.add_argument("--extract-layer", type=int, default=15)
    parser.add_argument(
        "--force-head-type",
        action="store_true",
        default=False,
        help="Force using --head-type instead of checkpoint's head_type",
    )

    # Trajectory pruning
    parser.add_argument(
        "--mode",
        type=str,
        default="prune",
        choices=["prune", "best_of_n", "random"],
        help="Generation mode: prune (ours), best_of_n (baseline), random (ablation)",
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=4, help="Initial trajectories (4 or 8)"
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[600, 400],
        help="Pruning checkpoint timesteps",
    )

    # Generation
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)

    # Range
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=160)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Select range
    indices = list(range(args.start, min(args.end, len(prompts))))
    logger.info(
        f"Processing prompts {args.start} to {args.end - 1} ({len(indices)} total)"
    )

    # Create config
    config = PhysicsGenConfig(
        physics_head_path=args.physics_head,
        head_type=args.head_type,
        extract_layer=args.extract_layer,
        force_head_type=args.force_head_type,
        mode=args.mode,
        num_trajectories=args.num_trajectories,
        checkpoints=args.checkpoints,
        num_inference_steps=args.steps,
        num_frames=args.frames,
        guidance_scale=args.guidance,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load existing log (if any) to support resume
    log_path = output_dir / "generation_log.json"
    results, completed_indices = load_existing_log(log_path)

    # Create index -> result mapping for updating existing entries
    results_by_index = {r["index"]: r for r in results}

    # Setup generator
    generator = TrajectoryPruningGenerator(config)
    generator.load()

    # Count stats
    skipped_count = 0
    generated_count = 0

    for idx in tqdm(indices, desc="Generating"):
        prompt_data = prompts[idx]
        prompt_text = prompt_data.get("caption", prompt_data.get("prompt", ""))
        category = prompt_data.get(
            "sub_category", prompt_data.get("main_category", "Unknown")
        )

        # Output path (1-indexed like baseline)
        video_path = output_dir / f"output_video_{idx + 1}.mp4"

        # Skip if already completed
        already_in_log = idx in completed_indices
        video_exists = video_path.exists()

        if args.skip_existing and (already_in_log or video_exists):
            if not already_in_log and video_exists:
                logger.info(f"[{idx}] Video exists but not in log, adding entry")
                results_by_index[idx] = {
                    "index": idx,
                    "category": category,
                    "prompt": prompt_text,
                    "output": str(video_path),
                    "time": 0,
                    "status": "success (pre-existing)",
                }
                results = list(results_by_index.values())
                save_log_sorted(log_path, results)
            else:
                logger.info(f"[{idx}] Skipping (already completed)")
            skipped_count += 1
            continue

        logger.info(f"\n[{idx}] {category}: {prompt_text[:60]}...")

        try:
            start_time = time.time()

            # Generate with pruning
            video, info = generator.generate(prompt_text, seed=args.seed + idx)

            gen_time = time.time() - start_time

            # Save video
            save_start = time.time()
            from diffusers.utils import export_to_video

            export_to_video(video, str(video_path), fps=8)
            save_time = time.time() - save_start

            total_time = time.time() - start_time

            result = {
                "index": idx,
                "category": category,
                "prompt": prompt_text,
                "output": str(video_path),
                "time": total_time,
                "time_breakdown": {
                    "generation": round(gen_time, 2),
                    "save": round(save_time, 2),
                    "sampling": round(info.get("sampling_time", 0), 2),
                    "decode": round(info.get("decode_time", 0), 2),
                },
                "status": "success",
                "mode": info["mode"],
                "best_trajectory": info["best_trajectory"],
                "best_score": info["best_score"],
                "all_scores": info["all_scores"],
                "scoring_history": info["scoring_history"],
                "num_trajectories": args.num_trajectories,
            }

            logger.info(
                f"  Done in {total_time:.1f}s "
                f"(sample={info.get('sampling_time', 0):.1f}s, "
                f"decode={info.get('decode_time', 0):.1f}s), "
                f"score={info['best_score']:.3f}"
            )
            generated_count += 1

        except Exception as e:
            import traceback

            logger.error(f"  Error: {e}")
            logger.error(traceback.format_exc())
            result = {
                "index": idx,
                "category": category,
                "prompt": prompt_text,
                "output": str(video_path),
                "time": 0,
                "status": f"error: {str(e)}",
            }

        # Update or add result
        results_by_index[idx] = result

        # Save incrementally (sorted by index)
        results = list(results_by_index.values())
        save_log_sorted(log_path, results)

    # Cleanup
    generator.cleanup()

    # Final summary
    successful = sum(
        1
        for r in results_by_index.values()
        if r.get("status", "").startswith("success")
    )
    total_time = sum(r.get("time", 0) for r in results_by_index.values())

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total in log: {len(results_by_index)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped (existing): {skipped_count}")
    logger.info(f"Generated this run: {generated_count}")
    logger.info(f"Total time this run: {total_time / 60:.1f} min")
    if generated_count > 0:
        logger.info(f"Avg time/video: {total_time / generated_count:.1f}s")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
