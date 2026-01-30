#!/usr/bin/env python3
"""
PhyGenBench Video Generation with Physics Head Trajectory Pruning

Generates videos for PhyGenBench prompts using CogVideoX-2B with
trajectory pruning guided by the trained physics discriminator head.

Output Structure:
    outputs/phygenbench/physics/
    ├── generation_log.json
    ├── output_video_1.mp4
    ├── output_video_2.mp4
    └── ...

Usage:
    # Generate all prompts (single GPU)
    python eval/generate_physics.py --start 0 --end 160

    # Generate specific range
    python eval/generate_physics.py --start 0 --end 80

    # Skip existing videos
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
    physics_head_path: str = (
        "results/training/physics_head/layers_ablation_20260127_074539/layer_10/best.pt"
    )
    head_type: str = "temporal_simple"
    extract_layer: int = 10

    # Trajectory pruning
    # 4 trajectories: 4 -> 2 -> 1 (faster, ~2.5x baseline time)
    # 8 trajectories: 8 -> 4 -> 2 -> 1 (slower, ~5x baseline time)
    num_trajectories: int = 4
    checkpoints: List[int] = field(default_factory=lambda: [600, 400])  # 2 prune points
    keep_ratio: float = 0.5

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
    ):
        self.device = device
        self.extract_layer = extract_layer
        self.captured_features = None
        self._hook_handle = None

        # Load physics head
        from src.models.physics_head import create_physics_head

        hidden_dim = 1920  # CogVideoX-2B
        self.head = create_physics_head(head_type, hidden_dim)

        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.head.load_state_dict(state)
        self.head = self.head.to(device).eval()

        logger.info(f"Physics head loaded from {head_path}")

    def setup_hook(self, transformer) -> None:
        """Register feature extraction hook."""

        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured_features = out.detach()

        block = transformer.transformer_blocks[self.extract_layer]
        self._hook_handle = block.register_forward_hook(hook_fn)

    def remove_hook(self) -> None:
        """Remove hook."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def score(self, timestep: int, num_frames: int = 49) -> float:
        """Score current captured features."""
        if self.captured_features is None:
            return 0.5  # Neutral score if no features

        feat = self.captured_features

        # Pool spatial dimensions
        num_latent_frames = (num_frames - 1) // 4 + 1  # 13 for 49 frames
        B, seq_len, D = feat.shape
        spatial_size = seq_len // num_latent_frames
        feat = feat.view(B, num_latent_frames, spatial_size, D).mean(dim=2)

        # Get score
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        logits = self.head(feat.float(), t.float())
        score = torch.sigmoid(logits).item()

        self.captured_features = None
        return score


# =============================================================================
# Trajectory Pruning Generator
# =============================================================================


class TrajectoryPruningGenerator:
    """
    Generates videos using trajectory pruning with physics head guidance.

    Strategy:
    - Maintain multiple trajectories (e.g., 8)
    - At checkpoints, evaluate all with physics head
    - Keep top half based on physics scores
    - Final: 8 -> 4 -> 2 -> 1
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

    @torch.no_grad()
    def generate(self, prompt: str, seed: int) -> Tuple[torch.Tensor, Dict]:
        """
        Generate video with trajectory pruning.

        Returns:
            video: Generated video tensor
            info: Dictionary with pruning history and timing
        """
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

            # Initialize latents manually (like pruning.py)
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
                    "seed": seed + i,
                }
            )

        # Setup scheduler
        self.pipe.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        checkpoint_steps = self._get_checkpoint_steps(timesteps)

        # Move timesteps to device
        timesteps = timesteps.to(self.device)

        logger.info(
            f"Trajectories: {num_traj}, Checkpoints at steps: {checkpoint_steps}"
        )
        logger.info(
            f"Latent shape: {latent_shape}, timesteps device: {timesteps.device}"
        )

        # Sampling loop
        pruning_history = []
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

                # Scheduler step - t needs to be on the right device
                tr["latents"] = self.pipe.scheduler.step(
                    noise_pred, t, tr["latents"], return_dict=False
                )[0]

            # Checkpoint: evaluate and prune
            if step in checkpoint_steps and num_active > 1:
                t_val = int(t.item()) if hasattr(t, "item") else int(t)

                # Score each trajectory (run one more forward to get features)
                for tr in active_trajs:
                    if self.evaluator:
                        # Quick forward to capture features
                        latent_input = self.pipe.scheduler.scale_model_input(
                            tr["latents"], t
                        )
                        t_input = t.expand(latent_input.shape[0])  # 1D array
                        _ = self.pipe.transformer(
                            hidden_states=latent_input,
                            timestep=t_input,
                            encoder_hidden_states=prompt_embeds[
                                :1
                            ],  # No CFG for scoring
                            return_dict=False,
                        )
                        tr["score"] = self.evaluator.score(
                            t_val, self.config.num_frames
                        )
                    else:
                        tr["score"] = torch.rand(1).item()

                # Sort by score and keep top half
                active_trajs.sort(key=lambda x: x["score"], reverse=True)
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

                logger.info(f"  Step {step} (t={t_val}): Keep {kept}, Prune {pruned}")
                pruning_history.append(
                    {
                        "step": step,
                        "timestep": t_val,
                        "kept": kept,
                        "pruned": pruned,
                    }
                )

                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()

        # Get best trajectory
        active_trajs = [tr for tr in trajectories if tr["active"]]
        best = max(active_trajs, key=lambda x: x["score"])

        timing["sampling"] = time.time() - sampling_start
        logger.info(
            f"Best trajectory: idx={best['idx']}, score={best['score']:.3f}, sampling_time={timing['sampling']:.1f}s"
        )

        # Decode to video
        decode_start = time.time()
        video = self._decode(best["latents"])
        timing["decode"] = time.time() - decode_start

        info = {
            "best_trajectory": best["idx"],
            "best_score": best["score"],
            "pruning_history": pruning_history,
            "sampling_time": timing["sampling"],
            "decode_time": timing["decode"],
            "encode_time": timing["encode"],
        }

        return video, info

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video."""
        # Move transformer to CPU to free memory
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

        # Post-process to numpy for export_to_video
        # output_type="np" returns list of [H, W, C] numpy arrays
        video = self.pipe.video_processor.postprocess_video(video, output_type="np")[0]

        # Move transformer back
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
        default="results/training/physics_head/layers_ablation_20260127_074539/layer_10/best.pt",
        help="Path to physics head checkpoint",
    )
    parser.add_argument("--head-type", type=str, default="temporal_simple")
    parser.add_argument("--extract-layer", type=int, default=10)

    # Trajectory pruning
    # 4 traj: ~2.5x baseline time (~500s/video)
    # 8 traj: ~5x baseline time (~1000s/video)
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

    # Setup generator
    generator = TrajectoryPruningGenerator(config)
    generator.load()

    # Generation loop
    results = []
    log_path = output_dir / "generation_log.json"

    for idx in tqdm(indices, desc="Generating"):
        prompt_data = prompts[idx]
        prompt_text = prompt_data.get("caption", prompt_data.get("prompt", ""))
        category = prompt_data.get(
            "sub_category", prompt_data.get("main_category", "Unknown")
        )

        # Output path (1-indexed like baseline)
        video_path = output_dir / f"output_video_{idx + 1}.mp4"

        # Skip if exists
        if args.skip_existing and video_path.exists():
            logger.info(f"[{idx}] Skipping (exists)")
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
                "best_trajectory": info["best_trajectory"],
                "best_score": info["best_score"],
                "num_trajectories": args.num_trajectories,
            }

            logger.info(
                f"  Done in {total_time:.1f}s (sample={info.get('sampling_time', 0):.1f}s, decode={info.get('decode_time', 0):.1f}s), score={info['best_score']:.3f}"
            )

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

        results.append(result)

        # Save log incrementally
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)

    # Cleanup
    generator.cleanup()

    # Final summary
    successful = sum(1 for r in results if r["status"] == "success")
    total_time = sum(r["time"] for r in results)

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Total time: {total_time / 60:.1f} min")
    logger.info(f"Avg time/video: {total_time / max(1, successful):.1f}s")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
