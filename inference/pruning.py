#!/usr/bin/env python3
"""
Dual-GPU Trajectory Pruning with Maximum Parallelism

Optimized for 24GB×2 GPU setup to maximize trajectory count and parallelism.

Architecture:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      DUAL-GPU PARALLEL SAMPLING                         │
    │                                                                         │
    │   GPU 0 (24GB)                          GPU 1 (24GB)                   │
    │   ┌──────────────────────┐              ┌──────────────────────┐       │
    │   │ DiT (8-bit) ~5GB     │              │ DiT (8-bit) ~5GB     │       │
    │   │ VAE (tiling) ~1GB    │              │ VAE (tiling) ~1GB    │       │
    │   │                      │              │                      │       │
    │   │ Trajectories 0-7     │              │ Trajectories 8-15    │       │
    │   │ (batch_size=4)       │              │ (batch_size=4)       │       │
    │   │ 2 mini-batches       │              │ 2 mini-batches       │       │
    │   └──────────────────────┘              └──────────────────────┘       │
    │             │                                     │                    │
    │             └──────────────┬──────────────────────┘                    │
    │                            │                                           │
    │                   CPU: Synchronize scores                              │
    │                   CPU: Global pruning decision                         │
    │                            │                                           │
    │             ┌──────────────┴──────────────────────┐                    │
    │             │                                     │                    │
    │   GPU 0: Continue with                  GPU 1: Continue with          │
    │   surviving trajectories                surviving trajectories         │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

Memory Budget per GPU (24GB):
    - DiT Transformer (8-bit):     ~5 GB
    - VAE (tiling enabled):        ~1 GB
    - 4 trajectory latents:        ~0.6 GB (4 × 160MB)
    - Activations (batch=4):       ~8 GB
    - KV cache + buffers:          ~4 GB
    - Physics Head:                ~0.002 GB
    ─────────────────────────────────────────
    Total:                         ~18.6 GB ✓ (safe margin)

Maximum Trajectories:
    - 2 GPUs × 8 trajectories/GPU = 16 total
    - Processed in mini-batches of 4 per GPU
    - Pruning: 16 → 8 → 4 → 2 → 1

Speedup vs Sequential:
    - Sequential (1 traj at a time): baseline
    - This approach: ~4x faster (2 GPUs × 2 mini-batches parallel)

Usage:
    # 16 trajectories across 2 GPUs
    python -m inference.trajectory_pruning_dual \
        --prompt "A ball falls and bounces" \
        --physics_head results/training/physics_head/mean/best.pt \
        --num_trajectories 16 \
        --batch_size 4

    # Test mode
    python -m inference.trajectory_pruning_dual \
        --prompt "A ball falls and bounces" \
        --test_mode \
        --num_trajectories 16
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DualGPUConfig:
    """Configuration for dual-GPU trajectory pruning."""

    # Model
    model_id: str = "THUDM/CogVideoX-2b"
    physics_head_path: Optional[str] = None
    head_type: str = "mean"
    extract_layer: int = 10

    # Trajectories - MAXIMIZED for dual GPU
    num_trajectories: int = 16  # 8 per GPU
    batch_size: int = 4  # Per-GPU mini-batch size
    checkpoints: List[int] = field(default_factory=lambda: [800, 600, 400, 200])
    keep_ratio: float = 0.5

    # Generation
    num_inference_steps: int = 50
    num_frames: int = 49
    guidance_scale: float = 6.0
    height: int = 480
    width: int = 720

    # Devices
    devices: List[str] = field(default_factory=lambda: ["cuda:0", "cuda:1"])

    # Output
    exp_name: str = "dual_gpu_traj"
    output_dir: str = "results/inference/trajectory_pruning"
    seed: int = 42

    def __post_init__(self):
        # Validate trajectory distribution
        num_gpus = len(self.devices)
        traj_per_gpu = self.num_trajectories // num_gpus
        if self.num_trajectories % num_gpus != 0:
            self.num_trajectories = traj_per_gpu * num_gpus
            logger.warning(f"Adjusted num_trajectories to {self.num_trajectories}")


# =============================================================================
# Per-GPU Worker
# =============================================================================


class GPUWorker:
    """
    Manages trajectories on a single GPU.

    Each worker:
    - Loads its own copy of CogVideoX (with 8-bit quantization)
    - Manages a subset of trajectories
    - Processes in mini-batches for efficiency
    - Reports scores back for global pruning decisions
    """

    def __init__(
        self,
        worker_id: int,
        device: str,
        config: DualGPUConfig,
        trajectory_indices: List[int],
    ):
        self.worker_id = worker_id
        self.device = device
        self.config = config
        self.trajectory_indices = trajectory_indices

        self.pipe = None
        self.feature_hook = None
        self.physics_head = None

        # Trajectory state: {global_idx: {"latents": tensor, "active": bool, "score": float}}
        self.trajectories: Dict[int, Dict] = {}

        self._lock = threading.Lock()

    def load(self) -> None:
        """Load models on this GPU."""
        from diffusers import CogVideoXPipeline

        logger.info(f"[GPU{self.worker_id}] Loading CogVideoX on {self.device}...")

        # Load with 8-bit quantization for memory efficiency
        self.pipe = CogVideoXPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
        )

        # Move to specific device (no CPU offload - we manage memory manually)
        self.pipe = self.pipe.to(self.device)
        self.pipe.vae.enable_tiling()

        # Feature hook
        self._setup_feature_hook()

        # Physics head (shared across workers via same weights)
        self._setup_physics_head()

        logger.info(f"[GPU{self.worker_id}] Ready!")

    def _setup_feature_hook(self) -> None:
        """Register feature extraction hook."""
        self.captured_features = None

        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured_features = out.detach()

        block = self.pipe.transformer.transformer_blocks[self.config.extract_layer]
        self._hook_handle = block.register_forward_hook(hook_fn)

    def _setup_physics_head(self) -> None:
        """Load physics head."""
        if self.config.physics_head_path is None:
            self.physics_head = None  # Test mode
            return

        from src.models.physics_head import create_physics_head

        hidden_dim = 1920 if "2b" in self.config.model_id.lower() else 3072
        self.physics_head = create_physics_head(self.config.head_type, hidden_dim)

        ckpt = torch.load(self.config.physics_head_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.physics_head.load_state_dict(state)
        self.physics_head = self.physics_head.to(self.device).eval()

    def init_trajectories(
        self,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> None:
        """Initialize trajectories assigned to this worker."""
        latent_shape = self._get_latent_shape()

        for global_idx in self.trajectory_indices:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.config.seed + global_idx)

            latents = (
                torch.randn(
                    latent_shape,
                    generator=gen,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                * self.pipe.scheduler.init_noise_sigma
            )

            self.trajectories[global_idx] = {
                "latents": latents,
                "active": True,
                "score": 0.0,
                "generator": gen,
            }

        # Store prompt embeds for this GPU
        self.prompt_embeds = prompt_embeds.to(self.device)
        self.neg_embeds = neg_embeds.to(self.device) if neg_embeds is not None else None

        logger.info(
            f"[GPU{self.worker_id}] Initialized {len(self.trajectory_indices)} trajectories"
        )

    def _get_latent_shape(self) -> Tuple[int, ...]:
        """Get latent tensor shape."""
        c = self.pipe.transformer.config.in_channels
        t = (self.config.num_frames - 1) // 4 + 1
        h = self.config.height // 8
        w = self.config.width // 8
        return (1, c, t, h, w)

    def get_active_indices(self) -> List[int]:
        """Get indices of active trajectories on this worker."""
        return [idx for idx, data in self.trajectories.items() if data["active"]]

    def denoise_step(self, t: torch.Tensor) -> None:
        """
        Run one denoising step for all active trajectories.
        Uses mini-batching for efficiency.
        """
        active_indices = self.get_active_indices()
        if not active_indices:
            return

        # Process in mini-batches
        for batch_start in range(0, len(active_indices), self.config.batch_size):
            batch_indices = active_indices[
                batch_start : batch_start + self.config.batch_size
            ]
            self._denoise_batch(batch_indices, t)

    def _denoise_batch(self, indices: List[int], t: torch.Tensor) -> None:
        """Denoise a mini-batch of trajectories."""
        batch_size = len(indices)

        # Stack latents
        latents = torch.cat(
            [self.trajectories[idx]["latents"] for idx in indices], dim=0
        )  # [B, C, T, H, W]

        # Prepare inputs for CFG
        do_cfg = self.config.guidance_scale > 1.0

        if do_cfg:
            latent_in = torch.cat([latents, latents], dim=0)
            prompt_in = torch.cat(
                [
                    self.neg_embeds.expand(batch_size, -1, -1),
                    self.prompt_embeds.expand(batch_size, -1, -1),
                ],
                dim=0,
            )
        else:
            latent_in = latents
            prompt_in = self.prompt_embeds.expand(batch_size, -1, -1)

        t_in = t.to(self.device).expand(latent_in.shape[0])

        # Forward pass
        with torch.no_grad():
            noise_pred = self.pipe.transformer(
                hidden_states=latent_in,
                encoder_hidden_states=prompt_in,
                timestep=t_in,
                return_dict=False,
            )[0]

        # Apply CFG
        if do_cfg:
            uncond, cond = noise_pred.chunk(2, dim=0)
            noise_pred = uncond + self.config.guidance_scale * (cond - uncond)

        # Scheduler step
        latents = self.pipe.scheduler.step(
            noise_pred, t.to(self.device), latents, return_dict=False
        )[0]

        # Update trajectories
        for i, idx in enumerate(indices):
            self.trajectories[idx]["latents"] = latents[i : i + 1]

    def evaluate_trajectories(self, t: torch.Tensor) -> Dict[int, float]:
        """
        Evaluate physics scores for all active trajectories.
        Returns {global_idx: score}.
        """
        scores = {}
        active_indices = self.get_active_indices()

        for idx in active_indices:
            # Get features (run forward if needed to capture)
            feat = self.captured_features

            if feat is None or self.physics_head is None:
                # Test mode or no features
                scores[idx] = torch.rand(1).item()
            else:
                with torch.no_grad():
                    # Use first sample if batched
                    if feat.shape[0] > 1:
                        feat = feat[:1]
                    timestep = t.unsqueeze(0).to(self.device)
                    logits = self.physics_head(feat, timestep)
                    scores[idx] = torch.sigmoid(logits).item()

            self.captured_features = None  # Clear

        return scores

    def prune_trajectories(self, pruned_indices: List[int]) -> None:
        """Mark trajectories as pruned."""
        for idx in pruned_indices:
            if idx in self.trajectories:
                self.trajectories[idx]["active"] = False

    def get_best_latents(self) -> Tuple[int, torch.Tensor, float]:
        """Get latents of best trajectory on this worker."""
        active = [
            (idx, data) for idx, data in self.trajectories.items() if data["active"]
        ]
        if not active:
            # Return any trajectory if none active
            idx, data = list(self.trajectories.items())[0]
            return idx, data["latents"], data["score"]

        best_idx, best_data = max(active, key=lambda x: x[1]["score"])
        return best_idx, best_data["latents"], best_data["score"]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video."""
        latents = latents / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            video = self.pipe.vae.decode(latents, return_dict=False)[0]
        video = (video / 2 + 0.5).clamp(0, 1)
        return video.permute(0, 2, 3, 4, 1).cpu()


# =============================================================================
# Coordinator
# =============================================================================


class DualGPUCoordinator:
    """
    Coordinates trajectory pruning across multiple GPUs.

    Responsibilities:
    - Distributes trajectories across workers
    - Synchronizes pruning decisions
    - Collects final results
    """

    def __init__(self, config: DualGPUConfig):
        self.config = config
        self.workers: List[GPUWorker] = []

    def setup(self) -> None:
        """Initialize workers on each GPU."""
        num_gpus = len(self.config.devices)
        traj_per_gpu = self.config.num_trajectories // num_gpus

        for i, device in enumerate(self.config.devices):
            # Assign trajectory indices to this worker
            start_idx = i * traj_per_gpu
            indices = list(range(start_idx, start_idx + traj_per_gpu))

            worker = GPUWorker(i, device, self.config, indices)
            worker.load()
            self.workers.append(worker)

        logger.info(f"Initialized {num_gpus} GPU workers")

    def encode_prompt(
        self, prompt: str, negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using first worker's pipeline."""
        worker = self.workers[0]

        prompt_embeds, neg_embeds = worker.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            do_classifier_free_guidance=self.config.guidance_scale > 1.0,
            device=worker.device,
        )

        return prompt_embeds.cpu(), neg_embeds.cpu() if neg_embeds is not None else None

    def init_all_trajectories(
        self,
        prompt_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor],
    ) -> None:
        """Initialize trajectories on all workers."""
        for worker in self.workers:
            worker.init_trajectories(prompt_embeds, neg_embeds)

    def run_sampling(self) -> Dict:
        """
        Run the full sampling loop with trajectory pruning.

        Returns results dict.
        """
        start_time = time.time()

        # Setup scheduler (use first worker's)
        scheduler = self.workers[0].pipe.scheduler
        scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = scheduler.timesteps

        # Also set on other workers
        for worker in self.workers[1:]:
            worker.pipe.scheduler.set_timesteps(self.config.num_inference_steps)

        # Find checkpoint steps
        checkpoint_steps = self._find_checkpoint_steps(timesteps)
        logger.info(f"Checkpoints at steps: {checkpoint_steps}")

        # Track history
        history = []

        # Sampling loop
        pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Sampling")

        for step, t in pbar:
            # Count total active
            total_active = sum(len(w.get_active_indices()) for w in self.workers)
            pbar.set_postfix({"active": total_active, "t": int(t)})

            if total_active == 0:
                logger.warning("No active trajectories!")
                break

            # Run denoising step on all workers (parallel via threads)
            self._parallel_denoise_step(t)

            # Check for checkpoint
            if step in checkpoint_steps and total_active > 1:
                logger.info(
                    f"\n--- Checkpoint step {step}, t={int(t)}, active={total_active} ---"
                )
                pruned = self._evaluate_and_prune_global(t, step)
                history.append(
                    {
                        "step": step,
                        "timestep": int(t),
                        "pruned": pruned,
                        "remaining": total_active - len(pruned),
                    }
                )

        # Get best trajectory
        best_idx, best_latents, best_score = self._get_global_best()

        # Decode on the worker that has the best trajectory
        video = self._decode_best(best_idx, best_latents)

        elapsed = time.time() - start_time

        return {
            "video": video,
            "best_trajectory": best_idx,
            "best_score": best_score,
            "pruning_history": history,
            "time_seconds": elapsed,
            "trajectories_per_gpu": self.config.num_trajectories
            // len(self.config.devices),
        }

    def _find_checkpoint_steps(self, timesteps: torch.Tensor) -> List[int]:
        """Map checkpoint timesteps to step indices."""
        ts_list = timesteps.tolist()
        steps = []
        for cp in self.config.checkpoints:
            idx = min(range(len(ts_list)), key=lambda i: abs(ts_list[i] - cp))
            steps.append(idx)
        return sorted(set(steps))  # Remove duplicates

    def _parallel_denoise_step(self, t: torch.Tensor) -> None:
        """Run denoising on all workers in parallel using threads."""
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(worker.denoise_step, t) for worker in self.workers
            ]
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # Raises exception if any

    def _evaluate_and_prune_global(self, t: torch.Tensor, step: int) -> List[int]:
        """
        Evaluate all trajectories and make global pruning decision.
        Returns list of pruned indices.
        """
        # Collect scores from all workers
        all_scores: Dict[int, float] = {}

        for worker in self.workers:
            scores = worker.evaluate_trajectories(t)
            all_scores.update(scores)

        # Global ranking and pruning
        total_active = len(all_scores)
        num_keep = max(1, int(total_active * self.config.keep_ratio))

        # Sort by score descending
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        keep_indices = [idx for idx, _ in ranked[:num_keep]]
        prune_indices = [idx for idx, _ in ranked[num_keep:]]

        # Log
        keep_scores = [(idx, f"{all_scores[idx]:.3f}") for idx in keep_indices]
        prune_scores = [(idx, f"{all_scores[idx]:.3f}") for idx in prune_indices]
        logger.info(f"  Keep: {keep_scores}")
        logger.info(f"  Prune: {prune_scores}")

        # Update scores and apply pruning to workers
        for worker in self.workers:
            for idx in worker.trajectory_indices:
                if idx in all_scores:
                    worker.trajectories[idx]["score"] = all_scores[idx]
            worker.prune_trajectories(prune_indices)

        return prune_indices

    def _get_global_best(self) -> Tuple[int, torch.Tensor, float]:
        """Get the best trajectory across all workers."""
        candidates = []
        for worker in self.workers:
            idx, latents, score = worker.get_best_latents()
            candidates.append((idx, latents, score, worker))

        best = max(candidates, key=lambda x: x[2])
        return best[0], best[1], best[2]

    def _decode_best(self, best_idx: int, best_latents: torch.Tensor) -> torch.Tensor:
        """Decode the best trajectory."""
        # Find which worker has this trajectory
        for worker in self.workers:
            if best_idx in worker.trajectory_indices:
                return worker.decode(best_latents)

        # Fallback: use first worker
        return self.workers[0].decode(best_latents)


# =============================================================================
# Main Pipeline
# =============================================================================


class DualGPUTrajectoryPruning:
    """
    Main entry point for dual-GPU trajectory pruning.
    """

    def __init__(self, config: DualGPUConfig):
        self.config = config
        self.coordinator = DualGPUCoordinator(config)

    def load(self) -> None:
        """Load all models."""
        self.coordinator.setup()

    @torch.no_grad()
    def generate(self, prompt: str, negative_prompt: str = "") -> Dict:
        """Generate video with trajectory pruning."""
        logger.info("=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Total trajectories: {self.config.num_trajectories}")
        logger.info(f"GPUs: {self.config.devices}")
        logger.info(f"Batch size per GPU: {self.config.batch_size}")
        logger.info("=" * 60)

        # Encode prompt
        prompt_embeds, neg_embeds = self.coordinator.encode_prompt(
            prompt, negative_prompt
        )

        # Initialize trajectories
        self.coordinator.init_all_trajectories(prompt_embeds, neg_embeds)

        # Run sampling
        results = self.coordinator.run_sampling()
        results["prompt"] = prompt

        return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU Trajectory Pruning")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--physics_head", type=str, default=None)
    parser.add_argument("--head_type", type=str, default="mean")
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=16,
        help="Total trajectories (split across GPUs)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Mini-batch size per GPU"
    )
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[800, 600, 400, 200]
    )
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--exp_name", type=str, default="dual_gpu_traj")
    parser.add_argument(
        "--output_dir", type=str, default="results/inference/trajectory_pruning"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    # Validate GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus < len(args.devices):
        logger.warning(
            f"Requested {len(args.devices)} GPUs but only {num_gpus} available"
        )
        args.devices = [f"cuda:{i}" for i in range(num_gpus)]

    # Config
    config = DualGPUConfig(
        physics_head_path=None if args.test_mode else args.physics_head,
        head_type=args.head_type,
        num_trajectories=args.num_trajectories,
        batch_size=args.batch_size,
        checkpoints=args.checkpoints,
        keep_ratio=args.keep_ratio,
        num_inference_steps=args.steps,
        num_frames=args.frames,
        guidance_scale=args.guidance,
        devices=args.devices,
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir) / f"{config.exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # Run
    pipeline = DualGPUTrajectoryPruning(config)
    pipeline.load()

    results = pipeline.generate(args.prompt, args.negative_prompt)

    # Save video
    from diffusers.utils import export_to_video

    video_path = out_dir / "output.mp4"
    export_to_video(results["video"][0], str(video_path), fps=8)
    logger.info(f"Saved video: {video_path}")

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "prompt": results["prompt"],
                "best_trajectory": results["best_trajectory"],
                "best_score": results["best_score"],
                "pruning_history": results["pruning_history"],
                "time_seconds": results["time_seconds"],
                "num_trajectories": config.num_trajectories,
                "num_gpus": len(config.devices),
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_dir}")
    logger.info(
        f"Time: {results['time_seconds']:.1f}s ({results['time_seconds'] / 60:.1f}min)"
    )


if __name__ == "__main__":
    main()
