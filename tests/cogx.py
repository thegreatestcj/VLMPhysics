"""
CogVideoX-5B Text-to-Video Generation Test
Usage: python test_cogvideox.py --prompt "A ball falls and bounces"
"""

import argparse
import time
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="tests/cogx.mp4")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading CogVideoX-5B...")

    # Load model with memory optimization
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16
    )

    # Memory optimization for 24GB GPU
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    #pipe.vae.enable_slicing()

    print(f"Generating video for: {args.prompt}")

    # Generate
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    torch.cuda.synchronize()
    start_time = time.time()

    video = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.steps,
        num_frames=args.num_frames,
        guidance_scale=6.0,
        generator=generator,
    ).frames[0]

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"Inference time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")

    # Save
    export_to_video(video, args.output, fps=8)
    print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()

