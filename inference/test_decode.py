# test_decode_compare.py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# 加载你保存的 latents
# 在 pruning.py 的 _decode_best 之前添加: torch.save(best_latents, "debug_latents.pt")
latents = torch.load("debug_latents.pt")
print(f"Loaded latents: {latents.shape}")  # 应该是 [1, 13, 16, 60, 90]

# 方法 1: 官方 pipeline decode
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,  # 全 FP32
).to("cuda")

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
# 官方 decode_latents 期望 [B, C, T, H, W]
latents_bcthw = latents.permute(0, 2, 1, 3, 4).float().cuda()
print(f"After permute: {latents_bcthw.shape}")  # [1, 16, 13, 60, 90]

# 用官方方法
video = pipe.decode_latents(latents_bcthw)
print(f"Video shape: {video.shape}, range: [{video.min():.3f}, {video.max():.3f}]")

# 保存
if video.dim() == 5:  # [B, C, T, H, W]
    video = (video / 2 + 0.5).clamp(0, 1)
    video = video[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    video = (video * 255).astype("uint8")
    export_to_video(video, "official_decode.mp4", fps=8)
    print("Saved official_decode.mp4")
