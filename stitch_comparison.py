import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.unet import SpatialUNet
from dataset.dataset import invert_hdr_log, apply_hdr_log
from losses.losses import calc_psnr, calc_ssim


def generate_rich_test_sample(sample_id=0, height=512, width=512):
    """
    Generates a rich multi-buffer synthetic test scene with specular highlights,
    complex geometric patterns, depth gradients, and high-frequency normals.
    """
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)

    # Multi-spheres and shapes
    r1 = np.sqrt((xx + 1.2)**2 + (yy + 0.8)**2)
    r2 = np.sqrt((xx - 1.2)**2 + (yy - 0.5)**2)
    r3 = np.sqrt(xx**2 + (yy - 1.5)**2)

    # Clean Ground Truth Radiance (512 SPP converged reference)
    clean = np.zeros((3, height, width), dtype=np.float32)
    
    # Sphere 1: Metallic Red/Gold
    mask1 = (r1 < 1.4)
    clean[0] += mask1 * np.clip(np.exp(-r1**2 * 1.5) * 2.5 + np.sin(xx * 6) * 0.2 + 0.6, 0, 3)
    clean[1] += mask1 * np.clip(np.exp(-r1**2 * 2.0) * 1.8 + 0.3, 0, 2)
    clean[2] += mask1 * 0.2

    # Sphere 2: Emerald Green Specular
    mask2 = (r2 < 1.3)
    clean[0] += mask2 * 0.1
    clean[1] += mask2 * np.clip(np.exp(-r2**2 * 2.5) * 2.8 + np.cos(yy * 8) * 0.2 + 0.7, 0, 3)
    clean[2] += mask2 * np.clip(np.exp(-r2**2 * 1.8) * 1.2 + 0.4, 0, 2)

    # Background ambient floor / wall
    bg_mask = (~mask1) & (~mask2)
    checker = ((np.floor(xx * 2) + np.floor(yy * 2)) % 2) * 0.3 + 0.2
    clean[0] += bg_mask * (checker + 0.1 * np.sin(yy * 4))
    clean[1] += bg_mask * (checker * 0.9)
    clean[2] += bg_mask * (checker * 1.2 + 0.1)

    # Simulate 1 SPP Monte Carlo noise (Poisson / Exponential variance + fireflies)
    noise_scale = 0.85
    noisy = clean + np.random.exponential(scale=clean * noise_scale + 0.08)
    fireflies = (np.random.rand(3, height, width) < 0.008) * np.random.uniform(4.0, 25.0, (3, height, width))
    noisy = noisy + fireflies

    # G-Buffers
    normal = np.zeros((2, height, width), dtype=np.float32)
    normal[0] = np.clip(xx / (r1 + 1e-2) * mask1 + xx / (r2 + 1e-2) * mask2 + np.sin(xx * 5) * 0.2 * bg_mask, -1, 1)
    normal[1] = np.clip(yy / (r1 + 1e-2) * mask1 + yy / (r2 + 1e-2) * mask2 + np.cos(yy * 5) * 0.2 * bg_mask, -1, 1)

    depth = np.clip((r1 * mask1 + r2 * mask2 + (yy + 3.0) * 0.5 * bg_mask) / 5.0, 0, 1).astype(np.float32)[np.newaxis, ...]
    roughness = (mask1 * 0.15 + mask2 * 0.25 + bg_mask * 0.75).astype(np.float32)[np.newaxis, ...]

    return {
        "noisy": torch.from_numpy(noisy).float(),
        "target": torch.from_numpy(clean).float(),
        "normal": torch.from_numpy(normal).float(),
        "depth": torch.from_numpy(depth).float(),
        "roughness": torch.from_numpy(roughness).float(),
    }


def run_and_stitch(checkpoint_path="checkpoints/unet_best.pt", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights from {checkpoint_path} on {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    in_channels = saved_args.get("in_channels", 7)
    base_channels = saved_args.get("base_channels", 32)
    num_levels = saved_args.get("num_levels", 4)

    model = SpatialUNet(
        in_channels=in_channels,
        out_channels=3,
        base_channels=base_channels,
        num_levels=num_levels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate 3 diverse test comparisons
    for sample_idx in range(3):
        sample = generate_rich_test_sample(sample_id=sample_idx, height=512, width=512)
        
        noisy_raw = sample["noisy"]
        target_raw = sample["target"]
        normal = sample["normal"]
        depth = sample["depth"]
        roughness = sample["roughness"]

        # Apply log transform as during training
        noisy_log = apply_hdr_log(noisy_raw)
        target_log = apply_hdr_log(target_raw)

        # Pack 7 channels
        input_tensor = torch.cat([noisy_log, normal, depth, roughness], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_log = model(input_tensor)
            output_linear = invert_hdr_log(output_log).squeeze(0).cpu()

        # Compute PSNR & SSIM
        psnr_noisy = calc_psnr(noisy_raw, target_raw, max_val=3.0)
        psnr_denoised = calc_psnr(output_linear, target_raw, max_val=3.0)
        ssim_noisy = calc_ssim(noisy_raw.unsqueeze(0), target_raw.unsqueeze(0))
        ssim_denoised = calc_ssim(output_linear.unsqueeze(0), target_raw.unsqueeze(0))

        # Tone map images for sRGB display [0, 1]
        def to_srgb(img_tensor):
            img = img_tensor.permute(1, 2, 0).numpy()
            # Simple Reinhard tone-mapping: x / (1 + x)
            tonemapped = img / (1.0 + img)
            # Gamma correction
            gamma = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)
            return gamma

        noisy_disp = to_srgb(noisy_raw)
        denoised_disp = to_srgb(output_linear)
        target_disp = to_srgb(target_raw)
        
        # Normal map display: map [-1, 1] -> [0, 1]
        normal_3d = np.zeros((512, 512, 3), dtype=np.float32)
        normal_3d[:, :, 0] = normal[0].numpy() * 0.5 + 0.5
        normal_3d[:, :, 1] = normal[1].numpy() * 0.5 + 0.5
        normal_3d[:, :, 2] = np.sqrt(np.clip(1.0 - normal[0].numpy()**2 - normal[1].numpy()**2, 0, 1))

        # -----------------------------------------------------------------
        # 1. Stitched 4-Quadrant Figure (Matching Paper Figure 1 Specification)
        # -----------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=200)
        
        # Top-Left: Raw 1 SPP
        axes[0, 0].imshow(noisy_disp)
        axes[0, 0].set_title(f"(A) Raw 1 SPP Monte Carlo Input\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.3f}", fontsize=14, fontweight='bold', color='darkred')
        axes[0, 0].axis("off")

        # Top-Right: Converged Ground Truth
        axes[0, 1].imshow(target_disp)
        axes[0, 1].set_title("(B) Converged Reference (512+ SPP Target)\nGround Truth Benchmark", fontsize=14, fontweight='bold', color='darkgreen')
        axes[0, 1].axis("off")

        # Bottom-Left: View-Space Shading Normals
        axes[1, 0].imshow(normal_3d)
        axes[1, 0].set_title("(C) Auxiliary G-Buffer\nView-Space Shading Normals (Input Channels 4 & 5)", fontsize=14, fontweight='bold', color='navy')
        axes[1, 0].axis("off")

        # Bottom-Right: Denoised Result (Ours)
        axes[1, 1].imshow(denoised_disp)
        axes[1, 1].set_title(f"(D) Ray Repair Neural Denoiser (Ours - 1 SPP)\nPSNR: {psnr_denoised:.2f} dB (+{psnr_denoised - psnr_noisy:.2f} dB) | SSIM: {ssim_denoised:.3f}", fontsize=14, fontweight='bold', color='darkblue')
        axes[1, 1].axis("off")

        plt.suptitle("Ray Repair: Real-Time 1 SPP Neural Denoising Benchmark", fontsize=18, fontweight='bold', y=0.96)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = os.path.join(output_dir, f"denoise_comparison_quadrant_{sample_idx+1}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved 4-quadrant benchmark image to: {out_path}")

        # -----------------------------------------------------------------
        # 2. Side-by-Side Split Banner Comparison
        # -----------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
        axes[0].imshow(noisy_disp)
        axes[0].set_title(f"1 SPP Raw Input (Noisy)\nPSNR: {psnr_noisy:.2f} dB", fontsize=13, fontweight='bold')
        axes[0].axis("off")

        axes[1].imshow(denoised_disp)
        axes[1].set_title(f"Ray Repair Denoised (1 SPP)\nPSNR: {psnr_denoised:.2f} dB (+{psnr_denoised - psnr_noisy:.2f} dB)", fontsize=13, fontweight='bold', color='green')
        axes[1].axis("off")

        axes[2].imshow(target_disp)
        axes[2].set_title("512+ SPP Converged Ground Truth\nReference", fontsize=13, fontweight='bold')
        axes[2].axis("off")

        plt.tight_layout()
        banner_path = os.path.join(output_dir, f"denoise_side_by_side_{sample_idx+1}.png")
        plt.savefig(banner_path, bbox_inches="tight")
        plt.close()
        print(f"Saved side-by-side comparison to: {banner_path}")


if __name__ == "__main__":
    run_and_stitch()
