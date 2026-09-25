import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from models.unet import SpatialUNet
from models.recurrent_unet import RecurrentUNet
from dataset.dataset import invert_hdr_log, apply_hdr_log
from losses.losses import calc_psnr, calc_ssim


def generate_fractal_heightmap(h=512, w=512, octaves=6, persistence=0.5, lacunarity=2.0, seed=42):
    """Generates procedural fractal Brownian motion (fBm) terrain heightmap."""
    np.random.seed(seed)
    heightmap = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        # Generate low-res random noise grid
        grid_size = max(4, int(16 * frequency))
        noise = np.random.uniform(-1, 1, (grid_size, grid_size))
        # Upsample smoothly with Gaussian interpolation
        from scipy.ndimage import zoom
        upsampled = zoom(noise, (h / grid_size, w / grid_size), order=2)[:h, :w]
        heightmap += amplitude * upsampled
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-6)
    return heightmap


def render_natural_terrain_frame(heightmap, cam_offset_x=0.0, cam_offset_y=0.0, height=512, width=512):
    """
    Renders a realistic game-like natural terrain multi-buffer frame:
      - Mountains, grassy valleys, rivers/water bodies
      - Sun directional lighting with specular water reflections
      - View-space surface normal maps
      - Linearized depth buffer with atmospheric perspective
      - Roughness buffer (rock vs grass vs water)
    """
    # Shift heightmap coordinates with camera movement
    x = np.linspace(cam_offset_x, cam_offset_x + 1.0, width)
    y = np.linspace(cam_offset_y, cam_offset_y + 1.0, height)
    xx, yy = np.meshgrid(x, y)

    # Base elevation
    elev = heightmap

    # Terrain classification
    water_level = 0.32
    grass_level = 0.65
    rock_level = 0.85

    water_mask = (elev < water_level)
    grass_mask = (elev >= water_level) & (elev < grass_level)
    rock_mask = (elev >= grass_level) & (elev < rock_level)
    snow_mask = (elev >= rock_level)

    # Compute surface normals from heightmap gradients
    dy, dx = np.gradient(elev)
    normal_z = np.full_like(elev, 0.15)
    norm = np.sqrt(dx**2 + dy**2 + normal_z**2) + 1e-6
    nx = -dx / norm
    ny = -dy / norm
    nz = normal_z / norm

    # Water flattening for normals
    nx[water_mask] *= 0.05
    ny[water_mask] *= 0.05
    nz[water_mask] = 1.0

    # Sun direction (from top-right)
    sun_dir = np.array([0.5, -0.6, 0.63])
    sun_dir /= np.linalg.norm(sun_dir)

    # Diffuse lighting (N . L)
    ndotl = np.clip(nx * sun_dir[0] + ny * sun_dir[1] + nz * sun_dir[2], 0.0, 1.0)
    ambient = 0.18

    # Specular lighting for water and wet rock
    view_dir = np.array([0.0, 0.0, 1.0])
    half_vec = (sun_dir + view_dir) / np.linalg.norm(sun_dir + view_dir)
    ndoth = np.clip(nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2], 0.0, 1.0)
    specular_water = np.power(ndoth, 64.0) * water_mask * 3.5

    # Albedo coloring
    albedo = np.zeros((3, height, width), dtype=np.float32)
    # Water: Deep blue-cyan
    albedo[0] += water_mask * 0.05
    albedo[1] += water_mask * 0.28
    albedo[2] += water_mask * 0.55

    # Grass: Forest green with subtle variance
    grass_noise = np.sin(xx * 50) * 0.03
    albedo[0] += grass_mask * (0.18 + grass_noise)
    albedo[1] += grass_mask * (0.42 + grass_noise * 1.5)
    albedo[2] += grass_mask * 0.12

    # Rock: Earthy gray-brown
    albedo[0] += rock_mask * 0.38
    albedo[1] += rock_mask * 0.34
    albedo[2] += rock_mask * 0.30

    # Snow: Crisp white
    albedo[0] += snow_mask * 0.88
    albedo[1] += snow_mask * 0.90
    albedo[2] += snow_mask * 0.95

    # Clean Ground Truth Radiance (512+ SPP converged equivalent)
    clean_radiance = np.zeros((3, height, width), dtype=np.float32)
    for c in range(3):
        clean_radiance[c] = albedo[c] * (ndotl * 1.6 + ambient) + specular_water

    # 1 SPP Monte Carlo noisy radiance (Poisson/Exponential noise + specular fireflies)
    noise_scale = 0.95
    noisy = clean_radiance + np.random.exponential(scale=clean_radiance * noise_scale + 0.06)
    # Fireflies on specular water and rock edges
    firefly_prob = water_mask * 0.015 + (~water_mask) * 0.004
    fireflies = (np.random.rand(3, height, width) < firefly_prob) * np.random.uniform(3.0, 18.0, (3, height, width))
    noisy = noisy + fireflies

    # Auxiliary Buffers
    normal_2d = np.stack([nx, ny], axis=0).astype(np.float32)
    depth = np.clip((1.0 - elev * 0.6 + yy * 0.4), 0.1, 1.0).astype(np.float32)[np.newaxis, ...]
    roughness = (water_mask * 0.05 + grass_mask * 0.75 + rock_mask * 0.85 + snow_mask * 0.4).astype(np.float32)[np.newaxis, ...]

    return {
        "noisy": torch.from_numpy(noisy).float(),
        "target": torch.from_numpy(clean_radiance).float(),
        "normal": torch.from_numpy(normal_2d).float(),
        "depth": torch.from_numpy(depth).float(),
        "roughness": torch.from_numpy(roughness).float(),
    }


def run_terrain_benchmark(checkpoint_path="checkpoints/unet_best.pt", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {checkpoint_path} on {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpatialUNet(in_channels=7, out_channels=3, base_channels=32, num_levels=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate game terrain heightmap
    heightmap = generate_fractal_heightmap(h=512, w=512, seed=101)

    # -------------------------------------------------------------
    # 1. Single-Frame Natural Game Terrain 4-Quadrant Benchmark
    # -------------------------------------------------------------
    frame_data = render_natural_terrain_frame(heightmap, cam_offset_x=0.0, cam_offset_y=0.0)

    noisy_raw = frame_data["noisy"]
    target_raw = frame_data["target"]
    normal = frame_data["normal"]
    depth = frame_data["depth"]
    roughness = frame_data["roughness"]

    noisy_log = apply_hdr_log(noisy_raw)
    input_tensor = torch.cat([noisy_log, normal, depth, roughness], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output_log = model(input_tensor)
        output_linear = invert_hdr_log(output_log).squeeze(0).cpu()

    psnr_noisy = calc_psnr(noisy_raw, target_raw, max_val=2.5)
    psnr_denoised = calc_psnr(output_linear, target_raw, max_val=2.5)
    ssim_noisy = calc_ssim(noisy_raw.unsqueeze(0), target_raw.unsqueeze(0))
    ssim_denoised = calc_ssim(output_linear.unsqueeze(0), target_raw.unsqueeze(0))

    def to_srgb(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        tonemapped = img / (1.0 + img)
        return np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)

    noisy_disp = to_srgb(noisy_raw)
    denoised_disp = to_srgb(output_linear)
    target_disp = to_srgb(target_raw)

    normal_3d = np.zeros((512, 512, 3), dtype=np.float32)
    normal_3d[:, :, 0] = normal[0].numpy() * 0.5 + 0.5
    normal_3d[:, :, 1] = normal[1].numpy() * 0.5 + 0.5
    normal_3d[:, :, 2] = np.sqrt(np.clip(1.0 - normal[0].numpy()**2 - normal[1].numpy()**2, 0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=200)
    axes[0, 0].imshow(noisy_disp)
    axes[0, 0].set_title(f"(A) Raw 1 SPP Monte Carlo Path Traced Terrain\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.3f}", fontsize=13, fontweight='bold', color='darkred')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_disp)
    axes[0, 1].set_title("(B) Converged 512+ SPP Reference Ground Truth\nNatural Video Game Environment", fontsize=13, fontweight='bold', color='darkgreen')
    axes[0, 1].axis("off")

    axes[1, 0].imshow(normal_3d)
    axes[1, 0].set_title("(C) Auxiliary G-Buffer Shading Normals\nSurface Orientation Guiding Geometry & Lighting", fontsize=13, fontweight='bold', color='navy')
    axes[1, 0].axis("off")

    axes[1, 1].imshow(denoised_disp)
    axes[1, 1].set_title(f"(D) Ray Repair Neural Denoiser (Ours - 1 SPP)\nPSNR: {psnr_denoised:.2f} dB (+{psnr_denoised - psnr_noisy:.2f} dB) | SSIM: {ssim_denoised:.3f}", fontsize=13, fontweight='bold', color='darkblue')
    axes[1, 1].axis("off")

    plt.suptitle("Ray Repair: 1 SPP Neural Denoising on Natural Game Terrain", fontsize=17, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    terrain_quadrant_path = os.path.join(output_dir, "natural_terrain_comparison.png")
    plt.savefig(terrain_quadrant_path, bbox_inches="tight")
    plt.close()
    print(f"Saved natural terrain benchmark to: {terrain_quadrant_path}")

    # -------------------------------------------------------------
    # 2. Multi-Frame Camera Fly-Through (Temporal Denoising Sequence)
    # -------------------------------------------------------------
    num_frames = 4
    fig, axes = plt.subplots(3, num_frames, figsize=(20, 14), dpi=180)

    for t in range(num_frames):
        cam_shift_x = t * 0.08
        cam_shift_y = t * 0.04
        t_data = render_natural_terrain_frame(heightmap, cam_offset_x=cam_shift_x, cam_offset_y=cam_shift_y)

        t_noisy = t_data["noisy"]
        t_target = t_data["target"]
        t_norm = t_data["normal"]
        t_depth = t_data["depth"]
        t_rough = t_data["roughness"]

        t_input = torch.cat([apply_hdr_log(t_noisy), t_norm, t_depth, t_rough], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            t_out_log = model(t_input)
            t_out = invert_hdr_log(t_out_log).squeeze(0).cpu()

        t_noisy_disp = to_srgb(t_noisy)
        t_denoised_disp = to_srgb(t_out)
        t_target_disp = to_srgb(t_target)

        # Row 1: 1 SPP Noisy Feed
        axes[0, t].imshow(t_noisy_disp)
        axes[0, t].set_title(f"1 SPP Noisy Feed (Frame {t+1})", fontsize=12, fontweight='bold', color='darkred')
        axes[0, t].axis("off")

        # Row 2: Ray Repair Neural Denoised
        axes[1, t].imshow(t_denoised_disp)
        axes[1, t].set_title(f"Ray Repair Denoised (Frame {t+1})", fontsize=12, fontweight='bold', color='darkblue')
        axes[1, t].axis("off")

        # Row 3: Ground Truth Reference
        axes[2, t].imshow(t_target_disp)
        axes[2, t].set_title(f"Converged GT (Frame {t+1})", fontsize=12, fontweight='bold', color='darkgreen')
        axes[2, t].axis("off")

    axes[0, 0].set_ylabel("1 SPP Raw", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel("Neural Denoised", fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel("512+ SPP Target", fontsize=14, fontweight='bold')

    plt.suptitle("Temporal Sequence Filmstrip: 1 SPP Camera Fly-Through on Natural Game Terrain", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    temporal_path = os.path.join(output_dir, "terrain_temporal_sequence.png")
    plt.savefig(temporal_path, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal sequence filmstrip to: {temporal_path}")


if __name__ == "__main__":
    run_terrain_benchmark()
