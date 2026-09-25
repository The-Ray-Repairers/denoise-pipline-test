import os
import argparse
import numpy as np
import torch


def generate_synthetic_scene(
    num_samples: int = 50,
    height: int = 256,
    width: int = 256,
    output_dir: str = "data/synthetic",
    is_sequence: bool = False,
    seq_length: int = 5,
):
    """
    Generates synthetic 1 SPP Monte Carlo buffers with ground truth for pipeline testing.
    Packs: noisy radiance (1spp), converged target, normals, depth, roughness.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} synthetic multi-buffer samples in '{output_dir}'...")

    for i in range(num_samples):
        if is_sequence:
            # Generate animated sequence
            frames_noisy = []
            frames_target = []
            frames_normal = []
            frames_depth = []
            frames_roughness = []

            for t in range(seq_length):
                shift = t * 2.0
                # Generate clean synthetic ground truth
                x = np.linspace(-2 + shift * 0.1, 2 + shift * 0.1, width)
                y = np.linspace(-2, 2, height)
                xx, yy = np.meshgrid(x, y)
                radius = np.sqrt(xx**2 + yy**2)

                clean = np.zeros((3, height, width), dtype=np.float32)
                clean[0] = np.exp(-radius**2) * 2.0  # Red highlight
                clean[1] = np.clip(np.sin(xx * 3.0) * 0.5 + 0.5, 0, 1)
                clean[2] = np.clip(np.cos(yy * 3.0) * 0.5 + 0.5, 0, 1)

                # Simulate 1 SPP Monte Carlo noise (Poisson + High Variance Spikes)
                noise_scale = 1.0 / np.sqrt(1.0)
                noisy = clean + np.random.exponential(scale=clean * noise_scale)
                # Random fireflies
                fireflies = (np.random.rand(3, height, width) < 0.005) * np.random.uniform(5.0, 30.0, (3, height, width))
                noisy = noisy + fireflies

                # G-Buffers
                normal = np.zeros((2, height, width), dtype=np.float32)
                normal[0] = np.clip(xx / (radius + 1e-3), -1, 1)
                normal[1] = np.clip(yy / (radius + 1e-3), -1, 1)

                depth = np.clip((radius + 1.0) / 4.0, 0, 1).astype(np.float32)[np.newaxis, ...]
                roughness = np.full((1, height, width), 0.2 + 0.1 * (i % 5), dtype=np.float32)

                frames_noisy.append(noisy)
                frames_target.append(clean)
                frames_normal.append(normal)
                frames_depth.append(depth)
                frames_roughness.append(roughness)

            np.savez_compressed(
                os.path.join(output_dir, f"sample_seq_{i:04d}.npz"),
                noisy=np.array(frames_noisy, dtype=np.float32),
                target=np.array(frames_target, dtype=np.float32),
                normal=np.array(frames_normal, dtype=np.float32),
                depth=np.array(frames_depth, dtype=np.float32),
                roughness=np.array(frames_roughness, dtype=np.float32),
            )
        else:
            # Single frame
            x = np.linspace(-2, 2, width)
            y = np.linspace(-2, 2, height)
            xx, yy = np.meshgrid(x, y)
            radius = np.sqrt(xx**2 + yy**2)

            clean = np.zeros((3, height, width), dtype=np.float32)
            clean[0] = np.exp(-radius**2) * 2.0
            clean[1] = np.clip(np.sin(xx * 3.0 + i) * 0.5 + 0.5, 0, 1)
            clean[2] = np.clip(np.cos(yy * 3.0 + i) * 0.5 + 0.5, 0, 1)

            noise_scale = 1.0
            noisy = clean + np.random.exponential(scale=clean * noise_scale + 0.05)
            fireflies = (np.random.rand(3, height, width) < 0.005) * np.random.uniform(5.0, 30.0, (3, height, width))
            noisy = noisy + fireflies

            normal = np.zeros((2, height, width), dtype=np.float32)
            normal[0] = np.clip(xx / (radius + 1e-3), -1, 1)
            normal[1] = np.clip(yy / (radius + 1e-3), -1, 1)

            depth = np.clip((radius + 1.0) / 4.0, 0, 1).astype(np.float32)[np.newaxis, ...]
            roughness = np.full((1, height, width), 0.3, dtype=np.float32)

            np.savez_compressed(
                os.path.join(output_dir, f"sample_{i:04d}.npz"),
                noisy=noisy,
                target=clean,
                normal=normal,
                depth=depth,
                roughness=roughness,
            )

    print(f"Successfully generated {num_samples} sample files in {output_dir}")


def download_huggingface_ommatidia(output_dir: str = "data/ommatidia", max_samples: int = 100):
    """
    Downloads sample subset from Hugging Face dataset mad-bot/ommatidia.
    """
    try:
        from datasets import load_dataset
        print(f"Connecting to Hugging Face dataset 'mad-bot/ommatidia'...")
        ds = load_dataset("mad-bot/ommatidia", split="train", streaming=True)
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            # Save processed item
            torch.save(item, os.path.join(output_dir, f"hf_sample_{count:04d}.pt"))
            count += 1
            if count % 10 == 0:
                print(f"Downloaded {count}/{max_samples} samples...")
        print(f"Completed Hugging Face download into {output_dir}")
    except Exception as e:
        print(f"Could not load Hugging Face dataset directly: {e}")
        print("Falling back to synthetic data generation...")
        generate_synthetic_scene(num_samples=max_samples, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate Monte Carlo sample datasets")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic multi-buffer data")
    parser.add_argument("--hf", action="store_true", help="Download from Hugging Face (mad-bot/ommatidia)")
    parser.add_argument("--sequence", action="store_true", help="Generate multi-frame temporal sequences")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output_dir", type=str, default="data/train", help="Target output folder")
    args = parser.parse_args()

    if args.hf:
        download_huggingface_ommatidia(output_dir=args.output_dir, max_samples=args.num_samples)
    else:
        generate_synthetic_scene(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            is_sequence=args.sequence
        )
