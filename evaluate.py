import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import SpatialUNet, RecurrentUNet
from dataset import DenoisingDataset, SequenceDenoisingDataset
from dataset.dataset import invert_hdr_log
from losses import calc_psnr, calc_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ray Repair Neural Denoiser")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_renders", action="store_true", help="Save visual comparison images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save render comparisons")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    saved_args = checkpoint.get("args", {})

    model_type = saved_args.get("model", "unet")
    in_channels = saved_args.get("in_channels", 7)
    base_channels = saved_args.get("base_channels", 32)
    num_levels = saved_args.get("num_levels", 4)
    is_recurrent = (model_type == "recurrent_unet")

    if is_recurrent:
        model = RecurrentUNet(in_channels=in_channels, base_channels=base_channels, num_levels=num_levels)
        test_dataset = SequenceDenoisingDataset(data_dir=args.test_dir, is_training=False, patch_size=None)
    else:
        model = SpatialUNet(in_channels=in_channels, base_channels=base_channels, num_levels=num_levels)
        test_dataset = DenoisingDataset(data_dir=args.test_dir, is_training=False, patch_size=None)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    noisy_psnr_list, denoised_psnr_list = [], []
    noisy_ssim_list, denoised_ssim_list = [], []
    latencies = []

    if args.save_renders:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nEvaluating on {len(test_dataset)} samples...")

    # Warmup GPU
    dummy_in = torch.randn(1, in_channels, 512, 512, device=args.device)
    if is_recurrent:
        dummy_in = dummy_in.unsqueeze(1)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_in)
    if args.device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs = batch["input"].to(args.device)
            targets = batch["target"].to(args.device)

            # Measure inference latency
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if args.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # in ms

            # Raw 1 SPP radiance (first 3 channels of input)
            raw_noisy = inputs[:, :3] if not is_recurrent else inputs[:, :, :3]

            # Invert log transform back to linear space for metric evaluation
            noisy_linear = invert_hdr_log(raw_noisy)
            pred_linear = invert_hdr_log(outputs)
            target_linear = invert_hdr_log(targets)

            # Compute PSNR & SSIM
            psnr_noisy = calc_psnr(noisy_linear, target_linear, max_val=1.0)
            psnr_denoised = calc_psnr(pred_linear, target_linear, max_val=1.0)
            ssim_noisy = calc_ssim(noisy_linear, target_linear)
            ssim_denoised = calc_ssim(pred_linear, target_linear)

            noisy_psnr_list.append(psnr_noisy)
            denoised_psnr_list.append(psnr_denoised)
            noisy_ssim_list.append(ssim_noisy)
            denoised_ssim_list.append(ssim_denoised)

    avg_noisy_psnr = np.mean(noisy_psnr_list)
    avg_denoised_psnr = np.mean(denoised_psnr_list)
    avg_noisy_ssim = np.mean(noisy_ssim_list)
    avg_denoised_ssim = np.mean(denoised_ssim_list)
    avg_latency = np.mean(latencies)

    print("\n" + "="*50)
    print("           EVALUATION BENCHMARK RESULTS")
    print("="*50)
    print(f" Raw 1 SPP Input:   PSNR = {avg_noisy_psnr:.2f} dB | SSIM = {avg_noisy_ssim:.4f}")
    print(f" Denoised Output:   PSNR = {avg_denoised_psnr:.2f} dB | SSIM = {avg_denoised_ssim:.4f}")
    print(f" PSNR Gain:        +{avg_denoised_psnr - avg_noisy_psnr:.2f} dB")
    print(f" Inference Latency: {avg_latency:.2f} ms / frame (Target: <33.3 ms for 30fps)")
    print("="*50)


if __name__ == "__main__":
    evaluate()
