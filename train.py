import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from models import SpatialUNet, RecurrentUNet
from dataset import DenoisingDataset, SequenceDenoisingDataset
from losses import CombinedDenoisingLoss, calc_psnr, calc_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ray Repair 1 SPP Neural Denoiser")
    # Architecture & Model
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "recurrent_unet"],
                        help="Model architecture to train")
    parser.add_argument("--in_channels", type=int, default=7,
                        help="Input channels (3 Radiance + 2 Normals + 1 Depth + 1 Roughness)")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channel width for U-Net")
    parser.add_argument("--num_levels", type=int, default=4,
                        help="Number of encoder/decoder resolution levels")
    
    # Dataset & Training
    parser.add_argument("--train_dir", type=str, default="data/train", help="Path to training data folder")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Path to validation data folder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (can increase on A40 48GB)")
    parser.add_argument("--patch_size", type=int, default=256, help="Spatial crop patch size")
    parser.add_argument("--seq_length", type=int, default=5, help="Temporal sequence length for Recurrent U-Net")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Precision & Hardware (NVIDIA A40 optimization)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                        help="Mixed precision dtype (bfloat16 recommended for A40)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoints & Logs
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 dry step to test pipeline")
    return parser.parse_args()


def train():
    args = parse_args()
    print(f"=== Ray Repair Denoiser Training ===")
    print(f"Model: {args.model} | Device: {args.device} | Batch Size: {args.batch_size}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name} ({vram:.1f} GB VRAM)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 1. Initialize Model
    is_recurrent = (args.model == "recurrent_unet")
    if is_recurrent:
        model = RecurrentUNet(
            in_channels=args.in_channels,
            out_channels=3,
            base_channels=args.base_channels,
            num_levels=args.num_levels,
        )
    else:
        model = SpatialUNet(
            in_channels=args.in_channels,
            out_channels=3,
            base_channels=args.base_channels,
            num_levels=args.num_levels,
        )

    model = model.to(args.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameter Count: {param_count / 1e6:.2f} M")

    # 2. Datasets & Loaders
    if is_recurrent:
        train_dataset = SequenceDenoisingDataset(
            data_dir=args.train_dir,
            seq_length=args.seq_length,
            patch_size=args.patch_size,
            is_training=True,
        )
        val_dataset = SequenceDenoisingDataset(
            data_dir=args.val_dir if os.path.exists(args.val_dir) else args.train_dir,
            seq_length=args.seq_length,
            patch_size=args.patch_size,
            is_training=False,
        )
    else:
        train_dataset = DenoisingDataset(
            data_dir=args.train_dir,
            patch_size=args.patch_size,
            is_training=True,
        )
        val_dataset = DenoisingDataset(
            data_dir=args.val_dir if os.path.exists(args.val_dir) else args.train_dir,
            patch_size=args.patch_size,
            is_training=False,
        )

    if len(train_dataset) == 0:
        print(f"[Notice] No data found in {args.train_dir}. Generating fallback synthetic dataset...")
        from dataset.download_sample import generate_synthetic_scene
        generate_synthetic_scene(num_samples=30, output_dir=args.train_dir, is_sequence=is_recurrent)
        if is_recurrent:
            train_dataset = SequenceDenoisingDataset(data_dir=args.train_dir, seq_length=args.seq_length, is_training=True)
            val_dataset = SequenceDenoisingDataset(data_dir=args.train_dir, seq_length=args.seq_length, is_training=False)
        else:
            train_dataset = DenoisingDataset(data_dir=args.train_dir, is_training=True)
            val_dataset = DenoisingDataset(data_dir=args.train_dir, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
        drop_last=True if len(train_dataset) >= args.batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )

    # 3. Loss & Optimizer
    criterion = CombinedDenoisingLoss(
        weight_l1=1.0,
        weight_grad=0.2,
        weight_temporal=0.5 if is_recurrent else 0.0,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Precision setup (NVIDIA A40 supports bfloat16 natively)
    amp_dtype = torch.bfloat16 if (args.precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device_type = "cuda" if "cuda" in args.device else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(args.amp and device_type == "cuda" and amp_dtype == torch.float16))

    start_epoch = 1
    best_val_psnr = -1.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_psnr = checkpoint.get("best_val_psnr", -1.0)

    # 4. Training Loop
    global_step = 0
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(args.device)
            targets = batch["target"].to(args.device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type, enabled=(args.amp and device_type == "cuda"), dtype=amp_dtype):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take denoised output

                loss_dict = criterion(outputs, targets, is_sequence=is_recurrent)
                total_loss = loss_dict["loss"]

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += total_loss.item()
            global_step += 1

            if batch_idx % 10 == 0:
                writer.add_scalar("Train/Total_Loss", total_loss.item(), global_step)
                writer.add_scalar("Train/Rel_L1_Loss", loss_dict["rel_l1"].item(), global_step)
                writer.add_scalar("Train/Grad_Loss", loss_dict["grad"].item(), global_step)
                if is_recurrent:
                    writer.add_scalar("Train/Temporal_Loss", loss_dict["temporal"].item(), global_step)

            if args.dry_run:
                print("[Dry Run] Successfully completed 1 training forward/backward step.")
                return

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, len(train_loader))

        # 5. Validation Loop
        model.eval()
        val_psnr_list, val_ssim_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(args.device)
                targets = batch["target"].to(args.device)

                with torch.cuda.amp.autocast(enabled=args.amp and (args.device == "cuda"), dtype=amp_dtype):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                psnr_val = calc_psnr(outputs, targets)
                ssim_val = calc_ssim(outputs, targets)
                val_psnr_list.append(psnr_val)
                val_ssim_list.append(ssim_val)

        mean_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
        mean_ssim = np.mean(val_ssim_list) if val_ssim_list else 0.0

        writer.add_scalar("Val/PSNR", mean_psnr, epoch)
        writer.add_scalar("Val/SSIM", mean_ssim, epoch)
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {avg_loss:.5f} | Val PSNR: {mean_psnr:.2f} dB | Val SSIM: {mean_ssim:.4f} | Time: {epoch_time:.1f}s")

        # Save Checkpoint
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_psnr": best_val_psnr,
            "args": vars(args),
        }

        # Save latest
        torch.save(checkpoint_payload, os.path.join(args.checkpoint_dir, f"{args.model}_latest.pt"))

        # Save best
        if mean_psnr > best_val_psnr:
            best_val_psnr = mean_psnr
            torch.save(checkpoint_payload, os.path.join(args.checkpoint_dir, f"{args.model}_best.pt"))
            print(f"[*] New best validation PSNR ({mean_psnr:.2f} dB) saved to {args.checkpoint_dir}/{args.model}_best.pt")

        if epoch % args.save_interval == 0:
            torch.save(checkpoint_payload, os.path.join(args.checkpoint_dir, f"{args.model}_epoch_{epoch:03d}.pt"))

    writer.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    train()
