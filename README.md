# Ray Repair: 1 SPP Real-Time Neural Denoiser (CARC Package)

This repository contains the training, evaluation, and export pipeline for **Ray Repair: Open-sourcing Real-Time Neural Denoising for Path-Traced Rendering**, configured specifically for **USC's CARC computing cluster** with **NVIDIA A40 GPUs (48GB VRAM)**.

---

## 📁 Package Structure

```
ray_repair_carc/
├── models/
│   ├── unet.py                 # 7-channel Spatial U-Net (Real-time baseline)
│   └── recurrent_unet.py       # Temporal Recurrent U-Net with ConvGRU bottleneck
├── dataset/
│   ├── dataset.py              # Multi-buffer loader (Radiance, Normals, Depth, Roughness)
│   └── download_sample.py      # Bootstrap script (Synthetic 1 SPP data or Hugging Face)
├── losses/
│   └── losses.py               # Relative L1, Gradient Edge loss, and Temporal loss
├── train.py                    # Main PyTorch training loop (AMP bfloat16 for A40, checkpointing)
├── evaluate.py                 # Benchmark script (PSNR, SSIM, Latency ms/frame)
├── export_onnx.py              # ONNX / TensorRT exporter for Unreal Engine 5
├── slurm_train.sh              # SLURM batch submission script for CARC
├── environment.yml             # Conda environment definition
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart on USC CARC

### Step 1: Upload to CARC
From your local machine, rsync or scp the folder to your CARC scratch directory:
```bash
rsync -avz /path/to/ray_repair_carc username@discovery.usc.edu:/scratch1/username/
```

### Step 2: Set Up Conda Environment on CARC
Log into CARC and create the environment:
```bash
ssh username@discovery.usc.edu
cd /scratch1/username/ray_repair_carc

# Load conda module and create environment
module load conda
conda env create -f environment.yml
conda activate ray_repair
```

### Step 3: Launch Training via SLURM
Submit the training job to the A40 GPU partition:
```bash
sbatch slurm_train.sh
```
Check job status:
```bash
squeue -u $USER
```
View live training logs:
```bash
tail -f logs/train_<JOB_ID>.out
```

---

## 🧪 Training & Evaluation Options

### Spatial U-Net (Single Frame Baseline)
```bash
python train.py --model unet --batch_size 16 --precision bfloat16 --epochs 100
```

### Recurrent U-Net (Temporal Sequence Denoising)
```bash
python train.py --model recurrent_unet --seq_length 5 --batch_size 8 --precision bfloat16 --epochs 100
```

### Evaluation & Benchmarking
Run the benchmark to verify PSNR gain and latency (<33 ms / frame):
```bash
python evaluate.py --checkpoint checkpoints/unet_best.pt --test_dir data/val
```

---

## 🎮 Unreal Engine 5 Integration (ONNX Export)

Export your trained checkpoint to ONNX for use inside Unreal Engine 5 (via UE5 NNE plugin or TensorRT):
```bash
python export_onnx.py --checkpoint checkpoints/unet_best.pt --output ray_repair.onnx
```

---

## ⚙️ Buffer Specification (7 Channels)

Following the paper specification, each pixel is packed into a 7-channel input tensor:
1. **[0:3] Noisy HDR Radiance** (1 SPP, log-compressed $\log(1+x)$)
2. **[3:5] View-Space Shading Normal** (2D projected vector, $[-1, 1]$)
3. **[5:6] Linearized Depth** ($[0, 1]$)
4. **[6:7] Material Roughness** ($[0, 1]$)
