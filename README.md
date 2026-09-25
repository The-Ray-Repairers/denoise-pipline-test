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

---

## 🌐 Online Datasets & Denoising Benchmarks

Below is a curated list of open-access datasets and scene repositories containing multi-buffer Monte Carlo path-traced images (radiance, albedo, normal, depth, roughness) and converged ground-truth pairs for training and benchmarking:

| Dataset / Resource | Source / Platform | Description & Buffer Contents | Link |
| :--- | :--- | :--- | :--- |
| **`mad-bot/ommatidia`** | Hugging Face | Monte Carlo path-traced samples packed with Radiance, World Normals, Linear Depth, Diffuse Albedo, Specular $F_0$, and Roughness. | [Hugging Face Dataset](https://huggingface.co/datasets/mad-bot/ommatidia) |
| **Real-Time Denoising Neural Bilateral Grid** | GitHub (Meng et al.) | ~19 GB open benchmark dataset containing 1 SPP path-traced images paired with G-buffers (normals, depth, diffuse albedo) and converged ground truth. | [GitHub Repository](https://github.com/xmeng525/RealTimeDenoisingNeuralBilateralGrid) |
| **Disney Research Denoising Dataset** | Disney Research Studios | Production-grade multi-SPP Monte Carlo renders with auxiliary feature buffers (albedo, normal, depth) generated via the Tungsten renderer. | [Disney Research Archive](https://studios.disneyresearch.com/data-sets/) |
| **Noisebase** | GitHub (Balint et al.) | Framework and dataset containing asynchronous `.exr` data loaders for neural Monte Carlo denoising pipelines. | [GitHub Repository](https://github.com/balintio/noisebase) |
| **Sample-Based MC Denoising (SBMC)** | Adobe Research | Multi-sample per-pixel path-tracing sequences and ground truth references with full auxiliary passes. | [GitHub Repository](https://github.com/adobe/sbmc) |
| **Benedikt Bitterli Rendering Resources** | Academic Resource | 32 standardized PBR 3D benchmark scenes (Classroom, Living Room, Kitchen, Cornell Box, San Miguel) in Tungsten, Mitsuba, and PBRT-v4 formats. | [Bitterli Resources](https://benedikt-bitterli.me/resources/) |
| **McGuire Computer Graphics Archive** | Casual Effects | Industry-standard graphics test scenes (Crytek Sponza, Rungholt, Bistro, Conference Room) for custom path-tracing dataset generation. | [Graphics Archive](https://casual-effects.com/data/) |
| **Open Image Denoise (OIDN) Toolkit** | RenderKit / Intel | Official preprocessing scripts and training utilities for training auxiliary-buffer HDR denoising models. | [GitHub Repository](https://github.com/RenderKit/oidn) |

