#!/bin/bash
# ==============================================================================
# SLURM Submission Script for USC CARC (Center for Advanced Research Computing)
# Hardware: NVIDIA A40 GPU (48GB VRAM)
# Project: Ray Repair - 1 SPP Real-Time Neural Denoising
# ==============================================================================

#SBATCH --job-name=ray_repair_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Node: $(hostname)"
echo "Allocated GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start Time: $(date)"
echo "=========================================================="

# 1. Load CARC Modules
module purge
module load conda
module load cuda/12.1.1

# 2. Activate Conda Environment
# (Ensure you created it beforehand via: conda env create -f environment.yml)
source activate ray_repair || conda activate ray_repair

# Create required directories
mkdir -p logs checkpoints runs data/train data/val

# 3. Optional: Download or generate synthetic bootstrap data if training folder is empty
if [ ! "$(ls -A data/train 2>/dev/null)" ]; then
    echo "No training data found in data/train. Generating synthetic bootstrap data..."
    python dataset/download_sample.py --num_samples 200 --output_dir data/train
    python dataset/download_sample.py --num_samples 50 --output_dir data/val
fi

# 4. Launch Training
# NVIDIA A40 48GB allows batch sizes 16-32 and bfloat16 mixed precision
python train.py \
    --model unet \
    --in_channels 7 \
    --base_channels 32 \
    --num_levels 4 \
    --epochs 100 \
    --batch_size 16 \
    --patch_size 256 \
    --lr 2e-4 \
    --precision bfloat16 \
    --amp \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir runs

echo "=========================================================="
echo "Training completed at: $(date)"
echo "=========================================================="
