# CUT Phase Tutorial: Foggy Cityscapes Dataset Generation

A comprehensive guide for generating synthetic foggy/clear image pairs using Contrastive Unpaired Translation (CUT) on Vast.ai GPU instances.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Part 1: Vast.ai Setup](#part-1-vastai-setup)
- [Part 2: Environment Configuration](#part-2-environment-configuration)
- [Part 3: Dataset Preparation](#part-3-dataset-preparation)
- [Part 4: Training CUT Models](#part-4-training-cut-models)
- [Part 5: Generating Synthetic Images](#part-5-generating-synthetic-images)
- [Part 6: Checkpoint Management](#part-6-checkpoint-management)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This tutorial guides you through the process of using **Contrastive Unpaired Translation (CUT)** to generate synthetic image pairs for domain adaptation research. Specifically, we will:

1. Train a CUT model to translate **Clear → Foggy** images (A→B)
2. Train a CUT model to translate **Foggy → Clear** images (B→A)
3. Generate synthetic datasets for downstream tasks

### What is CUT?

CUT (Contrastive Unpaired Translation) is a state-of-the-art unpaired image-to-image translation method that:
- Uses patchwise contrastive learning
- Requires **less memory** than CycleGAN
- Trains **faster** than traditional methods
- Does not require paired training data

**Paper:** [Contrastive Learning for Unpaired Image-to-Image Translation](https://arxiv.org/pdf/2007.15651) (ECCV 2020)

---

## Prerequisites

### Hardware Requirements (Vast.ai)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 4070 | RTX 4090 / A100 |
| VRAM | 8 GB | 16+ GB |
| RAM | 12 GB | 32+ GB |
| Storage | 50 GB | 100+ GB |
| GPU Series | RTX 40xx+ | RTX 40xx / A100 |

### Software Requirements

- Python 3.8
- PyTorch >= 1.4.0
- CUDA-compatible GPU drivers

---

## Part 1: Vast.ai Setup

### Step 1.1: Create a Vast.ai Account

1. Navigate to [Vast.ai](https://vast.ai/)
2. Create an account and add credits
3. Navigate to the **Console** → **Search** tab

### Step 1.2: Select GPU Instance

Filter instances with the following criteria:

```
GPU: RTX 4070, RTX 4080, RTX 4090, A100, or higher
RAM: >= 12 GB
Storage: >= 50 GB
CUDA Version: >= 11.0
```

**Recommended Instance Types:**
- **Budget:** RTX 4070 (12GB VRAM) - Good for batch_size=4
- **Performance:** RTX 4090 (24GB VRAM) - Allows batch_size=8+
- **Production:** A100 (40GB/80GB) - Maximum throughput

### Step 1.3: Select Docker Image

Choose one of the following base images:
- `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`
- `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`

### Step 1.4: Launch and Connect

1. Click **RENT** to launch the instance
2. Wait for the instance to initialize (1-3 minutes)
3. Connect via SSH or Jupyter:

```bash
# SSH Connection (copy from Vast.ai dashboard)
ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
```

---

## Part 2: Environment Configuration

### Step 2.1: System Updates

```bash
# Update system packages
sudo apt update

# Install required system dependencies
sudo apt install -y software-properties-common

# Add Python 3.8 repository
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package list
sudo apt update

# Install Python 3.8 and development tools
sudo apt install -y python3.8 python3.8-venv python3.8-dev
```

### Step 2.2: Project Setup

```bash
# Navigate to workspace (adjust path as needed)
cd /workspace

# Clone or upload the project
# Option A: Clone from repository
git clone <your-repository-url> contrastive-unpaired-translation

# Option B: Upload and extract zip file
unzip contrastive-unpaired-translation.zip

# Enter project directory
cd contrastive-unpaired-translation
```

### Step 2.3: Python Environment

```bash
# Create virtual environment with Python 3.8
python3.8 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2.4: Verify Installation

```bash
# Verify Python version
python --version
# Expected: Python 3.8.x

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output:**
```
PyTorch: 1.x.x
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090 (or similar)
```

---

## Part 3: Dataset Preparation

### Step 3.1: Create Dataset Directory

```bash
# Create dataset folder structure
mkdir -p datasets/cityscapes
```

### Step 3.2: Download Clear Images (Domain A)

```bash
# Download clear images from Google Drive
gdown 1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b

# Extract to dataset directory
unzip clear_image.zip -d datasets/cityscapes

# Clean up zip file to save storage
rm clear_image.zip
```

### Step 3.3: Download Foggy Images (Domain B)

```bash
# Download foggy images from Google Drive
gdown 1Y8GA79Hng9MD6llcSi3ys4NgWouxTnZb

# Extract to dataset directory
unzip foggy_image.zip -d datasets/cityscapes

# Clean up zip file
rm foggy_image.zip
```

### Step 3.4: Organize Dataset Structure

The dataset should follow this structure:

```
datasets/
└── cityscapes/
    ├── trainA/          # Clear images (Domain A)
    │   ├── image001.png
    │   ├── image002.png
    │   └── ...
    ├── trainB/          # Foggy images (Domain B)
    │   ├── image001.png
    │   ├── image002.png
    │   └── ...
    ├── testA/           # (Optional) Test clear images
    └── testB/           # (Optional) Test foggy images
```

**If folder names don't match, rename them:**

```bash
# Example: Rename folders if needed
cd datasets/cityscapes
mv clear trainA      # Rename clear folder to trainA
mv foggy trainB      # Rename foggy folder to trainB
cd ../..
```

### Step 3.5: Verify Dataset

```bash
# Check dataset structure
ls -la datasets/cityscapes/

# Count images in each domain
echo "Domain A (Clear) images: $(ls datasets/cityscapes/trainA | wc -l)"
echo "Domain B (Foggy) images: $(ls datasets/cityscapes/trainB | wc -l)"
```

---

## Part 4: Training CUT Models

### Step 4.1: Training Clear → Foggy (A→B)

This model learns to add fog to clear images.

```bash
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_AtoB \
    --CUT_mode CUT \
    --preprocess scale_shortside_and_crop \
    --load_size 640 \
    --crop_size 640 \
    --batch_size 4 \
    --gpu_ids 0 \
    --display_id -1
```

**Parameter Explanation:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--dataroot` | `./datasets/cityscapes` | Path to dataset |
| `--name` | `cityscapes_CUT_AtoB` | Experiment name (checkpoint folder) |
| `--CUT_mode` | `CUT` | Use full CUT model (vs FastCUT) |
| `--preprocess` | `scale_shortside_and_crop` | Preprocessing strategy |
| `--load_size` | `640` | Scale short side to this size |
| `--crop_size` | `640` | Random crop size for training |
| `--batch_size` | `4` | Batch size (adjust based on VRAM) |
| `--gpu_ids` | `0` | GPU device ID |
| `--display_id` | `-1` | Disable visdom visualization |

**Training Time Estimates:**

| GPU | Batch Size | ~Time per Epoch | Total (~200 epochs) |
|-----|------------|-----------------|---------------------|
| RTX 4070 | 4 | 15-20 min | 50-67 hours |
| RTX 4090 | 4 | 8-12 min | 27-40 hours |
| A100 | 8 | 5-8 min | 17-27 hours |

### Step 4.2: Monitor Training Progress

Training progress is saved to:
- **Checkpoints:** `./checkpoints/cityscapes_CUT_AtoB/`
- **Loss logs:** `./checkpoints/cityscapes_CUT_AtoB/loss_log.txt`
- **Sample images:** `./checkpoints/cityscapes_CUT_AtoB/web/`

```bash
# Monitor loss in real-time
tail -f checkpoints/cityscapes_CUT_AtoB/loss_log.txt
```

### Step 4.3: Training Foggy → Clear (B→A)

After the A→B model completes, train the reverse direction:

```bash
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_BtoA \
    --CUT_mode CUT \
    --preprocess scale_shortside_and_crop \
    --load_size 640 \
    --crop_size 640 \
    --batch_size 4 \
    --gpu_ids 0 \
    --display_id -1 \
    --direction BtoA
```

> **Note:** The only difference is `--direction BtoA`, which swaps source and target domains.

---

## Part 5: Generating Synthetic Images

### Step 5.1: Generate Foggy Images (A→B)

```bash
python test.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_AtoB \
    --CUT_mode CUT \
    --preprocess none \
    --phase train \
    --gpu_ids 0
```

**Parameter Notes:**
- `--preprocess none`: Keep original image dimensions
- `--phase train`: Use training set images for generation

**Output Location:** `./results/cityscapes_CUT_AtoB/train_latest/images/`

```
results/cityscapes_CUT_AtoB/train_latest/images/
├── fake_B/      # Generated foggy images ⭐
├── real_A/      # Original clear images (input)
└── idt_B/       # Identity preserved images
```

### Step 5.2: Generate Clear Images (B→A)

```bash
python test.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_BtoA \
    --CUT_mode CUT \
    --preprocess none \
    --phase train \
    --gpu_ids 0
```

**Output Location:** `./results/cityscapes_CUT_BtoA/train_latest/images/`

```
results/cityscapes_CUT_BtoA/train_latest/images/
├── fake_B/      # Generated clear images ⭐
├── real_A/      # Original foggy images (input)
└── idt_B/       # Identity preserved images
```

### Step 5.3: Batch Processing Tips

For large datasets, you can process in batches:

```bash
# Process with specific batch size for inference
python test.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_AtoB \
    --CUT_mode CUT \
    --preprocess none \
    --phase train \
    --gpu_ids 0 \
    --num_test 1000  # Limit number of images
```

---

## Part 6: Checkpoint Management

### Step 6.1: Checkpoint Structure

```
checkpoints/
├── cityscapes_CUT_AtoB/
│   ├── latest_net_G.pth       # Generator (main model) ⭐
│   ├── latest_net_F.pth       # Feature extractor
│   ├── latest_net_D.pth       # Discriminator
│   ├── latest_optimizer_G.pth # Optimizer state
│   ├── latest_optimizer_D.pth # Optimizer state
│   ├── loss_log.txt           # Training losses
│   ├── opt.txt                # Training options
│   └── web/                   # Sample images
└── cityscapes_CUT_BtoA/
    └── ... (same structure)
```

### Step 6.2: Save Checkpoints

**Compress for backup/transfer:**

```bash
# Create compressed archive of A→B checkpoint
tar -czvf cityscapes_CUT_AtoB_checkpoint.tar.gz checkpoints/cityscapes_CUT_AtoB/

# Create compressed archive of B→A checkpoint
tar -czvf cityscapes_CUT_BtoA_checkpoint.tar.gz checkpoints/cityscapes_CUT_BtoA/
```

**Upload to cloud storage:**

```bash
# Install rclone for cloud uploads (optional)
curl https://rclone.org/install.sh | sudo bash

# Configure and upload to Google Drive, S3, etc.
rclone copy cityscapes_CUT_AtoB_checkpoint.tar.gz gdrive:backups/
```

### Step 6.3: Download Results

**Compress generated images:**

```bash
# Compress A→B results
tar -czvf results_AtoB.tar.gz results/cityscapes_CUT_AtoB/

# Compress B→A results
tar -czvf results_BtoA.tar.gz results/cityscapes_CUT_BtoA/
```

**Transfer via SCP:**

```bash
# From local machine
scp -P <PORT> root@<IP>:/workspace/contrastive-unpaired-translation/results_*.tar.gz ./
```

### Step 6.4: Clean Up (Free Storage)

```bash
# Remove checkpoint after backing up (saves ~5-10 GB each)
rm -rf checkpoints/cityscapes_CUT_AtoB/
rm -rf checkpoints/cityscapes_CUT_BtoA/

# Remove intermediate files
rm -rf results/*/train_latest/images/real_*
rm -rf results/*/train_latest/images/idt_*
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--crop_size` (try 512 or 256)
- Use `--CUT_mode FastCUT` (uses less memory)

```bash
# Memory-efficient configuration
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_AtoB \
    --CUT_mode FastCUT \
    --crop_size 512 \
    --batch_size 2 \
    --gpu_ids 0 \
    --display_id -1
```

#### 2. gdown Download Fails

```
Access denied or quota exceeded
```

**Solutions:**
```bash
# Use gdown with cookies
gdown --id <FILE_ID> --cookies cookies.txt

# Alternative: Manual download + SCP upload
# Download file locally, then:
scp -P <PORT> local_file.zip root@<IP>:/workspace/
```

#### 3. Python Version Mismatch

```
ModuleNotFoundError or syntax errors
```

**Solution:** Ensure virtual environment is activated:
```bash
source venv/bin/activate
which python  # Should show /workspace/.../venv/bin/python
```

#### 4. Training Interrupted

**Resume from checkpoint:**
```bash
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_CUT_AtoB \
    --CUT_mode CUT \
    --continue_train \
    --epoch_count <last_epoch + 1> \
    ...
```

### Performance Optimization

| Scenario | Recommendation |
|----------|----------------|
| Limited VRAM | Use `FastCUT`, smaller `crop_size` |
| Faster training | Use `FastCUT`, `--netG resnet_6blocks` |
| Better quality | Use `CUT`, larger `crop_size`, more epochs |
| Large dataset | Use `--max_dataset_size` to limit |

---

## References

### Citation

If you use CUT in your research, please cite:

```bibtex
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

### Resources

- **CUT Paper:** [arXiv:2007.15651](https://arxiv.org/pdf/2007.15651)
- **Official Repository:** [GitHub](https://github.com/taesungp/contrastive-unpaired-translation)
- **Project Website:** [taesung.me/ContrastiveUnpairedTranslation](http://taesung.me/ContrastiveUnpairedTranslation/)
- **Vast.ai Documentation:** [docs.vast.ai](https://docs.vast.ai/)

---

## Quick Reference Commands

```bash
# === SETUP ===
sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# === DATASET ===
mkdir -p datasets/cityscapes
gdown 1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b && unzip clear_image.zip -d datasets/cityscapes && rm clear_image.zip
gdown 1Y8GA79Hng9MD6llcSi3ys4NgWouxTnZb && unzip foggy_image.zip -d datasets/cityscapes && rm foggy_image.zip

# === TRAINING ===
# A→B (Clear to Foggy)
python train.py --dataroot ./datasets/cityscapes --name cityscapes_CUT_AtoB --CUT_mode CUT --preprocess scale_shortside_and_crop --load_size 640 --crop_size 640 --batch_size 4 --gpu_ids 0 --display_id -1

# B→A (Foggy to Clear)
python train.py --dataroot ./datasets/cityscapes --name cityscapes_CUT_BtoA --CUT_mode CUT --preprocess scale_shortside_and_crop --load_size 640 --crop_size 640 --batch_size 4 --gpu_ids 0 --display_id -1 --direction BtoA

# === INFERENCE ===
python test.py --dataroot ./datasets/cityscapes --name cityscapes_CUT_AtoB --CUT_mode CUT --preprocess none --phase train --gpu_ids 0
python test.py --dataroot ./datasets/cityscapes --name cityscapes_CUT_BtoA --CUT_mode CUT --preprocess none --phase train --gpu_ids 0

# === BACKUP ===
tar -czvf checkpoints_backup.tar.gz checkpoints/
tar -czvf results_backup.tar.gz results/
```

---

*Last updated: January 2026*
