#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# FusionDA — GPU server environment setup
# Run from the project root:  bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────
set -e

# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. PyTorch (CUDA 11.8 — adjust the index-url if your CUDA differs)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Project dependencies
pip install -r requirements.txt
pip install gdown

# 4. Datasets — pulls all 4 paired domains from Google Drive
mkdir -p datasets
cd datasets

echo "Downloading source_real ..."
gdown 1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b -O source_real.zip
mkdir -p source_real && unzip -q source_real.zip -d source_real && rm source_real.zip

echo "Downloading source_fake ..."
gdown 1Mh2dm0lzBk8gi7R8zTj0e2V9tXus8Tmb -O source_fake.zip
mkdir -p source_fake && unzip -q source_fake.zip -d source_fake && rm source_fake.zip

echo "Downloading target_real ..."
gdown 1Y8GA79Hng9MD6llcSi3ys4NgWouxTnZb -O target_real.zip
mkdir -p target_real && unzip -q target_real.zip -d target_real && rm target_real.zip

echo "Downloading target_fake ..."
gdown 15xPIUiQ7Io2W8xrPBQYcwPoFiwsNKddW -O target_fake.zip
mkdir -p target_fake && unzip -q target_fake.zip -d target_fake && rm target_fake.zip
cd ..

# 5. Verify dataset layout (datasets/<domain>/<domain>/{train,val}/{images,labels})
echo ""
echo "=== Verifying dataset structure ==="
ls -d datasets/*/*/train/images 2>/dev/null || echo "WARN: dataset layout unexpected"

# 6. Pretrained YOLO26-s weights
echo ""
echo "Downloading YOLO26s weights ..."
wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26s.pt

echo ""
echo "=== Setup complete ==="
echo "Reproduce the latest ablation suite with:"
echo "  bash scripts/run_all_ablations.sh"
