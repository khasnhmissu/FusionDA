#!/bin/bash
# FusionDA Setup Script - GPU Server
# ===================================

# 1. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt
pip install gdown

# 4. Create datasets directory
mkdir -p datasets
cd datasets

# 5. Download & Extract datasets
# ZIP structure: xxx.zip contains xxx/ folder with images/ and labels/

echo "Downloading source_real..."
gdown 1p_pJjOIngZ6ylbPSs2fD3Eo7QR8q5C2b -O source_real.zip
mkdir -p source_real && unzip -q source_real.zip -d source_real && rm source_real.zip

echo "Downloading source_fake..."
gdown 1Mh2dm0lzBk8gi7R8zTj0e2V9tXus8Tmb -O source_fake.zip
mkdir -p source_fake && unzip -q source_fake.zip -d source_fake && rm source_fake.zip

echo "Downloading target_real..."
gdown 1Y8GA79Hng9MD6llcSi3ys4NgWouxTnZb -O target_real.zip
mkdir -p target_real && unzip -q target_real.zip -d target_real && rm target_real.zip

echo "Downloading target_fake..."
gdown 15xPIUiQ7Io2W8xrPBQYcwPoFiwsNKddW -O target_fake.zip
mkdir -p target_fake && unzip -q target_fake.zip -d target_fake && rm target_fake.zip

cd ..

# 6. Verify structure
echo ""
echo "=== Verifying dataset structure ==="
echo "Expected: datasets/xxx/xxx/images/train/"
ls -d datasets/*/*/images/train 2>/dev/null || echo "Warning: Check folder structure!"

# 7. Download weights
echo ""
echo "Downloading YOLOv8l weights..."
wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt

echo ""
echo "=== Setup Complete ==="
echo "Run training:"
echo "python train.py --data data.yaml --weights yolov8l.pt --epochs 200 --batch 16 --use-grl --enable-monitoring"
