#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# FusionDA — WIDERFACE 4-domain full environment & dataset setup
# Run from project root:  bash scripts/setup_wider.sh
#
# Layout produced (datasets/<domain>/<domain>/{train,val}/{images,labels}):
#   datasets/wider/wider/...               (source_real, clear)
#   datasets/wider_dehazed/...             (source_fake, GAN-dehazed clear)
#   datasets/wider_foggy/wider_foggy/...   (target_real, foggy)
#   datasets/wider_fake_fog/...            (target_fake, fog applied)
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

# 4. Datasets — pulls all 4 paired WIDERFACE domains from Google Drive
mkdir -p datasets
cd datasets

download_and_extract() {
  local NAME="$1"
  local FILE_ID="$2"
  if [ -d "$NAME/$NAME" ]; then
    echo "[skip] $NAME already extracted"
    return
  fi
  echo "[get ] $NAME (id=$FILE_ID)"
  gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$NAME.zip"
  mkdir -p "$NAME"
  unzip -q "$NAME.zip" -d "$NAME"
  rm "$NAME.zip"
}

download_and_extract "wider"          "1sb0Lt6bb5RdgsKsCQpY8RnBPmhqJ3FFO"
download_and_extract "wider_dehazed"  "1TRkpyZAJKP_SfLEdw2iD2fndpaadN40E"
download_and_extract "wider_foggy"    "15g_xesQmq6Ll-eUoezlg_F-bMCKoHRdP"
download_and_extract "wider_fake_fog" "1PU_HLAuTco02AkMzh-yz6KkYP3k2sQRp"

cd ..

# 5. Verify dataset structure
echo ""
echo "=== Verifying dataset structure ==="
for d in wider wider_dehazed wider_foggy wider_fake_fog; do
  for split in train val; do
    p="datasets/$d/$d/$split/images"
    if [ -d "$p" ]; then
      n=$(ls "$p" | wc -l)
      echo "  OK  $p  ($n files)"
    else
      echo "  MISS $p"
    fi
  done
done

# 6. Pretrained YOLO26-s weights
echo ""
if [ -f "yolo26s.pt" ]; then
  echo "[skip] yolo26s.pt already present"
else
  echo "Downloading YOLO26s weights ..."
  wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26s.pt
fi

echo ""
echo "=== Setup complete ==="
echo "Run an experiment with:  bash scripts/wider_01_baseline.sh"
echo "Or run the full ablation suite:  bash scripts/wider_run_all.sh"
