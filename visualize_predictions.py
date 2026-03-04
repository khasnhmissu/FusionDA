"""
Visualize Predictions — Vẽ bounding box lên ảnh gốc
=====================================================
Đọc file .txt prediction (output từ inference.py) và vẽ lên ảnh gốc
để kiểm tra nhãn dự đoán bằng mắt.

Input format (.txt):
    class_id  confidence  x_center  y_center  width  height
    (tọa độ normalized [0, 1])

Usage:
    python visualize_predictions.py \
        --images path/to/images \
        --predicts predicts \
        --output visualized \
        --conf-thres 0.25
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 1. CLASS COLORS & CONFIG
# ──────────────────────────────────────────────

# Bảng màu đẹp, dễ phân biệt (BGR format cho OpenCV)
COLORS = [
    (0, 255, 0),      # Green  — person
    (255, 128, 0),    # Orange — car
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 128, 255),    # Light orange
    (128, 255, 0),    # Light green
    (255, 255, 0),    # Cyan
]

DEFAULT_CLASS_NAMES = {0: 'person', 1: 'car'}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ──────────────────────────────────────────────
# 2. LOAD PREDICTIONS
# ──────────────────────────────────────────────

def load_prediction(txt_path):
    """
    Đọc file .txt prediction.

    Returns:
        list of (class_id, confidence, x_c, y_c, w, h) — normalized [0,1]
    """
    predictions = []
    if not os.path.exists(txt_path):
        return predictions

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                continue

            cls_id = int(parts[0])
            conf = float(parts[1])
            xc, yc, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            predictions.append((cls_id, conf, xc, yc, w, h))

    return predictions


# ──────────────────────────────────────────────
# 3. DRAW BOXES
# ──────────────────────────────────────────────

def draw_boxes(img, predictions, class_names, conf_thres=0.0, line_thickness=2):
    """
    Vẽ bounding box lên ảnh.

    Args:
        img:          ảnh BGR (H, W, 3)
        predictions:  list of (class_id, confidence, x_c, y_c, w, h)
        class_names:  dict {id: name}
        conf_thres:   ngưỡng confidence tối thiểu để vẽ
        line_thickness: độ dày viền box

    Returns:
        img_draw: ảnh với bounding box
        n_drawn:  số box đã vẽ
    """
    img_draw = img.copy()
    h_img, w_img = img_draw.shape[:2]
    n_drawn = 0

    # Sort theo confidence giảm dần (vẽ box conf thấp trước, cao sau → nổi bật hơn)
    predictions_sorted = sorted(predictions, key=lambda x: x[1])

    for cls_id, conf, xc, yc, w, h in predictions_sorted:
        if conf < conf_thres:
            continue

        # Chuyển normalized → pixel
        x1 = int((xc - w / 2) * w_img)
        y1 = int((yc - h / 2) * h_img)
        x2 = int((xc + w / 2) * w_img)
        y2 = int((yc + h / 2) * h_img)

        # Clamp
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        # Màu theo class
        color = COLORS[cls_id % len(COLORS)]
        cls_name = class_names.get(cls_id, f'cls_{cls_id}')

        # Vẽ box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, line_thickness)

        # Label text
        label = f'{cls_name} {conf:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness_text = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_text)

        # Background cho label (phía trên box)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        cv2.rectangle(img_draw, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)
        cv2.putText(img_draw, label, (x1 + 2, label_y2 - baseline - 2),
                    font, font_scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

        n_drawn += 1

    # Info text góc trên trái
    info = f'Detections: {n_drawn} | conf >= {conf_thres:.2f}'
    cv2.putText(img_draw, info, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return img_draw, n_drawn


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

def run_visualization(opt):
    """Duyệt ảnh + prediction → vẽ box → lưu."""

    images_dir = Path(opt.images)
    predicts_dir = Path(opt.predicts)
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class names
    class_names = DEFAULT_CLASS_NAMES.copy()
    if opt.class_names:
        import yaml
        if os.path.exists(opt.class_names):
            with open(opt.class_names, encoding='utf-8') as f:
                data_dict = yaml.safe_load(f)
            if 'names' in data_dict:
                if isinstance(data_dict['names'], dict):
                    class_names = data_dict['names']
                elif isinstance(data_dict['names'], list):
                    class_names = {i: n for i, n in enumerate(data_dict['names'])}

    # Lấy danh sách ảnh
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong {images_dir}")
        return

    print(f"\n{'='*60}")
    print(f"{'VISUALIZE PREDICTIONS':^60}")
    print(f"{'='*60}")
    print(f"  Images:      {images_dir} ({len(image_files)} ảnh)")
    print(f"  Predicts:    {predicts_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Conf thres:  {opt.conf_thres}")
    print(f"  Classes:     {class_names}")
    print(f"{'='*60}\n")

    total_boxes = 0
    n_with_det = 0
    n_no_pred = 0

    for img_path in image_files:
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Không đọc được: {img_path}")
            continue

        # Tìm file prediction tương ứng
        txt_path = predicts_dir / (img_path.stem + '.txt')
        predictions = load_prediction(str(txt_path))

        if not txt_path.exists():
            n_no_pred += 1

        # Vẽ boxes
        img_draw, n_drawn = draw_boxes(
            img, predictions, class_names,
            conf_thres=opt.conf_thres,
            line_thickness=opt.line_thickness
        )

        total_boxes += n_drawn
        if n_drawn > 0:
            n_with_det += 1

        # Lưu
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), img_draw)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'VISUALIZATION COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Total images:         {len(image_files)}")
    print(f"  Images with det:      {n_with_det}")
    print(f"  Images no pred file:  {n_no_pred}")
    print(f"  Total boxes drawn:    {total_boxes}")
    print(f"  Output folder:        {output_dir}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Predictions — Vẽ bounding box lên ảnh gốc để kiểm tra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python visualize_predictions.py \\
      --images datasets/target_real/target_real/val/images \\
      --predicts predicts \\
      --output visualized \\
      --conf-thres 0.25
        """
    )

    parser.add_argument('--images', type=str, required=True,
                        help='Folder chứa ảnh gốc')
    parser.add_argument('--predicts', type=str, required=True,
                        help='Folder chứa file .txt prediction (output từ inference.py)')
    parser.add_argument('--output', type=str, default='visualized',
                        help='Folder output cho ảnh đã vẽ box (default: visualized)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold tối thiểu để vẽ (default: 0.25)')
    parser.add_argument('--class-names', type=str, default='data.yaml',
                        help='YAML file chứa class names (default: data.yaml)')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Độ dày viền box (default: 2)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_visualization(args)
