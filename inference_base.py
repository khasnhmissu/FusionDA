"""
FusionDA Inference Script
=========================
Chạy inference trên folder ảnh, xuất file .txt prediction (YOLO format).

Hỗ trợ YOLO26s và YOLOv8 checkpoints.

Output format mỗi dòng:
    class_id  confidence  x_center  y_center  width  height
    (tọa độ chuẩn hóa [0, 1])

Tương thích với map_evaluation.py để đánh giá mAP.

Usage:
    python inference.py \
        --weights yolov8n.pt \
        --checkpoint runs/fda/exp/weights/best.pt \
        --source path/to/images \
        --output predicts \
        --conf-thres 0.25 \
        --iou-thres 0.45
"""

import argparse
import os
import time
import yaml
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


# ──────────────────────────────────────────────
# 1. LETTERBOX PREPROCESSING
# ──────────────────────────────────────────────

def letterbox(img, new_shape=640, color=(114, 114, 114), auto=False, stride=32):
    """
    Resize + pad ảnh về kích thước vuông (letterbox) giữ tỉ lệ.

    Returns:
        img_padded: ảnh đã letterbox (H, W, 3)
        ratio:      tỉ lệ scale (r, r)
        (dw, dh):   padding (left, top)
    """
    shape = img.shape[:2]  # (H, W)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Tỉ lệ scale (chọn min để ảnh vừa khung)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)

    # Padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride

    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


# ──────────────────────────────────────────────
# 2. UNDO LETTERBOX — CHUYỂN TỌA ĐỘ VỀ ẢNH GỐC
# ──────────────────────────────────────────────

def scale_boxes_to_original(boxes_xyxy, img_shape_orig, ratio, pad):
    """
    Chuyển tọa độ bounding box (xyxy pixel trên ảnh letterbox)
    → tọa độ pixel trên ảnh gốc.

    Args:
        boxes_xyxy:       tensor [N, 4] — (x1, y1, x2, y2) trên ảnh letterbox
        img_shape_orig:   (H_orig, W_orig) ảnh gốc
        ratio:            (r, r) — tỉ lệ scale đã dùng khi letterbox
        pad:              (dw, dh) — padding đã thêm

    Returns:
        boxes_xyxy trên ảnh gốc (tensor [N, 4])
    """
    dw, dh = pad
    r = ratio[0]

    # Trừ padding
    boxes_xyxy[:, 0] -= dw
    boxes_xyxy[:, 1] -= dh
    boxes_xyxy[:, 2] -= dw
    boxes_xyxy[:, 3] -= dh

    # Chia cho scale ratio
    boxes_xyxy[:, :4] /= r

    # Clamp
    h_orig, w_orig = img_shape_orig
    boxes_xyxy[:, 0].clamp_(0, w_orig)
    boxes_xyxy[:, 1].clamp_(0, h_orig)
    boxes_xyxy[:, 2].clamp_(0, w_orig)
    boxes_xyxy[:, 3].clamp_(0, h_orig)

    return boxes_xyxy


def xyxy_to_xywhn(boxes_xyxy, img_w, img_h):
    """
    Convert xyxy (pixel) → xywh normalized [0, 1].

    Args:
        boxes_xyxy: tensor [N, 4]
        img_w, img_h: kích thước ảnh gốc

    Returns:
        tensor [N, 4] — (x_center, y_center, width, height) normalized
    """
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]

    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return torch.stack([xc, yc, w, h], dim=1)


# ──────────────────────────────────────────────
# 3. LOAD MODEL
# ──────────────────────────────────────────────

def load_model(weights, checkpoint, device, half=False):
    """
    Load YOLO26/YOLOv8 architecture + FusionDA checkpoint.

    Args:
        weights:    đường dẫn đến base weights (e.g. 'yolo26s.pt')
        checkpoint: đường dẫn đến FusionDA checkpoint (.pt)
        device:     torch.device
        half:       dùng FP16

    Returns:
        model (nn.Module)
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        epoch = ckpt.get('epoch', '?')
        best_fitness = ckpt.get('best_fitness', '?')
        print(f"📦 Loaded checkpoint: epoch={epoch}, best_mAP@50={best_fitness}")
        ckpt_model = ckpt['model']
    else:
        print(f"📦 Loaded raw checkpoint")
        ckpt_model = ckpt

    # If checkpoint is a full model object (Ultralytics default), use it directly.
    if isinstance(ckpt_model, torch.nn.Module):
        model = ckpt_model.to(device)
    else:
        # Plain state_dict — build architecture from base weights then load
        yolo = YOLO(weights)
        model = yolo.model.to(device)
        model.load_state_dict(ckpt_model, strict=True)

    model.eval()

    # Always explicitly cast to the target dtype so that the model dtype
    # matches the input tensor dtype, regardless of how the checkpoint was saved.
    if half:
        model.half()
        print("⚡ FP16 inference enabled")
    else:
        model.float()  # Ensure FP32 even if checkpoint was saved in FP16

    n_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Model params: {n_params:,}")

    return model


# ──────────────────────────────────────────────
# 4. PARSE RAW OUTPUT
# ──────────────────────────────────────────────

def parse_model_output(pred, device):
    """
    Parse raw model output → tensor phù hợp cho NMS.

    YOLO26 output (eval mode):  tuple (tensor_decoded, None) hoặc tensor [B, N, 6]
    YOLOv8 output (train mode): dict {'one2one': ..., 'one2many': ...}
                                 hoặc tuple/list
    YOLOv8 output (eval mode):  tensor [batch, 4+nc, num_preds]

    NMS input cần: [batch, 4+nc, num_preds]

    Returns:
        pred_tensor: [batch, 4+nc, num_preds]
    """
    if isinstance(pred, dict):
        # Training mode output (dict)
        # NMS cần output của nhánh dense features, nên ta lấy one2many
        pred_tensor = pred.get('one2many', pred.get('one2one', None))
        if pred_tensor is not None:
            pred_tensor = pred_tensor[0] if isinstance(pred_tensor, (list, tuple)) else pred_tensor
        else:
            raise ValueError("Cannot parse model dict output: missing 'one2one' / 'one2many'")
    elif isinstance(pred, (list, tuple)):
        pred_tensor = pred[0]
    else:
        pred_tensor = pred

    if len(pred_tensor.shape) == 3 and pred_tensor.shape[-1] == 6:
        return pred_tensor

    # Chuẩn bị Tensor cho NMS Ultralytics, yêu cầu format là [B, 4+nc, N]
    # Ví dụ: YOLO26/v10 có thể trả về dạng [B, N, 4+nc] -> [1, 8400, 84] -> cần transpose (N > 4+nc <=> shape[1] > shape[-1])
    if len(pred_tensor.shape) == 3:
        if pred_tensor.shape[1] > pred_tensor.shape[-1]:
            # [B, N, 4+nc] format → transpose sang [B, 4+nc, N]
            pred_tensor = pred_tensor.permute(0, 2, 1)

    # Apply sigmoid nếu class scores chưa qua activation
    if len(pred_tensor.shape) == 3:
        cls_scores = pred_tensor[:, 4:, :]
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            pred_tensor = torch.cat([
                pred_tensor[:, :4, :],
                torch.sigmoid(cls_scores)
            ], dim=1)

    return pred_tensor


# ──────────────────────────────────────────────
# 5. MAIN INFERENCE
# ──────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def get_image_files(source_dir):
    """Lấy danh sách ảnh từ folder."""
    source = Path(source_dir)
    files = []
    for f in sorted(source.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return files


@torch.no_grad()
def run_inference(opt):
    """
    Pipeline inference đầy đủ:
    1. Load model
    2. Duyệt từng ảnh → preprocess → forward → NMS → undo letterbox → normalize
    3. Ghi file .txt
    """
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    half = opt.half and device.type != 'cpu'

    # Load class names
    class_names = {0: 'person', 1: 'car'}  # Default
    if opt.class_names and os.path.exists(opt.class_names):
        with open(opt.class_names, encoding='utf-8') as f:
            data_dict = yaml.safe_load(f)
        if 'names' in data_dict:
            if isinstance(data_dict['names'], dict):
                class_names = data_dict['names']
            elif isinstance(data_dict['names'], list):
                class_names = {i: n for i, n in enumerate(data_dict['names'])}
        nc = data_dict.get('nc', len(class_names))
    else:
        nc = len(class_names)

    print(f"\n{'='*60}")
    print(f"{'FusionDA Inference':^60}")
    print(f"{'='*60}")
    print(f"  Weights:     {opt.weights}")
    print(f"  Checkpoint:  {opt.checkpoint}")
    print(f"  Source:      {opt.source}")
    print(f"  Output:      {opt.output}")
    print(f"  Image size:  {opt.imgsz}")
    print(f"  Conf thres:  {opt.conf_thres}")
    print(f"  IoU thres:   {opt.iou_thres}")
    print(f"  Device:      {device}")
    print(f"  Classes:     {class_names}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(opt.weights, opt.checkpoint, device, half=half)

    # Get stride
    gs = max(int(model.stride.max()), 32)

    # Tạo output folder
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh
    image_files = get_image_files(opt.source)
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong {opt.source}")
        return

    print(f"📸 Tìm thấy {len(image_files)} ảnh\n")

    # Statistics
    total_detections = 0
    total_images_with_det = 0
    class_counts = {}

    t_start = time.time()

    for img_path in tqdm(image_files, desc="Inference"):
        # 1. Đọc ảnh
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️  Không đọc được: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # 2. Letterbox preprocessing
        img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=opt.imgsz, stride=gs)

        # 3. Chuyển sang tensor
        img_tensor = img_lb.transpose(2, 0, 1)  # HWC → CHW
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(device).float()
        img_tensor /= 255.0  # Normalize [0, 255] → [0, 1]

        if half:
            img_tensor = img_tensor.half()

        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

        # 4. Forward
        pred_raw = model(img_tensor)

        # 5. Parse output
        pred_tensor = parse_model_output(pred_raw, device)

        # 6. NMS
        detections = non_max_suppression(
            pred_tensor,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            max_det=opt.max_det,
        )

        # 7. Process detections cho ảnh này
        det = detections[0]  # Batch size = 1
        txt_name = img_path.stem + '.txt'
        txt_path = output_dir / txt_name

        if det is not None and len(det) > 0:
            # det format: [x1, y1, x2, y2, confidence, class_id]

            # Undo letterbox → tọa độ pixel ảnh gốc
            det[:, :4] = scale_boxes_to_original(
                det[:, :4].clone(), (h_orig, w_orig), ratio, (dw, dh)
            )

            # Chuyển xyxy → xywh normalized
            boxes_norm = xyxy_to_xywhn(det[:, :4], w_orig, h_orig)

            # Clamp normalized values
            boxes_norm = boxes_norm.clamp(0.0, 1.0)

            # Ghi file
            with open(txt_path, 'w') as f:
                for j in range(len(det)):
                    cls_id = int(det[j, 5].item())
                    conf = det[j, 4].item()
                    xc, yc, w, h = boxes_norm[j].cpu().numpy()

                    # Format: class_id confidence x_center y_center width height
                    f.write(f"{cls_id} {conf:.6f} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                    # Thống kê
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            total_detections += len(det)
            total_images_with_det += 1
        else:
            # Ghi file rỗng (không có detection)
            with open(txt_path, 'w') as f:
                pass

    elapsed = time.time() - t_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'INFERENCE COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Total images:       {len(image_files)}")
    print(f"  Images with det:    {total_images_with_det}")
    print(f"  Total detections:   {total_detections}")
    print(f"  Time:               {elapsed:.2f}s ({len(image_files)/max(elapsed,1e-6):.1f} img/s)")
    print(f"  Output folder:      {output_dir}")
    print()
    print(f"  Per-class counts:")
    for cls_id in sorted(class_counts.keys()):
        name = class_names.get(cls_id, f'class_{cls_id}')
        print(f"    [{cls_id}] {name}: {class_counts[cls_id]}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 6. ARGUMENT PARSER
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='FusionDA Inference — Chạy detection trên folder ảnh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python inference.py \\
      --weights yolov8n.pt \\
      --checkpoint runs/fda/exp/weights/best.pt \\
      --source datasets/target_real/target_real/val/images \\
      --output predicts \\
      --conf-thres 0.25

Output format (mỗi file .txt):
  class_id  confidence  x_center  y_center  width  height
  (tọa độ normalized [0, 1])
        """
    )

    # Model
    parser.add_argument('--weights', type=str, required=True,
                        help='YOLOv8 base weights (kiến trúc), e.g. yolov8n.pt')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='FusionDA checkpoint (.pt) chứa state_dict')

    # Data
    parser.add_argument('--source', type=str, required=True,
                        help='Folder chứa ảnh đầu vào')
    parser.add_argument('--output', type=str, default='predicts',
                        help='Folder output cho file .txt prediction (default: predicts)')

    # Inference params
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Kích thước ảnh inference (default: 640)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Số detection tối đa mỗi ảnh (default: 300)')

    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (default: 0)')
    parser.add_argument('--half', action='store_true',
                        help='Dùng FP16 inference')

    # Class names
    parser.add_argument('--class-names', type=str, default='data.yaml',
                        help='YAML file chứa class names (default: data.yaml)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)