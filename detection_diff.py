"""
Detection Diff Visualization — Visual Evidence
===============================================
Lấy ảnh foggy khó, chạy 4 models, vẽ boxes cùng GT.
Màu: Green = TP, Red = FP, Yellow dashed = FN (missed GT).

Usage:
    python detection_diff.py \
        --weights yolo26s.pt \
        --checkpoints \
            results/full-yolov8/weights/best.pt \
            results/nogrl-yolov8/weights/best.pt \
            results/freeze_teacher-yolov8/weights/best.pt \
            results/freeze_teacher-nogrl-yolov8/weights/best.pt \
        --names "Full (GRL)" "NoGRL" "Freeze Full" "Freeze NoGRL" \
        --images target_test/target_test/val/images \
        --labels target_test/target_test/val/labels \
        --output results_explainable/detection_diff \
        --n-images 8 \
        --device 0
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Colors (BGR for OpenCV)
COLOR_TP = (0, 200, 0)       # Green — True Positive
COLOR_FP = (0, 0, 220)       # Red — False Positive
COLOR_FN = (0, 200, 255)     # Yellow — False Negative (missed)
COLOR_GT = (255, 180, 0)     # Cyan — Ground Truth


def letterbox(img, new_shape=640, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, r), (dw, dh)


def scale_boxes_to_original(boxes_xyxy, img_shape_orig, ratio, pad):
    dw, dh = pad
    r = ratio[0]
    boxes_xyxy[:, 0] -= dw
    boxes_xyxy[:, 1] -= dh
    boxes_xyxy[:, 2] -= dw
    boxes_xyxy[:, 3] -= dh
    boxes_xyxy[:, :4] /= r
    h_orig, w_orig = img_shape_orig
    boxes_xyxy[:, 0].clamp_(0, w_orig)
    boxes_xyxy[:, 1].clamp_(0, h_orig)
    boxes_xyxy[:, 2].clamp_(0, w_orig)
    boxes_xyxy[:, 3].clamp_(0, h_orig)
    return boxes_xyxy


def parse_model_output(pred):
    if isinstance(pred, dict):
        pred_tensor = pred.get('one2one', pred.get('one2many', None))
        if pred_tensor is not None:
            pred_tensor = pred_tensor[0] if isinstance(pred_tensor, (list, tuple)) else pred_tensor
        else:
            raise ValueError("Cannot parse model output")
    elif isinstance(pred, (list, tuple)):
        pred_tensor = pred[0]
    else:
        pred_tensor = pred

    if len(pred_tensor.shape) == 3 and pred_tensor.shape[-1] == 6:
        return pred_tensor

    # YOLO26 eval: có thể trả về [B, N, 4+nc] thay vì [B, 4+nc, N]
    if len(pred_tensor.shape) == 3:
        if pred_tensor.shape[-1] > pred_tensor.shape[1]:
            pred_tensor = pred_tensor.permute(0, 2, 1)
    if len(pred_tensor.shape) == 3:
        cls_scores = pred_tensor[:, 4:, :]
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            pred_tensor = torch.cat([
                pred_tensor[:, :4, :],
                torch.sigmoid(cls_scores)
            ], dim=1)
    return pred_tensor


def load_gt_labels(label_path, img_w, img_h):
    """Load YOLO GT → [cls, x1, y1, x2, y2] pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return np.array([]).reshape(0, 5)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([cls_id, x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.array([]).reshape(0, 5)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def classify_detections(preds_np, gt_boxes, iou_threshold=0.5):
    """
    Classify each prediction as TP or FP, and find FN GT boxes.
    
    Returns:
        tp_boxes: list of [x1,y1,x2,y2,conf,cls]
        fp_boxes: list of [x1,y1,x2,y2,conf,cls]
        fn_boxes: list of [cls,x1,y1,x2,y2] (unmatched GT)
    """
    tp_boxes = []
    fp_boxes = []
    matched_gt = set()
    
    if len(preds_np) > 0:
        sorted_idx = np.argsort(-preds_np[:, 4])
        for idx in sorted_idx:
            pred_box = preds_np[idx, :4]
            pred_cls = int(preds_np[idx, 5])
            
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                gt_cls = int(gt[0])
                if pred_cls != gt_cls:
                    continue
                iou = compute_iou(pred_box, gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_boxes.append(preds_np[idx])
                matched_gt.add(best_gt_idx)
            else:
                fp_boxes.append(preds_np[idx])
    
    fn_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return tp_boxes, fp_boxes, fn_boxes


CLASS_NAMES = {0: 'person', 1: 'car'}


def draw_detection_diff(img_bgr, tp_boxes, fp_boxes, fn_boxes, model_name):
    """Draw TP (green), FP (red), FN (yellow dashed) on image."""
    img = img_bgr.copy()
    
    # Draw FN first (dashed yellow — behind other boxes)
    for gt in fn_boxes:
        cls_id = int(gt[0])
        x1, y1, x2, y2 = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
        # Dashed rectangle
        for start in range(x1, x2, 10):
            end = min(start + 5, x2)
            cv2.line(img, (start, y1), (end, y1), COLOR_FN, 2)
            cv2.line(img, (start, y2), (end, y2), COLOR_FN, 2)
        for start in range(y1, y2, 10):
            end = min(start + 5, y2)
            cv2.line(img, (x1, start), (x1, end), COLOR_FN, 2)
            cv2.line(img, (x2, start), (x2, end), COLOR_FN, 2)
        cls_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, f'FN:{cls_name}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FN, 1)
    
    # Draw FP (red)
    for pred in fp_boxes:
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        conf = pred[4]
        cls_id = int(pred[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FP, 2)
        cls_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, f'FP:{cls_name} {conf:.2f}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FP, 1)
    
    # Draw TP (green)
    for pred in tp_boxes:
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        conf = pred[4]
        cls_id = int(pred[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TP, 2)
        cls_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, f'TP:{cls_name} {conf:.2f}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TP, 1)
    
    # Title bar
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    text = f'{model_name}  TP:{len(tp_boxes)} FP:{len(fp_boxes)} FN:{len(fn_boxes)}'
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    return img


def draw_gt_only(img_bgr):
    """Draw GT boxes only."""
    img = img_bgr.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, 'Ground Truth', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def draw_gt_boxes_on(img_bgr, gt_boxes):
    """Draw GT boxes on image in cyan."""
    img = img_bgr.copy()
    for gt in gt_boxes:
        cls_id = int(gt[0])
        x1, y1, x2, y2 = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GT, 2)
        cls_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, cls_name, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, f'Ground Truth ({len(gt_boxes)} objects)', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def load_model(weights_arch, checkpoint_path, device):
    yolo = YOLO(weights_arch)
    model = yolo.model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_inference_single(model, img_path, device, imgsz=640, conf_thres=0.25,
                         iou_thres=0.45, class_mapping=None):
    """Run inference on single image, return detections [x1,y1,x2,y2,conf,cls] in original coords."""
    gs = max(int(model.stride.max()), 32)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    
    img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, stride=gs)
    img_tensor = img_lb.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    pred_raw = model(img_tensor)
    pred_tensor = parse_model_output(pred_raw)
    
    # NMS / E2E filter
    # YOLO26 E2E format [B, N, 6]: đã decoded, chỉ cần filter by confidence
    if len(pred_tensor.shape) == 3 and pred_tensor.shape[-1] == 6:
        detections = []
        for bi in range(pred_tensor.shape[0]):
            mask = pred_tensor[bi, :, 4] > conf_thres
            detections.append(pred_tensor[bi, mask])
    else:
        detections = non_max_suppression(
            pred_tensor, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300,
        )
    
    det = detections[0]
    if det is not None and len(det) > 0:
        det[:, :4] = scale_boxes_to_original(
            det[:, :4].clone(), (h_orig, w_orig), ratio, (dw, dh)
        )
        if class_mapping:
            valid_mask = torch.zeros(len(det), dtype=torch.bool, device=device)
            for coco_cls, dataset_cls in class_mapping.items():
                mask = (det[:, 5].long() == coco_cls)
                det[mask, 5] = dataset_cls
                valid_mask |= mask
            det = det[valid_mask]
        return det.cpu().numpy()
    return np.array([]).reshape(0, 6)


def select_images(images_dir, n_images=8, seed=42):
    image_files = sorted([
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if len(image_files) <= n_images:
        return image_files
    np.random.seed(seed)
    indices = np.linspace(0, len(image_files) - 1, n_images, dtype=int)
    return [image_files[i] for i in indices]


def create_comparison_figure(image_files, all_results, model_names, labels_dir, output_dir):
    """Create matplotlib comparison figure."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    n_images = len(image_files)
    n_cols = len(model_names) + 1  # +1 for GT column
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4.5 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        h_orig, w_orig = img_bgr.shape[:2]
        
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        gt_boxes = load_gt_labels(str(label_path), w_orig, h_orig)
        
        # Column 0: GT only
        gt_img = draw_gt_boxes_on(img_bgr, gt_boxes)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(gt_rgb)
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Columns 1..n: Each model
        for j, name in enumerate(model_names):
            preds_np = all_results[name][i]
            tp, fp, fn = classify_detections(preds_np, gt_boxes, iou_threshold=0.5)
            
            diff_img = draw_detection_diff(img_bgr, tp, fp, fn, name)
            diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
            axes[i, j + 1].imshow(diff_rgb)
            if i == 0:
                axes[i, j + 1].set_title(name, fontsize=11, fontweight='bold')
            axes[i, j + 1].axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='TP (correct)'),
        Patch(facecolor='red', label='FP (false alarm)'),
        Patch(facecolor='yellow', label='FN (missed)'),
        Patch(facecolor='cyan', label='Ground Truth'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Detection Comparison on Target Domain (Foggy)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'detection_diff_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def run(opt):
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_mapping = {0: 0, 2: 1}
    
    all_image_files = sorted([f for f in Path(opt.images).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

    if opt.best_matches:
        print(f"🔍 Searching for most accurate images (high TP, low FN) among {len(all_image_files)} candidates...")
        model0 = load_model(opt.weights, opt.checkpoints[0], device)
        pool = all_image_files[:200]  # Giới hạn 200 ảnh để tìm kiếm nhanh
        scored_images = []
        for img_path in tqdm(pool, desc="  Scoring images"):
            preds = run_inference_single(
                model0, img_path, device, imgsz=opt.imgsz,
                conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
                class_mapping=class_mapping,
            )
            img_bgr = cv2.imread(str(img_path))
            h_orig, w_orig = img_bgr.shape[:2]
            label_path = Path(opt.labels) / (img_path.stem + '.txt')
            gt_boxes = load_gt_labels(str(label_path), w_orig, h_orig)
            
            tp, fp, fn = classify_detections(preds, gt_boxes)
            # Trọng số: Khuyến khích TP, phạt nặng FN (miss) và phạt FP
            score = len(tp) * 2 - len(fn) * 1.5 - len(fp) * 1.0
            
            # Khởi tạo filter nhẹ để lấy ảnh có label thực tế
            if len(gt_boxes) > 0:
                scored_images.append((score, img_path))
                
        scored_images.sort(key=lambda x: x[0], reverse=True)
        image_files = [p for s, p in scored_images[:opt.n_images]]
        del model0
        torch.cuda.empty_cache()
    else:
        image_files = select_images(opt.images, opt.n_images)

    print(f"📸 Selected {len(image_files)} images for comparison")
    
    all_results = {}
    
    for ckpt_path, name in zip(opt.checkpoints, opt.names):
        print(f"\n{'='*50}")
        print(f"Running inference: {name}")
        print(f"{'='*50}")
        
        model = load_model(opt.weights, ckpt_path, device)
        
        results = []
        for img_path in tqdm(image_files, desc=f"  {name}"):
            preds = run_inference_single(
                model, img_path, device, imgsz=opt.imgsz,
                conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
                class_mapping=class_mapping,
            )
            results.append(preds)
        
        all_results[name] = results
        del model
        torch.cuda.empty_cache()
    
    # Create comparison figure
    create_comparison_figure(image_files, all_results, opt.names, opt.labels, output_dir)
    
    # Save individual images
    indiv_dir = output_dir / 'individual'
    indiv_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        h_orig, w_orig = img_bgr.shape[:2]
        label_path = Path(opt.labels) / (img_path.stem + '.txt')
        gt_boxes = load_gt_labels(str(label_path), w_orig, h_orig)
        
        for name in opt.names:
            preds = all_results[name][i]
            tp, fp, fn = classify_detections(preds, gt_boxes)
            diff_img = draw_detection_diff(img_bgr, tp, fp, fn, name)
            save_name = f'{img_path.stem}_{name.replace(" ", "_").lower()}.jpg'
            cv2.imwrite(str(indiv_dir / save_name), diff_img)
    
    print(f"\n{'='*60}")
    print(f"{'DETECTION DIFF COMPLETE':^60}")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='Detection Diff Visualization')
    
    parser.add_argument('--weights', type=str, default='yolo26s.pt')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--images', type=str, required=True, help='Target images dir')
    parser.add_argument('--labels', type=str, required=True, help='Target labels dir')
    parser.add_argument('--output', type=str, default='results_explainable/detection_diff')
    parser.add_argument('--n-images', type=int, default=8)
    parser.add_argument('--best-matches', action='store_true', help='Pre-evaluate and pick most accurate images')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
