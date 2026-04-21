"""
Pseudo-Label Quality over Epochs — RQ1 Direct Evidence
=======================================================
Chạy teacher model tại mỗi checkpoint qua target images,
so sánh predictions với ground truth → vẽ đường quality metrics.

EMA teacher: quality tăng dần qua epochs (teacher adapt)
Frozen teacher: quality phẳng (teacher không cập nhật)

Đồng thời tạo Confidence Distribution histogram.

Usage:
    python pseudo_label_quality.py \
        --checkpoints-dir results/full-yolov8/weights \
        --option-name "Full (GRL)" \
        --weights yolo26s.pt \
        --target-images datasets/target_real/target_real/val/images \
        --target-labels datasets/target_real/target_real/val/labels \
        --output results_explainable/pseudo_label_quality \
        --device 0

    # So sánh tất cả 4 options:
    python pseudo_label_quality.py --compare \
        --compare-dirs \
            results/full-yolov8/weights \
            results/nogrl-yolov8/weights \
            results/freeze_teacher-yolov8/weights \
            results/freeze_teacher-nogrl-yolov8/weights \
        --compare-names "Full (GRL)" "NoGRL" "Freeze Full" "Freeze NoGRL" \
        --weights yolo26s.pt \
        --target-images datasets/target_real/target_real/val/images \
        --target-labels datasets/target_real/target_real/val/labels \
        --output results_explainable/pseudo_label_quality \
        --device 0
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


# ──────────────────────────────────────────────
# Letterbox + helpers (from inference.py)
# ──────────────────────────────────────────────

def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
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


def parse_model_output(pred, device):
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


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_gt_labels(label_path, img_w, img_h):
    """Load YOLO format GT labels → xyxy pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return np.array([]).reshape(0, 5)  # [cls, x1, y1, x2, y2]
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
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_predictions_to_gt(preds, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to GT boxes.
    
    Args:
        preds: np.array [N, 6] — [x1, y1, x2, y2, conf, cls]
        gt_boxes: np.array [M, 5] — [cls, x1, y1, x2, y2]
        
    Returns:
        tp, fp, fn, matched_ious, all_confs
    """
    if len(preds) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, [], []
    if len(preds) == 0:
        return 0, 0, len(gt_boxes), [], []
    if len(gt_boxes) == 0:
        return 0, len(preds), 0, [], preds[:, 4].tolist()
    
    matched_gt = set()
    tp = 0
    fp = 0
    matched_ious = []
    all_confs = preds[:, 4].tolist()
    
    # Sort by confidence descending
    sorted_idx = np.argsort(-preds[:, 4])
    
    for idx in sorted_idx:
        pred_box = preds[idx, :4]
        pred_cls = int(preds[idx, 5])
        
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
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious, all_confs


# ──────────────────────────────────────────────
# Core: Evaluate one checkpoint
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(model, image_files, label_dir, device, imgsz=640,
                        conf_thres=0.5, iou_thres=0.45, max_det=20, class_mapping=None):
    """
    Run teacher/model inference on target images, compare with GT.
    
    Returns:
        dict with precision, recall, mean_iou, f1, all_confidences
    """
    gs = max(int(model.stride.max()), 32)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    all_confidences = []
    
    for img_path in tqdm(image_files, desc="Evaluating", leave=False):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        
        # Load GT
        label_path = Path(label_dir) / (img_path.stem + '.txt')
        gt_boxes = load_gt_labels(str(label_path), w_orig, h_orig)
        
        # Preprocess
        img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, stride=gs)
        img_tensor = img_lb.transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Forward
        pred_raw = model(img_tensor)
        pred_tensor = parse_model_output(pred_raw, device)
        
        # NMS / E2E filter
        # YOLO26 E2E format [B, N, 6]: đã decoded, chỉ cần filter by confidence
        if len(pred_tensor.shape) == 3 and pred_tensor.shape[-1] == 6:
            detections = []
            for bi in range(pred_tensor.shape[0]):
                mask = pred_tensor[bi, :, 4] > conf_thres
                detections.append(pred_tensor[bi, mask])
        else:
            detections = non_max_suppression(
                pred_tensor, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
            )
        
        det = detections[0]
        if det is not None and len(det) > 0:
            det[:, :4] = scale_boxes_to_original(
                det[:, :4].clone(), (h_orig, w_orig), ratio, (dw, dh)
            )
            
            # Apply class mapping if provided
            if class_mapping:
                valid_mask = torch.zeros(len(det), dtype=torch.bool, device=device)
                for coco_cls, dataset_cls in class_mapping.items():
                    mask = (det[:, 5].long() == coco_cls)
                    det[mask, 5] = dataset_cls
                    valid_mask |= mask
                det = det[valid_mask]
            
            preds_np = det.cpu().numpy()
        else:
            preds_np = np.array([]).reshape(0, 6)
        
        tp, fp, fn, matched_ious, confs = match_predictions_to_gt(
            preds_np, gt_boxes, iou_threshold=0.5
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)
        all_confidences.extend(confs)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(all_ious) if all_ious else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'n_predictions': total_tp + total_fp,
        'all_confidences': all_confidences,
    }


def find_checkpoints(checkpoint_dir):
    """Find all checkpoint files sorted by epoch."""
    ckpt_dir = Path(checkpoint_dir)
    checkpoints = []
    
    for f in ckpt_dir.glob('*.pt'):
        # Match patterns: checkpoint_ep050.pt, teacher_ep050.pt, etc.
        match = re.search(r'ep(\d+)', f.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, f))
    
    # Also include best.pt and last.pt with special epoch markers
    best = ckpt_dir / 'best.pt'
    last = ckpt_dir / 'last.pt'
    
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        # Fallback: use best/last if no epoch checkpoints found
        if best.exists():
            checkpoints.append((-1, best))
        if last.exists():
            checkpoints.append((-2, last))
    
    return checkpoints


def load_model_from_checkpoint(weights_arch, checkpoint_path, device):
    """Load YOLOv8 architecture + checkpoint state_dict."""
    yolo = YOLO(weights_arch)
    model = yolo.model.to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict):
        if 'teacher_state_dict' in ckpt:
            state_dict = ckpt['teacher_state_dict']
            print(f"  → Loaded teacher state_dict (epoch {ckpt.get('epoch', '?')})")
        elif 'model' in ckpt:
            state_dict = ckpt['model']
            print(f"  → Loaded model state_dict (epoch {ckpt.get('epoch', '?')})")
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_quality_curves(all_results, output_dir):
    """Plot pseudo-label quality metrics over epochs for all options."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        'Full (GRL)': '#2E86AB',
        'NoGRL': '#7EBDC2',
        'Freeze Full': '#F18F01',
        'Freeze NoGRL': '#C73E1D',
    }
    markers = {
        'Full (GRL)': 'o',
        'NoGRL': 's',
        'Freeze Full': '^',
        'Freeze NoGRL': 'D',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('precision', 'Precision', axes[0, 0]),
        ('recall', 'Recall', axes[0, 1]),
        ('mean_iou', 'Mean IoU (matched)', axes[1, 0]),
        ('f1', 'F1 Score', axes[1, 1]),
    ]
    
    for metric_key, metric_name, ax in metrics:
        for option_name, results in all_results.items():
            epochs = [r['epoch'] for r in results]
            values = [r[metric_key] for r in results]
            color = colors.get(option_name, 'gray')
            marker = markers.get(option_name, 'o')
            ax.plot(epochs, values, '-' + marker, label=option_name,
                    color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Pseudo-label {metric_name} over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        # Bỏ set_ylim(bottom=0) để matplotlib tự động scale vừa khít (zoom in) vào vùng data
        ax.autoscale(enable=True, axis='y', tight=False)
    
    fig.suptitle('Pseudo-Label Quality: EMA Teacher vs Frozen Teacher',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pseudo_label_quality.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'pseudo_label_quality.png'}")


def plot_confidence_distribution(all_results, output_dir):
    """Plot confidence distribution histograms at final epoch."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    output_dir = Path(output_dir)
    
    colors = {
        'Full (GRL)': '#2E86AB',
        'NoGRL': '#7EBDC2',
        'Freeze Full': '#F18F01',
        'Freeze NoGRL': '#C73E1D',
    }
    
    n_options = len(all_results)
    # Tắt sharey=True để mỗi biểu đồ histogram có Y-axis riêng biệt, tránh bị khoảng trắng
    fig, axes = plt.subplots(1, n_options, figsize=(5 * n_options, 4), sharey=False)
    if n_options == 1:
        axes = [axes]
    
    for idx, (option_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        # Use last epoch's confidences
        if results and 'all_confidences' in results[-1]:
            confs = results[-1]['all_confidences']
            color = colors.get(option_name, 'gray')
            ax.hist(confs, bins=50, range=(0, 1), alpha=0.7, color=color,
                    edgecolor='black', linewidth=0.5)
            ax.axvline(x=np.mean(confs), color='red', linestyle='--',
                       label=f'Mean={np.mean(confs):.3f}')
            ax.axvline(x=np.median(confs), color='green', linestyle='--',
                       label=f'Median={np.median(confs):.3f}')
            ax.set_title(f'{option_name}\n(n={len(confs)} dets)', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
        else:
            ax.set_title(f'{option_name}\n(no data)', fontsize=11)
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Confidence Distribution on Target Domain (Final Epoch)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'confidence_distribution.png'}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_single(opt):
    """Evaluate pseudo-label quality for a single option."""
    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    
    image_dir = Path(opt.target_images)
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
    print(f"📸 Found {len(image_files)} target images")
    
    class_mapping = {0: 0, 1: 1, 2: 1}  # COCO person→0, bicycle→1(car), car→1
    
    checkpoints = find_checkpoints(opt.checkpoints_dir)
    if not checkpoints:
        print(f"❌ No checkpoints found in {opt.checkpoints_dir}")
        return []
    
    print(f"📦 Found {len(checkpoints)} checkpoints: {[f'ep{e}' for e, _ in checkpoints]}")
    
    results = []
    for epoch, ckpt_path in checkpoints:
        print(f"\n{'='*50}")
        print(f"Evaluating: {opt.option_name} — Epoch {epoch}")
        print(f"Checkpoint: {ckpt_path.name}")
        print(f"{'='*50}")
        
        model = load_model_from_checkpoint(opt.weights, ckpt_path, device)
        
        metrics = evaluate_checkpoint(
            model, image_files, opt.target_labels, device,
            imgsz=opt.imgsz, conf_thres=0.25, iou_thres=0.45,
            class_mapping=class_mapping,
        )
        metrics['epoch'] = epoch
        metrics['checkpoint'] = str(ckpt_path)
        results.append(metrics)
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        
        del model
        torch.cuda.empty_cache()
    
    return results


def run_compare(opt):
    """Compare pseudo-label quality across all 4 options."""
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for ckpt_dir, name in zip(opt.compare_dirs, opt.compare_names):
        opt.checkpoints_dir = ckpt_dir
        opt.option_name = name
        results = run_single(opt)
        all_results[name] = results
        
        # Save individual CSV
        csv_path = output_dir / f'{name.replace(" ", "_").lower()}_quality.csv'
        save_results_csv(results, csv_path)
    
    # Plot comparisons
    plot_quality_curves(all_results, output_dir)
    plot_confidence_distribution(all_results, output_dir)
    
    # Save all results as JSON
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = [{k: v for k, v in r.items() if k != 'all_confidences'}
                              for r in results]
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"{'ALL RESULTS SAVED':^60}")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def save_results_csv(results, csv_path):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = ['epoch', 'precision', 'recall', 'f1', 'mean_iou', 'tp', 'fp', 'fn', 'n_predictions']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved CSV: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Pseudo-Label Quality Analysis')
    
    # Compare mode
    parser.add_argument('--compare', action='store_true', help='Compare all 4 options')
    parser.add_argument('--compare-dirs', nargs='+', help='Checkpoint directories for each option')
    parser.add_argument('--compare-names', nargs='+', help='Names for each option')
    
    # Single mode
    parser.add_argument('--checkpoints-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--option-name', type=str, default='Model', help='Option name')
    
    # Common
    parser.add_argument('--weights', type=str, default='yolo26s.pt', help='YOLO26/YOLOv8 architecture weights')
    parser.add_argument('--target-images', type=str, required=True, help='Target validation images dir')
    parser.add_argument('--target-labels', type=str, required=True, help='Target validation labels dir')
    parser.add_argument('--output', type=str, default='results_explainable/pseudo_label_quality')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.compare:
        run_compare(args)
    else:
        results = run_single(args)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results_csv(results, output_dir / f'{args.option_name.replace(" ", "_").lower()}_quality.csv')
        plot_quality_curves({args.option_name: results}, output_dir)
        plot_confidence_distribution({args.option_name: results}, output_dir)
