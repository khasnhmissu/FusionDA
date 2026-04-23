"""Detection diff — TP / FP / FN visualisation across models.

Draws bounding boxes colour-coded by match status:
  * GREEN : TP (prediction matches a GT, IoU≥thres, correct class)
  * RED   : FP (prediction does not match any GT)
  * YELLOW (dashed): FN (GT with no matching prediction)
  * CYAN  : Ground truth (in the GT-only column)

Differences from the old root-level `detection_diff.py`:
  * Fixed bug: old default `class_mapping={0:0, 2:1}` silently dropped all
    class-1 (car) predictions for the user's 2-class model. The new default
    is `None` (no remapping). Use `--class-mapping "0:0,2:1"` to re-enable
    if ever needed (e.g. evaluating a COCO-pretrained model on this data).
  * Fair image selection: `--best-matches` no longer biases toward the
    first model. It now picks images where models DISAGREE the most
    (highest std of TP count) → actually useful for comparison figures.
  * Optional --size-filter={small,medium,large} to focus on a size bucket.

Inputs
------
  --checkpoint        One or more model checkpoints to compare.
  --names             Display label per checkpoint.
  --data              data.yaml (for class names).
  --images            Directory with target val images.
  --labels            Directory with target val labels.
  --n-images          How many images in the comparison grid. Default 8.
  --iou-thres         Matching IoU. Default 0.5.
  --conf-thres        Prediction conf threshold. Default 0.25.
  --imgsz             Default 1024.
  --class-mapping     Optional "src:dst,src:dst" remap applied to predictions.
                      Leave unset for user's native 2-class models.
  --best-matches      Pick images where models disagree most (std of TP count).
  --size-filter       {small,medium,large} — focus on a single COCO size bucket.
  --device            GPU id. Default 0.
  --output            Default explain_out/detection_diff.

Outputs
-------
  detection_diff_comparison.png   Grid: rows=images, cols=[GT] + [each model].
  individual/                     Per-image, per-model detail images.

Example
-------
  python explain/detection_diff.py \\
      --checkpoint runs/ablation/baseline/weights/best.pt \\
                   runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "Baseline" "Paired DA" \\
      --data data.yaml \\
      --images datasets/target_real/target_real/val/images \\
      --labels datasets/target_real/target_real/val/labels \\
      --n-images 8 --best-matches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # type: ignore
    IMAGE_EXTENSIONS, box_area_xyxy, load_class_names, load_gt_labels,
    load_model_from_ckpt, match_predictions_to_gt, run_inference_single,
    size_bucket,
)


COLOR_TP = (0, 200, 0)
COLOR_FP = (0, 0, 220)
COLOR_FN = (0, 200, 255)
COLOR_GT = (255, 180, 0)


# ---------------------------------------------------------------------------
# Class remap (explicit, no hidden default)
# ---------------------------------------------------------------------------

def parse_class_mapping(spec):
    if not spec:
        return None
    out = {}
    for pair in spec.split(','):
        src, dst = pair.split(':')
        out[int(src)] = int(dst)
    return out


def apply_class_mapping(det, class_mapping, device='cpu'):
    """Remap `det[:, 5]` via `class_mapping`. Drops rows whose class is not in
    the mapping keys (dangerous — prefer None for user's native models)."""
    if class_mapping is None or len(det) == 0:
        return det
    valid = np.zeros(len(det), dtype=bool)
    for src, dst in class_mapping.items():
        mask = det[:, 5].astype(int) == src
        det[mask, 5] = dst
        valid |= mask
    return det[valid]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=10, gap=5):
    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(img, (x, y1), (xe, y1), color, thickness)
        cv2.line(img, (x, y2), (xe, y2), color, thickness)
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, ye), color, thickness)
        cv2.line(img, (x2, y), (x2, ye), color, thickness)


def draw_detection_diff(img_bgr, tp_boxes, fp_boxes, fn_boxes, title, class_names):
    img = img_bgr.copy()
    # FN (dashed yellow) — draw first so predictions overlap on top.
    for gt in fn_boxes:
        cls_id = int(gt[0])
        x1, y1, x2, y2 = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
        draw_dashed_rect(img, x1, y1, x2, y2, COLOR_FN, 2)
        label = class_names.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, f'FN:{label}', (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FN, 1)
    # FP
    for p in fp_boxes:
        x1, y1, x2, y2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        conf, cls = float(p[4]), int(p[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FP, 2)
        label = class_names.get(cls, f'cls{cls}')
        cv2.putText(img, f'FP:{label} {conf:.2f}', (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FP, 1)
    # TP
    for p in tp_boxes:
        x1, y1, x2, y2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        conf, cls = float(p[4]), int(p[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TP, 2)
        label = class_names.get(cls, f'cls{cls}')
        cv2.putText(img, f'TP:{label} {conf:.2f}', (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TP, 1)
    # Title bar
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, f'{title}  TP:{len(tp_boxes)} FP:{len(fp_boxes)} FN:{len(fn_boxes)}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def draw_gt(img_bgr, gt_boxes, class_names):
    img = img_bgr.copy()
    for gt in gt_boxes:
        cls_id = int(gt[0])
        x1, y1, x2, y2 = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GT, 2)
        label = class_names.get(cls_id, f'cls{cls_id}')
        cv2.putText(img, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, f'Ground Truth ({len(gt_boxes)} objects)',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------------------
# Image selection: disagreement-based
# ---------------------------------------------------------------------------

def score_disagreement(tp_counts_per_model):
    """Higher → more disagreement → more useful for comparison figure."""
    arr = np.array(tp_counts_per_model, dtype=np.float32)
    return float(arr.std())


# ---------------------------------------------------------------------------
# Size filter
# ---------------------------------------------------------------------------

def filter_by_size(gt_boxes, preds, bucket):
    """Keep GT whose area is in `bucket`. Keep preds whose area is in bucket."""
    if len(gt_boxes):
        areas = box_area_xyxy(gt_boxes[:, 1:5])
        gt_mask = np.array([size_bucket(a) == bucket for a in areas])
        gt_boxes = gt_boxes[gt_mask]
    if len(preds):
        areas = box_area_xyxy(preds[:, :4])
        pr_mask = np.array([size_bucket(a) == bucket for a in areas])
        preds = preds[pr_mask]
    return gt_boxes, preds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def classify_and_split(preds, gt_boxes, iou_thres):
    matched, fp_idx, fn_idx = match_predictions_to_gt(preds, gt_boxes, iou_thres=iou_thres)
    tp = [preds[m[0]] for m in matched] if len(preds) else []
    fp = [preds[i] for i in fp_idx] if len(preds) else []
    fn = [gt_boxes[i] for i in fn_idx] if len(gt_boxes) else []
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--weights', type=str, default='yolo26s.pt')
    ap.add_argument('--data', type=str, default='data.yaml')
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--n-images', type=int, default=8)
    ap.add_argument('--iou-thres', type=float, default=0.5)
    ap.add_argument('--conf-thres', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--class-mapping', type=str, default=None,
                    help='Format: "0:0,1:1" — applied to prediction class ids. '
                         'LEAVE UNSET for the user\'s native 2-class models. '
                         'The old default was buggy.')
    ap.add_argument('--best-matches', action='store_true',
                    help='Pick images where models disagree most (std of TP count).')
    ap.add_argument('--size-filter', choices=['small', 'medium', 'large'], default=None,
                    help='Keep only GT + preds within a size bucket.')
    ap.add_argument('--which', choices=['student', 'teacher'], default='student')
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--output', type=str, default='explain_out/detection_diff')
    opt = ap.parse_args()

    if len(opt.checkpoint) != len(opt.names):
        ap.error('--names count must match --checkpoint count')

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'individual').mkdir(exist_ok=True)
    print(f'[det-diff] Output → {out_dir}')

    class_names = load_class_names(opt.data)
    class_mapping = parse_class_mapping(opt.class_mapping)
    if class_mapping:
        print(f'[det-diff] class_mapping = {class_mapping}')
    else:
        print(f'[det-diff] class_mapping = None (predictions kept as-is)')

    # Pool of images.
    all_images = sorted([f for f in Path(opt.images).iterdir()
                         if f.suffix.lower() in IMAGE_EXTENSIONS])

    # Run all models on a pool first — either for best-matches selection or
    # directly for the final grid. We cap pool size to the first 150 images
    # to keep this practical.
    pool = all_images[:max(150, opt.n_images * 5)] if opt.best_matches else all_images[:opt.n_images]

    # Per (image, model) → preds np array.
    preds_per = {name: {} for name in opt.names}
    models = {}

    for ckpt, name in zip(opt.checkpoint, opt.names):
        print(f'\n[det-diff] Inference: {name} ({ckpt})')
        model = load_model_from_ckpt(opt.weights, ckpt, device, which=opt.which)
        models[name] = model  # keep for individual images later
        for img_path in tqdm(pool, desc=f'  {name}', leave=False):
            det = run_inference_single(model, img_path, device,
                                        imgsz=opt.imgsz,
                                        conf_thres=opt.conf_thres,
                                        iou_thres=0.45)
            det = apply_class_mapping(det, class_mapping)
            preds_per[name][img_path.name] = det
        # Free model after we've collected predictions for selection.
        # Keep for best-matches compute, then free at end of loop.
    # Done with models (we only re-render boxes — no more inference needed).
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    # Select n_images.
    if opt.best_matches:
        print('\n[det-diff] Ranking images by model disagreement (std of TP count)…')
        scored = []
        for img_path in pool:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            gt = load_gt_labels(Path(opt.labels) / (img_path.stem + '.txt'), w, h)
            tp_counts = []
            for name in opt.names:
                preds = preds_per[name][img_path.name]
                gt_use, preds_use = (gt, preds)
                if opt.size_filter:
                    gt_use, preds_use = filter_by_size(gt, preds, opt.size_filter)
                tp, _, _ = classify_and_split(preds_use, gt_use, opt.iou_thres)
                tp_counts.append(len(tp))
            if sum(tp_counts) == 0:
                continue  # skip images where no model finds anything
            scored.append((score_disagreement(tp_counts), img_path))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [f for _, f in scored[:opt.n_images]]
    else:
        selected = pool[:opt.n_images]

    print(f'[det-diff] Selected {len(selected)} images: {[f.stem for f in selected]}')

    # Build comparison grid.
    n_cols = 1 + len(opt.names)
    fig, axes = plt.subplots(len(selected), n_cols,
                              figsize=(4.5 * n_cols, 4.5 * len(selected)),
                              squeeze=False)

    for r, img_path in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]
        gt = load_gt_labels(Path(opt.labels) / (img_path.stem + '.txt'), w, h)

        gt_use_display, _ = filter_by_size(gt, np.array([]).reshape(0, 6), opt.size_filter) \
            if opt.size_filter else (gt, None)
        # GT col
        gt_img = draw_gt(img_bgr, gt_use_display if opt.size_filter else gt, class_names)
        axes[r, 0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        if r == 0:
            axes[r, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes[r, 0].set_ylabel(img_path.stem[:20], fontsize=8, rotation=0, labelpad=60, va='center')
        axes[r, 0].axis('off')

        for c, name in enumerate(opt.names):
            preds = preds_per[name][img_path.name]
            gt_use, preds_use = filter_by_size(gt, preds, opt.size_filter) \
                if opt.size_filter else (gt, preds)
            tp, fp, fn = classify_and_split(preds_use, gt_use, opt.iou_thres)
            diff_img = draw_detection_diff(img_bgr, tp, fp, fn, name, class_names)
            axes[r, c + 1].imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
            if r == 0:
                axes[r, c + 1].set_title(name, fontsize=11, fontweight='bold')
            axes[r, c + 1].axis('off')

            # Individual save.
            save_name = f'{img_path.stem}_{name.replace(" ", "_").lower()}.jpg'
            cv2.imwrite(str(out_dir / 'individual' / save_name), diff_img)

    title_suffix = f' (size={opt.size_filter})' if opt.size_filter else ''
    fig.suptitle(f'Detection diff — target domain{title_suffix}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = out_dir / 'detection_diff_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {save_path}')
    print(f'  ✅ {len(selected) * len(opt.names)} individuals in {out_dir / "individual"}')
    print('\n[det-diff] Done.')


if __name__ == '__main__':
    main()
