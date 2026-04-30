"""RQ3 — D-RISE: black-box saliency for detection.

Answers:
  * WHERE in the image does the detector "look" when it produces a given
    detection? Compare a DA-trained student vs a baseline (no-DA) on
    foggy target images.

Why D-RISE (Petsiuk et al. 2021), not GradCAM / EigenCAM
--------------------------------------------------------
  * Black-box: treats the detector as a function; no hooks / gradients.
  * Detection-specific: similarity is box-IoU × class-score × objectness.
    Maps *per-detection*, not per-layer.
  * Robust to architecture. Works identically on YOLO26 E2E, YOLOv8, etc.
  * Drawback: N masked forwards per image. Use N = 1000–2000 for paper.

Algorithm (summary)
-------------------
  For each image we want to explain:
    1. Run the detector once, pick TARGET detections (by confidence + optional
       filtering to TP only if GT is available).
    2. Generate N random binary masks at a coarse grid (e.g. 8×8), upsample
       to input size with bilinear smoothing + random shift.
    3. For each mask m_i: fill masked-out pixels with gray, forward again.
    4. For each target detection t = (box_t, cls_t, conf_t):
         for each mask m_i: compute similarity s_i = max over model
         detections d of IoU(box_d, box_t) * P_d(cls_t).
       Saliency = Σ_i m_i * s_i  /  Σ_i s_i  (intensity-weighted average mask).
    5. Saliency is in input space, 1 map per target detection. Normalise to
       [0, 1] and overlay.

Inputs
------
  --checkpoint        Two or more checkpoints to compare (best.pt of each).
  --names             Display names (e.g. "Baseline", "Paired DA").
  --images            Directory with target-domain images.
  --labels            OPTIONAL: labels directory. If provided, we prefer
                      target detections that are TP (IoU≥0.5 with some GT).
  --n-images          How many images to compute saliency for. Default 6.
  --n-masks           D-RISE mask count. Default 1500. 500 = quick, 2000 = paper.
  --mask-grid         Base mask resolution (e.g. 8 → 8×8). Default 8.
  --mask-p            Probability a cell is unmasked (kept). Default 0.5.
  --max-targets       How many target detections per image to explain. Default 3.
  --conf-thres        Target detection confidence threshold. Default 0.3.
  --imgsz             Default 1024.
  --device            GPU id. Default 0.
  --output            Output dir. Default explain_out/rq3_drise.

Outputs
-------
  drise_<img_stem>.png    Grid: [original] × [saliency of det 1 under model A] etc.
                          Rows = target detections, columns = checkpoints.
  summary.json            Number of detections explained per image per model.

Example
-------
  python explain/rq3_d_rise.py \\
      --checkpoint runs/ablation/baseline/weights/best.pt \\
                   runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "Baseline" "Paired DA" \\
      --images datasets/target_real/target_real/val/images \\
      --labels datasets/target_real/target_real/val/labels \\
      --n-images 6 --n-masks 1500

Runtime estimate
----------------
  Per (image × model): ~n_masks forward passes on letterboxed 1024².
  YOLO26s @ batch=1 on RTX 4080 Super: ~5 ms → 1500 masks ≈ 7.5 s.
  6 images × 2 models × 7.5 s ≈ 90 s. Add image selection overhead → ~2–3 min total.
"""
from __future__ import annotations

import argparse
import json
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
    IMAGE_EXTENSIONS, compute_iou, letterbox, load_class_names, load_gt_labels,
    load_model_from_ckpt, match_predictions_to_gt, parse_yolo26_output,
    scale_boxes_to_original,
)


# ---------------------------------------------------------------------------
# D-RISE mask generation
# ---------------------------------------------------------------------------

def generate_drise_masks(n_masks, input_h, input_w, grid=8, p_keep=0.5,
                          rng=None):
    """Generate N binary masks using D-RISE recipe:
       1. Binary grid M ~ Bernoulli(p_keep) at (grid, grid).
       2. Upsample to (grid+1) * s where s = ceil(input / grid) (bilinear).
       3. Randomly crop back to (input_h, input_w) with random (dy, dx) in [0, s).

    Returns: np.ndarray [N, H, W] float32 in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    s_h = int(np.ceil(input_h / grid))
    s_w = int(np.ceil(input_w / grid))
    up_h = (grid + 1) * s_h
    up_w = (grid + 1) * s_w

    # Generate low-res masks in one shot.
    low = (rng.random((n_masks, grid, grid)) < p_keep).astype(np.float32)

    # Upsample via bilinear (cv2.resize).
    masks = np.empty((n_masks, input_h, input_w), dtype=np.float32)
    for i in range(n_masks):
        up = cv2.resize(low[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        dy = int(rng.integers(0, s_h))
        dx = int(rng.integers(0, s_w))
        masks[i] = up[dy:dy + input_h, dx:dx + input_w]
    return masks


# ---------------------------------------------------------------------------
# Detection extraction on masked images
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect_letterbox(model, img_lb_t, conf_thres=0.01):
    """Forward a letterboxed tensor [1, 3, H, W] already on device.
    Returns np [N, 6] in letterbox pixel space.
    Keep conf_thres low to capture candidate boxes for similarity matching.
    """
    pred = parse_yolo26_output(model(img_lb_t))
    if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
        det = pred[0]
        det = det[det[:, 4] > conf_thres]
        return det.cpu().numpy() if det.numel() else np.array([]).reshape(0, 6)
    # Fallback: apply NMS.
    from ultralytics.utils.nms import non_max_suppression
    dets = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45, max_det=300)
    det = dets[0]
    if det is None or len(det) == 0:
        return np.array([]).reshape(0, 6)
    return det.cpu().numpy()


# ---------------------------------------------------------------------------
# D-RISE core
# ---------------------------------------------------------------------------

def target_similarity(target, preds):
    """Max over `preds` of IoU(pred.box, target.box) * conf_pred IFF cls matches.

    target: np array (6,) [x1, y1, x2, y2, conf, cls]
    preds : np array (M, 6)
    """
    if len(preds) == 0:
        return 0.0
    target_cls = int(target[5])
    t_box = target[:4]
    best = 0.0
    for d in preds:
        if int(d[5]) != target_cls:
            continue
        iou = compute_iou(t_box, d[:4])
        if iou <= 0:
            continue
        s = iou * float(d[4])
        if s > best:
            best = s
    return best


@torch.no_grad()
def drise_saliency_one_image(model, img_lb_rgb, targets_letterbox, device,
                              n_masks=1500, grid=8, p_keep=0.5,
                              background=(114, 114, 114), rng=None, batch=8):
    """Compute D-RISE saliency for each target detection on a letterboxed image.

    Parameters
    ----------
    img_lb_rgb   : np.ndarray (H, W, 3) uint8 — letterboxed image in RGB.
    targets_letterbox : np.ndarray (T, 6) — target detections to explain.

    Returns
    -------
    saliency : np.ndarray (T, H, W) float32 in [0, 1] — one map per target.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = img_lb_rgb.shape[:2]
    if len(targets_letterbox) == 0:
        return np.zeros((0, H, W), dtype=np.float32)

    masks = generate_drise_masks(n_masks, H, W, grid=grid, p_keep=p_keep, rng=rng)

    img = img_lb_rgb.astype(np.float32)  # (H, W, 3)
    bg = np.array(background, dtype=np.float32)[None, None, :]

    T = len(targets_letterbox)
    sim_per_mask = np.zeros((n_masks, T), dtype=np.float32)

    for start in tqdm(range(0, n_masks, batch), desc='  D-RISE', leave=False):
        end = min(start + batch, n_masks)
        b = end - start
        batch_np = np.empty((b, 3, H, W), dtype=np.float32)
        for j in range(b):
            m = masks[start + j][..., None]  # (H, W, 1)
            masked = img * m + bg * (1.0 - m)
            batch_np[j] = masked.transpose(2, 0, 1)
        x = torch.from_numpy(batch_np).to(device).div_(255.0)
        pred = parse_yolo26_output(model(x))
        # Handle each item in batch
        if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
            for j in range(b):
                item = pred[j]
                item = item[item[:, 4] > 0.01]
                item_np = item.cpu().numpy() if item.numel() else np.array([]).reshape(0, 6)
                for ti, tgt in enumerate(targets_letterbox):
                    sim_per_mask[start + j, ti] = target_similarity(tgt, item_np)
        else:
            # Non-E2E path — do per-item NMS (slower).
            from ultralytics.utils.nms import non_max_suppression
            dets = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.45, max_det=300)
            for j, det in enumerate(dets):
                det_np = det.cpu().numpy() if (det is not None and len(det)) else np.array([]).reshape(0, 6)
                for ti, tgt in enumerate(targets_letterbox):
                    sim_per_mask[start + j, ti] = target_similarity(tgt, det_np)

    # Weighted average: saliency(y, x) = Σ_i masks[i,y,x] * sim[i, t]  /  Σ_i sim[i, t]
    saliency = np.zeros((T, H, W), dtype=np.float32)
    for ti in range(T):
        w = sim_per_mask[:, ti]  # (N,)
        denom = float(w.sum())
        if denom <= 1e-8:
            continue
        # Weighted sum of masks.
        # shape: (N,) × (N, H, W) → (H, W)
        acc = (masks * w[:, None, None]).sum(axis=0) / denom
        a_min, a_max = acc.min(), acc.max()
        if a_max > a_min:
            acc = (acc - a_min) / (a_max - a_min)
        saliency[ti] = acc.astype(np.float32)

    return saliency


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def overlay_heatmap(img_rgb, heat, alpha=0.55):
    heat = np.power(np.clip(heat, 0, 1), 0.7)
    heat_u8 = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb.astype(np.float32) * (1 - alpha)
                   + colored.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def draw_one_box(img, box, color=(255, 255, 255), label=None, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Image + target selection
# ---------------------------------------------------------------------------

def select_images_and_targets(model, image_dir, label_dir, device, class_names,
                                imgsz, n_images, conf_thres, max_targets, rng):
    """Pick images that have detections. If labels available, prefer TPs."""
    files = sorted([f for f in Path(image_dir).iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS])

    # Scan up to 5x n_images to ensure we find good candidates.
    pool = files[:max(n_images * 5, 30)]
    scored = []
    for f in pool:
        img_bgr = cv2.imread(str(f))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_o, w_o = img_rgb.shape[:2]
        img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, stride=32)
        t = torch.from_numpy(img_lb.transpose(2, 0, 1)).to(device).float().div_(255.0).unsqueeze(0)
        det_lb = detect_letterbox(model, t, conf_thres=conf_thres)
        # Prefer images with GT (for TP filtering later).
        n_det = len(det_lb)
        gt_n = 0
        if label_dir is not None:
            gt = load_gt_labels(Path(label_dir) / (f.stem + '.txt'), w_o, h_o)
            gt_n = len(gt)
        scored.append((n_det + 0.5 * gt_n, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [f for _, f in scored[:n_images]]
    return chosen


def extract_targets_for_image(model, img_path, label_dir, device, imgsz,
                                conf_thres, max_targets):
    """Pick up to `max_targets` target detections to explain on this image.

    Prefers TP detections (match GT) when labels are available.
    Returns list: (img_lb_rgb [H,W,3], targets_letterbox np [T,6]).
    """
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_o, w_o = img_rgb.shape[:2]
    img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, stride=32)
    t = torch.from_numpy(img_lb.transpose(2, 0, 1)).to(device).float().div_(255.0).unsqueeze(0)

    # Detections in LETTERBOX space (this is what D-RISE operates in).
    det_lb = detect_letterbox(model, t, conf_thres=conf_thres)
    if len(det_lb) == 0:
        return img_lb, np.array([]).reshape(0, 6)

    # If we have labels, rank by TP first (dets that match some GT).
    if label_dir is not None:
        gt = load_gt_labels(Path(label_dir) / (img_path.stem + '.txt'), w_o, h_o)
        if len(gt):
            # Scale GT to letterbox space: pixel * r + pad
            r = ratio[0]
            gt_lb = gt.copy()
            gt_lb[:, 1] = gt[:, 1] * r + dw
            gt_lb[:, 2] = gt[:, 2] * r + dh
            gt_lb[:, 3] = gt[:, 3] * r + dw
            gt_lb[:, 4] = gt[:, 4] * r + dh
            matched, _, _ = match_predictions_to_gt(det_lb, gt_lb, iou_thres=0.5)
            tp_idx = set(pi for pi, _, _ in matched)
            tps = [i for i in range(len(det_lb)) if i in tp_idx]
            fps = [i for i in range(len(det_lb)) if i not in tp_idx]
            # Sort TPs and FPs each by conf desc.
            tps.sort(key=lambda i: -det_lb[i, 4])
            fps.sort(key=lambda i: -det_lb[i, 4])
            order = tps + fps
            det_lb = det_lb[order[:max_targets]]
        else:
            det_lb = det_lb[np.argsort(-det_lb[:, 4])[:max_targets]]
    else:
        det_lb = det_lb[np.argsort(-det_lb[:, 4])[:max_targets]]

    return img_lb, det_lb


# ---------------------------------------------------------------------------
# Plot per-image comparison
# ---------------------------------------------------------------------------

def plot_compare_drise(img_stem, img_lb_per_ckpt, targets_per_ckpt,
                        saliency_per_ckpt, names, class_names, out_dir):
    """Rows = target detections (up to max across checkpoints).
       Cols = [original] + one per checkpoint.
    """
    n_ckpts = len(names)
    # We base layout on the FIRST checkpoint's targets.
    ref_img = img_lb_per_ckpt[0]
    ref_targets = targets_per_ckpt[0]
    if len(ref_targets) == 0:
        print(f'  ⚠ no targets for {img_stem} — skipping.')
        return

    n_rows = len(ref_targets)
    n_cols = 1 + n_ckpts
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for r, tgt in enumerate(ref_targets):
        # Col 0: original with the target box highlighted.
        base = ref_img.copy()
        cls_name = class_names.get(int(tgt[5]), str(int(tgt[5])))
        draw_one_box(base, tgt[:4], color=(255, 255, 0),
                     label=f'Target: {cls_name} {tgt[4]:.2f}')
        axes[r, 0].imshow(base)
        axes[r, 0].set_title(f'Target {r+1}', fontsize=10, fontweight='bold')
        axes[r, 0].axis('off')

        # Cols 1..n: saliency per checkpoint.
        for c, name in enumerate(names):
            tgts_here = targets_per_ckpt[c]
            sal_here = saliency_per_ckpt[c]
            # Try to match THIS ckpt's target to the reference target by IoU.
            best_idx = -1
            best_iou = 0.0
            for ti, td in enumerate(tgts_here):
                if int(td[5]) != int(tgt[5]):
                    continue
                iou = compute_iou(tgt[:4], td[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, ti
            ax = axes[r, c + 1]
            if best_idx >= 0 and best_iou > 0.1:
                sal = sal_here[best_idx]
                img_to_draw = img_lb_per_ckpt[c].copy()
                over = overlay_heatmap(img_to_draw, sal, alpha=0.55)
                draw_one_box(over, tgts_here[best_idx, :4], color=(255, 255, 255),
                             label=f'{name}  IoU={best_iou:.2f}')
                ax.imshow(over)
            else:
                # Checkpoint didn't produce a matching detection.
                ax.imshow(img_lb_per_ckpt[c])
                ax.text(10, 30, f'{name}: NO matching det',
                        color='red', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7))
            ax.set_title(name if r == 0 else '', fontsize=11, fontweight='bold')
            ax.axis('off')

    fig.suptitle(f'D-RISE saliency — {img_stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save = out_dir / f'drise_{img_stem}.png'
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {save}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--weights', type=str, default='yolo26s.pt')
    ap.add_argument('--data', type=str, default='configs/data/data.yaml')
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--labels', type=str, default=None,
                    help='Optional labels dir — used to prefer TP detections for explanation.')
    ap.add_argument('--n-images', type=int, default=6)
    ap.add_argument('--n-masks', type=int, default=1500)
    ap.add_argument('--mask-grid', type=int, default=8)
    ap.add_argument('--mask-p', type=float, default=0.5)
    ap.add_argument('--max-targets', type=int, default=3)
    ap.add_argument('--conf-thres', type=float, default=0.3)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--which', choices=['student', 'teacher'], default='student')
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--batch', type=int, default=8,
                    help='Batch of masked forwards per GPU call.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='explain_out/rq3_drise')
    opt = ap.parse_args()

    if len(opt.checkpoint) != len(opt.names):
        ap.error('--names count must match --checkpoint count')

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[RQ3] Output → {out_dir}')
    print(f'[RQ3] {opt.n_masks} masks × {len(opt.checkpoint)} models × {opt.n_images} images')

    class_names = load_class_names(opt.data)
    rng = np.random.default_rng(opt.seed)

    # Pick images using the FIRST model → same set for all models (fair compare).
    first_model = load_model_from_ckpt(opt.weights, opt.checkpoint[0], device, which=opt.which)
    image_files = select_images_and_targets(
        first_model, opt.images, opt.labels, device, class_names,
        opt.imgsz, opt.n_images, opt.conf_thres, opt.max_targets, rng,
    )
    print(f'[RQ3] Selected {len(image_files)} images: {[f.stem for f in image_files]}')
    del first_model
    torch.cuda.empty_cache()

    summary = {}

    # For each model, compute saliency per image.
    per_ckpt_results = {name: {} for name in opt.names}
    for ckpt, name in zip(opt.checkpoint, opt.names):
        print(f'\n[RQ3] Checkpoint "{name}"')
        model = load_model_from_ckpt(opt.weights, ckpt, device, which=opt.which)
        summary[name] = {'checkpoint': ckpt, 'per_image': {}}

        for img_path in image_files:
            print(f'  Image {img_path.stem}')
            img_lb, targets_lb = extract_targets_for_image(
                model, img_path, opt.labels, device, opt.imgsz,
                opt.conf_thres, opt.max_targets,
            )
            if len(targets_lb) == 0:
                print(f'    ⚠ no targets; skipping')
                per_ckpt_results[name][img_path.stem] = {
                    'img_lb': img_lb, 'targets': targets_lb, 'saliency': None,
                }
                summary[name]['per_image'][img_path.stem] = 0
                continue

            saliency = drise_saliency_one_image(
                model, img_lb, targets_lb, device,
                n_masks=opt.n_masks, grid=opt.mask_grid, p_keep=opt.mask_p,
                rng=np.random.default_rng(opt.seed),
                batch=opt.batch,
            )
            per_ckpt_results[name][img_path.stem] = {
                'img_lb': img_lb, 'targets': targets_lb, 'saliency': saliency,
            }
            summary[name]['per_image'][img_path.stem] = int(len(targets_lb))

        del model
        torch.cuda.empty_cache()

    # Plot per-image compare
    print('\n[RQ3] Writing figures…')
    for img_path in image_files:
        stem = img_path.stem
        img_lb_per = [per_ckpt_results[n][stem]['img_lb'] for n in opt.names]
        tgt_per = [per_ckpt_results[n][stem]['targets'] for n in opt.names]
        sal_per = [per_ckpt_results[n][stem]['saliency'] for n in opt.names]
        if sal_per[0] is None or len(tgt_per[0]) == 0:
            continue
        plot_compare_drise(stem, img_lb_per, tgt_per, sal_per,
                            opt.names, class_names, out_dir)

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'  ✅ {out_dir / "summary.json"}')
    print('\n[RQ3] Done.')


if __name__ == '__main__':
    main()
