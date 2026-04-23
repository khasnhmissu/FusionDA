"""RQ1 — Pseudo-label quality over epochs, with per-size breakdown + calibration.

Answers:
  * Does the EMA teacher's pseudo-label quality improve with training?
  * Is the quality gain concentrated in small / medium / large objects?
  * Are pseudo-labels well-calibrated (confidence ≈ actual precision)?

Inputs
------
  --run-dir         Path to the training run directory (e.g. runs/ablation/paired_full_40ep)
                    The script auto-discovers weights/checkpoint_ep*.pt files
                    (these are the only ones that contain teacher_state_dict).
  --data            data.yaml (for class names).
  --target-images   Directory with target val images (e.g. datasets/target_real/target_real/val/images)
  --target-labels   Directory with target val labels
  --conf-thres      Confidence threshold to match TEACHER_CONF_THRES from training
                    (0.5 is the current value in train.py). Default 0.5.
  --iou-thres       Matching IoU threshold for TP/FP. Default 0.5.
  --imgsz           Inference imgsz. Default 1024 (matches training).
  --device          GPU id or 'cpu'. Default 0.
  --output          Output directory. Default <run-dir>/explain/rq1.

Can also compare multiple runs:
  --compare-runs    One or more run dirs. --names gives labels.

Outputs (in --output/):
  pseudo_label_quality.png          4-panel curves (P, R, F1, meanIoU) over epochs
  per_size_quality.png              3x4 grid: {small, medium, large} × {P, R, F1, n_gt}
  confidence_distribution.png       Histogram of teacher confidence at final epoch
  calibration_diagram.png           Reliability curve: bin confidence vs actual precision
  quality_per_size.csv              Raw numbers for plotting / paper tables
  quality_overall.csv               Overall metrics per epoch
  summary.json                      All numbers collected

Example
-------
Single run:
  python explain/rq1_pseudo_label_quality.py \\
      --run-dir runs/ablation/paired_full_40ep \\
      --data data.yaml \\
      --target-images datasets/target_real/target_real/val/images \\
      --target-labels datasets/target_real/target_real/val/labels \\
      --conf-thres 0.5 --imgsz 1024

Compare multiple runs:
  python explain/rq1_pseudo_label_quality.py \\
      --compare-runs runs/ablation/paired_full_40ep runs/ablation/frozen_teacher \\
      --names "EMA Teacher" "Frozen Teacher" \\
      --data data.yaml \\
      --target-images datasets/target_real/target_real/val/images \\
      --target-labels datasets/target_real/target_real/val/labels
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Make `from _common import ...` work when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # type: ignore
    IMAGE_EXTENSIONS, box_area_xyxy, load_class_names, load_gt_labels,
    load_model_from_ckpt, match_predictions_to_gt, run_inference_single,
    size_bucket,
)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_teacher_checkpoints(run_dir):
    """Find `weights/checkpoint_ep*.pt` files (only these have teacher_state_dict).

    Returns a list of (epoch:int, path:Path), sorted by epoch.
    """
    weights_dir = Path(run_dir) / 'weights'
    if not weights_dir.exists():
        raise FileNotFoundError(f"{weights_dir} does not exist.")
    ckpts = []
    for f in weights_dir.glob('checkpoint_ep*.pt'):
        m = re.search(r'ep(\d+)', f.name)
        if m:
            ckpts.append((int(m.group(1)), f))
    ckpts.sort(key=lambda x: x[0])
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint_ep*.pt in {weights_dir}. "
            f"These are the only files with teacher_state_dict. "
            f"Verify the training run saved intermediate checkpoints "
            f"(train.py saves at epochs in `checkpoint_epochs`, default [10,20,30, final])."
        )
    return ckpts


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def eval_checkpoint(arch_weights, ckpt_path, image_files, labels_dir,
                    device, imgsz, conf_thres, iou_thres,
                    which='teacher'):
    """Run one checkpoint over all images, aggregate TP/FP/FN per size bucket.

    Returns a dict with overall + per-size {tp, fp, fn, n_gt, ious, confs},
    plus two arrays for calibration: (confs_of_preds, is_tp_flag).
    """
    model = load_model_from_ckpt(arch_weights, ckpt_path, device, which=which)

    buckets = ['small', 'medium', 'large']
    stats = {
        'overall': {'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0,
                    'ious': [], 'confs': [], 'calib_conf': [], 'calib_tp': []},
    }
    for b in buckets:
        stats[b] = {'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0, 'ious': []}

    for img_path in tqdm(image_files, desc=f'  Eval {Path(ckpt_path).name}', leave=False):
        import cv2
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]
        gt_boxes = load_gt_labels(
            Path(labels_dir) / (img_path.stem + '.txt'), w_orig, h_orig
        )
        preds = run_inference_single(
            model, img_path, device, imgsz=imgsz,
            conf_thres=conf_thres, iou_thres=iou_thres,
        )

        matched, fp_idx, fn_idx = match_predictions_to_gt(preds, gt_boxes, iou_thres=iou_thres)

        # Per-prediction TP flag for calibration.
        tp_flag = np.zeros(len(preds), dtype=bool)
        for (pi, gi, iou) in matched:
            tp_flag[pi] = True

        # Overall accumulation
        stats['overall']['tp'] += len(matched)
        stats['overall']['fp'] += len(fp_idx)
        stats['overall']['fn'] += len(fn_idx)
        stats['overall']['n_gt'] += len(gt_boxes)
        stats['overall']['ious'].extend([m[2] for m in matched])
        if len(preds):
            stats['overall']['confs'].extend(preds[:, 4].tolist())
            stats['overall']['calib_conf'].extend(preds[:, 4].tolist())
            stats['overall']['calib_tp'].extend(tp_flag.tolist())

        # Per-size accumulation — bucket by GT area for TP/FN, pred area for FP
        # (FPs have no matched GT, so classify by their OWN box area).
        # This is the COCO convention.
        if len(gt_boxes):
            gt_areas = box_area_xyxy(gt_boxes[:, 1:5])
        else:
            gt_areas = np.array([])

        # TPs — bucket by matched GT area.
        for (pi, gi, iou) in matched:
            b = size_bucket(gt_areas[gi])
            stats[b]['tp'] += 1
            stats[b]['ious'].append(iou)
        # FNs — bucket by GT area.
        for gi in fn_idx:
            b = size_bucket(gt_areas[gi])
            stats[b]['fn'] += 1
        # n_gt per size
        for gi in range(len(gt_boxes)):
            b = size_bucket(gt_areas[gi])
            stats[b]['n_gt'] += 1
        # FPs — bucket by prediction box area.
        if len(preds) and len(fp_idx):
            pred_areas = box_area_xyxy(preds[:, :4])
            for pi in fp_idx:
                b = size_bucket(pred_areas[pi])
                stats[b]['fp'] += 1

    del model
    torch.cuda.empty_cache()
    return stats


def metrics_from_counts(tp, fp, fn, ious):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    miou = float(np.mean(ious)) if ious else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'mean_iou': miou,
            'tp': tp, 'fp': fp, 'fn': fn, 'n_preds': tp + fp}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LINE_COLORS = ['#2E86AB', '#F18F01', '#C73E1D', '#7EBDC2', '#6B4C93', '#4CAF50']
LINE_MARKERS = ['o', 's', '^', 'D', 'v', 'X']


def _get_style(i):
    return LINE_COLORS[i % len(LINE_COLORS)], LINE_MARKERS[i % len(LINE_MARKERS)]


def plot_overall_curves(runs, output_dir):
    """4-panel: precision / recall / f1 / mean_iou vs epoch, one line per run."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [('precision', axes[0, 0]), ('recall', axes[0, 1]),
               ('f1', axes[1, 0]), ('mean_iou', axes[1, 1])]

    for i, (name, per_epoch) in enumerate(runs.items()):
        color, marker = _get_style(i)
        epochs = [e['epoch'] for e in per_epoch]
        for m, ax in metrics:
            ys = [metrics_from_counts(e['overall']['tp'], e['overall']['fp'],
                                       e['overall']['fn'], e['overall']['ious'])[m]
                  for e in per_epoch]
            ax.plot(epochs, ys, '-' + marker, color=color, label=name,
                    linewidth=2, markersize=7)

    for m, ax in metrics:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(m.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'Pseudo-label {m} over epochs', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle('Pseudo-label quality (overall)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'pseudo_label_quality.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def plot_per_size_curves(runs, output_dir):
    """3×4 grid: rows=(small, medium, large), cols=(P, R, F1, n_gt)."""
    sizes = ['small', 'medium', 'large']
    cols = [('precision', 'Precision'), ('recall', 'Recall'),
            ('f1', 'F1'), ('n_gt', 'GT count')]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    for row_i, sz in enumerate(sizes):
        for col_i, (key, label) in enumerate(cols):
            ax = axes[row_i, col_i]
            for i, (name, per_epoch) in enumerate(runs.items()):
                color, marker = _get_style(i)
                epochs = [e['epoch'] for e in per_epoch]
                if key == 'n_gt':
                    ys = [e[sz]['n_gt'] for e in per_epoch]
                else:
                    ys = [metrics_from_counts(e[sz]['tp'], e[sz]['fp'],
                                               e[sz]['fn'], e[sz]['ious'])[key]
                          for e in per_epoch]
                ax.plot(epochs, ys, '-' + marker, color=color, label=name,
                        linewidth=2, markersize=6)
            ax.set_title(f'{sz.capitalize()} — {label}', fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
            if row_i == 2:
                ax.set_xlabel('Epoch')
            if col_i == 0:
                ax.set_ylabel(sz.capitalize(), fontsize=11, fontweight='bold')
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=9, loc='best')

    fig.suptitle('Pseudo-label quality by object size (COCO buckets)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'per_size_quality.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def plot_confidence_distribution(runs, output_dir):
    """Histogram of prediction confidences at the LAST epoch of each run."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for i, (name, per_epoch) in enumerate(runs.items()):
        ax = axes[i]
        confs = per_epoch[-1]['overall']['confs']
        color, _ = _get_style(i)
        if confs:
            ax.hist(confs, bins=40, range=(0, 1), color=color, alpha=0.75,
                    edgecolor='black', linewidth=0.5)
            mean = float(np.mean(confs))
            median = float(np.median(confs))
            ax.axvline(mean, color='red', linestyle='--', label=f'mean={mean:.3f}')
            ax.axvline(median, color='green', linestyle=':', label=f'median={median:.3f}')
        ax.set_title(f'{name}\n(final epoch, n={len(confs)} preds)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Pseudo-label confidence distribution (final epoch)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'confidence_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


def plot_calibration(runs, output_dir, n_bins=10):
    """Reliability diagram at the final epoch: bin by conf, plot actual precision."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    for i, (name, per_epoch) in enumerate(runs.items()):
        confs = np.array(per_epoch[-1]['overall']['calib_conf'])
        is_tp = np.array(per_epoch[-1]['overall']['calib_tp'])
        if len(confs) == 0:
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(confs, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        xs, ys, counts = [], [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            xs.append(confs[mask].mean())
            ys.append(is_tp[mask].mean())
            counts.append(int(mask.sum()))
        color, marker = _get_style(i)
        ax.plot(xs, ys, '-' + marker, color=color, linewidth=2,
                markersize=8, label=f'{name} (n={len(confs)})')
        for x, y, c in zip(xs, ys, counts):
            ax.annotate(str(c), (x, y), fontsize=7, color=color,
                        xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Predicted confidence (bin mean)', fontsize=11)
    ax.set_ylabel('Actual precision (TP rate in bin)', fontsize=11)
    ax.set_title('Pseudo-label calibration (final epoch)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'calibration_diagram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {path}')


# ---------------------------------------------------------------------------
# CSV / JSON dumps
# ---------------------------------------------------------------------------

def dump_csv_overall(runs, output_dir):
    path = output_dir / 'quality_overall.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'precision', 'recall', 'f1', 'mean_iou',
                    'tp', 'fp', 'fn', 'n_preds', 'n_gt'])
        for name, per_epoch in runs.items():
            for e in per_epoch:
                m = metrics_from_counts(e['overall']['tp'], e['overall']['fp'],
                                         e['overall']['fn'], e['overall']['ious'])
                w.writerow([name, e['epoch'], f"{m['precision']:.4f}",
                            f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                            f"{m['mean_iou']:.4f}", m['tp'], m['fp'], m['fn'],
                            m['n_preds'], e['overall']['n_gt']])
    print(f'  ✅ {path}')


def dump_csv_per_size(runs, output_dir):
    path = output_dir / 'quality_per_size.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'size', 'precision', 'recall', 'f1',
                    'mean_iou', 'tp', 'fp', 'fn', 'n_gt'])
        for name, per_epoch in runs.items():
            for e in per_epoch:
                for sz in ['small', 'medium', 'large']:
                    m = metrics_from_counts(e[sz]['tp'], e[sz]['fp'],
                                             e[sz]['fn'], e[sz]['ious'])
                    w.writerow([name, e['epoch'], sz, f"{m['precision']:.4f}",
                                f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                                f"{m['mean_iou']:.4f}", m['tp'], m['fp'], m['fn'],
                                e[sz]['n_gt']])
    print(f'  ✅ {path}')


def dump_json_summary(runs, output_dir):
    out = {}
    for name, per_epoch in runs.items():
        out[name] = []
        for e in per_epoch:
            entry = {'epoch': e['epoch']}
            for scope in ['overall', 'small', 'medium', 'large']:
                s = e[scope]
                m = metrics_from_counts(s['tp'], s['fp'], s['fn'], s['ious'])
                entry[scope] = {**m, 'n_gt': s['n_gt']}
            per_epoch_e = entry
            out[name].append(per_epoch_e)
    path = output_dir / 'summary.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  ✅ {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)

    # Single-run OR compare-runs (mutually exclusive semantically)
    ap.add_argument('--run-dir', type=str,
                    help='Single training run directory.')
    ap.add_argument('--compare-runs', nargs='+',
                    help='Two or more training run directories to compare on the same chart.')
    ap.add_argument('--names', nargs='+',
                    help='Display names for runs (must match --compare-runs count).')

    ap.add_argument('--weights', type=str, default='yolo26s.pt',
                    help='Architecture weights file (for loading model topology).')
    ap.add_argument('--data', type=str, default='data.yaml')
    ap.add_argument('--target-images', type=str, required=True)
    ap.add_argument('--target-labels', type=str, required=True)

    ap.add_argument('--which', choices=['teacher', 'student'], default='teacher',
                    help="'teacher' uses teacher_state_dict (only in checkpoint_ep*.pt); "
                         "'student' uses the student's 'model' key instead.")
    ap.add_argument('--conf-thres', type=float, default=0.5,
                    help='Confidence threshold (default 0.5 matches TEACHER_CONF_THRES in train.py).')
    ap.add_argument('--iou-thres', type=float, default=0.5,
                    help='IoU threshold for TP matching.')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--device', type=str, default='0')

    ap.add_argument('--output', type=str, default=None,
                    help='Output dir. Default: <run-dir>/explain/rq1 for single run, '
                         './explain_out/rq1_compare for compare mode.')

    opt = ap.parse_args()

    if opt.run_dir is None and opt.compare_runs is None:
        ap.error('either --run-dir or --compare-runs is required')

    if opt.compare_runs and (opt.names is None or len(opt.names) != len(opt.compare_runs)):
        ap.error('--names must match --compare-runs count')

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)

    # Resolve run list
    if opt.run_dir:
        run_dirs = [opt.run_dir]
        run_names = [Path(opt.run_dir).name]
    else:
        run_dirs = opt.compare_runs
        run_names = opt.names

    # Resolve output dir
    if opt.output:
        out_dir = Path(opt.output)
    elif opt.run_dir:
        out_dir = Path(opt.run_dir) / 'explain' / 'rq1'
    else:
        out_dir = Path('explain_out/rq1_compare')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[RQ1] Output → {out_dir}')

    # List target val images
    img_dir = Path(opt.target_images)
    image_files = sorted([f for f in img_dir.iterdir()
                          if f.suffix.lower() in IMAGE_EXTENSIONS])
    print(f'[RQ1] {len(image_files)} target images in {img_dir}')

    # Eval each run × each checkpoint epoch
    runs = {}
    for run_dir, name in zip(run_dirs, run_names):
        print(f'\n[RQ1] Run "{name}" at {run_dir}')
        try:
            ckpts = discover_teacher_checkpoints(run_dir)
        except FileNotFoundError as e:
            print(f'  ⚠ {e}')
            continue
        print(f'  Found {len(ckpts)} checkpoints: {[f"ep{e}" for e, _ in ckpts]}')

        per_epoch = []
        for epoch, ckpt_path in ckpts:
            print(f'  ▶ epoch {epoch}')
            s = eval_checkpoint(
                opt.weights, ckpt_path, image_files, opt.target_labels,
                device, opt.imgsz, opt.conf_thres, opt.iou_thres,
                which=opt.which,
            )
            s['epoch'] = epoch
            per_epoch.append(s)
            m = metrics_from_counts(s['overall']['tp'], s['overall']['fp'],
                                     s['overall']['fn'], s['overall']['ious'])
            print(f'    overall  P={m["precision"]:.3f} R={m["recall"]:.3f} '
                  f'F1={m["f1"]:.3f}  mIoU={m["mean_iou"]:.3f}  '
                  f'TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}')
        runs[name] = per_epoch

    if not runs:
        print('[RQ1] No runs evaluated. Exiting.')
        sys.exit(1)

    print(f'\n[RQ1] Generating plots + CSVs in {out_dir}')
    plot_overall_curves(runs, out_dir)
    plot_per_size_curves(runs, out_dir)
    plot_confidence_distribution(runs, out_dir)
    plot_calibration(runs, out_dir)
    dump_csv_overall(runs, out_dir)
    dump_csv_per_size(runs, out_dir)
    dump_json_summary(runs, out_dir)
    print(f'[RQ1] Done.')


if __name__ == '__main__':
    main()
