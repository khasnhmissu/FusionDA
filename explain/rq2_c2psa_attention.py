"""RQ2b — YOLO26 C2PSA self-attention visualisation.

Answers:
  * WHERE does the model attend inside the C2PSA (Partial Self-Attention)
    block at the backbone end, and does that attention concentrate on
    foreground objects for a DA-trained student vs a baseline?

Mechanism
---------
YOLO26's C2PSA block contains a PSABlock whose core is a multi-head attention
(Q·K^T softmax). This script monkey-patches the `Attention.forward` of that
block to capture `attn = softmax(q·k)` weights per layer call.

For each inspected image:
  * Run the student model (forward, no_grad).
  * Pull `attn` tensor of shape (1, heads, N, N) where N = H_feat × W_feat.
  * For a set of *query tokens* — chosen as the feature-map cells under the
    centre of detected bounding boxes — plot the attention map (reshape
    1×N → H×W, upsample to image size) overlaid on the letterboxed input.
  * Aggregate (mean over heads) or per-head (optional).

Inputs
------
  --checkpoint    One or more student checkpoints.
  --names         Display label per checkpoint.
  --data          data.yaml (for class names).
  --images        Image directory (typically target val images). A subset is sampled.
  --n-images      Number of images to visualise. Default 6.
  --imgsz         Default 1024.
  --conf-thres    Detection threshold to pick query-token locations. Default 0.25.
  --max-queries   Max number of query points per image. Default 4.
  --per-head      Also save per-head heatmaps (multi-panel).

Outputs (in --output, default explain_out/rq2_attention/):
  attention_<name>__<img_stem>.png    For each image + each checkpoint.
                                      Panels: [original with boxes]
                                              [attention from query token 1]
                                              [attention from query token 2] ...
  compare_<img_stem>.png              If multiple checkpoints — side-by-side
                                      comparison (one row per checkpoint).

Example
-------
Single DA model:
  python explain/rq2_c2psa_attention.py \\
      --checkpoint runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "Paired DA" \\
      --data data.yaml \\
      --images datasets/target_real/target_real/val/images \\
      --n-images 6

Compare DA vs no-DA:
  python explain/rq2_c2psa_attention.py \\
      --checkpoint runs/ablation/baseline/weights/best.pt \\
                   runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "Baseline" "Paired DA" \\
      --data data.yaml \\
      --images datasets/target_real/target_real/val/images

Limitations
-----------
  * Attention weights are meaningful ONLY if the hooked module has the
    signature of a standard SDPA: q, k, v = split(qkv), attn = softmax(q·k).
    YOLO26's C2PSA uses exactly this pattern. If the layer structure
    changes across Ultralytics versions, this script prints which module
    classes it found and falls back gracefully.
  * Hooked attention is at the BACKBONE END — coarse (e.g. 32×32 for
    imgsz=1024). We upsample bilinearly to input size; small objects may
    fall in a single feature-map cell.
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # type: ignore
    IMAGE_EXTENSIONS, letterbox, load_class_names, load_model_from_ckpt,
    parse_yolo26_output, scale_boxes_to_original,
)
from domain_adaptation import find_last_backbone_layer  # type: ignore


# ---------------------------------------------------------------------------
# Attention weight capture
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Monkey-patch `Attention.forward` inside C2PSA to capture softmax(QK)/scale.

    Assumes Ultralytics' Attention has the pattern:
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, H, ..., N).split([key_dim, key_dim, head_dim], dim=2)
        attn = (q.transpose(-2,-1) @ k) * self.scale
        attn = attn.softmax(dim=-1)     # ← we want this
        x = v @ attn.transpose(-2,-1)
        ...
    If the attribute names differ, we gracefully fall back.
    """

    def __init__(self, model):
        self.captured = []         # list of tensors (B, heads, N, N) per call
        self.feat_shapes = []      # list of (H, W) of the feature map
        self.orig_forwards = []    # saved originals for remove()
        self.attn_modules = []
        self._install(model)

    @staticmethod
    def _find_attn_modules(model):
        """Find Attention submodules within C2PSA (backbone end)."""
        layers = model.model if hasattr(model, 'model') else model
        out = []
        # Only go into the last-backbone layer (C2PSA on YOLO26).
        last_idx = find_last_backbone_layer(model)
        backbone_end = layers[last_idx]
        # Scan children for nn.Module named 'Attention' (YOLO26 Attention block).
        for m in backbone_end.modules():
            if m.__class__.__name__ in ('Attention', 'PSA', 'PSABlock'):
                # PSABlock wraps an Attention; Attention is what we want.
                if m.__class__.__name__ == 'Attention':
                    out.append(m)
        if not out:
            # Fallback: any module with 'qkv' linear/conv attribute.
            for m in backbone_end.modules():
                if hasattr(m, 'qkv'):
                    out.append(m)
        return out

    def _install(self, model):
        modules = self._find_attn_modules(model)
        if not modules:
            print('[AttnCap] ⚠ No Attention module found inside C2PSA. '
                  'Will fallback to activation-based visualisation.')
            return
        print(f'[AttnCap] Hooked {len(modules)} Attention module(s).')

        capture_ref = self

        for m in modules:
            orig = m.forward

            def wrapped(x, _m=m, _orig=orig):
                # Replicate Attention.forward while intercepting softmax attn.
                B, C, H, W = x.shape
                N = H * W
                try:
                    num_heads = getattr(_m, 'num_heads', None) or getattr(_m, 'heads', None)
                    key_dim = getattr(_m, 'key_dim', None)
                    head_dim = getattr(_m, 'head_dim', None)
                    if num_heads is None or key_dim is None or head_dim is None:
                        # Can't split qkv ourselves — just call original.
                        out = _orig(x)
                        return out
                    qkv = _m.qkv(x)
                    qkv = qkv.view(B, num_heads, key_dim * 2 + head_dim, N)
                    q, k, v = qkv.split([key_dim, key_dim, head_dim], dim=2)
                    attn = (q.transpose(-2, -1) @ k) * getattr(_m, 'scale', 1.0 / (key_dim ** 0.5))
                    attn = attn.softmax(dim=-1)
                    capture_ref.captured.append(attn.detach().cpu())
                    capture_ref.feat_shapes.append((H, W))
                    x_out = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
                    pe = getattr(_m, 'pe', None)
                    if pe is not None:
                        x_out = x_out + pe(v.reshape(B, -1, H, W))
                    proj = getattr(_m, 'proj', None)
                    if proj is not None:
                        x_out = proj(x_out)
                    return x_out
                except Exception as e:
                    # Fall through to original forward on any mismatch.
                    if not capture_ref._warned:
                        print(f'[AttnCap] fallback to original forward ({e})')
                        capture_ref._warned = True
                    return _orig(x)

            self._warned = False
            m.forward = wrapped
            self.orig_forwards.append((m, orig))
            self.attn_modules.append(m)

    def reset(self):
        self.captured = []
        self.feat_shapes = []

    def remove(self):
        for m, orig in self.orig_forwards:
            m.forward = orig
        self.orig_forwards = []


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def pick_queries_from_detections(detections_letterbox, H_feat, W_feat,
                                  img_h, img_w, max_queries=4,
                                  conf_thres=0.25):
    """Given detection boxes in letterbox coords, pick up to max_queries query
    token indices (centre of each detection projected to feature-map cells).
    Returns list of (token_index, (cy_px, cx_px, cls, conf))."""
    queries = []
    if detections_letterbox is None or len(detections_letterbox) == 0:
        return queries
    dets = detections_letterbox
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().numpy()
    # Sort by conf desc, keep above threshold, take top max_queries
    order = np.argsort(-dets[:, 4])
    for i in order:
        if dets[i, 4] < conf_thres:
            continue
        x1, y1, x2, y2, conf, cls = dets[i, :6]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        fx = int(np.clip(cx / img_w * W_feat, 0, W_feat - 1))
        fy = int(np.clip(cy / img_h * H_feat, 0, H_feat - 1))
        token = fy * W_feat + fx
        queries.append((token, (int(cy), int(cx), int(cls), float(conf),
                                 int(x1), int(y1), int(x2), int(y2))))
        if len(queries) >= max_queries:
            break
    return queries


def build_attention_heatmap(attn, token_idx, H_feat, W_feat, head='mean'):
    """Turn a (1, heads, N, N) attention into a (H, W) map from query=token_idx.

    head: 'mean' aggregate over heads, or int head index.
    """
    if attn.ndim == 4:
        a = attn[0]  # (heads, N, N)
    else:
        a = attn
    if isinstance(head, int):
        row = a[head, token_idx, :]
    else:
        row = a.mean(dim=0)[token_idx, :]
    row = row.numpy().reshape(H_feat, W_feat)
    # Normalise to [0, 1].
    r_min, r_max = row.min(), row.max()
    if r_max > r_min:
        row = (row - r_min) / (r_max - r_min)
    return row


def overlay_heatmap(img_rgb, heat, alpha=0.55, colormap=cv2.COLORMAP_JET):
    heat = np.power(np.clip(heat, 0, 1), 0.7)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_c = cv2.applyColorMap(heat_u8, colormap)
    heat_c = cv2.cvtColor(heat_c, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb * (1 - alpha) + heat_c * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_image_for_attn(model, img_path, device, capture, imgsz=1024,
                       conf_thres=0.25):
    """Forward once, return (img_lb_rgb, detections_letterbox, attn_tensor, feat_shape)."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, _, _ = letterbox(img_rgb, new_shape=imgsz, stride=32)
    t = torch.from_numpy(img_lb.transpose(2, 0, 1)).to(device).float().div_(255.0).unsqueeze(0)

    capture.reset()
    pred = parse_yolo26_output(model(t))

    if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
        det = pred[0]
        det = det[det[:, 4] > conf_thres]
    else:
        det = None

    if not capture.captured:
        return img_lb, det, None, None
    # Use the LAST captured attention (deepest in C2PSA).
    attn = capture.captured[-1]
    H, W = capture.feat_shapes[-1]
    return img_lb, det, attn, (H, W)


def draw_detections(img_rgb, detections, class_names, color=(255, 80, 80)):
    out = img_rgb.copy()
    if detections is None:
        return out
    dets = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections
    for d in dets:
        x1, y1, x2, y2, conf, cls = d[:6]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, 2)
        name = class_names.get(int(cls), str(int(cls)))
        cv2.putText(out, f'{name} {conf:.2f}', (int(x1), max(0, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_per_image_fig(img_lb, detections, attn, feat_shape, class_names,
                       name, img_stem, out_dir, max_queries=4, conf_thres=0.25):
    H_img, W_img = img_lb.shape[:2]
    if attn is None or feat_shape is None:
        print(f'  ⚠ no attention captured for {img_stem} — is C2PSA present?')
        return
    H_feat, W_feat = feat_shape
    queries = pick_queries_from_detections(
        detections, H_feat, W_feat, H_img, W_img,
        max_queries=max_queries, conf_thres=conf_thres,
    )
    if not queries:
        print(f'  ⚠ no detections above {conf_thres} on {img_stem} — skipping.')
        return

    n_panels = 1 + len(queries)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5), squeeze=False)
    axes = axes[0]

    axes[0].imshow(draw_detections(img_lb, detections, class_names))
    axes[0].set_title(f'{name}\n{img_stem}', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for qi, (tok_idx, meta) in enumerate(queries):
        cy, cx, cls, conf = meta[:4]
        x1, y1, x2, y2 = meta[4:]
        heat = build_attention_heatmap(attn, tok_idx, H_feat, W_feat, head='mean')
        heat_up = cv2.resize(heat.astype(np.float32), (W_img, H_img))
        over = overlay_heatmap(img_lb, heat_up, alpha=0.6)
        # Mark query location
        cv2.circle(over, (cx, cy), 10, (255, 255, 255), 2)
        cv2.circle(over, (cx, cy), 5, (0, 0, 0), -1)
        # Box of the detection
        cv2.rectangle(over, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cls_name = class_names.get(cls, str(cls))
        axes[qi + 1].imshow(over)
        axes[qi + 1].set_title(
            f'Query {qi+1}: {cls_name} (conf={conf:.2f})\nat feat=({cy*H_feat//H_img},{cx*W_feat//W_img})',
            fontsize=10)
        axes[qi + 1].axis('off')

    fig.suptitle('C2PSA attention: what does the query token attend to?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    slug_name = name.replace(' ', '_').replace('/', '_').lower()
    save = out_dir / f'attention_{slug_name}__{img_stem}.png'
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {save}')


def save_compare_fig(per_image_rows, img_stem, class_names, out_dir):
    """Side-by-side comparison across checkpoints for one image.
    per_image_rows: list of (name, img_lb, detections, attn, feat_shape, queries_list).
    """
    # Find the max number of queries across rows to align columns.
    max_q = max(len(row['queries']) for row in per_image_rows)
    if max_q == 0:
        return
    n_rows = len(per_image_rows)
    n_cols = 1 + max_q
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for r, row in enumerate(per_image_rows):
        name = row['name']
        img_lb = row['img_lb']
        detections = row['detections']
        attn = row['attn']
        feat_shape = row['feat_shape']
        queries = row['queries']

        axes[r, 0].imshow(draw_detections(img_lb, detections, class_names))
        axes[r, 0].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[r, 0].axis('off')

        H_img, W_img = img_lb.shape[:2]
        for qi in range(max_q):
            ax = axes[r, qi + 1]
            if qi < len(queries) and attn is not None:
                tok_idx, meta = queries[qi]
                cy, cx, cls, conf = meta[:4]
                x1, y1, x2, y2 = meta[4:]
                H_feat, W_feat = feat_shape
                heat = build_attention_heatmap(attn, tok_idx, H_feat, W_feat, head='mean')
                heat_up = cv2.resize(heat.astype(np.float32), (W_img, H_img))
                over = overlay_heatmap(img_lb, heat_up, alpha=0.6)
                cv2.circle(over, (cx, cy), 10, (255, 255, 255), 2)
                cv2.circle(over, (cx, cy), 5, (0, 0, 0), -1)
                cv2.rectangle(over, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cls_name = class_names.get(cls, str(cls))
                ax.imshow(over)
                if r == 0:
                    ax.set_title(f'Query {qi+1}\n({cls_name})', fontsize=10)
            else:
                ax.imshow(np.zeros_like(img_lb))
            ax.axis('off')

    fig.suptitle(f'C2PSA attention comparison — {img_stem}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save = out_dir / f'compare_{img_stem}.png'
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {save}')


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def pick_images(images_dir, model, device, capture, n_images, imgsz,
                 conf_thres):
    """Pick images that produce ≥1 detection — ensures attention queries exist."""
    all_files = sorted([f for f in Path(images_dir).iterdir()
                        if f.suffix.lower() in IMAGE_EXTENSIONS])
    # Scan up to ~5× needed to find images with detections.
    pool = all_files[:max(n_images * 5, 30)]
    scored = []
    for f in pool:
        img_lb, det, attn, feat_shape = run_image_for_attn(
            model, f, device, capture, imgsz=imgsz, conf_thres=conf_thres,
        )
        score = 0 if det is None else int(len(det))
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for s, f in scored[:n_images]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--weights', type=str, default='yolo26s.pt')
    ap.add_argument('--data', type=str, default='data.yaml')
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--n-images', type=int, default=6)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--conf-thres', type=float, default=0.25)
    ap.add_argument('--max-queries', type=int, default=4)
    ap.add_argument('--which', choices=['student', 'teacher'], default='student')
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--output', type=str, default='explain_out/rq2_attention')
    opt = ap.parse_args()

    if len(opt.checkpoint) != len(opt.names):
        ap.error('--names count must match --checkpoint count')

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[RQ2-attn] Output → {out_dir}')

    class_names = load_class_names(opt.data)

    # Pick images using the first checkpoint (so all checkpoints visualise
    # the same set of images → fair comparison).
    first_model = load_model_from_ckpt(opt.weights, opt.checkpoint[0], device, which=opt.which)
    first_cap = AttentionCapture(first_model)
    image_files = pick_images(opt.images, first_model, device, first_cap,
                               opt.n_images, opt.imgsz, opt.conf_thres)
    print(f'[RQ2-attn] Selected {len(image_files)} images with detections: '
          f'{[f.stem for f in image_files]}')
    first_cap.remove()
    del first_model, first_cap
    torch.cuda.empty_cache()

    # For each image, collect per-checkpoint attention for a compare figure.
    per_image_rows = {f.stem: [] for f in image_files}

    for ckpt, name in zip(opt.checkpoint, opt.names):
        print(f'\n[RQ2-attn] Checkpoint "{name}"')
        model = load_model_from_ckpt(opt.weights, ckpt, device, which=opt.which)
        capture = AttentionCapture(model)

        for img_path in image_files:
            img_lb, det, attn, feat_shape = run_image_for_attn(
                model, img_path, device, capture,
                imgsz=opt.imgsz, conf_thres=opt.conf_thres,
            )
            if img_lb is None:
                continue
            H_img, W_img = img_lb.shape[:2]
            queries = pick_queries_from_detections(
                det,
                feat_shape[0] if feat_shape else 1,
                feat_shape[1] if feat_shape else 1,
                H_img, W_img,
                max_queries=opt.max_queries, conf_thres=opt.conf_thres,
            ) if feat_shape is not None else []

            save_per_image_fig(img_lb, det, attn, feat_shape, class_names,
                                name, img_path.stem, out_dir,
                                max_queries=opt.max_queries,
                                conf_thres=opt.conf_thres)
            per_image_rows[img_path.stem].append({
                'name': name, 'img_lb': img_lb, 'detections': det,
                'attn': attn, 'feat_shape': feat_shape, 'queries': queries,
            })

        capture.remove()
        del model, capture
        torch.cuda.empty_cache()

    # Compare figures (if multiple checkpoints)
    if len(opt.checkpoint) > 1:
        for img_stem, rows in per_image_rows.items():
            if rows:
                save_compare_fig(rows, img_stem, class_names, out_dir)

    print('\n[RQ2-attn] Done.')


if __name__ == '__main__':
    main()
