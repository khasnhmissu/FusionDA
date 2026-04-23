"""RQ2a — Feature-space domain alignment: UMAP/t-SNE + MMD.

Answers:
  * Do the student's backbone features for source vs target images overlap
    after training (= domain-invariant)?
  * Quantify the gap with MMD (Maximum Mean Discrepancy). DA training should
    shrink MMD toward 0.

Mechanism
---------
  * Hook the backbone-end layer (C2PSA for YOLO26, SPPF for YOLOv8). This is
    the same layer the training pipeline uses for GRL / feature-KD.
  * For each image: forward (no_grad), GAP the feature map → (C,) vector.
  * Collect N vectors from source val, N vectors from target val.
  * Compute:
     - UMAP 2D projection → scatter plot coloured by domain.
     - (optional) t-SNE 2D projection for paper sanity-check.
     - MMD (RBF, multi-kernel) between source and target vector clouds.

Inputs
------
  --checkpoint        One or more training checkpoints (best.pt, last.pt,
                      checkpoint_ep*.pt). Supply multiple to compare side-by-side.
  --names             Display label per checkpoint.
  --source-images     Dir with source-domain val images.
  --target-images     Dir with target-domain val images.
  --n-per-domain      How many images to sample from each domain. Default 500.
  --imgsz             Default 1024.
  --device            0 | cpu. Default 0.
  --output            Output dir. Default ./explain_out/rq2_feature.

Outputs
-------
  umap_<name>.png              Scatter of source vs target features for each checkpoint
  umap_grid.png                All checkpoints in a grid (easy paper figure)
  tsne_<name>.png              Optional t-SNE (enable with --tsne)
  mmd.csv                      MMD source↔target per checkpoint
  summary.json                 All numbers

Examples
--------
Single model (the final student):
  python explain/rq2_feature_umap.py \\
      --checkpoint runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "DA (best)" \\
      --source-images datasets/source_real/source_real/val/images \\
      --target-images datasets/target_real/target_real/val/images \\
      --n-per-domain 500 --imgsz 1024

Compare before/after DA (e.g. baseline vs full DA model):
  python explain/rq2_feature_umap.py \\
      --checkpoint runs/ablation/baseline/weights/best.pt \\
                   runs/ablation/paired_full_40ep/weights/best.pt \\
      --names "Baseline (no DA)" "Paired DA (full)" \\
      --source-images datasets/source_real/source_real/val/images \\
      --target-images datasets/target_real/target_real/val/images
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # type: ignore
    IMAGE_EXTENSIONS, letterbox, load_model_from_ckpt,
)

# Reuse existing utility for MMD (avoids re-implementing the multi-kernel version).
from utils.explainability.mmd import compute_mmd  # type: ignore
# Reuse last-backbone-layer auto-detection (same layer training uses).
from domain_adaptation import find_last_backbone_layer  # type: ignore


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class BackboneFeatureExtractor:
    """Hooks the backbone-end layer and returns GAP-pooled features (B, C)."""

    def __init__(self, model):
        self.features = None
        layer_idx = find_last_backbone_layer(model)
        layers = model.model if hasattr(model, 'model') else model
        target = layers[layer_idx]
        self.hook = target.register_forward_hook(self._hook_fn)
        self.layer_name = target.__class__.__name__
        self.layer_idx = layer_idx
        print(f'[FeatureExtractor] Hook at layer {layer_idx} ({self.layer_name})')

    def _hook_fn(self, module, input, output):
        self.features = output.detach()

    @torch.no_grad()
    def extract(self, img_tensor):
        """Forward once, return GAP-pooled features (B, C) as numpy."""
        self.features = None
        _ = img_tensor  # trigger forward happens outside
        return None

    def pull(self):
        """Pop cached features and GAP-pool → (B, C) numpy."""
        if self.features is None:
            return None
        f = self.features
        # (B, C, H, W) → (B, C)
        if f.ndim == 4:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        elif f.ndim == 3:
            f = f.mean(dim=-1)
        self.features = None
        return f.cpu().numpy()

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


@torch.no_grad()
def extract_features(model, image_files, device, imgsz=1024, batch=4):
    """Forward a batch of images, pool backbone-end features."""
    extractor = BackboneFeatureExtractor(model)
    all_feats = []
    batch_imgs = []
    batch_count = 0
    import cv2

    for img_path in tqdm(image_files, desc='  extract'):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _, _ = letterbox(img_rgb, new_shape=imgsz, stride=32)
        t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        batch_imgs.append(t)
        batch_count += 1
        if batch_count == batch:
            x = torch.stack(batch_imgs).to(device)
            _ = model(x)
            feats = extractor.pull()
            if feats is not None:
                all_feats.append(feats)
            batch_imgs, batch_count = [], 0

    if batch_imgs:
        x = torch.stack(batch_imgs).to(device)
        _ = model(x)
        feats = extractor.pull()
        if feats is not None:
            all_feats.append(feats)

    extractor.remove()
    return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap_single(source_feat, target_feat, title, save_path,
                     mmd_val=None, n_neighbors=15, min_dist=0.1, seed=42):
    """UMAP 2D scatter of source vs target features."""
    try:
        import umap
    except ImportError:
        print('  ⚠ umap-learn not installed. pip install umap-learn')
        return

    X = np.concatenate([source_feat, target_feat], axis=0).astype(np.float32)
    domain = np.array([0] * len(source_feat) + [1] * len(target_feat))

    # Standardise per-feature for stable UMAP.
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric='euclidean', random_state=seed)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    for d, color, label, alpha in [
        (0, '#2E86AB', f'Source (n={len(source_feat)})', 0.55),
        (1, '#F18F01', f'Target (n={len(target_feat)})', 0.55),
    ]:
        mask = domain == d
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=label,
                   alpha=alpha, s=18, edgecolors='none')
    subtitle = f'\nMMD(src↔tgt) = {mmd_val:.4f}' if mmd_val is not None else ''
    ax.set_title(f'{title}{subtitle}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlabel('UMAP dim 1')
    ax.set_ylabel('UMAP dim 2')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_umap_grid(all_embeddings, output_dir):
    """All checkpoints side-by-side in one figure."""
    n = len(all_embeddings)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), squeeze=False)
    axes = axes[0]

    for ax, (name, payload) in zip(axes, all_embeddings.items()):
        emb = payload['emb']
        n_src = payload['n_src']
        mmd = payload.get('mmd')
        ax.scatter(emb[:n_src, 0], emb[:n_src, 1], c='#2E86AB', label='Source',
                   alpha=0.55, s=18, edgecolors='none')
        ax.scatter(emb[n_src:, 0], emb[n_src:, 1], c='#F18F01', label='Target',
                   alpha=0.55, s=18, edgecolors='none')
        subtitle = f'\nMMD = {mmd:.4f}' if mmd is not None else ''
        ax.set_title(f'{name}{subtitle}', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Backbone feature space — source vs target',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save = output_dir / 'umap_grid.png'
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ {save}')


def plot_tsne(source_feat, target_feat, title, save_path, seed=42):
    from sklearn.manifold import TSNE
    X = np.concatenate([source_feat, target_feat], axis=0).astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std
    emb = TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(emb[:len(source_feat), 0], emb[:len(source_feat), 1],
               c='#2E86AB', label='Source', alpha=0.55, s=18, edgecolors='none')
    ax.scatter(emb[len(source_feat):, 0], emb[len(source_feat):, 1],
               c='#F18F01', label='Target', alpha=0.55, s=18, edgecolors='none')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--weights', type=str, default='yolo26s.pt')
    ap.add_argument('--source-images', type=str, required=True)
    ap.add_argument('--target-images', type=str, required=True)
    ap.add_argument('--n-per-domain', type=int, default=500)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--which', choices=['student', 'teacher'], default='student',
                    help="State to load. 'teacher' only works with checkpoint_ep*.pt.")
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--tsne', action='store_true',
                    help='Also compute t-SNE (slower, for paper sanity-check).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='explain_out/rq2_feature')
    opt = ap.parse_args()

    if len(opt.checkpoint) != len(opt.names):
        ap.error('--names count must match --checkpoint count')

    device = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[RQ2-feat] Output → {out_dir}')

    rng = np.random.default_rng(opt.seed)

    def pick(dir_, n):
        files = sorted([f for f in Path(dir_).iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        if len(files) <= n:
            return files
        idx = rng.choice(len(files), size=n, replace=False)
        return [files[i] for i in sorted(idx)]

    src_files = pick(opt.source_images, opt.n_per_domain)
    tgt_files = pick(opt.target_images, opt.n_per_domain)
    print(f'[RQ2-feat] Source: {len(src_files)}  Target: {len(tgt_files)}')

    all_embeddings = {}
    mmd_rows = []

    for ckpt, name in zip(opt.checkpoint, opt.names):
        print(f'\n[RQ2-feat] Checkpoint "{name}" = {ckpt}')
        model = load_model_from_ckpt(opt.weights, ckpt, device, which=opt.which)

        print('  Source features…')
        src_feat = extract_features(model, src_files, device, imgsz=opt.imgsz, batch=opt.batch)
        print('  Target features…')
        tgt_feat = extract_features(model, tgt_files, device, imgsz=opt.imgsz, batch=opt.batch)
        print(f'  src feat shape={src_feat.shape}  tgt feat shape={tgt_feat.shape}')

        # MMD (multi-kernel RBF, from utils/explainability/mmd.py)
        mmd = None
        try:
            s_t = torch.from_numpy(src_feat).float()
            t_t = torch.from_numpy(tgt_feat).float()
            mmd = float(compute_mmd(s_t, t_t))
            print(f'  MMD(source, target) = {mmd:.4f}')
        except Exception as e:
            print(f'  ⚠ MMD computation failed: {e}')
        mmd_rows.append((name, ckpt, len(src_feat), len(tgt_feat), mmd))

        # UMAP
        try:
            import umap
            X = np.concatenate([src_feat, tgt_feat], axis=0).astype(np.float32)
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mean) / std
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                 metric='euclidean', random_state=opt.seed)
            emb = reducer.fit_transform(X)
            all_embeddings[name] = {
                'emb': emb, 'n_src': len(src_feat), 'n_tgt': len(tgt_feat),
                'mmd': mmd,
            }
            slug = name.replace(' ', '_').replace('/', '_').lower()
            save = out_dir / f'umap_{slug}.png'
            plot_umap_single(src_feat, tgt_feat, title=f'{name}',
                             save_path=save, mmd_val=mmd, seed=opt.seed)
            print(f'  ✅ {save}')
        except ImportError:
            print('  ⚠ umap-learn not installed — skipping UMAP plot.')

        if opt.tsne:
            slug = name.replace(' ', '_').lower()
            save = out_dir / f'tsne_{slug}.png'
            plot_tsne(src_feat, tgt_feat, title=f'{name} — t-SNE',
                      save_path=save, seed=opt.seed)
            print(f'  ✅ {save}')

        del model
        torch.cuda.empty_cache()

    if len(all_embeddings) >= 1:
        plot_umap_grid(all_embeddings, out_dir)

    # Dump CSV + JSON
    csv_path = out_dir / 'mmd.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'checkpoint', 'n_source', 'n_target', 'mmd'])
        for row in mmd_rows:
            w.writerow([row[0], row[1], row[2], row[3],
                        f'{row[4]:.6f}' if row[4] is not None else ''])
    print(f'  ✅ {csv_path}')

    summary = {
        name: {'checkpoint': ckpt, 'n_source': n_src, 'n_target': n_tgt, 'mmd': mmd}
        for (name, ckpt, n_src, n_tgt, mmd) in mmd_rows
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'  ✅ {out_dir / "summary.json"}')
    print('\n[RQ2-feat] Done.')


if __name__ == '__main__':
    main()
