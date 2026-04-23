"""Visual sanity check for `utils.copy_paste_small.apply_small_copy_paste`.

What it does
------------
Load `--n-images` training images via PairedMultiDomainDataset with
`copy_paste_small=True`, save side-by-side:

    [original with GT boxes]  |  [augmented with GT + pasted boxes]

Inspect the output JPGs in `test_copy_paste_out/` — pasted patches should:
  * NOT look like sharp rectangles against a foggy/washed-out background
    (if they do → color match is not working).
  * NOT have hard rectangular borders (if they do → Gaussian blend broken).
  * Land in plausible places: persons on sidewalks / road edges, cars on
    the road surface.
  * Not overlap existing GT boxes severely.

Usage (from repo root)
----------------------
  python test_copy_paste_small.py \
      --data data.yaml \
      --n-images 12 \
      --source-real datasets/source_real/source_real/train/images \
      --target-real datasets/target_real/target_real/train/images

Two "sets" are produced — one on source (clear) and one on target (foggy) —
so you can verify color-matching works in BOTH domains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.copy_paste_small import apply_small_copy_paste


def tensor_img_to_bgr_hwc(t):
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.ascontiguousarray(arr[..., ::-1])   # RGB → BGR
    return arr


def draw_boxes(img, bboxes_xywhn, color, thickness=2):
    out = img.copy()
    H, W = out.shape[:2]
    if isinstance(bboxes_xywhn, torch.Tensor):
        bboxes_xywhn = bboxes_xywhn.cpu().numpy()
    for box in np.asarray(bboxes_xywhn):
        cx, cy, bw, bh = [float(v) for v in box[:4]]
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def run(name, img_path_dir, data_dict, out_dir, imgsz,
        n_images, small_thr, max_copies, prob, seed):
    """Load a YOLODataset, pick random indices, run augmentation, save figures."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.cfg import get_cfg

    print(f'\n{"=" * 70}')
    print(f'  {name}  ({img_path_dir})')
    print('=' * 70)

    cfg = get_cfg()
    ds = YOLODataset(
        img_path=img_path_dir,
        imgsz=imgsz, augment=True, cache=False,
        hyp=cfg, data=data_dict, task='detect', stride=32,
        rect=True, single_cls=False, pad=0.5, classes=None,
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_images, len(ds)), replace=False)
    indices = sorted(indices.tolist())

    out_sub = out_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)

    # Copy-paste does not modify in place safely if we want to compare;
    # run over the original dict's bboxes/cls both BEFORE (copy) and
    # AFTER.
    for idx in indices:
        sample = ds[int(idx)]
        img_before = tensor_img_to_bgr_hwc(sample['img'])
        bboxes_before = sample['bboxes'].clone() if isinstance(sample['bboxes'], torch.Tensor) \
            else np.array(sample['bboxes'], copy=True)
        cls_before = sample['cls'].clone() if isinstance(sample['cls'], torch.Tensor) \
            else np.array(sample['cls'], copy=True)
        n_before = len(bboxes_before)

        # Run augmentation with prob=1.0 to guarantee application on every sample.
        aug = apply_small_copy_paste(
            {
                'img': sample['img'].clone() if isinstance(sample['img'], torch.Tensor)
                    else sample['img'].copy(),
                'cls': cls_before.clone() if isinstance(cls_before, torch.Tensor)
                    else cls_before.copy(),
                'bboxes': bboxes_before.clone() if isinstance(bboxes_before, torch.Tensor)
                    else bboxes_before.copy(),
                'batch_idx': sample.get('batch_idx', torch.zeros(n_before)),
            },
            small_thr=small_thr, max_copies=max_copies, prob=1.0,   # force
        )
        img_after = tensor_img_to_bgr_hwc(aug['img'])
        bboxes_after = aug['bboxes']
        n_after = len(bboxes_after)
        n_pasted = n_after - n_before

        # Draw: GT green on BEFORE, (GT green + pasted red) on AFTER
        img_before_draw = draw_boxes(img_before, bboxes_before, color=(0, 255, 0))

        img_after_draw = img_after.copy()
        if n_pasted > 0:
            gt_part = bboxes_after[:n_before] if isinstance(bboxes_after, torch.Tensor) \
                else bboxes_after[:n_before]
            paste_part = bboxes_after[n_before:] if isinstance(bboxes_after, torch.Tensor) \
                else bboxes_after[n_before:]
            img_after_draw = draw_boxes(img_after_draw, gt_part, color=(0, 255, 0))
            img_after_draw = draw_boxes(img_after_draw, paste_part, color=(0, 0, 255))
        else:
            img_after_draw = draw_boxes(img_after_draw, bboxes_after, color=(0, 255, 0))

        # Side-by-side.
        if img_before_draw.shape == img_after_draw.shape:
            combined = np.hstack([img_before_draw, img_after_draw])
        else:
            combined = img_after_draw   # fallback

        # Title bar
        H = combined.shape[0]
        cv2.putText(combined, f'idx={idx}  BEFORE (green=GT)',
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(combined, f'AFTER  GT(green)+PASTED(red, n={n_pasted})',
                    (combined.shape[1] // 2 + 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        save = out_sub / f'cp_{idx:05d}.jpg'
        cv2.imwrite(str(save), combined)
        print(f'  ✓ idx={idx:5d}  n_gt={n_before:3d}  n_pasted={n_pasted:2d}  → {save}')


def run_paired(out_dir, data_dict, imgsz, n_images,
               small_thr, max_copies, prob, seed):
    """Verify the actual TRAINING path: PairedMultiDomainDataset with
    copy_paste_small=True → source_real and source_fake should both receive
    the SAME pastes at the SAME positions, with CLEAR Cityscapes pixel
    content in both (source_fake's patches get colour-matched to fog)."""
    from fusion_da import PairedMultiDomainDataset
    from ultralytics.cfg import get_cfg

    print(f'\n{"=" * 70}')
    print(f'  PAIRED mode  (source_real ↔ source_fake via PairedMultiDomainDataset)')
    print('=' * 70)

    root = Path(data_dict.get('path', ''))

    def resolve(key):
        v = data_dict.get(key)
        if isinstance(v, list):
            v = v[0]
        return str(root / v)

    cfg = get_cfg()
    pds = PairedMultiDomainDataset(
        source_real_path=resolve('train_source_real'),
        source_fake_path=resolve('train_source_fake'),
        target_real_path=resolve('train_target_real'),
        target_fake_path=resolve('train_target_fake'),
        imgsz=imgsz, augment=True, hyp=cfg, data=data_dict, stride=32,
        copy_paste_small=True,
        copy_paste_max_copies=max_copies,
        copy_paste_small_thr=small_thr,
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pds), size=min(n_images, len(pds)), replace=False)

    out_sub = out_dir / 'paired_source'
    out_sub.mkdir(parents=True, exist_ok=True)

    for idx in sorted(indices.tolist()):
        item = pds[int(idx)]
        sr = item['source_real']
        sf = item['source_fake']

        img_r = tensor_img_to_bgr_hwc(sr['img'])
        img_f = tensor_img_to_bgr_hwc(sf['img'])
        bb_r = sr['bboxes']
        bb_f = sf['bboxes']

        n_r = len(bb_r)
        n_f = len(bb_f)

        img_r_draw = draw_boxes(img_r, bb_r, color=(0, 255, 0))
        img_f_draw = draw_boxes(img_f, bb_f, color=(0, 0, 255))

        if img_r_draw.shape == img_f_draw.shape:
            combined = np.hstack([img_r_draw, img_f_draw])
        else:
            combined = img_r_draw

        cv2.putText(combined, f'idx={idx}  source_real  (green=all labels)',
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(combined, f'source_fake  (red=all labels)',
                    (combined.shape[1] // 2 + 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        save = out_sub / f'pair_{idx:05d}.jpg'
        cv2.imwrite(str(save), combined)

        # Correctness: after paired copy-paste, bboxes of source_real and
        # source_fake should match (same length, same coordinates).
        sync_status = 'SYNCED' if n_r == n_f else f'DESYNC n_r={n_r} n_f={n_f}'
        print(f'  {"✓" if n_r == n_f else "✗"}  idx={idx:5d}  '
              f'n_labels={n_r}  {sync_status}  → {save}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=str, default='data.yaml')
    ap.add_argument('--source-real', type=str, default=None,
                    help='Override source-real train images dir. Default: from data.yaml.')
    ap.add_argument('--target-real', type=str, default=None,
                    help='Override target-real train images dir. Default: from data.yaml.')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--n-images', type=int, default=12)
    ap.add_argument('--small-thr', type=float, default=32.0)
    ap.add_argument('--max-copies', type=int, default=3)
    ap.add_argument('--prob', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=str, default='test_copy_paste_out')
    ap.add_argument('--paired-only', action='store_true',
                    help='Only run the paired test (skip standalone source/target).')
    opt = ap.parse_args()

    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)

    root = Path(data_dict.get('path', ''))

    def resolve(key, override):
        if override:
            return override
        v = data_dict.get(key)
        if isinstance(v, list):
            v = v[0]
        return str(root / v)

    src_dir = resolve('train_source_real', opt.source_real)
    tgt_dir = resolve('train_target_real', opt.target_real)

    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not opt.paired_only:
        run('source_real', src_dir, data_dict, out_dir, opt.imgsz,
            opt.n_images, opt.small_thr, opt.max_copies, opt.prob, opt.seed)
        run('target_real', tgt_dir, data_dict, out_dir, opt.imgsz,
            opt.n_images, opt.small_thr, opt.max_copies, opt.prob, opt.seed + 1)

    # Paired mode — reflects actual training behaviour.
    run_paired(out_dir, data_dict, opt.imgsz, opt.n_images,
               opt.small_thr, opt.max_copies, opt.prob, opt.seed + 2)

    print(f'\n✅ Outputs in {out_dir}/')
    print('   standalone/   — quick sanity on single-image crop-paste')
    print('   paired_source/ — full paired pipeline. LEFT (green) = source_real,')
    print('                    RIGHT (red) = source_fake. Labels must match 1:1,')
    print('                    and pasted patches should look CLEAR on real side,')
    print('                    FOG-tinted (but same shape) on fake side.')


if __name__ == '__main__':
    main()
