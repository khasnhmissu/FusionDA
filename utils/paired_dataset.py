"""PairedAugDataset — RNG-synced twin YOLODataset loader.

Why
---
`YOLODataset.__getitem__(idx)` applies random augmentation on every call:
mosaic picks 3 random partners, flip decides randomly, scale/translate/rotate
are sampled from distributions. Calling two independent YOLODataset instances
at the same idx therefore produces two DIFFERENTLY augmented tensors — even
if the underlying images represent the same scene (e.g. source_real and
source_fake, where fake = real + fog).

This breaks paired losses (consistency, distillation, paired source_fake
detection) because the label/prediction coordinates no longer align at the
pixel level.

How
---
Around each twin __getitem__ call, seed Python / NumPy / Torch RNGs with
the same integer. Ultralytics transforms (Mosaic, RandomPerspective,
RandomFlip, HSV, copy-paste, mixup) all use `random.*` or `np.random.*`.
Identical seed → identical random draws → identical augmentation.

The worker's RNG state is saved/restored around the seeded block so that
our seeding does not disrupt the dataloader worker's own random sequence
(e.g. shuffling handled by the parent DataLoader).

Invariants / expectations for the two folders
---------------------------------------------
- Exactly the same filenames on both sides (enforced by `_assert_paired`).
- Same image dimensions per filename (required for mosaic to produce the
  same composite).
- Same YOLO labels per filename (fog augmentation must not move objects).

Caveats
-------
- `rect=True` is supported: YOLODataset sorts im_files by aspect ratio and
  computes batch_shapes. Twin datasets built from identical filenames with
  identical image dimensions MUST end up with the same `im_files` order and
  the same `batch_shapes`, otherwise idx→shape mapping would drift and
  pairing would silently break. `_assert_paired` verifies both.
- Albumentations (Blur / MedianBlur / ToGray / CLAHE at p=0.01 each) uses
  its own RNG that we do not touch. These are colour-only transforms
  firing on rare frames; a miss of alignment on them only perturbs the
  fog-like pixel difference slightly and does not affect geometry.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.dataset import YOLODataset


class PairedAugDataset(Dataset):
    """Twin YOLODataset with augmentation synchronised via RNG seeding."""

    def __init__(self, real_path: str, fake_path: str, **yolo_ds_kwargs):
        """Build two YOLODatasets (real / fake) with identical kwargs.

        Args:
            real_path: path passed to the real YOLODataset's img_path.
            fake_path: path passed to the fake YOLODataset's img_path.
            **yolo_ds_kwargs: forwarded to both YOLODataset constructors
                (imgsz, augment, cache, hyp, data, task, stride, rect,
                single_cls, pad, classes, ...).
        """
        self.real_ds = YOLODataset(img_path=real_path, **yolo_ds_kwargs)
        self.fake_ds = YOLODataset(img_path=fake_path, **yolo_ds_kwargs)
        self._assert_paired()

    def _assert_paired(self):
        """Fail loudly if real/fake folders are not exact 1:1 by filename.

        When `rect=True`, YOLODataset also sorts `im_files` by aspect ratio
        and computes `batch_shapes`. Twin datasets built from identical
        filenames with identical image dimensions MUST end up with the same
        `im_files` order and the same `batch_shapes`, otherwise idx→shape
        mapping would drift and pairing would silently break.
        """
        real_stems = [Path(f).stem for f in self.real_ds.im_files]
        fake_stems = [Path(f).stem for f in self.fake_ds.im_files]

        if len(real_stems) != len(fake_stems):
            raise AssertionError(
                f"[PairedAugDataset] Count mismatch: "
                f"real={len(real_stems)} vs fake={len(fake_stems)}. "
                f"Both folders must contain exactly the same filenames."
            )

        if real_stems != fake_stems:
            if sorted(real_stems) == sorted(fake_stems):
                raise AssertionError(
                    "[PairedAugDataset] Filenames match but ORDER differs "
                    "after YOLODataset init — this drifts pairing. "
                    "Likely cause: image dimensions differ between real "
                    "and fake so `rect=True` sorts them differently. "
                    "Verify dims with `identify -format '%wx%h\\n' *.jpg`."
                )
            missing_in_fake = sorted(set(real_stems) - set(fake_stems))
            missing_in_real = sorted(set(fake_stems) - set(real_stems))
            raise AssertionError(
                f"[PairedAugDataset] Filename sets differ:\n"
                f"  in real but not fake ({len(missing_in_fake)}): "
                f"{missing_in_fake[:5]}{'...' if len(missing_in_fake) > 5 else ''}\n"
                f"  in fake but not real ({len(missing_in_real)}): "
                f"{missing_in_real[:5]}{'...' if len(missing_in_real) > 5 else ''}"
            )

        # rect=True sanity check: batch_shapes must agree, otherwise
        # letterboxing would differ and pairing breaks.
        real_bs = getattr(self.real_ds, 'batch_shapes', None)
        fake_bs = getattr(self.fake_ds, 'batch_shapes', None)
        if real_bs is not None and fake_bs is not None:
            import numpy as _np
            real_bs_arr = _np.asarray(real_bs)
            fake_bs_arr = _np.asarray(fake_bs)
            if real_bs_arr.shape != fake_bs_arr.shape or not _np.array_equal(
                real_bs_arr, fake_bs_arr
            ):
                raise AssertionError(
                    "[PairedAugDataset] batch_shapes mismatch between real "
                    "and fake (rect=True is brittle when image dims drift). "
                    f"real[:3]={real_bs_arr[:3].tolist()} "
                    f"fake[:3]={fake_bs_arr[:3].tolist()}. "
                    "Either make real/fake dims identical, or pass rect=False."
                )

        rect_mode = 'rect=True (aspect-sorted, matched batch_shapes)' \
            if real_bs is not None else 'rect=False (all shapes=imgsz)'
        print(f"[PairedAugDataset] {len(real_stems)} paired samples "
              f"verified [{rect_mode}].")

    def __len__(self):
        return len(self.real_ds)

    def __getitem__(self, idx):
        # 1. Draw a single seed for this __getitem__ invocation.
        seed = random.randint(0, 2**31 - 1)

        # 2. Snapshot worker RNG state so seeding does not derail the
        #    DataLoader worker's subsequent random sequence.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        try:
            # 3a. Seed → real side.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            real = self.real_ds[idx]

            # 3b. Reseed with the SAME seed → fake side.
            #     Mosaic partners, flip, scale, translate, rotate, shear
            #     all get identical random draws → identical augmentation.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            fake = self.fake_ds[idx]
        finally:
            # 4. Restore worker RNG — do NOT leak our seeding outward.
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

        return {'real': real, 'fake': fake}
