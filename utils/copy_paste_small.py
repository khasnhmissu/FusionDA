"""Small-object copy-paste augmentation for Cityscapes-style dashcam scenes.

Method
------
Follows the spirit of Kisantal et al. 2019 — "Augmentation for Small
Object Detection" (ICAR, ~500 citations, proven SOTA-simple baseline with
+9.7% AP_small on COCO) — with four practical adaptations for foggy
dashcam scenes after mosaic + letterbox:

  1. **Source = medium / large objects, target = small.** Kisantal clones
     already-small objects; in our foggy data those small instances are
     noise-limited. We instead sample from objects whose max-side is in
     [small_thr, 4·small_thr] (i.e. medium / large), extract their crop,
     and **downsample** to a target size randomly in [24, 40] px. This
     yields CLEAN small-object samples from high-detail sources. 24 px
     lower bound keeps pastes above the "dot" regime where no detection
     signal survives; 40 px upper bound stays inside the small band.
     Falls back to small-source only when the image has no medium/large.
  2. **Content-area aware placement.** Letterbox / mosaic fills unused
     area with neutral gray (114,114,114). We detect the non-gray bounding
     box and restrict pastes to within — so labels never end up on the
     padding.
  3. **Perspective-aware vertical placement.** Small objects in a
     dashcam view cluster near the vanishing point (~image centre
     horizontally) and appear at the horizon line. Zone is narrow in x
     (25-75% of content) and y depends on the target size:
         size 20 (far) → y ∈ [35%, 50%] (near horizon)
         size 45 (close) → y ∈ [50%, 65%] (mid frame)
  4. **Fog-safe compositing.** Two tricks to avoid the "sharp rectangle
     on fog" shortcut: per-channel mean colour match to the destination
     region, and Gaussian α-blend at the boundary (α≈1 centre → 0 edge).

Additional cheap regularisation:
  - Horizontal flip of the cropped object with probability 0.5.
  - Total pastes per image in [1, max_copies] (not per source object,
    so the paste count stays bounded regardless of how many source
    candidates the image contains).

Compatibility notes
-------------------
* This augment is applied AFTER the rest of the YOLODataset pipeline (post-
  mosaic). Placement uses image-relative zones which still work because
  persons/cars remain in the same rough image regions after mosaic.
* **Paired-dataset safe**: when called with an explicit `seed`, every
  random decision (source object index, target position, scale jitter,
  flip, prob check) becomes deterministic. Call the function twice with
  the SAME seed on the two twin images (source_real and source_fake) and
  both will receive identical pastes at identical positions, differing
  only in the pixel values of the cropped object (clear vs foggy) — so
  pair-pixel alignment is preserved and consistency loss can stay on.
  See `PairedMultiDomainDataset.__getitem__` for the calling pattern.

Usage:
    from utils.copy_paste_small import apply_small_copy_paste
    # standalone (per-image independent):
    batch = apply_small_copy_paste(batch, small_thr=32.0, max_copies=3)
    # paired (real + fake get the same pastes):
    seed = random.randint(0, 2**31 - 1)
    real = apply_small_copy_paste(real, seed=seed, ...)
    fake = apply_small_copy_paste(fake, seed=seed, ...)
"""

import random
from typing import Optional

import numpy as np
import torch


def _xywhn_to_xyxy(bboxes_xywhn, img_h, img_w):
    """Convert normalized xywh center to pixel xyxy."""
    cx = bboxes_xywhn[:, 0] * img_w
    cy = bboxes_xywhn[:, 1] * img_h
    w  = bboxes_xywhn[:, 2] * img_w
    h  = bboxes_xywhn[:, 3] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _xyxy_to_xywhn(boxes_xyxy, img_h, img_w):
    """Convert pixel xyxy to normalized xywh center."""
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / img_w
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / img_h
    w  = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
    h  = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
    return torch.stack([cx, cy, w, h], dim=-1)


def _compute_iou_xyxy(box, boxes):
    """Compute IoU between one box and multiple boxes (all in xyxy pixel format).
    
    Args:
        box:   (4,) single box [x1, y1, x2, y2]
        boxes: (N, 4) existing boxes
    Returns:
        ious: (N,)
    """
    if boxes.numel() == 0:
        return torch.tensor([], device=box.device)
    
    inter_x1 = torch.max(box[0], boxes[:, 0])
    inter_y1 = torch.max(box[1], boxes[:, 1])
    inter_x2 = torch.min(box[2], boxes[:, 2])
    inter_y2 = torch.min(box[3], boxes[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6
    
    return inter / union


def _find_content_area(img_hwc, padding_val=114, tol=8):
    """Bounding box (x1, y1, x2, y2) of the non-padding region.

    Letterbox / mosaic outputs fill unused area with a neutral gray
    (default 114 in Ultralytics). Without masking this out, pastes
    land on the padding and carry labels outside the real scene.
    """
    if img_hwc is None or img_hwc.size == 0:
        return 0, 0, 0, 0
    if img_hwc.ndim == 3:
        diff = np.abs(img_hwc.astype(np.int16) - padding_val).max(axis=-1)
    else:
        diff = np.abs(img_hwc.astype(np.int16) - padding_val)
    mask = diff > tol
    if not mask.any():
        return 0, 0, img_hwc.shape[1], img_hwc.shape[0]
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _perspective_zone(cls_id, content_box, target_size,
                      size_min=24.0, size_max=40.0):
    """Placement zone that concentrates small pastes near the dashcam
    vanishing point, with perspective-consistent y.

    Dashcam geometry:
      * Vanishing point ~ image centre → distant (small) objects cluster
        in the horizontal middle.
      * Road plane sits BELOW the horizon (~50 % from top) → all pastes
        land at y ≥ 50 %, with smaller (far) objects just below horizon
        and larger (closer) objects further down on the road surface.

    Mapping (linear in `target_size`):
      size=24  → y ∈ [50%, 60%]  (far, right below horizon)
      size=40  → y ∈ [60%, 72%]  (closer, mid-road)

    Returns (x_min, x_max, y_min, y_max) in absolute pixel coordinates,
    all inside `content_box = (cx1, cy1, cx2, cy2)`.
    """
    cx1, cy1, cx2, cy2 = content_box
    cw = max(cx2 - cx1, 1)
    ch = max(cy2 - cy1, 1)

    # x ∈ [35%, 65%] of content — tight band around the vanishing point.
    # Distant small objects almost always sit near the road vanishing
    # point in a dashcam view, so a narrow 30 %-wide centre band is both
    # realistic and avoids pasting onto buildings / sidewalks far from
    # the road.
    x_min = cx1 + int(cw * 0.35)
    x_max = cx1 + int(cw * 0.65)

    # y band sits BELOW the horizon line (y ≥ 50 % of content) and shifts
    # further down as target size grows (perspective: smaller = farther =
    # just below horizon; larger = closer = further down on the road).
    t = float(target_size - size_min) / max(size_max - size_min, 1e-6)
    t = max(0.0, min(1.0, t))
    y_band_min_top = 0.50   # smallest target: y starts right at horizon
    y_band_min_bot = 0.60
    y_band_max_top = 0.60   # largest target: y starts further down
    y_band_max_bot = 0.72
    y_top = y_band_min_top + t * (y_band_max_top - y_band_min_top)
    y_bot = y_band_min_bot + t * (y_band_max_bot - y_band_min_bot)
    y_min = cy1 + int(ch * y_top)
    y_max = cy1 + int(ch * y_bot)

    # Sanity clamp
    x_min = max(cx1, min(x_min, cx2 - 1))
    x_max = max(x_min + 1, min(x_max, cx2))
    y_min = max(cy1, min(y_min, cy2 - 1))
    y_max = max(y_min + 1, min(y_max, cy2))
    return x_min, x_max, y_min, y_max


def _gaussian_alpha(h, w, sigma_ratio=3.0):
    """Soft alpha mask — 1 at the centre, fades toward 0 at the edges.

    Used for blending a pasted rectangular crop over a background without
    a hard edge artefact. σ is chosen so that α ≈ 0.05 at the corner and
    α ≈ 1.0 at the centre.

    Returns
    -------
    alpha : np.ndarray (h, w, 1) float32 in [0, 1]
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma = max(min(h, w) / sigma_ratio, 1.0)
    alpha = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)[..., None]


def _color_match(crop, target_region):
    """Shift `crop`'s per-channel mean to match `target_region`'s mean.

    Rationale: a clear-domain RGB crop pasted into a foggy scene would
    otherwise show up as a sharp, highly-saturated rectangle → the
    detector can latch onto the contrast as a shortcut feature rather
    than the object shape. Matching per-channel mean is a cheap way to
    pull the pasted pixels into the local colour palette.
    """
    if target_region.size == 0 or crop.size == 0:
        return crop
    c_mean = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    t_mean = target_region.reshape(-1, target_region.shape[-1]).mean(axis=0)
    shifted = crop.astype(np.float32) + (t_mean - c_mean)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_small_copy_paste(
    batch_item: dict,
    small_thr: float = 32.0,
    max_copies: int = 3,
    max_overlap_iou: float = 0.0,
    prob: float = 0.5,
    seed: Optional[int] = None,
    source_image=None,
) -> dict:
    """Apply Kisantal-style small-object copy-paste (with fog-safe adaptations).

    Args
    ----
    batch_item : dict with keys 'img' ([C,H,W] uint8 tensor OR [H,W,C] ndarray),
                 'cls' (N,1), 'bboxes' (N,4 xywhn), 'batch_idx', etc.
                 The paste DESTINATION — this image is the one modified.
    small_thr  : max side length (pixels) for an object to be considered "small".
    max_copies : maximum total pastes per image (1..max_copies attempts).
    max_overlap_iou : max IoU allowed with any existing or already-pasted box.
                 Default 0.0 → strict non-overlap (any pixel touch is rejected),
                 which matches the intent "paste only on free road surface,
                 never on top of an existing labelled object".
    prob       : probability of applying augmentation at all.
    seed       : if given, drives every randomised decision — two invocations
                 with the same seed on two twin images (e.g. source_real vs
                 source_fake) produce identical paste plans, preserving
                 pair-pixel alignment.
    source_image : optional CHW tensor / HWC ndarray from which PIXEL CROPS are
                 extracted (instead of `batch_item['img']`). The function
                 still uses `batch_item['bboxes']` to locate source object
                 regions, so `source_image` must have the SAME shape as
                 `batch_item['img']` and the SAME bbox layout (guaranteed
                 by PairedAugDataset for twin source_real/source_fake).
                 Typical use: always pass source_real's image so that both
                 source_real AND source_fake paste calls receive CLEAR
                 (clean-domain) object pixels, which are then colour-matched
                 to their respective destination palettes.

    Returns the augmented `batch_item` dict (modified in place).
    """
    rng = random.Random(seed) if seed is not None else random

    # Random skip.
    if rng.random() > prob:
        return batch_item
    
    img = batch_item.get('img')
    bboxes = batch_item.get('bboxes')   # (N, 4) normalized xywhn
    cls = batch_item.get('cls')         # (N, 1) or (N,)
    
    if img is None or bboxes is None or cls is None:
        return batch_item
    if len(bboxes) == 0:
        return batch_item
    
    # img shape: (C, H, W) uint8 tensor
    if isinstance(img, torch.Tensor):
        _, img_h, img_w = img.shape
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] <= 4:  # CHW
            _, img_h, img_w = img.shape
        else:  # HWC
            img_h, img_w = img.shape[:2]
    else:
        return batch_item
    
    # Ensure tensors
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).float()
    if isinstance(cls, np.ndarray):
        cls = torch.from_numpy(cls).float()
    
    cls_flat = cls.view(-1)

    # ------------------------------------------------------------------
    # Source selection — prefer medium / large objects whose rich detail
    # produces CLEAN small samples after downscaling. Falls back to small
    # sources only if no medium/large exists (rare in Cityscapes).
    # ------------------------------------------------------------------
    w_px = bboxes[:, 2] * img_w
    h_px = bboxes[:, 3] * img_h
    max_side = torch.max(w_px, h_px)

    medium_mask = (max_side >= small_thr) & (max_side <= small_thr * 4.0)
    small_mask = max_side < small_thr
    if medium_mask.sum() > 0:
        source_indices = torch.where(medium_mask)[0].tolist()
    elif small_mask.sum() > 0:
        source_indices = torch.where(small_mask)[0].tolist()
    else:
        return batch_item

    # Current boxes in xyxy for IoU checking.
    all_boxes_xyxy = _xywhn_to_xyxy(bboxes, img_h, img_w)

    new_boxes = []
    new_cls = []

    # Convert img to HWC numpy for pixel manipulation.
    if isinstance(img, torch.Tensor):
        img_np = img.numpy() if not img.is_cuda else img.cpu().numpy()
        img_is_tensor = True
    else:
        img_np = img.copy()
        img_is_tensor = False

    if img_np.ndim == 3 and img_np.shape[0] <= 4:
        img_np = img_np.transpose(1, 2, 0)  # CHW → HWC
        was_chw = True
    else:
        was_chw = False

    img_modified = img_np.copy()

    # ------------------------------------------------------------------
    # Resolve the SOURCE image from which pixel crops will be extracted.
    # - If `source_image` is supplied (typical in paired mode: always pass
    #   the CLEAR source_real image for both real/fake calls), use that.
    # - Otherwise fall back to the destination image itself (standalone
    #   mode: crop and paste within the same image — original Kisantal).
    # ------------------------------------------------------------------
    if source_image is not None:
        src_img = source_image
        if isinstance(src_img, torch.Tensor):
            src_img = src_img.cpu().numpy() if src_img.is_cuda else src_img.numpy()
        else:
            src_img = np.asarray(src_img)
        if src_img.ndim == 3 and src_img.shape[0] <= 4:
            src_img = src_img.transpose(1, 2, 0)   # CHW → HWC
        # Shape must match destination so that normalised bboxes map to the
        # same pixel regions.
        if src_img.shape[:2] != img_modified.shape[:2]:
            # Fallback: silently use destination if shapes disagree.
            src_img = img_modified
    else:
        src_img = img_modified

    # ------------------------------------------------------------------
    # Detect actual content area on the DESTINATION image — letterbox /
    # mosaic padding is masked out so pastes never land on gray bars.
    # ------------------------------------------------------------------
    content_box = _find_content_area(img_modified)
    cx1, cy1, cx2, cy2 = content_box
    if (cx2 - cx1) < 32 or (cy2 - cy1) < 32:
        return batch_item   # content region degenerate — skip augmentation

    # ------------------------------------------------------------------
    # Per-image paste budget (not per source object — avoids exploding
    # paste counts when the image already has many objects).
    # ------------------------------------------------------------------
    n_pastes = rng.randint(1, max_copies)

    # Target paste size in COCO small-band, bounded to [24, 40] px.
    # Lower bound 24 keeps pasted objects above the "dot" regime where the
    # detector has no hope of learning useful features (very-tiny <24 px
    # crops also lose most of their downsampled detail). Upper bound 40
    # stays within the small-object band.
    target_size_min = 24.0
    target_size_max = min(40.0, float(small_thr) * 1.4)

    for _ in range(n_pastes):
        # Pick a random source object from the pool.
        si = rng.choice(source_indices)
        obj_cls = int(cls_flat[si].item())
        obj_box = bboxes[si]

        src_cx = obj_box[0].item() * img_w
        src_cy = obj_box[1].item() * img_h
        src_w = obj_box[2].item() * img_w
        src_h = obj_box[3].item() * img_h

        src_x1 = max(0, int(src_cx - src_w / 2))
        src_y1 = max(0, int(src_cy - src_h / 2))
        src_x2 = min(img_w, int(src_cx + src_w / 2))
        src_y2 = min(img_h, int(src_cy + src_h / 2))

        # Crop from the SOURCE image (may be the clear source_real when
        # pasting onto foggy source_fake; or the image itself in standalone).
        crop = src_img[src_y1:src_y2, src_x1:src_x2].copy()
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 6 or crop_w < 6:
            continue

        # Skip if the source crop is essentially uniform colour (likely a
        # region of letterbox padding inside the source image). Deterministic
        # given the same `src_img` across paired real/fake calls → both sides
        # skip/keep identically.
        try:
            crop_std = crop.reshape(-1, crop.shape[-1]).std(axis=0).mean() \
                if crop.ndim == 3 else float(crop.std())
        except Exception:
            crop_std = 0.0
        if crop_std < 5.0:
            continue

        # Target max-side in the COCO "small" band (20-45 px).
        target_size = rng.uniform(target_size_min, target_size_max)

        # Preserve aspect ratio when resizing to target_size.
        aspect = crop_w / float(crop_h)
        if aspect >= 1.0:
            new_w = max(3, int(round(target_size)))
            new_h = max(3, int(round(target_size / aspect)))
        else:
            new_h = max(3, int(round(target_size)))
            new_w = max(3, int(round(target_size * aspect)))

        # Perspective-aware zone (narrow around vanishing point, y depends
        # on the chosen target_size).
        zone_x1, zone_x2, zone_y1, zone_y2 = _perspective_zone(
            obj_cls, content_box, target_size,
            size_min=target_size_min, size_max=target_size_max,
        )

        # Random centre inside that zone.
        target_cx = rng.randint(zone_x1, max(zone_x1 + 1, zone_x2))
        target_cy = rng.randint(zone_y1, max(zone_y1 + 1, zone_y2))

        tgt_x1 = max(cx1, target_cx - new_w // 2)
        tgt_y1 = max(cy1, target_cy - new_h // 2)
        tgt_x2 = min(cx2, tgt_x1 + new_w)
        tgt_y2 = min(cy2, tgt_y1 + new_h)

        actual_w = tgt_x2 - tgt_x1
        actual_h = tgt_y2 - tgt_y1
        if actual_w < 3 or actual_h < 3:
            continue

        # Skip if target area collides with any existing or already-pasted
        # box above the allowed IoU — prevents stacking.
        # `all_boxes_xyxy` is identical between twin images (PairedAugDataset
        # guarantees same bboxes), so the IoU decision is deterministic
        # given the same seed.
        new_box_xyxy = torch.tensor(
            [tgt_x1, tgt_y1, tgt_x2, tgt_y2], dtype=torch.float32
        )
        check_boxes = all_boxes_xyxy.clone()
        if new_boxes:
            extra = torch.stack(new_boxes)
            check_boxes = torch.cat([check_boxes, extra], dim=0)
        ious = _compute_iou_xyxy(new_box_xyxy, check_boxes)
        if ious.numel() > 0 and ious.max() > max_overlap_iou:
            continue

        # Destination region (used for colour match + alpha blend).
        target_region = img_modified[tgt_y1:tgt_y2, tgt_x1:tgt_x2]
        if target_region.size == 0:
            continue

        # Optional horizontal flip of the source crop for variety
        # (Kisantal 2019 recipe, adds ~0 cost).
        crop_to_use = crop
        if rng.random() < 0.5:
            crop_to_use = crop[:, ::-1].copy()

        # Resize — INTER_AREA is the recommended downsample filter
        # (most source crops are medium/large → we downsample to small);
        # INTER_LINEAR for the occasional upsample.
        try:
            import cv2
            is_down = (actual_w * actual_h) < (crop_w * crop_h)
            interp = cv2.INTER_AREA if is_down else cv2.INTER_LINEAR
            resized_crop = cv2.resize(
                crop_to_use, (actual_w, actual_h), interpolation=interp,
            )
        except ImportError:
            from PIL import Image
            pil_crop = Image.fromarray(crop_to_use)
            pil_crop = pil_crop.resize((actual_w, actual_h), Image.BILINEAR)
            resized_crop = np.array(pil_crop)

        # Match local mean colour so clear-domain patches do not stand out
        # as sharp rectangles in foggy / washed-out scenes.
        resized_crop = _color_match(resized_crop, target_region)

        # Gaussian α-blend at the rectangular boundary — hides the edge
        # artefact without needing a segmentation mask.
        alpha = _gaussian_alpha(actual_h, actual_w)
        dst = target_region.astype(np.float32)
        blended = alpha * resized_crop.astype(np.float32) + (1.0 - alpha) * dst
        img_modified[tgt_y1:tgt_y2, tgt_x1:tgt_x2] = \
            np.clip(blended, 0, 255).astype(np.uint8)

        new_boxes.append(new_box_xyxy)
        new_cls.append(obj_cls)
    
    if not new_boxes:
        return batch_item
    
    # Convert back to CHW if needed
    if was_chw:
        img_modified = img_modified.transpose(2, 0, 1)  # HWC → CHW
    
    # Update image
    if img_is_tensor:
        batch_item['img'] = torch.from_numpy(img_modified)
    else:
        batch_item['img'] = img_modified
    
    # Append new boxes to labels
    new_boxes_xyxy = torch.stack(new_boxes)  # (M, 4) pixel xyxy
    new_boxes_xywhn = _xyxy_to_xywhn(new_boxes_xyxy, img_h, img_w)  # (M, 4) normalized
    new_cls_tensor = torch.tensor(new_cls, dtype=cls.dtype).view(-1, 1) if cls.dim() == 2 \
        else torch.tensor(new_cls, dtype=cls.dtype)
    
    batch_item['bboxes'] = torch.cat([bboxes, new_boxes_xywhn], dim=0)
    batch_item['cls'] = torch.cat([cls, new_cls_tensor], dim=0)
    
    # Update batch_idx if present (all new boxes belong to same image)
    if 'batch_idx' in batch_item and isinstance(batch_item['batch_idx'], torch.Tensor):
        if batch_item['batch_idx'].numel() > 0:
            batch_idx_val = batch_item['batch_idx'][0].item()
        else:
            batch_idx_val = 0
        new_batch_idx = torch.full((len(new_boxes),), batch_idx_val,
                                   dtype=batch_item['batch_idx'].dtype)
        batch_item['batch_idx'] = torch.cat([batch_item['batch_idx'], new_batch_idx], dim=0)
    
    return batch_item
