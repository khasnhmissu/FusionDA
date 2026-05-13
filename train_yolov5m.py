"""FusionDA training entry-point for YOLOv5m / YOLOv8 / non-E2E backbones.

``fusion_da.py`` sets ``_IS_E2E = True`` whenever Ultralytics 8.4 is installed,
because ``E2ELoss`` is always importable regardless of which model is loaded.
``FDALoss`` then routes every model through the E2E path, which expects
``{"one2many": ..., "one2one": ...}`` — but YOLOv5mu emits a flat tensor,
so the first forward raises ``KeyError: 'one2many'``.

Fix: patch ``fusion_da._IS_E2E = False`` *before* train.py is imported,
so FDALoss selects ``v8DetectionLoss`` — the correct path for non-E2E heads.
train.py / fusion_da.py are left untouched; this wrapper is opt-in.

Usage
=====
    python train_yolov5m.py --weights yolov5mu.pt --data configs/data/data.yaml
    python train_yolov5m.py --config configs/train_config_yolov5m.yaml

All CLI flags from train.py are accepted unchanged.
"""
from __future__ import annotations

import sys


def _force_v8_loss_path() -> None:
    """Patch fusion_da._IS_E2E to False.

    Must run BEFORE ``from train import ...`` because train imports fusion_da.
    FDALoss reads _IS_E2E at call time (not import time), so patching the
    module attribute here is sufficient.
    """
    import fusion_da
    if getattr(fusion_da, "_IS_E2E", False):
        fusion_da._IS_E2E = False
        # Also re-bind the module-level _DetLoss to v8DetectionLoss so any
        # downstream code that reads it gets the consistent class.
        from ultralytics.utils.loss import v8DetectionLoss
        fusion_da._DetLoss = v8DetectionLoss
        print("[train_yolov5m] Patched fusion_da._IS_E2E → False "
              "(forces v8DetectionLoss for non-E2E heads such as YOLOv5mu).")


def main() -> None:
    _force_v8_loss_path()

    # Import AFTER patch — train triggers fusion_da import which reads _IS_E2E.
    from train import parse_args, train

    args = parse_args()

    if getattr(args, "config", None):
        from utils.config_loader import (
            load_config, config_to_namespace, merge_cli_args,
        )
        config = load_config(args.config)
        config = merge_cli_args(config, args)
        args = config_to_namespace(config)

    print("=" * 70)
    print("FusionDA YOLOv5m wrapper — non-E2E backbone path")
    print("=" * 70)
    print(f"Model:   {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  cuda:{args.device}")
    if not getattr(args, "baseline", False):
        print(f"GRL:     {'Enabled' if args.use_grl else 'Disabled'}")
        print(f"Teacher: EMA")
    else:
        print("Mode:    Source-only detection (no teacher, GRL, consistency)")
        print("Val:     Source mAP + Target mAP (dual-domain)")
    print("=" * 70)

    train(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
