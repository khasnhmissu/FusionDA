"""FusionDA training entry-point — torchvision Faster R-CNN R-50-FPN backend.

Algorithmic parity with train.py
================================
This file is a *parallel sibling* of ``train.py``.  It replicates FusionDA's
algorithm — image translation, paired source-fake supervised loss, cosine
consistency on backbone features, Mean Teacher pseudo-distillation on
target images, and adversarial GRL alignment — onto a two-stage Faster
R-CNN backbone instead of single-stage YOLO.  The eight-step "loss
recipe" at the top of ``train.py`` carries over verbatim:

    loss_source        = detection(student(I^s)            , GT_s)
    loss_source_fake   = detection(student(I^sf)           , GT_s)
    loss_consistency   = 1 - cos(φ(I^sf), φ(I^s).detach())
    loss_distillation  = detection(student(I^t),
                                   pseudo := teacher(I^tf).filter())
    loss_domain        = BCE(disc(GRL(φ(I^s))), disc(GRL(φ(I^t))))

Faster R-CNN's loss dict (``loss_classifier`` + ``loss_box_reg`` +
``loss_objectness`` + ``loss_rpn_box_reg``) is summed into a single
scalar by ``FasterRCNNLoss`` so the rest of the loop stays YOLO-shaped.

Why a sibling and not a flag in train.py
========================================
``train.py`` imports ``FDALoss`` (which wraps ``v8DetectionLoss`` /
``E2ELoss``) at module-import time and calls ``YOLO(...)`` for model
construction.  Both are YOLO-only abstractions that would require
substantial branching to support a torchvision detector.  Adding the
sibling keeps the YOLO path **byte-for-byte unchanged** and gives the
Faster R-CNN backend a clean home.

"""
from __future__ import annotations

import argparse
import gc
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_adaptation import (
    DomainDiscriminator,
    compute_domain_loss,
    get_grl_alpha,
)
from fusion_da import PairedMultiDomainDataset, WeightEMA
from utils.FDA_helpers import get_progressive_lambda

from fasterrcnn import (
    FPNFeatureHook,
    FasterRCNNLoss,
    build_fasterrcnn,
    torchvision_predictions_to_pseudo_targets,
    yolo_batch_to_torchvision,
)


def _validate(model, opt, device, tag: str, save_dir: Path) -> Optional[dict]:
    """Run fasterrcnn.eval.evaluate on the val set.

    Returns {map50, map50_95, ap_small, ap_medium, ap_large} or None if
    val_target_path is not configured.
    """
    val_path = opt.val_target_path
    if not val_path or not Path(val_path).exists():
        print(f"[VAL/{tag}] No target val path configured — skipping.")
        return None

    # Late import: keeps eval optional for users who only train (so no
    # pycocotools required at runtime if val is disabled).
    from fasterrcnn.eval import evaluate as evaluate_fasterrcnn

    eval_save_dir = save_dir / f"val_{tag}_{int(time.time())}"
    eval_save_dir.mkdir(parents=True, exist_ok=True)

    tmp_ckpt = eval_save_dir / "tmp_for_val.pt"
    torch.save({"model": model.state_dict()}, tmp_ckpt)

    try:
        summary = evaluate_fasterrcnn(
            str(tmp_ckpt),
            val_path,
            f"{tag}",
            num_classes=opt.nc,
            names=opt.names,
            min_size=opt.imgsz,
            max_size=opt.imgsz,
            device=str(device),
            score_thresh=0.001,
            save_json_dir=eval_save_dir,
        )
    finally:
        try:
            tmp_ckpt.unlink()
        except OSError:
            pass

    return summary


def train(opt):
    device = torch.device(f"cuda:{opt.device}" if str(opt.device).isdigit() else opt.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] device={opt.device} requested, CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "weights").mkdir(exist_ok=True)
    print("=" * 70)
    print(f"FusionDA / Faster R-CNN R-50-FPN — {'BASELINE' if opt.baseline else 'FULL'}")
    print(f"  Save dir:   {save_dir}")
    print(f"  Device:     {device}")
    print(f"  imgsz:      {opt.imgsz}")
    print(f"  batch:      {opt.batch}")
    print(f"  epochs:     {opt.epochs}")
    print("=" * 70)

    with open(opt.data, encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)
    nc = int(data_dict["nc"])
    names = data_dict["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    opt.nc = nc
    opt.names = names

    # Resolve full target val path so _validate() can find it later.
    root = Path(data_dict.get("path", ""))
    val_target = data_dict.get("val_target_real") or data_dict.get("val")
    if isinstance(val_target, list):
        val_target = val_target[0] if val_target else None
    if val_target:
        val_target_dir = Path(root) / val_target
        # Walk up to the nearest directory that contains both images/ and labels/
        # (the val_dir convention used by eval_v5m / fasterrcnn.eval).
        opt.val_target_path = str(val_target_dir.parent.parent)
    else:
        opt.val_target_path = None

    print(f"  Classes:    {nc} ({names})")
    print(f"  Val path:   {opt.val_target_path}")

    model_student = build_fasterrcnn(
        num_classes=nc,
        min_size=opt.imgsz,
        max_size=opt.imgsz,
        pretrained_backbone=True,                  # ImageNet only — no COCO leakage
    ).to(device)

    teacher_ema = None
    model_teacher = None
    student_hook = None
    teacher_hook = None
    domain_disc = None
    grl_optimizer = None

    if not opt.baseline:
        teacher_ema = WeightEMA(
            model_student,
            alpha=opt.teacher_alpha,
            device=device,
            update_after_step=getattr(opt, "update_after_step", 500),
            alpha_rampup_steps=getattr(opt, "alpha_rampup_steps", 2000),
        )
        model_teacher = teacher_ema.module
        student_hook = FPNFeatureHook(model_student, level=opt.fpn_level)
        teacher_hook = FPNFeatureHook(model_teacher, level=opt.fpn_level)

        if opt.use_grl:
            # FPN OUT_CHANNELS=256 in torchvision Faster R-CNN R-50-FPN.
            domain_disc = DomainDiscriminator(
                in_channels=256,
                hidden_dim=opt.grl_hidden_dim,
                dropout=opt.grl_dropout,
            ).to(device)
            grl_optimizer = optim.Adam(
                domain_disc.parameters(), lr=opt.grl_lr,
            )
            print(f"  GRL:        in_channels=256 (FPN P{opt.fpn_level})")
        else:
            print(f"  GRL:        disabled")

    n_student = sum(p.numel() for p in model_student.parameters())
    print(f"  Params:     student={n_student/1e6:.1f}M"
          + (f", teacher={sum(p.numel() for p in model_teacher.parameters())/1e6:.1f}M"
             if model_teacher is not None else ""))

    # AdamW matches FusionDA's YOLO convention (lr=1e-4).  For users who
    # want to reproduce ALDI's R-50-FPN recipe (SGD lr=0.005), pass
    # ``--optimizer sgd --lr0 0.005``.
    if opt.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            [p for p in model_student.parameters() if p.requires_grad],
            lr=opt.lr0, momentum=0.9, weight_decay=0.0001, nesterov=True,
        )
    else:
        optimizer = optim.AdamW(
            [p for p in model_student.parameters() if p.requires_grad],
            lr=opt.lr0, weight_decay=0.0005,
        )

    warmup_epochs_lr = max(1, int(getattr(opt, "warmup_epochs", 3)))
    def _lr_factor(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs_lr:
            return (epoch_idx + 1) / warmup_epochs_lr
        progress = (epoch_idx - warmup_epochs_lr) / max(opt.epochs - warmup_epochs_lr, 1)
        return opt.lrf + (1.0 - opt.lrf) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)

    from ultralytics.cfg import get_cfg
    default_cfg = get_cfg()

    def _get_path(key, fallback=None):
        v = data_dict.get(key, fallback)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(Path(root) / v) if v else None

    source_real = _get_path("train_source_real", data_dict.get("train"))
    source_fake = _get_path("train_source_fake")
    target_real = _get_path("train_target_real")
    target_fake = _get_path("train_target_fake")
    print(f"  Source real:  {source_real}")
    print(f"  Source fake:  {source_fake}")
    print(f"  Target real:  {target_real}")
    print(f"  Target fake:  {target_fake}")

    # FPN max stride is 32 — same alignment requirement as YOLO.
    paired_dataset = PairedMultiDomainDataset(
        source_real_path=source_real if source_real else "",
        source_fake_path=source_fake if not opt.baseline else None,
        target_real_path=target_real if not opt.baseline else None,
        target_fake_path=target_fake if not opt.baseline else None,
        imgsz=opt.imgsz,
        augment=True,
        hyp=default_cfg,
        data=data_dict,
        stride=32,
    )
    loader = DataLoader(
        paired_dataset,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=PairedMultiDomainDataset.collate_fn,
    )
    print(f"  Dataset:    {len(paired_dataset)} samples, {len(loader)} batches/epoch")

    compute_loss = FasterRCNNLoss(model_student)
    use_amp = bool(getattr(opt, "amp", False)) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)

    csv_path = save_dir / "metrics.csv"
    csv_keys = [
        "epoch", "iter", "lr",
        "loss", "loss_source", "loss_source_fake", "loss_consistency",
        "loss_distillation", "loss_domain",
    ]
    if not csv_path.exists():
        csv_path.write_text(",".join(csv_keys) + "\n")

    best_map50 = 0.0
    global_step = 0
    t_start = time.time()

    for epoch in range(opt.epochs):
        model_student.train()
        if model_teacher is not None:
            model_teacher.eval()      # teacher always in eval mode

        running = defaultdict(float)
        n_iters = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{opt.epochs}", dynamic_ncols=True)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            if grl_optimizer is not None:
                grl_optimizer.zero_grad(set_to_none=True)

            for domain in ("source_real", "source_fake", "target_real", "target_fake"):
                if domain not in batch:
                    continue
                bd = batch[domain]
                bd["img"] = bd["img"].to(device, non_blocking=True).float()
                if "img" in bd and bd["img"].max() > 1.5:    # YOLODataset returns uint8 [0,255]
                    bd["img"] = bd["img"] / 255.0
                for k in ("cls", "bboxes", "batch_idx"):
                    if k in bd:
                        bd[k] = bd[k].to(device, non_blocking=True)

            with amp.autocast(enabled=use_amp):
                imgs_sr, tgts_sr = yolo_batch_to_torchvision(batch["source_real"])
                # If a whole batch ends up empty (all images stripped of GT
                # by augmentation crops), torchvision FasterRCNN can still
                # train on negative samples — but skip the iter cleanly to
                # avoid wasted GPU time.
                if all(len(t["boxes"]) == 0 for t in tgts_sr):
                    continue

                loss_source, items_s = compute_loss(imgs_sr, tgts_sr)

                if opt.baseline:
                    loss_total = loss_source
                    loss_sf = torch.zeros((), device=device)
                    loss_consistency = torch.zeros((), device=device)
                    loss_distill = torch.zeros((), device=device)
                    loss_domain = torch.zeros((), device=device)
                else:
                    feat_sr = student_hook.get_features().detach().clone()

                    imgs_sf, tgts_sf = yolo_batch_to_torchvision(batch["source_fake"])
                    loss_sf, _ = compute_loss(imgs_sf, tgts_sf)
                    feat_sf = student_hook.get_features()

                    cos = F.cosine_similarity(
                        feat_sf.flatten(2), feat_sr.flatten(2), dim=1,
                    ).mean()
                    loss_consistency = 1.0 - cos

                    if epoch >= opt.burn_in_epochs and target_fake is not None:
                        imgs_tf, _ = yolo_batch_to_torchvision(batch["target_fake"])
                        with torch.no_grad():
                            model_teacher.eval()
                            teacher_preds = model_teacher(imgs_tf)
                        pseudo = torchvision_predictions_to_pseudo_targets(
                            teacher_preds, conf_thres=opt.conf_thres,
                        )
                    else:
                        pseudo = None

                    if pseudo is not None and any(len(t["boxes"]) > 0 for t in pseudo):
                        imgs_tr, _ = yolo_batch_to_torchvision(batch["target_real"])
                        loss_distill = compute_loss.compute_distillation_loss(
                            imgs_tr, pseudo,
                        )
                        feat_tr = student_hook.get_features()
                    else:
                        loss_distill = torch.zeros((), device=device)
                        # We still need feat_tr for GRL — do a no-target
                        # forward to fire the hook.  Don't compute loss here.
                        if opt.use_grl:
                            imgs_tr, _ = yolo_batch_to_torchvision(batch["target_real"])
                            # Faster R-CNN forward in eval mode populates the
                            # hook but doesn't compute loss — perfect.
                            model_student.eval()
                            with torch.no_grad():
                                _ = model_student(imgs_tr)
                            model_student.train()
                            feat_tr = student_hook.get_features()
                        else:
                            feat_tr = None

                    # ── Domain (GRL) loss ─────────────────────────
                    if opt.use_grl and feat_tr is not None and epoch >= opt.grl_warmup:
                        alpha_grl = get_grl_alpha(
                            epoch=epoch - opt.grl_warmup,
                            total_epochs=max(1, opt.epochs - opt.grl_warmup),
                            max_alpha=opt.grl_max_alpha,
                        )
                        logits_s = domain_disc(feat_sr, alpha=alpha_grl)
                        logits_t = domain_disc(feat_tr, alpha=alpha_grl)
                        loss_domain = compute_domain_loss(logits_s, logits_t)
                        if isinstance(loss_domain, tuple):
                            loss_domain = loss_domain[0]
                    else:
                        loss_domain = torch.zeros((), device=device)

                    if getattr(opt, "use_progressive_lambda", False):
                        lam = get_progressive_lambda(
                            epoch=epoch,
                            total_epochs=opt.epochs,
                            warmup_epochs=opt.burn_in_epochs,
                            max_lambda=opt.lambda_weight,
                        )
                    else:
                        lam = opt.lambda_weight

                    loss_total = (
                        loss_source
                        + opt.source_fake_weight * loss_sf
                        + opt.consistency_weight * loss_consistency
                        + lam * loss_distill
                        + opt.grl_weight * loss_domain
                    )

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model_student.parameters(),
                max_norm=opt.gradient_clip,
            )
            if domain_disc is not None:
                # Mirror train.py:1019 — clip discriminator with same norm.
                torch.nn.utils.clip_grad_norm_(
                    domain_disc.parameters(),
                    max_norm=opt.gradient_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            if grl_optimizer is not None:
                grl_optimizer.step()

            # WeightEMA tracks the step internally; signature is .update(model).
            if teacher_ema is not None:
                teacher_ema.update(model_student)

            running["loss"]              += float(loss_total.detach())
            running["loss_source"]       += float(loss_source.detach())
            running["loss_source_fake"]  += float(loss_sf.detach()) if torch.is_tensor(loss_sf) else 0.0
            running["loss_consistency"]  += float(loss_consistency.detach()) if torch.is_tensor(loss_consistency) else 0.0
            running["loss_distillation"] += float(loss_distill.detach()) if torch.is_tensor(loss_distill) else 0.0
            running["loss_domain"]       += float(loss_domain.detach()) if torch.is_tensor(loss_domain) else 0.0
            n_iters += 1

            pbar.set_postfix({
                "loss":  f"{loss_total.item():.3f}",
                "src":   f"{loss_source.item():.3f}",
                "lr":    f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # CSV row for every iter
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{global_step},{optimizer.param_groups[0]['lr']:.6e},"
                    f"{float(loss_total.detach()):.4f},"
                    f"{float(loss_source.detach()):.4f},"
                    f"{float(loss_sf.detach()) if torch.is_tensor(loss_sf) else 0:.4f},"
                    f"{float(loss_consistency.detach()) if torch.is_tensor(loss_consistency) else 0:.4f},"
                    f"{float(loss_distill.detach()) if torch.is_tensor(loss_distill) else 0:.4f},"
                    f"{float(loss_domain.detach()) if torch.is_tensor(loss_domain) else 0:.4f}"
                    "\n"
                )
            global_step += 1

        scheduler.step()
        avg = {k: v / max(1, n_iters) for k, v in running.items()}
        print(f"[epoch {epoch+1}/{opt.epochs}] " +
              " ".join(f"{k}={v:.3f}" for k, v in avg.items()))

        if (epoch + 1) % opt.val_interval == 0 or (epoch + 1) == opt.epochs:
            summary = _validate(model_student, opt, device,
                                tag=f"student_e{epoch+1:03d}",
                                save_dir=save_dir)
            if summary:
                print(f"[VAL student e{epoch+1}] mAP50={summary.get('map50', 0):.4f} "
                      f"mAP50-95={summary.get('map50_95', 0):.4f} "
                      f"AP_s={summary.get('ap_small', 0):.4f} "
                      f"AP_m={summary.get('ap_medium', 0):.4f} "
                      f"AP_l={summary.get('ap_large', 0):.4f}")
                if summary.get("map50", 0) > best_map50:
                    best_map50 = summary["map50"]
                    torch.save(
                        {"model": model_student.state_dict(),
                         "epoch": epoch, "names": names, "nc": nc},
                        save_dir / "weights" / "best.pt",
                    )
                    print(f"  ↳ saved new best.pt (mAP50={best_map50:.4f})")

            if model_teacher is not None:
                summary_t = _validate(model_teacher, opt, device,
                                      tag=f"teacher_e{epoch+1:03d}",
                                      save_dir=save_dir)
                if summary_t:
                    print(f"[VAL teacher e{epoch+1}] mAP50={summary_t.get('map50',0):.4f} "
                          f"mAP50-95={summary_t.get('map50_95',0):.4f}")

        if (epoch + 1) % opt.save_interval == 0:
            torch.save(
                {"model": model_student.state_dict(),
                 "epoch": epoch, "names": names, "nc": nc},
                save_dir / "weights" / f"epoch_{epoch+1:03d}.pt",
            )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(
        {"model": model_student.state_dict(),
         "epoch": opt.epochs - 1, "names": names, "nc": nc},
        save_dir / "weights" / "last.pt",
    )
    if model_teacher is not None:
        torch.save(
            {"model": model_teacher.state_dict(),
             "epoch": opt.epochs - 1, "names": names, "nc": nc},
            save_dir / "weights" / "last_teacher.pt",
        )
    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"\nDone. {opt.epochs} epochs in {elapsed_h:.2f} h. "
          f"Best mAP50={best_map50:.4f}. Save dir: {save_dir}")

    if student_hook is not None:
        student_hook.remove()
    if teacher_hook is not None:
        teacher_hook.remove()


def parse_args():
    p = argparse.ArgumentParser(description="FusionDA training on Faster R-CNN R-50-FPN.")
    p.add_argument("--config",       default=None, help="optional YAML config")
    p.add_argument("--data",         default="configs/data/data.yaml")
    p.add_argument("--imgsz",        type=int, default=640)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--batch",        type=int, default=4)

    # Schedule  — defaults match FusionDA YOLO (train.py:278 hardcodes
    # warmup_epochs_lr=5; train.py:1016 grad_clip default 2.0).
    p.add_argument("--epochs",       type=int, default=40)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr0",          type=float, default=1e-4)
    p.add_argument("--lrf",          type=float, default=0.01)
    p.add_argument("--optimizer",    default="adamw", choices=("adamw", "sgd"))
    p.add_argument("--gradient-clip", type=float, default=2.0,
                   help="Max grad norm — matches train.py's default 2.0. "
                        "ALDI's R-50-FPN config uses 5.0; bump if loss diverges.")
    p.add_argument("--device",       default="0")

    p.add_argument("--baseline",     action="store_true",
                   help="Source-only Faster R-CNN (no DA).")

    # Defaults match FusionDA YOLO's train.py CLI for cross-backend parity.
    p.add_argument("--teacher-alpha",        type=float, default=0.9999)
    p.add_argument("--update-after-step",    type=int,   default=500)
    p.add_argument("--alpha-rampup-steps",   type=int,   default=2000)
    p.add_argument("--burn-in-epochs",       type=int,   default=5)
    p.add_argument("--conf-thres",           type=float, default=0.5,
                   help="Teacher pseudo-label score threshold "
                        "(train.py:79 hardcodes 0.5; matched here).")
    p.add_argument("--lambda-weight",        type=float, default=0.2,
                   help="Distillation loss weight (train.py CLI default).")
    p.add_argument("--use-progressive-lambda", action="store_true")
    p.add_argument("--consistency-weight",   type=float, default=0.5)
    p.add_argument("--source-fake-weight",   type=float, default=0.1)
    p.add_argument("--use-grl",              action="store_true")
    p.add_argument("--grl-warmup",           type=int,   default=5)
    p.add_argument("--grl-max-alpha",        type=float, default=1.0)
    p.add_argument("--grl-weight",           type=float, default=0.05)
    p.add_argument("--grl-hidden-dim",       type=int,   default=512)
    p.add_argument("--grl-dropout",          type=float, default=0.1)
    p.add_argument("--grl-lr",               type=float, default=5e-5)
    p.add_argument("--fpn-level",            default="2",
                   help="FPN level for hook ('0'=P2, '1'=P3, '2'=P4 (default), '3'=P5).")
    p.add_argument("--project",              default="runs/fda_fasterrcnn")
    p.add_argument("--name",                 default="exp")
    p.add_argument("--amp",                  action="store_true",
                   help="Enable mixed-precision (CUDA only).")
    p.add_argument("--val-interval",         type=int, default=5)
    p.add_argument("--save-interval",        type=int, default=10)

    return p.parse_args()


def _is_cli_set(key: str) -> bool:
    """Heuristic: did the user pass ``--<key>`` on the command line?

    We can't tell from argparse alone whether a value is the default or
    user-set — argparse erases that distinction.  Inspecting ``sys.argv``
    is the simplest robust signal.
    """
    import sys
    flag = "--" + key.replace("_", "-")
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    # Optional YAML config — takes precedence over CLI defaults but not over
    # explicitly-passed CLI flags.  Same convention as train.py.
    if args.config and Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        for k, v in yaml_cfg.items():
            # Only override if user didn't pass the flag explicitly on CLI.
            if not _is_cli_set(k):
                setattr(args, k.replace("-", "_"), v)

    train(args)
