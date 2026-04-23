# Explainability Scripts

Standalone CLI scripts that explain a trained FusionDA model. Each script
runs **after training** on saved checkpoints. They share a common utility
module (`_common.py`) so checkpoint loading, output parsing and matching
logic stay consistent.

These scripts are designed to answer three research questions:

| File | RQ | Evidence it produces |
|---|---|---|
| `rq1_pseudo_label_quality.py` | RQ1 — does the teacher improve? | P/R/F1/mIoU curves over training + **per-size (small/medium/large) breakdown** + **confidence calibration** |
| `rq2_feature_umap.py` | RQ2a — is the feature space domain-aligned? | UMAP / t-SNE scatter (source vs target) + **MMD** quantitative gap |
| `rq2_c2psa_attention.py` | RQ2b — where inside YOLO26 does the backbone attend? | **C2PSA self-attention heatmaps** per query token |
| `rq3_d_rise.py` | RQ3 — what does the detector look at? | **D-RISE** (black-box) saliency per detection, DA vs no-DA |
| `detection_diff.py` | Supporting | TP/FP/FN visualisation across models |

---

## Prerequisites

### Python packages

```bash
pip install matplotlib umap-learn scikit-learn tqdm pyyaml opencv-python pillow
# ultralytics + torch are already required by training
```

`rq2_feature_umap.py` uses `umap-learn`. If absent, the script still runs
MMD + t-SNE but skips UMAP plotting.

### Checkpoint files

`train.py` saves three kinds of checkpoints:

| File | Contains | Use for |
|---|---|---|
| `best.pt` | `{'model', 'optimizer', 'epoch', 'best_fitness'}` — STUDENT only | RQ2, RQ3, detection_diff |
| `last.pt` | Same as best.pt | Same as above |
| `checkpoint_ep<EP>.pt` | `{'model', 'teacher_state_dict', 'epoch'}` — STUDENT **+ TEACHER** | **RQ1** (needs teacher) |

By default `train.py` writes intermediate checkpoints at `epochs [10, 20, 30, final]`
(`checkpoint_epochs` argument). If you want more granularity for the RQ1
quality curve, add more epochs to that list via the `--checkpoint-epochs` CLI
flag (or edit the default in train.py).

### Dataset layout assumed

```
datasets/
 ├── source_real/source_real/val/{images,labels}/   # clear source val set
 └── target_real/target_real/val/{images,labels}/   # foggy target val set
data.yaml                                            # `nc`, `names`
```

Your `yolo26s.pt` is the 2-class architecture base. Scripts load it only for
the architecture; weights come from the checkpoint.

---

## Run order (typical paper workflow)

1. Train the main model (paired pipeline):
   ```bash
   python train.py --data data.yaml --weights yolo26s.pt --epochs 40 \
       --project runs/ablation --name paired_full_40ep \
       --burn-in-epochs 5 --eval-source
   ```

2. Train a baseline to compare against (for RQ3):
   ```bash
   python train.py --data data.yaml --weights yolo26s.pt --epochs 40 \
       --project runs/ablation --name baseline_40ep \
       --baseline
   ```

3. Run the three evidence scripts (below).

---

## RQ1 — Pseudo-label quality over epochs

Answers "does the teacher actually improve?" and "is the gain in small /
medium / large objects?".

```bash
python explain/rq1_pseudo_label_quality.py \
    --run-dir runs/ablation/paired_full_40ep \
    --data data.yaml \
    --target-images datasets/target_real/target_real/val/images \
    --target-labels datasets/target_real/target_real/val/labels \
    --conf-thres 0.5 --imgsz 1024
```

**Compare two runs (e.g. EMA teacher vs Frozen teacher):**
```bash
python explain/rq1_pseudo_label_quality.py \
    --compare-runs runs/ablation/paired_full_40ep runs/ablation/frozen_teacher \
    --names "EMA Teacher" "Frozen Teacher" \
    --data data.yaml \
    --target-images datasets/target_real/target_real/val/images \
    --target-labels datasets/target_real/target_real/val/labels
```

Outputs (in `<run-dir>/explain/rq1/`):
- `pseudo_label_quality.png` — 2×2 grid P / R / F1 / mIoU over epochs
- `per_size_quality.png` — 3×4 grid (small/medium/large × P/R/F1/n_gt)
- `confidence_distribution.png` — histogram at final epoch
- `calibration_diagram.png` — reliability curve (bin conf vs actual TP rate)
- `quality_overall.csv`, `quality_per_size.csv`, `summary.json`

**Reading it**:
- If the **EMA teacher** curve rises monotonically while the Frozen one is flat → positive evidence that EMA matters.
- If **AP_small** rises faster than AP_medium/large → evidence the pipeline helps small objects specifically.
- Calibration diagram: if the curve sits below the y=x line for bin conf=0.5 (actual TP rate < 0.5) → pseudo-labels are over-confident; raise `TEACHER_CONF_THRES` in `train.py`.

---

## RQ2a — Feature space UMAP + MMD

Answers "are the backbone features for source vs target overlapping?".

**Single model** (the final student):
```bash
python explain/rq2_feature_umap.py \
    --checkpoint runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Paired DA (best)" \
    --source-images datasets/source_real/source_real/val/images \
    --target-images datasets/target_real/target_real/val/images \
    --n-per-domain 500
```

**Compare baseline vs DA (most convincing)**:
```bash
python explain/rq2_feature_umap.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline (no DA)" "Paired DA (full)" \
    --source-images datasets/source_real/source_real/val/images \
    --target-images datasets/target_real/target_real/val/images \
    --n-per-domain 500 --tsne
```

Outputs (in `explain_out/rq2_feature/`):
- `umap_<name>.png` — per-checkpoint scatter
- `umap_grid.png` — all checkpoints side-by-side
- `tsne_<name>.png` (if `--tsne`)
- `mmd.csv`, `summary.json`

**Reading it**:
- Baseline model: blue (source) and orange (target) form **two separated clouds** → domain gap exists at the feature level.
- DA model: **clouds overlap**, MMD is lower (often 30–60% reduction) → DA has actually pulled target features toward source features. That is the hypothesis of the whole paired-augmentation pipeline and this is the direct visual + quantitative proof.

---

## RQ2b — C2PSA self-attention

Answers "where does the backbone attend inside its self-attention block,
and does attention focus on objects rather than background for the DA model?".

**Single model:**
```bash
python explain/rq2_c2psa_attention.py \
    --checkpoint runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --n-images 6
```

**Compare DA vs baseline:**
```bash
python explain/rq2_c2psa_attention.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --n-images 6
```

Outputs (in `explain_out/rq2_attention/`):
- `attention_<name>__<img_stem>.png` — per-image panel: original with boxes + one heatmap per query token.
- `compare_<img_stem>.png` — if multiple checkpoints, rows = checkpoint, cols = query tokens.

**Reading it**: for a query token placed at the **centre of a detected object**, the attention heatmap should concentrate **on the same object** (and similar objects in the scene). A baseline model often shows diffuse attention leaking into road / sky / building texture. DA model → tighter, object-localised attention.

**Caveat**: the hook depends on Ultralytics' C2PSA Attention structure (has `qkv`, `num_heads`, `key_dim`, `head_dim`, optional `pe`, `proj`). If a future Ultralytics version changes the module, the script prints a warning and falls back to the original forward (no attention captured). Open a PR to update the shim in that case.

---

## RQ3 — D-RISE saliency

Answers "where does the detector look when producing a specific detection?"
on foggy target images. Compares a DA-trained model to a baseline.

```bash
python explain/rq3_d_rise.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --labels datasets/target_real/target_real/val/labels \
    --n-images 6 --n-masks 1500
```

Outputs (in `explain_out/rq3_drise/`):
- `drise_<img_stem>.png` — one figure per image. Rows = target detections. Col 0 = original with target box highlighted. Col 1… = D-RISE saliency per model.
- `summary.json` — number of detections explained per image per model.

**Reading it**: DA model's saliency should light up the object pixels themselves. Baseline saliency often spreads to context (road, other cars, sky), or shifts off-object — evidence that the baseline relies on spurious correlations that the paired-augmentation regularises away.

**Runtime**:
- ~7 ms per forward × `n_masks` × `n_images` × number of checkpoints.
- For 1500 masks × 6 images × 2 models ≈ 3 min on an RTX 4080 Super.
- Increase `--n-masks` to 2000+ for paper-quality. Decrease `--batch` if you hit OOM.

**If no `--labels` passed**: all target detections are explained. With `--labels`, TP detections (match a GT at IoU≥0.5) are preferred over FPs.

---

## detection_diff.py — TP/FP/FN side-by-side

Supporting visualisation: show where each model wins / loses on the target set.

```bash
python explain/detection_diff.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --labels datasets/target_real/target_real/val/labels \
    --n-images 8 --best-matches
```

`--best-matches` picks images where the models disagree most (highest std of TP count) — useful for comparison. Use `--size-filter small` to focus on small objects.

**Important**: if your model was trained from COCO pretrained weights without head replacement (80 classes), pass `--class-mapping "0:0,2:1"` to remap. For the user's native 2-class `yolo26s.pt`, leave it unset.

---

## Recommended batch of runs for the paper

Full evidence package — run once when the final models are ready:

```bash
# RQ1 — teacher quality on the main run
python explain/rq1_pseudo_label_quality.py \
    --run-dir runs/ablation/paired_full_40ep \
    --data data.yaml \
    --target-images datasets/target_real/target_real/val/images \
    --target-labels datasets/target_real/target_real/val/labels \
    --conf-thres 0.5 --imgsz 1024

# RQ2a — feature UMAP + MMD, baseline vs DA
python explain/rq2_feature_umap.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --source-images datasets/source_real/source_real/val/images \
    --target-images datasets/target_real/target_real/val/images \
    --n-per-domain 500 --tsne

# RQ2b — C2PSA attention, baseline vs DA
python explain/rq2_c2psa_attention.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --n-images 6

# RQ3 — D-RISE, baseline vs DA
python explain/rq3_d_rise.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --labels datasets/target_real/target_real/val/labels \
    --n-images 6 --n-masks 1500

# Supporting — TP/FP/FN diff, baseline vs DA, small objects
python explain/detection_diff.py \
    --checkpoint runs/ablation/baseline_40ep/weights/best.pt \
                 runs/ablation/paired_full_40ep/weights/best.pt \
    --names "Baseline" "Paired DA" \
    --data data.yaml \
    --images datasets/target_real/target_real/val/images \
    --labels datasets/target_real/target_real/val/labels \
    --n-images 8 --best-matches --size-filter small
```

Collect figures from `runs/ablation/paired_full_40ep/explain/rq1/` and
`explain_out/rq2_feature/`, `explain_out/rq2_attention/`, `explain_out/rq3_drise/`,
`explain_out/detection_diff/`.

---

## What did NOT get ported from the old scripts

- Root-level `gradcam_explain.py` (EigenCAM) has been **deprecated in favour of `rq3_d_rise.py`**. D-RISE is the SOTA choice for detection saliency — unlike EigenCAM it is detection-specific (accounts for box + class + objectness) and doesn't need layer hooks or suffer from SVD sign ambiguity. The old file is deleted.
- Root-level `pseudo_label_quality.py` → superseded by `explain/rq1_pseudo_label_quality.py` (adds per-size breakdown + calibration + bug-free defaults). Old file deleted.
- Root-level `detection_diff.py` → superseded by `explain/detection_diff.py` with the class-mapping bug fixed and fairer image selection. Old file deleted.

The library helpers at `utils/explainability/` (MMD, feature_viz, domain_metrics) are **kept** — they are used during training by `DomainMonitor` and by `rq2_feature_umap.py` for MMD.
