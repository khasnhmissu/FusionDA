#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Variant 07 — "hyperparameter sensitivity" (RQ3)
# ─────────────────────────────────────────────────────────────────
# Sweeps the THREE most influential hyper-parameters of the full
# FusionDA pipeline (same setup as 05_grl.sh: teacher-student EMA +
# pseudo-label distillation + source/target consistency + GRL).
#
#  H1.  lambda-weight  (λ_distill)   — pseudo-label distillation weight
#       defaults: 0.2   sweep: {0.05, 0.1, 0.2*, 0.5, 1.0}
#       refs   : DA-Faster (Chen'18), Unbiased Teacher (Liu'21),
#                Adaptive Teacher (Li CVPR'22)
#
#  H2.  grl-weight     (λ_adv)       — adversarial domain-confusion weight
#       defaults: 0.05  sweep: {0.01, 0.05*, 0.1, 0.2, 0.5}
#       refs   : DANN (Ganin'15), DA-Faster (Chen'18),
#                Probabilistic Teacher (Chen CVPR'22)
#
#  H3.  teacher-alpha  (EMA momentum) — Mean-Teacher EMA decay
#       defaults: 0.9999 sweep: {0.99, 0.999, 0.9995, 0.9999*, 0.99995}
#       refs   : Mean Teacher (Tarvainen'17), SoftTeacher (Xu'21),
#                Unbiased / Adaptive Teacher (~0.9996)
#
# Total: 1 shared baseline + 4 × 3 = 13 runs.
# Each run logs to runs/ablation/07_sens_<param>_<value>/train.log
# Final summary table prints Source/Target mAP@50 per run.
# ─────────────────────────────────────────────────────────────────
set -e

# ── shared setup (mirror 05_grl.sh) ──────────────────────────────
WEIGHTS="yolo26s.pt"
DATA="configs/data/data.yaml"
PROJECT="runs/ablation"
EPOCHS=40
BATCH=4

# ── default (center) values ──────────────────────────────────────
DEF_LAMBDA=0.2
DEF_GRL=0.05
DEF_ALPHA=0.9999

# ── sweep grids (default value will be skipped → reuse baseline) ──
LAMBDA_VALUES=(0.05 0.1 0.2 0.5 1.0)
GRL_VALUES=(0.01 0.05 0.1 0.2 0.5)
ALPHA_VALUES=(0.99 0.999 0.9995 0.9999 0.99995)

# ── helper: run one config ───────────────────────────────────────
run_one () {
  local NAME="$1"; local LAMBDA="$2"; local GRL="$3"; local ALPHA="$4"
  local OUT="$PROJECT/$NAME"
  if [[ -f "$OUT/train.log" ]]; then
    echo "[skip] $NAME already has a train.log — delete to re-run."
    return
  fi
  mkdir -p "$OUT"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  [$NAME]  λ_distill=$LAMBDA  λ_adv=$GRL  α_ema=$ALPHA"
  echo "════════════════════════════════════════════════════════════"
  python train.py \
    --weights        "$WEIGHTS" \
    --data           "$DATA" \
    --epochs         "$EPOCHS" \
    --batch          "$BATCH" \
    --project        "$PROJECT" \
    --name           "$NAME" \
    --lambda-weight  "$LAMBDA" \
    --grl-weight     "$GRL" \
    --teacher-alpha  "$ALPHA" \
    --use-grl \
    --eval-source \
    2>&1 | tee "$OUT/train.log"
}

# ── 1. shared baseline (default of all three) ────────────────────
BASE_NAME="07_sens_baseline"
run_one "$BASE_NAME" "$DEF_LAMBDA" "$DEF_GRL" "$DEF_ALPHA"

# ── 2. sweep H1: lambda-weight ───────────────────────────────────
for v in "${LAMBDA_VALUES[@]}"; do
  [[ "$v" == "$DEF_LAMBDA" ]] && continue   # baseline already covers default
  NAME="07_sens_lambda_${v}"
  run_one "$NAME" "$v" "$DEF_GRL" "$DEF_ALPHA"
done

# ── 3. sweep H2: grl-weight ──────────────────────────────────────
for v in "${GRL_VALUES[@]}"; do
  [[ "$v" == "$DEF_GRL" ]] && continue
  NAME="07_sens_grl_${v}"
  run_one "$NAME" "$DEF_LAMBDA" "$v" "$DEF_ALPHA"
done

# ── 4. sweep H3: teacher-alpha ───────────────────────────────────
for v in "${ALPHA_VALUES[@]}"; do
  [[ "$v" == "$DEF_ALPHA" ]] && continue
  NAME="07_sens_alpha_${v}"
  run_one "$NAME" "$DEF_LAMBDA" "$DEF_GRL" "$v"
done

# ── 5. summary table ─────────────────────────────────────────────
print_section () {
  local title="$1"; shift
  local -a names=("$@")
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $title"
  echo "════════════════════════════════════════════════════════════"
  printf "%-32s | %-15s | %-15s\n" "run" "Source mAP@50" "Target mAP@50"
  printf "%-32s-+-%-15s-+-%-15s\n" \
    "$(printf '─%.0s' {1..32})" "$(printf '─%.0s' {1..15})" "$(printf '─%.0s' {1..15})"
  for n in "${names[@]}"; do
    log="$PROJECT/$n/train.log"
    if [[ -f "$log" ]]; then
      line=$(grep -E "(Source mAP|Student mAP|Target mAP)" "$log" | tail -1)
      src=$(echo "$line" | grep -oP 'Source mAP@50=\K[0-9.]+' || echo "—")
      tgt=$(echo "$line" | grep -oP 'Target mAP@50=\K[0-9.]+' || \
            echo "$line" | grep -oP 'Student mAP@50=\K[0-9.]+' || echo "—")
      printf "%-32s | %-15s | %-15s\n" "$n" "$src" "$tgt"
    else
      printf "%-32s | %-15s | %-15s\n" "$n" "NO LOG" "NO LOG"
    fi
  done
}

# Build per-sweep name lists (default → baseline)
LAMBDA_NAMES=(); for v in "${LAMBDA_VALUES[@]}"; do
  if [[ "$v" == "$DEF_LAMBDA" ]]; then LAMBDA_NAMES+=("$BASE_NAME");
  else LAMBDA_NAMES+=("07_sens_lambda_${v}"); fi
done
GRL_NAMES=(); for v in "${GRL_VALUES[@]}"; do
  if [[ "$v" == "$DEF_GRL" ]]; then GRL_NAMES+=("$BASE_NAME");
  else GRL_NAMES+=("07_sens_grl_${v}"); fi
done
ALPHA_NAMES=(); for v in "${ALPHA_VALUES[@]}"; do
  if [[ "$v" == "$DEF_ALPHA" ]]; then ALPHA_NAMES+=("$BASE_NAME");
  else ALPHA_NAMES+=("07_sens_alpha_${v}"); fi
done

print_section "H1: lambda-weight  (pseudo-label distillation)" "${LAMBDA_NAMES[@]}"
print_section "H2: grl-weight     (adversarial domain alignment)" "${GRL_NAMES[@]}"
print_section "H3: teacher-alpha  (Mean-Teacher EMA momentum)"   "${ALPHA_NAMES[@]}"
