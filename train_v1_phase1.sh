#!/bin/bash
# =============================================================================
# train_v1_phase1.sh — Phase-1 backbone feature distillation, warm-started from
#                      a phase-2 best_feat.pt checkpoint.
#
# USAGE
#   bash train_v1_phase1.sh
#       Fresh phase-1 run, weights initialised from INIT_CKPT below.
#       Creates work_dirs/v0_v1_phase1/<timestamp>/.
#
#   RESUME=auto bash train_v1_phase1.sh
#       Resume the most recent phase-1 run from its newest ckpt_epoch*.pt.
#       Falls back to INIT_CKPT warm-start if no periodic checkpoint exists.
#
#   RESUME_RUN=20260623_123456 bash train_v1_phase1.sh
#       Resume a specific phase-1 run dir.
#
# INIT_CKPT warm-start note:
#   A fresh phase-1 run uses --pretrained INIT_CKPT: it loads ONLY model weights
#   from the phase-2 checkpoint, so the optimizer/scheduler/epoch/LR are freshly
#   initialised (start_epoch=0, LR=learning_rate with warmup). Use RESUME=auto or
#   RESUME_RUN=<dir> to --resume an interrupted phase-1 run instead (that path
#   restores optimizer + scheduler + epoch from the run's own ckpt_epoch*.pt).
#
# RULES
#   - Phase-1 only computes feat_loss; early_stop_metric is "feat" (SSIM-based).
#   - Do NOT resume a run that is still training.
#   - Resume requires the SAME config / #GPUs.
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd):$PYTHONPATH"

INIT_CKPT="work_dirs/v1_tenth/20260622_062707/best_feat.pt"

run_train() {
  local work_root="$1"; shift
  local config="$1"; shift

  local resume_requested=""
  local run_name
  if [ -n "${RESUME_RUN:-}" ]; then
    run_name="$RESUME_RUN"; resume_requested=1
  elif [ "${RESUME:-}" = "auto" ] || [ "${RESUME:-}" = "1" ]; then
    run_name="$(ls -1 "$work_root" 2>/dev/null | sort | tail -1)" || true
    [ -z "$run_name" ] && run_name="$(date '+%Y%m%d_%H%M%S')"
    resume_requested=1
  else
    run_name="$(date '+%Y%m%d_%H%M%S')"
  fi
  export RUN_NAME="$run_name"

  local run_dir="${work_root}/${run_name}"
  mkdir -p "$run_dir"

  # Priority: own periodic ckpt (interrupted phase-1 run, full-state --resume)
  #            > INIT_CKPT (warm-start weights ONLY via --pretrained, fresh schedule)
  local ckpt resume_arg=""
  ckpt="$(ls -1 "${run_dir}"/ckpt_epoch*.pt 2>/dev/null | sort | tail -1)" || true
  if [ -n "$ckpt" ]; then
    resume_arg="--resume ${ckpt}"
    echo "[resume phase-1] ${run_dir}  <-  ${ckpt}"
  elif [ -n "${INIT_CKPT}" ]; then
    resume_arg="--pretrained ${INIT_CKPT}"
    [ -n "$resume_requested" ] && echo "[warn] no ckpt_epoch*.pt in ${run_dir}; falling back to INIT_CKPT warm-start"
    echo "[warm-start] ${run_dir}  <-  ${INIT_CKPT} (fresh schedule: start_epoch=0, LR reset)"
  else
    echo "[fresh] ${run_dir}"
  fi

  torchrun "$@" --config "${config}" ${resume_arg} 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done \
    | tee -a "${run_dir}/train.log"
}

run_train work_dirs/v1_tenth_phase1 configs/v1_tenth_phase1.yaml --nproc_per_node=4 --master_port=29502 tools/train_v1.py
