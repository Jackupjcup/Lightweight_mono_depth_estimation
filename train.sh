#!/bin/bash
# =============================================================================
# train.sh — launcher for tools/train.py with safe, opt-in auto-resume.
#
# This script runs TWO trainings sequentially:
#     1) configs/v0_tenth.yaml      -> work_dirs/v0_tenth/<run>
#     2) configs/v0_tenth_v1.yaml   -> work_dirs/v0_tenth_v1/<run>
# (With `set -eo pipefail`, if the 1st training fails the 2nd will NOT start.)
#
# -----------------------------------------------------------------------------
# HOW TO RUN WITHOUT RESUME (default — start fresh, never touches old runs):
#
#     bash train.sh
#
#   Each training creates a brand-new work_dirs/<cfg>/<timestamp>/ and trains
#   from scratch. This is the safe default.
#
# -----------------------------------------------------------------------------
# HOW TO RESUME (opt-in via env var):
#
#   Resume the LATEST run of each training from its newest ckpt_epoch*.pt:
#     RESUME=auto bash train.sh
#
#   Resume a SPECIFIC run dir (same dir name is used under each work_root):
#     RESUME_RUN=20260620_035604 bash train.sh
#
# -----------------------------------------------------------------------------
# RULES (read before resuming):
#   - Resume picks ckpt_epoch*.pt (periodic), NOT best_*.pt — best snapshots are
#     metric-best and would roll training progress back.
#   - Do NOT resume a run that is still training: two processes writing the same
#     dir corrupt logs/checkpoints.
#   - Resume requires the SAME config / #GPUs / dataset: the LR scheduler restores
#     step count only, so changing steps_per_epoch desyncs the schedule.
#   - Logs are appended (tee -a), so resuming never truncates the old train.log.
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd):$PYTHONPATH"

# run_train <work_root> <config> <torchrun args...>
# work_root MUST equal work_dirs/<config-stem> (train.py derives its log dir
# from RUN_NAME + the config filename).
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

  local ckpt resume_arg=""
  ckpt="$(ls -1 "${run_dir}"/ckpt_epoch*.pt 2>/dev/null | sort | tail -1)" || true
  if [ -n "$ckpt" ]; then
    resume_arg="--resume ${ckpt}"
    echo "[resume] ${run_dir}  <-  ${ckpt}"
  else
    [ -n "$resume_requested" ] && echo "[warn] resume requested but no ckpt_epoch*.pt in ${run_dir}; starting fresh"
    echo "[fresh ] ${run_dir}"
  fi

  torchrun "$@" --config "${config}" ${resume_arg} 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done \
    | tee -a "${run_dir}/train.log"
}

run_train work_dirs/v0_tenth    configs/v0_tenth.yaml    --nproc_per_node=4 tools/train.py
# run_train work_dirs/v0_tenth_v1 configs/v0_tenth_v1.yaml --nproc_per_node=4 tools/train.py
