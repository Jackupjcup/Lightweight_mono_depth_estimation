#!/bin/bash
# Training launcher with safe, opt-in auto-resume.
#
#   bash train_all.sh                       # fresh run (new timestamped dir) — default
#   RESUME=auto bash train_all.sh           # resume the LATEST run dir from its newest ckpt_epoch*.pt
#   RESUME_RUN=20260620_035604 bash train_all.sh   # resume a specific run dir
#
# Notes:
#   - Resume uses ckpt_epoch*.pt (periodic checkpoints), NOT best_*.pt (those are
#     metric-best snapshots and would roll training progress back).
#   - Only resume a run that is NOT currently training: two processes writing the
#     same dir will corrupt logs/checkpoints.
#   - Resume requires the SAME config / #GPUs / dataset (the LR scheduler restores
#     step count only; changing steps_per_epoch desyncs the schedule).
set -eo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd):$PYTHONPATH"

# run_train <work_root> <config> <torchrun args...>
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

run_train work_dirs/v0_v1 configs/v0_v1.yaml --nproc_per_node=4 tools/train.py
