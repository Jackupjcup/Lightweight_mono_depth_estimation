#!/bin/bash
# =============================================================================
# train_v1.sh — launcher for tools/train_v1.py with safe, opt-in auto-resume.
#
# USAGE
#   bash train_v1.sh
#       Start a FRESH run. Creates work_dirs/v1_tenth/<timestamp>/ and trains
#       from scratch. Never touches any existing run. (default behaviour)
#
#   RESUME=auto bash train_v1.sh
#       RESUME the most recent run under work_dirs/v1_tenth/, continuing from its
#       newest ckpt_epoch*.pt. Falls back to a fresh run if none exists.
#
#   RESUME_RUN=20260620_035604 bash train_v1.sh
#       RESUME a SPECIFIC run dir (work_dirs/v1_tenth/20260620_035604).
#
# RULES (read before resuming)
#   - Resume picks ckpt_epoch*.pt (periodic), NOT best_*.pt (metric-best snapshots
#     would roll training progress back).
#   - Do NOT resume a run that is still training: two processes writing the same
#     dir corrupt logs/checkpoints.
#   - Resume requires the SAME config / #GPUs / dataset: the LR scheduler restores
#     step count only, so changing steps_per_epoch desyncs the schedule.
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd):$PYTHONPATH"

# run_train <work_root> <config> <torchrun args...>
# work_root MUST equal work_dirs/<config-stem> (train_v1.py derives its log dir
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

run_train work_dirs/v1 configs/v1.yaml --nproc_per_node=4 --master_port=29501 tools/train_v1.py


