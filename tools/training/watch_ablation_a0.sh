#!/usr/bin/env bash
# Watchdog for A0 ablation training: if the train process dies before epoch_1000,
# resume from the latest saved checkpoint.
set -euo pipefail

REPO="/home/ls/lc/TreeLearn"
WORKDIR="$REPO/work_dirs/train_ablation_baseline"
CONFIG="configs/training/ablations/train_ablation_baseline.yaml"
PY="/home/ls/anaconda3/envs/TreeLearn/bin/python"
STUB="$REPO/work_dirs/train_ablation_rgb_gate/libittnotify_stub.so"
INTERVAL_SEC="${WATCH_INTERVAL_SEC:-1800}"  # default 30 min
LOG="$WORKDIR/watchdog.log"
TRAIN_PATTERN="tools/training/train.py --config configs/training/ablations/train_ablation_baseline"

mkdir -p "$WORKDIR"
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

latest_ckpt() {
  ls -1 "$WORKDIR"/epoch_*.pth 2>/dev/null \
    | sed -n 's/.*epoch_\([0-9][0-9]*\)\.pth/\1 &/p' \
    | sort -n \
    | awk '{print $2}' \
    | tail -1
}

train_running() {
  pgrep -f "$TRAIN_PATTERN" >/dev/null 2>&1
}

start_or_resume() {
  local done_ckpt="$WORKDIR/epoch_1000.pth"
  if [[ -f "$done_ckpt" ]]; then
    log "A0 complete: found $done_ckpt"
    return 0
  fi
  if train_running; then
    log "A0 train already running"
    return 0
  fi

  local ckpt
  ckpt="$(latest_ckpt || true)"
  export LD_PRELOAD="$STUB"
  export LD_LIBRARY_PATH="/home/ls/anaconda3/envs/TreeLearn/lib:${LD_LIBRARY_PATH:-}"

  if [[ -n "${ckpt:-}" ]]; then
    log "A0 train not running; resuming from $ckpt"
    nohup "$PY" -u tools/training/train.py --config "$CONFIG" --resume "$ckpt" \
      >> "$WORKDIR/nohup_stdout.log" 2>&1 &
  else
    log "A0 train not running; no checkpoint found, starting from scratch"
    nohup "$PY" -u tools/training/train.py --config "$CONFIG" \
      >> "$WORKDIR/nohup_stdout.log" 2>&1 &
  fi
  log "launched PID=$!"
}

log "watchdog started (interval=${INTERVAL_SEC}s)"
start_or_resume

while true; do
  sleep "$INTERVAL_SEC"
  if [[ -f "$WORKDIR/epoch_1000.pth" ]]; then
    log "A0 finished (epoch_1000.pth present); watchdog exiting"
    exit 0
  fi
  if train_running; then
    latest="$(latest_ckpt || echo none)"
    log "OK running; latest_ckpt=$latest"
  else
    log "WARN train process missing; attempting resume"
    start_or_resume
  fi
done
