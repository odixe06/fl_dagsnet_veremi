#!/usr/bin/env bash
# Kéo output kernel Kaggle với vòng retry (CLI hay chết/treo giữa chừng — CONTEXT §9).
# usage: scripts/pull_output.sh <acct> <owner/slug> <dest-dir> [max-attempts]
#   PULL_PATTERN=<regex> (tuỳ chọn) → --file-pattern (re.search trên đường dẫn), vd. bỏ trọng số các round đã có ở phiên trước.
# Mỗi lượt: xoá file 0 byte (trừ *.done, __init__.py — hợp lệ 0 byte) → chạy `kaggle kernels output` (CLI bỏ qua file đã đủ
# kích thước) → giết khi KHÔNG nhận byte nào qua mạng > 4 phút, hoặc lượt vượt 3 giờ.
# (CLI đệm cả file trong RAM rồi mới ghi, nên file đích đứng ở 0 byte suốt lúc tải: với ~1 MB/s ở đây, weights 81–160 MB
# và preds 215 MB–1,08 GB đều "đứng yên" > 4 phút một cách hợp lệ — 22-09 lượt kéo 20c/50c bị giết vì thế.)
set -u
ACCT=$1; KERNEL=$2; DEST=$3; MAX=${4:-12}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KAS="$ROOT/../.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
mkdir -p "$DEST"
for ((i=1; i<=MAX; i++)); do
  find "$DEST" -type f -size 0 ! -name '*.done' ! -name '__init__.py' -delete
  echo "[$(date -u +%H:%M:%SZ)] attempt $i/$MAX: $KERNEL -> $DEST"
  python "$KAS" "$ACCT" -- kaggle kernels output "$KERNEL" -p "$DEST" --page-size 200 ${PULL_PATTERN:+--file-pattern "$PULL_PATTERN"} \
      > "${DEST}.cli.log" 2>&1 &
  pid=$!
  t0=$(date +%s); stuck=0; idle=0
  rx() { awk '/^ *(eth|ens|enp|wlan)[0-9]*:/ {s+=$2} END {print s+0}' /proc/net/dev; }
  rx_prev=$(rx)
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    if (( now - t0 > 10800 )); then echo "  lượt quá 3 giờ → kill"; pkill -P $pid; kill $pid; break; fi
    rx_now=$(rx); if (( rx_now - rx_prev < 65536 )); then idle=$((idle+30)); else idle=0; fi; rx_prev=$rx_now
    if (( idle >= 240 )); then
      stuck=1; echo "  không nhận byte nào > 4 phút → kill"; pkill -P $pid; kill $pid; break
    fi
  done
  wait $pid; rc=$?
  n=$(find "$DEST" -type f ! -name '.pull.log' | wc -l); z=$(find "$DEST" -type f -size 0 ! -name '*.done' ! -name '__init__.py' | wc -l)
  echo "  rc=$rc files=$n zero=$z  $(tail -1 "${DEST}.cli.log")"
  if [[ $rc -eq 0 && $z -eq 0 && $stuck -eq 0 ]]; then echo "PULL_OK $KERNEL"; exit 0; fi
  sleep 20
done
echo "PULL_FAIL $KERNEL"; exit 1
