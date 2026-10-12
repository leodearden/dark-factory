#!/usr/bin/env bash
# Usage: run_phase.sh <phase> <tag> [probe args...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
phase="$1"; tag="$2"; shift 2
logs="$here/logs/$tag"; mkdir -p "$logs"
cd /home/leo/src/whisper-writer
source .venv/bin/activate
nvlibs=""
for d in .venv/lib/python*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && nvlibs="${nvlibs:+$nvlibs:}$(realpath "$d")"
done
export LD_LIBRARY_PATH="${nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 100 > "$logs/gpu.csv" &
s1=$!
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader,nounits -lms 250 > "$logs/apps.csv" &
s2=$!
trap 'kill $s1 $s2 2>/dev/null || true' EXIT
sleep 2
rc=0; python "$here/probe.py" --phase "$phase" --out "$logs/events.jsonl" "$@" > "$logs/probe.log" 2>&1 || rc=$?

sleep 3
echo "probe rc=$rc"
