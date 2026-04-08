#!/bin/bash
# Fire the full vLLM eval matrix: every RunPod-targetable config × every
# eval task. Each config gets its own RunPod pod, and all pods are fired
# IN PARALLEL — different pods have no resource conflict, so wallclock
# drops by the number of configs while total pod-hours (and therefore $)
# stays the same.
#
# Concurrency tuning:
#   - PER-POD concurrency defaults to 5 (the per-pod max — vLLM's
#     --max-num-seqs is 16 but reviewer fanout is the binding constraint).
#     5 pods × 5 concurrency = 25 simultaneous implementer slots and up
#     to 125 reviewer-call waves; that's why this requires the multi-
#     account Claude Max setup (5 uncapped plans available per the user).
#   - Wallclock at concurrency=5: each pod runs all 5 tasks in parallel,
#     so wallclock per pod ≈ ~30 min slowest task; all 5 pods running
#     concurrently → total wallclock ≈ ~30 min + pod cold-start.
#
# Cost: ~$1 per task × 5 tasks × 5 configs = ~$25 total. Same as
# sequential — only wallclock differs.
#
# Logs:
#   /var/tmp/dark-factory-evals/matrix-<config>-<timestamp>.log (this
#   script's per-pod log) and the launcher's per-task logs underneath.
#
# Usage:
#   bash scripts/run_eval_matrix.sh                  # all 5 configs × 5 tasks
#   CONFIGS=qwen3-coder-next-fp8-new bash ...        # single config
#   TASKS=df_task_12 bash ...                        # single task per config
#   CONCURRENCY=2 bash ...                           # bump per-pod concurrency

set -uo pipefail

# Default to all 5 vLLM configs and all 5 eval tasks. Override via env vars.
CONFIGS="${CONFIGS:-qwen3-coder-next-fp8-new reap-139b-nvfp4-new reap-172b-nvfp4-gb10-new minimax-m25-nvfp4-new minimax-m25-fp8-new}"
TASKS="${TASKS:-df_task_12,df_task_13,df_task_18,reify_task_12,reify_task_27}"
CONCURRENCY="${CONCURRENCY:-5}"
VERIFY="${VERIFY:-warn}"
# Inner orchestrator-internal timeout: how long the actual eval can run.
# Default 360 min (6h) — large enough that "would-be-done" runs have time
# to actually finish (matrix run #2 had several runs that hit tests/lint/
# typecheck=True at the 60 min default mark and only needed more time).
ORCH_TIMEOUT_MIN="${ORCH_TIMEOUT_MIN:-360}"
# Outer subprocess limit: must be > ORCH_TIMEOUT_MIN so the orchestrator
# hits its own timeout first and produces a clean result file rather than
# a hard SIGKILL with no result.
TIMEOUT_MIN="${TIMEOUT_MIN:-420}"

mkdir -p /var/tmp/dark-factory-evals

PIDS=()
declare -A LOG_FOR_PID
PORT=8200  # Each config gets its own port; incremented per launch.

for cfg in $CONFIGS; do
    LOG="/var/tmp/dark-factory-evals/matrix-$cfg-$(date +%Y%m%d-%H%M%S).log"
    echo "[$(date +%H:%M:%S)] LAUNCH $cfg → $LOG (port $PORT)"
    python3 /home/leo/src/dark-factory/scripts/run_vllm_eval.py \
        --config "$cfg" \
        --tasks "$TASKS" \
        --concurrency "$CONCURRENCY" \
        --verify-baseline-clean "$VERIFY" \
        --task-timeout-min "$TIMEOUT_MIN" \
        --orch-timeout-min "$ORCH_TIMEOUT_MIN" \
        --port "$PORT" \
        --no-volume \
        > "$LOG" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    LOG_FOR_PID[$pid]="$LOG"
    PORT=$((PORT + 1))
    # Stagger pod creates by 5s so RunPod's API doesn't see a thundering herd.
    sleep 5
done

echo "[$(date +%H:%M:%S)] all ${#PIDS[@]} configs launched in parallel; pids: ${PIDS[*]}"
echo "[$(date +%H:%M:%S)] tail any per-pod log to follow live; this script will wait."

# Wait for each pid and report.
RC_AGG=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] pid=$pid OK (${LOG_FOR_PID[$pid]})"
    else
        rc=$?
        RC_AGG=1
        echo "[$(date +%H:%M:%S)] pid=$pid FAILED rc=$rc — see ${LOG_FOR_PID[$pid]}"
    fi
done

echo "[$(date +%H:%M:%S)] matrix complete; results under orchestrator/src/orchestrator/evals/results/"
exit "$RC_AGG"
