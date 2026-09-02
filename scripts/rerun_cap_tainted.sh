#!/bin/bash
# Rerun cap-tainted final-run eval pairs. A FALLBACK, not part of the
# normal campaign loop.
#
# Identified 2026-04-14: every result below had $0 cost, 0 tokens, and
# sub-10s workflow duration — pure cap starvation, no real inference.
#
# Two groups:
#   1. Cloud baselines (Opus/Sonnet) — run via orchestrator eval directly
#   2. vLLM on RunPod — run via run_vllm_eval.py (creates pods)
#
# WHY A CAP-TAINTED RERUN SHOULD NOW BE RARE. An eval run no longer dies
# on the first cap it meets: run_architect_eval routes every invocation
# through invoke_with_cap_retry with 48h of patience, resuming the capped
# session via --resume on whichever account frees up first, so banked
# spend survives the failover. A trial reaching this script means that
# tolerance was exhausted, not merely tested. Reach for it when a campaign
# actually recorded cap_tainted trials — not as a routine second pass.
#
# THE ROSTER. Eval runs draw on the SHARED fleet pool
# (config/usage-accounts.yaml), the same seven accounts as every other
# invocation, and get their cap tolerance from the retry-plus-resume path
# above rather than from a private reserve. This script used to synthesise
# its own roster of "shared pool + max-a"; account A is reserved for
# INTERACTIVE use only — Leo's own sessions exhaust its weekly cap most
# weeks, so it never was the uncapped reserve that injection assumed
# (ruling 2026-08-30, tasks 4741/4945). USAGE_ACCOUNTS_FILE is still
# exported below: the override is how a run selects its roster, and the
# orchestrator config's own default is a hardcoded absolute path into the
# main checkout. What was retired is what it pointed AT.

set -uo pipefail

# Source .env for OAuth tokens
if [ -f /home/leo/src/dark-factory/.env ]; then
    set -a
    source /home/leo/src/dark-factory/.env
    set +a
fi

PYTHON=/home/leo/src/runpod-toolkit/.venv/bin/python
LAUNCHER=/home/leo/src/dark-factory/scripts/run_vllm_eval.py
LOGDIR=/var/tmp/dark-factory-evals
mkdir -p "$LOGDIR"

ORCH_TIMEOUT_MIN=150
TASK_TIMEOUT_MIN=180
CONCURRENCY=5

# ---- Group 1: Cloud baselines (no pod needed) ----
# These run orchestrator eval directly. Each (config, tasks) pair runs as
# a single orchestrator eval invocation.
#
# final-claude-opus-max:   reify_task_12 reify_task_27
# final-claude-sonnet-max: df_task_18 reify_task_12 reify_task_27

echo "[$(date +%H:%M:%S)] === Cloud baseline reruns ==="

# Point at the shared fleet pool verbatim — no synthesised roster. This
# mirrors run_vllm_eval.py's build_eval_env(), which was corrected in the
# same change (task 4945). The variable name is kept so the per-invocation
# override further down needs no restructuring.
EVAL_ACCOUNTS_FILE=/home/leo/src/dark-factory/config/usage-accounts.yaml
export USAGE_ACCOUNTS_FILE="$EVAL_ACCOUNTS_FILE"
echo "[$(date +%H:%M:%S)] Eval accounts file: $EVAL_ACCOUNTS_FILE"

CLOUD_PIDS=()

for pair in \
    "final-claude-opus-max:reify_task_12,reify_task_27" \
    "final-claude-sonnet-max:df_task_18,reify_task_12,reify_task_27" \
; do
    cfg="${pair%%:*}"
    tasks="${pair#*:}"
    LOG="$LOGDIR/rerun-$cfg-$(date +%Y%m%d-%H%M%S).log"
    echo "[$(date +%H:%M:%S)] LAUNCH cloud: $cfg × [$tasks] → $LOG"

    # Run each task sequentially within this config (cloud evals are fast
    # to start and the orchestrator handles its own timeouts).
    (
        IFS=','
        for task in $tasks; do
            TASK_PATH="orchestrator/src/orchestrator/evals/tasks/${task}.json"
            # reify tasks target a different project
            if [[ "$task" == reify_* ]]; then
                ORCH_CONFIG="/home/leo/src/reify/orchestrator.yaml"
            else
                ORCH_CONFIG="/home/leo/src/dark-factory/dark-factory-orchestrator.yaml"
            fi
            echo "[$(date +%H:%M:%S)] START $cfg × $task (config=$ORCH_CONFIG)"
            cd /home/leo/src/dark-factory
            USAGE_ACCOUNTS_FILE="$EVAL_ACCOUNTS_FILE" \
            uv run --project orchestrator orchestrator eval \
                --task "$TASK_PATH" \
                --config-name "$cfg" \
                --force \
                --timeout "$ORCH_TIMEOUT_MIN" \
                --config "$ORCH_CONFIG" \
                2>&1
            echo "[$(date +%H:%M:%S)] DONE  $cfg × $task (rc=$?)"
        done
    ) > "$LOG" 2>&1 &
    CLOUD_PIDS+=($!)
    sleep 2
done

# ---- Group 2: vLLM on RunPod ----
# These use run_vllm_eval.py which creates a RunPod pod per config.
#
# final-minimax-m27-fp8:       ALL 5 tasks  (uses :latest + HF download — new entrypoint)
# final-reap-139b-awq:         ALL 5 tasks  (every result was cap, no real data)
# final-qwen3-coder-next-fp8:  ALL 5 tasks  (only reify_27 is cap-only, but pod is
#                               going up anyway — marginal cost to run the rest)
#
# NOT rerunning final-reap-139b-fp8: all 5 tasks already have REAL results
# (128-1052 lines, $17-$79 cost). The cap results are just duplicates.

echo "[$(date +%H:%M:%S)] === vLLM RunPod reruns ==="

ALL_TASKS="df_task_12,df_task_13,df_task_18,reify_task_12,reify_task_27"
VLLM_PIDS=()
PORT=8200

for cfg in \
    final-minimax-m27-fp8 \
    final-reap-139b-awq \
    final-qwen3-coder-next-fp8 \
; do
    LOG="$LOGDIR/rerun-$cfg-$(date +%Y%m%d-%H%M%S).log"
    echo "[$(date +%H:%M:%S)] LAUNCH vLLM: $cfg × ALL → $LOG (port $PORT)"

    $PYTHON $LAUNCHER \
        --config "$cfg" \
        --tasks "$ALL_TASKS" \
        --concurrency "$CONCURRENCY" \
        --verify-baseline-clean warn \
        --task-timeout-min "$TASK_TIMEOUT_MIN" \
        --orch-timeout-min "$ORCH_TIMEOUT_MIN" \
        --port "$PORT" \
        > "$LOG" 2>&1 &

    VLLM_PIDS+=($!)
    PORT=$((PORT + 1))
    sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] All launched:"
echo "  Cloud PIDs: ${CLOUD_PIDS[*]}"
echo "  vLLM PIDs:  ${VLLM_PIDS[*]}"
echo ""
echo "  Tail any log: tail -f $LOGDIR/rerun-*.log"
echo ""

# Wait for all
ALL_PIDS=("${CLOUD_PIDS[@]}" "${VLLM_PIDS[@]}")
RC_AGG=0
for pid in "${ALL_PIDS[@]}"; do
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] pid=$pid OK"
    else
        rc=$?
        RC_AGG=1
        echo "[$(date +%H:%M:%S)] pid=$pid FAILED rc=$rc"
    fi
done

echo "[$(date +%H:%M:%S)] rerun complete; results under orchestrator/src/orchestrator/evals/results/"
exit "$RC_AGG"
