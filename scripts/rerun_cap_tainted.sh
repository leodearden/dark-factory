#!/bin/bash
# Rerun all cap-tainted final-run eval pairs.
#
# Identified 2026-04-14: every result below had $0 cost, 0 tokens, and
# sub-10s workflow duration — pure cap starvation, no real inference.
#
# Two groups:
#   1. Cloud baselines (Opus/Sonnet) — run via orchestrator eval directly
#   2. vLLM on RunPod — run via run_vllm_eval.py (creates pods)
#
# The USAGE_ACCOUNTS_FILE mechanism (d33db0c) ensures eval runs get
# account A in addition to the shared pool.

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

# Generate an eval accounts file that includes account A (eval-only).
# Without this, cloud evals use only the shared pool which may be capped.
# This mirrors what run_vllm_eval.py's build_eval_env() does.
EVAL_ACCOUNTS_FILE=$(python3 -c "
import yaml, tempfile, os
base = '/home/leo/src/dark-factory/config/usage-accounts.yaml'
with open(base) as f:
    data = yaml.safe_load(f)
accounts = list(data.get('accounts', []))
# Add max-a via env var reference (oauth_token_env, not direct oauth_token)
# to match the Pydantic schema that requires the oauth_token_env field.
if not any(a.get('name') == 'max-a' for a in accounts):
    accounts.append({'name': 'max-a', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_A'})
fd, path = tempfile.mkstemp(suffix='.yaml', prefix='eval-accounts-')
with os.fdopen(fd, 'w') as f:
    yaml.safe_dump({'accounts': accounts}, f)
print(path)
")
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
                ORCH_CONFIG="/home/leo/src/dark-factory/orchestrator/config.yaml"
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
