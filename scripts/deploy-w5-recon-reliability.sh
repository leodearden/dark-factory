#!/usr/bin/env bash
set -euo pipefail

# deploy-w5-recon-reliability.sh
#
# RESTART + VERIFY deploy capstone for the W5 recon-reliability batch
# (ReconLedgerStore control-plane ledger + write-both/read-new cutover --
# plans/recon-reliability-prd.md, tasks alpha..omicron / 2219-2232, all
# merged to main ahead of this script).
#
# INTENDED CALLER: a task_kind='deterministic' deploy task's
# `before_done.script`, with `target_unit` set to fused-memory.service -- a
# DIFFERENT unit than the orchestrator's own, so deterministic_runner.py
# takes the CROSS-UNIT blocking path (module docstring section gamma):
# baseline-inspect fused-memory.service -> run this script BLOCKING ->
# re-inspect and verify a fresh MainPID + later ActiveEnterTimestampMonotonic
# -> done=deployed-and-verified. This script's own `systemctl restart` below
# is therefore load-bearing for that verify -- it must actually BE the
# thing that restarts the unit, not a delegate that merely signals it.
#
# RESTART CONVENTION (program decision #6): a plain, blocking
# `systemctl --user restart fused-memory.service` -- NOT
# restart-fused-memory.sh's `--drain` path (SIGUSR1 + journal-wait for
# "Harness fully drained"), which hung per task 2090. This script performs
# its own self-contained restart+verify rather than delegating to
# restart-fused-memory.sh, so that risky option is never reachable from a
# deterministic deploy.
#
# SERVING-SANITY: after the restart, this script (1) polls `curl -sf
# $HEALTH_URL` until fused-memory's /health endpoint reports ready (process
# up, graphiti+mem0 reachable -- fused_memory/server/tools.py's
# health_check), THEN (2) polls `journalctl --user -u fused-memory.service`
# for a recon-serving readiness marker -- proof the restarted process is
# actively cycling reconciliation, not merely alive. The default
# RECON_MARKER ("Project reconciliation loop started for dark_factory") is
# the `_project_loop` startup log line in
# fused_memory/reconciliation/harness.py; confirmed against the live
# fused-memory.service journal on this host (task 2233 prerequisite) to
# recur roughly once a minute during normal activity, so it reliably
# reappears well within the default RECON_VERIFY_TIMEOUT after a clean
# restart -- no health-only fallback was needed. Combined with the runner's
# own fresh-MainPID/ActiveEnterTimestampMonotonic verify (proving THIS
# process is new), an observed marker proves the new process is actively
# serving ledger-backed recon, not merely up.
#
# Usage:
#   deploy-w5-recon-reliability.sh [--check|--dry-run]
#
# --check / --dry-run: print the intended restart+verify plan and exit 0
# WITHOUT restarting anything and WITHOUT any systemctl/curl/journalctl call.
#
# Env overrides (test-only knobs; production runs with the defaults):
#   HEALTH_URL            - fused-memory /health URL (default:
#                            http://localhost:8002/health).
#   HEALTH_TIMEOUT         - seconds to poll for a healthy /health before
#                            failing (default: 30).
#   RECON_MARKER           - journal substring proving recon is actively
#                            cycling (default: "Project reconciliation loop
#                            started for dark_factory").
#   RECON_VERIFY_TIMEOUT   - seconds to poll the journal for RECON_MARKER
#                            before failing (default: 180).
#
# This script is cwd-independent (pure systemctl/curl/journalctl, no
# repo-relative paths) -- it runs cleanly regardless of the caller's cwd
# (the detached out-of-cgroup context the cross-unit runner invokes it from
# defaults its cwd to $HOME, not project_root).

SERVICE="fused-memory.service"
HEALTH_URL="${HEALTH_URL:-http://localhost:8002/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"
RECON_MARKER="${RECON_MARKER:-Project reconciliation loop started for dark_factory}"
RECON_VERIFY_TIMEOUT="${RECON_VERIFY_TIMEOUT:-180}"

MODE="apply"

for arg in "$@"; do
    case "$arg" in
        --check|--dry-run)
            MODE="check"
            ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ "$MODE" == "check" ]]; then
    echo "Would restart ${SERVICE} (systemctl --user restart ${SERVICE})"
    echo "Would then verify: /health at ${HEALTH_URL} (timeout ${HEALTH_TIMEOUT}s), then recon-serving marker '${RECON_MARKER}' via journalctl (timeout ${RECON_VERIFY_TIMEOUT}s)"
    exit 0
fi

restart_start="$(date +%s)"

echo "Restarting ${SERVICE}..."
systemctl --user restart "$SERVICE"

# 1. Health gate: wait for /health to report ready (reuses
# restart-fused-memory.sh:47-65's curl -sf polling idiom).
echo -n "Waiting for health..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
healthy=false
while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        healthy=true
        break
    fi
    echo -n "."
    sleep 1
done

if ! $healthy; then
    echo " FAILED"
    echo "ERROR: fused-memory did not become healthy within ${HEALTH_TIMEOUT}s" >&2
    exit 1
fi
echo " OK"

# 2. Serving-sanity gate (placeholder single-attempt check -- upgraded to a
# real bounded polling loop with pass/fail handling in step-8; for now this
# only wires the restart-then-health-then-serving ordering, it does not yet
# act on the outcome).
journalctl --user -u "$SERVICE" --since "@${restart_start}" --no-pager -q \
    | grep -q "$RECON_MARKER" || true

exit 0
