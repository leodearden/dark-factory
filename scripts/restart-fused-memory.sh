#!/usr/bin/env bash
set -euo pipefail

# Cycle-aware restart of fused-memory (task 2703 δ).
#
# Usage:
#   restart-fused-memory.sh            # DEFAULT: defer while a full recon
#                                      #   cycle is in flight, restart between
#                                      #   cycles (or after the ~35-min cap)
#   restart-fused-memory.sh --now      # bypass the gate, restart immediately
#   restart-fused-memory.sh --drain    # signal SIGUSR1 to drain recon first,
#                                      #   wait for the drained marker, restart
#
# The DEFAULT path polls /health and pipes the body to recon_busy_check.py:
# while a full reconciliation cycle is running (recon_busy non-empty) it
# defers, up to RECON_GATE_TIMEOUT, then proceeds. An unreachable/unreadable
# /health is treated as NOT busy (fail-safe proceed) rather than wedging the
# restart. All paths keep the post-start /health verification.

HEALTH_URL="${HEALTH_URL:-http://localhost:8002/health}"
SERVICE="fused-memory"
DRAIN_MARKER="Harness fully drained"

# Recon defer gate (DEFAULT path). Cap basis: longest observed full cycle
# 29:58 (run 97b49a64) plus headroom. All env-overridable so tests can inject
# short caps/intervals to exercise the cap/defer paths in seconds.
RECON_GATE_TIMEOUT="${RECON_GATE_TIMEOUT:-2100}"          # 35 min cap
RECON_GATE_POLL_INTERVAL="${RECON_GATE_POLL_INTERVAL:-15}"
CURL_MAX_TIME="${CURL_MAX_TIME:-5}"

# --drain: cycle-scale wait for the drained marker (replaces the old 120s,
# which was sized for the idle fast path; a genuine in-flight cycle runs
# 9-30+ min — task 2702's drain now converges).
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-2100}"
DRAIN_POLL_INTERVAL="${DRAIN_POLL_INTERVAL:-5}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="default"
case "${1:-}" in
    "")       mode="default" ;;
    --now)    mode="now" ;;
    --drain)  mode="drain" ;;
    *)
        echo "ERROR: unknown argument: ${1}" >&2
        echo "Usage: restart-fused-memory.sh [--now|--drain]" >&2
        exit 2
        ;;
esac

# 1. DEFAULT: defer while a full recon cycle is in flight.
if [[ "$mode" == "default" ]]; then
    echo "recon_gate: polling ${HEALTH_URL} for in-flight reconciliation cycles (cap ${RECON_GATE_TIMEOUT}s)"
    gate_start=$SECONDS
    while true; do
        body="$(curl -s --max-time "$CURL_MAX_TIME" "$HEALTH_URL" || true)"
        gate_output="$(printf '%s' "$body" | python3 "$SCRIPT_DIR/recon_busy_check.py" || true)"
        # Pure-bash first-line extraction (no pipeline): `head -n1` can lose
        # the SIGPIPE race against a still-writing producer once gate_output
        # is large (many recon_busy_cycle lines), which under `pipefail` +
        # `set -e` aborted this script with exit 141 mid-gate (task 3838).
        verdict="${gate_output%%$'\n'*}"
        elapsed=$((SECONDS - gate_start))

        if [[ "$verdict" != "busy" ]]; then
            echo "recon_gate: clear (verdict=${verdict}) after ${elapsed}s — proceeding"
            break
        fi

        if [[ $elapsed -ge $RECON_GATE_TIMEOUT ]]; then
            echo "recon_gate: cap reached (${elapsed}s >= ${RECON_GATE_TIMEOUT}s) — proceeding anyway"
            break
        fi

        echo "recon_gate: deferring — reconciliation cycle in flight (elapsed ${elapsed}s / cap ${RECON_GATE_TIMEOUT}s)"
        printf '%s\n' "$gate_output" | grep '^recon_busy_cycle' || true
        sleep "$RECON_GATE_POLL_INTERVAL"
    done
fi

# 2. --drain: signal the harness to stop new cycles and finish current ones,
#    then wait for the drained marker before restarting.
if [[ "$mode" == "drain" ]]; then
    echo "Draining fused-memory recon (SIGUSR1 via systemctl kill)..."
    systemctl --user kill --kill-who=main --signal=SIGUSR1 "$SERVICE" \
        || echo "WARNING: failed to signal ${SERVICE}; proceeding to drain wait anyway." >&2

    drain_deadline=$((SECONDS + DRAIN_TIMEOUT))
    while [[ $SECONDS -lt $drain_deadline ]]; do
        if journalctl --user -u "$SERVICE" --since "10 seconds ago" --no-pager -q \
            | grep -q "$DRAIN_MARKER"; then
            echo "${DRAIN_MARKER}."
            break
        fi
        sleep "$DRAIN_POLL_INTERVAL"
    done

    if [[ $SECONDS -ge $drain_deadline ]]; then
        echo "WARNING: Drain timed out after ${DRAIN_TIMEOUT}s, proceeding with restart anyway." >&2
    fi
fi

# 3. Restart.
echo "Restarting fused-memory..."
systemctl --user restart "$SERVICE"

# 4. Post-start /health verification (all paths).
echo -n "Waiting for health..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo " OK"
        echo "fused-memory restarted successfully."
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " FAILED"
echo "ERROR: fused-memory did not become healthy within ${HEALTH_TIMEOUT}s" >&2
exit 1
