#!/usr/bin/env bash
set -euo pipefail

# Cycle-aware restart of fused-memory (task 2703 δ).
#
# Usage: restart-fused-memory.sh [--now|--drain] [--rebind-socket]
#
# Two independent choices. WHEN to restart (the gate, at most one flag):
#   (no gate flag)     # DEFAULT: defer while a full recon cycle is in
#                      #   flight, restart between cycles (or after the
#                      #   ~35-min cap)
#   --now              # bypass the gate, restart immediately
#   --drain            # signal SIGUSR1 to drain recon first, wait for the
#                      #   drained marker, restart
# HOW to restart:
#   (no flag)          # `systemctl --user restart fused-memory`
#   --rebind-socket    # adopt the committed fused-memory.socket template,
#                      #   then stop and start socket and service together
#
# --rebind-socket composes with --now or --drain, in either order. It refuses
# (exit 3) while fm-staleness-redeploy.service runs, asking before the chosen
# gate and again after it. Seconds before it stops then starts, it installs
# scripts/fused-memory.socket.template verbatim and daemon-reloads. Once
# /health answers, it exits with scripts/check_network_exposure.py's
# status. On a converged host it is
# idempotent: the same unit is installed and the same endpoints come back.
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
NETWORK_EXPOSURE_CHECK="${NETWORK_EXPOSURE_CHECK:-$SCRIPT_DIR/check_network_exposure.py}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# scripts/orchestrator-watchdog.py::FM_STALENESS_REDEPLOY_UNIT — a transient
# systemd-run unit, so `is-active` answers inactive when it is absent.
STALENESS_REDEPLOY_UNIT="fm-staleness-redeploy.service"

usage_error() {
    echo "ERROR: $1" >&2
    echo "Usage: restart-fused-memory.sh [--now|--drain] [--rebind-socket]" >&2
    exit 2
}

gate="recon"
rebind_socket=0

set_gate() {
    if [[ "$gate" != "recon" && "$gate" != "$1" ]]; then
        usage_error "--${gate} and --$1 are mutually exclusive"
    fi
    gate="$1"
}

for arg in "$@"; do
    case "$arg" in
        --now)           set_gate now ;;
        --drain)         set_gate drain ;;
        --rebind-socket) rebind_socket=1 ;;
        *)               usage_error "unknown argument: ${arg}" ;;
    esac
done

# The fused-memory tier has no restart lease, so a staleness redeploy in
# flight would restart the service again underneath the rebind.
refuse_while_staleness_redeploy_runs() {
    local state
    state="$(systemctl --user is-active "$STALENESS_REDEPLOY_UNIT" || true)"
    if [[ "$state" == "inactive" || "$state" == "failed" ]]; then
        return 0
    fi
    echo "rebind_socket: REFUSED: ${STALENESS_REDEPLOY_UNIT} is '${state:-no answer}'; the socket unit was not installed and fused-memory was not restarted." >&2
    echo "rebind_socket: re-run once \`systemctl --user is-active ${STALENESS_REDEPLOY_UNIT}\` answers inactive." >&2
    exit 3
}

# 0. --rebind-socket: refuse before a gate that can hold for RECON_GATE_TIMEOUT
#    or DRAIN_TIMEOUT.
if [[ "$rebind_socket" == 1 ]]; then
    refuse_while_staleness_redeploy_runs
fi

# 1. DEFAULT: defer while a full recon cycle is in flight.
if [[ "$gate" == "recon" ]]; then
    echo "recon_gate: polling ${HEALTH_URL} for in-flight reconciliation cycles (cap ${RECON_GATE_TIMEOUT}s)"
    gate_start=$SECONDS
    while true; do
        body="$(curl -s --max-time "$CURL_MAX_TIME" "$HEALTH_URL" || true)"
        gate_output="$(printf '%s' "$body" | python3 "$SCRIPT_DIR/recon_busy_check.py" || true)"
        # Pure-bash first-line extraction (no pipeline): a `head -n1`
        # consumer can lose the SIGPIPE race against a still-writing
        # producer once gate_output exceeds the pipe buffer (many
        # recon_busy_cycle lines), which under `pipefail` + `set -e` would
        # abort this script with exit 141 mid-gate, before the restart,
        # silently. A pipeline that can SIGPIPE at all has no business in a
        # `set -euo pipefail` gate, so this is hardened defensively — the
        # exact gate_output size behind the observed task 3838 exit-141
        # incident is not yet confirmed (recon_busy_snapshot() emits one
        # entry per concurrently in-flight full cycle, realistically 1-5;
        # see task 3838 follow-up).
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
if [[ "$gate" == "drain" ]]; then
    echo "Draining fused-memory recon (SIGUSR1 via systemctl kill)..."
    systemctl --user kill --kill-who=main --signal=SIGUSR1 "$SERVICE" \
        || echo "WARNING: failed to signal ${SERVICE}; proceeding to drain wait anyway." >&2

    drain_deadline=$((SECONDS + DRAIN_TIMEOUT))
    while [[ $SECONDS -lt $drain_deadline ]]; do
        # Pure-bash substring match (no pipeline): `grep -q` exits the
        # instant it finds a match, which can lose the SIGPIPE race against
        # a still-writing journalctl on a burst larger than the pipe
        # buffer (plausible during an active recon cycle — exactly when
        # --drain is used). Since this pipeline is an `if` condition,
        # `set -e` would not abort on that SIGPIPE — the condition would
        # just silently evaluate false, missing a marker it actually saw
        # and spinning to the full DRAIN_TIMEOUT (task 3838).
        journal="$(journalctl --user -u "$SERVICE" --since "10 seconds ago" --no-pager -q || true)"
        if [[ "$journal" == *"$DRAIN_MARKER"* ]]; then
            echo "${DRAIN_MARKER}."
            break
        fi
        sleep "$DRAIN_POLL_INTERVAL"
    done

    if [[ $SECONDS -ge $drain_deadline ]]; then
        echo "WARNING: Drain timed out after ${DRAIN_TIMEOUT}s, proceeding with restart anyway." >&2
    fi
fi

# 3. Restart. A socket that is already listening cannot rebind while the
#    service holds the old listener, and a combined restart of socket and
#    service is refused, so --rebind-socket stops then starts both
#    (OPERATIONS.md §"Socket activation (MCP ports survive restarts)").
#    The install and reload sit here, seconds before the start: after the
#    reload, a restart of fused-memory may no longer inherit a listener whose
#    endpoint the template changed.
if [[ "$rebind_socket" == 1 ]]; then
    refuse_while_staleness_redeploy_runs
    install -m 0644 "$SCRIPT_DIR/fused-memory.socket.template" "$UNIT_DIR/fused-memory.socket"
    systemctl --user daemon-reload
    echo "rebind_socket: installed $SCRIPT_DIR/fused-memory.socket.template as $UNIT_DIR/fused-memory.socket"
    echo "rebind_socket: stopping fused-memory (its ExecStopPost stops fused-memory.socket), then starting both"
    systemctl --user stop fused-memory
    systemctl --user start fused-memory.socket fused-memory
else
    echo "Restarting fused-memory..."
    systemctl --user restart "$SERVICE"
fi

# 4. Post-start /health verification (all paths).
wait_for_health() {
    echo -n "Waiting for health..."
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    while [[ $SECONDS -lt $deadline ]]; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            echo " OK"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " FAILED"
    echo "ERROR: fused-memory did not become healthy within ${HEALTH_TIMEOUT}s" >&2
    return 1
}

wait_for_health || exit 1

# 5. --rebind-socket post-condition: the exposure check's verdict is the
#    script's status.
if [[ "$rebind_socket" == 1 ]]; then
    echo "rebind_socket: post-condition: ${NETWORK_EXPOSURE_CHECK}"
    check_rc=0
    python3 "$NETWORK_EXPOSURE_CHECK" || check_rc=$?
    echo "rebind_socket: exposure check exited ${check_rc}"
    exit "$check_rc"
fi

echo "fused-memory restarted successfully."
exit 0
