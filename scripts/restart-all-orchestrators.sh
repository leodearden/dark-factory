#!/usr/bin/env bash
set -euo pipefail

# Restart EVERY running orchestrator-*.service user unit and verify a fresh
# ActiveEnterTimestampMonotonic for each.
#
# All orchestrator units run the same orchestrator package from this repo, so
# a change to orchestrator/src must be deployed by restarting all of them —
# not only orchestrator-dark-factory (see restart-orchestrator.sh for the
# single-unit variant this generalizes).
#
# INTENDED CALLER: an operator shell, or a task_kind='deterministic' deploy
# task's `before_done.script` with target_unit='orchestrator-dark-factory.service'.
# Because the unit list includes the dark-factory orchestrator itself, the
# deploy task MUST set target_unit to the dark-factory unit so the
# DeterministicRunner routes execution through its cgroup-escaping detached
# `systemd-run --user` path (deterministic_runner.py,
# _default_schedule_detached_restart; done = 'scheduled'). A BLOCKING
# invocation from inside any orchestrator's own cgroup would be killed
# mid-script under KillMode=control-group — same constraint documented in
# restart-orchestrator.sh, extended here to every unit in the list.
#
# Defensive ordering: the invoking orchestrator's own unit (default
# orchestrator-dark-factory.service, override via SELF_UNIT) is restarted
# LAST, so if this script is ever mistakenly run blocking from inside that
# unit, every other orchestrator still gets restarted before the caller dies.
#
# Usage:
#   restart-all-orchestrators.sh [--drain]
#
# --drain enables the per-unit merge-drain gate (task 2397, γ of the
# orchestrator fleet-redeploy PRD): before restarting each unit, its
# α-produced (task 2395) merge-idle heartbeat
# (orchestrator/src/orchestrator/fleet_heartbeat.py) is read via
# scripts/drain_check.py, classified into idle/busy/stale/absent. Two
# deliberately-opposite fail directions:
#   - busy (fresh, mid-merge): restart is DEFERRED — rechecked every
#     ORCH_DRAIN_POLL_INTERVAL_SECS — until the unit drains (goes idle) or
#     ORCH_RESTART_FORCE_FIRE_AFTER_SECS elapses, at which point the restart
#     is FORCED anyway (one re-verified merge accepted; recover_pending_merges
#     makes this crash-safe). Fails toward PROTECTING the merge.
#   - stale/absent (heartbeat missing, unreadable, or too old): given a
#     shorter ORCH_DRAIN_UNKNOWN_GRACE_SECS grace to start reporting,
#     rechecked every ORCH_DRAIN_POLL_INTERVAL_SECS; a fresh idle/busy
#     reading that appears during the grace re-classifies into the idle/busy
#     handling above, and if the heartbeat is still stale/absent once the
#     grace elapses, the restart proceeds anyway. Fails toward CONVERGENCE
#     (the opposite direction from busy — a not-reporting unit must not block
#     the fleet restart forever).
# Without --drain, the script behaves exactly as before: an immediate,
# uncapped restart-all with zero heartbeat reads.
#
# Env knobs (all optional, ${VAR:-default} style):
#   ORCH_FLEET_DIR                       fleet-common heartbeat dir
#                                         (default: /home/leo/src/dark-factory/data/fleet)
#   ORCH_RESTART_FORCE_FIRE_AFTER_SECS    busy-grace before a forced restart
#                                         (default: 4500 = 75m)
#   ORCH_DRAIN_FRESH_WINDOW_SECS          heartbeat freshness window
#                                         (default: 120 = 2x run-loop tick)
#   ORCH_DRAIN_POLL_INTERVAL_SECS         re-check interval while deferring
#                                         (busy) or awaiting a stale/absent
#                                         heartbeat (default: 30)
#   ORCH_DRAIN_UNKNOWN_GRACE_SECS         grace for a stale/absent heartbeat
#                                         before proceeding anyway (default: 120)

FIELDS="MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic"
VERIFY_TIMEOUT="${RESTART_VERIFY_TIMEOUT:-30}"
SELF_UNIT="${SELF_UNIT:-orchestrator-dark-factory.service}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_DIR="${ORCH_FLEET_DIR:-/home/leo/src/dark-factory/data/fleet}"
FORCE_FIRE_AFTER_SECS="${ORCH_RESTART_FORCE_FIRE_AFTER_SECS:-4500}"
DRAIN_FRESH_WINDOW_SECS="${ORCH_DRAIN_FRESH_WINDOW_SECS:-120}"
DRAIN_POLL_INTERVAL_SECS="${ORCH_DRAIN_POLL_INTERVAL_SECS:-30}"
DRAIN_UNKNOWN_GRACE_SECS="${ORCH_DRAIN_UNKNOWN_GRACE_SECS:-120}"

DRAIN_ENABLED=0
for arg in "$@"; do
    case "$arg" in
        --drain)
            DRAIN_ENABLED=1
            ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            exit 1
            ;;
    esac
done

read_field() {
    # $1 = `systemctl show` output blob, $2 = field name
    printf '%s\n' "$1" | grep "^$2=" | cut -d= -f2-
}

get_state() {
    # $1 = unit name
    systemctl --user show -p "$FIELDS" "$1"
}

restart_and_verify() {
    # $1 = unit name.  Returns 0 on verified-fresh restart, 1 otherwise.
    local unit="$1"
    local baseline baseline_pid baseline_mono state pid active mono deadline
    baseline="$(get_state "$unit")"
    baseline_pid="$(read_field "$baseline" MainPID)"
    baseline_mono="$(read_field "$baseline" ActiveEnterTimestampMonotonic)"

    echo "Restarting ${unit} (baseline MainPID=${baseline_pid})..."
    systemctl --user restart "$unit"

    echo -n "Verifying fresh start for ${unit}..."
    deadline=$((SECONDS + VERIFY_TIMEOUT))
    while [[ $SECONDS -lt $deadline ]]; do
        state="$(get_state "$unit")"
        pid="$(read_field "$state" MainPID)"
        active="$(read_field "$state" ActiveState)"
        mono="$(read_field "$state" ActiveEnterTimestampMonotonic)"
        # ActiveEnterTimestampMonotonic (+ ActiveState=active + a live pid) is
        # the authoritative freshness signal.  PID inequality is NOT required:
        # the kernel can reuse the old MainPID for the new process, which
        # would make a clean restart look spuriously stale.
        if [[ "$pid" -gt 0 && "$active" == "active" && "$mono" -gt "$baseline_mono" ]]; then
            echo " OK (new MainPID=${pid})"
            return 0
        fi
        sleep 1
    done
    echo " FAILED"
    return 1
}

drain_check_verdict() {
    # $1 = unit name.  Prints one of idle/busy/stale/absent to stdout.
    python3 "$SCRIPT_DIR/drain_check.py" --unit "$1" --fleet-dir "$FLEET_DIR" \
        --fresh-window "$DRAIN_FRESH_WINDOW_SECS"
}

drain_gate() {
    # $1 = unit name.  Only called when --drain was passed.  Blocks
    # (poll-and-recheck) until it is safe to restart $1.
    #
    # - idle (fresh, merge_idle) returns immediately (transparent).
    # - stale/absent (heartbeat missing or too old) is given a bounded
    #   ORCH_DRAIN_UNKNOWN_GRACE_SECS grace, re-polling every
    #   DRAIN_POLL_INTERVAL_SECS: a fresh idle/busy reading that appears
    #   during the grace re-classifies into the branch below; if the grace
    #   elapses with the heartbeat still stale/absent, the restart proceeds
    #   anyway (fail-toward-convergence -- the opposite fail direction from
    #   a confirmed-busy unit, which fails toward protecting the merge).
    # - busy (fresh, mid-merge) defers with a journal line and polls every
    #   DRAIN_POLL_INTERVAL_SECS until the unit drains (idle) or
    #   FORCE_FIRE_AFTER_SECS elapses, at which point it force-proceeds
    #   anyway.
    local unit="$1"
    local verdict start_secs elapsed grace_start elapsed_grace
    verdict="$(drain_check_verdict "$unit")"

    grace_start=$SECONDS
    while [[ "$verdict" == "stale" || "$verdict" == "absent" ]]; do
        elapsed_grace=$((SECONDS - grace_start))
        if [[ $elapsed_grace -ge $DRAIN_UNKNOWN_GRACE_SECS ]]; then
            echo "proceeding with restart of ${unit}: heartbeat ${verdict} after ${DRAIN_UNKNOWN_GRACE_SECS}s grace"
            return 0
        fi
        sleep "$DRAIN_POLL_INTERVAL_SECS"
        verdict="$(drain_check_verdict "$unit")"
    done

    if [[ "$verdict" == "idle" ]]; then
        return 0
    fi

    # verdict == "busy" here: idle returned above, and the stale/absent
    # grace loop only exits once a fresh idle/busy reading appears.
    echo "deferring restart of ${unit}: mid-merge (grace $((FORCE_FIRE_AFTER_SECS / 60))m)"
    start_secs=$SECONDS
    while true; do
        elapsed=$((SECONDS - start_secs))
        if [[ $elapsed -ge $FORCE_FIRE_AFTER_SECS ]]; then
            echo "force-restarting ${unit}: mid-merge grace of ${FORCE_FIRE_AFTER_SECS}s exceeded"
            return 0
        fi
        sleep "$DRAIN_POLL_INTERVAL_SECS"
        verdict="$(drain_check_verdict "$unit")"
        if [[ "$verdict" == "idle" ]]; then
            echo "resuming restart of ${unit}: drained"
            return 0
        fi
    done
}

# Enumerate running orchestrator units at run time (robust to which projects
# are enabled on this host), deferring SELF_UNIT to the end.
mapfile -t running_units < <(
    systemctl --user list-units 'orchestrator-*.service' \
        --state=running --no-legend --plain 2>/dev/null | awk '{print $1}'
)

if [[ ${#running_units[@]} -eq 0 ]]; then
    echo "No running orchestrator-*.service units found; nothing to restart."
    exit 0
fi

ordered_units=()
self_present=0
for unit in "${running_units[@]}"; do
    if [[ "$unit" == "$SELF_UNIT" ]]; then
        self_present=1
    else
        ordered_units+=("$unit")
    fi
done
if [[ $self_present -eq 1 ]]; then
    ordered_units+=("$SELF_UNIT")
fi

echo "Restarting ${#ordered_units[@]} orchestrator unit(s): ${ordered_units[*]}"

failures=()
for unit in "${ordered_units[@]}"; do
    if [[ $DRAIN_ENABLED -eq 1 ]]; then
        drain_gate "$unit"
    fi
    if ! restart_and_verify "$unit"; then
        failures+=("$unit")
    fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
    echo "ERROR: restart did not verify fresh for: ${failures[*]}" >&2
    exit 1
fi

echo "All ${#ordered_units[@]} orchestrator unit(s) restarted and verified fresh."
exit 0
