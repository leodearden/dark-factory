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
# --drain is accepted-and-ignored (generic-caller compatibility; see
# restart-orchestrator.sh — orchestrator graceful drain is delegated to
# SIGTERM via `systemctl restart`, bounded by TimeoutStopSec).

FIELDS="MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic"
VERIFY_TIMEOUT="${RESTART_VERIFY_TIMEOUT:-30}"
SELF_UNIT="${SELF_UNIT:-orchestrator-dark-factory.service}"

for arg in "$@"; do
    case "$arg" in
        --drain)
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
