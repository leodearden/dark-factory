#!/usr/bin/env bash
set -euo pipefail

# Restart orchestrator-dark-factory.service and verify a fresh MainPID.
#
# INTENDED CALLER: an operator shell, or a task_kind='deterministic' deploy
# task's `before_done.script`, invoked OUTSIDE the orchestrator's own
# systemd cgroup (e.g. an external host/session). This script performs a
# BLOCKING `systemctl --user restart` and then externally verifies a fresh
# MainPID -- it must NEVER be exec'd from *inside* the orchestrator process
# it is restarting: under this unit's default KillMode=control-group, a
# spawned child is killed alongside the parent before it can observe the
# service coming back. A true in-process self-restart needs the
# cgroup-escaping `systemd-run --user` pattern in
# orchestrator/src/orchestrator/deterministic_runner.py
# (_default_schedule_detached_restart), and even that cannot self-verify.
#
# Usage:
#   restart-orchestrator.sh

SERVICE="orchestrator-dark-factory.service"
FIELDS="MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic"

if [[ $# -gt 0 ]]; then
    echo "ERROR: unexpected argument: $1" >&2
    exit 1
fi

read_field() {
    # $1 = `systemctl show` output blob, $2 = field name
    printf '%s\n' "$1" | grep "^$2=" | cut -d= -f2-
}

get_state() {
    systemctl --user show -p "$FIELDS" "$SERVICE"
}

baseline="$(get_state)"
baseline_pid="$(read_field "$baseline" MainPID)"

echo "Restarting ${SERVICE} (baseline MainPID=${baseline_pid})..."
systemctl --user restart "$SERVICE"
