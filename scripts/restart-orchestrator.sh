#!/usr/bin/env bash
set -euo pipefail

# Restart the orchestrator unit (default orchestrator-dark-factory.service)
# and verify a fresh MainPID.
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
#   restart-orchestrator.sh [--drain]
#
# --drain is accepted-and-ignored -- a no-op alias kept for generic-caller
# compatibility (mirrors restart-dashboard.sh's convention for services
# without a drain layer). The orchestrator has no SIGUSR1 drain handler
# (unlike fused-memory's --drain/"Harness fully drained" mechanism); its
# graceful drain is delegated entirely to SIGTERM -> cancel-main-task ->
# finally (cli.py _make_cancel_handler), bounded by the unit's
# TimeoutStopSec=90s. A plain `systemctl restart` already triggers that
# path, so there is nothing extra for --drain to do here.
#
# Env knobs (all optional, ${VAR:-default} style):
#   ORCH_RESTART_UNIT        the unit this script restarts
#                             (default: orchestrator-dark-factory.service)
#   RESTART_VERIFY_TIMEOUT   seconds to wait for a fresh MainPID (default: 30)
#
# ORCH_RESTART_UNIT, and deliberately NOT `ORCH_UNIT` (task 3950): every
# installed scripts/orchestrator-*.service sets
# `Environment=ORCH_UNIT=<its own unit name>`, so reading THAT variable would
# silently retarget this script at whichever orchestrator's cgroup happened to
# spawn it -- restarting the wrong unit, in production, with no test able to
# see it. `SELF_UNIT` is likewise taken, with a different meaning, by
# restart-all-orchestrators.sh.

# Test isolation is achieved by SETTING ORCH_RESTART_UNIT to a synthetic
# `orchestrator-fake*` unit, never by changing this default. The default is
# pinned against scripts/orchestrator-watchdog.py's WATCHED table and
# restart-all-orchestrators.sh's SELF_UNIT by
# tests/scripts/test_orchestrator_watchdog.py::test_restart_orchestrator_unit_default_matches_across_tiers
# (task 3950) -- that pin is the only remaining coverage of this value, since
# the two fake-systemctl harnesses now supply their own synthetic names. Read
# it before changing this line.
SERVICE="${ORCH_RESTART_UNIT:-orchestrator-dark-factory.service}"
FIELDS="MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic"
VERIFY_TIMEOUT="${RESTART_VERIFY_TIMEOUT:-30}"

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
    systemctl --user show -p "$FIELDS" "$SERVICE"
}

baseline="$(get_state)"
baseline_pid="$(read_field "$baseline" MainPID)"
baseline_mono="$(read_field "$baseline" ActiveEnterTimestampMonotonic)"

echo "Restarting ${SERVICE} (baseline MainPID=${baseline_pid})..."
systemctl --user restart "$SERVICE"

echo -n "Verifying fresh MainPID..."
deadline=$((SECONDS + VERIFY_TIMEOUT))
while [[ $SECONDS -lt $deadline ]]; do
    state="$(get_state)"
    pid="$(read_field "$state" MainPID)"
    active="$(read_field "$state" ActiveState)"
    mono="$(read_field "$state" ActiveEnterTimestampMonotonic)"
    # ActiveEnterTimestampMonotonic (+ ActiveState=active + a live pid) is the
    # authoritative freshness signal. PID inequality is NOT required: on rare
    # occasions the kernel reuses the old MainPID for the new process, which
    # would otherwise make a clean restart look spuriously stale.
    if [[ "$pid" -gt 0 && "$active" == "active" && "$mono" -gt "$baseline_mono" ]]; then
        echo " OK"
        echo "${SERVICE} restarted successfully (new MainPID=${pid})."
        exit 0
    fi
    sleep 1
done

echo " FAILED"
echo "ERROR: restart did not take (stale MainPID); ${SERVICE} did not come back within ${VERIFY_TIMEOUT}s" >&2
exit 1
