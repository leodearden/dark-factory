#!/usr/bin/env bash
# Predicate: did legibility-trickle-health@<project_id>.service run within
# the last <hours> hours AND reach a verdict? Exit 0 = alive, non-zero =
# escalate.
#
# Usage: check_trickle_health_liveness.sh <project_id> <hours>
#
# Probes systemd UNIT STATE (`systemctl --user show`), never git history,
# for check_trickle_liveness.sh's decision-7 reason.
#
# ALIVE is exactly success/0 (healthy) or exit-code/3 (a DEGRADED verdict).
# 3 is scripts/legibility/check_trickle_health.py::EXIT_DEGRADED, its single
# definition; scripts/tests/test_check_trickle_health_liveness.py pins this
# literal against it. Rationale: OPERATIONS.md §"Legibility trickle health
# probe (04:30)".
#
# Bound by: the delayed-milestone predicates task 10045 files once this
# script is on main. Task 10045 replaces this with the filed ids.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: check_trickle_health_liveness.sh <project_id> <hours>" >&2
    exit 1
fi
PROJECT_ID="$1"
HOURS="$2"

SERVICE="legibility-trickle-health@${PROJECT_ID}.service"
FIELDS="Result,ExecMainStatus,ExecMainExitTimestamp"

read_field() {
    # $1 = `systemctl show` output blob, $2 = field name
    printf '%s\n' "$1" | grep "^$2=" | cut -d= -f2-
}

state="$(systemctl --user show -p "$FIELDS" "$SERVICE")"
result="$(read_field "$state" Result)"
exit_status="$(read_field "$state" ExecMainStatus)"
timestamp="$(read_field "$state" ExecMainExitTimestamp)"

# Before `date -d`: GNU date reads an empty string as "today", and a
# never-started instance of a loaded template reports Result=success,
# ExecMainStatus=0 with an empty timestamp.
if [ -z "$timestamp" ] || [ "$timestamp" = "n/a" ]; then
    echo "ERROR: ${SERVICE} has never run (no ExecMainExitTimestamp)" >&2
    exit 1
fi

case "${result}/${exit_status}" in
    success/0|exit-code/3) ;;
    *)
        echo "ERROR: ${SERVICE} last run was Result='${result}'" \
            "ExecMainStatus='${exit_status}' (expected success/0 = healthy or" \
            "exit-code/3 = DEGRADED verdict; anything else means the health" \
            "unit itself is broken)" >&2
        exit 1
        ;;
esac

if ! epoch="$(date -d "$timestamp" +%s 2>/dev/null)"; then
    echo "ERROR: could not parse ExecMainExitTimestamp '${timestamp}' for ${SERVICE}" >&2
    exit 1
fi

now="$(date +%s)"
age_secs=$((now - epoch))
max_secs=$((HOURS * 3600))

if [ "$age_secs" -gt "$max_secs" ]; then
    echo "ERROR: ${SERVICE} last ran $((age_secs / 3600))h ago," \
        "exceeding the ${HOURS}h liveness window" >&2
    exit 1
fi

echo "OK: ${SERVICE} last ran $((age_secs / 3600))h ago (within ${HOURS}h)" \
    "with Result='${result}' ExecMainStatus='${exit_status}'."
exit 0
