#!/usr/bin/env bash
# Predicate: did the legibility-trickle timer for <project_id> run
# successfully within the last <hours> hours? Exit 0 = alive, non-zero =
# escalate.
#
# Usage: check_trickle_liveness.sh <project_id> <hours>
#
# Probes systemd UNIT STATE (`systemctl --user show`), NEVER git history
# (PRD decision 7): a quiet night that finds nothing to code commits
# nothing (see nightly.py's no-change-night gate), so a git-history probe
# would false-alarm on a healthy-but-quiet timer. ExecMainExitTimestamp /
# Result / ExecMainStatus are the authoritative last-run signal instead.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: check_trickle_liveness.sh <project_id> <hours>" >&2
    exit 1
fi
PROJECT_ID="$1"
HOURS="$2"

SERVICE="legibility-trickle@${PROJECT_ID}.service"
FIELDS="Result,ExecMainStatus,ExecMainExitTimestamp"

# read_field/get_state: mirrors restart-orchestrator.sh's identical
# `systemctl --user show -p "$FIELDS" <unit>` + grep/cut convention.
read_field() {
    # $1 = `systemctl show` output blob, $2 = field name
    printf '%s\n' "$1" | grep "^$2=" | cut -d= -f2-
}

state="$(systemctl --user show -p "$FIELDS" "$SERVICE")"
result="$(read_field "$state" Result)"
exit_status="$(read_field "$state" ExecMainStatus)"
timestamp="$(read_field "$state" ExecMainExitTimestamp)"

# The never-ran case: an unset property answers as an empty string (real
# systemd behavior); "n/a" is handled defensively too. Checked BEFORE
# `date -d` -- GNU date silently accepts an empty string as "today", which
# would otherwise let a never-ran service slip under the age threshold.
if [ -z "$timestamp" ] || [ "$timestamp" = "n/a" ]; then
    echo "ERROR: ${SERVICE} has never run (no ExecMainExitTimestamp)" >&2
    exit 1
fi

if [ "$result" != "success" ]; then
    echo "ERROR: ${SERVICE} last run Result was '${result}' (expected success)" >&2
    exit 1
fi

if [ "$exit_status" != "0" ]; then
    echo "ERROR: ${SERVICE} last run ExecMainStatus was '${exit_status}' (expected 0)" >&2
    exit 1
fi

if ! epoch="$(date -d "$timestamp" +%s 2>/dev/null)"; then
    echo "ERROR: could not parse ExecMainExitTimestamp '${timestamp}' for ${SERVICE}" >&2
    exit 1
fi

now="$(date +%s)"
age_secs=$((now - epoch))
max_secs=$((HOURS * 3600))

if [ "$age_secs" -gt "$max_secs" ]; then
    echo "ERROR: ${SERVICE} last ran successfully $((age_secs / 3600))h ago," \
        "exceeding the ${HOURS}h liveness window" >&2
    exit 1
fi

echo "OK: ${SERVICE} last ran successfully $((age_secs / 3600))h ago (within ${HOURS}h)."
exit 0
