#!/usr/bin/env bash
# Predicate: did the legibility-transcript-check timer for <project_id> run
# ON SCHEDULE within the last <hours> hours? Exit 0 = alive, non-zero =
# escalate.
#
# Usage: check_transcript_check_liveness.sh <project_id> <hours>
#
# Mirrors check_trickle_liveness.sh (systemd UNIT STATE via `systemctl --user
# show`, NEVER git history) with ONE principled divergence (task 2901 DD4):
#
#   The transcript-persistence DETECTOR uses exit 1 as its NORMAL, expected
#   lost-transcript ALARM -- it found a missing transcript and escalated. A
#   probe that flagged exit 1 as unhealthy would conflate "detector working
#   and firing" with "detector broken". So this probe treats BOTH a clean run
#   (Result=success, ExecMainStatus=0) AND a firing run (Result=exit-code,
#   ExecMainStatus=1) as ALIVE. The liveness question for the detector is
#   purely "did it run on schedule?" => freshness + not-never-ran are the
#   load-bearing assertions.
#
#   Contrast check_trickle_liveness.sh, which requires Result=success &&
#   ExecMainStatus=0: for the trickle, exit 1 means the pipeline BROKE
#   (fail-loud), so its probe must reject it.
#
#   A genuinely broken timer is still caught here: an abnormal Result
#   (timeout / signal / core-dump / oom-kill / ...), or ExecMainStatus empty
#   or >= 2 (argparse error / malformed ExecStart), or never-ran, or
#   staleness all still FAIL.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: check_transcript_check_liveness.sh <project_id> <hours>" >&2
    exit 1
fi
PROJECT_ID="$1"
HOURS="$2"

SERVICE="legibility-transcript-check@${PROJECT_ID}.service"
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

# DD4 verdict: a clean run (success/0) and a firing lost-transcript run
# (exit-code/1) are both ALIVE; anything else is a broken timer, not merely a
# detector that fired.
if [ "$result" != "success" ] && [ "$result" != "exit-code" ]; then
    echo "ERROR: ${SERVICE} last run Result was '${result}' (expected success or exit-code)" >&2
    exit 1
fi

if [ "$exit_status" != "0" ] && [ "$exit_status" != "1" ]; then
    echo "ERROR: ${SERVICE} last run ExecMainStatus was '${exit_status}'" \
        "(expected 0=clean or 1=lost-transcript finding; empty/>=2 is a broken timer)" >&2
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
    echo "ERROR: ${SERVICE} last ran $((age_secs / 3600))h ago," \
        "exceeding the ${HOURS}h liveness window" >&2
    exit 1
fi

echo "OK: ${SERVICE} last ran $((age_secs / 3600))h ago (within ${HOURS}h)."
exit 0
