#!/usr/bin/env bash
# check_merge_flakiness.sh — exemplar predicate for a milestone 'predicate' check.
#
# A dependency-free, deterministic exit-code VERDICT check: compares an
# injected --value against a --threshold and reports whether the invariant
# holds (exit 0) or is violated (exit 1).  A real deployment would compute
# --value from logs (out of scope — PRD §12); this script only owns the
# threshold comparison and the verdict contract.  The orchestrator parses no
# output — only the exit code (PRD §5.5).
#
# Usage:
#   check_merge_flakiness.sh --threshold RATE --value RATE [--window-days N] [--sleep-secs N]
#
# Flags:
#   --window-days N   Accepted for realism (documents the log window a real
#                      deployment would scan); unused by this exemplar.
#   --threshold RATE  Max acceptable flakiness rate before the check reports
#                      VIOLATED.
#   --value RATE      The measured flakiness rate to compare against
#                      --threshold.
#   --sleep-secs N    Sleep N seconds before comparing (default 0) — lets the
#                      SAME fixture drive a check timeout under test.
#
# Exit codes:
#   0  invariant holds    (value < threshold)
#   1  invariant VIOLATED (value >= threshold)
#   2  usage error (unknown or missing argument)
set -euo pipefail

WINDOW_DAYS=""
THRESHOLD=""
VALUE=""
SLEEP_SECS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --window-days)
            WINDOW_DAYS="${2:-}"
            shift 2
            ;;
        --threshold)
            THRESHOLD="${2:-}"
            shift 2
            ;;
        --value)
            VALUE="${2:-}"
            shift 2
            ;;
        --sleep-secs)
            SLEEP_SECS="${2:-}"
            shift 2
            ;;
        *)
            echo "check_merge_flakiness.sh: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$THRESHOLD" ] || [ -z "$VALUE" ]; then
    echo "check_merge_flakiness.sh: --threshold and --value are required" >&2
    exit 2
fi

if [ "$SLEEP_SECS" != "0" ]; then
    sleep "$SLEEP_SECS"
fi

if awk "BEGIN { exit !($VALUE < $THRESHOLD) }"; then
    echo "check_merge_flakiness: value=${VALUE} threshold=${THRESHOLD} window_days=${WINDOW_DAYS} -- invariant holds"
    exit 0
else
    echo "check_merge_flakiness: value=${VALUE} threshold=${THRESHOLD} window_days=${WINDOW_DAYS} -- invariant VIOLATED"
    exit 1
fi
