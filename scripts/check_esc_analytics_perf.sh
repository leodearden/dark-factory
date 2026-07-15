#!/usr/bin/env bash
# Read-only perf predicate for the escalation-analytics dashboard endpoint.
#
# Contract (plans/escalation-lifecycle-dashboard-prd.md, Seam 3): run as a
# deterministic-task predicate (before_done.kind='predicate') — decision is by
# exit code only; the stdout tail is carried as the note. Prints one
# structured measurement line. Exit 0 iff the median response time over
# --attempts requests is within --threshold-ms; connection failure exits 1.
#
# This script owns only the measurement + threshold comparison. The default
# --url pins the endpoint path delivered by the PRD's γ task; a route change
# must update this default in the same change.
set -u

URL="http://127.0.0.1:8080/api/v2/dashboard/escalation-analytics"
# attempts=5: observed 2026-07-15 that under fleet load one request can block
# ~10s on a busy worker while warm requests are sub-ms — median-of-5 needs 3
# slow attempts to trip, so scheduling noise alone can't file a false L2.
THRESHOLD_MS=2000
ATTEMPTS=5

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --threshold-ms) THRESHOLD_MS="$2"; shift 2 ;;
    --attempts) ATTEMPTS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

times_ms=()
for i in $(seq 1 "$ATTEMPTS"); do
  t=$(curl -sf -o /dev/null -w '%{time_total}' --max-time 30 "$URL") || {
    echo "measured=connection_failed attempt=${i} url=${URL} threshold_ms=${THRESHOLD_MS}"
    exit 1
  }
  times_ms+=("$(awk -v t="$t" 'BEGIN{printf "%d", t*1000}')")
done

median=$(printf '%s\n' "${times_ms[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
echo "measured_median_ms=${median} attempts_ms=${times_ms[*]} threshold_ms=${THRESHOLD_MS} url=${URL}"
[ "$median" -le "$THRESHOLD_MS" ] && exit 0
exit 1
