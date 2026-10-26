#!/usr/bin/env bash
# merge-deep-canary-predicate.sh -- read-only 7-day deep merge-ahead canary
# health predicate (task eta1's before_done, kind=predicate).
# PRD: plans/deep-merge-ahead-prd.md (task eta1; predicate exit-code contract,
# CLAUDE.md "Milestone tasks" / task-authoring.md section 6).
#
# PROVISIONAL THRESHOLDS -- task theta (week-after aggressiveness assessment)
# retunes these against real canary telemetry.  Defaults, all overridable:
#   deep-fail-rate    <= 0.35   (replay study saw 2/59 ~= 0.034 -- loose gate)
#   items-per-verify  >= 1.3    (a chain must land materially >1 item/verify)
#   verify-p90-secs   <= 5400   (75% of the 7200s verify timeout -- margin gate)
#
# Reads a target orchestrator runs.db (the append-only sqlite event store; no
# migration) over the window (now - N days, now], its bounds rendered in
# EventStore.emit's own UTC ISO-8601 layout so the lexical comparison against
# events.timestamp is exact.  Keyed on the truthful chain telemetry
# fields (merge_verify.chain_items >= 2 = a deep verify; merge_finalized.
# landed_via_chain = items a chain landed).  Read-only: opened mode=ro, no writes.
# verify_p90 reads merge_verify data['duration_ms']: the events.duration_ms
# column is NULL for that event type.
#
# The DeterministicRunner decides by EXIT CODE ONLY -- it parses no stdout:
#   0  canary healthy            -> task done (promote 6->32 proceeds)
#   1  unhealthy OR no signal    -> born-at-L2 milestone_check_failed (hold;
#                                    a canary that never exercised chains must
#                                    not silently promote)
#   2  usage/infra fault         -> never masquerades as a health verdict
#
# Usage:
#   merge-deep-canary-predicate.sh --runs-db PATH [--window-days N]
#       [--deep-fail-rate-max R] [--items-per-verify-min R] [--verify-p90-max-secs S]
#       [--now EPOCH_SECS]   (the reference instant; defaults to the current time)
set -euo pipefail

usage() { echo "merge-deep-canary-predicate: $*" >&2; exit 2; }

RUNS_DB=""
WINDOW_DAYS="7"
FAIL_MAX="0.35"
ITEMS_MIN="1.3"
P90_MAX="5400"
NOW=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --runs-db)              [ "$#" -ge 2 ] || usage "missing value for $1"; RUNS_DB="$2"; shift 2 ;;
        --window-days)          [ "$#" -ge 2 ] || usage "missing value for $1"; WINDOW_DAYS="$2"; shift 2 ;;
        --deep-fail-rate-max)   [ "$#" -ge 2 ] || usage "missing value for $1"; FAIL_MAX="$2"; shift 2 ;;
        --items-per-verify-min) [ "$#" -ge 2 ] || usage "missing value for $1"; ITEMS_MIN="$2"; shift 2 ;;
        --verify-p90-max-secs)  [ "$#" -ge 2 ] || usage "missing value for $1"; P90_MAX="$2"; shift 2 ;;
        --now)                  [ "$#" -ge 2 ] || usage "missing value for $1"; NOW="$2"; shift 2 ;;
        *) usage "unknown argument: $1" ;;
    esac
done

[ -n "$RUNS_DB" ] || usage "--runs-db is required"
[ -f "$RUNS_DB" ] || usage "runs.db not found: $RUNS_DB"

python3 - "$RUNS_DB" "$WINDOW_DAYS" "$FAIL_MAX" "$ITEMS_MIN" "$P90_MAX" "$NOW" <<'PY'
import json, math, sqlite3, sys, time
from datetime import datetime, timedelta, timezone
db, raw_window, raw_fail_max, raw_items_min, raw_p90_max, raw_now = sys.argv[1:7]

def usage_fault(message):
    print(f"predicate: {message}", file=sys.stderr); sys.exit(2)

def number(flag, raw):
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        usage_fault(f"{flag} must be a finite number; got {raw!r}")
    return value

window_days = number("--window-days", raw_window)
if window_days <= 0:
    usage_fault(f"--window-days must be > 0; got {raw_window!r}")
fail_max = number("--deep-fail-rate-max", raw_fail_max)
items_min = number("--items-per-verify-min", raw_items_min)
p90_max = number("--verify-p90-max-secs", raw_p90_max)
now_epoch = time.time() if raw_now == "" else number("--now", raw_now)

try:
    as_of = datetime.fromtimestamp(now_epoch, timezone.utc)
    window_lo = (as_of - timedelta(days=window_days)).isoformat(timespec="microseconds")
    window_hi = as_of.isoformat(timespec="microseconds")
except (OverflowError, OSError, ValueError) as exc:
    usage_fault(f"--now/--window-days put the window out of range: {exc}")

try:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
except sqlite3.Error as exc:
    print(f"predicate: cannot open runs.db read-only: {exc}", file=sys.stderr); sys.exit(2)

def load(evtype):
    try:
        rows = conn.execute(
            "SELECT data FROM events "
            "WHERE event_type=? AND timestamp > ? AND timestamp <= ? ORDER BY id",
            (evtype, window_lo, window_hi),
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"predicate: query failed ({evtype}): {exc}", file=sys.stderr); sys.exit(2)
    out = []
    for (data,) in rows:
        try:
            parsed = json.loads(data) if data else {}
        except json.JSONDecodeError:
            parsed = {}
        out.append(parsed)
    return out

mv = load("merge_verify")
mf = load("merge_finalized")

# A "deep" verify is one whose verified tree carried >= 2 chained items.
deep = [d for d in mv if isinstance(d.get("chain_items"), int) and d["chain_items"] >= 2]
n_deep = len(deep)
n_fail = sum(1 for d in deep if not d.get("passed"))
fail_rate = (n_fail / n_deep) if n_deep else None

landed = [d["landed_via_chain"] for d in mf
          if isinstance(d.get("landed_via_chain"), int) and d["landed_via_chain"] >= 1]
items_per = (sum(landed) / n_deep) if n_deep else None   # items landed per deep verify run

def payload_secs(d):
    ms = d.get("duration_ms")
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms):
        return None
    return ms / 1000.0

durs = sorted(s for s in map(payload_secs, deep) if s is not None)
def pct(xs, q):
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]
p90 = pct(durs, 0.90)

# None (insufficient signal) is treated as a FAIL: a canary that produced no
# chains this window must not green-light promotion to the higher cap.
checks = [
    ("deep_fail_rate",   fail_rate, fail_rate is not None and fail_rate <= fail_max, f"<= {fail_max}"),
    ("items_per_verify", items_per, items_per is not None and items_per >= items_min, f">= {items_min}"),
    ("verify_p90_secs",  p90,       p90 is not None and p90 <= p90_max,               f"<= {p90_max}"),
]
ok = all(passed for (_, _, passed, _) in checks)
summary = " ".join(
    f"{name}={('n/a' if val is None else round(val, 4))}({'ok' if passed else 'FAIL'} {bound})"
    for (name, val, passed, bound) in checks
)
print(f"merge-deep-canary window_days={window_days:g} n_deep_verifies={n_deep}: {summary} -> "
      f"{'PASS' if ok else 'HOLD'}")
sys.exit(0 if ok else 1)
PY
