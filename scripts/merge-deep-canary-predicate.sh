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
# migration) over a trailing window, keyed on the truthful chain telemetry
# fields (merge_verify.chain_items >= 2 = a deep verify; merge_finalized.
# landed_via_chain = items a chain landed).  Read-only: opened mode=ro, no writes.
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
set -euo pipefail

usage() { echo "merge-deep-canary-predicate: $*" >&2; exit 2; }

RUNS_DB=""
WINDOW_DAYS="7"
FAIL_MAX="0.35"
ITEMS_MIN="1.3"
P90_MAX="5400"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --runs-db)              [ "$#" -ge 2 ] || usage "missing value for $1"; RUNS_DB="$2"; shift 2 ;;
        --window-days)          [ "$#" -ge 2 ] || usage "missing value for $1"; WINDOW_DAYS="$2"; shift 2 ;;
        --deep-fail-rate-max)   [ "$#" -ge 2 ] || usage "missing value for $1"; FAIL_MAX="$2"; shift 2 ;;
        --items-per-verify-min) [ "$#" -ge 2 ] || usage "missing value for $1"; ITEMS_MIN="$2"; shift 2 ;;
        --verify-p90-max-secs)  [ "$#" -ge 2 ] || usage "missing value for $1"; P90_MAX="$2"; shift 2 ;;
        *) usage "unknown argument: $1" ;;
    esac
done

[ -n "$RUNS_DB" ] || usage "--runs-db is required"
[ -f "$RUNS_DB" ] || usage "runs.db not found: $RUNS_DB"

python3 - "$RUNS_DB" "$WINDOW_DAYS" "$FAIL_MAX" "$ITEMS_MIN" "$P90_MAX" <<'PY'
import json, sqlite3, sys
db, window = sys.argv[1], sys.argv[2]
fail_max, items_min, p90_max = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

try:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
except sqlite3.Error as exc:
    print(f"predicate: cannot open runs.db read-only: {exc}", file=sys.stderr); sys.exit(2)

def load(evtype):
    try:
        rows = conn.execute(
            "SELECT data, duration_ms FROM events "
            "WHERE event_type=? AND timestamp > datetime('now', ?) ORDER BY id",
            (evtype, f"-{window} days"),
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"predicate: query failed ({evtype}): {exc}", file=sys.stderr); sys.exit(2)
    out = []
    for data, dur in rows:
        try:
            parsed = json.loads(data) if data else {}
        except json.JSONDecodeError:
            parsed = {}
        out.append((parsed, dur))
    return out

mv = load("merge_verify")
mf = load("merge_finalized")

# A "deep" verify is one whose verified tree carried >= 2 chained items.
deep = [(d, dur) for (d, dur) in mv if isinstance(d.get("chain_items"), int) and d["chain_items"] >= 2]
n_deep = len(deep)
n_fail = sum(1 for (d, _) in deep if not d.get("passed"))
fail_rate = (n_fail / n_deep) if n_deep else None

landed = [d["landed_via_chain"] for (d, _) in mf
          if isinstance(d.get("landed_via_chain"), int) and d["landed_via_chain"] >= 1]
items_per = (sum(landed) / n_deep) if n_deep else None   # items landed per deep verify run

durs = sorted(dur / 1000.0 for (_, dur) in deep if isinstance(dur, (int, float)))
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
print(f"merge-deep-canary window_days={window} n_deep_verifies={n_deep}: {summary} -> "
      f"{'PASS' if ok else 'HOLD'}")
sys.exit(0 if ok else 1)
PY
