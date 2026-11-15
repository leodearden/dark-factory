#!/usr/bin/env bash
# routing_experiment_arm.sh -- arm a routing split experiment on the running fleet.
# PRD: plans/routing-experiment-substrate-prd.md (leaf delta). The before_done of the
# deterministic arming task: exit 0 = the split is balanced on an unchanged window
# and every reachable unit reloaded; non-zero = the caller escalates born-at-L2.
#
# 1. Confirms main carries the experiment's rule ids (gamma landed), so a reload
#    cannot silently arm an older table.
# 2. Runs scripts/routing_experiment_readout.py --aa (leaf beta) over the trailing
#    window with the experiment salt: the A/A check that seat N showed is required
#    before any split is read (task-id parity skew, drift). Non-zero there stops here.
# 3. Calls reload_config on every unit listed in UNITS through the shared transport
#    (scripts/_config_reload_gate.py). A unit whose MCP is not listening is SKIPPED
#    with a warning, never a failure: defaults.yaml is re-read at its next start, so
#    a halted or idle unit arms itself when it comes back. A unit that answers with
#    a reload error FAILS the arming.
# 4. Appends one JSON line to data/orchestrator/routing_experiments.jsonl recording
#    the salt, rule ids, flip time and per-unit dispositions: the readout gate's
#    anchor for "when did arm 1 start".
#
# Usage: routing_experiment_arm.sh <salt> <rule_id_csv> [days]
#   e.g. routing_experiment_arm.sh routing-exp-1 large-plan-modules-4,large-plan-modules 30
set -euo pipefail

die() { echo "routing_experiment_arm: $*" >&2; exit 1; }
[ "$#" -ge 2 ] || die "usage: routing_experiment_arm.sh <salt> <rule_id_csv> [days]"
SALT="$1"; RULES="$2"; DAYS="${3:-30}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python3"; [ -x "$PY" ] || PY=python3

# Fleet units: <project_root>:<escalation_port>. Idle or halted units are skipped
# when their MCP does not answer (see header). Override with ROUTING_EXPERIMENT_UNITS.
UNITS="${ROUTING_EXPERIMENT_UNITS:-\
/home/leo/src/dark-factory:8102 \
/home/leo/src/reify:8100 \
/home/leo/src/solar-challenge-platform:8107 \
/home/leo/src/solar-challenge:8106 \
/home/leo/src/stream-touch:8110 \
/home/leo/src/scoping-intake:8109 \
/home/leo/src/pump-web-ui:8108 \
/home/leo/src/know-live:8105 \
/home/leo/src/autopilot-video:8101}"

# 1. The rule ids must be on main's defaults.yaml (the editable install every unit reads).
IFS=',' read -r -a RULE_IDS <<< "$RULES"
for rid in "${RULE_IDS[@]}"; do
  git -C "$ROOT" show main:orchestrator/src/orchestrator/defaults.yaml | grep -q "id: ${rid}\$" \
    || die "rule id '${rid}' is not in main's defaults.yaml; land gamma before arming"
done

# 2. A/A check on an unchanged window.
[ -f "$ROOT/scripts/routing_experiment_readout.py" ] \
  || die "scripts/routing_experiment_readout.py is missing; land beta before arming"
"$PY" "$ROOT/scripts/routing_experiment_readout.py" --aa --salt "$SALT" --buckets 2 --days "$DAYS" \
  || die "A/A check failed for salt '$SALT': the split is skewed on an unchanged window; do not arm"

# 3. Reload every reachable unit; 4. record the flip.
"$PY" - "$HERE" "$SALT" "$RULES" "$ROOT/data/orchestrator/routing_experiments.jsonl" $UNITS <<'PYEOF'
import json, sys, datetime, pathlib
here, salt, rules, ledger, *units = sys.argv[1:]
sys.path.insert(0, here)
import _config_reload_gate as gate

record = {
    "salt": salt, "rule_ids": rules.split(","),
    "armed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "units": [],
}
failed = False
for unit in units:
    root, port = unit.rsplit(":", 1)
    try:
        report = gate.fetch_reload_report(port, committed_as=f"routing split {salt}", config_path=None)
        applied = report.get("applied")
        record["units"].append({"root": root, "port": port, "status": "reloaded", "applied": applied})
        print(f"routing_experiment_arm: {root}:{port} reloaded; applied={applied}")
    except gate.ReloadNotConfirmed as exc:
        kind = exc.failure.value
        if kind in (gate.ReloadFailure.NO_RELOAD_REPORT.value, gate.ReloadFailure.REPLY_TIMED_OUT.value):
            record["units"].append({"root": root, "port": port, "status": "skipped", "reason": kind})
            print(f"routing_experiment_arm: WARNING {root}:{port} not reachable ({kind}); it arms at its next start", file=sys.stderr)
        else:
            failed = True
            record["units"].append({"root": root, "port": port, "status": "failed", "reason": kind, "detail": exc.detail})
            print(f"routing_experiment_arm: FAILED {root}:{port}: {exc.detail}", file=sys.stderr)

path = pathlib.Path(ledger); path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a") as fh:
    fh.write(json.dumps(record) + "\n")
print(f"routing_experiment_arm: recorded flip in {path}")
sys.exit(1 if failed else 0)
PYEOF
