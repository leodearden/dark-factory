#!/usr/bin/env bash
# routing_experiment_arm.sh -- arm a routing split experiment on the running fleet.
# PRD: plans/routing-experiment-substrate-prd.md (leaf delta). The before_done of the
# deterministic arming task: exit 0 = the split is balanced on an unchanged window
# and every reachable unit reloaded; non-zero = the caller escalates born-at-L2.
#
# 1. Confirms main carries the experiment's rule ids and its split salt (a
#    `salt: <salt>` line), so a reload cannot silently arm an older table. They
#    are read from main's copy of the file named by ROUTING_EXPERIMENT_RULES_FILE
#    (repo-relative, default orchestrator/src/orchestrator/defaults.yaml), read
#    once, and each is tested against it.
# 2. Anchors the treatment start, TREATMENT_FROM: the commit time, in UTC, of the
#    earliest first-parent main commit of the trailing run in which that file has
#    carried every id and the salt. Units read main's checkout, so no treated
#    decision can precede it, however early a restarted unit picked the rules up.
#    A commit that deletes the file, drops an id, or drops the split or moves it
#    to another salt ends the run; one whose copy git cannot read stops the
#    arming. The salt is matched file-wide, not per rule, so a commit that drops
#    the split from only some of the rules does not end the run.
# 3. Runs scripts/routing_experiment_readout.py --aa (leaf beta) over the [days]
#    window ending at TREATMENT_FROM, with the experiment salt, reading only the
#    experiment units' roots: the A/A check that seat N showed is required before
#    any split is read (task-id parity skew, drift). Non-zero there stops here.
# 4. Calls reload_config on every experiment unit through the shared transport
#    (scripts/_config_reload_gate.py). A unit whose MCP is not listening is SKIPPED
#    with a warning, never a failure: its config is re-read at its next start, so
#    a halted or idle unit arms itself when it comes back. A unit that answers with
#    a reload error FAILS the arming.
# 5. Appends one JSON line to data/orchestrator/routing_experiments.jsonl recording
#    the salt, rule ids, treatment_from and the commit it names (rules_landed_in),
#    armed_at and per-unit dispositions. treatment_from is the readout gate's anchor
#    for "when did arm 1 start"; armed_at records when every reachable unit
#    reloaded. plans/routing-experiment-substrate-prd.md, section "Experiment 1:
#    rules (γ, δ)", "Sequencing hazard".
#
# Usage: [ROUTING_EXPERIMENT_UNITS='<project_root>:<port> ...'] \
#        [ROUTING_EXPERIMENT_RULES_FILE=<repo-relative path>] \
#          routing_experiment_arm.sh <salt> <rule_id_csv> [days]
#   e.g. routing_experiment_arm.sh routing-exp-1 large-plan-modules-4,large-plan-modules 30
set -euo pipefail

die() { echo "routing_experiment_arm: $*" >&2; exit 1; }
first_missing_experiment_line() {
  local rid
  for rid in "${RULE_IDS[@]}"; do
    grep -q -- "id: ${rid}\$" <<< "$1" || { printf "rule id '%s'" "$rid"; return 0; }
  done
  grep -q -- "salt: ${SALT}\$" <<< "$1" || printf "split salt '%s'" "$SALT"
}
[ "$#" -ge 2 ] || die "usage: routing_experiment_arm.sh <salt> <rule_id_csv> [days]"
SALT="$1"; RULES="$2"; DAYS="${3:-30}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python3"; [ -x "$PY" ] || PY=python3

# The experiment's units, <project_root>:<escalation_port> with an absolute
# project_root, whitespace-separated.
# Step 3's A/A reads exactly their roots and step 4 reloads exactly them. The
# default is the whole fleet; ROUTING_EXPERIMENT_UNITS names a subset.
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

IFS=$' \t\n' read -r -d '' -a UNIT_LIST <<< "$UNITS" || true
[ "${#UNIT_LIST[@]}" -gt 0 ] \
  || die "ROUTING_EXPERIMENT_UNITS names no unit; an experiment needs at least one <project_root>:<port>"
ROOTS=(); ROOT_PORT_PAIRS=()
for unit in "${UNIT_LIST[@]}"; do
  [[ "$unit" =~ ^(/[^,]+):([0-9]+)$ ]] \
    || die "unit '${unit}' is not <project_root>:<port> with an absolute, comma-free project_root"
  ROOTS+=("${BASH_REMATCH[1]}")
  ROOT_PORT_PAIRS+=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
done
ROOTS_CSV="$(IFS=,; printf '%s' "${ROOTS[*]}")"

# 1. The rule ids and the salt must be on main's copy of the rules file (main is
#    what every unit's editable install reads).
RULES_FILE="${ROUTING_EXPERIMENT_RULES_FILE:-orchestrator/src/orchestrator/defaults.yaml}"
RULES_TEXT="$(git -C "$ROOT" show "main:${RULES_FILE}")" \
  || die "cannot read ${RULES_FILE} from main (git's message is above); ROUTING_EXPERIMENT_RULES_FILE names it, repo-relative"
IFS=',' read -r -a RULE_IDS <<< "$RULES"
missing="$(first_missing_experiment_line "$RULES_TEXT")"
[ -z "$missing" ] \
  || die "${missing} is not in main's ${RULES_FILE}; land the experiment's rules before arming"

# 2. The treatment start: walk main's first-parent history newest first while
#    every commit's copy of the rules file carries every id and the salt.
HISTORY="$(TZ=UTC0 git -C "$ROOT" log main --first-parent --date=iso-strict-local --format='%H %cd' -- "$RULES_FILE")" \
  || die "cannot read main's first-parent history of ${RULES_FILE}"
TREATMENT_FROM=""; RULES_LANDED_IN=""
while read -r sha landed_at; do
  listed="$(git -C "$ROOT" ls-tree --full-tree --name-only "$sha" -- "$RULES_FILE")" \
    || die "cannot list main's tree at ${sha} (git's message is above)"
  [ -n "$listed" ] || break
  text="$(git -C "$ROOT" show "${sha}:${RULES_FILE}")" \
    || die "cannot read ${RULES_FILE} at main's ${sha} (git's message is above); its landing cannot be anchored"
  [ -z "$(first_missing_experiment_line "$text")" ] || break
  TREATMENT_FROM="$landed_at"; RULES_LANDED_IN="$sha"
done <<< "$HISTORY"
[ -n "$TREATMENT_FROM" ] \
  || die "no first-parent commit on main carrying every rule id and the salt was found in the history of ${RULES_FILE}"
echo "routing_experiment_arm: treatment from ${TREATMENT_FROM}: main ${RULES_LANDED_IN:0:10} first carried every rule id and the salt in ${RULES_FILE}; the A/A window ends there"

# 3. A/A check on an unchanged window, which ends where treatment can begin.
[ -f "$ROOT/scripts/routing_experiment_readout.py" ] \
  || die "scripts/routing_experiment_readout.py is missing; land beta before arming"
"$PY" "$ROOT/scripts/routing_experiment_readout.py" --aa --salt "$SALT" --buckets 2 --days "$DAYS" \
    --until "$TREATMENT_FROM" --roots "$ROOTS_CSV" \
  || { rc=$?; die "the readout's A/A check did not pass for salt '$SALT' (exit $rc: 1 = skewed, 3 = no ledger read, 4 = unproven; its own message above says which); do not arm"; }

# 4. Reload every reachable unit; 5. record the flip.
"$PY" - "$HERE" "$SALT" "$RULES" "$TREATMENT_FROM" "$RULES_LANDED_IN" \
    "$ROOT/data/orchestrator/routing_experiments.jsonl" "${ROOT_PORT_PAIRS[@]}" <<'PYEOF'
import json, sys, datetime, pathlib
here, salt, rules, treatment_from, rules_landed_in, ledger, *root_port_pairs = sys.argv[1:]
sys.path.insert(0, here)
import _config_reload_gate as gate

record = {
    "salt": salt, "rule_ids": rules.split(","),
    "treatment_from": treatment_from, "rules_landed_in": rules_landed_in,
    "armed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "units": [],
}
failed = False
for root, port in zip(root_port_pairs[0::2], root_port_pairs[1::2]):
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
