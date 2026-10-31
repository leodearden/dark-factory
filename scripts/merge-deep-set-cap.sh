#!/usr/bin/env bash
# merge-deep-set-cap.sh -- thin operator deploy for the deep merge-ahead canary.
# PRD: plans/deep-merge-ahead-prd.md (tasks zeta cap=6, eta2 cap=32; decisions
# #6 cap staging, #7 green-tier hot-reload).
#
# Sets merge_deep.chain_cap to <cap> in a target project's
# dark-factory-orchestrator.yaml by LANDING that one-file edit through the
# target's own merge queue (scripts/_merge_queue_landing.py), never by a direct
# commit to its main: such a commit is drift the queue never verified, which
# forces every merge in flight there to re-verify in full
# (orchestrator/src/orchestrator/merge_lane/gates.py::_disjoint_skip_blockers).
# The edit is computed against main's committed copy of the file, so the target
# checkout is never written by this script; only the queue's own landing syncs it.
#
# Only once the queue has landed the edit does the script hot-reload that
# project's running orchestrator via its escalation MCP (reload_config, which
# always re-reads the orchestrator's OWN config path -- it cannot retarget),
# and assert the reload's `applied` disposition carries the merge_deep.chain_cap
# knob (proving it hot-applied on the green tier, not silently deferred to a
# restart).  The escalation MCP is stateful, so every call goes through the
# shared transport (scripts/_config_reload_gate.py), never a single-shot POST.
# Forward-transition deploy (0->6->32); it is NOT idempotent at a fixed value
# -- re-running at the cap main already carries has nothing to land and dies
# before any MCP call.
#
# The landing waits for the target's merge verify, which runs the full gate:
# minutes to hours.  MERGE_QUEUE_LANDING_TIMEOUT_SECS bounds the wait (default
# _merge_queue_landing.py::DEFAULT_LANDING_TIMEOUT_SECS); a calling task's
# before_done.timeout_secs must be at least that ceiling plus 300 s, because
# calls in flight at the ceiling, the cancel and the reload run past it
# (OPERATIONS.md, "Deploying a knob flip").  A request still in flight at that
# ceiling, or at SIGTERM/SIGINT, is cancelled before the script exits.
#
# Usage: merge-deep-set-cap.sh <cap> <config_yaml_path> <escalation_port>
#   <cap>              non-negative integer, no leading zero (0 = kill switch)
#   <config_yaml_path> absolute path to the target dark-factory-orchestrator.yaml
#   <escalation_port>  the target orchestrator's escalation MCP port
#
# Exit 0 = knob landed on the target's main, reloaded, and observed in
# `applied`.  Non-zero = a failure the caller (a deterministic deploy task)
# escalates born-at-L2; one that did not land names the request's last merge
# state.
set -euo pipefail

KNOB_KEY="merge_deep.chain_cap"

die() { echo "merge-deep-set-cap: $*" >&2; exit 1; }

[ "$#" -eq 3 ] || die "usage: merge-deep-set-cap.sh <cap> <config_yaml_path> <escalation_port>"
CAP="$1"; CONFIG="$2"; PORT="$3"

# Canonical spelling only: the yaml edit, commit message and applied check must agree.
[[ "$CAP" =~ ^(0|[1-9][0-9]*)$ ]] \
    || die "cap must be a non-negative integer with no leading zero (YAML reads 010 as octal 8), got: $CAP"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "port must be an integer, got: $PORT"
[ -f "$CONFIG" ]          || die "config not found: $CONFIG"

# Step 3's landing and reload run under the SCRIPT's checkout venv and import
# from the SCRIPT's checkout -- never from $REPO, the config's checkout (the
# contract is scripts/_config_reload_gate.py's).  Step 1 is stdlib-only plain
# python3.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/../.venv/bin/python3"
[ -x "$PY" ] || PY=python3

# REPO, REL, BASE, and main's copy of the config (MAIN_COPY) beside the copy
# step 1 edits (EDITED), with the traps that let step 3 cancel on a killpg.
# shellcheck source=scripts/lib/deploy_main_copy.sh
source "$SCRIPT_DIR/lib/deploy_main_copy.sh"
main_copy_failed() { die "$2"; }
deploy_main_copy "$CONFIG" main_copy_failed

# 1. Upsert `merge_deep:\n  chain_cap: <cap>` (stdlib python3; line-oriented so
#    the raw-shell deploy layer needs no yaml library).
python3 - "$EDITED" "$CAP" <<'PY'
import re, sys
path, cap = sys.argv[1], sys.argv[2]
lines = open(path, encoding="utf-8").read().splitlines()
out, i, done = [], 0, False
while i < len(lines):
    line = lines[i]
    if re.match(r'^merge_deep:\s*(#.*)?$', line):
        out.append(line); i += 1
        body = []
        while i < len(lines) and (lines[i][:1] in (' ', '\t') or lines[i].strip() == ''):
            body.append(lines[i]); i += 1
        while body and body[-1].strip() == '':   # don't strand chain_cap after blanks
            body.pop()
        replaced = False
        for b in body:
            if re.match(r'^\s+chain_cap:\s', b):
                out.append(f'  chain_cap: {cap}'); replaced = True
            else:
                out.append(b)
        if not replaced:
            out.append(f'  chain_cap: {cap}')
        done = True
        continue
    out.append(line); i += 1
if not done:
    if out and out[-1].strip() != '':
        out.append('')
    out += ['merge_deep:', f'  chain_cap: {cap}']
open(path, 'w', encoding="utf-8").write('\n'.join(out) + '\n')
PY

# 2. Nothing to land is the documented non-idempotence, refused before any MCP call.
if cmp -s "$MAIN_COPY" "$EDITED"; then
    die "nothing to land: main already carries ${KNOB_KEY}=${CAP}"
fi

# 3. Land the edit through the target's merge queue, then hot-reload the running
#    orchestrator and assert the knob is live.  The landing is
#    scripts/_merge_queue_landing.py's; the transport, its failure diagnostics
#    and the argv-only heredoc contract are scripts/_config_reload_gate.py's.
#    This checks only the knob's disposition.
"$PY" - "$SCRIPT_DIR" "$KNOB_KEY" "$CAP" "$PORT" "$REPO" "$REL" "$BASE" "$EDITED" <<'LAND_PY'
import os
import sys
script_dir, knob, cap, port, repo, rel, base, edited = sys.argv[1:9]
sys.path.insert(0, script_dir)
from _config_reload_gate import die, reload_or_exit
from _merge_queue_landing import land_or_exit

PROG = "merge-deep-set-cap"
with open(edited, "rb") as edit:
    content = edit.read()
landing = land_or_exit(
    repo, rel, base, content, prog=PROG, port=port,
    label=f"merge-deep-chain-cap-{cap}",
    message=f"chore(merge-deep): set {knob}={cap} (deep merge-ahead canary)",
    timeout_secs=os.environ.get("MERGE_QUEUE_LANDING_TIMEOUT_SECS") or None,
)
print(f"{PROG}: landed {landing.branch} request_id={landing.request_id} as {landing.landed_as}",
      flush=True)

report = reload_or_exit(port, prog=PROG, committed_as=landing.landed_as)
applied = report.get("applied") or {}
entry = applied.get(knob)
if not isinstance(entry, dict):
    die(PROG, f"applied disposition missing {knob}: applied_keys={sorted(applied)} "
              f"restart_required_keys={sorted(report.get('restart_required') or {})}")
if str(entry.get("new")) != cap:
    die(PROG, f"applied {knob} does not carry {cap}: entry={entry}")
print(f"{PROG}: {knob} applied old={entry.get('old')} new={entry.get('new')}")
LAND_PY

echo "merge-deep-set-cap: done cap=${CAP} config=${CONFIG} port=${PORT}"
