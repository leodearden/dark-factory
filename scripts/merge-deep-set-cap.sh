#!/usr/bin/env bash
# merge-deep-set-cap.sh -- thin operator deploy for the deep merge-ahead canary.
# PRD: plans/deep-merge-ahead-prd.md (tasks zeta cap=6, eta2 cap=32; decisions
# #6 cap staging, #7 green-tier hot-reload).
#
# Sets merge_deep.chain_cap to <cap> in a target project's
# dark-factory-orchestrator.yaml, commits ONLY that file, hot-reloads that
# project's running orchestrator via its escalation MCP (reload_config, which
# always re-reads the orchestrator's OWN config path -- it cannot retarget),
# and asserts the reload's `applied` disposition carries the merge_deep.chain_cap
# knob (proving it hot-applied on the green tier, not silently deferred to a
# restart).  Forward-transition deploy (0->6->32); it is NOT idempotent at a
# fixed value -- re-running at the current value has nothing to commit and the
# knob would not appear in `applied`.
#
# Usage: merge-deep-set-cap.sh <cap> <config_yaml_path> <escalation_port>
#   <cap>              non-negative integer (0 = kill switch)
#   <config_yaml_path> absolute path to the target dark-factory-orchestrator.yaml
#   <escalation_port>  the target orchestrator's escalation MCP port
#
# Exit 0 = knob committed, reloaded, and observed in `applied`.  Non-zero =
# a failure the caller (a deterministic deploy task) escalates born-at-L2.
set -euo pipefail

KNOB_KEY="merge_deep.chain_cap"

die() { echo "merge-deep-set-cap: $*" >&2; exit 1; }

[ "$#" -eq 3 ] || die "usage: merge-deep-set-cap.sh <cap> <config_yaml_path> <escalation_port>"
CAP="$1"; CONFIG="$2"; PORT="$3"

[[ "$CAP" =~ ^[0-9]+$ ]]  || die "cap must be a non-negative integer, got: $CAP"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "port must be an integer, got: $PORT"
[ -f "$CONFIG" ]          || die "config not found: $CONFIG"

REPO="$(cd "$(dirname "$CONFIG")" && pwd)"
CONFIG_BASE="$(basename "$CONFIG")"

# 1. Upsert `merge_deep:\n  chain_cap: <cap>` (stdlib python3; line-oriented so
#    the raw-shell deploy layer needs no yaml library).
python3 - "$CONFIG" "$CAP" <<'PY'
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

# 2. Commit only that file (machine-operated-checkout hygiene: never sweep up
#    unrelated dirty/staged state -- CLAUDE.md "Working in the main checkout").
git -C "$REPO" commit --only "$CONFIG_BASE" \
    -m "chore(merge-deep): set ${KNOB_KEY}=${CAP} (deep merge-ahead canary)" \
    || die "git commit --only ${CONFIG_BASE} failed (nothing to commit / already at ${CAP}?)"

# 3. Hot-reload the running orchestrator via its escalation MCP.  The
#    streamable-HTTP MCP transport accepts a single-shot tools/call POST when
#    Accept carries BOTH media types (a bare application/json 406s pre-dispatch).
RESP="$(curl -sS -X POST "http://127.0.0.1:${PORT}/mcp" \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"reload_config","arguments":{}}}')" \
    || die "reload_config request to 127.0.0.1:${PORT} failed"

# 4. Assert the applied disposition carries the knob (unwrap the JSON-RPC
#    envelope -> tool result -> {applied, restart_required, ...}).
printf '%s' "$RESP" | python3 - "$KNOB_KEY" <<'PY'
import json, sys
knob = sys.argv[1]
env = json.load(sys.stdin)
res = env.get("result", {})
tool = res.get("structuredContent")
if not isinstance(tool, dict):
    content = res.get("content") or []
    tool = json.loads(content[0]["text"]) if content and content[0].get("text") else {}
if tool.get("error"):
    print(f"reload_config error: {tool['error']}", file=sys.stderr); sys.exit(1)
applied = tool.get("applied") or {}
entry = applied.get(knob)
if not isinstance(entry, dict):
    print(f"applied disposition missing {knob}: applied_keys={sorted(applied)} "
          f"restart_required_keys={sorted(tool.get('restart_required') or {})}", file=sys.stderr)
    sys.exit(1)
print(f"merge-deep-set-cap: {knob} applied old={entry.get('old')} new={entry.get('new')}")
PY

echo "merge-deep-set-cap: done cap=${CAP} config=${CONFIG} port=${PORT}"
