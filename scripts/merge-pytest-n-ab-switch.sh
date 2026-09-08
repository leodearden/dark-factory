#!/usr/bin/env bash
# scripts/merge-pytest-n-ab-switch.sh — flip the MERGE test leg's pytest-xdist
# `-n auto` pin (verify_env.PYTEST_XDIST_AUTO_NUM_WORKERS) to a new value,
# commit ONLY the config, and hot-reload the running orchestrator.
#
# Context: the 2026-09-08 load-tuning session pinned the merge leg at 16 and
# Leo asked for an in-place A/B against 8 (the value task 3589 proposes for
# addopts) rather than deciding from a 249-test ladder measured at saturation.
# This is the arm switch; scripts/merge-pytest-n-ab-analysis.py is the cut.
#
# Usage: merge-pytest-n-ab-switch.sh <value> [config_yaml] [escalation_port] [--dry-run]
#   <value>           the new PYTEST_XDIST_AUTO_NUM_WORKERS ("8", "16", ...)
#   [config_yaml]     default /home/leo/src/dark-factory/dark-factory-orchestrator.yaml
#   [escalation_port] default 8102 (dark-factory's escalation MCP)
#   --dry-run         edit a temp copy, print the diff, no commit, no reload
#
# Modelled on scripts/merge-deep-set-cap.sh (same commit --only + single-shot
# MCP tools/call reload). Exit 0 only when the reload's `applied` disposition
# carries verify_env with the new value; the last stdout line is a JSON verdict
# for a kind='predicate' before_done note.
set -euo pipefail
die() { echo "merge-pytest-n-ab-switch: $*" >&2; exit 1; }

DRY=0; ARGS=()
for a in "$@"; do case "$a" in --dry-run) DRY=1;; *) ARGS+=("$a");; esac; done
[ "${#ARGS[@]}" -ge 1 ] || die "usage: merge-pytest-n-ab-switch.sh <value> [config_yaml] [escalation_port] [--dry-run]"
VALUE="${ARGS[0]}"
CONFIG="${ARGS[1]:-/home/leo/src/dark-factory/dark-factory-orchestrator.yaml}"
PORT="${ARGS[2]:-8102}"
[[ "$VALUE" =~ ^[0-9]+$ ]] || die "value must be a positive integer, got '$VALUE'"
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
REPO="$(cd "$(dirname "$CONFIG")" && git rev-parse --show-toplevel)" || die "config is not inside a git checkout"
CONFIG_BASE="$(realpath --relative-to="$REPO" "$CONFIG")"
KEY='PYTEST_XDIST_AUTO_NUM_WORKERS'
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

TARGET="$CONFIG"
if [ "$DRY" -eq 1 ]; then TARGET="$(mktemp)"; cp "$CONFIG" "$TARGET"; fi

# 1. Rewrite the value inside the top-level `verify_env:` mapping. Line-based
#    on purpose (no yaml round-trip: the file is 1000 lines of load-bearing
#    comments). Replaces an existing A/B marker line rather than stacking them.
python3 - "$TARGET" "$KEY" "$VALUE" "$STAMP" <<'PY'
import re, sys
path, key, value, stamp = sys.argv[1:5]
lines = open(path, encoding='utf-8').read().split('\n')
out, i, done, in_block, prev_marker = [], 0, False, False, None
marker = f'  # A/B arm: {key} set to "{value}" at {stamp} by scripts/merge-pytest-n-ab-switch.sh'
while i < len(lines):
    line = lines[i]
    if re.match(r'^verify_env:\s*$', line):
        in_block = True; out.append(line); i += 1; continue
    if in_block:
        if line.startswith('  # A/B arm: '):
            prev_marker = line; i += 1; continue  # drop a previous marker (restored if the value is unchanged)
        m = re.match(rf'^(\s+){re.escape(key)}:\s*(.*)$', line)
        if m:
            if m.group(2).strip().strip('"\'') == value:
                # already at the value: keep the file byte-identical (idempotent re-run)
                if out and out[-1].startswith('  # A/B arm: '): out.pop()
                if prev_marker is not None: out.append(prev_marker)
                out.append(line); done = True; i += 1; continue
            out.append(marker); out.append(f'{m.group(1)}{key}: "{value}"'); done = True; i += 1; continue
        if line.strip() == '' or not line.startswith(' '):
            if not done:
                out.append(marker); out.append(f'  {key}: "{value}"'); done = True
            in_block = False
    out.append(line); i += 1
if not done:
    sys.exit(f'no top-level verify_env: block found in {path}; refusing to invent one')
open(path, 'w', encoding='utf-8').write('\n'.join(out))
PY

if [ "$DRY" -eq 1 ]; then
    diff -u "$CONFIG" "$TARGET" || true
    rm -f "$TARGET"
    echo "{\"dry_run\": true, \"would_set\": \"$VALUE\"}"
    exit 0
fi

# 2. Commit only that file (machine-operated checkout: never sweep up unrelated
#    state). Idempotent: if the file already carried the value (a re-run after a
#    runner crash-resume), there is nothing to commit and that is success — the
#    reload below still runs so the in-memory config is known to match the file.
if git -C "$REPO" diff --quiet -- "$CONFIG_BASE"; then
    SHA="already-at-${VALUE}"
else
    git -C "$REPO" commit --only "$CONFIG_BASE" -q \
        -m "config(ab): merge test leg ${KEY}=${VALUE} (pytest -n A/B arm switch, ${STAMP})" \
        || die "git commit --only ${CONFIG_BASE} failed"
    SHA="$(git -C "$REPO" rev-parse --short HEAD)"
fi

# 3. Hot-reload via the escalation MCP (single-shot tools/call; Accept must carry both media types).
RESP="$(curl -sS -X POST "http://127.0.0.1:${PORT}/mcp" \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"reload_config","arguments":{}}}')" \
    || die "reload_config request to 127.0.0.1:${PORT} failed (committed as ${SHA}; the value lands at the next restart)"

# 4. Assert the applied disposition carries verify_env with the new value.
printf '%s' "$RESP" | python3 - "$KEY" "$VALUE" "$SHA" <<'PY'
import json, sys
key, value, sha = sys.argv[1:4]
env = json.load(sys.stdin)
res = env.get('result', {})
tool = res.get('structuredContent')
if not isinstance(tool, dict):
    content = res.get('content') or []
    tool = json.loads(content[0]['text']) if content and content[0].get('text') else {}
if tool.get('error'):
    print(f'reload_config error: {tool["error"]}', file=sys.stderr); sys.exit(1)
entry = (tool.get('applied') or {}).get('verify_env')
new = (entry or {}).get('new') or {}
if str(new.get(key)) != value:
    print(f'applied.verify_env does not carry {key}={value}: applied_keys={sorted(tool.get("applied") or {})} '
          f'restart_required_keys={sorted(tool.get("restart_required") or {})} entry={entry}', file=sys.stderr)
    sys.exit(1)
print(json.dumps({'switched_to': value, 'commit': sha, 'reload_applied': True}))
PY
