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
# Shares scripts/merge-deep-set-cap.sh's `git commit --only` shape, and that
# alone: the escalation MCP is STATEFUL, so a single-shot `tools/call` POST is
# rejected at the TRANSPORT layer — `Bad Request: Missing session ID`, HTTP 400,
# measured live on 2026-09-12 — before any tool runs, and `curl` without -f
# exits 0 on it. The reload therefore goes through
# legibility.census_trigger.post_mcp_tool_call, which handshakes.
# (merge-deep-set-cap.sh still carries that single-shot defect; copying from it
# again would reintroduce this bug.)
#
# Exit 0 on either of the two ways the value can be live: the reload
# hot-applied it ('applied'), or the running config already carried it and the
# reload provably re-read THIS config file ('already_converged'). The last
# stdout line is a JSON verdict — {switched_to, commit, outcome} — for a
# kind='predicate' before_done note.
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
# The reload step imports its MCP transport from the checkout the SCRIPT lives
# in — never from $REPO below, which is the CONFIG's checkout and may be a
# different project entirely. That transport is not stdlib (httpx, pydantic),
# so the interpreter is resolved for the same reason: inheriting whatever
# `python3` a login shell offers makes this gate unreachable. This tree's one
# root .venv (CLAUDE.md, "Locating installed code"), else the caller's python3.
# Derived from the path rather than `git rev-parse` so a copy of this script
# outside any checkout still resolves under `set -e`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$SCRIPT_DIR/.."
PY="$CHECKOUT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3
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
#    reload below still runs, and step 4's converged branch reads its "nothing
#    changed here" answer as that same success rather than as a failure.
if git -C "$REPO" diff --quiet -- "$CONFIG_BASE"; then
    SHA="already-at-${VALUE}"
else
    git -C "$REPO" commit --only "$CONFIG_BASE" -q \
        -m "config(ab): merge test leg ${KEY}=${VALUE} (pytest -n A/B arm switch, ${STAMP})" \
        || die "git commit --only ${CONFIG_BASE} failed"
    SHA="$(git -C "$REPO" rev-parse --short HEAD)"
fi

# 3. Hot-reload via the escalation MCP, and assert the value is live.
#    `legibility.census_trigger.post_mcp_tool_call` is the single transport
#    definition every consumer of this server goes through (task 3644). It
#    supplies the four things one POST cannot: the session-less `initialize`
#    handshake the stateful server demands, SSE decoding of the reply, raising
#    on both an envelope-level JSON-RPC `error` and `result.isError`, and the
#    session-terminating DELETE without which every run leaks a live anyio task
#    in the long-lived escalation process. None of it is re-implemented here:
#    three legibility posters each hand-rolled this transport and all three
#    silently dropped every escalation they ever filed.
#
#    The value is live in either of two shapes. `applied` carrying verify_env
#    with the new value is the flip. ABSENCE of verify_env from `applied` is
#    the converged re-run — and absence is the ONLY converged signal on the
#    wire, because `unchanged` is a bare int COUNT of equal leaves
#    (config.py::ConfigDiff), naming no keys and carrying no values. Absence is
#    weaker than convergence, though: a rolled-back reload and a reload of a
#    DIFFERENT orchestrator both produce it, which is what the two corroborators
#    below exclude.
"$PY" - "$SCRIPT_DIR" "$KEY" "$VALUE" "$SHA" "$(realpath "$CONFIG")" "$PORT" <<'RELOAD_PY'
import json, os, sys
script_dir, key, value, sha, config_path, port = sys.argv[1:7]
sys.path.insert(0, script_dir)
try:
    from legibility import census_trigger
except ImportError as exc:
    # Not stdlib: httpx, and pydantic via census_trigger's legibility.config
    # import. Name the interpreter and the remedy rather than emitting a
    # traceback — an operator hitting this from a login shell has no other clue
    # that the interpreter, not the code, is what is wrong.
    print(f'the MCP reload transport is not importable under {sys.executable}: {exc}\n'
          f'  sys.path entry added: {script_dir}\n'
          f'  remedy: sync this checkout so {script_dir}/../.venv exists, or re-run\n'
          f'  this script under `uv run --project shared`',
          file=sys.stderr)
    sys.exit(1)

try:
    tool = census_trigger.post_mcp_tool_call(
        f'http://127.0.0.1:{port}/mcp', 'reload_config', {})
except Exception as exc:
    # Broad on purpose, and the breadth is the point: census_trigger raises
    # StatusFetchUnavailable for a malformed or error envelope, RuntimeError
    # for a failed handshake, and httpx's own exceptions for a dead socket —
    # all three mean the live config is UNKNOWN, and a deploy gate must not
    # pass on an unknown. Naming the TRANSPORT keeps this distinct from the
    # rolled-back-reload diagnostic below, which is about a reload that ran.
    print(f'reload_config never reached the tool: {type(exc).__name__}: {exc} '
          f'(committed as {sha}; the value lands at the next restart)', file=sys.stderr)
    sys.exit(1)

# reload_config's OWN error field: a config that failed to parse, reported by a
# perfectly successful tools/call. `_raise_on_mcp_error` inspects the JSON-RPC
# envelope and never sees this one.
if tool.get('error'):
    print(f'reload_config error: {tool["error"]}', file=sys.stderr); sys.exit(1)
entry = (tool.get('applied') or {}).get('verify_env')
if entry is None:
    reloaded = tool.get('reloaded')
    reported = tool.get('config_path')
    if not reloaded:
        print(f'reload reported no verify_env change, but did not commit it: reloaded={reloaded!r} '
              f'(a failed reload rolls every leaf back, so the live config is untouched)', file=sys.stderr)
        sys.exit(1)
    if not reported or os.path.realpath(reported) != config_path:
        print(f'reload reported no verify_env change, but re-read a different file: '
              f'config_path={reported!r} expected={config_path!r}', file=sys.stderr)
        sys.exit(1)
    outcome = 'already_converged'
else:
    new = (entry or {}).get('new') or {}
    if str(new.get(key)) != value:
        print(f'applied.verify_env does not carry {key}={value}: applied_keys={sorted(tool.get("applied") or {})} '
              f'restart_required_keys={sorted(tool.get("restart_required") or {})} entry={entry}', file=sys.stderr)
        sys.exit(1)
    outcome = 'applied'
print(json.dumps({'switched_to': value, 'commit': sha, 'outcome': outcome}))
RELOAD_PY
