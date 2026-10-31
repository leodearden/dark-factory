#!/usr/bin/env bash
# scripts/merge-pytest-n-ab-switch.sh — flip the MERGE test leg's pytest-xdist
# `-n auto` pin (verify_env.PYTEST_XDIST_AUTO_NUM_WORKERS) to a new value,
# land ONLY the config through its checkout's own merge queue, and hot-reload
# the running orchestrator.
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
#   --dry-run         print the diff against main's copy; no landing, no reload
#
# Shares scripts/merge-deep-set-cap.sh's shape. The edit is computed against
# main's committed copy of the config and landed through the merge queue
# (scripts/_merge_queue_landing.py), never committed to main directly: such a
# commit is drift the queue never verified, which forces every merge in flight
# to re-verify in full (orchestrator/src/orchestrator/merge_lane/gates.py::
# _disjoint_skip_blockers). The checkout is never written by this script; only
# the queue's own landing syncs it. The landing waits for a full merge verify:
# MERGE_QUEUE_LANDING_TIMEOUT_SECS bounds it (default
# _merge_queue_landing.py::DEFAULT_LANDING_TIMEOUT_SECS), and a calling task's
# before_done.timeout_secs must exceed it. The reload goes through
# scripts/_config_reload_gate.py, which handshakes: the escalation MCP is
# STATEFUL, so a single-shot `tools/call` POST is rejected at the TRANSPORT
# layer — `Bad Request: Missing session ID`, HTTP 400, measured live on
# 2026-09-12 — before any tool runs.
#
# Exit 0 on either of the two ways the value can be live: the reload
# hot-applied it ('applied'), or the running config already carried it and the
# reload provably re-read THIS config file ('already_converged'). The last
# stdout line of every designed exit is a JSON verdict, for a kind='predicate'
# before_done note: {switched_to, commit, outcome} on exit 0 (--dry-run prints
# its own {dry_run, would_set}), or one carrying `failure: <tag>` on exit 1.
# `commit` is the sha the queue landed the flip as, `already-at-<value>` when
# main already carried it, or null when nothing landed. The tags come from
# die() below, scripts/lib/deploy_main_copy.sh, the reload step's fail(),
# scripts/_merge_queue_landing.py::LandingFailure and
# scripts/_config_reload_gate.py::ReloadFailure.
set -euo pipefail
die() {
    local failure="$1"; shift
    printf '{"failure": "%s"}\n' "$failure"
    echo "merge-pytest-n-ab-switch: $*" >&2
    exit 1
}

DRY=0; ARGS=()
for a in "$@"; do case "$a" in --dry-run) DRY=1;; *) ARGS+=("$a");; esac; done
[ "${#ARGS[@]}" -ge 1 ] || die usage "usage: merge-pytest-n-ab-switch.sh <value> [config_yaml] [escalation_port] [--dry-run]"
VALUE="${ARGS[0]}"
CONFIG="${ARGS[1]:-/home/leo/src/dark-factory/dark-factory-orchestrator.yaml}"
PORT="${ARGS[2]:-8102}"
[[ "$VALUE" =~ ^[0-9]+$ ]] || die invalid_value "value must be a positive integer, got '$VALUE'"
[[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) || die invalid_port "escalation_port must be a TCP port in 1-65535, got '$PORT'"
[ -f "$CONFIG" ] || die config_not_found "config not found: $CONFIG"
# One interpreter, resolved once, runs both python steps below. The landing
# and reload step imports its MCP transport from the checkout the SCRIPT lives
# in — never from $REPO below, which is the CONFIG's checkout and may be a
# different project entirely. That transport is not stdlib (httpx, pydantic),
# so the interpreter is resolved for the same reason: inheriting whatever
# `python3` a login shell offers makes this gate unreachable. Step 1's editor
# is stdlib-only but runs under the same interpreter, so the script has one
# resolution mechanism rather than two. This tree's one root .venv (CLAUDE.md,
# "Locating installed code"), else the caller's python3. Derived from the path
# rather than `git rev-parse` so a copy of this script outside any checkout
# still resolves under `set -e`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$SCRIPT_DIR/.."
PY="$CHECKOUT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3
KEY='PYTEST_XDIST_AUTO_NUM_WORKERS'
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# REPO, REL, BASE, and main's copy of the config (MAIN_COPY) beside the copy
# step 1 edits (EDITED), with the traps that let step 3 cancel on a killpg.
# shellcheck source=scripts/lib/deploy_main_copy.sh
source "$SCRIPT_DIR/lib/deploy_main_copy.sh"
deploy_main_copy "$CONFIG" die

# 1. Rewrite the value inside the top-level `verify_env:` mapping. Line-based
#    on purpose (no yaml round-trip: the file is 1000 lines of load-bearing
#    comments). Replaces an existing A/B marker line rather than stacking them.
"$PY" - "$EDITED" "$KEY" "$VALUE" "$STAMP" <<'PY' || die config_edit_refused "step 1 left main's ${REL} unedited (reason above)"
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
    diff -u --label "main:$REL" --label "$REL (edited)" "$MAIN_COPY" "$EDITED" || true
    echo "{\"dry_run\": true, \"would_set\": \"$VALUE\"}"
    exit 0
fi

# 2. Idempotent: if main already carried the value (a re-run after a runner
#    crash-resume), there is nothing to land and that is success — the reload
#    below still runs, and its converged branch reads its "nothing changed
#    here" answer as that same success rather than as a failure.
TO_LAND=("$EDITED")
if cmp -s "$MAIN_COPY" "$EDITED"; then TO_LAND=(); fi

# 3. Land the edit through the merge queue, then hot-reload via the escalation
#    MCP and assert the value is live. The landing is
#    scripts/_merge_queue_landing.py's; the transport, its failure diagnostics,
#    the argv-only heredoc contract and the check that the reload committed a
#    re-read of THIS file are scripts/_config_reload_gate.py's; this checks
#    verify_env.
#
#    The value is live in either of two shapes. `applied` carrying verify_env
#    with the new value is the flip. ABSENCE of verify_env from `applied` is
#    the converged re-run — and absence is the ONLY converged signal on the
#    wire, because `unchanged` is a bare int COUNT of equal leaves
#    (config.py::ConfigDiff), naming no keys and carrying no values. Neither
#    shape says anything about OUR file on its own: a rolled-back reload and a
#    reload of a DIFFERENT orchestrator produce the absence, and that other
#    orchestrator's own pending change to the same value produces the flip. So
#    both shapes are held to the re-read check, by passing this config file to
#    fetch_reload_report; the converged shape is further gated on verify_env
#    not being bucketed as restart-required.
"$PY" - "$SCRIPT_DIR" "$KEY" "$VALUE" "$CONFIG" "$PORT" "$REPO" "$REL" "$BASE" "$STAMP" \
    "${TO_LAND[@]}" <<'RELOAD_PY'
import json, os, sys
from pathlib import Path
script_dir, key, value, config_path, port, repo, rel, base, stamp, *to_land = sys.argv[1:]
sys.path.insert(0, script_dir)
from _config_reload_gate import ReloadNotConfirmed, fetch_reload_report
from _merge_queue_landing import LandingNotConfirmed, land_file


def verdict(commit, **result):
    print(json.dumps({'switched_to': value, 'commit': commit, **result}))


def fail(failure, message, *, commit):
    print(f'merge-pytest-n-ab-switch: {message}', file=sys.stderr)
    verdict(commit, failure=failure)
    sys.exit(1)


def land(edited):
    """The sha the merge queue landed *edited* as."""
    try:
        landing = land_file(
            repo, rel, base, Path(edited).read_bytes(), port=port,
            label=f'merge-pytest-n-{value}',
            message=f'config(ab): merge test leg {key}={value} (pytest -n A/B arm switch, {stamp})',
            timeout_secs=os.environ.get('MERGE_QUEUE_LANDING_TIMEOUT_SECS') or None,
        )
    except LandingNotConfirmed as exc:
        fail(exc.failure.value, exc.detail, commit=None)
    print(f'merge-pytest-n-ab-switch: landed {landing.branch} request_id={landing.request_id} '
          f'as {landing.landed_as}', file=sys.stderr)
    return landing.landed_as


sha = land(*to_land) if to_land else f'already-at-{value}'
try:
    tool = fetch_reload_report(port, committed_as=sha, config_path=config_path)
except ReloadNotConfirmed as exc:
    fail(exc.failure.value, exc.detail, commit=sha)
entry = (tool.get('applied') or {}).get('verify_env')
if entry is None:
    # verify_env is in RELOADABLE_FIELDS today, so a change to it is never
    # bucketed as restart-required (config.py::diff_config) and this never
    # fires. One allowlist edit away it would: a genuine flip then produces
    # this branch's exact shape — reloaded, our file, nothing under `applied`
    # — while the arm is landed and NOT live.
    if 'verify_env' in (tool.get('restart_required') or {}):
        fail('restart_required', f'verify_env changed but is restart-required, not hot-applied: '
                                 f'restart_required_keys={sorted(tool.get("restart_required") or {})} '
                                 f'(committed as {sha}; the value lands at the next restart)', commit=sha)
    outcome = 'already_converged'
else:
    new = (entry or {}).get('new') or {}
    if str(new.get(key)) != value:
        fail('value_mismatch', f'applied.verify_env does not carry {key}={value}: applied_keys={sorted(tool.get("applied") or {})} '
                               f'restart_required_keys={sorted(tool.get("restart_required") or {})} entry={entry}', commit=sha)
    outcome = 'applied'
verdict(sha, outcome=outcome)
RELOAD_PY
