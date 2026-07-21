#!/usr/bin/env bash
#
# eval_bootstrap_smoke.sh — go/no-go gate for the eval-worktree bootstrap fix
# (task 2847). This MUST pass before re-running the task-2539 architect-effort
# campaign: it proves the two eval-worktree bootstrap defects are fixed
# end-to-end, so an eval architect/implementer run behaves like real dispatch —
#
#   BUG 1: plan-tools MCP is wired into the eval architect session, so the
#          architect actually produces a plan (architect result
#          metrics.plan_steps > 0 & outcome == done), not the starved
#          plan_steps=0 structural floor (esc-df_task_12-3).
#   BUG 2: eval verify runs under the worktree's OWN pinned 3.13 .venv, so each
#          implementer worktree's .venv/bin/python is Python 3.13.x
#          (esc-df_task_13-3).
#
# Cost: one live `eval-ofat` over the OFAT candidate set × a single fixture
# (default df_task_12) — real LLM budget and minutes. It is the operator
# go/no-go gate, NOT a CI test. Set SMOKE_SKIP_EVAL to run ONLY the Phase-2
# assertions over whatever results already sit in RESULTS_DIR — the
# assertion-logic path the test suite (test_eval_bootstrap_smoke.py) drives
# over synthetic result JSONs, with no LLM spend.
#
# Env knobs (all optional):
#   SMOKE_SKIP_EVAL    set to any value → skip the live eval-ofat; assert only.
#   SMOKE_RESULTS_DIR  results dir to assert over
#                      (default: orchestrator/src/orchestrator/evals/results —
#                      where eval-ofat writes).
#   SMOKE_FIXTURE      fixture id to run/assert (default: df_task_12).
#   SMOKE_CONFIG       orchestrator config YAML
#                      (default: <repo>/dark-factory-orchestrator.yaml).
#   SMOKE_TIMEOUT      per-eval-run timeout in MINUTES (default: 25).
#
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${SMOKE_CONFIG:-$REPO_ROOT/dark-factory-orchestrator.yaml}
FIXTURE=${SMOKE_FIXTURE:-df_task_12}
RESULTS_DIR=${SMOKE_RESULTS_DIR:-$REPO_ROOT/orchestrator/src/orchestrator/evals/results}
PYBIN=$REPO_ROOT/.venv/bin/python
TIMEOUT=${SMOKE_TIMEOUT:-25}

# --- Phase 0: tie the harness/JSON-parser to the pin -------------------------
# The result JSON below is parsed with $PYBIN; fail fast unless it is itself the
# pinned 3.13, so a stale/mismatched interpreter can't silently mask a BUG-2
# regression.
ver=$("$PYBIN" --version 2>&1 || true)
case "$ver" in
  "Python 3.13."*) : ;;
  *)
    echo "SMOKE FAIL (Phase 0): $PYBIN is '$ver', expected Python 3.13.*" >&2
    echo "  The smoke harness must run under the repo's pinned 3.13 .venv." >&2
    exit 1
    ;;
esac

# --- Phase 1: live eval-ofat (skipped entirely when SMOKE_SKIP_EVAL set) ------
shopt -s nullglob
if [[ -z "${SMOKE_SKIP_EVAL:-}" ]]; then
  # Snapshot the pre-existing result set so Phase 2 asserts only over THIS
  # run's freshly-written results, not stale ones from a prior run.
  declare -A _before
  for f in "$RESULTS_DIR/${FIXTURE}__"*.json; do _before["$f"]=1; done

  tasks_dir=$(mktemp -d)
  trap 'rm -rf "$tasks_dir"' EXIT
  cp "$REPO_ROOT/orchestrator/src/orchestrator/evals/tasks/${FIXTURE}.json" "$tasks_dir/"

  echo "SMOKE: running eval-ofat over ${FIXTURE} (timeout ${TIMEOUT}m/run)..."
  uv run --project "$REPO_ROOT/orchestrator" orchestrator eval-ofat \
    --config "$CONFIG" --tasks-dir "$tasks_dir" \
    --max-parallel 1 --trials 1 --timeout "$TIMEOUT"

  check_files=()
  for f in "$RESULTS_DIR/${FIXTURE}__"*.json; do
    [[ -n "${_before[$f]:-}" ]] || check_files+=("$f")
  done
else
  # Dry-run: assert over ALL of this fixture's results already in RESULTS_DIR.
  check_files=("$RESULTS_DIR/${FIXTURE}__"*.json)
fi

# --- Phase 2: assert BUG 1 & BUG 2 over the result JSONs (parsed via $PYBIN, no jq)
"$PYBIN" - "${check_files[@]}" <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

paths = sys.argv[1:]
results = []
for p in paths:
    try:
        with open(p) as fh:
            results.append((p, json.load(fh)))
    except (OSError, ValueError) as e:
        print(f"SMOKE FAIL: could not read result JSON {p}: {e}", file=sys.stderr)
        sys.exit(1)


def is_architect(r):
    cfg = r.get("config_name", "")
    role = (r.get("metrics") or {}).get("role_under_test")
    return cfg.startswith("architect-") or role == "architect"


def resolve_venv_pythons(wt):
    """Return EVERY interpreter uv created for this worktree (possibly several).

    Checks the direct <wt>/.venv/bin/python first (the top-level-venv layout —
    a single venv); else EVERY one-level subproject match <wt>/*/.venv/bin/python
    — where uv lands the venv when a fixture's setup does `cd <subproject> &&
    uv sync` with UV_PROJECT_ENVIRONMENT scrubbed (verify._target_subprocess_env),
    e.g. df_task_12 → <wt>/orchestrator/.venv. A worktree whose setup builds more
    than one subproject venv (e.g. both fused-memory/.venv and orchestrator/.venv)
    yields more than one path, and the BUG-2 gate version-checks EVERY one (not
    just the alphabetically-first) so a wrong interpreter in ANY subproject is
    caught. Returns [] when none exist. Direct-first keeps the top-level-venv
    smoke path unchanged (task 2875)."""
    direct = Path(wt) / ".venv" / "bin" / "python"
    if direct.exists():
        return [str(direct)]
    return [str(p) for p in sorted(Path(wt).glob("*/.venv/bin/python"))]


architects = [(p, r) for p, r in results if is_architect(r)]
implementers = [(p, r) for p, r in results if not is_architect(r)]

# --- BUG 1: plan-tools MCP wired into the eval architect session -------------
if not architects:
    print(
        f"SMOKE FAIL [BUG 1]: no architect result found among {len(results)} "
        "result(s). The eval architect must run and produce a scorable plan; "
        "plan-tools MCP must be wired into eval architect sessions "
        "(run_architect_eval).",
        file=sys.stderr,
    )
    sys.exit(1)

for p, r in architects:
    steps = (r.get("metrics") or {}).get("plan_steps", 0)
    outcome = r.get("outcome")
    if not (isinstance(steps, int) and steps > 0) or outcome != "done":
        print(
            f"SMOKE FAIL [BUG 1]: architect result {p} has plan_steps={steps!r} "
            f"outcome={outcome!r} (need plan_steps>0 & outcome=done). plan-tools "
            "was NOT wired into the eval architect session → the architect wrote "
            "no plan.",
            file=sys.stderr,
        )
        sys.exit(1)

# --- BUG 2: implementer verify ran under the pinned 3.13 worktree venv -------
for p, r in implementers:
    wt = r.get("worktree_path")
    if not wt:
        print(
            f"SMOKE FAIL [BUG 2]: result {p} has no worktree_path to check the "
            "pinned 3.13 .venv against.",
            file=sys.stderr,
        )
        sys.exit(1)
    venv_pys = resolve_venv_pythons(wt)
    if not venv_pys:
        print(
            f"SMOKE FAIL [BUG 2]: no worktree venv found for {p} at "
            f"{wt}/.venv/bin/python or {wt}/*/.venv/bin/python. The eval "
            "implementer worktree venv is missing — verify did not run under a "
            "pinned 3.13 .venv.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Version-check EVERY resolved venv (a worktree may build more than one
    # subproject venv); a wrong interpreter in ANY of them fails the gate.
    for venv_py in venv_pys:
        try:
            out = subprocess.run(
                [venv_py, "--version"], capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as e:
            print(
                f"SMOKE FAIL [BUG 2]: could not run {venv_py} --version: {e}. The "
                "eval implementer worktree venv is missing — verify did not run "
                "under a pinned 3.13 .venv.",
                file=sys.stderr,
            )
            sys.exit(1)
        pyver = (out.stdout + out.stderr).strip()
        if not pyver.startswith("Python 3.13."):
            print(
                f"SMOKE FAIL [BUG 2]: {venv_py} is '{pyver}', expected Python "
                "3.13.* — eval verify was NOT pinned to the worktree's own 3.13 "
                ".venv (verify_env UV_PYTHON / scrubbed setup env).",
                file=sys.stderr,
            )
            sys.exit(1)
        # BUG 2 hardening (task 2876): a 3.13 interpreter is necessary but NOT
        # sufficient. Eval verify's pytest collection imports orchestrator.config
        # → shared/__init__ → shared.async_sqlite_base → `import aiosqlite`, and
        # the eval worktree is checked out at an OLD pre_task_commit whose
        # orchestrator-only `uv sync` predates the shared[vllm] dep and so omits
        # aiosqlite. Probe it directly — a version-only gate emits a hollow SMOKE
        # PASS on a 3.13 venv whose verify actually dies at collection (the
        # false-negative that let 2875 merge). snapshots.ensure_eval_verify_deps
        # is what provisions it; this is the loud backstop if that ever regresses.
        try:
            imp = subprocess.run(
                [venv_py, "-c", "import aiosqlite"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as e:
            print(
                f"SMOKE FAIL [BUG 2]: could not probe `import aiosqlite` via "
                f"{venv_py}: {e}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if imp.returncode != 0:
            tail = (imp.stdout + imp.stderr).strip()[-300:]
            print(
                f"SMOKE FAIL [BUG 2]: {venv_py} cannot `import aiosqlite` "
                f"(rc={imp.returncode}) — eval verify would die at pytest "
                "collection (orchestrator.config → shared → import aiosqlite). "
                "The worktree venv is missing aiosqlite; "
                f"snapshots.ensure_eval_verify_deps must provision it.\n{tail}",
                file=sys.stderr,
            )
            sys.exit(1)

print(
    f"SMOKE PASS: {len(architects)} architect result(s) with plan_steps>0 & "
    f"done (BUG 1 fixed); {len(implementers)} implementer worktree venv(s) on "
    "Python 3.13.* (BUG 2 fixed)."
)
PYEOF
