# Eval-bootstrap RCA (layer 7): metric-collector ORCH_CONFIG_PATH env leak makes `tests_pass=true` unreachable — 2026-07-22

**Context.** Operator smoke re-runs (esc-2919-1 gate recipe / esc-2848-1 runbook step 1) after
tasks 2880 + 2881 landed. Run 1: 12:52Z, killed mid-cell-2 by a host background-task sweep.
Run 2: systemd unit `eval-smoke-0722`, 13:38:42Z → 17:20Z, completed — Phase-2 assertion
**FAIL [BUG 2]**: 0 of 5 implementer cells reached `outcome=done` + `metrics.tests_pass=true`.
Forensic: watcher-df sub-agent, per-cell ground truth + env-delta reproduction.

## Verdict: 2880 + 2881 WORK. A third, structural defect sits in the metric collector.

**Class-A exemplar (run 9351652d, claude-opus-high, run 1):** `outcome=done`, agent's final
verify attempt fully GREEN under the pinned worktree venv (pytest rc=0, pyright 0 errors —
after the agent fixed its own attempt-1 type error) — yet stamped `tests_pass=false`.

## Root cause (proven by minimal reproduction)

`collect_metrics` (evals/metrics.py:305) runs its own post-hoc
`run_verification(worktree, workflow.config)` with **no file scoping** → for evals this is the
fixture's `verify_commands` (runner.py:241): `cd orchestrator && uv run pytest tests/ -x` —
the whole test dir, fail-fast, inside the pre_task_commit snapshot.

The verify subprocess env (`_target_subprocess_env`, verify.py:2902-2921) scrubs
`_VENV_ISOLATION_KEYS` but **not `ORCH_*`** — and `eval-ofat --config ...` had already done
`os.environ['ORCH_CONFIG_PATH'] = str(config_path)` in-process (config.py:4012). So the
snapshot-era `tests/test_config.py::TestDefaults::test_default_values` (which instantiates
`OrchestratorConfig()` with no env isolation; main's current version monkeypatches
`ORCH_CONFIG_PATH` for exactly this reason, but the fixture snapshot 0b741353c2 predates that
hardening) loads the **current production `dark-factory-orchestrator.yaml`** and fails
`AssertionError: assert 24 == 3` (`max_concurrent_tasks`). `-x` aborts the suite there —
alphabetically before the task's own (passing) tests — and `tests_pass=false` is stamped.

**Reproduction (run-8181499e worktree, `cd <wt>/orchestrator`):**
- plain pinned-venv pytest on the test → **1 passed**
- `+ ORCH_CONFIG_PATH=/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` → **FAILED
  assert 24 == 3** (repr shows production `project_root`)
- `UV_PYTHON`/`DF_VERIFY_ROLE` alone → passed. **`ORCH_CONFIG_PATH` is the sole delta.**
- The log line `Verification env (mode=concurrent): ['DF_VERIFY_ROLE', 'UV_PYTHON']` is a red
  herring — it prints only the `verify_env` overlay; the base is the runner's full `os.environ`.

## This-run compounders (run 2, not structural)

1. **Capacity starvation:** max-g/e weekly-capped, max-a AUTH-FAILED (HTTP 401 — needs token
   refresh), and max-f became capped during the run → all 5 implementer cells burned their
   25-min budgets mostly in usage-gate probe sleeps → `outcome=cancelled` at ~1500s each
   (workflow time only 251-566s).
2. **Recurring fixture-induced agent red:** every cell's implementer edit trips pyright
   `"proc" is possibly unbound` in verify.py — cells that had time (run 1 cell 1) fix it and
   go green; the capacity-starved run-2 cells were cancelled before their fix iteration.
3. Architect cells (plan-only): 4/4 done, plan_quality 0.97-1.0 — unaffected throughout.

## Fix direction (filed as a task; not applied in the read-only forensic)

Scrub `ORCH_CONFIG_PATH`/`ORCH_*` in `_target_subprocess_env` (or set `ORCH_CONFIG_PATH=''`
in the eval `verify_env` overlay, mirroring main's own test hardening). Either kills the whole
class: any snapshot-era env-sensitive test is poisoned by any `ORCH_*` inherited from the
runner process.

## Sequencing for the campaign (esc-2848-1)

Next smoke attempt needs BOTH: (a) the env-scrub fix landed+available, and (b) real account
headroom (wait for max-a token refresh or a cap reset window). The layer chain so far:
2847 → 2851 → 2875 → 2876 → 2880/2881 → **this** (metric-collector env leak).

Evidence: result JSONs `df_task_12__*__{8181499e,fb3e66f2,c498b567,fd853cf7,9cf2162c,9351652d}.json`,
worktrees `/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-*`, smoke logs (session
scratchpad `smoke-run2.log`), forensic transcript in watcher-df session 2026-07-22.
