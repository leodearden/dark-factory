# Eval-bootstrap RCA (layer 5+6): smoke gate STILL cannot pass after 2875+2876 — 2026-07-22

**Context.** Operator re-run of the architect-effort eval campaign smoke gate
(esc-2848-1 runbook step 1), authorized by Leo, after task 2875 (3.14t venv
pin) and task 2876 (aiosqlite full-workspace provisioning + hardened BUG-2
smoke assertion, merged `19fe347569`) both landed. Run window
2026-07-22 09:04–09:37 UTC, `scripts/eval_bootstrap_smoke.sh` (fixture
`df_task_12`, default `SMOKE_TIMEOUT=25`), operator-killed (scoped
`kill -TERM -- -<pgid>` of the smoke's own session, pgid 3913592) after the
outcome was proven structurally doomed — see defects below. Spend: ~$3.06
(cell 1 agent) + cell 2 killed pre-agent (~$0).

**Verdict: the 2875+2876 fixes WORK — but two further, previously-masked
bootstrap defects still make a genuine smoke pass impossible.** The gate's
hardened assertion (>=1 implementer cell `outcome=done` with
`metrics.tests_pass=true`) remains unreachable. This is bootstrap layer 5+6
in the 2847 → 2851 → 2875 → 2876 sequence; each earlier layer masked the
next.

## What 2875+2876 fixed, confirmed working this run

- Worktree setup completed in **3 s** (`cd orchestrator && uv sync` OK),
  worktree venv = Python 3.13.9 (2875 confirmed).
- `Provisioned eval-verify deps ['aiosqlite'] into <wt>/orchestrator/.venv`
  (2876's `snapshots.ensure_eval_verify_deps` confirmed live).
- The implementer agent ran end-to-end: 56 turns, $3.06, "All three plan
  steps are complete, committed, and marked done" (cell
  `df_task_12 × claude-opus-high`, run `ca3f40d4`).

## Defect A — EVAL_PROFILE null-sentinel FM URL × 120 s transport retry window

**Mechanism.** `EVAL_PROFILE['fused_memory.url'] = 'http://127.0.0.1:1'`
(D8 — "non-routable null sentinel; immediate ECONNREFUSED, never
production", `orchestrator/src/orchestrator/evals/profile.py`). The intent
is instant-fail. But the transport it lands on —
`McpSession._raw_call` (`orchestrator/src/orchestrator/mcp_lifecycle.py`) —
wraps EVERY logical call in `fm_retry_backoffs()`
(`FM_RESTART_RETRY_WINDOW_SECS = 120.0`, hardcoded, no env knob;
`orchestrator/src/orchestrator/fm_retry.py`). So each eval-side FM call
burns a full ~120 s retry window (13–14 attempts) before failing, instead
of failing in milliseconds.

**Evidence (this run's log).** 11 distinct `initialize … attempt 1/N` retry
cycles in a 33-minute run ≈ **22 min of pure dead overhead**. Cell 1
timeline: 4 cycles (≈8.3 min) before the agent was even invoked
(09:04:32→09:12:50Z), agent ran 698 s to success (09:25:08Z), then
post-workflow cycles consumed the remainder; the runner killed the cell at
exactly the wall-clock timeout: `Eval complete: df_task_12 ×
claude-opus-high → cancelled (total=1502.6s, workflow=698.3s)` — **verify
never started, after the workflow had already succeeded**. The two 07-21
cells died identically (wall 1502/1503 s vs workflow 208/410 s).

**Impact.** With ~8–12 min of overhead + a ~12 min opus implementer +
verify, no implementer cell can finish inside the default 25-min timeout —
the smoke can only ever produce `cancelled` implementer cells. The full
trimmed OFAT (~198 cells) would waste ~10+ min/cell of the same overhead
(~33+ machine-hours) and likely timeout-kill every long cell.

**Fix directions (for the follow-up task to decide).** Fail-fast the
null-sentinel: when `fused_memory.url` is the D8 sentinel (or on immediate
`ECONNREFUSED` to a loopback sentinel), skip/shrink the fm-restart retry
window — the 120 s window exists to ride out a *fused-memory restart*,
which by construction cannot happen for `127.0.0.1:1`. Alternatively (or
additionally) raise the smoke/OFAT per-cell timeout defaults. Constraint:
no edits to `evals/runner.py` scoring/benchmark/judge machinery.

## Defect B — main-repo root `conftest.py` shadows the fixture's pinned code in eval verify

**Mechanism.** Eval worktrees live at
`<repo>/.eval-worktrees/<fixture>/<run>` — **nested under the main repo**.
Any pytest run inside them picks up the main repo's root `conftest.py`
(ancestor conftest collection), which `sys.path.insert(0, …)`s the
**current-main** `<subproject>/src` dirs and pre-imports `orchestrator`,
`shared`, etc. Every `import orchestrator.*` in the fixture's test run then
resolves to **today's code, not the fixture's pinned `pre_task_commit`
code**.

**Proof (cheap, reproducible, no LLM).** In the cell-1 worktree
(`.eval-worktrees/df_task_12/run-ca3f40d4`, base `0b741353c2` + the agent's
verify-only commits):

```
cd <wt>/orchestrator
env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python -m pytest tests/test_config.py -q
# → 1 failed (test_load_from_yaml: models.implementer 'sonnet' != expected 'opus'), 3 passed
env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python -m pytest tests/test_config.py -q --confcutdir=<wt>/orchestrator
# → 4 passed
```

Same file copied outside the repo tree: 4 passed. The fixture-era code
default is `implementer='opus'`; the failure value `'sonnet'` is
current-main's default arriving through the shadowed import. The agent saw
the same class of failure live and (reasonably) misread it as
"pre-existing".

**Impact.** (1) Deterministic false-red on `df_task_12` → `tests_pass=true`
is unreachable → the hardened smoke gate can NEVER genuinely pass while the
shadowing exists, at any timeout. (2) Broader eval validity: any historical
eval verify that ran pytest in a nested worktree partially tested
current-main code against fixture-era tests. (3) The same walk-up applies
to any tool with ancestor config discovery run from inside eval worktrees.

**Fix directions (for the follow-up task to decide).**
(i) Relocate `.eval-worktrees/` outside the repo root (precedent: the
fix-forward rule "worktree OUTSIDE `.worktrees/`" after crash-recovery
deletions, df#2633) — the structural fix, kills the whole
ancestor-contamination class; (ii) or make eval verify pytest invocations
hermetic (`--confcutdir=<wt>` / rootdir pinning / `PYTEST_ADDOPTS`);
(iii) or a neutralizing `conftest.py`/marker at the worktree boundary.
Option (i) also protects lint/typecheck config walk-up; whichever is
chosen must land in the bootstrap (snapshots/verify seam), not the scoring
machinery.

## Also observed (not blockers, recorded for completeness)

- **Agent-env config leak (cosmetic here):** the in-agent test run resolved
  `fused_memory.url` to `:8002` (agent env carries live-config vars via the
  workflow env), while the clean/verify env resolves the fixture default
  `:8000`. Verify's `_target_subprocess_env` does not inherit the agent's
  overlay, so this did not affect the verify verdict — worth a scrub only
  if it recurs.
- **Launch-env hygiene:** the operator launch leaked `CLAUDE_SPAWN_*` into
  the eval env (known false-red class for `test_session_hooks` on old
  fixture trees, cf. task 2643). Scrub on the next operator launch.
- **Capacity:** max-g / max-f / max-e all weekly-capped during the run
  (resets: f → 07-22 13:00Z; e → 07-25 17:00Z; g → 07-26 13:00Z); pool
  effectively max-c / max-b / max-d + eval-injected max-a. Cell 1 burned
  three 429 probes before landing on an uncapped account.

## Disposition

- Smoke gate: **FAILED (structural, proven mid-run; run killed to stop
  doomed spend)**. esc-2848-1 left **PENDING** per the runbook.
- Follow-up bootstrap-fix tasks filed (one per defect):
  **task 2880** (Defect A: null-sentinel fail-fast / timeout adequacy) and
  **task 2881** (Defect B: eval-worktree conftest isolation).
- Re-run sequence once both land: fresh
  `scripts/eval_bootstrap_smoke.sh` (clean env, scrubbed `CLAUDE_SPAWN_*`;
  consider `SMOKE_TIMEOUT=60` until Defect A lands) → REAL pass → trimmed
  OFAT per esc-2848-1 → committed verdict → resolve the gate.
