# RCA: why judge/triage invocation rows are EMPTY in runs.db (δ-rollup instrumentation "gap")

**Date:** 2026-07-20 · **Investigator:** claude-interactive (spawn `investigate-df-1062034`)
**Verdict:** **BENIGN / expected-by-design.** This is **not** a missing-`cost_store.record`
bug. All three roles are correctly instrumented; they are simply **gated off (judge),
rarely triggered (triage), or batch-scoped (module_tagger)** in production. The δ
`model_role_rollup` being empty for these roles is a *consequence of those roles being
dark in production*, not of a broken record site. **Filing an "add instrumentation" fix
task would double-instrument the already-landed 2461/2534 sites — do not.**

The one genuine finding is a **documentation/expectations mismatch** (the "judge already
records via `_invoke`" wording is code-accurate but production-misleading) plus a
**pilot-instrument limitation** the κ re-scope should account for. Both are judgment calls
entangled with Leo's active κ re-scope → **parked for Leo, not self-actioned.**

---

## Q1 — Confirm on current main

`data/orchestrator/runs.db` (`sqlite3 -readonly`), **15,644** rows,
window **2026-04-09T16:54:51Z → 2026-07-20T15:27:48Z** (live; the fleet has run since
07-19, so this is current, not the stale 07-19 snapshot).

| role | rows | task_id NULL | note |
|---|---|---|---|
| `judge` | **0** | — | absent from table |
| `triage` | **0** | — | absent from table |
| `module_tagger` | **9** | **9 (100%)** | up from 6 on 07-19; +3 new rows since, all `sonnet`, ~$0.30–0.70, `capped=0` |
| `steward` | 169 | 0 | first row 2026-07-16 — steward telemetry *works* |

(Populated roles unchanged: implementer 5343, reviewer_comprehensive 4592, architect 3531,
debugger 1157, escalation-watcher-auto 424 [all NULL task_id], unblock_auto 199, steward
169, simple_task 127, merger 51, deep_reviewer 42 [all NULL], module_tagger 9 [all NULL].)

The 07-19 finding reproduces exactly.

## Q2 — "not instrumented" vs "not invoked" → **NOT INVOKED** (on the recording path)

The labels are **not** hiding under a different `role` string. `role='judge'` and
`role='triage'` appear at exactly one write-path site each (below); no judge/triage rows
are folded into `reviewer_comprehensive` / `deep_reviewer` / `steward`. The roles simply
do not *run* on any path that writes to this DB in the window.

## Q3 — Trace the record sites (both are fully wired; both are gated)

**Single writer:** `shared/src/shared/cost_store.py::save_invocation` (`invocations` table).
Called from three places: `workflow.py::_invoke` (9601-9619, `capped=False`),
`review_checkpoint.py:260`, and `shared/cli_invoke.py::invoke_with_cap_retry` (1401-1421,
`capped=unattributed_cap`). steward/triage/module_tagger delegate to the cli_invoke
internal save (steward.py:681-688 comment documents this to avoid double-counting).

### Judge — instrumented, but gated OFF in production
- `workflow.py:5938` `_run_completion_judge` → `self._invoke(JUDGE, …)` → `_invoke`'s
  `save_invocation(role='judge')` at 9603-9619. **The record site exists and is correct.**
- **Gate:** `workflow.py:5717` `if self.config.judge_after_each_iteration:`.
  `config.py:2214` `judge_after_each_iteration: bool = Field(default=False)`. **Not set in
  the live `dark-factory-orchestrator.yaml` → False.** In production the completion judge
  **never runs.**
- Eval mode flips it on (`evals/runner.py:232` defaults it to `True`), **but** the eval
  path builds the workflow via `build_workflow(...)` (runner.py:385) **without a
  `cost_store`** → `_invoke`'s `if self.cost_store:` guard (9601) skips the write. So even
  eval-mode judge invocations write **nothing** to `data/orchestrator/runs.db`.
- Net: the "judge already records via `_invoke`" claim is **code-accurate but
  production-inert** — the site is live only under a flag that's off in prod, and the only
  path that flips it on has no cost_store. This is exactly what confused the 07-19 session.

### Triage — instrumented (task 2461), but rarely triggered
- `steward.py:706` `_pre_triage_suggestions` → `790` `invoke_with_cap_retry(role='triage',
  cost_store=self.cost_store, …)` → `cli_invoke.py:1403` `save_invocation(role='triage',
  capped=unattributed_cap)`. Landed by **task 2461 step-6** (`5e653a2bf1`,
  "GREEN - triage forwards cost telemetry kwargs"). **The record site exists and is
  correct. This is the ONLY `role='triage'` path (grep-confirmed).**
- **Gate:** `steward.py:426-432` — fires only when `escalation.category ==
  'review_suggestions'` **and** `len(suggestions) >= config.suggestion_triage_threshold`
  (default **10**, `config.py:2650`). No such large-batch review-suggestion escalation has
  occurred in the window. (steward itself only began recording 2026-07-16; the pre-triage
  sub-path is strictly rarer.)

### 2461 delivered what it claimed (no double-instrumentation)
2461 threaded `cost_store` into `TaskSteward` (steps 1-2) and forwarded cost-telemetry
kwargs for **both** steward (step-4) and triage (step-6). steward.py:681-688 / 804-808
explicitly document the internal-save delegation so no external second `save_invocation`
is added — **there is no double-count risk, and no missing record call.**

## Q4 — module_tagger `task_id=NULL` → **BY DESIGN, not a bug**
- `harness.py:2267-2271`: task_id is **intentionally omitted** — this is a *batch* tag over
  many untagged tasks, not scoped to one task; `invoke_with_cap_retry`'s `task_id or None`
  normalizes the default `''` to a NULL row. Landed by **task 2534 step-10** (`710acdcca3`).
- `digest.py:516-518` documents that the `LEFT JOIN invocations→task_results` handles this:
  module_tagger rows "still contribute invocation_count/cost/cap-hit but count as neither
  done nor blocked — an honest representation, not a dropped row."
- Consequence: `done_rate` / `blocked_rate` / `cost_per_done` are **structurally
  undefined** for module_tagger (no per-task outcome to join to). That is *honest* for a
  batch role with no single task outcome — not a broken metric to fix.

## Q5 — Verdict, double-count check, PRD ownership, pilot impact

**Verdict: benign / expected-by-design.** No missing `cost_store.record`. The 07-19
observation is real but its implied cause ("instrumentation gap → add a record call") is
wrong. Confirmed against the double-instrumentation warning in CLAUDE.md: the sites already
exist (2461 for steward/triage, 2534 for module_tagger); adding more would double-count.

**Impact on the κ/ι pilots** (which intend to use δ's `model_role_rollup` as a post-flip
watch + cost baseline):

| κ target | rollup observability today | why |
|---|---|---|
| **module_tagger** (κ 2540) | cost / cap-hit / invocation-volume ✅ · done / blocked / $-per-done ❌ | NULL task_id by design → no outcome join |
| **judge** (κ 2815) | **entirely blind in production** ❌ | judge is eval-only (`judge_after_each_iteration=False` in prod) and the eval path wires no cost_store. A production judge model-flip has *no production invocations to watch* — but also *no production effect*, since the judge runs only in evals. |
| **triage** (κ 2816, deferred) | observable but **sparse-to-empty** | fires only on ≥10-suggestion `review_suggestions` batches |

This corroborates the already-landed 07-19 graphiti fact: *"the G6 premise that pass/fail
thresholds derive from δ's model_role_rollup was determined to be false"* — the rollup
cannot serve as the pass/fail instrument for exactly these three roles.

## Recommendation (for Leo — parked, not self-actioned per brief)

1. **Do NOT file an instrumentation fix task.** The record sites are already wired
   (2461, 2534). The roles are dark/rare/batch by design.
2. **Recommended (clear-cut, docs-only):** soften the "the judge already records via
   `_invoke`" wording in `CLAUDE.md` / `plans/adaptive-model-routing-prd.md` to note the
   judge only records when `judge_after_each_iteration=True` — **eval-only today; the
   production gate is off and the eval path wires no cost_store**, so no judge rows reach
   `data/orchestrator/runs.db`. This is the sentence that tripped the 07-19 session. I did
   **not** file/commit it: the wording is entangled with the κ re-scope Leo is actively
   driving, so it's his call whether to fold it into that work.
3. **Decision-relevant for the κ judge pilot (2815):** the production `runs.db` rollup is
   *not* a viable watch instrument for the judge role — the judge only runs in evals. If
   Leo wants judge model telemetry, the options are (a) turn the judge on in production
   (a real cost/behaviour change) or (b) wire a cost_store into the eval harness (a new
   eval-telemetry feature) — both are new scoped work, his call, not a bug fix.
4. **module_tagger (2540) / triage (2816):** the rollup can watch module_tagger cost/cap/
   volume (not outcomes) and will be sparse for triage — consistent with the already-noted
   "δ rollup = OUTCOMES not quality → post-flip watch" framing.

## Evidence index
- runs.db: `data/orchestrator/runs.db` — `invocations`, `task_results`
- Judge gate: `orchestrator/src/orchestrator/workflow.py:5717`, `config.py:2214`,
  `evals/runner.py:232` + `385` (no cost_store), `workflow.py:9601` (guard)
- Judge record: `workflow.py:5938`, `9603-9619`
- Triage: `steward.py:426-432` (gate), `706-814` (`_pre_triage_suggestions`),
  `cli_invoke.py:1401-1421` (write); task 2461 (`3659966237`)
- module_tagger: `harness.py:2249-2273`, `digest.py:506-584`; task 2534 (`f2100595ba`)
- Writer: `shared/src/shared/cost_store.py:102-145`
