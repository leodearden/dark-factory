# PRD — Curator single-call budget: pool-size scaling + restart-durable over-budget visibility

**Origin:** esc-task-curator-191 (`curator_failure` / `error_max_budget_usd`, reify, 2026-06-14 23:15Z) root-cause investigation.

## Problem / consumer + user-observable surface

The TaskCurator's **single-candidate** LLM call (`TaskCurator._call_llm`, task_curator.py:1692) passes a
**flat** `max_budget_usd = 0.30`. For projects whose corpus is dense with near-duplicate tasks
(reify's `reify-compiler:` lowering backlog), the assembled pool is routinely full (29–30/30 detail-rich
entries) → a ~41K-token prompt. At that size a legitimate 2-turn sonnet classification costs ~$0.29–0.32
(dominated by the CLI's default 1-hour prompt-cache *creation* at 2× base input, ~$0.246, plus ~$0.018
CLI-internal haiku overhead). esc-task-curator-191's call did real work (`num_turns=2`, `output_tokens=1995`,
`stop_reason=end_turn`, total `$0.30574`) and was essentially answering when the $0.30 ceiling tripped.

Two consequences:

1. **Budget is structurally too tight for a full pool.** 5 such trips in the past week, **all** single-call
   `task-curator[reify]`, **zero** batch (the R1 batch-budget scaling — `per_item_budget_usd`,
   `batch_budget_cap_usd=2.00`, task_curator.py:1758-1765 — already covers batches; the **single-call path,
   and the bisect-to-size-1 fallback that routes through `curate()`→`_call_llm`, were never given the same
   scaling**). Each trip degrades to `action='create'`, so a candidate that genuinely needed dedupe (4611 was
   later blocked as a superseded duplicate) enters the backlog anyway.
   - **Consumer:** the curator dedupe path / orchestrator backlog (fewer duplicate tasks created).
   - **User-observable surface:** a full-pool single-call curation renders a real drop/combine/create decision
     instead of the `error_max_budget_usd` → degrade-to-create fallback.

2. **Recurrence is invisible.** `CuratorEscalator._failure_log` is in-memory (curator_escalator.py:83). The
   fused-memory service is currently watchdog-restarting every ~10–20 min, which resets the burst counter, so
   every trip re-reads as "failure 1 of 3" and never crosses the 3-strike loud-note threshold — the same
   masking that hid the StructuredOutput outage. And `error_max_budget_usd` (a call that **did work**) is
   indistinguishable, in the escalation, from a hang.
   - **Consumer:** the L2 escalation-watcher / operator triage.
   - **User-observable surface:** an over-budget escalation whose "N of N" count is accurate **across restarts**
     and whose detail carries `cost_usd=` and `pool_sizes=` so triage can tell "working-but-over-budget"
     (→ raise budget) from a hang.

## Approach (sketch)

One cohesive task (both levers edit `task_curator.py` on the same `CuratorFailureError`/`report_failure` path;
splitting would force an awkward intermediate state under the narrow-file-lock model).

**Lever 1 — pool-size-scaled single-call budget.** Mirror the R1 batch formula onto `_call_llm`, scaling by
**pool size** (the cost driver), not candidate count:
`budget = min(max_budget_usd + per_pool_entry_budget_usd * len(pool), single_call_budget_cap_usd)`.
Add the two new knobs to `CuratorConfig` (config/schema.py). Choose `per_pool_entry_budget_usd` /
`single_call_budget_cap_usd` so a full 30-entry pool yields ≈ $0.60–0.75 (~2× the observed ~$0.31 legitimate
cost — headroom for harder 2–3-turn decisions + cache + haiku overhead) while an empty pool stays at the
$0.30 baseline. This fix automatically covers the bisect-to-size-1 fallback (it calls `curate()`→`_call_llm`).

**Lever 2 — restart-durable visibility.** (a) Persist the per-`(project_id, subtype)` failure timestamps so
the burst counter survives a restart and the existing "3rd-in-window → loud note" actually fires (store under
the reconciliation data dir, alongside `curator_events.db`; reload on `CuratorEscalator` construction).
(b) Thread `cost_usd` (add to `CuratorFailureError`, sourced from `AgentResult.cost_usd`) and `pool_sizes`
into `report_failure`, and render both in the `curator_failure` escalation detail.

## Pre-conditions (substrate — G3, verified)

- `AgentResult.cost_usd` exists (shared/src/shared/cli_invoke.py:231); not yet on `CuratorFailureError` —
  this task adds it.
- `pool_sizes` is in scope at the `report_failure` call site in `curate()` (task_curator.py:894-908).
- Reconciliation data dir + sqlite are available for escalator persistence (`curator_events.db` resolved at
  server/main.py:1282-1290); `CuratorEscalator()` currently holds no store — persistence is net-new but the
  substrate exists.
- `CuratorConfig` already carries the R1 batch knobs (config/schema.py:576-590); the two new single-call knobs
  are additive.

## Resolved design decisions

- **Scale by pool size, not a flat bump.** Empty-pool calls are cheap and must stay cheap; only full pools need
  headroom. Matches the R1 philosophy (scale the cost driver).
- **One task, not two.** Shared edit surface + atomic `CuratorFailureError` signature change.
- **`error_max_budget_usd` keeps the generic burst path, not the always-surface ZOT/schema-denied path.** A
  call that did work shouldn't flood per-occurrence; the right fix is durable counting + cost/pool tagging so a
  genuine pattern surfaces while one-off trips don't spam.

## Out of scope

- The fused-memory **watchdog crash-loop** (SIGABRT every ~10–20 min) — under separate investigation/fix
  (user-confirmed). It is the *amplifier* of the visibility problem, not its cause; lever 2 is correct
  regardless of restart cadence.
- Changing the CLI's 1-hour prompt-cache TTL (the largest single cost contributor) — not selected; not
  clearly controllable from `cli_invoke`. Revisit only if pool-size scaling proves insufficient.
- Shrinking `pool_total_cap` / `entry_details_chars` (dedupe-quality tradeoff) — not selected.

## Cross-PRD relationship

None. Self-contained within fused-memory (`task_curator.py`, `curator_escalator.py`, `config/schema.py`).
No seam owner needed (G4 N/A).

## Decomposition plan

- **α — Curator single-call budget: pool-size scaling + restart-durable over-budget visibility.**
  *Signal:* (1) a unit test asserting `_call_llm`'s `max_budget_usd` is `min(base + per_pool_entry*len(pool),
  cap)` across empty / partial / full / over-cap pools (mirrors the R1 test, test_task_curator.py:1851-1980);
  (2) a unit test asserting the escalator burst counter resumes from persisted state after a simulated restart
  (fresh `CuratorEscalator` instance reloads the log and the next failure reads "N+1 of 3"), and that a
  `curator_failure` escalation detail for an `error_max_budget_usd` trip contains `cost_usd=` and `pool_sizes=`.
  *Consumer:* curator dedupe path (lever 1) + L2 escalation-watcher/operator triage (lever 2).

## Open questions (tactical, implementation-time)

- Persistence medium for the escalator log: a small table in the existing `curator_events.db` vs a dedicated
  `curator_escalator_state.db` vs a JSON file under the data dir. Implementer's choice; keep it isolated from
  the cost store's schema.
- Exact `per_pool_entry_budget_usd` / `single_call_budget_cap_usd` values within the ≈$0.60–0.75 full-pool
  target.
