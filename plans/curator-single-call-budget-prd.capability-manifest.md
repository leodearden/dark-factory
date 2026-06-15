# Capability manifest — curator-single-call-budget-prd

One leaf (α). Each capability its signal asserts, bound to evidence. No binding resolves to a FAIL sentinel
(`declared-only | test-only | producer-downstream | producer-absent | producer-extent-short | fixture-ERROR |
bound≤floor | rejection-absent`), so the batch is clear to queue.

## α — Curator single-call budget: pool-size scaling + restart-durable visibility

| # | Capability asserted by signal | Binding | Evidence | Status |
|---|---|---|---|---|
| 1 | Single-call path passes `max_budget_usd` into the CLI call | `grep` wired | `fused-memory/src/fused_memory/middleware/task_curator.py:1692` (`max_budget_usd=self._config.curator.max_budget_usd`) | PASS |
| 2 | R1 batch scaling exists to mirror (formula + knobs) | `grep` wired | task_curator.py:1758-1765 (`budget = min(... + per_item_budget_usd*(n-1), batch_budget_cap_usd)`); config/schema.py:576-590 | PASS |
| 3 | New single-call knobs (`per_pool_entry_budget_usd`, `single_call_budget_cap_usd`) | producer:α (additive `CuratorConfig` fields) | config/schema.py:525-619 `CuratorConfig` — additive, produced by this task | PASS |
| 4 | `cost_usd` is available to thread into the escalation | `grep` wired | `shared/src/shared/cli_invoke.py:231` (`cost_usd: float = 0.0` on `AgentResult`) | PASS |
| 5 | `pool_sizes` is in scope at the `report_failure` call site | `grep` wired | task_curator.py:894-924 (`curate()` holds `pool_sizes` from `_build_corpus`, passes to the degrade `CuratorDecision`) | PASS |
| 6 | `CuratorFailureError` can carry `cost_usd` | producer:α (additive attribute, default-safe) | task_curator.py:61-… (existing pattern: timed_out/duration_ms/subtype/schema_tool_denied already attached) | PASS |
| 7 | Escalation detail is free-text and can render `cost_usd=` / `pool_sizes=` | `grep` wired | curator_escalator.py:235-262 (`detail_lines` builder for the generic `curator_failure` path) | PASS |
| 8 | Writable persistence substrate for the burst log exists | `grep` wired | `fused-memory/src/fused_memory/server/main.py:1282-1290` (reconciliation data dir / `curator_events.db`); sqlite available process-wide | PASS |
| 9 | Burst counter currently in-memory (the thing being made durable) | `grep` wired | curator_escalator.py:83 (`self._failure_log: dict[str, list[float]]`); construction main.py:516 (`CuratorEscalator()`, no store) | PASS |

**Numeric premise (G6):** the ≈$0.60–0.75 full-pool budget target is backed by the measured legitimate
full-pool cost `$0.30574` (esc-task-curator-191 result JSON); the cap is a ~2× headroom bound over a measured
value, not an unbacked exactness claim. No rejection-mechanism capability is asserted (no `rejection-absent`
risk).
