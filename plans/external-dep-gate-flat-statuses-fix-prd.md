# PRD — Cross-project external-dep gate: consumer reads the flat `{dep: status}` shape

**Status:** ready to decompose
**Author path:** `plans/external-dep-gate-flat-statuses-fix-prd.md`
**Project:** `dark_factory` (`/home/leo/src/dark-factory`)

## Problem / consumer + user-observable surface

The orchestrator scheduler's cross-project external-dep gate **never resolves a real
external dep to `done`**, so any task carrying `metadata.external_deps` is held forever.

- **Consumer / user-observable surface:** reify cross-project tasks `reify:4697` and
  `reify:4699` (warm-lane activation), whose sole unsatisfied gate is the external dep
  `dark_factory:1846` — which is genuinely `done` — currently sit `pending` and undispatched.
  Both emit `external_dep_gate_held{cause:"resolver_degraded"}` every tick (observed at
  `ticks≈3252` and climbing). The user-observable result of this PRD: those tasks dispatch
  (transition to `in-progress`) and the held-events stop.

### Root cause (diagnosed 2026-06-19)

`Scheduler.get_external_statuses` (`orchestrator/src/orchestrator/scheduler.py:1542`) parses
the MCP response with:

```python
statuses, parse_err = parse_tool_result(result, 'statuses', dict)
```

i.e. it expects the decoded payload to be a dict containing a top-level **`statuses`** key.
But the real fused-memory tool `get_external_statuses`
(`fused-memory/src/fused_memory/server/tools.py:2231`) does `return result`, where `result`
is a **flat** `{dep: status}` dict — e.g. `{"dark_factory:1846": "done"}`, **no `statuses`
wrapper**. So `parse_tool_result` reports `EnvelopeShape.KEY_ABSENT key='statuses'`
(`shared/src/shared/mcp_envelope.py:180-181`) → `ExternalResolverError` →
`external_resolver_failed` fail-safe-wait in `_deps_satisfied`
(`scheduler.py:1978-1985`) → the dep is **never** satisfied.

The sibling tool `get_statuses` **does** wrap (`tools.py:2150`, `return {'statuses': result}`),
which is why the consumer was (incorrectly) written to look for `'statuses'` for both.

**Lineage — why this survived two prior fixes:**
- `1578` built the tool returning a flat `dict[str,str]` (the documented, canonical contract).
- `1580` built the consumer gate using the `'statuses'` key — bug born here, but **silent**.
- `1799` (critical) root-caused the *symptom* (reify 4635 sat invisibly) and fixed
  **visibility only** — it set the error slot so the gate fail-safe-waits *loudly*
  (`external_dep_gate_held` events) instead of silently filtering. It did **not** correct the key.
- `1807` (`193023af62a`) folded the resolver onto `parse_tool_result`, faithfully
  **preserving** the wrong `'statuses'` key.

Both prior fixes iterated on the error-handling of a parse that can **never** succeed against
the real tool. Neither crossed the producer→consumer seam. The tests passed via **mock drift**:
every fixture feeds the *wrapped* shape — `test_cross_project_dispatch_integration.py:195`
(`json.dumps({'statuses': statuses})`) and `test_scheduler.py:4062/4082/4154` — encoding the
consumer's assumption, not the producer's flat output. No test exercises the real tool's shape.

## Approach

**Consumer-side fix (user-confirmed direction).** The tool's documented contract is flat
(`get_external_statuses(deps) -> dict[str, str]` keyed by the verbatim dep string; direct
callers and `CLAUDE.md` rely on flat). The producer is correct; **do not change the tool**.
Fix the consumer to read the flat dict, fix the drifted mocks, and add a **two-way seam test**
that pins the producer's real output shape against the consumer parser so this can never
silently re-drift.

Plus a defense-in-depth escalation backstop: a *permanent* resolver/parse failure
(`cause:"resolver_degraded"`) currently waits forever with no escalation — only the sentinel
causes (`unknown_project`/`unknown_task`/`malformed`) escalate. Make a persistent
`resolver_degraded` hold escalate to a human after a bounded number of ticks, consistent with
the project directive *prefer loud escalation over silent degradation*.

## Pre-conditions for activating

- None external. All substrate exists on `main` (verified 2026-06-19) — see capability manifest.
- No cross-project dependency for this PRD's own tasks (it fixes the cross-project machinery itself).

## Resolved design decisions

1. **Fix the consumer, not the producer.** The flat `{dep: status}` contract is canonical
   (1578, CLAUDE.md, live direct call). Wrapping the tool would break every direct caller.
2. **Pin the seam with a two-way test.** Task A's signal includes a test that runs the *real*
   `get_external_statuses` output shape through the scheduler resolver — the guard 1799/1807
   lacked. Drifted wrapped mocks are corrected in the same task.
3. **Escalation backstop is a separate task (B), depends on A.** With A landed,
   `resolver_degraded` means a *genuine* contract/transport failure, so escalating a persistent
   one is correct (not noise). Keeping it separate keeps A's blast radius minimal and lets the
   urgent unblock land first.

## Out of scope

- Changing the `get_external_statuses` tool return shape (producer stays flat).
- Reworking the sentinel-escalation path (`unknown_project`/`unknown_task`/`malformed`) — it
  already escalates; B only adds the `resolver_degraded` cause to the escalation set.
- The reify-side warm-lane work (`reify:4697`/`reify:4699`) — those dispatch once A lands.

## Cross-PRD relationship

No cross-PRD seams. The only "seam" is the in-repo producer (`fused-memory` tool) ↔ consumer
(`orchestrator` scheduler) contract, and this PRD explicitly fixes only the consumer side and
pins the contract with a test. No contested ownership.

## Decomposition plan

- **Task A (critical) — Consumer reads the flat `{dep: status}` shape + two-way seam test.**
  In `Scheduler.get_external_statuses`, extract the flat dict the tool actually returns
  (the decoded payload *is* the `{dep: status}` map) instead of `inner['statuses']`. Preserve
  the existing `missing`-dep resolver-degraded guard and the `ExternalResolverError` error-slot
  semantics from 1799. Correct the drifted mocks
  (`test_cross_project_dispatch_integration.py:195`, `test_scheduler.py:4062/4082/4154`) to emit
  the real **flat** shape. Add a two-way seam test that drives the *real*
  `get_external_statuses` tool output (flat `{dep:'done'}`) through the scheduler resolver and
  asserts the dep is marked satisfied (and that a non-`done` dep is **not** satisfied).
  *Observable signal:* the new seam test is green AND a pending task whose only unmet gate is a
  `done` external dep dispatches — no `external_dep_gate_held{resolver_degraded}` for it.
  *Consumer:* `reify:4697`, `reify:4699` dispatch.
  *Modules:* `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/tests/test_cross_project_dispatch_integration.py`,
  `orchestrator/tests/test_scheduler.py`.

- **Task B (high, depends on A) — Escalate persistent `resolver_degraded` holds.**
  Extend the external-dep hold path so a `resolver_degraded` cause that persists for a bounded
  number of consecutive ticks escalates to a human (via the existing `_on_external_dep_block`
  → `_mark_blocked(escalate_to_human=True)` pathway used by the sentinel path), instead of
  waiting silently forever. Reuse `max_external_dep_unresolved_cycles` (or an explicit sibling
  knob) as the threshold. `_note_external_hold` currently deliberately does **not** touch the
  sentinel counter (`scheduler.py:1589`); add a parallel bounded counter for the
  `resolver_degraded` cause.
  *Observable signal:* a unit test drives N consecutive `resolver_degraded` ticks and asserts an
  escalation is filed (`_on_external_dep_block` fires / a human escalation appears); below N, no
  escalation.
  *Consumer:* the human escalation queue (operator-visible).
  *Modules:* `orchestrator/src/orchestrator/scheduler.py` (`_note_external_hold` /
  `_apply_external_dep_policy` region), `orchestrator/tests/test_scheduler.py` (or the
  cross-project dispatch test module).

## Open questions (tactical — safe for an architect to decide)

- **Extraction mechanism for Task A:** extract the flat dict inline in
  `Scheduler.get_external_statuses` (orchestrator-only blast radius — preferred default), **or**
  add a whole-inner-dict / `key=None` mode to `shared.mcp_envelope.parse_tool_result` and reuse
  it. Either is acceptable; the inline form keeps the change single-package. Tactical.
- **Threshold value for Task B:** reuse `max_external_dep_unresolved_cycles` verbatim vs a
  dedicated `max_external_dep_resolver_degraded_cycles`. Either is fine; reuse is the default.
