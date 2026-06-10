# Capability manifest — retroactive merge-queue train coalescing

PRD: `plans/merge-train-queue-coalescing-design.md` (committed 0231712eab).
Binds each leaf signal's asserted capabilities to evidence on main (G3+G6 mechanized).
Verified 2026-06-10 against main @ 0231712eab.

## Batch shape

| Label | Task | Leaf? |
|---|---|---|
| α | workflow `superseded` merge-outcome consumer | intermediate (→ ζ; also unblocks γ2 gate precondition #1) |
| β | harness-injected train-callback factory for the merge worker | intermediate (→ γ) |
| γ | retroactive coalescing pass in SpeculativeMergeWorker | intermediate (→ δ, ε, ζ) |
| δ | merge-ready confidence gate for coalescing candidates | intermediate (→ ζ) |
| ε | merge_status `superseded_by` surfacing + /merge-queue skill follow-the-train | intermediate (→ ζ) |
| ζ | integration gate: end-to-end boundary scenarios + enablement notes | **leaf** |

All foundation tasks are roped into the ζ integration-gate leaf (C-as-integration-gate pattern).

## ζ (integration gate) — capability → evidence

| Capability | Evidence | Verdict |
|---|---|---|
| `'superseded'` in `MergeOutcome.status` Literal | grep `orchestrator/src/orchestrator/merge_queue.py:3091` — declared on main | PASS |
| `MergeOutcome.superseded_by` field | grep `merge_queue.py:3109` — declared; populated by γ (producer upstream of ζ) | PASS |
| Workflow consumer for `'superseded'` outcomes | ABSENT on main (named γ2 blocker, `merge_queue.py:153-166`) → producer: task α, upstream of ζ | PASS (producer queued upstream) |
| Waiter-detach check (`req.result.cancelled()`) | grep `merge_queue.py:3980`, `:4676` (`waiter_alive`) — wired on the production worker path | PASS |
| Waiting-queue introspection (`self._queue._queue`) | grep `merge_queue.py:4727` — `snapshot()` reads it on the production path | PASS |
| `_select_train_members` (stackability selection) | grep `workflow.py:380`, wired into `_maybe_form_train` `workflow.py:958` (production β-former path) | PASS |
| `git_ops.get_changed_line_ranges` | wired `workflow.py:953` (β-former fan-out) | PASS |
| `git_ops.stack_train_branches` (conflict → eject) | wired `workflow.py:970` | PASS |
| `GroupMergeRequest` + worker dispatch via `_do_train_merge` | grep `merge_queue.py:3055` (type), `:5086` (SpeculativeMergeWorker dispatch), `:4162` (plain worker) — wired | PASS |
| Train tip full-scope verify before advance (correctness invariant) | `_do_train_merge` routes through the shared post-merge core (task 1596 done); one `verify.sh --scope all` on the merged tip | PASS |
| `mark_member_done` / `status_check` callbacks constructible worker-side | ABSENT on main (only `workflow.py:665-787` builds them, scheduler captured; worker ctor `harness.py:3273` has no scheduler) → producer: task β, upstream of γ → ζ | PASS (producer queued upstream) |
| Registry fan-out of resolved future to attached waiters | `InFlightMergeRegistry` done-callback mirroring `merge_queue.py:2374-2397`; detach-cancel race guard `:2421-2430` | PASS |
| Terminal-outcome retention records `superseded_by` | grep `merge_queue.py:2689`, `:2708`, `:2730` (γ2 scaffolding, landed) | PASS |
| `EventType.train_formed` | grep `workflow.py:1020` (emission), event_store StrEnum — wired | PASS |
| `EventType.train_coalesced` | NEW → producer: task γ, upstream of ζ | PASS (producer queued upstream) |
| `merge_status` durable-tier outcome passthrough | grep `escalation/src/escalation/server.py:1266-1275` — wired; `superseded_by` NOT yet included → producer: task ε, upstream of ζ | PASS (producer queued upstream) |
| Confidence signal source reachable from the worker | worker receives `event_store` at construction (`harness.py:3277`) — recent merge_attempt / terminal-outcome history queryable; scheduler metadata NOT reachable (excluded from δ's design) | PASS |
| `merge_train_max_members` config (cap=3, ge=2) | grep `orchestrator/src/orchestrator/config.py:923` | PASS |
| `merge_train_coalesce_enabled` config knob | NEW (default False, human-flip) → producer: task γ, upstream of ζ | PASS (producer queued upstream) |
| Economic premise (train derail cost +EV for N≤3) | external decision GO-N3, reify esc-4455-16, s(3)=0.962 — no new numeric premise asserted by any leaf signal | PASS |

No FAIL bindings. Every absent capability has a named in-batch producer strictly upstream
of the leaf that asserts it (no DAG inversion). No leaf signal references the live
2026-06-10 queue snapshot (4442/4455/cargo-run-prebuilt-fix) — it is illustrative only.

## DAG

```
α ──────────────┐
β → γ → δ ──────┤→ ζ (leaf)
        γ → ε ──┤
        γ ──────┘
```
