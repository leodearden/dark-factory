# Capability manifest — dashboard-park-stack-visibility-prd

Mechanizes G3 + G6 per **leaf** task (γ, η, ζ). Each capability the leaf's signal asserts is bound to evidence:
`grep:<file>:<line>` (wired on main) · `producer:task-<label> upstream` (in the dependency closure) ·
`existing` (landed substrate verified this session). Any `declared-only | test-only | producer-downstream |
producer-absent | producer-extent-short | rejection-absent` binding **blocks** the batch. **No FAIL bindings.**

Intermediates (α, β, δ, ε) carry no user-observable leaf signal; their deliverables are the producers the leaves
bind against. δ additionally owns the **negative/rejection** mechanism for boundary test B4.

---

## γ — Scheduler tab shows buried owners + flags stranded; `modules[].parked_by` populated

| Capability asserted by signal | Binding | Verdict |
|---|---|---|
| `park_stacks` snapshot key includes shadowed/buried owners | `producer:task-α upstream` (γ→β→α; α adds `snapshot_park_stacks()` with `shadowed` flag) | PASS |
| `read_scheduler_state` passes `park_stacks` through to the dashboard | `producer:task-β upstream` (β adds the skeleton key + pass-through test) | PASS |
| Liveness oracle to detect stranded owners (owner ∉ live set) | `existing` — dashboard already joins the active-task set at `data/scheduler.py:255`; `get_statuses` MCP tool exists | PASS |
| Active park owner per module for `parked_by` | `producer:task-α upstream` (park_stacks top) / also derivable from existing `parks` key | PASS |
| `modules_conflict` prefix-rule to resolve `parked_by` like `holder` | `grep:data/scheduler.py:210` (already used to resolve `holder`) | PASS (existing) |

## η — parked-unheld module renders orange w/ owner (Task-detail card + Orchestrators view)

| Capability asserted by signal | Binding | Verdict |
|---|---|---|
| `modules[].parked_by` present in SCHEDULER data | `producer:task-γ upstream` (η→γ) | PASS |
| Shared chip reads per-module fields via `buildSchedLockInfo` | `grep:tabs.jsx:179` + `grep:tab_tasks.jsx:334-338` (both already read `m.holder` off the same module entry; `parked_by` is the same access path) | PASS (existing) |
| `lock-parked` orange CSS class | owned by η (the leaf creates it) | PASS |
| Precedence held(red) > parked(orange) > available(grey) | logic owned by η; both inputs (`holder` existing, `parked_by` from γ) upstream | PASS |

## ζ — clicking evict clears the ghost park; alert clears next snapshot

| Capability asserted by signal | Binding | Verdict |
|---|---|---|
| `request_park_eviction` MCP tool enqueues a request | `producer:task-ε upstream` (ζ→ε) | PASS |
| Scheduler tick drains request + `force_clear(owner)` | `producer:task-δ upstream` (ζ→ε→δ; δ builds on existing `prune_owners`, `scheduler.py:2968`) | PASS |
| `force_clear` removes the park from the next snapshot (alert clears) | `producer:task-δ upstream`, reflected via α's snapshot rendered by γ | PASS |
| Dashboard `_scheduler_proxy` POST→MCP pattern | `grep:dashboard/src/dashboard/app.py:1160` (existing override proxy) | PASS (existing) |
| Evict offered only for verified-stranded rows (UI guard) | `producer:task-γ upstream` (`parked_owner_live`/stranded rows) + owned by ζ | PASS |

### Rejection mechanism (G6 branch 4) — boundary test B4, owned by δ (intermediate)

| Negative assertion | Binding | Verdict |
|---|---|---|
| Evict request against a **LIVE** owner is **refused** (no force_clear; `reservation_force_evict_refused` emitted) | `rejection-check` owned by **δ**: δ authors a live-owner request and asserts the refusal event fires + park intact (B4 test). Authoritative guard is server-side, not UI. | PASS — bound, not deferred |

**Result:** all leaf capabilities resolve to PASS / existing-substrate / upstream-producer. The safety rejection
(B4) is bound in δ. Batch is clear to queue.
