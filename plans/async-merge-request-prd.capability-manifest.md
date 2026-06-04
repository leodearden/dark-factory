# Capability manifest — async merge_request redesign PRD (P2–P4)

Per-leaf capability→evidence bindings (mechanized G3+G6). Built at decompose time 2026-06-04.
PRD: `plans/async-merge-request-prd.md`. Verdict: **no FAIL bindings — batch clear to queue.**

One DAG-direction fix applied during manifest build: skill-migration leaves (β3–β7) assert
the `merge_status` polling protocol, whose producer α3 was not in their transitive dep
closure (β→β1→α2→α1 omits α3). Resolved by adding α3 as a direct prereq of β3–β7.

## α2 — merge_request returns request_id + already_merged fast-path (escalation)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `merge_request` MCP tool wired in live server | grep:escalation/src/escalation/server.py:533-534 `@mcp.tool() async def merge_request` | wired |
| `request_id` on MergeRequest | producer: **α1 upstream** (dep wired); kw_only precedent at merge_queue.py:1836 (`enqueued_at`) proves the dataclass-inheritance path past `GroupMergeRequest`'s non-default fields | producer-upstream |
| Ancestor check for fast-path | grep:git_ops.py — `is_ancestor` (`git merge-base --is-ancestor`), referenced at :1047-1102 companion docs; train PRD verified :896-902 | wired |
| git_ops handle reachable from the MCP tool | grep:server.py:596 `git_ops_for_scan = getattr(harness, 'git_ops', None)` — already used by the coalesce gate | wired |
| "no merge_queued event" assertable | emit chokepoint is `enqueue_merge_request` (merge_queue.py:1619-1636); fast-path returns before it | wired |

## α3 — merge_status MCP tool (escalation)

| Capability | Evidence | Verdict |
|---|---|---|
| Retention ring + `merge_finalized` events keyed by request_id | producer: **α1 upstream** (dep wired) | producer-upstream |
| Live queue snapshot (position/age/state) | producer: **task 1605 upstream** (in-progress 2026-06-04; dep wired). `get_merge_queue` not yet on main — verified absent via grep | producer-upstream |
| Event store persists across restarts | grep:event_store.py:44 `EventType(StrEnum)`, :230 `emit` → `data/orchestrator/runs.db` (sqlite) | wired |
| Restart-unknown hint semantics | I3 encodes existing lesson (memory: verify merge_request outcome via git log) — no new substrate | wired |

## β2 — merge_cancel tool (escalation)

| Capability | Evidence | Verdict |
|---|---|---|
| Worker drops cancelled-future entries without halting | grep:merge_queue.py:2409-2418 `_request_abandoned` ("abandoned by waiter (future cancelled) — dropping request without halting queue") | wired |
| Server-side waiter record holding the future (so cancel-by-request_id can reach it) | producer: **β1 upstream** (dep wired) | producer-upstream |
| Registry slot auto-release on future cancel | grep:merge_queue.py:1559 `future.add_done_callback(... _release)` — fires on cancellation too (docstring :1549-1551) | wired |

## β3–β7 — skill migrations (skills/*/SKILL.md, one file each)

| Capability | Evidence | Verdict |
|---|---|---|
| The five skill files exist and call merge_request | grep -l: skills/{merge-queue,unblock,unblock-low-risk,escalation-watcher,escalation-watcher-auto}/SKILL.md (all five verified 2026-06-04) | wired |
| Bounded `wait_secs` + non-blocking response shape | producer: **β1 upstream** (dep wired) | producer-upstream |
| `merge_status` polling target | producer: **α3 upstream** (dep wired — added during manifest build, see header) | producer-upstream |
| `merge_cancel` for β5's abort path | producer: **β2 upstream** (dep wired) | producer-upstream |
| Hard rule to retire is where the PRD says | grep:skills/escalation-watcher/SKILL.md:145 "**Hard rule: this session never calls `merge_request` at top level — no exceptions.**" | wired |
| Skills propagate without copy step | symlinked into ~/.claude/commands/ (memory: feedback_skill_symlinks) | wired |

## β8 — the flip: default wait_secs=0, delete unbounded branch (escalation)

| Capability | Evidence | Verdict |
|---|---|---|
| Unbounded await site to delete | grep:server.py:625 `outcome = await future` | wired |
| All five callers migrated first | producers: **β3–β7 upstream** (deps wired) — D5 compat ladder guarantees no caller window | producer-upstream |
| ≤100 s clamp achievable under MCP ceiling | MCP framework tool-call ceiling 120 s > 100 s clamp; bounded wait via `wait_for(shield(fut))` is stdlib asyncio (G6 numeric: bound > floor trivially) | wired |

## γ2 — bounded generation auto-chaining (orchestrator)

| Capability | Evidence | Verdict |
|---|---|---|
| Stranded-delta detection site | grep:merge_queue.py:477-504 `_check_post_merge_equivalence` → blocked outcome after main advanced | wired |
| Multi-waiter entry + chaining substrate | producer: **γ1 upstream** (dep wired) | producer-upstream |
| Bounded-counter escalation exemplar | grep:workflow.py:2577 `_check_merge_outcome_thrash` (counter+signature; memory: feedback_check_thrash_helper_pattern); blocked-outcome → workflow waiter escalation path exists (workflow.py:654 `_mark_blocked(escalate_to_human=True)`) | wired |
| `superseded_by` surfacing | retention record (α1, transitively upstream via γ1→β8→…→α1) + merge_status (α3 upstream of γ1's chain) | producer-upstream |
| Numeric premise: "≤2 generations, no human intervention" | bound = design constant (D3, settled); not a measured threshold — no floor check applicable; 3rd advance escalates by construction | valid |

## γ3 — workflow attaches as peer waiter; soft-cancel detaches (orchestrator)

| Capability | Evidence | Verdict |
|---|---|---|
| Workflow merge submission sites to convert | grep:workflow.py:3815-3833 (single-task `MergeRequest` + `enqueue_merge_request`), :592-619 (train tip path — preserved per D9) | wired |
| Soft-cancel hook | grep:workflow.py:3831-3833 `_await_cancellable` → `_handle_soft_cancel('merge')`; train :617-619 | wired |
| Outcome-mapping surface | grep:workflow.py:3835-3854 (status dispatch incl. conflict → `_resolve_and_resubmit`) | wired |
| Attach/detach API | producer: **γ1 upstream** (dep wired) | producer-upstream |

## δ1 — integration gate: 14 boundary scenarios (tests)

| Capability | Evidence | Verdict |
|---|---|---|
| Every mechanism under test | producers: **γ1, γ2, γ3 upstream** (deps wired; transitively the whole batch) | producer-upstream |
| Test homes exist | orchestrator/tests/ (existing merge-queue tests); escalation/tests/ (test_server_chokepoint.py — the existing merge_request chokepoint suite) | wired |
| Train non-regression baseline (scenario 12) | existing train tests on main (atomic-train PRD ε₁ landed; `_do_train_merge` + GroupMergeRequest at merge_queue.py:1839-1868, :1908) | wired |
| Skill-protocol checks avoid the negative-assertion trap | δ1's signal asserts the new protocol is documented, not literal-string absence (memory: feedback_negative_assert_prompt_self_conflict) | valid |

## Intermediates (substrate noted, not leaf-gated)

- **α1** [mq, ev]: `MergeRequest` dataclass at merge_queue.py:1824-1836; `EventType` StrEnum
  open for extension (event_store.py:44); done_callback chokepoint precedent :1559.
  Consumers: α2, α3, β1.
- **β1** [esc, mq]: blocking await to wrap at server.py:625; coalesce gate returns
  `inflight_task_id` today (server.py:605-623) — D8 upgrades it to the existing entry's
  request_id; position/queue_depth from **1605 upstream** (dep wired). Consumers: β2–β8.
- **γ1** [mq]: single-slot registry to restructure at merge_queue.py:1526-1592; MCP-only
  coalesce gate :1706-1821 (workflow bypass documented :1734-1736 — closed by **1604
  upstream**, dep wired); `git patch-id` = plain git CLI (no repo precedent; standard
  tool). Consumers: γ2, γ3, δ1.

## Excluded signals (G6 guard)

- "An interactive session's call never blocks 4 h again" is the incident's negation, not a
  testable leaf signal — covered by β8's clamp assertion (boundary tests 1+2).
- Live-queue soak behaviour (real cold-cargo verifies) is not unit-producible; the verify
  pipeline's latency is environmental, mocked at the worker boundary in δ1.
