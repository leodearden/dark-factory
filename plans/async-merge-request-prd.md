# PRD — Async merge_request: non-blocking MCP merge API + multi-waiter queue entries (P2–P4)

**Status:** active, 2026-06-04
**Slug:** `async-merge-request`
**Approach:** B + H (contract + two-way boundary tests). See § Contract and § Boundary-test sketch.
**Origin:** 2026-06-04 incident investigation (reify task 3112); settled design brief at
`~/.claude/spawn-briefs/merge-request-async-redesign-2026-06-04.md`. The brief's settled
decisions are final and are restated (not relitigated) in § Resolved design decisions.

## 1. Goal

No MCP caller can ever block unboundedly on the orchestrator merge queue, and a merge
submitted once is durable intent that lands (or escalates) without babysitting:

- From an interactive session against a backlogged queue, `merge_request` returns within
  seconds with `{status: queued, request_id, …}`; `merge_status(request_id)` later reports
  `done`. No MCP merge call can block > 100 s.
- Submitting a branch that is already on main returns `{status: already_merged, commit}`
  at the door — no redundant queue entry.
- (P4) Pushing new commits to a branch whose merge is in flight lands the delta
  automatically (≤ 2 auto-chained generations), with no human intervention.

## 2. Background — the 2026-06-04 incident (G6 premise, incident-validated)

`merge_request` (`escalation/src/escalation/server.py:534`) blocks until the merge worker
finalizes the request (`outcome = await future`, server.py:625) — unbounded latency. On
2026-06-04 an interactive /unblock session's call blocked ~4 h behind a saturated serial
verify pipeline (reify: 12–90 min cold-cargo verifies per entry). The entry was also a
guaranteed-redundant duplicate of an already-queued workflow-path merge the dedup gate
could not see. Three further pathologies:

1. **Stranded content** — the post-merge equivalence check
   (`merge_queue.py:477-504`) detects the branch tip advanced past the merged snapshot,
   returns a `blocked` outcome to the waiter (main has already advanced), and nothing
   re-queues the delta. Two amend commits were stranded; a coincidental second queue
   entry landed them hours later.
2. **Zombie entries** — the workflow's soft-cancel/retry loop re-enqueues new entries
   instead of re-attaching; cancelled-future entries are dropped later by the worker
   with "abandoned by waiter (future cancelled)" (`merge_queue.py:2409-2418`).
3. **Accidental cancellation semantics** — the only way a request gets cancelled today
   is implicitly, by the waiter's future being cancelled (client disconnect / task
   cancellation) — never by explicit intent.

## 3. Pre-conditions for activating

P1 interim tasks (already queued — **not** in this PRD's scope; dependencies wired at
decompose time):

| Task | Status (2026-06-04) | What it delivers | This PRD's use |
|---|---|---|---|
| 1604 | pending | harness-owned shared `InFlightMergeRegistry` across workflow + MCP enqueue paths | γ1 (multi-waiter core) builds on the unified registry |
| 1605 | in-progress | `get_merge_queue` read-only MCP tool — live MergeWorker snapshot (position, age, waiter_alive) | α3 (`merge_status`) and β1 (position/queue_depth in submit response) share this snapshot infrastructure |
| 1606 | pending | dashboard active-merges list | out of scope here (consumes 1605, not this PRD) |
| 1607 | done | dashboard recent-merges 24 h window | none |

## 4. Sketch of approach

Three phases over the existing chokepoint:

- **P2 (additive, non-breaking):** every `MergeRequest` carries a `request_id` and
  `snapshot_tip`; terminal outcomes are retained (in-memory ring ~200 + `merge_finalized`
  event in the event store, surviving restarts); new `merge_status` MCP tool; submit-time
  `already_merged` fast-path. Blocking `merge_request` keeps working, now also returning
  `request_id`.
- **P3 (breaking flip, via compat ladder):** `merge_request` gains `wait_secs`
  (server-clamped ≤ 100 s); a submitted merge becomes durable intent decoupled from the
  MCP call's lifetime (disconnect no longer cancels); explicit `merge_cancel` added; all
  five MCP-calling skills migrate to submit→poll; final server-only flip removes the
  unbounded-blocking path.
- **P4 (multi-waiter entries + generations):** a queue entry becomes
  `(branch, snapshot_tip, generation, waiters[])`; coalescing = attaching a waiter;
  workflow and MCP callers are peers on the same entry; post-merge equivalence failure
  auto-chains a bounded next generation; workflow soft-cancel detaches instead of
  cancelling — retiring the zombie-entry class.

## 5. Resolved design decisions

From the settled brief (final):

- **D1 — `wait_secs` clamped ≤ 100 s.** Bounded blocking convenience survives; no
  unbounded blocking mode on the MCP surface (MCP framework ceiling is 120 s).
- **D2 — Durable intent.** A submitted merge runs to completion unless explicitly
  `merge_cancel`led. Client disconnect no longer cancels (accepted semantic change).
- **D3 — Generation bound.** Max 2 auto-chained generations, then
  `_mark_blocked(escalate_to_human=True)` via the `_check_*_thrash` counter+signature
  pattern.
- **D4 — Phasing.** P2 → P3 → P4; P1 (1604–1607) already queued.

Resolved in this session:

- **D5 — P3 lands via a compat ladder, not one atomic batch.** `wait_secs=None`
  (default during the ladder) preserves today's blocking semantics; `wait_secs >= 0`
  opts into bounded/non-blocking. Each skill migrates independently — a migrated skill
  (explicit `wait_secs` + `merge_status` polling) is correct under BOTH server modes, so
  no window exists where any caller misreads `queued` as failure. The final flip (β8) is
  a small server-only change: default becomes `0`, the unbounded branch is deleted. This
  honours the brief's atomicity requirement (no broken window) without a multi-package
  mega-task that would blow the architect budget (memory: split multi-package tasks).
- **D6 — Submit-time fast-path is the ancestor check; patch-id containment arrives
  with P4.** At submit, `git_ops.is_ancestor(branch_tip, main)` answers the
  already-merged case (the incident's shape). Patch-id comparison (rebase/cherry-pick
  rewrites) requires the P4 identity machinery and lands there (γ1); the fast-path is
  upgraded in place.
- **D7 — Retention hooks the future, not the 12 resolve sites.** `enqueue_merge_request`
  registers `result.add_done_callback` to record the terminal outcome into the ring and
  emit `merge_finalized` — one chokepoint, mirroring how `InFlightMergeRegistry.acquire`
  already auto-releases (`merge_queue.py:1559`). Worker resolve paths stay untouched.
- **D8 — Pre-P4, `attached` means "your intent is represented by the existing entry."**
  A coalesced submission returns the existing entry's `request_id` (today it returns
  only `inflight_task_id`); the caller polls that id. True multi-waiter attach semantics
  (per-waiter terminal answers, tip-relation handling) arrive in P4 without changing the
  response shape.
- **D9 — Trains stay on the direct path.** `GroupMergeRequest` (atomic-train PRD) is
  harness-internal, single-waiter by construction, and continues to use
  `enqueue_merge_request` directly. Multi-waiter entries apply to single-branch
  requests only; the worker's train handling is preserved bit-identical.
- **D10 — `merge_status` accepts `request_id | branch | task_id`; `request_id` is
  authoritative.** Branch/task_id lookups resolve to the most-recent entry (live first,
  then retention ring, then event store) and include the resolved `request_id` so
  callers can switch to the precise handle.

## 6. Cross-PRD relationship (G4)

| Other PRD / surface | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/orchestrator-atomic-train-merge-prd.md` | coexists | `GroupMergeRequest` + `_do_train_merge` in the same MergeWorker this PRD restructures | this PRD owns non-regression (D9: trains stay on direct path; boundary test 12) | wired |
| `plans/escalation-repend-state-machine-prd.md` (PRD-3) | none (verified disjoint) | PRD-3's merge gating lives in workflow gates **before** merge submission; it declares merge_queue.py untouched (its § seam table, register: nobody) | n/a | no intersection |
| P1 tasks 1604/1605 (no PRD; queued from same incident) | consumes | shared `InFlightMergeRegistry` (1604); `get_merge_queue` snapshot infra (1605) | P1 tasks produce; this PRD's α3/β1/γ1 consume via task deps | queued/in-progress |
| Dashboard | produces (indirect) | `merge_status` states surface via 1605/1606's snapshot rendering | 1606 owns rendering | out of scope |
| Skills `merge-queue`, `unblock`, `unblock-low-risk`, `escalation-watcher`, `escalation-watcher-auto` | produces | submit→poll calling convention; retirement of the "never call merge_request at top level" hard rule (`skills/escalation-watcher/SKILL.md:145`) | this PRD (β3–β7) | queued |

## 7. Contract (B+H) — signatures and invariants

### 7.1 MCP API (post-P3 shapes)

```
merge_request(task_id, branch, worktree, description='', wait_secs=0) → immediate:
  {status: 'queued'|'attached', request_id, snapshot_tip, generation,
   position, queue_depth, eta_seconds}
  {status: 'already_merged', commit}            # submit-time fast-path
  {status: 'unknown_branch'|'error', ...}       # immediate failures unchanged
  # wait_secs > 0: bounded convenience wait, SERVER-CLAMPED to ≤ 100 s;
  # returns the latest state (terminal outcome if reached) at expiry.

merge_status(request_id=None, branch=None, task_id=None) →
  {state: 'queued'|'verifying'|'gate'|'finalizing'|'done'|'conflict'|'blocked'|
          'abandoned'|'superseded'|'unknown',
   request_id, generation, superseded_by?, position?, eta_seconds?,
   outcome?, started_at?, finished_at?, hint?}

merge_cancel(request_id) →
  {cancelled: bool, state, reason?}   # explicit waiter detach / entry cancel
```

**Invariants:**

- I1 — No `merge_request` code path awaits longer than the clamp. The unbounded
  `await future` (server.py:625) is deleted at β8; bounded waits use
  `wait_for(shield(future), clamp)` so expiry never cancels the entry (D2).
- I2 — `request_id` is unique per entry-generation, assigned at `MergeRequest`
  construction, and stable across CAS retries / re-verify within the same generation.
- I3 — Every terminal outcome is observable via `merge_status` after the entry leaves
  the queue, and across orchestrator restarts (event-store row keyed by request_id).
  A truly unknown id returns `{state: 'unknown', hint: 'check git log main'}` — honest
  restart semantics (encodes the "verify merge_request outcome via git log" lesson).
- I4 — Submit-time fast-path: if `branch_tip` is an ancestor of main, return
  `already_merged` without enqueueing and without a `merge_queued` event.
- I5 — `merge_cancel` is the ONLY cancellation path. Pre-P4 it cancels the entry's
  future (worker drops it at the existing `_request_abandoned` checkpoint,
  `merge_queue.py:2409`); post-P4 it detaches the calling waiter and cancels the entry
  only when the waiter count reaches zero.
- I6 — Train requests (`GroupMergeRequest`) bypass all new machinery except request_id
  + retention; their worker behaviour is bit-identical (boundary test 12).

### 7.2 P4 entry model

```python
# conceptual; lives in orchestrator/src/orchestrator/merge_queue.py
Entry = (branch: str, snapshot_tip: sha, generation: int, waiters: list[Waiter])
Waiter = (request_id, source: 'mcp'|'workflow', submitted_tip: sha, future)
```

Identity semantics for a new request for branch B at tip `T_new` vs an entry holding
`T_old` (settled in the brief):

| Relation | Handling |
|---|---|
| `T_new == T_old` | Pure coalesce — attach waiter |
| `T_old` ancestor of `T_new` (superset) | Entry not yet verifying: re-snapshot entry to `T_new`. Already verifying: attach waiter + chain a generation-2 entry for the delta; waiter's terminal answer comes from gen 2 |
| `T_new` ancestor / patch-id-contained in `T_old` (subset) | Attach waiter; at finalize, per-waiter containment check answers done/already_merged |
| Divergent (rebase/cherry-pick rewrite) | Patch-id compare first: equal content → subset case. Genuinely divergent → same as superset (re-snapshot or chain gen 2). Last-writer-wins on the branch ref |

**Invariants:**

- I7 — Post-merge equivalence failure becomes "tip advanced since snapshot" →
  auto-chain the next generation. Bounded: max 2 auto-generations, then
  `_mark_blocked(escalate_to_human=True)` (counter+signature, the `_check_*_thrash`
  shape). Chained entries get fresh request_ids; the superseded entry's retention
  record carries `superseded_by`.
- I8 — Workflow at merge phase attaches to an existing entry instead of duplicating.
  Outcome mapping: done + tip contained → merge phase complete; done but tip not
  contained → waiter rides the chained generation; conflict/blocked → as if its own
  merge failed (existing retry/steward logic, `workflow.py:3835-3854`).
- I9 — Workflow soft-cancel = detach waiter, not cancel entry. Entry proceeds while
  waiters remain; dropped at the next checkpoint when waiter count hits zero.
- I10 — All registry mutations stay synchronous within the event loop (no `await`
  between check and write) — the race-freedom property the current registry documents
  (`merge_queue.py:1535-1537`).

### 7.3 Polling protocol (skill-facing)

Submit with `wait_secs=100`; if non-terminal, poll `merge_status(request_id)` with
backoff 15 s → 60 s, using `eta_seconds` as the hint when present. `state: 'unknown'`
after an orchestrator restart → fall back to `git log main` (I3 hint). The
escalation-watcher hard rule "never call merge_request at top level"
(`skills/escalation-watcher/SKILL.md:145`) is retired at β6 and replaced by the new
invariant: **every `merge_request` call passes an explicit bounded `wait_secs`;
completion is awaited only via `merge_status` polling** (per the
encode-the-invariant lesson, the rule names the protocol, not just the call site).

## 8. Boundary-test sketch (two-way: MCP/skill side ↔ worker side)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Non-blocking submit against busy queue | Worker mid-verify on another branch; `wait_secs=0` | Returns < 5 s with `{status: queued, request_id, position ≥ 1, queue_depth}`; `merge_queued` event emitted |
| 2 | Bounded wait expiry | Same; `wait_secs=600` | Returns ≤ ~100 s (clamped) with latest non-terminal state; entry still queued (shield: future not cancelled) |
| 3 | Submit-time already_merged | Branch tip is ancestor of main | `{status: already_merged, commit}` in seconds; no `merge_queued` event; queue untouched |
| 4 | merge_status across lifecycle | Entry queued → verifying → done | States observed in order; after worker finishes, `merge_status` returns `done` + outcome from retention ring |
| 5 | merge_status across restart | Merge finalized; orchestrator restarted (ring empty) | `merge_status(request_id)` returns terminal state from event store; unknown id → `{state: unknown, hint}` |
| 6 | Explicit cancel | Entry queued, not yet picked up | `merge_cancel` → worker drops at checkpoint ("abandoned by waiter" log); `merge_status` → `abandoned`; queue not halted |
| 7 | Disconnect ≠ cancel (durable intent) | MCP client disconnects after submit (wait_secs=0 return or mid-bounded-wait) | Entry survives, merge completes; `merge_status` from a NEW session returns `done` |
| 8 | Coalesce returns request_id (pre-P4 `attached`) | Entry in flight for branch B; second submit for B at same tip | `{status: attached, request_id == first entry's id}`; `merge_coalesced` event |
| 9 | P4 multi-waiter peer completion | Workflow waiter + MCP waiter on same entry | Both futures resolve with the same terminal outcome; one merge executed |
| 10 | P4 soft-cancel detach | Two waiters; workflow soft-cancels | Entry proceeds; MCP waiter gets `done`; no "abandoned by waiter" drop; workflow re-attach on retry coalesces |
| 11 | P4 generation chain | Entry verifying at snapshot T1; branch advances to T2 | Post-merge equivalence detects delta → gen-2 entry auto-chained for T2; gen-1 `merge_status` → `{state: superseded, superseded_by}`; gen-2 lands delta; 3rd consecutive advance → `_mark_blocked(escalate_to_human=True)` |
| 12 | Train non-regression | `GroupMergeRequest` for a 3-member train | Worker behaviour bit-identical to pre-PRD (train tests still green); request carries request_id + retention only |
| 13 | Subset waiter answer | Entry at T_old verifying; submit at T_new = ancestor of T_old | Waiter attached; at finalize waiter's answer is `already_merged`/`done` via containment check |
| 14 | Skill migration end-to-end | Backlogged queue; run /merge-queue flow | Submission returns in seconds; skill polls per § 7.3; final state reported correctly |

## 9. Decomposition plan

Labels are PRD-local; IDs assigned at decompose time. Modules: `esc` =
`escalation/src/escalation/server.py` (+ `escalation/tests/`), `mq` =
`orchestrator/src/orchestrator/merge_queue.py`, `wf` =
`orchestrator/src/orchestrator/workflow.py`, `ev` =
`orchestrator/src/orchestrator/event_store.py`.

### Phase P2 — identity + status surface (additive)

- **α1 — MergeRequest identity + terminal-outcome retention.** [mq, ev]
  `request_id: str` (kw_only, default_factory, following the `enqueued_at` precedent at
  mq:1836 — required because `GroupMergeRequest` adds non-default fields) and
  `snapshot_tip: str | None` (kw_only) on `MergeRequest`; retention ring (~200) +
  `merge_finalized` EventType keyed by request_id, both registered via
  `add_done_callback` in `enqueue_merge_request` (D7).
  *Intermediate — unlocks α2, α3, β1.* Sanity signal: `merge_finalized` rows with
  request_id appear in `data/orchestrator/runs.db` after any merge.
- **α2 — merge_request returns request_id + submit-time already_merged fast-path.** [esc]
  Response gains `request_id` on all shapes; before enqueue, resolve branch tip and
  `git_ops.is_ancestor(tip, main)` → `{status: already_merged, commit}` (I4).
  *Leaf.* Signal: submitting an already-merged branch returns `already_merged` within
  seconds with no `merge_queued` event; normal submissions carry `request_id`.
  Prereqs: α1.
- **α3 — merge_status MCP tool.** [esc]
  Lookup order: live snapshot (1605 infra) → retention ring → event store → unknown+hint
  (I3, D10). *Leaf.* Signal: `merge_status(request_id)` of a live entry returns
  state/position/eta; of a finalized merge after orchestrator restart returns the
  terminal outcome; unknown id returns `{state: unknown, hint: 'check git log main'}`.
  Prereqs: α1, **1605**.

### Phase P3 — non-blocking flip (compat ladder, D5)

- **β1 — Non-blocking submit + durable intent.** [esc; mq param wiring]
  `wait_secs` param: `None` (compat default) = legacy blocking; `0` = immediate;
  `>0` clamped ≤ 100 with `wait_for(shield(future), …)` (I1 mechanics). Server-side
  waiter record holds the future — call lifetime decoupled, disconnect no longer
  cancels (D2/I5 substrate). Non-blocking response per § 7.1 incl. position/queue_depth
  from 1605's snapshot; coalesced submissions return the existing entry's request_id
  with `status: attached` (D8). *Intermediate — unlocks β2…β8.*
  Prereqs: α2, **1605**.
- **β2 — merge_cancel tool.** [esc]
  Resolve request_id → cancel the entry's future → existing `_request_abandoned` drop
  (mq:2409) + retention records `abandoned`. Unknown/terminal ids return
  `{cancelled: false, state, reason}`. *Leaf.* Signal: boundary test 6 — cancelled
  entry shows `abandoned` via merge_status; worker log shows the drop without halting
  the queue. Prereqs: β1.
- **β3 — Migrate skill: merge-queue.** [skills/merge-queue/SKILL.md] Submit→poll per
  § 7.3. *Leaf.* Signal: a /merge-queue run against a backlogged queue returns from
  submission in seconds and reports the final state via polling; the skill prose
  contains the explicit-wait_secs invariant. Prereqs: β1, β2.
- **β4 — Migrate skill: unblock.** [skills/unblock/SKILL.md] Same shape. *Leaf.*
  Signal: as β3 for the /unblock merge step. Prereqs: β1, β2.
- **β5 — Migrate skill: unblock-low-risk.** [skills/unblock-low-risk/SKILL.md]
  Abort-unless-done becomes submit → poll → on any non-done doubt: `merge_cancel` +
  abort (preserving its abort-clean contract). *Leaf.* Signal: dry-run trace shows
  submit→poll→cancel-on-abort; no unbounded wait remains in the flow. Prereqs: β1, β2.
- **β6 — Migrate skill: escalation-watcher + retire the top-level hard rule.**
  [skills/escalation-watcher/SKILL.md] Replace the :145 hard rule with the § 7.3
  protocol invariant (encode-the-invariant lesson — name the protocol, not the call
  site). *Leaf.* Signal: SKILL.md documents bounded-submit+poll; no prose instructs
  unbounded blocking anywhere in the file. Prereqs: β1, β2.
- **β7 — Migrate skill: escalation-watcher-auto (L1 retry-land path).**
  [skills/escalation-watcher-auto/SKILL.md] Same shape, including its retry-land flow.
  *Leaf.* Signal: as β6 for the auto-watcher. Prereqs: β1, β2.
- **β8 — The flip: default wait_secs=0, delete the unbounded branch.** [esc]
  Remove `None`-compat; clamp is the only wait path (I1 complete). *Leaf.* Signal:
  boundary tests 1+2 — `merge_request(wait_secs=600)` against a busy queue returns
  ≤ ~100 s; default call returns immediately with `queued`; grep shows no unbounded
  `await future` in merge_request. Prereqs: β3, β4, β5, β6, β7.

### Phase P4 — multi-waiter entries + generations

- **γ1 — Multi-waiter entry core + identity semantics.** [mq]
  Entry = `(branch, snapshot_tip, generation, waiters[])`; attach/detach; the § 7.2
  relation table (same-tip coalesce; superset re-snapshot when not verifying; subset
  attach + per-waiter containment at finalize; divergent patch-id compare);
  waiter-count-zero drop at checkpoint; submit-time fast-path upgraded with patch-id
  containment (D6). Trains untouched (D9). *Intermediate — unlocks γ2, γ3, δ1.*
  Prereqs: **1604**, β8.
- **γ2 — Bounded generation auto-chaining.** [mq]
  Post-merge equivalence failure → auto-chain gen-(n+1) for the delta; max 2
  auto-generations then `_mark_blocked(escalate_to_human=True)` (I7); `superseded_by`
  in retention + merge_status. *Leaf.* Signal: boundary test 11 — commits pushed
  mid-verify land automatically as gen-2 with no human intervention; gen-1
  merge_status shows `superseded`/`superseded_by`; 3rd consecutive advance escalates.
  Prereqs: γ1.
- **γ3 — Workflow attaches as peer waiter; soft-cancel detaches.** [wf]
  Merge phase attach instead of duplicate enqueue (I8 outcome mapping at
  wf:3815-3854 + train-tip path untouched); soft-cancel = detach (I9). *Leaf.* Signal:
  boundary tests 9+10 — soft-cancelled workflow leaves the entry completing for the
  remaining MCP waiter; the zombie re-enqueue class ("abandoned by waiter" on
  multi-waiter entries) is gone. Prereqs: γ1.
- **δ1 — Integration gate: boundary-test table end-to-end.** [orchestrator/tests/,
  escalation/tests/] All 14 § 8 scenarios implemented as discrete tests (skill
  scenarios 14 as protocol-conformance checks against the skill prose, avoiding the
  negative-assertion trap — assert the new protocol is documented, not that a literal
  forbidden string is absent). *Leaf — the B+H integration gate.* Signal: all 14
  scenarios pass in CI. Prereqs: γ1, γ2, γ3.

**Topological order:** α1 → {α2, α3} → β1 → β2 → {β3…β7} → β8 → γ1 → {γ2, γ3} → δ1,
with out-of-batch deps 1605 → {α3, β1} and 1604 → γ1.

## 10. Out of scope

- **Dashboard rendering** of queue/status states — 1606's seam (consumes 1605).
- **Reconciliation / fused-memory systems** — untouched.
- **Train multi-waiter semantics** — trains stay single-waiter on the direct path (D9).
- **N>2 generation chains / configurable bound** — fixed at 2 then escalate (D3).
- **Foreign-process kill for stale merge worktrees** — existing reap-only stance
  preserved (mq:1771-1783).
- **Backpressure / queue prioritisation** — ordering stays FIFO + speculative worker.

## 11. Open questions (tactical)

1. **request_id format.** Suggested: `mr-<8-char-uuid4>` (short, log-greppable,
   no task_id collision when one task submits twice). Decide during α1.
2. **Ring buffer eviction.** Suggested: plain deque(maxlen=200); event store is the
   durable tier so eviction is lossless. Decide during α1.
3. **`merge_status` state names for worker-internal phases** (`verifying`/`gate`/
   `finalizing` granularity depends on what 1605's snapshot exposes). Map to whatever
   1605 lands; degrade to `verifying` if finer states aren't available. Decide during α3.
4. **Bounded-wait implementation detail** — `asyncio.wait_for(asyncio.shield(fut))` vs
   `asyncio.wait({fut}, timeout=…)`. Both satisfy I1/D2; pick during β1 (note the
   FastMCP off-loop lesson: sync tool callbacks run on threadpool workers — merge_request
   is async today and must stay async).
5. **Should β8 also rename `in_flight` → `attached` in any residual response shape?**
   Suggested: yes, with `in_flight` kept as a deprecated alias for one release. Decide
   during β8.

## 12. References

- Brief: `~/.claude/spawn-briefs/merge-request-async-redesign-2026-06-04.md`.
- Code anchors (verified on main, 2026-06-04):
  - `merge_request` tool + blocking await: `escalation/src/escalation/server.py:534-634`
    (await at :625); registry injection :56,93-98.
  - `InFlightMergeRegistry`: `orchestrator/src/orchestrator/merge_queue.py:1526-1592`
    (done_callback auto-release :1559; loop-synchronous invariant :1535-1537).
  - `coalesce_or_enqueue_merge_request` (MCP-only dedup gate): mq:1706-1821; explicitly
    NOT used by workflow paths (:1734-1736 — the 1604 gap).
  - `MergeRequest`: mq:1824-1836 (kw_only precedent `enqueued_at` :1836);
    `GroupMergeRequest`: mq:1839-1868; `MergeOutcome`: mq:1871-1886.
  - `_request_abandoned` (cancelled-future drop): mq:2409-2418.
  - Post-merge equivalence check (stranded-content site): mq:477-504.
  - Workflow merge submission: single-task `workflow.py:3815-3833` (+ outcome mapping
    :3835-3854, soft-cancel :3831-3833); train tip path :592-619.
  - Event store: `EventType` StrEnum `event_store.py:44` (merge_queued :66,
    merge_coalesced :68); `emit` :230; persists `data/orchestrator/runs.db`.
  - `git_ops.is_ancestor`: `orchestrator/src/orchestrator/git_ops.py` (~:896-902, plus
    companion check ~:1047-1102).
  - Skill callers: `skills/{merge-queue,unblock,unblock-low-risk,escalation-watcher,escalation-watcher-auto}/SKILL.md`;
    hard rule at `skills/escalation-watcher/SKILL.md:145`.
- Memory lessons encoded: verify-merge-outcome-via-git-log (I3), encode-the-invariant
  (β6), split-multi-package-tasks (D5), `_check_*_thrash` pattern (I7),
  negative-assertion trap (δ1), FastMCP off-loop callbacks (Open Q4).
