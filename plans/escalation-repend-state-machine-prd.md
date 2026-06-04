# PRD: Escalation re-pend state machine & merge gating

**Status**: active — authored 2026-06-04 (PRD-3 of the escalation-flow trio; origin: Brief 3 of
`plans/escalation-flow-2026-06-04-prd-briefs.md`, a 16-agent verified audit).
**Approach**: B + H (design-first; contract + boundary tests). One state machine, one
correctness-class issue — semantics decided here, before decomposition.
**Goal frame** (shared across the trio): increase throughput by reducing issue-handling and
scheduling latency, without sacrificing final correctness of code reaching main.

## Goal

One coherent escalation-resolution state machine in which:

- Every resolution **names its intent** (`action` enum) and every intent has exactly one,
  state-independent task outcome. The chronic ambiguity of `terminate=true` is removed, not
  re-documented.
- **Born-at-L2** (critical/urgent) escalations gate workflow progress and merge — today they are
  the *only* class that cannot stop a merge (gates filter `severity=='blocking' and level==0`).
- A human's dismissal is **durable**: a parked/abandoned task is never circularly re-asked
  (today: dismissed → blocked → ≤900s Fix#1b → `task_failure` L1 → auto-watcher promotes back to
  the same human, context lost).
- Re-block loops are **bounded** (signature-aware guard, threshold 3) instead of burning one full
  agent budget + one human round-trip per lap, unbounded.
- The resolve→flip **crash window self-heals** at startup and within one sweep period mid-run,
  without a human round-trip.
- Docs tell the truth: `resolve_issue` docstring, both watcher skills, dead `review_suggestions`
  handler.

## Consumers (G1)

- **L2 human / escalation-watcher sessions** — decisions become durable; never re-asked after a
  dismissal; the resolve API self-documents its effect.
- **The scheduler** — tasks re-enter (`pending`), park (`deferred`), or terminate (`cancelled`)
  promptly and correctly; no zombie `blocked` rows.
- **The orchestrator budget** — bounded re-dispatch loops; the verify/debugger grind (~$7-8/cycle
  post-critical) stops because gates trip immediately.
- **Main-branch correctness** — pending critical/urgent escalations gate MERGE entry.
- **All MCP users of `resolve_issue`** — steward, escalation-watcher, escalation-watcher-auto,
  /unblock, recon-escalation-watcher: one semantics table, enforced by signature.

User-observable surface: see §Boundary-test sketch — each row is a signal.

## Background

3-tier ladder: L0 agent→steward, L1 steward→auto-watcher, L2 auto-watcher→human. Born-at-L2 =
severity ∈ {critical, urgent} stamped `level=2` at the MCP chokepoint (server.py:156-157,
`BORN_AT_L2_SEVERITIES` = models.py:41), bypasses dedupe. `queue.resolve()` cascades L2→members
(`resolved_by='l2-cascade:<id>'`). Harness re-pend flips gate on `status=='resolved'` AND
`level==1` (cascade flip harness.py:4168-4184; Fix#1a orphan flip :4186-4221). Fix#1b backstop:
≤900s sweep re-files a `task_failure` L1 for blocked tasks with no pending escalation and no
active workflow (harness.py:1786-1858) — **re-files, never flips**, which protects the deliberate
`/unblock` blocked-park (open L1 → skipped by the pending-escalation check). All anchors
re-verified against the working tree 2026-06-04 (31/31 PASS).

Verified defects this PRD resolves (Brief 3, priority order):

1. **Born-at-L2 never gates** — all three gates (workflow.py:3093-3096, :3442-3445, :1186-1189)
   filter `severity=='blocking' and level==0`; a critical/urgent level-2 record fails both
   conjuncts; the merge worker reads no escalations. Dormant (only the harness watcher-outage
   sentinel, harness.py:2212-2234, files born-at-L2 today) — activates silently on any prompt
   change.
2. **`terminate=true` is broken everywhere** — it doesn't terminate anything on *any* path:
   the live L0 wait (workflow.py:4760-4767) resumes with empty guidance on dismissal; off-live,
   dismissal flips nothing and Fix#1b circles the task back to the human who dismissed it.
   The auto-watcher's `scope_violation`/`dependency_discovered` flows use `terminate=true`
   believing "task will be rescheduled" — so every routine scope expansion / dep discovery
   strands its task into the circular loop today.
3. **No cross-incarnation re-block guard** — all thrash counters are failure-mode-specific;
   cascade-re-pended tasks can loop blocked→L2→pending→blocked unbounded.
4. **Flip durability** — the blocked→pending flip is a scheduled coroutine fired after the
   durable resolve (silent drop when loop closed, harness.py:4088-4093); a crash in the window
   strands the task until a full second escalation round-trip.
5. **Small gates** — Fix#1b's `_workflow_cancel_at` check is membership-not-age (harness.py:1809
   vs the 30s grace at :1557-1565); `_escalation_events` registers deep in the slot coroutine
   (:2356) not at dispatch (:892), a sub-second Fix#1a race.
6. **Doc truth** — `resolve_issue` docstring (server.py:338-339) overclaims text delivery; the L2
   skill's `review_suggestions` handler (escalation-watcher/SKILL.md:257-368) is unreachable dead
   prose (live path = curator tickets, workflow.py:5923-6037; fallback = steward-consumed L0).

## Resolved design decisions

**D1 — `resolve_issue` action enum (replaces `terminate`).**
`resolve_issue(escalation_id, resolution, action='resume'|'restart'|'park'|'abandon'|'close_only',
resolved_by)`. The `terminate` parameter is **removed**; passing it raises a hard,
self-explaining error naming the five actions. Default `action='resume'` (the overwhelmingly
common, least-destructive intent). Rationale: five distinct caller intents existed with two
encodings, both lying; the enum names each intent and is state-independent — the cure for
"chronically unclear".

**D2 — park = `deferred`; abandon = `cancelled`.** No new "parked" marker: `deferred` is already
invisible to the scheduler and to the stranded sweep (blocked-only predicate), visible to the
operator in task lists, and pinned by `test_criterion_3_deferred_task_not_flipped`. Un-park =
explicit `set_task_status('pending'|'cancelled')`. This also avoids the
set_task_status-replaces-metadata hazard a metadata marker would carry.

**D3 — agent-filed critical/urgent: restrict at chokepoint.** The server submit path downgrades
task-attached critical/urgent from agent roles to `blocking` (logged, summary annotated).
Born-at-L2 stays reserved for harness sentinels and operator tools; agents reach humans via the
existing ladder (steward → auto-watcher → `promote_to_l2`). Protects the human-interrupt channel
from severity inflation ("escalations are exceptional").

**D4 — gates get the disjunct anyway (defense in depth).** All three workflow gates become
`(severity=='blocking' and level==0) or severity in BORN_AT_L2_SEVERITIES or level >= 2`.
Accepted consequence: a pending critical/urgent from a *prior incarnation* sinks a fresh run —
intended stop-the-line semantics (re-dispatch with one pending is unreachable except by operator
override). The stale-L1-must-not-sink-runs property (workflow.py:3088-3092 comment) is preserved
for plain blocking L1s.

**D5 — stranded recovery via auto-resolvable category.** The Fix#1b sweep files a dedicated
`stranded_blocked` L1 (instead of `task_failure`); the auto-watcher auto-resolves it with
`action='resume'` → the existing Fix#1a orphan flip re-pends. The sweep keeps its
**re-file-never-flip** discipline (preserves the `/unblock` blocked-park protection) and runs at
**startup** plus every `stranded_reconcile_interval_secs` (900s) — startup pass is the crash-window
replay. Humans only see genuinely re-failed tasks.

**D6 — signature-aware re-block guard, threshold 3.** Persisted per-task counter + block-reason
signature in task metadata, following the `_check_*_thrash` helper shape (reset to **1**, not 0,
on signature change). Incremented at **every** blocked→pending flip (cascade, orphan, sweep-driven).
At threshold the flip is withheld and a **born-at-L2** (`urgent`, category `task_failure`,
summary `persistent re-block: <n> redispatches, signature <sig>`) goes to a human. Counter write
ordering errs toward over-counting (see contract C5).

**D7 — memberless born-at-L2 `resume` re-pends.** The orphan-flip gate extends from `level==1` to
task-attached `level>=1`, so resolving a born-at-L2 with `action='resume'` flips its blocked task.
(Today it flips nothing — the task strands.)

**D8 — live-workflow born-at-L2 trip → ESCALATED wait.** Same behavior as blocking L0s: the
workflow waits in `_wait_for_resolution`-style ESCALATED state and inherits the existing
`_workflow_cancel_at` timeout-to-blocked machinery. Resolution text reaches the agent on this
live path (and only this path — unchanged for cascades; the L2 skill already documents writing
durable guidance to memory instead).

**D9 — kills route through existing soft-cancel machinery.** `restart`/`park`/`abandon` on a task
with a live workflow use the `release_workflow` substrate (`_workflow_cancel_events`,
`hard_cancel_workflow`, `_workflow_slot_tasks`). Invariant: action-driven teardown **suppresses
the workflow's own terminal status write** — the action's status wins (contract C3).

**D10 — legacy in-process mapping.** Records resolved via `queue.resolve()` directly (steward
:717, harness internals) carry `resolution_action=None`; the harness callback maps
`dismiss=True → close_only`, `dismiss=False → resume`. Never destructive — only the MCP layer can
express `restart`/`park`/`abandon`. No steward edit required.

**D11 — small gates folded in.** Fix#1b's `_workflow_cancel_at` check becomes age-based (reuse
the `_RECONCILE_CANCEL_GRACE_S`-style window); `_escalation_events` registration moves to the
dispatch point (harness.py:892 region) closing the sub-second Fix#1a race.

## Pre-conditions for activating

None hard. PRD-1's watcher-rotation fixes improve `stranded_blocked` *pickup latency* (L1 drain
cadence) but not correctness — the category lands and functions regardless. Live state at audit
time: queue drained, 0 blocked tasks — clean landing window.

## Cross-PRD relationship (G4)

Ownership per the static register in `plans/escalation-flow-2026-06-04-prd-briefs.md`; newly
discovered seams logged append-only in `plans/escalation-flow-gaps-prd3.md` (5 entries at
authoring time — read it before implementing).

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| PRD-1 (watcher/queue ops) | this PRD **reads** queue.py APIs (`submit`, `resolve`, `get_by_task`, parent-L2 lookup) | `escalation/queue.py` | PRD-1 (PRD-3 reads only — no queue.py edits anywhere in this decomposition) | wired |
| PRD-1 | produces | `Escalation.resolution_action` additive field must pass through PRD-1's queue/sweep/archive serialization | this-prd (field), PRD-1 (pass-through) | logged (gaps #1) |
| PRD-1 | shared file, disjoint regions | `server.py`: PRD-3 = resolve_issue handler + submit-path downgrade + CATEGORIES; PRD-1 = startup sweep wiring | split per gaps #2 | logged |
| PRD-1 | produces | `stranded_blocked` routing-table entry in escalation-watcher-auto/SKILL.md (Per-Category table = PRD-3; Main Loop = PRD-1) | this-prd | queued (task θ) |
| PRD-2 (B3 hardening) | consumes | `resolve_issue(action=...)` signature — B3/AFK-shift-2 snippets using `terminate=` break on removal | this-prd (server change + companion sweep), PRD-2 (new snippets use `action=`) | logged (gaps #3) |
| PRD-2 | produces | B3-applicability pointer form in PRD-3-owned `task_failure`/`review_issues` handlers (PRD-2 makes B3 posture-configurable; our prose must not restate posture) | this-prd (task η) | logged (gaps #6) |
| recon queue (8103, no PRD) | consumes | same server package; recon findings get `resume`/`close_only` disposition semantics only | this-prd (companion sweep covers recon-watcher skill prose) | logged (gaps #5) |
| merge_queue.py verify path | consumes (invariant) | post-merge re-rebase+verify remains the correctness backstop; untouched | nobody (per register) | wired |

## Contract (B+H §1)

### C1 — `resolve_issue` semantics table (the single source of truth; docstring reproduces it)

| `action` | Record disposition | Live workflow | Task status effect | Intent |
|---|---|---|---|---|
| `resume` (default) | `resolved` | resumes; resolution text injected (L0 live path) | `blocked` → `pending` (any task-attached level ≥ 1, incl. memberless born-at-L2); otherwise no-op | "Here's the answer — continue." |
| `restart` | `resolved` | killed (soft-cancel → grace → hard) | → `pending` (from `in-progress` or `blocked`) | "This run is off-course — re-run fresh." |
| `park` | `dismissed` | killed | → `deferred` (from any non-terminal status) | "Stop; human decides later; machine must not touch." |
| `abandon` | `dismissed` | killed | → `cancelled` | "Never run again." |
| `close_only` | `dismissed` | untouched | none | "Record is noise/duplicate — change nothing." |

- Terminal task statuses (`done`, `cancelled`) are never overwritten by any action (existing
  status-recheck discipline, harness.py:4244-4249).
- `terminate=` (any value) → hard error: `"'terminate' was removed; state your intent:
  action='resume'|'restart'|'park'|'abandon'|'close_only' — see resolve_issue docstring."`
- **L2 cluster cascade**: the action applies to the L2 and uniformly to every member task.
  Mechanism: `queue.resolve()` cascades members with the same dismiss flag (unchanged, PRD-1's
  file); the harness member callback resolves the **parent's** action by parsing
  `resolved_by='l2-cascade:<id>'` and reading the parent record via the queue **read** API.
- `resolution_action=None` (in-process legacy callers): `dismiss=True → close_only`,
  `dismiss=False → resume`.

### C2 — `Escalation.resolution_action` field (models.py, additive)

`resolution_action: str | None = None`, set by the server before `queue.resolve()`, persisted in
the record JSON (rides through queue/sweep/archive untouched). Validation at the server layer
only.

### C3 — harness action dispatch (`_on_escalation_resolved` region)

Ordering invariants:
1. **Status write precedes kill** for `restart`/`park`/`abandon` (scheduler must not re-dispatch
   a parked/abandoned task in the kill window; for `restart`, re-dispatch after `pending` is the
   point).
2. **Teardown suppression**: a workflow killed by an action must not write its own terminal
   status (`blocked`) over the action's status. Mechanism: stamp the task id in an
   action-teardown set checked by the workflow's terminal-status writer (analogous to the
   existing `_workflow_cancel_at` discipline).
3. Flip gate generalization: cascade flip + orphan flip accept task-attached `level >= 1` with
   disposition `resolved` (D7); cascade members inherit the parent action (C1).
4. Status writes go through fused-memory `set_task_status` (interceptor/recon events fire);
   metadata writes use `update_task(append=true)` (clobber hazard).

### C4 — chokepoint downgrade (server submit path)

Task-attached escalations from **agent roles** with severity ∈ `BORN_AT_L2_SEVERITIES` are
downgraded to `blocking`, logged at WARNING, summary prefixed `[downgraded:critical]`/
`[downgraded:urgent]`. Harness sentinels (`agent_role` in the harness-internal allowlist, e.g.
`harness-*`) and operator tools keep born-at-L2. roles.py documents the rule for agents.

### C5 — re-block guard (metadata schema)

`metadata.reblock_guard = {count: int, signature: str}` on the task. Signature = the blocking
escalation's `category` + normalized first 120 chars of its summary (tactical refinement allowed
at implementation; see Open questions). At every blocked→pending flip: same signature →
`count += 1`; different → `count = 1, signature = new`. Counter write (append=true) **before**
the status flip — a crash between over-counts, erring toward earlier human escalation, never
under-counts. At `count >= 3`: withhold the flip, file born-at-L2 per D6. Human resets by
clearing `metadata.reblock_guard` explicitly.

### C6 — stranded sweep (`stranded_blocked`)

Predicate (unchanged except the age fix): `status=='blocked'` AND no active workflow AND
`_workflow_cancel_at[tid]` absent-or-older-than-grace AND no pending escalation. Action: file L1
`category='stranded_blocked'`, `severity='blocking'`, `agent_role='harness-stranded-blocked-reaper'`
(re-file-never-flip preserved). Runs once at startup (after `_reconcile_stranded_in_progress`,
harness.py:743 region) + every `stranded_reconcile_interval_secs`. Auto-watcher routing:
`stranded_blocked` → verify the predicate still holds (task still blocked, no pending sibling,
no active workflow) → `resolve_issue(action='resume')` → Fix#1a flips (guard C5 applies).
Self-dedupe via the pending-escalation check (existing).

### C7 — workflow gates (three sites)

`workflow.py:3093, :3442, :1186` predicate becomes:
`(e.severity == 'blocking' and e.level == 0) or e.severity in BORN_AT_L2_SEVERITIES or e.level >= 2`.
Trip behavior unchanged (`return WorkflowOutcome.ESCALATED`). Import `BORN_AT_L2_SEVERITIES`
from `escalation.models` (constant exists; orchestrator already imports `escalation.models` in
the sweep).

## Boundary-test sketch (B+H §2)

Each row faces both sides of the escalation↔orchestrator seam; preconditions are synthetic queue
records + task rows, postconditions assert through the product's own read paths (task status via
fused-memory, queue state via escalation APIs, workflow outcomes via harness tests).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Pending level-2 critical gates MERGE | task in-progress at MERGE entry; pending task-attached level-2 record | workflow returns ESCALATED; no merge attempted |
| B2 | Pending critical gates post-implementer (budget grind stops) | execute-loop iteration; pending critical | ESCALATED before next verify/debugger iteration |
| B3 | Stale plain L1 still does NOT sink a fresh run | re-dispatched task; pending `blocking` level-1 from prior incarnation | gates pass (D4 preserves the L0-only filter for plain blocking) |
| B4 | `park` on L2 cluster | L2 with 2 member L1s; both member tasks blocked | both tasks `deferred`; records `dismissed`; no `stranded_blocked` filed across 2 sweep periods; no re-ask |
| B5 | `abandon` memberless born-at-L2 | blocked task; task-attached level-2 record | task `cancelled`; sweep never touches it |
| B6 | `restart` on live workflow | in-progress task, workflow mid-execute | workflow killed via soft-cancel; task `pending`; teardown did NOT write `blocked` (C3.2) |
| B7 | `resume` memberless born-at-L2 (D7) | blocked task; level-2 record resolved `action='resume'` | task `pending` (today: strands) |
| B8 | `close_only` on healthy in-progress task | live workflow, non-gating escalation | record `dismissed`; workflow untouched; task status unchanged |
| B9 | `terminate=` hard error | any record | error names the five actions; record unchanged |
| B10 | Restart-window replay (the missing regression test) | on-disk: resolved L1, task `blocked`, no pending esc, no workflow — harness started fresh | startup sweep files `stranded_blocked`; auto-resolve(`resume`) flips to `pending`; zero human round-trips |
| B11 | Re-block guard trips | task flipped blocked→pending 3× with same signature | 4th flip withheld; born-at-L2 `urgent` filed; counter survives the status cycles (C3.4 metadata discipline) |
| B12 | Signature change resets | 2 same-signature flips, then different block reason | `count == 1`, new signature; flip proceeds |
| B13 | Agent-filed critical downgraded | submit from agent role, severity `critical` | record `blocking` level-appropriate; summary prefixed; WARNING logged; harness sentinel still born-at-L2 |
| B14 | Legacy in-process dismiss is non-destructive (D10) | steward `queue.resolve(dismiss=True)` on L0, task in-progress | mapped `close_only`; task untouched |
| B15 | `/unblock` blocked-park still protected | blocked task with OPEN L1 | sweep skips (pending-escalation check) — unchanged |

## Decomposition plan

Pre-split per package (cross-package tasks exceed the architect budget). Greek labels; IDs at
decompose time. Phase 1 = foundation, Phase 2 = vertical slice + gates, Phase 3 = prose +
companion corrections. ζ is the integration gate (G2 escape-hatch pattern: α1/α2 are
intermediates).

| # | Task (≤70c) | Package | Observable signal | Prereqs |
|---|---|---|---|---|
| α1 | Add resolve_issue action enum + resolution_action field + docstring | escalation | B9: `terminate=` → hard error naming actions; `action='park'` persists `resolution_action='park'` in record JSON; docstring reproduces C1 table | — |
| α2 | Chokepoint severity downgrade + stranded_blocked category | escalation | B13: agent-filed critical lands as `blocking` w/ prefixed summary + WARNING; `stranded_blocked` accepted by CATEGORIES validation | — |
| β | Harness action dispatch + level≥1 flip + teardown suppression | orchestrator | B4/B5/B6/B7/B8/B14: park→deferred (cluster-wide), abandon→cancelled, restart→pending w/o teardown clobber, resume flips memberless born-at-L2, close_only/legacy no-ops | α1 |
| γ | Extend three workflow gates with born-at-L2 disjunct | orchestrator | B1/B2/B3: pending level-2 critical → ESCALATED at MERGE entry + post-implementer; stale plain L1 still passes | — |
| δ | Signature-aware re-block guard (threshold 3) | orchestrator | B11/B12: 4th same-signature flip withheld + born-at-L2 urgent filed; signature change resets to 1; counter survives status cycle | β |
| ε | Stranded sweep → stranded_blocked + startup pass + age-gate + event-at-dispatch | orchestrator | B10 (restart-window regression test) + B15: startup sweep files `stranded_blocked`, auto-resolution path flips, /unblock park untouched; `_escalation_events` registered at dispatch | α2 |
| ζ | Integration gate: boundary-test suite B1–B15 green | orchestrator | full §Boundary-test table runs in CI (`pytest orchestrator/tests/test_repend_state_machine.py`) | β, γ, δ, ε |
| η | Rewrite L2 escalation-watcher skill (owned sections) | skills | SKILL.md contains the C1 table; AFK shift 1 uses `action='park'`; `review_suggestions` handler replaced by one-line routing note; `task_failure`/`review_issues` handlers defer B3 applicability via pointer form (gaps: PRD-2 entry 2); watcher instant-fire-at-launch note added (gaps: PRD-1 entry 3); zero `terminate=` snippets in PRD-3-owned sections | α1, β |
| θ | Auto-watcher routing table: action migration + stranded_blocked | skills | scope_violation/dependency_discovered use `action='resume'`; `stranded_blocked` entry auto-resolves with `resume`; zero `terminate=` in the routing table | α1, β, ε |
| ι | roles.py severity-policy doc + orchestrator-tree caller migration | orchestrator | roles prompt states the downgrade rule; `grep -r 'terminate=' orchestrator/src` → no resolve_issue callers | α2 |
| κ | Companion sweep: legacy terminate= snippets in out-of-register skills | skills | `grep -rl 'terminate=' skills/unblock skills/recon-escalation-watcher` → empty; PRD-1/2-owned sections checked, residue logged to gaps file | α1 |

Capability manifest committed beside this PRD at decompose time
(`plans/escalation-repend-state-machine.capability-manifest.md`).

## Out of scope

- PRD-1 territory: watcher.py (rotation timeout, drain-before-up), queue.py/sweep.py/archive.py
  (hygiene, add_members race, ntfy mapping), server startup sweep wiring.
- PRD-2 territory: dry_run_unblock.py, unblock-low-risk skill, B3 subsection + AFK shift 2.
- Delivering resolution text to *fresh* (re-dispatched) workflows — cascade re-pend stays
  status-only; durable-guidance-via-memory remains the documented pattern.
- merge_queue.py verify path (existing invariant, register: nobody).
- Dedupe-folded child-spin issue (audit backlog; separate concern).

## Open questions (tactical)

1. **Signature derivation refinement.** C5 starts with category + normalized summary prefix; if
   summaries prove too volatile (timestamps, ids), switch to category + escalating `agent_role` +
   file-set hash. Decide during δ.
2. **Sentinel allowlist shape for C4.** `harness-*` prefix vs explicit role list. Decide during α2.
3. **Near-shutdown one-off sweep trigger.** The startup pass may make it redundant; add only if
   B10-style strands appear in soak. Decide after ζ lands.
4. **`stranded_blocked` pickup latency.** Bounded by PRD-1's rotation fixes; if PRD-1 slips,
   consider a direct inotify nudge. Observe, don't pre-build.
