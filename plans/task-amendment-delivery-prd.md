# PRD: Task-amendment delivery — make a post-plan amendment reach the work, and make descoping legible

**Status:** active · authored 2026-08-12 · approach **B + H** (contract + two-way boundary tests)
**Author session:** sweep-df-3334614 (investigation report:
`/home/leo/.claude/spawn-briefs/df-amendment-drop-sweep-RESULT.md`)

---

## 1. Goal

When a task's `description`/`details` are amended after its plan was authored, the
amendment must reach the work — today it reaches nothing, silently, and the task
still closes `done`. This PRD makes an amendment **trigger a re-plan**, makes the
re-plan **unable to silently destroy completed work**, gives the architect a
**sanctioned way to descope work that genuinely is no longer in scope**, and makes
the write itself **say so at the boundary**.

User-observable: amend a planned task's text, redispatch it, and the implementer's
briefing reflects the amendment. Amend it in a way that descopes done work, and the
plan records *which* commits were dropped and *what happened to them* — instead of a
hand-edited `plan.json` and four follow-up escalations.

---

## 2. Background — measured, not inferred

### 2.1 The delivery defect

`BriefingAssembler._format_task` (`agents/briefing.py:1478-1494`) is the **only**
renderer of `description`/`details`, and it renders them whole — no truncation, no
cap. It has **six** callers: architect (`:309`), revalidation (`:398`),
plan-completion (`:493`), plan-tightening (`:577`), simple-task (`:648`),
steward-initial (`:1181`). Six of the fourteen prompt builders.

Measured over **4,583 real dispatch briefings** (transcript corpus
`data/orchestrator/agent-transcripts/`, 2026-07-18 → 08-10; every dispatch's first
jsonl record is `{"operation":"enqueue","timestamp":…,"content":<the briefing>}`):

| briefing family | dispatches | carries task text |
|---|---:|---:|
| architect / plan-authoring / simple-task | 660 | **100%** |
| plan-revalidation | 164 | **100%** |
| steward-initial | 81 | **100%** |
| reviewer (`# Code Diff to Review`) | 1,543 | 0% |
| implementer (`# Plan Overview`) | 835 | 0% |
| amender (`# Amendment Pass`) | 442 | 0% |
| debugger (`# Task Context`) | 221 | 0% |
| resume (`# Resuming After Escalation`) | 130 | 0% |
| fixer (`# Review Feedback`) | 128 | 0% |

**916 / 4,583 = 20.0%.** The split is 100%/0%, not a gradient, and it matches the
caller list exactly. `build_implementer_prompt(self, plan: dict, …)`
(`briefing.py:709`) does not take the task as a parameter at all.

So an amendment made after the plan was authored is invisible **not only to a running
agent but to every later fresh dispatch**, because a fresh dispatch resumes from the
plan. Task **3199**: 38 dispatches over 5 days; only the 1st and 3rd carried task
text, both *before* the 2026-08-02 amendment; none of the 35 later ones did; merged
08-06 without the amendment's two probes.

**Outcome rate** (569 `done` tasks with ≥1 text-carrying briefing): 21 (3.7%) carry
text no briefing ever delivered; ~10 bore a deliverable; **6 cost something real** —
hard drops **3199** and **2852** (the latter still untracked), partials **3018**,
**3512**, **2951**, and **3832** whose dropped correction leaves a false claim live in
main today (`test_offline_lane_integration.py:269`).

Delivery is **incidental, not designed**: **59** tasks had an amendment delivered
because a later re-plan happened; **21** did not. Nothing causes that luck and nothing
detects its absence. **27 open tasks carry undelivered amendments right now, and
27/27 already hold a plan on disk.**

### 2.2 Why the trigger is the right lever (and why pushing text downstream is not)

The EXECUTE/JUDGE spine is **deliberately** plan-only: `roles.py:732` — *"The plan
structure is IMMUTABLE after creation"*; `roles.py:761/765/808` — *"Follow the plan
exactly … If you encounter an unexpected issue the plan doesn't account for, note it
and stop. Do NOT modify the plan."*; the judge and debugger carry enumerated
three-input contracts (`roles.py:1038-1049`, `:830-833`) corroborated by
`ARCHITECTURE.md:485`. In five months, all five `_format_task` additions went to
architect-family or steward builders — never to an EXECUTE-lane builder.

**Design principle (restated, not invented here):** *the architect is the sole
translator of the task record into a plan; from PLAN onward the plan is the executable
contract.* Rendering task text into the implementer/amender/debugger/judge would show
an agent required work it is explicitly barred from planning. **The sanctioned remedy
for stale task intent in the EXECUTE lane is a re-plan.** This PRD therefore adds no
task text to any EXECUTE-lane prompt.

C-A1 is not an obstacle: it scopes to `metadata.files` and ships
`test_include_files_false_keeps_description` (`test_briefing.py:135-141`)
*guaranteeing* the description survives.

### 2.3 The trigger's actual defect

`TaskWorkflow._plan()` (`workflow.py:4289-4762`) decides revalidate-vs-architect on:
plan exists ∧ `steps` ∧ `_session_id` ∧ `_old_plan_base`, then computes files changed
on `main` between the plan base and current main, overlapped with `plan.files`
(`:4359-4364`). **The task text plays no part.** And "Lever B" short-circuits even
that pass when `not overlap ∧ _can_skip_revalidation(plan)` (`:4377-4388`), removing
the last opportunity to re-read the task.

`_can_skip_revalidation` (`:4764-4830`) already receives the plan and can read
`self.task`; it has four conjuncts and **reads no task text**. That is the seam.

`updated_at` is **not** a usable trigger — `sqlite_task_backend.py:2761-2762`:
*"updated_at always advances, even on a no-op write."* Only a content fingerprint
works. `candidate_key` is also blind: recomputed only `if title is not None or
metadata is not None` (`:2719-2721`), so a description-only amendment leaves it
byte-identical.

### 2.4 Who amends text (all four machine writers)

`Scheduler.update_task(task_id, metadata, *, append, metadata_mode)`
(`scheduler.py:3794-3801`) **cannot write text** — verified across ~30 orchestrator
call sites, all metadata-only. But the orchestrator is not the only machine. Of
57,128 journalled `update_task` ops, **1,454 carried a text field** (details 766,
description 737, title 208) — **100% `source='mcp_tool'`**. And the largest writer is
**invisible to that journal**: the TaskCurator's `combine` calls `tm.update_task`
directly, bypassing `TaskInterceptor`.

### 2.5 The curator combine race (measured; this PRD's cheapest win)

The pending-only combine constraint **is implemented at selection and not re-checked
at execution**:

- selection — `task_curator.py:2793`: `combine_eligible=(status == 'pending')`,
  enforced at decision-parse (`:2949-2960`), downgrading a non-eligible target to
  `create`. Sound.
- execution — `task_interceptor.py:2143-2144`: the entire status predicate is
  `if target_status in {'done', 'cancelled'}`. Nothing blocks `in-progress`.

`_execute_combine` **does** re-read the live target (`:2126`) — this is a re-read with
the wrong predicate, not a missing one. The window spans an LLM round trip with a
**180 s** timeout, and the two actors are unsynchronised by design:
`_curator_lock`'s docstring (`task_interceptor.py:2298-2300`) — *"Short writes
(set_task_status etc.) never take this lock"* — and dispatch **is** a
`set_task_status`.

Measured over `data/combine_audit.jsonl` (640 records, independently recounted this
session): **511 pending, 129 `in-progress` (20.2%)**; of the 129, **128 rewrote the
description**, 127 the title; **23 carry an LLM justification asserting the target
"is pending"** in the same record whose `old.status` reads `in-progress`. All 36
dark_factory cases landed after `task_started` and before the architect's plan
`invocation_end` — mid-planning, after the briefing was assembled; 33/36 within the
180 s timeout. Task **2951**: combine landed **9.3 s** after dispatch.

The curator's own contract prompt (`task_curator.py:435-439`) names the failure mode
verbatim: *"Combining into a non-pending task would silently drop the candidate's
work because the workflow has already moved past planning for that task."* The abort
path is already safe — `# combine failed → fall through to create`
(`task_interceptor.py:3487`) — so refusing files the candidate as its own task. No
test pins the current behaviour.

### 2.6 Why a done-step count invariant is the wrong invariant

A re-plan can destroy completed work: `build_revalidation_prompt` offers option **(c)
"Plan is invalid: call `create_plan(...)`"** (`briefing.py:468`), and
`mcp/plan_tools.py:991-1001` writes `'steps': []`, overwriting plan.json wholesale
("*this overwrites plan.json wholesale*"). Step 5 of the same prompt says *"Do NOT
remove or replace steps with status done"* — enforced by nothing. Compounding it,
`committed_work` is deliberately excluded from branch B (`workflow.py:4413-4416`) on
the premise that B "already carries done-step semantics" — which a `create_plan`-shaped
revalidation falsifies.

**But a "done count must not decrease" invariant is keyed on the wrong quantity, and
fires backwards on both real cases:**

- **Task 2971** — the corpus's only fully-worked, human-ratified descope. Plan went
  from 2 steps (`step-1` test **done** @`309c22ab6f`, `step-2` impl pending) to 1
  step (`step-1` impl **done** @`c0609049a7`). Done count **1 → 1, no decrease** — yet
  a done step *was* destroyed and its commit deliberately orphaned (`309c22ab6f` is
  **not on main**; survives only on archived branch `task/2971-20260723T095312Z`).
  A count invariant **waves this through**.
- **Task 3143** — a revalidation called `create_plan()` and took **19 done → 0**
  (`esc-3143-4`). All 19 recorded SHAs were unreachable: phantom-done recovery, where
  wiping was correct. A count invariant **blocks this**.

The discriminator is not the count but **the set of `(step_id, commit)` pairs and
whether each dropped commit's disposition was recorded**.

### 2.7 SHA "validity" cannot be a reachability check

Task **3157**'s addendum (2026-08-05, from the esc-3404-5 forensic ruling) measured
exactly this:

- A bare `merge-base --is-ancestor` check fires on **185/200** live task branches with
  recorded done-step commits (**92.5%**); restricted to the 49 non-terminal branches a
  gate would inspect, **39/49 (79.6%)**. Almost every hit is ordinary merge-lane rebase
  churn, not loss.
- **991 of 1,973 (50.2%)** recorded done-step SHAs **no longer exist as git objects**
  (gc pruned the pre-rebase originals). "Verify the recorded SHA is reachable" is
  unimplementable as literally worded for half the corpus. **"SHA unresolvable" is an
  expected state, not a defect signal.**
- Phantom-done is genuinely **rare**: of the same 1,973 done steps, **zero** confirmed
  "recorded done, work nowhere"; four candidates all hand-falsified.
- **The machinery already exists and is sound; the gap is phase placement.**
  `TaskWorkflow._reconcile_done_step_commits` (`workflow.py:8440`) does
  WIP-filename-subset → `GitOps.find_equivalent_commit` patch-id/subject matching
  (task 2762) → non-blocking `severity='info'` escalation. It is called from
  **exactly one site** (`workflow.py:7976`, inside the implement loop). `_plan()` and
  `_apply_revalidation_skip` **never call it**.

So the correct check is **content/patch-based via the existing reconciler**, called
from the revalidation path — *not* a second, parallel reachability mechanism (INV-5).

### 2.8 A descope folk protocol already exists — codify, don't invent

Six worked precedents — **2971** (orphan-and-cherry-pick), **2540** (ride-along,
corrected in place; its escalation names *"a re-scope→re-plan gap in the pipeline"*),
**3225 / 3507** (hand-authored in-branch revert; `934ce79426` **is an ancestor of
main**, verified), **3446→3591** and **2169** (scope split to a follow-up task),
**3143** (create_plan wipe) — share five moves: record the descope in the description;
stamp a typed metadata key naming the authorising human/escalation; re-file or
known-issue the dropped half; dispose of the commit (orphan / revert / ride-along);
**fix `plan.json` afterwards, by hand, via follow-up escalations** (`esc-2971-6/-10/-11/-12`,
`esc-2540-18/-19`).

2971's operator invented the schema by hand — `.worktrees/.task-meta/2971/plan.json`
carries `_rescoped_at`, `_rescoped_by`, `_rescoped_note`, verified on disk, the note
reading *"…step-1 test done @orphaned 309c22ab6f…"*. That is a de-facto schema
proposal written by the only person who has ever done this, and it is this PRD's
starting point.

Also visible in that file: `_created_at` (14:22:41) **post-dates** `_rescoped_at`
(10:07:27) — the unconditional provenance re-stamp wrote straight over the manual
rescope. Any new plan field must be stamped with the same care (see §9 C4).

All three dispositions are legitimate in different cases, so the design **records
which was chosen** rather than picking one. Performing reverts is out of scope (§8).

---

## 3. Sketch of approach — six arms

1. **α — the trigger.** A task-text fingerprint stamped into the plan at authoring and
   mirrored onto `task.metadata`; a fifth conjunct in `_can_skip_revalidation`; a
   green-tier config flag. Fingerprint mismatch ⇒ the dispatch takes the revalidation
   path, which already renders the current task text.
2. **β — done-step integrity + sanctioned descope.** Call the existing reconciler from
   the revalidation path before trusting any done step; promote 2971's `_rescoped_*`
   fields plus a per-step `disposition` to schema; key the invariant on
   `(step_id, commit)` + disposition, not on count.
3. **γ — the converse merge gate**, at *submission* time: every commit on the task
   branch is owned by a plan step or named in a disposition.
4. **δ — write-boundary advisory + event** in fused-memory, so an amendment to a
   planned task says so at the moment it is written.
5. **ε — close the curator combine race**: make the execution-time predicate match the
   selection-time one.
6. **ζ — one-off migration sweep**: stamp every live plan, so no permanent
   legacy-tolerance branch is needed.

---

## 4. Resolved design decisions

**D1 — Trigger on a content fingerprint, never `updated_at`.** `updated_at` advances
on every no-op write (§2.3). Basis is `description + '\x00' + details` — **both**,
because `_format_task` renders both (`briefing.py:1485-1488`).

**D2 — Do NOT casefold; do NOT reuse recon's normalizer.** Recon's
`_normalize_content_description` (`flag_dedup.py:1375-1382`) casefolds because its
predicate is finding-dedup, where case is noise. Ours asks *"did the author change the
text"*, where a case change **is** a change. These are genuinely different contracts
that must **not** agree. The helper lands in `shared/` (a workspace dep of both
orchestrator and fused-memory — `orchestrator/pyproject.toml:21,26`,
`fused-memory/pyproject.toml:23,27`) with a docstring stating explicitly that it is
not the recon normalizer and must not be collapsed into it. This is the INV-5-correct
resolution: one home for *our* contract, and an explicit non-unification note so a
future tidy-up doesn't merge two predicates that must differ.

**D3 — Missing fingerprint is a fault, not a legacy case.** ζ stamps every live plan
during migration, so after transition a plan without `_task_text_fp` means a writer
skipped the stamp. The predicate therefore **declines the skip and emits a structured
fault** (INV-2/INV-4) rather than fail-open. No permanent transition-tolerance logic.

**D4 — ζ stamps a deliberately-invalid fingerprint on plans that need revising.** The
27 measured-exposed tasks (and any other live plan whose text has moved) get a sentinel
that cannot match, forcing exactly one re-plan on next dispatch. Plans whose text is
current get their true fingerprint and are undisturbed.

**D5 — The invariant is keyed on `(step_id, commit)` + disposition, not count.**
Refuted by 2971 (count 1→1, real destruction) and 3143 (19→0, legitimate wipe) —
§2.6. Every done step that disappears from the plan must be accounted for by a
recorded disposition.

**D6 — SHA integrity via the existing reconciler, not a new reachability check.**
Reachability measures 79.6–92.5% false positives and 50.2% of recorded SHAs are gc-pruned
(§2.7). β calls `_reconcile_done_step_commits` (`workflow.py:8440`) from the
revalidation path; "SHA unresolvable" is an expected state. **SHA uniqueness across
done steps** is asserted separately — it is cheap, sound, and closes the hole where a
count- or set-based check is gamed by duplicating a sha.

**D7 — Disposition vocabulary is `orphan | revert | ride-along`, recorded not
performed.** All three are attested legitimate (§2.8). v1 records which was chosen and
gates on the record existing; it does not perform reverts (§8).

**D8 — Split task 3157; β takes its done-step half.** 3157's two halves were welded by
a **curator combine** (`metadata.curator_action='combine'`), not by design — and that
weld gave the phantom-done half an inherited dependency on 3155→3154 (both `pending`,
inside `plans/task-meta-task-keying-prd.md`) which it does not need. 3157 narrows to
its `_base_commit` half; β takes the done-step-integrity half and depends only on
**3651** (`pending`, no deps, dispatchable today), which fixes the reconciler's
filename-subset collapse bug that β would otherwise inherit. A companion amendment to
3157 is performed at decompose (§7c).

**D9 — γ runs at merge SUBMISSION, not at verify or dequeue.** The existing
`_check_plan_files_touched_in_branch` already runs inside
`TaskWorkflow._submit_to_merge_queue` (`workflow.py:10003`; check at `:10090`) **before**
`register_and_enqueue_merge_request`, with an in-place remedy-and-recheck precedent at
`:10108` (`_try_narrow_plan`). γ goes at the same site so the workflow can address a
rejection immediately instead of waiting for the queue front.

**D10 — δ reads `task.metadata`, never the filesystem.** `plan.json` lives under
`<worktree_base>/.task-meta/…` and is invisible to fused-memory; a filesystem or git
probe on the fused-memory event loop is exactly the INV-8 incident of task 3778. α
therefore mirrors the fingerprint and plan-authored timestamp onto `task.metadata`
(writable by `Scheduler.update_task`, which is metadata-only), and δ compares against
that — a DB read already in hand.

**D11 — ε widens the execution predicate AND asserts no live claimant.** Status alone
has the same TOCTOU shape one level down: a target can be `pending` at the guard and
dispatched a moment later. `claimant_run_id is None` is cheap insurance on a race
measured at 20.2%.

**D12 — This PRD adds task text to no EXECUTE-lane prompt.** §2.2. The reviewer's
missing task context is a real defect but is owned elsewhere (§6).

**D13 — A β/γ refusal routes through the existing bounded failure path; no new hold is
invented.** β refuses *inside* the plan loop, so the first exit owner is the architect
itself: it receives the structured reason and may retry recording a disposition —
the same remedy-and-recheck shape `_try_narrow_plan` already uses at γ's site. If it
cannot resolve, the refusal routes to `_handle_no_plan_failure` (`workflow.py:4649`),
which already increments `consecutive_no_plan_failures` keyed by the main SHA inside
`metadata.retry_ledger` and **escalates to a human directly at ≥2** rather than
re-pending. That is a pre-existing owned-and-bounded hold with a storm escape, so β
and γ reuse it rather than adding a second failure path (INV-4, INV-5, INV-7 all
satisfied by reuse).

---

## 5. Pre-conditions for activating

| Prerequisite | State | Needed by |
|---|---|---|
| **3651** — `_reconcile_done_step_commits` filename-subset collapse fix | `pending`, no deps, dispatchable | β (else β inherits the collapse bug) |
| `_reconcile_done_step_commits` (`workflow.py:8440`) | **exists**, sound (tasks 2386, 2762 done) | β |
| `GitOps.find_equivalent_commit` (`git_ops.py:8855`) | **exists** | β |
| `shared` workspace dep of orchestrator + fused-memory | **verified** | α, δ |
| `UpdateTaskResult` (`task_backend_types.py:18-22`) | exists; **no advisory key** — needs `NotRequired` | δ |
| `should_reembed` text-field predicate (`task_interceptor.py:4467`) | **exists**, exactly the right chokepoint | δ |
| `RELOADABLE_FIELDS` (`config.py:4876`ff) | exists; revalidation flags are **absent** (red tier today) | α |
| `test_plan_revalidation_skip.py` (15 tests, 467 lines) | exists — the natural home | α, β |

**No novel substrate.** Every capability α–ζ invoke exists on main today; the only
additions are a new `shared/` helper, new plan/metadata keys, one `NotRequired`
TypedDict key, and one new event type. **G3: satisfied by inspection, no prerequisite
substrate task required** beyond 3651 (a correctness prerequisite, not a substrate one).

---

## 6. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task **3157** (`plans/task-meta-task-keying-prd.md` label δ) | contested — split | done-step SHA trust on the `_plan()` revalidation path | **this PRD (β)**; 3157 keeps `_base_commit` | resolved by D8; companion amendment §7c |
| task **3651** | consumes | `_reconcile_done_step_commits` correctness | 3651 | hard dependency of β |
| task **3659** (`docs/prds/memory-briefing-and-fusion.md` D7/D8) | adjacent | reviewer gains task-scoped **memory** sections | 3659 | untouched — D12 |
| task **3252** | adjacent | reviewer is memory-mute | 3252 | untouched — D12 |
| task **3160** | adjacent | prompt sites naming `.task/` | 3160 | no overlap: α/β touch `_plan()` + `_can_skip_revalidation`, not prompt paths |
| `plans/capability-delivered-checks-prd.md` | consumes | sidecar `delivered_check` schema | that PRD | this PRD's sidecar conforms |
| `docs/legibility/design-invariants.md` | consumes | INV slugs for the G7 walk | that doc | normative; walked in §12 |

**Reciprocal-ambiguity check.** 3157 is the only contested seam. Its own addendum
warns that its fix *landing alone* makes the destruction hazard worse — i.e. it
already asserts the two halves must be sequenced together. D8 resolves ownership by
splitting rather than by sequencing, removing the block instead of inheriting it.

---

## 7. Decomposition plan

### 7a. New leaves

| Label | Title | Modules | Observable signal | Prereqs |
|---|---|---|---|---|
| **α** | Task-text fingerprint gates plan-revalidation skip | `shared/`, `orchestrator/` | Amend a planned task's `description`; redispatch; the dispatch takes the **architect revalidation** path (event `plan_revalidated`) instead of emitting `phase_skipped{reason:'revalidation_skipped_no_overlap'}`, and the resulting briefing contains the amended text | — |
| **β** | Done-step integrity + sanctioned descope on the revalidation path | `orchestrator/` | A revalidation that drops a done step **without** a recorded disposition is refused with a structured reason; one that records `disposition` proceeds and the plan retains `_rescoped_at`/`_rescoped_by`/`_rescoped_note` + the dropped `(step_id, commit)`; a plan with duplicate done-step SHAs is refused | **3651**, α |
| **γ** | Converse merge gate at submission: branch commits ⊆ owned ∪ descoped | `orchestrator/` | Submitting a merge for a branch carrying a commit owned by no plan step and named in no disposition is refused **at `_submit_to_merge_queue`** (before enqueue), with the offending sha in the reason | β |
| **δ** | Write-boundary amendment advisory + event | `fused-memory/`, `shared/` | `update_task` changing `description`/`details` on a task whose `metadata.plan_text_fp` disagrees returns `amendment_advisory` in its result, and emits a structured `task_text_amended` event carrying the writer source | α |
| **ε** | Curator combine: execution-time predicate matches selection | `fused-memory/` | A combine whose live target is `in-progress` (or holds a claimant) is refused and the candidate is filed as its own task; `data/combine_audit.jsonl` records the refusal | — |
| **ζ** | One-off plan-fingerprint migration sweep | `scripts/` | Every live plan under `.worktrees/.task-meta/*/plan.json` carries `_task_text_fp`; the 27 measured-exposed tasks carry the forcing sentinel; the script reports counts per class | α |
| **η** | *(integration gate, B+H)* End-to-end amendment-delivery boundary tests | `orchestrator/`, `fused-memory/` | The §10 boundary-test table passes end-to-end, including the mid-EXECUTE non-refresh case and the fused-memory advisory case | α, β, γ, δ, ε |

α, ε, ζ are independently observable; β, γ, δ are observable in their own right and
also feed η. η is the B+H integration gate whose signal is §10.

### 7b. Amendments to existing records (performed at decompose, not new tasks)

- **3157** — narrow to its `_base_commit` half; move the done-step-SHA half to β with
  a note recording D8 and that the two halves were joined by a curator combine, not by
  design. Preserve its existing `delivered_checks` entry (`plan-carries-base-commit`),
  which already targets only the retained half.
- **2852** — file the dropped "option A" (state-walking poller reads `current()` each
  tick); currently owned by no task. *(Filed as a standalone follow-up, outside this
  PRD's arms — it is a merge-lifecycle fix, not an amendment-delivery one.)*

### 7c. Deliberately out of this batch

- Reverting descoped commits (D7, §8).
- Reviewer task-context (D12; 3659/3252).
- The Stage-2 `append=true` + `description=` clobber bug (§8).

---

## 8. Out of scope

- **Performing reverts.** All three dispositions are legitimate; v1 records the choice
  and gates on the record existing. A revert *affordance* is a separate design.
- **Task text in EXECUTE-lane prompts** (D12).
- **Reviewer task context** — owned by 3659/3252.
- **Recon Stage 2's `append=true` + `description=` clobber.** Observed on task 3095:
  Stage 2 believed it was appending while silently replacing the column, because
  `description` always overwrites regardless of `append`
  (`sqlite_task_backend.py:2633` vs `:2651`). A real data-loss bug, adjacent but
  independent — file separately.
- **Widening `revalidation_skip_enabled` / `max_revalidation_age_hours` to green
  tier.** Plausible in-scope bonus; deferred to keep α's blast radius minimal.
- **Retiring `metadata.memory_hints`** (task 3254) — unrelated channel.

---

## 9. Contract section (B + H)

### C1 — The fingerprint

```python
# shared/src/shared/task_text_fingerprint.py
#
# NOT the reconciliation normalizer.  fused_memory.reconciliation.flag_dedup
# ._normalize_content_description CASEFOLDS because its predicate is finding-dedup,
# where case is noise.  THIS predicate asks "did the author change the text", where a
# case change IS a change.  These two must NOT be unified (INV-5 applies to logic that
# must AGREE; these must DIFFER).

def task_text_fingerprint(description: str | None, details: str | None) -> str:
    """Stable fingerprint of a task's agent-visible prose."""
    basis = ' '.join((description or '').split()) + '\x00' + ' '.join((details or '').split())
    return 'tfp:' + hashlib.sha256(basis.encode('utf-8')).hexdigest()[:32]
```

- Prefix `tfp:` distinguishes it from recon's `fp:` at a glance.
- Whitespace-normalized (so reflowing prose is not an amendment); **not** casefolded.
- Basis is description **and** details, `\x00`-separated (unambiguous concatenation).

### C2 — Where it is stamped and read

| Site | File:function | Action |
|---|---|---|
| **Authoring** | `workflow.py` `_plan()` tail, at the `stamp_plan_provenance` call | stamp `plan['_task_text_fp']` |
| **Lever-B skip** | `workflow.py` `_apply_revalidation_skip` → `bump_revalidation_stamp` | re-stamp (unchanged by definition; keeps the field alive) |
| SIMPLE_TASK / eval / post-review replan | the other `stamp_plan_provenance` callers | stamp |
| **MUST NOT stamp** | the **per-implementer-iteration** provenance re-stamp | **`setdefault`/skip only** — refreshing here from the live task would make a mid-EXECUTE amendment silently self-clear, and the trigger would never fire. This is the same shape that overwrote 2971's manual `_rescoped_at` (§2.8) |
| **Mirror** | `Scheduler.update_task(metadata=…)` | mirror to `task.metadata.plan_text_fp` + `plan_authored_at` (metadata-only ⇒ legal for the orchestrator) |
| **Read (orchestrator)** | `workflow.py` `_can_skip_revalidation` | fifth conjunct |
| **Read (fused-memory)** | `task_interceptor.py` at the `should_reembed` predicate | δ's advisory |

`plan_tools._create_plan` writes a **fresh 8-key dict**, overwriting plan.json
wholesale — so the stamp must land **after** the architect returns, never before.

### C3 — The skip predicate

```
_can_skip_revalidation(plan) is True iff  <existing four conjuncts>
  AND '_task_text_fp' in plan                       # else: FAULT (D3)
  AND plan['_task_text_fp'] == task_text_fingerprint(task.description, task.details)
```

Missing key ⇒ decline the skip **and** emit a structured fault with a counter
(INV-2/INV-4), never a silent decline. Gated by a new green-tier bool in
`RELOADABLE_FIELDS`.

### C4 — Descope record (promoted from 2971's hand-rolled fields)

Plan document:

| Key | Type | Meaning |
|---|---|---|
| `_rescoped_at` | ISO-8601 | when the descope was enacted |
| `_rescoped_by` | str | authorising actor (agent id / human) |
| `_rescoped_note` | str | prose rationale |
| `_descoped_steps` | list | the dropped records — **new**, the machine-checkable twin of the note |

Each `_descoped_steps` entry:

```yaml
step_id:      "step-1"
commit:       "309c22ab6f"        # as recorded; MAY be unresolvable (50.2% are)
disposition:  orphan | revert | ride-along
authority:    "esc-2971-6"        # escalation id or human ratification marker
note:         "superseded by the Leo-ratified code-only scope"
```

### C5 — The β invariant (replacing "count must not decrease")

Let `D_before` / `D_after` be the `(step_id, commit)` sets of done steps.

```
DROPPED := D_before \ D_after
β PASSES iff
    every element of DROPPED appears in plan['_descoped_steps'] with a
        disposition and an authority
  AND the done-step commits in D_after are PAIRWISE UNIQUE
  AND for each element of D_after, _reconcile_done_step_commits has run and either
        resolved it or classified it unresolvable-but-equivalent
        (unresolvable alone is EXPECTED, never a failure — §2.7)
```

Note the asymmetry this buys: **2971 passes** (its dropped `(step-1, 309c22ab6f)` is
recorded with `disposition: orphan`), and **3143 passes** (all 19 dropped, each
recorded, disposition `orphan`, authority `esc-3143-4`) — while a silent
`create_plan()` wipe **fails**, because nothing recorded the drops.

### C6 — The γ merge gate (submission-time)

At `TaskWorkflow._submit_to_merge_queue`, alongside the existing
`_check_plan_files_touched_in_branch` call and **before**
`register_and_enqueue_merge_request`:

```
OWNED    := { commit of every plan step (any status) }
DESCOPED := { commit of every _descoped_steps entry }
γ PASSES iff  every commit in (base..branch_head)  ∈ OWNED ∪ DESCOPED
              OR is classified equivalent by find_equivalent_commit
```

Fail-open on git error (matching the sibling gates' documented convention). Rejection
returns the offending sha; the workflow may remedy in place and re-check, as
`_try_narrow_plan` already does at the same site. **INV-8:** the commit walk uses the
async git runner (`git_ops._run`), never a blocking `subprocess.run`, and its fan-out
is bounded by the branch's commit count with the loop-invariant probes hoisted.

### C7 — The δ advisory + event

```python
class UpdateTaskResult(TypedDict):
    id: str
    message: str
    updated: bool
    updated_task: dict | None
    amendment_advisory: NotRequired[dict]   # new; absent when not applicable
```

`amendment_advisory` = `{task_id, plan_authored_at, fields_changed, will_replan: bool,
plan_text_fp_before, task_text_fp_after}`. Emitted whenever the `should_reembed`
predicate fires **and** `task.metadata.plan_text_fp` exists and disagrees.

Because the curator bypasses `TaskInterceptor` entirely, δ **also** emits a structured
`task_text_amended` event carrying `writer_source`, so curator writes remain visible
to an operator even though no advisory can reach the curator's already-exited
classifier LLM.

### C8 — The ε predicate

```python
# task_interceptor.py, replacing `if target_status in {'done', 'cancelled'}`
if target_status != 'pending' or target.get('claimant_run_id') is not None:
    # abort -> falls through to create (the candidate becomes its own task)
```

This makes execution match selection (`combine_eligible == (status == 'pending')`).

---

## 10. Boundary-test sketch (B + H) — η's signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Amendment triggers a re-plan | plan `_task_text_fp` = A; task text now fingerprints B; no main overlap | dispatch takes the architect **revalidation** path; **no** `phase_skipped{revalidation_skipped_no_overlap}`; briefing contains the amended text |
| 2 | No amendment still skips | fingerprints equal; no main overlap | Lever-B skip fires exactly as today; `optimistic_path='revalidation_skip'` |
| 3 | Mid-EXECUTE amendment does not self-clear | task amended while implementer iterating; per-iteration provenance re-stamp runs | `_task_text_fp` is **unchanged** by the iteration stamp; the next dispatch still re-plans |
| 4 | Missing fingerprint is a fault | plan with no `_task_text_fp` (post-migration) | skip declined **and** a structured fault emitted with its counter — not a silent decline |
| 5 | Whitespace-only edit is not an amendment | text reflowed, tokens identical | fingerprint unchanged; skip still fires |
| 6 | Case-only edit **is** an amendment | `foo` → `Foo` | fingerprint changes; re-plan (D2) |
| 7 | Silent done-step wipe refused | revalidation returns a plan dropping a done step, no `_descoped_steps` | refused with a structured reason naming `(step_id, commit)` |
| 8 | Recorded descope proceeds (2971 replay) | dropped step recorded with `disposition: orphan` + authority | accepted; plan retains `_rescoped_*` and `_descoped_steps` |
| 9 | Phantom-done wipe proceeds (3143 replay) | 19 done steps dropped, all recorded, SHAs unresolvable | accepted — unresolvable is expected, not a failure |
| 10 | Duplicate done SHAs refused | two done steps share a commit | refused (closes the sha-duplication hole) |
| 11 | Ride-along commit blocked at submission | branch carries a commit owned by no step and named in no disposition | `_submit_to_merge_queue` refuses **before** enqueue; reason names the sha |
| 12 | Descoped ride-along permitted | same commit named with `disposition: ride-along` | submission proceeds |
| 13 | fused-memory advisory returned | `update_task(description=…)` on a task whose `metadata.plan_text_fp` disagrees | result carries `amendment_advisory`; `task_text_amended` event emitted |
| 14 | δ does no filesystem/git I/O | advisory path exercised under load | no filesystem or git call on the fused-memory loop thread (INV-8; D10) |
| 15 | Combine into `in-progress` refused (2951 replay) | live target `in-progress` at execution; snapshot said `pending` | combine aborts; candidate filed as its own task; refusal recorded in the combine audit |
| 16 | Combine into claimed `pending` refused | target `pending` but `claimant_run_id` set | refused (D11) |

---

## 11. Open questions (tactical only)

1. **Fault channel for D3/C3.** A `phase_skipped`-style event with a decline reason, a
   dedicated counter, or an `info` escalation? All three satisfy INV-2/INV-4. Decide
   in α against the existing event vocabulary.
2. **`_descoped_steps` vs per-step tombstone.** C4 puts dropped steps in a plan-level
   list; an alternative keeps them in `steps` with `status: 'descoped'`. The list keeps
   `steps` semantics unchanged for the ~20 existing readers. Decide in β.
3. **ζ's classification of "needs revising".** The 27 are known; the sweep should
   re-derive rather than hard-code. Whether it re-derives from transcripts or simply
   fingerprints current text against the plan's authoring-time briefing is an
   implementation choice for ζ.
4. **γ's treatment of merge-commits and rebase churn** on long-lived branches — the
   equivalence escape hatch is specified (C6) but its exact ordering versus the
   ownership check is tactical.
5. **Whether ε should also emit an advisory** to the curator's audit record beyond the
   existing refusal log line.

---

## 12. G7 walk (advisory; full walk at decompose)

| Invariant | Bearing | Disposition |
|---|---|---|
| `contracts-machine-checked` | The descope protocol is **prose today** (six hand-rolled instances). C4/C5 move it to schema + a checked predicate. | **Satisfied — this PRD is the fix** |
| `structured-facts-at-failure` | β/γ rejections name `(step_id, commit)`/sha structurally; δ's advisory and event carry typed fields, not a log string. | Satisfied |
| `corroborate-before-acting` | β corroborates done steps against git via the existing reconciler before trusting them; ε re-checks live status at execution instead of trusting the snapshot. | **Satisfied — ε is literally a corroboration fix** |
| `storm-escape-required` | D3's fault path and δ's advisory must carry counters, not silent decline/drop. | Addressed by C3 + open question 1 |
| `no-lockstep-duplication` | Two live copies of a description normalizer already exist. D2 puts ours in `shared/` with an explicit non-unification note; D6 forbids a second reachability mechanism. | **Satisfied — both are INV-5-driven** |
| `status-matches-liveness` | β/γ refusals must not leave a task in a status implying a live owner. | Satisfied via D13 — refusals route to `_handle_no_plan_failure`, an existing choke point that writes the successor status |
| `holds-owned-and-bounded` | A β refusal is a hold: who exits it, and what bounds it? | **Satisfied via D13** — exit owner is the architect in-loop; bound is the pre-existing `consecutive_no_plan_failures` streak, escalating to a human at ≥2 |
| `loop-thread-occupancy-bounded` | γ walks branch commits (git I/O); δ runs inside fused-memory's loop — the exact process of the task-3778 incident. | Addressed by C6 (async runner, hoisted invariants, bounded fan-out) and D10 (δ reads `task.metadata`, never the filesystem) |

No waivers proposed, and no live items: D13 closed the `holds-owned-and-bounded` and
`status-matches-liveness` rows by **reusing** an existing bounded path rather than
adding one. The decompose walk re-checks all eight against every task in the batch.

---

## 13. Provenance

Investigation: `/home/leo/.claude/spawn-briefs/df-amendment-drop-sweep-RESULT.md`
(instrument, corpus, and the 21-case table). Measurement artifacts:
`data/orchestrator/agent-transcripts/` (4,583 briefings), `data/combine_audit.jsonl`
(640 combines), `.worktrees/.task-meta/2971/plan.json` (the hand-rolled descope
schema), task **3157**'s 2026-08-05 addendum (the reachability and SHA-durability
measurements).
