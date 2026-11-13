# Merge pool, part 2: batches chosen from evidence (design notes, NOT a PRD)

**Status:** design notes, 2026-11-13. Not decomposable. This is the second full draft of a single PRD that was split;
part 1 is `plans/priority-at-later-gates-prd.md`. Two critic passes found 31 defects in the batch half, listed in
`plans/priority-at-later-gates-prd.evidence/critic-full-draft-pass1.md` and `critic-full-draft-pass2.md`; the
second pass's items are NOT folded into the text below. Read those before reusing anything here. Part 2 is to be
authored as its own PRD after task 8528 has landed and task 9908's rows exist. Where this text and part 1 differ
on priority, pins or age, part 1 is right.

## 1. Goal

Three gates decide which task goes next: the scheduler, the verify-admission gate and the merge queue. Today only the
scheduler reads priority. This PRD makes the two later gates read the scheduler's priority as one value, `(tier,
age)`, strict across tiers, makes a pin an ordinary tier, and replaces the arrival-ordered merge queue with a
**merge pool**: waiting requests are taken for verification in priority order, in batches whose members are chosen
from the recorded evidence about which requests are likely to fail.

What an operator observes when it has landed:

1. A pinned or high-tier task that reaches the merge lane after lower-tier requests is verified before them, unless
   they are already merged or being verified. `merge_dequeued` rows show the picked request first among the recorded
   waiting set by rank and age.
2. `set_task_priority_override(pinned=True)` lifts the pinned task's unfinished prerequisites as well as the task, at
   the scheduler and at both later gates. `get_scheduler_state().effective_priorities` shows them at tier `pinned`.
3. At the later gates a task's age counts from when it could first have run until it is done. It has no cap, and it
   is not reset when the task is re-pended or the orchestrator restarts.
4. Every merge batch carries a `merge_pool_decision` row with the predicted probability that the batch as verified
   would pass, and a committed readout checks that prediction against the verdict.
5. When a batch fails, its members are narrowed down by further batches. A waiting request whose probability of being
   the culprit reaches the conviction threshold goes back to its workflow with the failing evidence and a statement
   of how it was convicted.
6. A suspect past the suspect deadline is verified alone ahead of all other picks.

## 2. Background and premise validity (G6)

From the study (`evidence/REPORT.md`) and the measurements taken for this PRD (`evidence/measure.md`; dark_factory,
"recent" = 2026-10-30 to 2026-11-13, "paused" = 2026-11-09 to 2026-11-13):

- **Priority is read at one gate.** 39% of merge picks in the paused window took a request while a higher-tier or
  pinned one waited. `merge_lane/types.py::MergeRequest` has no tier, and the verify-admission key is `(role class,
  arrival, seq)` (`verify_admission_gate.py::AdmissionGate._pick`).
- **The size of the prize is modest and is stated as such.** Simulated on the recorded queue, a priority pick is
  worth under an hour to an urgent request at today's arrival rate, more when the queue is long. This PRD is
  justified by one mechanism replacing three (lanes set by hand, pins that stop at dispatch, per-gate ordering
  rules) and by better-chosen batches. No latency target is asserted.
- **No age survives past dispatch.** The scheduler's anchor, `metadata.pending_since`, is deleted by every
  `metadata_mode='replace'` write, by design (task 3816's "design decision 8";
  `sqlite_task_backend.py::SqliteTaskBackend.update_task`): 26 of 27 done tasks with id 9000 or above, and 15 of 18
  in-progress tasks, have none. The watchdog's `eligible_since` (`eligibility_clock.py::EligibilityClock`) is
  forgotten at dispatch.
- **A pin stops at the pinned task.** `Scheduler._compute_effective_priorities` takes the override's `boost_tier`
  and never `pinned`. A pinned task with unmet dependencies is not in the pin queue (`Scheduler._pin_queue`).
- **Batches are sized by one lane-wide number and blame nobody.** Pending task 8528 sizes a batch from a rolling
  pass rate and bisects a failed batch by halves (`merge-lane-policy-prd.md` D6, D7). A failed batch produces no
  outcome for anyone (`worker.py::SpeculativeMergeWorker._requeue_from_verify`).
- **Measured rates.** Single-request merge verifies passed 396 of 513 (0.772, recent) and 128 of 153 (0.837,
  paused), gate re-verifications excluded. Batches implied a per-request pass rate of 0.874 and 0.894. Task 8528's
  rolling 30-verdict estimate swings between 0.52 and 0.97 (10th to 90th percentile). Of failed single verifies,
  8% to 31% were followed by a pass with no agent run in between: a bound on false failures of 0.02 to 0.09 per
  verify of a sound tree. A batch takes no longer to verify than one request (thin: 59 batches). The waiting set at
  a pick has median 6, 95th percentile 13.

**Premises checked.**
- *Exact inference is cheap enough.* At most `max_suspects` = 16 requests are enumerated: 65,536 hypotheses, one
  bit-mask test per observation, 0.39 s in plain Python at the limit (the critic's timing). It runs off the event
  loop, per connected group, under a stated budget (6.3).
- *The batch rule does not justify bigger batches than the fallback.* With equal weights and nothing stacked behind,
  "add a request while it raises the expected number landed" admits the n-th member while q < 1/n: 14, 6 and 3 at
  pass estimates 0.93, 0.85 and 0.75, against the pending table's 8, 4 and 2. The table is smaller because a failed
  batch also cancels what is stacked on it, which the rule does not count. The pool therefore chooses who is in a
  batch and never takes more than the table would (6.4), until a replay with stacking counted says otherwise (ι).
- *Conviction by elimination may seldom fire, and that is acceptable.* When every other member of a failed batch has
  landed, the last suspect is bad with probability q / (q + (1 - q) x false_failure): 0.73 to 0.95 across the
  measured ranges, 0.86 at the seeded values (0.16, 0.03), which is below the 0.9 threshold. It is then verified
  alone, exactly as under task 8528. Whether elimination convicts in practice is decided by the refitted
  false-failure rate (ι), not asserted here.

## 3. Resolved design decisions

**Ruled by Leo (2026-11-12, 2026-11-13).**

- **R1.** Nothing already merged or being verified is displaced for a higher-priority arrival.
- **R2.** The merge queue becomes a pool taken in priority order. Priority at the pool is the scheduler's.
- **R3.** Priority is `(tier, age)`, strict across tiers, at both later gates, with a promotion after a long wait as
  the only starvation backstop. The backstop's default comes from real recorded waits.
- **R4.** Age keeps accruing after a task is scheduled, through to done, with no cap.
- **R5.** A pin is a tier above critical in one uniform tiering.
- **R6.** Batch size comes from per-request failure probabilities updated from failed batches. Requests after a
  bisection point return to the pool and may be reordered; those being verified may not.
- **R7.** A suspect must not sit in the pool. Conviction by elimination at 0.9 is acceptable.
- **R8.** The pick change lands after task 8528. The recording task 9908 is hand-carried ahead of everything.
- **R9.** The same-files rule is dropped from the pick; the test is narrowed and kept as the conflict-probe filter.
  The rebase cap is relaxed substantially.

**Awaiting Leo** (taken at the recommended value so the rest of the document is definite).

- **D2 [Leo]. Where age starts.** Task 8261's amendment records the ruling as "first-eligible". No such stamp
  exists. Two options:
  - *(recommended)* **First eligible.** A durable `metadata.eligible_since`, written once by the scheduler the first
    tick the task is a dispatch candidate, reset only on `cancelled → pending`. Time parked behind a dependency does
    not count. Cost: one write site in `scheduler.py`, and the metadata key.
  - **First pending.** Make `pending_since` durable instead, by reversing task 3816's design decision 8 in the
    fused-memory backend. No scheduler edit. Time parked behind a dependency counts as age: with no cap, a task
    parked for two months outranks, within its tier at the later gates, one that has been ready for one. About 177
    open tasks are parked behind task 8528 alone.
- **D3 [Leo]. The scheduler's own order within a tier.** Once age is uncapped, the scheduler's two other terms (the
  count of dependents and the work-product credit) cannot keep today's weight. Three options:
  - *(recommended)* **Leave dispatch order as it is.** The scheduler is already strict across tiers, and its age
    term already grows for as long as a task waits (it saturates; it does not stop). Only the later gates take raw
    uncapped age. No scoring change, no reversal of the earlier scoring PRD's worked case, and nothing of this PRD
    waits behind the scheduler restructuring chain except the pin tier.
  - **Pure age with fixed credits** (17 h for a work product, up to 43 h for dependents). Within a tier, dispatch
    becomes first-come order for any two tasks more than about two days apart; the earlier PRD's founding case
    (a chain head with 9 dependents, 93 h old, against a 214 h task with none) reverses.
  - **Age multiplied by a factor for dependents and work product**, coefficients fitted from a replay. Keeps today's
    behaviour; adds a fitted formula.
- **D18 [Leo]. Task 8261 becomes this PRD's β.** It would implement the shared comparator at verify admission
  directly, keep its id, pin and dependents, and wait for α. The alternative is to let it land its own blended key
  first and replace it later.

**Decided here** (listed in the hand-back; an unlisted one stands as written).

- **D1. One value, one comparator.** `shared/task_priority.py` (task 6718) owns the tier names and ranks, `pinned`
  included as an effective-only tier. `orchestrator/task_priority.py` (created by task 9908 as `TaskPriority(tier,
  pinned)`) imports them, grows `age_anchor` and `pin_order`, and gains the one ordering function `order_key`
  (6.1). Verify admission and the merge pool sort by `order_key` and by nothing of their own.
- **D5. `pinned` is an effective tier, never a stored one.** A task's own `priority` cannot be `pinned`. From α, the
  lookup returns tier `pinned` and the row's `pin_order` for a task whose own override row is pinned, so both later
  gates honour pins at once. θ adds inheritance: the existing walk (`Scheduler._compute_effective_priorities`) takes
  pins, so a pinned task's unfinished prerequisites become `pinned` with the smallest `pin_order` among their
  pinned dependents.
- **D6. Pin order is the order inside the `pinned` tier**, and `reorder_pin_queue` keeps working. The separate pin
  pass as a dispatch order goes; pinned-tier candidates are simply first in the scheduler's order.
- **D7. The pin reservation stays bounded.** A park installed for a `pinned`-tier owner still ranks above every
  other tier and is ordered by `pin_order`. At most `pin_reservation_max_active` owners hold one, chosen as today
  (the first lock-blocked, non-deterministic, not load-held candidates of the `pinned` tier), and
  `Scheduler._bound_pin_reservations` keeps enforcing it. Inherited prerequisites compete for that reservation by
  `pin_order`; they do not each get one. This stays inside the earlier scoring PRD's rejection of reserving for
  every starving task. `pin_blocked`, `pin_reservations` and `pin_queue` keep their meaning: `pin_queue` lists
  stored pins only, `effective_priorities` shows the inherited tier.
- **D8. No promotion backstop at the scheduler.** The starvation watchdog remains that gate's backstop.
- **D9. The promotion backstop is a step.** A waiter that has waited at a gate longer than that gate's
  `promote_after_secs` is treated as one tier higher for each such period, stopping at `high`: it never ties with
  real `critical` or `pinned` work. Waiting is counted from the request's own enqueue (merge pool) or the leg's
  arrival (verify admission). Defaults are the 95th percentile of the gate's own recorded wait, rounded up to the
  hour, per project, printed by a committed script; the value is committed in the project's config by the task that
  ships the knob. Derived once, not live: a live percentile would lengthen as the low tier waits longer.
- **D10. The "high" lane remains, for main repair only**, strictly ahead of everything.
- **D11. Inference is one pure function over recorded observations** (6.3). An observation is defined at the point
  a verdict counts, and has its own event row; the lane's in-memory list is a cache rebuilt from named row kinds.
- **D12. Suspects are ordered with clean work, never mixed with it** in one batch or one stack (6.5).
- **D13. Only a waiting request is convicted; a failed own verify convicts at once.** A conviction by elimination
  does not count as merge thrash the first time; the same content convicted twice does.
- **D14. Two switches, both hot.** `merge_pool.order` is `arrival` or `priority` and governs the pick.
  `merge_pool.mode` is `off`, `shadow` or `on` and governs batch membership, suspects and conviction. Both default
  to today's behaviour. Each is flipped by a human gate that presents a readout. With `mode` not `on`, task 8528's
  suspects stay first in the order so its bisect keeps working.
- **D15. Verdicts given on a red main are not evidence.** When main is confirmed red at commit M, every failed
  observation whose tested tree contained M is marked `main_red` retroactively and excluded, including the one that
  triggered the probe.
- **D16. Convictions have a storm limit.** More than `convict_storm_max` (3) in an hour stops convictions and files
  one L1; narrowing by verification continues.
- **D17. New lane code goes in new modules** under `merge_lane/`, each within the ratchet's ceilings (1,500 lines a
  file, cognitive 15 a function). `merge_pool` config is its own sub-model in its own file, mounted once.
- **D19. Sequencing.** Recording (9908) → the durable age and comparator (α) → verify admission (β) and the pool
  order (γ, after 8528, behind its switch) → the observation log and inference (ε) → shadow decisions (ζ, after the
  batch code has moved to its own file) → readout, replay and human gates (ι) → the pool drives (η1, η2). The pin
  tier inside the scheduler (θ1, θ2) waits for the shared tier vocabulary and the last `scheduler.py` leaf of the
  restructuring chain. κ is the integration gate.

**Considered and rejected.**
- *Folding the pool into task 8528.* It heads a protected chain with about 200 tasks behind it. It lands as planned
  and remains the fallback.
- *A separate urgency value for the merge lane* (what a request blocks upstream). Left to the lock hand-off work.
- *Testing urgent requests alone; deriving the lane from the tier; choosing the host by urgency.* Simulated as
  harmful or worth minutes.
- *Basing a batch off the frontier.* A real improvement and out of scope; 6.5's no-mixing rule is used instead.
- *Weighting batch membership by tier.* With weights 16/8/4/2/1 a low-tier request is never admitted behind a pinned
  head although an extra member costs no verify time. Weights default to 1; unequal weights are a config option.
- *A reason prefix for convictions.* The structured `conviction` field is the only discriminator.
- *A pair rule for requests that fail only together.* A batch that passes lands, so "each passed without the other"
  cannot occur in the pool; the second then fails its own run on top of the first, which the lane already labels.
- *Per-test blame.* Verify units on `merge_verify` are per module leg, and `MergeOutcome` carries failing tests only
  as text. A later PRD.

## 4. Pre-conditions (G3)

Verified on main, by symbol:

| Assumed | Evidence |
|---|---|
| One pick site in the lane | `merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`, called from `worker.py::SpeculativeMergeWorker._acquire_next_request` and `::_look_ahead` |
| The batch's members are added at dispatch, on the picked item's merge commit | `worker.py::SpeculativeMergeWorker._deep_chain_placement` |
| Members can be any set and any order | `merge_lane/chain_reservation.py::ChainReservations`; `MergeIntake.take_buffered` (by task id); `worker.py::build_chain` (any ordered sequence) |
| A still-waiting request can be resolved with an outcome | `suffix_graph.py::SuffixConflictTracker._escalate` (the precedent) |
| A blocked merge reaches a fixing agent | `workflow.py::TaskWorkflow._submit_to_merge_queue` generic blocked arm → `::_mark_blocked(merge_phase=True)` → the steward → retry in place |
| The thrash counter can be skipped from the outcome | `workflow.py::TaskWorkflow._check_merge_outcome_thrash` runs only when `_last_merge_block_reason` is set |
| `MergeOutcome` takes an optional field without touching the journal or outbox | it is a plain dataclass; `queue_store.py::PersistedMergeRequest` does not persist outcomes |
| A fork-point changed-files query | `git_ops.py::GitOps.get_branch_changed_files` (`main...ref`), returning `(files, error)` |
| Confirmed conflicts between waiting requests | `suffix_graph.py::SuffixConflictGraph.textual_edges` |
| The effective-tier walk with inheritance | `scheduler.py::Scheduler._compute_effective_priorities` |
| The bounded pin reservation | `scheduler.py::Scheduler._bound_pin_reservations`, `pin_reservation.py::PinOrder` |
| First-candidacy stamp (not durable) | `eligibility_clock.py::EligibilityClock.observe`, called from `Scheduler._apply_starvation_watchdog` |
| Hot-reloadable config sub-models | `config.py::MergeDeepConfig`, `RELOADABLE_FIELDS` via `_submodel_leaf_paths` |

Not on main:

| Needed | Delivered by | Status 2026-11-13 |
|---|---|---|
| `TaskPriority`, the injected lookup, and the recorded rows (`merge_queued` lane and tier, `merge_dequeued` waiting set, `merge_verify.chain_request_ids`) | task 9908 | deferred, hand-carry in progress |
| The suspect list, depth table and `chain_depth_reason` | task 8528, behind 8527 and 8526 | pending |
| The batch code in its own file | task 5041 (`merge_lane/speculation.py`), behind 5040; landing in task 5042 | pending |
| One tier vocabulary and the override store's home | task 6718 | pending |
| The last `scheduler.py` leaf of the restructuring chain | task 9683 (confirm at decompose) | pending |
| A stable identity for a request's content across rebases | built here (ε): `git patch-id --stable` of the fork-point diff; no wrapper exists in `git_ops.py` | — |
| A durable eligibility stamp | built here (α) | — |

Used if present, not required: task 9144 (`MergeOutcome.kind`), task 7921 (main-red verdict book), task 9709
(`config_sections/merge_lane.py`).

## 5. Cross-PRD relationships and seam owners (G4)

| Other work | Direction | Seam | Owner | Action |
|---|---|---|---|---|
| Task 9908 (recording) | consumes, extends | `TaskPriority`, `read_task_priority`, the lookup, pick records | 9908 builds; α extends | α edits the guard so new fields survive, and gives every new field a `None` default |
| Task 8528 (bisect and depth table) | consumes; replaced when `mode=on` | the suspect list; the depth for the current estimate | 8528 owns its code; γ keeps its suspects first; ζ reads its depth | — |
| Task 8532 (seven-day readout) | none | reads the lane with both switches at their defaults | 8532 | ι's gates do not fire until 8532 has read |
| Task 8932 (denser depth table design) | superseded | the depth table | this PRD | cancel at decompose with Leo's word |
| Tasks 5040, 5041, 5042, 5306, 5045 (split `worker.py`) | consumes | where placement and landing live | they own the moves | ζ depends on 5041; η1 and η2 on 5041 and 5042 |
| `merge-lane-quality-prd.md` frozen line | amends | "main advances strictly in submission order" becomes "in the order verifies were dispatched" | γ | dated delta note |
| `two-layer-merge-queue-prd.md`, task 1891 | supersedes | the same-files pick rule | γ | dated delta note |
| `deep-merge-ahead-prd.md` decision 1 | amends | "in submission order" becomes "in pool order" | γ | dated delta note |
| Task 7921 (main-red hold) | consumes | held requests are not candidates; its verdict book becomes the "main is red" read | η1 owns the switch: if the verdict book exists when η1 runs it reads it, else η1 files the follow-up as a dependent of 7921 | dated note on 7921 at decompose |
| Task 9144 (`MergeOutcome.kind`) | produces | the `conviction` field | η2 owns both orders: if `kind` exists when η2 runs it adds the member, else it files the follow-up as a dependent of 9144 | dated note on 9144 at decompose |
| Task 8261 (verify admission by tier) | becomes β (D18) | the admission key | this PRD | amend at decompose: implements `order_key`, depends on α; its knob becomes the promotion period |
| Task 5623 (journal loses the first-enqueue stamp) | none | the recorded field | 5623 | stays open |
| Task 8933 (verify on a dead base after a failed batch) | must land before or with η1 | the failed-batch requeue arm | 8933 | η1 depends on it |
| Tasks 9182, 9152, 9723 | same file | `merge_lane/intake_buffers.py` | each its own | serialised by the file lock; γ is filed against that file |
| Task 9188 (merge tools move) | consumes | queue position in `get_merge_queue` and `merge_status` reads the pool order | 9188 moves; γ changes what they read | note on 9188 |
| Task 9137 (host selection in the allocator) | none | no per-purpose slot is added | 9137 | — |
| `scheduler-dispatch-scoring-and-lock-layer-prd.md` | amends | the pin pass; the pin reservation exception (kept, D7) | θ1 | dated delta note |
| Tasks 6328, 3824 (reservations for critical work; durable parks) | same rule | the park order | each its own | θ1 depends on neither; note on both that the reservation band is unchanged |
| Task 9636 (watchdog cap and restart-safe inputs) | consumes | watchdog pins become `pinned`-tier like operator pins | 9636 | θ1 after 9636 if it is in flight; note |
| Tasks 6718, 6642, 6722, 7009, 4709 | consumes | the tier vocabulary, validators, park payload, override events | each its own | θ2 after 6718; `pinned` is added once, in the shared module |
| Task 9705, 9709 (config moves) | same knobs | where admission and lane knobs live | each its own | `merge_pool` is its own file from the start (D17) |
| Task 6658 (derailed members lose their place) | none | its premise is unchanged while D3 stands | 6658 | — |
| Lock hand-off study (unruled) | none | reservations beyond the bounded one | that study | not designed here |

## 6. Contract (B + H)

### 6.1 Priority and its order

```
@dataclass(frozen=True)
class TaskPriority:
    tier: str | None = None             # effective tier, from shared/task_priority.py
    pinned: bool | None = None          # task 9908's field, kept
    pin_order: int | None = None        # set only when tier == 'pinned'
    age_anchor: datetime | None = None  # metadata.eligible_since (D2)

def order_key(p: TaskPriority, *, now: datetime, waited_secs: float,
              promote_after_secs: float | None) -> tuple[int, int, float]
```

- Smaller sorts first: `(rank, pin_order or 0, -age)`. `rank` is the tier's index with `pinned` first. `age` is
  `now - age_anchor` in seconds, 0 when the anchor is absent.
- Promotion: `rank` is reduced by `floor(waited_secs / promote_after_secs)`, never past the rank of `high`.
- An unknown tier or a failed lookup gives the rank of `medium` and age 0, and the caller counts it on its event. A
  waiter with no task (a background leg at verify admission) is not a failed lookup. The lookup never raises.
- Backstop defaults, 95th percentile rounded up: dark_factory verify admission 4 h (13,135 s), merge pool 9 h
  (32,018 s, wait to first pick). The dark_factory window includes the days when grouped merges distorted the
  merge-wait figures; β and γ re-run the script and commit what it prints. Other projects: an operator reload.

Invariants:
- **P1.** Verify admission and the merge pool order waiters by `order_key`, after role class at admission and after
  the high lane and (while `mode` is not `on`) task 8528's suspects at the pool.
- **P2.** Ties break on the caller's existing sequence (`seq` at admission, request id in the pool).
- **P3.** Among waiters not promoted, a higher tier always precedes a lower.

### 6.2 The age anchor

- `metadata.eligible_since` is written once, by the scheduler, the first tick the task is a dispatch candidate. It is
  blessed and machine-authored in `shared/task_metadata.py`. It survives every metadata write, `replace` mode
  included: the backend re-attaches it after `strip_machine_authored_metadata`. Only `cancelled → pending` clears it.
- Rows that are already past dispatch get a one-off repair that can only under-age: the **latest** time the task can
  be shown to have been a candidate by (its first `lock_acquired` or `task_started` event, else `updated_at`),
  marked `eligible_since_repaired`.
- **A1.** For any task, `eligible_since` read after any sequence of writes, dispatches, re-pends and restarts equals
  the value first stamped, unless a `cancelled → pending` transition intervened.

### 6.3 Observations and inference

**An observation exists when an entry is finalized as head of the frontier with a test verdict.** Its base is then
on main, so the tree tested was main plus exactly its members. A verify that was cancelled, voided by a failed base,
requeued without a verdict (lease contention, runner unavailable, an error on the batch arm), or re-run as a gate
re-verification is not an observation. Members dropped when `build_chain` truncated are not in it.

The pool emits one `merge_pool_observation` row per observation: members by request id and content id, `passed`,
`alone`, `main_red`, and the `merge_verify` event it came from. The cache is rebuilt at start from four row kinds
only, bounded to the requests in the journal: `merge_pool_observation`, `merge_finalized` (landings),
`merge_convicted`, and `merge_queued` (the content id of a re-submission).

A request's **content id** is `git patch-id --stable` of its fork-point diff, computed when the request is enqueued
and carried on `MergeRequest` and `merge_queued`. It survives a rebase. "Unchanged" means equal content id.

```
@dataclass(frozen=True)
class Observation:
    members: frozenset[str]     # request ids
    passed: bool
    alone: bool                 # verified alone when dispatched
    main_red: bool
    at: datetime

class Posterior:
    def p_bad(self, request_id: str) -> float
    def p_pass(self, members: Collection[str]) -> float
    suspects: frozenset[str]
    groups: tuple[frozenset[str], ...]   # suspects joined by a remaining failed observation

def infer(priors: Mapping[str, float], observations: Sequence[Observation], *,
          false_failure: float, max_suspects: int) -> Posterior
```

- A hypothesis is the set of bad requests. Its prior weight is the product of each request's prior or one minus it.
  An observation with no bad member passes with probability `1 - false_failure`; with a bad member it fails. The
  posterior is the normalised product, computed per group. Requests in no remaining failed observation are not
  enumerated.
- A landed member is removed from the member sets of earlier failed observations; a failed observation all of whose
  members have landed contributes nothing and is counted as a false failure. A convicted member takes with it every
  failed observation it was in. An observation is dropped when a member's content id changes. `main_red`
  observations are excluded.
- Two priors. `prior_batch` sizes batches: one minus task 8528's rolling estimate. `prior_conviction` is slow: the
  14-day single-request failure share (seed 0.16), refitted by ι. Conviction uses only the slow one, so convictions
  do not rise during a run of failures.
- `false_failure` default 0.03 (measured bound 0.02 to 0.09), provisional, refitted by ι. Validators:
  0 < `false_failure` < 0.5; 0.5 < `convict_threshold` <= 1.
- A group larger than `max_suspects` raises `TooManySuspects`. With `mode=on` the caller verifies its lowest-priority
  members alone until it fits. In `shadow` the decision row records it.
- **I1.** `infer` is pure and deterministic, runs off the event loop, and one decision's inference is budgeted at
  2 s; over budget is a `decision_error`, and the dispatch proceeds by task 8528's rule.
- **I2.** A request that passed in a verify whose landing was then aborted has `p_bad` 0 until its content id
  changes.
- **I3.** The posterior depends only on the observations, never on the order they are supplied in.

### 6.4 Heads and clean batches

A batch has a **head** and **members**. The head is the request the merger picked; members are added when the head
is dispatched. Both decisions are the pool's.

```
def next_head(order: Sequence[PoolRequest], posterior: Posterior, in_flight: Frontier) -> Head | None
# Head(request, kind: 'clean' | 'suspect')            -- what MergeIntake.pop_next_pickable returns
def select_clean_batch(head, candidates, posterior, *, max_batch, conflicts) -> BatchPlan
# BatchPlan(members, p_pass_chosen, skipped: tuple[(request_id, reason), ...])
```

- With `mode=on`, a clean head is the first non-suspect of `pool_order()`; the rules for a suspect head are in 6.5.
  With `mode` not `on`, the head is the first of `pool_order()` and members are task 8528's.
- `select_clean_batch` runs in the placement step. `candidates` are the waiting, unreserved, unheld, non-suspect
  requests in `pool_order()`, the head excluded. The head is member 1 and lands first; the rest follow in order.
- Each candidate c is added when `(n + 1) * p_pass(S + c) > n * p_pass(S)` for a batch of n so far (weights 1), else
  skipped with reason `risk`; the walk continues. A candidate with a confirmed conflict with a member is skipped
  with reason `conflict`.
- `max_batch` at run time is the smaller of the config value (8) and the depth task 8528's table gives for the
  current estimate.
- A head whose `p_bad` exceeds 0.5, or that was picked `clean` and has become a suspect by dispatch, is verified
  alone.
- **B1.** A skipped candidate keeps its place in the pool.
- **B2.** The pool never dispatches more members than task 8528's table would at that moment.

### 6.5 Suspects

- **Own run.** A failed observation with `alone` true resolves that request at once through the lane's existing
  own-verify arm (`_run_post_merge_verify`'s blocked outcome), with `conviction.by = 'own_run'`. The threshold
  applies only to requests with no failed own run.
- **Only waiting requests are convicted.** A request that is merging, reserved or being verified is never
  convicted; its verdict decides. Before convicting, the lane re-checks that the request still waits with the same
  content id.
- **Elimination.** A waiting request with `p_bad >= convict_threshold` (0.9, computed with `prior_conviction`) is
  resolved back to its workflow (6.6).
- **Ordering.** Suspect work is ordered with clean work, not given a slot. A suspect head is taken when the first
  suspect precedes the first clean candidate in `pool_order()`, or when a suspect is past the deadline. Any number
  of suspect batches may be in flight, one per group.
- **The suspect batch** for a group is the prefix, in `pool_order()`, whose `p_pass` is nearest 0.5; on a tie the
  shorter. Its first request is the head.
- **No mixing in the stack.** The pool does not pick a head of the other kind while one is in flight ahead of it,
  unless the frontier is empty. A suspect batch is therefore never stacked on an unverified clean batch, nor a clean
  batch on a suspect batch.
- **Deadline.** A suspect past `suspect_deadline_secs` is verified alone, oldest first, ahead of all other picks.
  Default 2 h: twice the 95th percentile of a passed verify on the slowest host (3,633 s). The clock stops while the
  suspect's lane is halted.
- **S1.** A suspect waits at most `suspect_deadline_secs` plus one verify for each suspect ahead of it past the
  deadline, divided by the number of hosts.
- **S2.** A suspect is never a member of a clean batch.

### 6.6 Conviction back to the workflow

```
@dataclass(frozen=True)
class ConvictionEvidence:
    by: Literal['own_run', 'elimination']
    p_bad: float
    content_id: str
    failed_batches: tuple[tuple[str, ...], ...]   # task ids of each failed batch it was in
    failing_units: tuple[str, ...]                # module legs that ran red in those batches
```

- The outcome is `MergeOutcome('blocked', reason, block_category=POST_MERGE_VERIFY, conviction=...)`. `reason` is
  built by the existing "Post-merge verification failed" builder from the failed batch's report. No new prefix.
- The workflow takes its existing generic arm; the escalation detail states how the request was convicted.
- `consecutive_merge_thrash` advances for `by == 'own_run'`, for an outcome with no conviction record, and for a
  second elimination of the same content id. It does not advance for a first elimination.
- A request whose task was convicted and that returns with the same content id is verified alone before anything
  else. The lane learns this from `merge_convicted` rows keyed by task id and content id.
- A wrong conviction costs one steward pass, one of the task's merge retries, a rebase and one verify.
- **C1.** Every conviction emits `merge_convicted` with the evidence. η2's readout extension joins it to the later
  verdict on the same content id; the count of wrong convictions is a lower bound (it sees only unchanged returns).
- **C2.** More than `convict_storm_max` (3) convictions in an hour stops convictions and files one L1 (D16).

### 6.7 Pool order, conflicts, config, events

- `MergeIntake.pool_order()` returns the waiting requests in this order: the high lane (D10); then, while `mode` is
  not `on`, the requests in task 8528's suspect list in the order it requeued them; then the rest, by `order_key`
  when `merge_pool.order` is `priority` and as today when it is `arrival`.
- Every reader of "the order requests will be taken in" reads `pool_order()`: the pick, `chain_snapshot`,
  `unfrozen_suffix`, `snapshot()`, the conflict graph's node order, `take_buffered`, `get_merge_queue` and
  `merge_status` positions, `merge_heartbeat.head_of_line`, task 9908's pick records, and
  `_rebucket_relaned_requests`. The shutdown drain is exempt.
- With `order=priority` the same-files rule (`MergeIntake._clique_minimal_visible`) is not applied.
- The conflict graph's footprint is each branch's own changes (`GitOps.get_branch_changed_files`). A git error is a
  probe error, not an empty footprint.
- `merge_pool.bounce_cap` replaces `MERGE_BOUNCE_CAP`; default 20, a loop-breaker (the cap of 3 was exceeded 0 times
  in 19 days).
- Config `merge_pool` (own file, all leaves hot): `order`, `mode`, `promote_after_secs`, `max_batch`,
  `max_suspects`, `convict_threshold`, `convict_storm_max`, `false_failure`, `prior_conviction`,
  `suspect_deadline_secs`, `bounce_cap`.
- Events. New: `merge_pool_observation`; `merge_pool_decision` (mode, kind, members chosen, members dispatched,
  `p_pass_chosen`, `p_pass_dispatched` = the same posterior evaluated on the members the verify covered, skipped
  with reasons, suspects, priors, `decision_error`); `merge_convicted`. New keys: `content_id` on `merge_queued`;
  `age_secs` and `rank` on the picked and each waiting entry of `merge_dequeued`. No existing key changes meaning.
- **O1.** Main advances in the order verifies were dispatched. Arrival order is not preserved.
- **O2.** A request being merged or verified is never reordered, re-based or returned to the pool by a later
  arrival (R1).
- **O3.** A decision that cannot be computed writes a `merge_pool_decision` row with `decision_error`; the absence
  of a row means `mode=off`.

## 7. Boundary-test sketch (B + H)

Driven through the lane's public seam (requests in, events and outcomes out), the admission gate's entry and the
scheduler's dispatch entry.

| # | Scenario | Preconditions | Asserts |
|---|---|---|---|
| 1 | Later high-tier request | `order=priority`; three low-tier requests wait, none in flight; a high-tier request arrives | it is picked next; `merge_dequeued` lists the three with `rank` and `age_secs` |
| 2 | Nothing in flight is displaced | a low-tier batch is being verified; a request whose task is pinned arrives | the batch's verify is not cancelled; the pinned request is the next pick |
| 3 | Promotion backstop | a low-tier request has waited past `promote_after_secs`; fresh medium-tier requests keep arriving | it is picked ahead of them; a fresh high-tier request still precedes it after one promotion |
| 4 | Main-repair lane | a high-lane low-tier request and a pinned normal-lane request wait | the high-lane request is picked first |
| 5 | 8528's bisect under the pool order | `mode=off`, `order=priority`; a batch of 4 fails; a high-tier request arrives | the next dispatch verifies two of the four suspects, not the arrival |
| 6 | Age survives | a task is dispatched, a `replace`-mode metadata write occurs, it is re-pended and the orchestrator restarts | `eligible_since` is unchanged; its age at the merge pool is from the original stamp |
| 7 | Parked time does not count | two same-tier tasks; one filed 30 days ago behind a dependency that landed today, one eligible for 5 days | at the merge pool the 5-day task precedes |
| 8 | Batch rule | `mode=on`; eight equal-tier waiting requests at prior 0.16; the table's depth is 8 | the batch has 6 members; two are skipped with reason `risk` and stay in the pool |
| 9 | Table bound | as 8, with the table's depth 4 | the batch has 4 members |
| 10 | Failed batch narrows | a batch of 6 is finalized failed | the six are suspects; the next suspect batch is the prefix with `p_pass` nearest 0.5; no clean head is picked while it is at the back of the frontier |
| 11 | Own run | a request verified alone fails | it is resolved `blocked` with `conviction.by == 'own_run'` after one verify |
| 12 | Elimination | `false_failure` 0.01; of four suspects three land | the fourth, still waiting, is resolved with `by == 'elimination'` and no verify of its own; the thrash counter does not advance. With `false_failure` 0.03 it is verified alone instead |
| 13 | Never in flight | a suspect is being verified alone when its co-members land | it is not convicted; its verdict decides |
| 14 | False failure | a batch of 4 fails; both halves then pass | all four land, no suspect remains, and the readout counts one false failure |
| 15 | Convicted content returns | a request convicted by elimination returns with the same content id | it is verified alone before any other pick; a second elimination of it advances the thrash counter |
| 16 | Deadline | three suspects pass the deadline together | they are verified alone, oldest first, ahead of other picks |
| 17 | Main red, retroactive | two batches fail; main is then confirmed red at a commit both trees contained | both observations are marked `main_red`; nobody is a suspect because of them |
| 18 | Storm | four convictions in one hour | the fourth does not happen; one L1 is filed; narrowing continues |
| 19 | Shadow | `mode=shadow`; a batch is dispatched by task 8528's rule | a `merge_pool_decision` row carries the would-be members and `p_pass_dispatched` for the members actually verified; the dispatched members are 8528's |
| 20 | Decision error | inference raises in shadow | a row with `decision_error` is written and the dispatch is unaffected |
| 21 | Kill switch | `mode` goes from `on` to `off` while suspects exist | the next dispatch follows task 8528's rule with the suspects first; no request is lost |
| 22 | Restart | the orchestrator restarts with suspects, a landing and a conviction recorded | the rebuilt cache gives the same next head as without the restart; the convicted request's co-members are not suspects again |
| 23 | Rebase keeps identity | a suspect is rebased in place by the conflict bounce | its content id is unchanged and its observations are kept |
| 24 | Confirmed conflict | two waiting requests have a confirmed conflict | they are never members of one batch |
| 25 | Verify admission | a high-tier task leg arrives after a low-tier one | the high-tier leg is granted first; a background leg's class rule is unchanged and it is not counted as a failed lookup |
| 26 | Pin reaches a prerequisite | task B pinned, depends on unfinished A (low tier); critical work is ready | A dispatches before the critical work; `effective_priorities[A] == 'pinned'`; `pin_queue` lists B only |
| 27 | Reservation stays bounded | B pinned with three unfinished prerequisites, all lock-blocked | exactly one holds a pin reservation |
| 28 | Prerequisite at the later gates | as 26; A's request and a critical request wait in the pool | A's request precedes |

## 8. Decomposition plan

Greek labels are this plan's; ids are assigned at decompose. "Signal" is what an operator observes.

| Label | Task | Touches | Signal or what it unlocks | Needs |
|---|---|---|---|---|
| **α** | The durable eligibility stamp; `TaskPriority` with `age_anchor`, `pin_order` and `order_key`; the lookup returns `pinned` for an own pinned row | the scheduler's candidacy site and lookup; `orchestrator/task_priority.py` (and its guard); fused-memory task backend and its test; `shared/task_metadata.py`; `docs/task-authoring.md`; a repair script | Row 6 and 7's stamps; after a `replace`-mode write and a re-pend, `get_task` shows the original `eligible_since`; `merge_queued` rows carry `age_secs`; the repair reports how many rows it stamped. Unlocks β, γ | 9908 |
| **β** | Verify admission orders by `order_key` with the promotion backstop (task 8261, amended) | `verify_admission_gate.py`, `verify.py`, `config.py`, the defaults script | Row 25; `verify_admission_plan` rows carry tier and age; the script prints the project's p95 | α |
| **γ** | The pool order behind `merge_pool.order`: one order function and its readers, the narrowed footprint, `bounce_cap`, the `merge_pool` config file, the three delta notes | `merge_lane/intake_buffers.py`, new `merge_lane/pool_order.py`, new `orchestrator/merge_pool_config.py` (mounted in `config.py`), `suffix_graph.py`, the three earlier PRDs | Rows 1 to 5 with `order=priority`; with `order=arrival` the recorded pick order is unchanged | α, 8528 |
| **ε** | Content id, the observation row, the cache and its rebuild, inference | new `merge_lane/pool_observations.py`, `merge_lane/pool_inference.py`; the enqueue and finalize sites; `merge_lane/types.py`; `event_store.py`; a fixture exported from real rows, dated | With both switches at their defaults, every finalized verdict writes a `merge_pool_observation` row and every `merge_queued` row carries `content_id`. Row 23. Unlocks ζ | 9908, 8528 |
| **ζ** | Head and batch selection, and shadow decisions | new `merge_lane/pool_batches.py`; wiring at the pick and the placement | Rows 19 and 20: with `mode=shadow` every batch dispatch writes a `merge_pool_decision` row; none with `mode=off` | ε, γ, 5041 |
| **ι** | The readout, the replay and the two human gates | a script under `scripts/`; two deterministic gate tasks | The readout prints from real rows: `p_pass_dispatched` against actual by bin, the refitted `false_failure` and `prior_conviction`, and a replay of recorded batches under 6.4 and 6.5 giving verifies per landing and suspect wait at p50 and p95 beside task 8528's figures. Fewer than 200 decisions with a verdict prints `insufficient rows (n of 200)`; no rows prints `no merge_pool_decision rows yet`. Gate 1 flips `order`; gate 2 flips `mode` | ζ and γ observed running (a delivered check on real rows); 8532 has read |
| **η1** | The pool drives batches and suspects: heads by kind, suspect batches, no mixing in the stack, deadline, restart rebuild, the main-red read | `merge_lane/pool_batches.py`, new `merge_lane/pool_suspects.py`, wiring at the pick, the placement and the failed-batch arm | Rows 8 to 10, 13, 14, 16, 17, 21, 22, 24 with `mode=on` | ζ, ι's second gate, 5041, 5042, 8933 |
| **η2** | Conviction: own run, elimination, the evidence record, the thrash rule, the storm limit, the readout's conviction join | `merge_lane/pool_suspects.py`, `merge_lane/types.py`, `workflow.py`, the readout script | Rows 11, 12, 15, 18; `merge_convicted` rows; the readout prints wrong convictions, or `no conviction rows` | η1 |
| **θ1** | `pinned` inside the scheduler: the walk takes pins, pin order inherited, the pin pass removed as a dispatch order, the bounded reservation kept | `scheduler.py`, `pin_reservation.py`, scheduler tests, the scoring PRD's delta note | Rows 26 and 27; on a recorded pin queue with no prerequisites involved, dispatch order is unchanged | α, 6718, 9683 |
| **θ2** | `pinned` in the vocabulary and its readers | `shared/task_priority.py`, fused-memory and dashboard validators, `shared/scheduler_state.py`, the dashboard's scheduler view | `set_task_priority_override(boost_tier='pinned')` is refused with a message naming `pinned=True`; the dashboard shows inherited `pinned` | θ1 |
| **κ** | Integration gate | a boundary-test module driving scheduler, admission and lane together | Section 7 rows 1 to 28 against the real modules, row 28 included | β, γ, η2, θ2 |

Notes for decompose:
- Every leaf that touches a function pinned in `orchestrator/tests/merge_lane_ratchet_baseline.json` names the
  authorised-raise path in its text.
- "Observed running" and "has read" are delivered checks or deterministic gate tasks, not dependency edges.
- With Leo's word at decompose: cancel 8932; amend 8261 into β; add the dated notes to 9144, 7921, 9188, 6328, 3824
  and 9636.
- Critical path to the pool driving: 9908 → α → γ (after 8528) → ε → ζ (after 5041) → ι (after 8532 and 200
  decisions) → η1 (after 5042 and 8933) → η2.

## 9. Out of scope

- Any change to the scheduler's order within a tier (D3), a promotion backstop at the scheduler, per-tier slot caps.
- Lock reservations beyond the bounded pin reservation; releasing locks before landing (ruled out by Leo).
- Deriving the high lane from the tier; choosing a verify host by priority; a verify slot reserved for suspects.
- Basing a batch off the frontier; verifying both halves of a failed batch in parallel.
- Per-test blame. Priorities per PRD. Batches larger than task 8528's table.
- The other projects' adoption beyond the defaults the script prints.

## 10. Open questions (tactical)

1. **Where the content id is computed.** Suggested: on the lane's enqueue path, to keep `workflow.py` out of ε.
   Decide in ε.
2. **Whether `merge_pool_decision` is one row per dispatch or per fill.** Suggested: per dispatch. Decide in ζ.
3. **The exact order of solo verifies when a group exceeds `max_suspects`.** Suggested: lowest in `pool_order()`
   first. Decide in η1.
4. **Whether the repair in α prefers `lock_acquired` or `task_started`.** Suggested: the earlier of the two, which is
   still at or after first candidacy. Decide in α.
5. **Whether `prior_conviction` is one value or per tier.** Suggested: one value. Decide in ι.
