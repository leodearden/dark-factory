# PRD — durable, discriminating `merge_status`: the not-landed half

**Status:** active · authored 2026-08-28 · approach **B + H** (contract + two-way boundary tests)

**Code anchors** are cited by **symbol**, never `path:line`. Measurements in §Background were taken
2026-08-28 against main `516ab65125` on the live `dark_factory` stores
(`data/orchestrator/runs.db`, `data/orchestrator/merge_queue.json`) and are dated **provenance** —
do not re-measure them into this document later.

---

## Goal

`merge_status` stops answering `unknown` for propositions it can actually decide, and its answer
survives orchestrator restarts and multi-day spans.

What an operator or agent observes after this PRD:

- Polling a merge that finalized `blocked` in a **previous run** returns
  `state: 'stale_record'` with a labelled `last_outcome` naming the run, the timestamp and the
  reason — instead of `unknown`.
- Polling a request that is sitting in the **durable journal** after a restart returns
  `state: 'journaled'` — instead of `unknown`.
- Polling something genuinely never submitted returns `state: 'no_record'` — distinguishable, for
  the first time, from "we lost the record".
- `unknown` survives with exactly one meaning: **the probe itself failed**.
- Every response carries a machine-readable `reason` and, whenever any durable record exists, a
  `last_outcome` block — so the three consuming skills read a field instead of re-deriving the
  answer from `git merge-base --is-ancestor` exit codes.

## Background

### The measured defect

`merge_status`'s lookup order is: live snapshot → retention ring → event store → git-authority
(Tier 3.5) → `{state: 'unknown', hint: 'check git log main'}`. Measured 2026-08-28:

| Tier | Survives restart? | Survives days? | Can answer |
|---|---|---|---|
| live snapshot (`SpeculativeMergeWorker.snapshot`) | ✗ RAM | ✗ | all states |
| retention ring (`TerminalOutcomeRetention`) | ✗ RAM — **and never constructed in production** | ✗ | all states |
| event store (`EventStore.latest_merge_finalized`) | durable on disk, **filtered `run_id = self.run_id`** | ✗ | all states |
| git-authority (Tier 3.5) | ✓ run-independent | decays | **`done` only** |
| Tier 4 | — | — | `unknown`, static hint |

**There is exactly one run-spanning tier and it can only ever prove `done`.** Every not-landed
proposition therefore degrades to `unknown` the moment the run that observed it ends.

Numbers, on the live store:

- `runs.db` holds **2,620 `merge_finalized` rows across 228 `run_id`s spanning 80 days**
  (2026-06-09 → 2026-08-28). The current run can see **17 of them — 0.65%**.
- **885 of those rows are non-`done`** and permanently unreachable once their run ends:
  `blocked` 699, `superseded` 86, `conflict` 35, `unknown_branch` 17, `abandoned` 11,
  `wip_halted` 3.
- Run duration in the current regime: **median 8.26 h over the last 7 days** (20 runs), median
  8.86 h over 30 days — the 8 h fleet-deploy clock. RAM tiers flush ~3×/day; the durable tier's
  memory window **equals one run**. "Several days" is ~9 flushes.
- `data/orchestrator/merge_queue.json` — the durable journal `recover_pending_merges` reads at
  startup — is **never consulted by `merge_status`**, so a durably-journaled request reads
  `unknown` in the restart→recovery window.

So `unknown` conflates at least five distinct propositions: *never submitted* · *submitted and
finalized in a previous run* · *journaled and awaiting recovery* · *a coalesce/train id that
resolves through no tier by construction* · *landed but unattributable*.

### The consumer damage, measured

The three skills that poll `merge_status` have grown a hand-maintained decision tree whose sole
purpose is to re-derive what the server declined to say. Across `skills/merge-queue/SKILL.md`,
`skills/unblock/SKILL.md` and `skills/unblock-low-risk/SKILL.md` — 20,108 words total — there are
**44 lines mentioning `unknown`** and **59 lines of `rc=0`/`rc=1`/`rc=128` ancestry
disambiguation**: per-arm exceptions, 20-minute wall-clock ceilings, and rules of the form *"this
block's resubmit line does not apply to the `poll_by == 'branch'` arm"*.

That prose is also **wrong**. `skills/merge-queue/SKILL.md` explains `unknown` as *"the orchestrator
restarted and the retention ring no longer holds this request"* — but no production path ever
constructs a `TerminalOutcomeRetention` (task 3149). The consumers are navigating by a mechanism
that does not exist; the real causes are the RAM snapshot dying and the `run_id` filter.

This is INV-1 `contracts-machine-checked` (a contract living in prose) and INV-5
`no-lockstep-duplication` (three copies of one decision tree, one of which — task 4269 — has
already drifted out of sync).

### The failure mode this produces

An agent reads `unknown`, concludes "not landed", and re-submits work that is already on `main`.
Observed on task 6873 (2026-08-28): a rebase-landing read as not-landed three separate ways
(`merge_status` → `unknown` on all of `request_id`/`task_id`/`branch`; `merge-base --is-ancestor`
→ false; a `Merge task/6873 into main` commit that is not itself on `main`).

### Why task 1750 does not cover this

Task 1750 was the "run-spanning landed-check (cross-run `done` resolution)", filed DEFERRED/GATED
on *"pursue ONLY if α (1748) and β (1749) leave a residual unknown after an orchestrator restart"*.
It closed as subsumed because git self-resolution covers the restart case. **That reasoning is
correct for the landed case and only the landed case.** It is the ruling that left the not-landed
half unowned, and this PRD re-opens exactly that residual — without disturbing 1750's conclusion
about landings.

## Sketch of approach

**`merge_status` conflates two propositions into one field.**

1. *Liveness* — "is a merge in flight for this branch right now?" Correctly run-scoped.
2. *History* — "what is the last thing we know happened?" Correctly run-spanning.

Today a single `state` answers (1) and collapses to `unknown` when it cannot, discarding (2)
entirely even while 80 days of it sit indexed on disk. The fix separates them.

**This is a read-path PRD.** Nothing that needs to be durable is missing: the `merge_finalized`
payload already carries `request_id`, `branch`, `state`, `snapshot_tip`, `merge_sha`,
`superseded_by`, `generation`, `reason`, `landed_via_chain`, and the journal is already written and
already read at startup. One field is added to the finalize payload (`absorbed_request_ids`); no
new store is introduced. That is deliberate — a second durable outcome store would violate INV-9
`one-fact-one-home`, and this PRD *removes* a competing home rather than adding one.

## Resolved design decisions

**D1 — `state` is partitioned into outcome states and epistemic states.**

- **Outcome states** (unchanged vocabulary): `queued`, `verifying`, `gate`, `finalizing`, `done`,
  `conflict`, `blocked`, `abandoned`, `superseded`. Sourced **only** from current-run evidence or
  from git. A prior-run row may never set one.
- **Epistemic states** (new): describe what the *record* says, never what the merge did.
  - `no_record` — nothing anywhere, and git cannot prove a landing. Genuinely never heard of it.
  - `stale_record` — a durable prior-run terminal row exists; read `last_outcome`.
  - `journaled` — a durable in-flight journal entry exists but no live worker entry (the
    restart→recovery window).
  - `unknown` — **retained, and narrowed to one meaning: the probe itself failed** (git raised, the
    event store raised). Never again "we have no record".

**D2 — history never sets an outcome state.** The invariant, machine-checked:

```
state ∈ OUTCOME_STATES  ⟹  evidence is current-run or git
state ∈ EPISTEMIC_STATES ⟹  state names the record, not the merge
last_outcome may come from any run, and is ALWAYS labelled (run_id, finished_at, is_current_run)
```

This preserves in full the property `latest_merge_finalized`'s `run_id` filter exists to protect —
*"prevents `merge_status` from silently surfacing stale prior-run terminal outcomes"* — while making
the response informative. A prior run's `conflict` can never appear in `state`; it appears in
`last_outcome` wearing its run id and timestamp. `stale_record` is an assertion about our records,
not about the merge.

*Rejected alternative — a recency window* (surface a prior-run outcome in `state` when no newer
activity exists for that branch): the freshness comparison is itself an uncorroborated inference
about live state, which is the INV-3 `corroborate-before-acting` failure shape. *Rejected
alternative — drop the filter, caller beware*: re-opens the hazard and pushes the judgment back
into the skills' prose, which is the thing this PRD exists to retire.

**D3 — the state vocabulary gets one home, with a drift guard.** Extending `state` while three
skills carry hand-maintained vocabulary lists in prose is, unmitigated, a deliberate instance of
INV-5 — the exact shape task 4269 already records drifting. So the vocabulary moves to
`shared/src/shared/merge_state.py` (enum + `OUTCOME_STATES` / `EPISTEMIC_STATES` / `TERMINAL_STATES`
partitions), following the established convention of `shared/src/shared/task_statuses.py`, which
`plans/task-status-authority-prd.md` used to collapse four duplicated status-set copies into one
module. The escalation server's wire `Literal` derives from it, and a drift guard modelled on
`scripts/tests/test_design_invariants_consistency.py` (task 3802 — which already pins a
`skills/`-family markdown file against a normative source) cross-checks the three SKILL.md
vocabulary lists. This is what converts "extend `state`" from a hazard into a one-time investment,
and it closes 4269's class rather than widening it.

**D4 — `last_outcome` is an enrichment, not a tier.** It is attached whenever a durable row exists,
independent of which tier set `state` — so even a `state: 'done'` response carries the history. One
pass, one place, no per-tier duplication.

**D5 — tier order, by strength of claim.**

1. live snapshot → outcome states
2. event store, **current run** → outcome states
3. git-authority → `done` *(consumed from task 4649's extracted module, under task 4647's
   non-decaying `branch_work_landed` contract — never re-derived here)*
4. durable journal (`MergeQueueStore`) → `journaled`
5. event store, **cross-run** → `stale_record` + `last_outcome`
6. `no_record` — or `unknown` when a probe raised

`done` outranks `journaled`: a proven landing is terminal truth, a journal entry is only a pending
expectation. Cost is immaterial — measured below.

**D6 — the retention ring is deleted, not wired.** Task 3149 poses this as an explicit
wire-it-vs-delete-it judgement call and this PRD rules **delete**. A cross-run event-store read
subsumes the ring's Tier-2 purpose durably rather than for a few RAM-resident minutes, and
`absorbed_request_ids` (D7) subsumes `record_alias` — the one capability 3149 flags as a genuine
gap — likewise durably. Keeping an unwired RAM ring alongside a durable answer would be a second
home for one fact (INV-9). Deleting it also removes the docstrings the skills are currently
navigating by.

**D7 — alias resolution replaces `record_alias`, durably.**

- A **superseded member** already emits its own `merge_finalized` row carrying `superseded_by`, so
  it resolves cross-run today, once tier 5 exists.
- A **`coalesce-*` train id** is never any row's `request_id`. It resolves by reverse index: find
  rows whose `superseded_by` equals the train id, then follow a member's branch. Read path only.
- An **attach/door-coalesced loser** never gets its own row (its future resolves in-process via the
  attach mirror). The primary's `merge_finalized` payload gains `absorbed_request_ids: [...]` — the
  single new write in this PRD — so a loser's id resolves to the primary's outcome, cross-run and
  across restarts.

**D8 — `hint` becomes derived, not constant.** `_MERGE_STATUS_UNKNOWN_HINT` is a module constant
today. It becomes a function of `reason`, so the prose an agent reads names the actual situation.
`reason` is the machine-readable field; `hint` is its human-readable projection, generated from one
table, never hand-written per call site.

## Pre-conditions for activating

- **Task 4649** (extract the git-authority tier into an importable module) — same file pair as this
  PRD's read-path work; must not run concurrently. Hard prerequisite for **β**.
- **Task 4498** (Tier-3.5 call sites) — same file. Hard prerequisite for **β**; see the seam table,
  it also needs a paired amendment.
- **Tasks 4647 / 4648** (`branch_work_landed`, non-decaying contract) are *not* blocking: this PRD
  consumes whatever the git tier returns and never re-derives a landing verdict. If they land
  first, tier 3 simply gets better. Boundary row **B5** pins the consumption, not the derivation.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/landed-not-done-recovery.md` (4647, 4648, 4651, 4652, 4646, 4496) | consumes | the `done` half — `branch_work_landed`, Tier-3.5's landing verdict | **landed-not-done-recovery** | wired (B5 pins consumption only) |
| Task **4649** (extract git-authority tier) | consumes | the extracted importable module | 4649 | queued — hard prereq of β |
| Task **4498** (Tier-3.5 call sites) | **contested — resolved here** | 4498 scope item 2 is literally *"decide whether `unknown` remains the right degradation"*. **That decision is this PRD's (D1/D2).** 4498 retains the `delivered_checks` wiring only. | **this PRD** owns the vocabulary; 4498 owns the wiring | **needs paired amendment at decompose** |
| Task **3149** (retention ring dead code) | **absorbed** | ruled DELETE by D6 | this PRD (ε) | closes 3149 |
| Task **2932** (thread retention into recovery) | **closed by** D6 | there is no ring to thread | this PRD (ε) | closes 2932 |
| Task **4269** (`unblock-low-risk` omits `superseded`) | **closed by** D3 | the drift guard makes the omission a test failure | this PRD (α) | closes 4269 |
| Task **3860** (finalized entries never retired → phantom `queued`) | adjacent | Tier-1 *liveness correctness*; this PRD does not touch the live snapshot's contents | 3860 | unchanged |
| Task **3047** (recovery drops PRD-branch entries) | adjacent | `recover_pending_merges` branch reconstruction | 3047 | unchanged — but tier 4 makes its symptom visible as `journaled`-absent rather than `unknown` |
| Task **3015** (stale in-flight slot → `unknown` on resubmit) | adjacent | `InFlightMergeRegistry` slot lifecycle | 3015 | unchanged |
| Task **4589** (merge-phase task with `request_id=null` in all tiers) | adjacent | its detection recipe gets sharper: `no_record` is a positive signal where `unknown` was ambiguous | 4589 | unchanged |
| Task **3846** (recon's own rebase-landed misread) | adjacent | a different consumer of the same root shape; recon does not call `merge_status` | 3846 | unchanged |
| `plans/task-status-authority-prd.md` | pattern precedent | `shared/task_statuses.py` as the one-home vocabulary convention D3 follows | n/a | precedent only |

## Contract (B + H)

Home: `shared/src/shared/merge_state.py`. Pure vocabulary module — defines the enum and its derived
partitions only, and re-wires no call sites (the `task_statuses.py` convention).

```python
class MergeState(enum.StrEnum):
    # --- outcome states: what the merge did. Current-run evidence or git only. ---
    queued = 'queued'; verifying = 'verifying'; gate = 'gate'; finalizing = 'finalizing'
    done = 'done'; conflict = 'conflict'; blocked = 'blocked'
    abandoned = 'abandoned'; superseded = 'superseded'
    # --- epistemic states: what our records say. Never assert a merge outcome. ---
    no_record = 'no_record'        # nothing anywhere; git cannot prove a landing
    stale_record = 'stale_record'  # durable prior-run terminal row exists; read last_outcome
    journaled = 'journaled'        # durable journal entry, no live worker entry
    unknown = 'unknown'            # NARROWED: the probe itself failed

OUTCOME_STATES: frozenset[MergeState]
EPISTEMIC_STATES: frozenset[MergeState]
TERMINAL_STATES: frozenset[MergeState]   # what a poll loop may stop on


class MergeSubmitStatus(enum.StrEnum):
    """`merge_request`'s OWN vocabulary — distinct from `merge_status`'s `state`."""
    queued = 'queued'; attached = 'attached'; already_merged = 'already_merged'
    unknown_branch = 'unknown_branch'; failed = 'failed'; superseded = 'superseded'
```

**Two vocabularies, not one.** `merge_request`'s response `status` and `merge_status`'s `state` are
different vocabularies that the consuming skills' terminal-set lists currently interleave
(`skills/merge-queue/SKILL.md`'s "Terminal at submit time" set mixes `already_merged` and
`unknown_branch` — submit-only — with `done`, `conflict`, `blocked` — both). A guard pinning those
lists against a poll-only enum would fire falsely, so α homes **both** vocabularies and pins each
list against the right one. Homing only the poll half would leave the guard either wrong or
scoped-out, which is the half-measure D3 exists to avoid.

Response shape (additions to today's; existing fields unchanged):

```python
{
  'state': MergeState,          # partitioned per D1
  'reason': MergeReason,        # ALWAYS present on an epistemic state; enum, one home
  'source': Literal['live', 'event_store', 'git', 'journal', 'event_store_cross_run', 'none'],
  'last_outcome': {             # present whenever ANY durable row exists — even beside state='done'
      'state': MergeState,      # an OUTCOME state
      'run_id': str,
      'finished_at': str,       # ISO-8601
      'is_current_run': bool,   # False ⟹ this is history, not present truth
      'reason': str | None,
      'merge_sha': str | None,
      'superseded_by': str | None,
  } | None,
  'hint': str,                  # DERIVED from reason via one table — never a constant
}
```

**Invariants** (each pinned by a boundary row):

- **I1** — `state ∈ OUTCOME_STATES ⟹ source ∈ {live, event_store, git}`. A cross-run or journal
  read can never produce an outcome state. *(B1)*
- **I2** — `last_outcome.is_current_run is False ⟹ last_outcome.state never equals state`. History
  and present truth are never the same assertion. *(B1, B2)*
- **I3** — `state == unknown ⟹ reason == 'probe_failed'`. Every other former `unknown` has a
  discriminating state. *(B8)*
- **I4** — every `state` value emitted by the server is a member of `shared.merge_state.MergeState`
  and every `status` a member of `MergeSubmitStatus`; each terminal-set list in the three SKILL.md
  consumers is pinned against whichever of the two vocabularies it actually enumerates. *(B9)*
- **I5** — the landing verdict is read, never derived: this module makes no `is_ancestor` /
  `find_merge_marker` / patch-id call of its own. *(B5)*

## Boundary-test sketch (B + H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| **B1** | Prior-run `blocked`, polled after restart. **THE REGRESSION PIN — must not be waived.** | a `merge_finalized` row with `state='blocked'` under a run_id ≠ current; branch not on main | `state == 'stale_record'`, `reason == 'prior_run_terminal'`, `last_outcome.state == 'blocked'`, `is_current_run is False`. **`state` is never `'blocked'`.** |
| **B2** | Same branch resubmitted this run after a prior-run `conflict` | prior-run `conflict` row + a live current-run entry | `state` reflects the current-run entry only; `last_outcome` still shows the prior run, labelled. The prior outcome never leaks into `state`. |
| **B3** | Journaled, not yet recovered | entry present in `merge_queue.json`, absent from the live snapshot, no finalize row | `state == 'journaled'`, `source == 'journal'` |
| **B4** | Never submitted | no row, no journal entry, no branch | `state == 'no_record'`, `reason == 'never_submitted'`, `last_outcome is None` |
| **B5** | Rebase landing (the 6873 shape) | branch tip not an ancestor of main; work present on main | `state == 'done'`, `kind == 'found_on_main'`, `source == 'git'`, **and this module made no landing-primitive call of its own** (assert the non-call, per I5) |
| **B6** | `coalesce-*` train id | member rows carry `superseded_by == <train id>` | resolves via the reverse index; **not** `no_record` |
| **B7** | Attach-coalesced loser's `request_id`, cross-run | primary's row carries `absorbed_request_ids` including the loser | resolves to the primary's outcome, after a restart |
| **B8** | Probe failure | git invocation raises | `state == 'unknown'`, `reason == 'probe_failed'` — the only surviving `unknown` |
| **B9** | Vocabulary drift | a member added to `MergeState` without updating all three SKILL.md lists | the drift guard **fails** — seeded-violation fixture, mirroring `docs/legibility/design-invariants-fixtures.md` |
| **B10** | Cross-run lookup cost | a store of ≥300k events / ≥2,500 `merge_finalized` rows | p99 < 50 ms. *Achievability basis: measured 2026-08-28 at 0.0 ms cross-run vs 0.2 ms run-scoped on 364,134 rows / 161 MB; `idx_events_type` already covers the predicate, and `merge_finalized` is 0.7% of the table. The bound carries ~3 orders of magnitude of margin over the measured floor.* (INV-8 pin) |

Rows B1–B8 face **both** sides of the seam: each asserts a server response *and* the consuming
skill's branch on it.

## Decomposition plan

Labels are Greek; task ids are assigned at decompose. Sizing per the overlay's provisional bands
(~300–1,500 LOC, ≤10–12 files).

- **α — Vocabulary one-home + drift guard.**
  Modules: `shared/`, `escalation/`, `skills/`, `scripts/tests/`.
  Lands `shared/src/shared/merge_state.py` carrying **both** vocabularies (`MergeState` and
  `MergeSubmitStatus`), derives the server's wire `Literal`s from them, updates the three SKILL.md
  **vocabulary lists** (mechanical list edits only — δ owns the reasoning prose), and lands the
  drift guard plus its seeded-violation fixture.
  *Observable signal:* adding a member to `MergeState` without touching all three SKILL.md lists
  makes `scripts/tests/test_merge_state_vocabulary_consistency.py` **fail** (B9) — a rejection
  assertion, bound by the seeded fixture.
  *Prereqs:* none. **Closes 4269.**

- **γ — Cross-run reads + alias resolution.**
  Modules: `orchestrator/`.
  `EventStore.latest_merge_finalized` gains an explicit cross-run mode returning the row **with its
  `run_id`**; adds the `superseded_by` reverse index for train ids; adds `absorbed_request_ids` to
  the `merge_finalized` payload at the finalize site.
  *Observable signal:* a scripted `latest_merge_finalized(..., cross_run=True)` against the live
  store returns a prior-run row carrying its own `run_id`, where today it returns `None` (B7's data
  half).
  *Prereqs:* none. Runs in parallel with α (disjoint files).

- **β — The read-path rebuild.**
  Modules: `escalation/`.
  Tier reorder per D5, epistemic states, `reason`, `source`, derived `hint`, always-on
  `last_outcome` enrichment (D4), journal tier, alias resolution wired.
  *Observable signal:* `mcp__escalation__merge_status(task_id=…)` for a task whose merge finalized
  `blocked` in a previous run returns `state: 'stale_record'` with a populated `last_outcome` —
  where today it returns `unknown` (B1, B2, B3, B4, B8).
  *Prereqs:* α (vocabulary), γ (the cross-run read must exist before β can emit `stale_record`),
  **out-of-batch 4649 and 4498** (identical file; must serialize).

- **δ — Collapse the skills' decision trees onto `reason`.**
  Modules: `skills/`.
  Rewrites the `unknown`-handling prose in `merge-queue`, `unblock` and `unblock-low-risk` to branch
  on `reason`, deleting the per-arm rc=0/1/128 reconstruction and the incorrect retention-ring
  explanation. Net deletion.
  *Observable signal:* the count of `rc=`-disambiguation lines across the three files falls from 59
  to 0 for the `unknown` path, and each file's `unknown` handling reduces to a `reason` switch —
  verified by the α guard plus the boundary rows' consumer half.
  *Prereqs:* α, β.

- **ε — Delete the retention ring.**
  Modules: `orchestrator/`, `escalation/`.
  Removes `TerminalOutcomeRetention`, the dead Tier-2 read, `_get_terminal_retention`, the
  `retention=` parameters, `record_alias`, `forget`, and the docstrings describing production
  behaviour that never existed. Mostly deletion.
  *Observable signal:* `grep -r TerminalOutcomeRetention` returns nothing outside git history, and
  the full merge-lane suite stays green.
  *Prereqs:* β (Tier 2 must be gone from the read path first), γ (shares `merge_queue.py`).
  **Closes 3149 and 2932.**

- **ζ — Integration gate: the boundary suite B1–B10 green as one suite.** *(leaf)*
  Modules: new tests under `escalation/tests/` and `orchestrator/tests/`.
  Lands the full matrix against a real git fixture and a **simulated restart** (a fresh `run_id`
  over a populated store), exercising both the server response and the consuming skill's branch.
  *Observable signal:* the suite passes as one suite, including B1 (the anti-staleness pin) and B10
  (the INV-8 latency pin).
  *Prereqs:* β, γ, δ, ε, and out-of-batch 4649.

**Same-file serialization** (load-bearing under the narrow-file-lock model):
`escalation/server.py` — 4649 → 4498 → β → ε. `orchestrator/merge_queue.py` — γ → ε.

**G7 walk (advisory, author mode).** INV-1 and INV-5 are the defect being retired (D3). INV-2
`structured-facts-at-failure` is what `reason` + `last_outcome` deliver — Tier 4 currently discards
everything it knows. INV-9 `one-fact-one-home` drove D6 (delete the competing RAM home) and the
choice to read existing rows rather than build a projection table. INV-3 `corroborate-before-acting`
is why the recency-window alternative was rejected. INV-6 `status-matches-liveness` is enforced by
the D1 partition. INV-8 is pinned by B10 and measured clear. **No waivers.**

## Out of scope

- **The landed half.** `docs/prds/landed-not-done-recovery.md` owns everything about proving work is
  on `main` — patch-id attribution, non-decaying evidence, the periodic reconciler. This PRD
  *consumes* that verdict (I5, B5) and must never grow a second landing authority.
- **`get_merge_queue`.** The dashboard's merge surface reads that tool, not `merge_status`; it is
  equally RAM-only and equally restart-blind. A real gap, deliberately not folded in — recorded so
  it is not lost.
- **Live-snapshot *contents*.** Phantom `queued` entries (3860), stale in-flight slots (3015) and
  dropped PRD-branch recovery (3047) are Tier-1/recovery correctness. This PRD changes which tiers
  are consulted and what the response says, not what the live snapshot holds.
- **Making the journal itself more complete.** Tier 4 reads `merge_queue.json` as it is; 3047 owns
  what goes into it.
- **Recon's own landed-detection** (3846) — a different consumer that does not call `merge_status`.

## Open questions (tactical)

1. **Where the journal tier reads from** — a fresh `MergeQueueStore(path)` per call, or a
   harness-mounted instance. **Suggested:** harness-mounted, mirroring the existing
   `_get_merge_worker` accessor convention. Decide in β.
2. **Whether `no_record` needs a compatibility window** emitting both it and `unknown`.
   **Suggested:** no — the α drift guard makes the flip atomic across all three consumers, which is
   the whole point of D3. Decide in α.
3. **Full `MergeReason` membership** beyond the five named (`never_submitted`, `prior_run_terminal`,
   `journaled_not_live`, `coalesce_train_id`, `landing_unattributable`, `probe_failed`). Decide in α.
4. **Whether `absorbed_request_ids` should be backfilled** for existing rows or only written going
   forward. **Suggested:** forward-only; the reverse index covers superseded members historically,
   and door-coalesced losers are a narrow class. Decide in γ.
