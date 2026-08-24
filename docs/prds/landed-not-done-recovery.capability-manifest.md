# Capability manifest — landed-not-done-recovery

PRD: `docs/prds/landed-not-done-recovery.md` · verified against main `d8f165756b` ·
machine-readable twin: `docs/prds/landed-not-done-recovery.capability-manifest.yaml`

Mechanises G3 + G6 per leaf: every capability each task's signal asserts, bound to evidence, so a
dispatch-time architect diffs intent against substrate instead of re-deriving the check.
**68 capabilities across 8 labels — 61 PASS, 4 OPEN, 3 FAIL.** 29 mechanical `delivered_check`s
(copied into producer `metadata.delivered_checks` at `commit_planning`), 39 `manual` (recorded here,
excluded from the dispatch gate).

## The three FAIL bindings, and how each was discharged

A FAIL blocks queueing until resolved. All three are recorded rather than silently flipped to PASS,
because each was a real gate firing and the resolution changed the batch.

| # | Label | Capability | Discharge |
|---|---|---|---|
| 1 | **α** | `alpha-site-has-no-competing-queued-producer` | **α was not filed.** Task **4496** (pending/high, `dependencies=[]`, unstarted) owns the identical site with the identical remedy and strictly supersets it. θ depends on 4496 in α's place. The α block is retained in the sidecar with `task_id: null` so B13 keeps a home and 4496's implementer inherits the D6 constraint and the release-edge gap. |
| 2 | **δ** | `verdict-must-carry-a-probe-for-the-escalating-arms` | **Spec amended** — PRD correction 6bis. The Contract's `LandingVerdict` declares four fields and no `probe`, but two dispatch-gate arms feed `verdict.probe` into the L1 escalation body. δ must carry a probe mapping forward. |
| 3 | **ε** | `rejected-verdict-carries-structured-facts-for-the-escalating-arms` | Same defect, consumer side. ε must not repoint an escalating arm until δ's verdict carries the probe. |

## Why this decompose corrected the PRD

The PRD was authored 2026-08-23 against main `e0c859f566` (2026-08-22T08:01). **Task 3116 merged at
`40c39cd8ee` on 2026-08-22T20:41 — 12h40m later, and is not an ancestor of the stated baseline.** It
rewrote the exact predicate §Background argues against. Eight corrections are recorded in the PRD's
new §Post-authoring corrections; the load-bearing ones:

- `commit_effect_present_in_main` is **no longer byte-identity** — it is a threshold added-line
  survival test (0.98 aggregate / 0.90 per-file / 25-line floor). 3116's own full-corpus measurement:
  of 2,680 previously-rejected merges it now accepts 1,050 (**39.2 %**), residual **60.8 %**. D1's
  direction survives on that residual; its 95.5 % / 0.04 % arithmetic does not.
- `branch_content_in_main` is **untouched** and still byte-identity (`git_ops.py:9155`) — and it is
  the dispatch gate's third-arm **entry condition**, so B1's decay claim survives where it matters.
- `landing_evidence.py` **already exists** (1,096 lines, task 2678) with `LandingEvidenceVerdict`,
  `validate_landing_evidence` and **`branch_is_degenerate`** — which is B6's predicate, already wired
  at five production sites. δ extends a shared authority; it does not create a module.
- **D6's ratified half is right; its stated escape hatch does not exist.**
  `veto_streak_min_span_secs` is an inner condition of a filer only ever called behind
  `if site in STREAK_CHARGING_SITES` (`harness.py:9755`), so it never runs for a non-charging site.
- **D8's conclusion holds; its mechanism is wrong.** The two helpers are siblings inside
  `create_server` (`:596`), not closures inside `merge_status` — proved by `merge_request` calling
  `_git_authority_task_metadata` at `:2293`, which a closure inside `merge_status` could not do. ζ's
  surface is two tools and the second is pinned by nothing.
- **D7's fold is in-progress-gated.** `is_stranded` returns False unconditionally for any status
  other than `in-progress`, so η's whole population would read LIVE forever — B12 greens while B11
  reds. η takes an out-of-batch dependency on task **4623**.

## Scoping is load-bearing, not tidiness

Every `grep` check carries an explicit `paths:` scope, and every 0-count is paired with a
known-positive control in the same scope. Three traps this caught, all real:

| pattern | at its `paths` scope | one level wider | what satisfies the wider scope |
|---|---|---|---|
| `async def branch_work_landed` | **0** (`landing_evidence.py`) | **3** in `orchestrator/src/`, 14 repo-wide | `TaskWorkflow._branch_work_landed_on_main` (`workflow.py:13251`) — a *sixth* landing predicate one underscore from the contract's name |
| `no_op_landing` | **0** | **4** repo-wide | all four are the PRD's own prose |
| `preexisting_main_break` | **0** (`harness.py`) | **99** under `orchestrator/`, 10 files under `docs/`+`plans/` | the producers β is reacting to |
| `is_stranded_any_status` | **0** in any `.py` | **14** repo-wide | entirely PRD prose in two documents |

Four `expect: absent` checks are genuine false-before/true-after rejection bindings — the string is
present today and the work removes it: `self.git_ops.branch_content_in_main` (1),
`a later commit on main removed the deliverable` (2), `stamped done unconditionally (the task-1175
clobber` (1), `def _found_on_main_response` / `def _git_authority_task_metadata` (1 each). Two
candidates were **rejected as vacuous** and demoted to `manual`: D6's "must not add
`already_landed_gate` to `STREAK_CHARGING_SITES`" (0 today, must stay 0 — a regression guard, not an
assertion) and γ's "don't edit the `('pruned_not_landed', 'delivered_checks_withheld')` tuple" (a
considered *non*-edit is not greppable).

## Per-label bindings

Full evidence strings are in the YAML twin; this is the reviewer's summary.

### α — NOT FILED (superseded by task 4496)
Almost none of α was greenfield, and the part that was is already queued. `RecoverySite.already_landed_gate`
exists (`recovery_emission.py:149`), `_emit_recovery_disposition` has 10 live call sites
(`harness.py:9643`), and **this very gate already emits twice** (`:11593`, `:11625`). The event-type
discriminator needs no edit either — `harness.py:9791` routes anything that is not
`escalation_pinned`/`provenance_arbitration` to `recovery_left` by fall-through. One genuine gap
worth inheriting: the gate's only streak-release edge (`:11655`) sits *above* all git work on the
escalation path, so a decline signature accumulated below it is never popped — on a per-tick site
that is an unbounded tracker footprint.

### β — narrow the veto, on-main only (leaf; no prereqs)
The **only** leaf whose PRD anchors survived the drift check verbatim: `:5262-5270`, `:5281`,
`:5286`, `:5566`, `:5596`. D3's premise checks out — the two clauses are separate call sites on the
same static method, so widening the shared frozenset really would widen both.
`EscalationRef.category` is confirmed non-sentinel-populated (`task_ground_truth.py:740`), so β
widens a test over a real value, not a `''` default.

### γ — preserve the parked row (leaf; no prereqs)
Two things the PRD row omitted are load-bearing. `reconcile_landed_outbox` does
`report[disposition] += 1` **inside** the per-row `try/except` that tallies `errors`
(`merge_queue.py:6788-6797`), so an unregistered label raises `KeyError` and is silently absorbed as
an error — hence a separate capability for seeding the key to `0`. And the operator-visible half of
the signal lives in **`harness.py:11232-11244`**, which is why `harness.py` joined γ's module set
(without it the new label is counted and never displayed — a G1 orphan). Reassuringly
`reconcile_landed_task` already defaults a new label to *do-not-dispatch* (`:6719`), the right answer
for a parked landed task — but γ should state that rather than inherit it.

### δ — the producer, on shifted ground (intermediate; unlocks ε, η)
Extends an existing 1,096-line authority. B6 needs no new algorithm. The useful find is
**`merge_queue.patch_content_contained`** (`merge_queue.py:3740-3767`) — an existing production
`git cherry` containment helper that *is* δ's attribution primitive — but `merge_queue.py:46-49`
imports *from* `landing_evidence`, so a top-level reverse import is a cycle; use the house lazy
import (precedent `merge_queue.py:816`). The enumeration deliverable came out far smaller than D1
implies: **3 production sites**, not an open-ended list. And `format_unattributed_landing_detail`
renders the literal `'Unrecognized reason code: <x>'` for anything missing from
`_REASON_EXPLANATIONS`, so the `no_citation` → `no_attribution` rename is a *registration* change.

### ε — repoint the consumers (leaf; prereqs δ, 4496, 4498)
Re-scoped to **re-pointing only**: 4496 owns the four `harness.py` sites and 4498 owns the two
`escalation/server.py` sites, both unstarted, so ε depends on them rather than contending. The
task-1175 correction is **three** copies, not the two the PRD names. The Tier-3.5 repoint travels
through the existing runtime-only reverse import (`server.py:3441-3445`), so no new layering.

### ζ — extract the git-authority tier (intermediate; prereq 4498; unlocks η)
The declared signal holds up better than expected — the suite constructs a real server through
`create_server(...)` nine times with a real-git fixture. One caveat re-verified at `:244-247`: it
deliberately avoids `assert_called_once_with`, so a **signature** change is invisible to it. Harmless
for ζ, and precisely why it will not catch ε's repoint. Module placement is OPEN;
`escalation/src/escalation/git_authority.py` is recommended because `orchestrator/pyproject.toml:20`
already declares `escalation` as a workspace dependency, so η can import it statically.

### η — the reachable actor (leaf; prereqs δ, ζ, γ, 4623)
The disposition read path is concrete and needs no new work — `emit_recovery_event` → `EventStore`
→ `get_scheduler_events` → `dashboard/data/scheduler.py:112-116`, which filters on event *type*, so a
new `RecoverySite` is visible for free. The writer exists too. D7 is where it breaks, hence the 4623
edge. **Boundary with task 3539 — ruled 2026-08-24 (gate 4673 / `esc-4673-1`), no conflict.** The
"same population, incompatible remedies" framing recorded here was a misreading and is retracted
(PRD corrections 9-10, landed in `f34b4bce0b`). The populations are disjoint on status: 3539's rows
are all keyed `IN_PROGRESS`; η owns `pending`/`merge-deferred`, filtered out at `harness.py:4959-4960`
and never reaching `_RECOVERY`; landed `blocked` is β's. That proof binds 3539 *structurally* — η's
own actor sits outside `_RECOVERY` and `_RECONCILE_SWEEP_STATUSES`, so on η's side the boundary is
convention with no automated check. The *"something re-pended it"* question is **answered**: it is the
mark-done applier's own reject arm (`harness.py:5715-5723`), so η need not hunt a re-pender — but the
same decayed-evidence reject is why δ is a hard prereq. Two claims made when this was ruled do **not**
survive re-measurement: `done` is not an absorbing state (`shared/task_transitions.py:222`), and the
3539→β adjacency is not a handoff that completes — a converted task arrives blocked still pinned on
`task_failure`, which β's narrowed veto does not admit, so it rests there until a human closes the
record. Tasks 4645 and 3539 were corrected accordingly on 2026-08-24.

### θ — the integration gate (leaf; prereqs β, γ, ε, η, 4496)
A genuine DAG sink. Two rows are currently vacuous-by-construction, which is the honest false-before
state: B7 cannot fire because `preexisting_main_break` appears zero times in `harness.py`, and
B4/B6's reason codes appear zero times outside PRD prose (controls 3 and 69 prove the greps reach the
files). The real-git fixture pattern exists as ~10 independent per-file copies with no shared helper —
θ should reuse one rather than mint a tenth.
