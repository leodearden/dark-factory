# Triage of the 23 reviewer findings the off-contract verdict shape dropped

**Task 5430** · escalation `esc-2896-10` · triaged 2026-09-14T16:19:32Z ·
branch `task/5430` · branch base `99ab62335a` · audit code at
`907353ec41ce9944567ab1bf9cf0bf4626508e91`

Between 2026-07-19 and 2026-08-10, `reviewer_comprehensive` emitted verdicts
whose `verdict.issues[]` entries used an off-contract shape: `severity` outside
`{blocking, suggestion}`, and `file`+`line` instead of `location`. Both review
gates key on the contract fields, so those findings were classified as
suggestions AND skipped by the in-scope filter — the amendment gate never fired
and no implementer ever saw them. **They were never judged wrong. They were
never read.** The mechanism and its durable guard belong to the companion task;
this artifact is the residue, and the question it answers is per-finding: is
each one (a) already fixed incidentally, (b) still a live defect on current
main, or (c) not a defect / no longer applicable?

Every number below is cited from `report.json` in this directory. Every
derivation and measurement is in `provenance.json`. The 14 verdict files the
triage read are frozen byte-for-byte in `corpus/`, and that copy is now the
only committed record — see **The evidence was perishable** below, which is the
most load-bearing thing this task learned.

## The answer

**7 (a) · 15 (b) · 1 (c).** Fifteen of twenty-three findings — nearly
two-thirds — describe defects that are still live on main seven weeks after
being emitted and never read. Fourteen follow-up tasks were filed; the
fifteenth was deliberately not re-filed, because an open task already tracks it.

| severity as emitted | (a) fixed | (b) live | (c) n/a |
|---|---|---|---|
| high | 1 | 3 | 0 |
| major | 1 | 0 | 0 |
| moderate | 0 | 0 | 1 |
| medium | 5 | 12 | 0 |

`python scripts/audit_offcontract_review_findings.py --validate
docs/offcontract-review-finding-triage-2026-09-14/report.json` exits **0** with
an empty violation list: all 23 roster ids dispositioned exactly once, each
with a non-empty note, every (b) citing a follow-up, and no ticket hung off an
(a) or (c). That is deliberate — "we accounted for all 23" is a structural
property of a data file, so it is enforced rather than asserted. The
a/b/c judgements themselves carry no test, because whether a finding is live is
a judgement over code that has moved for seven weeks; a test encoding one would
pin a conclusion rather than a behaviour.

## Per-finding dispositions

| finding | sev | disp | reading |
|---|---|---|---|
| `2896-correctness-0` | high | **a** | storm dedup key fixed by `b57f8c1402` — an *evil merge*; 3 tests now pin it |
| `3031-integration-gap-0` | medium | **a** | `report_ready_to_merge` is in `_PLAN_CREATOR_TOOLS` now; exit reachable |
| `3031-correctness-1` | medium | **b** | "blocked + no escalation is LEAVE-shaped" still asserts the inverse of the sweep |
| `3041-correctness-0` | high | **b** | 2-of-6 tombstone coverage unchanged; 1 of 3 overclaiming texts fixed |
| `3041-architectural-coherence-1` | medium | **b** | stage2 still mandates the narrative the pool evicts first |
| `3064-efficiency-0` | medium | **b** | description **grew again**, 8,328 → 9,376 chars |
| `3075-correctness-0` | high | **b** | ASSIGNED preserve still unbounded; `--disk-pressure` still cannot override |
| `3142-correctness-0` | high | **b** | reproduced at runtime; already tracked by task 4205 — **not re-filed** |
| `3142-correctness-1` | medium | **b** | `'Merged PR #4521'` → task 4521, reproduced byte-identical |
| `3142-correctness-2` | medium | **b** | `git rev-parse` still synchronous on the event loop |
| `3308-contract-drift-0` | moderate | **c** | the PRD caught up: task 4088 annotated both rows |
| `3340-correctness-0` | medium | **b** | streak escalation still edge-triggered on `!=` |
| `3340-test-coverage-1` | medium | **b** | the level-triggered probe still has no caller |
| `3340-correctness-2` | medium | **b** | path divergence intact, but **latent** — gated on the missing binding |
| `3363-correctness-0` | major | **a** | task 3442 fixed it, more thoroughly than asked |
| `3363-design-1` | medium | **a** | scoped `internal_error` is cacheable now |
| `3367-correctness-0` | medium | **b** | gate still checks `test.rc != 0`, not the test leg's classification |
| `3367-correctness-1` | medium | **b** | merge-queue infra routing still keys on the aggregate category |
| `3453-correctness-0` | medium | **b** | `find_spec` canary; comment and warning still claim importability |
| `3453-blast-radius-1` | medium | **b** | `--setenv=PYTHONPATH` still unit-wide |
| `3454-correctness-0` | medium | **a** | capacity markers promoted to `shared/cap_markers.py` (task 3483) |
| `3679-test-coverage-0` | medium | **a** | the cited pin test exists now — **recovered by another route** |
| `3757-correctness-0` | medium | **a** | duplicate-uuid check moved out of the manifest-blame clause |

## The five priority findings

The task's own suggested order was the four `high` plus the one `major` first,
because those are the reason it was worth doing. Three of the five are live.

**2896 (a).** The task description asserted this was fixed by hand in
`b57f8c1402`. Verifying rather than accepting it found something the
description could not have known: `b57f8c1402` is a **merge commit**, and the
fix is an *evil merge* — `task_id=entity_uuid` appears in neither parent's tree
(parent1 still has `task_id=''` with the `entity_uuid` lookup; parent2 has no
such function at all). It exists on main solely through that merge's by-hand
conflict resolution, which is exactly consistent with "fixed by hand" and also
means `git log -S'task_id=entity_uuid' origin/main` finds **nothing** without
`-m`. A fix that is invisible to the ordinary archaeology tool is worth knowing
about. Three tests now pin the write-key/read-key agreement.

**3041 (b).** The code measurement reproduces exactly — still 2 of 6 recon Mem0
delete paths wired to `record_mem0_deletion_tombstones`. One of the three
overclaiming texts was correctly fixed (task 4421), one is tracked (task 4472),
and one is not: the live `get_memory_by_id` MCP tool description still hedges
tombstone absence on TTL alone. **The finding is now worse than when it was
written, in a way it could not have predicted.** Task 4421's fix means
`prompts/stage2.py` now tells an agent that "several Mem0 delete paths write no
tombstone at all", while the tool description that same agent reads says
absence means only that a tombstone expired. Two live agent-facing texts that
contradict each other. The finding's own stated mitigation has also expired: it
noted `render_cycle_summary_section` was "not yet imported by any prompt file";
it is now.

**3075 (b), narrowed.** Every mechanical fact is verbatim intact, including one
the finding under-stated: the gate cannot see `updated_at` even in principle,
because `lib_lane_state.sh` exports exactly three variables and none is an age.
"Permanently" is now false for one case — task 2891's
`_reclaim_terminal_lane_records` heals a stuck record whose task is terminal.
But the core ENOSPC scenario survives intact, and the reason is worth stating
plainly: **the healing pass releases through the same OSError-swallowing write,
so it fails while the disk is still full.** The self-healing path is blocked by
the very condition it would heal, during exactly the window an operator is
running `reclaim --disk-pressure`. An age bound now exists in the tree
(`_stale_lane_assignment_census`) and terminates in a digest line.

**3142 (b).** Reproduced at runtime rather than inferred, and carried one step
further than the original: `build_unverified_flag` was also run, confirming
end-to-end that a truthful "Filed follow-up task 3200" yields
`tag='unverified_claim'`. Already tracked by task 4205 as its CLASS 1, and
independently quantified by `docs/unverified-completion-claim-sweep-2026-08-11`
at 125–128 of ~166–172 production mismatches, none terminal. **No duplicate
filed.**

**3363 (a).** Fixed by task 3442, and further than the review suggested. The
review proposed `parity.endsWith('_open')`; what landed is an enumerated
13-state table whose *unlisted* states fail loudly as "unrecognised parity"
rather than degrading to muted, plus a bidirectional producer/consumer coverage
gate asserted from both Python and JS. `verdictBadge` also moved out of the
JSX (task 3481), so the finding's cited consumer path is itself stale now. The
decisive check — producer-emits versus consumer-handles — is empty in both
directions.

## The gate decision: continue

The task authorised stopping after the five priority findings if they all
dispositioned (a) or (c), recording the remaining 18 as low yield. **Stop was
not available**: three of the five are (b). A 3-of-5 live rate in the
high/major band is also the opposite of the premise stopping would rest on, so
all 18 medium findings were triaged rather than deferred. No finding carries
`deferred_by_gate`. Recorded verbatim in `report.json -> gate`.

The 18 vindicated the decision: 12 of them are live, including both `3453`
findings, both `3367` findings, and all three `3340` findings.

## Two findings that recovered without the gate

Worth separating from the rest, because they bound how bad the drop actually
was. **`3679` was independently recovered**: the pin test its docstring falsely
claimed now exists, and the work that wrote it is labelled in-code as "task
4126 (recovered from the task-3679 review)". Something read that review by
another route. The repo also grew a structural guard for the whole defect class
— `orchestrator/tests/test_cited_test_class_drift.py` asserts every src-cited
test class resolves, and names this incident as its motivation ("carried one
for TEN DAYS naming a class that existed nowhere in the repo"). And **`3308`
was resolved by the document catching up**: task 4088 annotated both superseded
PRD Contract rows on 2026-08-12, eleven days after the verdict was emitted and
never read.

So the dropped population was not uniformly lost. It was *mostly* lost.

## The evidence was perishable, and had already eroded

This is the finding with the longest reach, and it is why `corpus/` exists.

`.worktrees/` is gitignored, so the verdict corpus is untracked runtime state
that later re-reviews **overwrite in place**. Task 2896's verdict was already
gone before this task began: its file now holds a 2026-09-12 re-review with
three on-contract `suggestion` issues, and both archived `reviews/` copies parse
to zero issues. The highest-severity finding in this task's own roster — the one
already known to have been a genuine defect — survives only as one line of the
task description. It is `2896-correctness-0`, the single roster entry with
`corpus_record: "absent (overwritten 2026-09-12)"`.

That is not a one-off. Re-deriving the task description's own provenance census
shows erosion in progress:

| description claimed | measured 2026-09-14 | |
|---|---|---|
| 1228 stored verdicts | 1267 | drifted up (new tasks) |
| 3158 issues / 905 verdicts | 3331 / 943 | drifted up (new tasks) |
| 76 off-contract severity | **74** | **shrank** |
| 123 lacking `location` | **121** | **shrank** |
| all off-contract from `reviewer_comprehensive` | 74/74 | confirmed, with a caveat |
| window 2026-07-19 .. 2026-08-10 | 2026-07-19T01:24 .. 2026-08-10T12:33 | **confirmed exactly** |

The description concluded "this residue is closed-ended and will not grow." It
does not grow. **It shrinks.** And the arithmetic of the shrinkage is exact:

    described   76 off-contract = 23 medium-and-above + 53 below
    measured    74 off-contract = 22 medium-and-above + 52 below

One finding was lost from each band, and the one lost from the triage band is
2896's `high`. The description counts "the four `high`"; only three remain
measurable. **A population described as closed-ended lost its most important
member in the three days between being written and being read.**

Two further measurements, recorded because each cuts against a natural reading:

- *The role claim is true and nearly vacuous.* All 74 off-contract issues came
  from `reviewer_comprehensive` — but the tree also holds 29 `merger.json`
  verdicts, and all 29 carry **zero** issues. `reviewer_comprehensive` is the
  only role that emits issues at all here, so the measurement says which role
  writes issues, not which writes them badly.
- *The two gates have different blast radii.* All 74 off-contract findings also
  lacked `location`, but **47 further findings lacked `location` while carrying
  a contract severity** — including 2 `blocking`. Those 47 were adjudicated
  normally, because severity is what the gates key on for scope. Only the
  intersection was silently dropped. This is why the census keeps the two
  tallies separate: folding them together would hide it.

The frozen `corpus/` reconciles the roster exactly. The 22 live findings, the 22
frozen findings, and 22 of the 23 roster ids are the same set; the only roster
entry with no record anywhere is 2896's.

## Follow-ups filed

Fourteen tasks, one per (b) finding, no bundling — including the two `3453`
findings that share a module and the three `3340` findings whose remedies are
coupled. Coupled is not identical, and a bundled task would lose the
per-finding accounting this triage exists to produce; the coupling travels in
each description as a cross-reference instead. Every id below is a **curator
ticket**, not a task id: the curator runs asynchronously and decides
create/combine/drop, so no task existed at filing time.

| finding | ticket |
|---|---|
| `3041-correctness-0` | `tkt_0RTN0677778P5F1XT85R7DTN0F` |
| `3075-correctness-0` | `tkt_0RTN06S7S2FA8BX9EA4WX015GV` |
| `3367-correctness-0` | `tkt_0RTN07JSXP72V67AS14REFW529` |
| `3367-correctness-1` | `tkt_0RTN085P18105P4H30YNXMPH14` |
| `3031-correctness-1` | `tkt_0RTN091H7E8CR02RVAN13R6HBY` |
| `3453-correctness-0` | `tkt_0RTN09HRZC1NSSDEKVVK07KYVW` |
| `3453-blast-radius-1` | `tkt_0RTN0AEH2C6E6HF5BGREK4ZRJH` |
| `3041-architectural-coherence-1` | `tkt_0RTN0AZENJ1T3NCPJZJ1KW8P23` |
| `3064-efficiency-0` | `tkt_0RTN0BTYZJJ4F35Q3NV59FA1NG` |
| `3142-correctness-1` | `tkt_0RTN0CDNGBA7VN4VPHGA1CCQX6` |
| `3142-correctness-2` | `tkt_0RTN0D8MN4NB038DEGEXS25QV0` |
| `3340-correctness-0` | `tkt_0RTN0DPCN7ZBT5A09Z31MVRT8Y` |
| `3340-test-coverage-1` | `tkt_0RTN0EG727M3EMNQ2ADKA7T712` |
| `3340-correctness-2` | `tkt_0RTN0EVPSGZ31EX57G2V904HP7` |
| `3142-correctness-0` | **none — task 4205 already tracks it** |

Idempotency keys: `escalation_id` `agent-followup-5430` on all fourteen, and
`suggestion_hash = sha256('5430:<finding id>')[:16]`, so a re-filed candidate
dedupes per finding rather than per batch.

Three descriptions record a re-measurement that **weakened** the original
finding, so nobody picks them up believing a sharper case than the evidence
supports: `3453[1]`'s exposure is confined to escalation-spec-carrying payloads;
`3064`'s token cost does not apply to a client that defers tool schemas (as this
session's did); and `3075`'s "permanently" is false for terminal-task records.
One description carries a precondition rather than a bug: `3340-correctness-2`
is dormant only because `3340-test-coverage-1`'s binding does not exist, so
filing that binding without fixing the path derivation would convert a dormant
defect into a born-at-L2 escalation for a healthy pipeline.

## What this task did not do

No defect was fixed, and none of the 17 subsystem files named by the findings
was modified — the triage is read-and-report by design, and every fix belongs
to its follow-up. The 52 off-contract findings emitted at
low/minor/nit/trivial were not triaged (the description says 53; the same
single-loss-per-band drift applies). The 47 location-less but on-contract
findings are not in scope: they were adjudicated normally, and are measured
here only to bound the two defects' blast radii. Recurrence prevention belongs
to the companion `submit_review_verdict` validation task.
