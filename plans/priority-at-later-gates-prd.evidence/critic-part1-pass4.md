# Critic, fourth pass: the fold-in to `/tmp/prd-merge-pool/prd1-priority.md`

Read-only, 2026-11-13. Checked: the diff against `prd1-priority.v1.md` and the code and tasks it touches. Main
`a8bd0b5eb1`; task store, `scheduler_state.json` and the eligibility store read without writing. Nothing written but
this file.

Counts: **1 answer-changing (A)**, **11 minor (B)**. Verdict: **not clean**, by one decomposition change.

## 1. Third-pass items as folded

All seven are folded correctly: A1 (start-up fill, paused re-read, row 21, row 22), A2 (D5, row 11, open question 4
removed), A3 (6.4 rewrite, O3, rows 6 and 14), A4 (6.3 key, lookup on the plan, decisions 2, 3 and 7, 8261's text
replaced, plan arrival), A5 (D6 additions, rows 19 and 24), A6 (α creates the config file; γ needs α and 8528), A7
(the status read, the accepted race, the seed). No "leg's arrival" remains. No `merge_lane/merge_pool_facts.py`
remains. Residue from the fold-in is in B.

Checks that came out clean:
- **Cycle.** None. Task 9636 is `pending`, medium, effective tier `high`, with no open task upstream; its dependents
  are 9727 and 7718.
- **The snapshot file.** The scheduler writes `scheduler_state.json` and never reads it today
  (`shared/scheduler_state.py::SchedulerStateSnapshot.from_json` exists for other readers). The live file holds
  3,478 effective tiers, including all 18 in-progress tasks, so a task that was in progress at a stop is covered.
- **Promoted background leg.** "The oldest `high` waiter" is a key: `(0, (rank of high, 0, -infinity), seq)`.
- **κ1** needs nothing from ι.

---

## A. Answer-changing

### A1. α is now outside the sizing band and waits for a task that queues for a heavily contended file

**Draft.** Section 8, α: touches "`eligibility_clock.py`, `scheduler.py` (the lookup, its start-up fill, the cleanup
read), `orchestrator/task_priority.py`, `orchestrator/merge_pool_facts.py`, new `orchestrator/merge_pool_config.py`
mounted in `config.py`, a seed script, tests"; needs "9908, 9636". Critical path: "9908 → α → γ".

**Wrong.**
1. Size. Counted out, α is about 15 files: the six source files named, `defaults.yaml` (the config stanza), the
   `merge_queued` emit site in `merge_lane/intake.py`, the seed script, and six test files (eligibility clock, task
   priority, pool facts, lane recording, the lookup and its start-up fill, config reload, the seed). My estimate is
   600 to 700 source lines and 900 to 1,100 test lines, so 1,500 to 1,800 against a band of 300 to 1,500 and 10 to
   12 files. It is also two units of work: the value and its lookup, and the durable table.
2. The wait. α is the head of the critical path, and it now waits for 9636. 9636 has no blocker of its own, but it
   needs `scheduler.py`, which 88 pending tasks and one in-progress task declare today, one of them pinned (6328).
   For scale: task 8261, at high tier and pinned, waited 9.9 days for `config.py` and `verify.py`. Every leaf of this PRD would wait with α. (The edge direction is the one I advised in
   the third pass; the advice did not weigh this.)
3. Only the first-eligible table shares a file with 9636. `order_key` already orders a waiter with no anchor from
   its fallback age, so β and γ can be built and tested before the table exists.

**Replace α with two leaves.**

| Label | Task | Touches | Signal or what it unlocks | Needs |
|---|---|---|---|---|
| **α1** | The value and the lookup: `TaskPriority` with `base_tier`, `pin_order`, `age_anchor`; `PINNED`, `GATE_RANK`, `order_key`; the lookup returns `pinned` for an own pinned row and answers before the first tick and while paused; the unknown-tier streak; the `merge_pool` config file; the new `merge_queued` keys | `orchestrator/task_priority.py`, `scheduler.py` (the lookup and its start-up fill), `orchestrator/merge_pool_facts.py`, `merge_lane/intake.py`, new `orchestrator/merge_pool_config.py` mounted in `config.py`, `defaults.yaml`, tests | `merge_queued` rows carry `base_tier`, `age_secs` and `age_from: enqueue`, and tier `pinned` for a pinned task before the first tick after a restart. Unlocks β, γ, θ2, α2 | 9908 |
| **α2** | The first-eligible table, its status-read cleanup and its seed; the lookup returns the anchor | `eligibility_clock.py`, `scheduler.py` (the cleanup read, the anchor in the lookup), a seed script, tests | `merge_queued` rows carry `age_from: anchor`; the seed reports how many rows it wrote from each source. Unlocks ι's gate and κ1 | α1, 9636 |

Then: β needs α1. γ needs α1 and 8528. θ2 needs α1 and 6718. ι needs "γ observed running; α2 landed; 8532 has
read", so the switch is not flipped while age is still the fallback. κ1 needs β, γ and α2 (rows 7, 8 and 10 need the
table). Critical path: "9908 → α1 → γ (after 8528) → ι (after α2 and after 8532 has read); β in parallel". Section
5's 9636 row names α2.

---

## B. Minor

1. **6.4, "`pool_order()` returns that order."** Ambiguous for `arrival`, where the pick is a selection rule, not a
   total order, and `chain_snapshot` is deliberately plain buffer order. Replace with: "With `arrival`,
   `pool_order()` is the lane buffers in buffer order, high lane first; the pick applies the same-files rule over
   it, as today; every other reader keeps the filter it has today." Readers can then share the one function with no
   change in arrival mode. γ owns the readers.
2. **6.4, the footprint depends on the switch, and the switch is hot.** The conflict graph's debounce signature must
   include `merge_pool.order`, or a flip leaves the old footprint in place until the waiting set changes.
3. **6.2, "While paused it re-reads the override store".** Three details. The pins the lookup reads must be a map
   of its own, not `_prev_overrides_snapshot`: that one is the baseline for the override change events, and
   refreshing it during a pause would swallow `task_pinned` on resume. The re-read runs from the paused return of
   `Scheduler.acquire_next`, off the loop thread; the lookup stays a dict read. And 9908's lookup answers a pin
   only once `_overrides_initialized` is set, which happens in the first tick; α1 drops that condition.
4. **6.2, the start-up fill after θ1.** The snapshot's `effective_priorities` will then say `pinned` for an
   inherited prerequisite and carry no order for it. Filled from that, the prerequisite sorts as order 0, ahead of
   every operator pin, until a tick runs; in a pause that is until resume. θ1 adds the pin orders and base tiers to
   the snapshot (`shared/scheduler_state.py` is already in its files) and the fill reads them. Say also: a missing
   or unreadable file leaves tiers unknown, counted. (The live file was 37 minutes old when I read it.)
5. **6.1 and row 25, "Ten consecutive picks at one gate with an unknown tier file one L1".** As worded, a process
   with no scheduler (a gate with no lookup wired) would file it. Replace with: "Where a lookup is wired, ten
   consecutive picks whose picked waiter has no tier file one L1, not re-filed while it is open." No leaf builds
   it: α1 builds the streak, β and γ call it. (My wording.)
6. **Row 23.** Its assertion mentions a pinned or critical leg that its preconditions do not have. Add "and one
   critical task leg waits". Take the background key from a constant in `task_priority.py`, so D1's "build no key
   of their own" still holds.
7. **Row 24.** Owned by κ2 only. Add it to θ1's signal ("Rows 15 to 19 and 24").
8. **Rows 21 and 22 in α.** The pick they assert is γ's. With the split, α1's signal states what α1 alone shows
   (above); κ1 keeps the rows.
9. **D10, "ordered inside by `order_key`".** Only under `priority`; under `arrival` the high lane is as today. Add
   "when `order` is `priority`".
10. **A stale tier during a long pause.** Accepted in 6.2 for a tier; say the same of an inherited pin whose
    dependent was unpinned during the pause: it stays `pinned` at the later gates until resume.
11. **Open question 4 (whether γ splits).** With B1 and B2 folded γ is about 13 files. Keep the question.

---

## Invariants: hits that remain

| Invariant | Mechanism | Item |
|---|---|---|
| `loop-thread-occupancy-bounded` | the paused re-read, if it is done inside the lookup | B3 |
| `storm-escape-required` | the unknown-tier L1 with no "lookup wired" condition and no dedupe | B5 |
| `guards-exercise-behaviour` | row 23 asserts on a leg it does not set up; row 24 has no implementing leaf | B6, B7 |

## META

With the three [Leo] decisions confirmed as recommended, the design passes: the contract is complete and no open
design question remains. It is held back by A1 alone, which is about how the work is cut and what it waits for, not
about what is built. With α split as above and B1 to B5 folded, I would call it clean.

Not clean.
