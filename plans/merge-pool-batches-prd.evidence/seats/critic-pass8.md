**Verdict: clean pass. No A-defect; 5 B, all in the new task κ or wording.**

Read-only against main `9856ed15dc`. Scratch is `/tmp/gate-10069/critic8/` (`d89.diff`, `rule9.py`, `c1.py`). No repository test was run.

What I did to convince myself:
- Diffed draft 9 against draft 8 and ran `rule9.py` (staged copy identical); every figure in B3, D14 and the section-9 row matches its output, and B1 (4 of 246, at thresholds), B2 and the first-kind minimum (0 of 1,278 cells off) hold.
- Re-ran my own corner grid on the new `F`: 4,032 cells (q from 0 to 0.999999, seven cap pairs, three starts) all converge with no dependence on the start. `F(0)`, `F(1)`, `F(2)` are 0, 1, 1.56 at q 0.14.
- Read κ against the code and tasks 9176, 4838 and 5623, and re-walked the cross-references.

## 1. Pass-7 items against DRAFT 9

| Pass 7 | Status |
|---|---|
| A1 (`F(1)`) | answered: one new member costs one verify; a cleared front now takes 12 new up to q 0.17 |
| B1 (changed tip) | answered: the conviction lapses and row 30 asserts the pick |
| B2 (D13 carried) | answered: `FALLBACK` bullet, D12, V1, rows 3 and 47; D13 says it narrows L9 |
| B3 (joint price) | answered: 9 to 22% less, 11% at 0.14; a new front does not form it from 0.11 |
| B4 (small) | answered |

## 2. New defects

### A. Answer-changing

None.

### B. Minor

1. **κ's lifetime sentence is wrong for two counters ("An entry is dropped when its request leaves the lane").** The timeout and full-disk counters are keyed by task and span submissions: `merge_lane/worker.py::SpeculativeMergeWorker.__init__` says "lives across submissions … cleared on a successful CAS advance", and `_merger_loop` reads the count when the next submission is dequeued, to abandon it. Dropped on leaving, the loop-breaker never trips. Existing tests pin it (`orchestrator/tests/test_merge_queue.py` among five files). *Requirement:* each entry keeps today's lifetime, and a persisted entry has an age bound.
2. **κ's list is narrower than the inventory it closes.** L12 and κ's signal name the bounce count, the landing-race, timeout and full-disk retries, the window and the halt. The inventory row also has gate and narrowed retries, chain-generation counts and the probe round counter. Say which κ carries; ι2 cannot get a ruling for the rest.
3. **κ's restored halt has no owner.**
   - `harness.py::halt_merge_queue`'s docstring says a restart clears an operator halt. κ touches the file, so say that this changes.
   - Task 4838's false critical for an ownerless halt would then also fire after a restart. Section 5 notes only 9176.
4. **The recent-verify window is reset at start on purpose.** The comment at `_recent_verify_outcomes` says a restarted worker "never suppresses on phantom flakiness". With `probe_fraction: 0.0` in dark_factory it has no effect today. Give the restored window an age bound.
5. **Wording.**
   - L4 says the whole chain of section 8 lands ahead of task 5041; the decompose note says κ and ι2 only share `worker.py` with it through the file lock.
   - Section 10's last bullet ("each item's owner is named there") predates L12.
   - `F` at p = 0 is n in `rule9.py` and 0/0 in the text.

## 3. Leo's call

1. **Does task 5041 wait for κ and ι2 (L4's words), or only for β2 (the note)?**

## 4. Checked and sound

- **κ's files hold the state it names.**
  - The bounce count is `suffix_graph.py::MergeBounceRegistry`, reached through `intake_buffers.py`; `MERGE_BOUNCE_CAP` is 3, so "two of three used, one left" is right.
  - The retry counters and `_recent_verify_outcomes` are in `merge_lane/worker.py`. `merge_lane/gates.py` only mutates the counters through the mapping interface, so a persistent map can replace them.
  - The operator halt is `merge_lane/halt.py::HaltState._operator_halt`; `unhalt_merge_queue` clears it through `unhalt_all_lanes`.
- **Other tasks.**
  - 5623 touches the journal only: no conflict.
  - 9176 and 4838 edit the same files, serialised by the file lock.
- **κ's signal.** Observable through restart scene tests and `get_merge_halt_status`. κ is intermediate: ι2 needs it.
- **Cross-references.**
  - Rows 1 to 49 are each assigned (rows 3, 30 and 47 edited in place).
  - κ and ι2 are in "Notes for decompose".
  - 6.6's row kinds carry `alone` and `uncleared` with the observation.
  - L13's cancelled tasks match section 8.
  - No reference to `rule7` or `rule8` remains.

## 5. Not checked

- `REPORT.md` and the staging tree beyond the `rule9` copy.
- Nothing outside the diff was re-read, apart from the cross-references above.
- No scan was run on a real tree and the simulator was not rerun.
