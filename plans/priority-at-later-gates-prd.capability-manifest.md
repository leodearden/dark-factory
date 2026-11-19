# Capability manifest: priority at the later gates, and a pin as a tier

**PRD:** `plans/priority-at-later-gates-prd.md` · **Sidecar:** `plans/priority-at-later-gates-prd.capability-manifest.yaml`
**Code anchors** verified against main `a8bd0b5eb1` (2026-11-13). Cited by symbol.

Each task's signal is bound to the capabilities it asserts. Every binding is PASS: none is declared-only, test-only,
a downstream producer or an absent producer. Each `grep:` pointer was resolved against main by parsing the file at
`main:<path>`. Each built-here pattern was confirmed absent on main with `git grep -E` over the stated paths, so its
check goes from failing to passing when its producer lands. One check is the reverse: γ's removal of the fixed
rebase cap is expressed as a pattern that must become absent.

**G5 (B + H).** κ1 and κ2 are the integration gates; their signal is the PRD's section 7 rows against the real
modules. `signal_leaves` = κ1, κ2.

**G7.** The design was walked against `docs/legibility/design-invariants.md` in four critic passes
(`plans/priority-at-later-gates-prd.evidence/critic-part1-pass3.md`, `critic-part1-pass4.md`). The last pass named
three remaining hits (`loop-thread-occupancy-bounded`: the paused re-read; `storm-escape-required`: the
unknown-tier alarm; `guards-exercise-behaviour`: rows 23 and 24). Each was resolved in the PRD text before
decompose (6.2, 6.1, section 7 and θ1's row). No waiver is recorded.

**Producers outside this batch.** Task 9908 (the priority value, the lookup, the recorded pick rows; hand-carried, in
the merge queue at decompose time), task 9636 (watchdog inputs), task 8528 (the suspect list), task 8532 (the
lane's seven-day readout), task 6718 (the shared tier vocabulary), task 9683 (the last scheduler restructuring
leaf). Each is wired as a real dependency of the task that needs it.

## α1 — the priority value, the comparator and the lookup
| Capability | Binding | Verdict |
|---|---|---|
| The value and the injected lookup exist to extend | producer:task-9908 upstream | PASS |
| Pin rows can be read at construction | `grep:orchestrator/src/orchestrator/overrides.py::OverrideStore.get_overrides` | PASS |
| The scheduler writes the state file α1 reads back | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.get_state_snapshot` | PASS |
| `order_key`, `GATE_RANK`, the `merge_pool` config, `age_from` on `merge_queued` | built here; four mechanical greps, absent today | PASS |

## α2 — the first-eligible table
| Capability | Binding | Verdict |
|---|---|---|
| Every candidate is stamped each tick | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._apply_starvation_watchdog` wired: it calls `EligibilityClock.observe` | PASS |
| A status read exists for tasks absent from the fetch | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.get_statuses` | PASS |
| The field the anchor is returned in | producer:α1 upstream | PASS |
| The `first_eligible` table | built here; mechanical grep, absent today | PASS |

## β — verify admission (task 8261)
| Capability | Binding | Verdict |
|---|---|---|
| One pick site and one policy builder | `grep:orchestrator/src/orchestrator/verify_admission_gate.py::AdmissionGate._pick`, `grep:orchestrator/src/orchestrator/verify.py::admission_policy` | PASS |
| The second gate site | `grep:orchestrator/src/orchestrator/verify_hosts.py::HostPool` | PASS |
| `order_key` | producer:α1 upstream | PASS |
| The gate orders by `order_key` | built here; mechanical grep on the gate module, absent today | PASS |

## γ — the pool order
| Capability | Binding | Verdict |
|---|---|---|
| One pick, two callers | `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._acquire_next_request`, `grep:orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._look_ahead` wired: the only callers of `MergeIntake.pop_next_pickable` | PASS |
| A fork-point changed-files query | `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.get_branch_changed_files` | PASS |
| The footprint is computed in one place | `grep:orchestrator/src/orchestrator/suffix_graph.py::SuffixConflictTracker.recompute` wired | PASS |
| The suspect list that stays first | producer:task-8528 upstream | PASS |
| The config and the comparator | producer:α1 upstream | PASS |
| `pool_order` | built here; mechanical grep, absent today | PASS |
| The rebase cap is read from config | built here; mechanical grep that the fixed constant is gone | PASS |

## ι1 — the order readout
| Capability | Binding | Verdict |
|---|---|---|
| The rows it reads are written by real producers | producer:task-9908 upstream (the waiting set), producer:γ upstream (rank and age on each entry), producer:α2 upstream (age from the anchor) | PASS |
| The readout, with its "no records" state | built here; mechanical grep for the state's text, absent today | PASS |

## ι2 — the human gate
| Capability | Binding | Verdict |
|---|---|---|
| The readout to present | producer:ι1 upstream | PASS |
| The lane's seven-day readout has been taken | producer:task-8532 upstream | PASS |
| The readout covers picks recorded after γ deployed | read by a human at the gate; manual | PASS |

## θ2 — `pinned` in the shared vocabulary
| Capability | Binding | Verdict |
|---|---|---|
| The shared module to add it to | producer:task-6718 upstream | PASS |
| The name and rank order to move | producer:α1 upstream | PASS |
| `PINNED` in `shared/` with the refusals | built here; mechanical grep, absent today | PASS |

## θ1 — `pinned` inside the scheduler
| Capability | Binding | Verdict |
|---|---|---|
| The inheritance walk runs each tick | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_compute_priorities` wired | PASS |
| The reservation cap is enforced in one place | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._bound_pin_reservations` wired | PASS |
| Only the top-scored candidate accrues skips | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_select_scored` wired | PASS |
| The separate pin pass that is folded in | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_select_pins` | PASS |
| The vocabulary | producer:θ2 upstream | PASS |
| Pin orders in the state snapshot | built here; mechanical grep, absent today | PASS |

## κ1, κ2 — integration gates
| Capability | Binding | Verdict |
|---|---|---|
| κ1's producers | producer:β upstream, producer:γ upstream, producer:α2 upstream | PASS |
| κ1's boundary module (`TestPriorityGatesBoundary`) | built here; mechanical grep, absent today | PASS |
| κ2's producers | producer:κ1 upstream, producer:θ1 upstream | PASS |
| κ2's boundary module (`TestPinTierBoundary`) | built here; mechanical grep, absent today | PASS |
