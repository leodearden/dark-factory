# Capability manifest: merge-lane dispatch and speculation policy

**PRD:** `plans/merge-lane-policy-prd.md` · **Sidecar:** `plans/merge-lane-policy-prd.capability-manifest.yaml`
**Code anchors** verified against main `72239dfd5d` (2026-11-05). Cited by symbol.

Each leaf's signal is bound to the capabilities it asserts. Every binding is PASS; none is declared-only,
test-only, a downstream producer, or an absent producer. Mechanical checks (`grep`) were each confirmed
absent on main today (`git grep -c` over `orchestrator/src shared/src scripts hooks dashboard/src
fused-memory/src` returned 0 files for every pattern), so each flips red → green when its producer lands.

**G5 note (B + H).** Each `worker.py` leaf owns both sides of the seam it touches and names its own boundary
rows (PRD §7); ι is the readout gate that proves the whole chain through the real producers' rows.
`signal_leaves` = η, ι.

## α — instrumentation
| Capability | Binding | Verdict |
|---|---|---|
| `merge_verify` rows are emitted from one site that can carry new fields | `worker.py::_run_inflight_verify` (chain arm and plain arm), `_emit_merge_attempt`; `merge_lane/types.py` event dataclasses | PASS |
| The allocator knows why it chose a host | `verify_runner.py::HostAllocator.acquire(policy=...)`, `free_host_count` | PASS |
| A cancelled verify has a teardown site | `worker.py::_teardown_verify_task` | PASS |
| Void provenance is known at the void site | `worker.py::_chain_dead_link`, `_record_item_commit_dead` | PASS |
| The throughput script is the reader | `scripts/merge_lane_throughput.py` (task 5050, done) | PASS |
| `runner_reason` is delivered | built here; mechanical grep, absent today | PASS |

## β — re-anchor recovery re-merges
| Capability | Binding | Verdict |
|---|---|---|
| The five re-merge consumers go through one producer | `worker.py::_remerge` → `_remerge_inner` (`remerge_recovery` marker) | PASS |
| Live-tip inputs exist | `worker.py::frozen_prefix_tip`, `_dead_base_commits`, `_frozen_inflight_entries`; `types.py::InflightEntry.vacated` | PASS |
| Permit ledger exposes a non-blocking slot test | `speculation_controller.py::PermitLedger.slot_available` | PASS |
| Merger state to re-point | `SpeculationController.spec_base`, `pending_spec_base` | PASS |
| Accounting audit stays | `worker.py::speculation_accounting_violations` | PASS |
| `on_base_invalidated` is delivered | built here; mechanical grep, absent today | PASS |

## γ — gate re-verify as a dispatch
| Capability | Binding | Verdict |
|---|---|---|
| Gate decision row written before any verify | `reverify_gate_decision.py::emit_reverify_gate_decision`; `gates.py::_reverify_rebased_tree` | PASS |
| The park site to replace | `worker.py::_land_rebased_or_park_gate`, `_gate_reverify_on_head_host` | PASS |
| Front redispatch exists | `worker.py::request_redispatch` (used by `_void_and_remerge`) | PASS |
| `base_relation=rebased_head` is delivered | built here (field from α); mechanical grep `rebased_head`, absent today | PASS |

## δ — K = hosts + 1, lease released at verdict
| Capability | Binding | Verdict |
|---|---|---|
| K is a single source | `harness.py::Harness._speculation_k` | PASS |
| Lease release site | `worker.py::_finalize_inflight` → `_resolve_and_release` | PASS |
| Entry status vocabulary | `types.py::InflightEntry` (`status`, `lease`) | PASS |
| `verified_pending_landing` status is delivered | built here; mechanical grep, absent today | PASS |

## ε — chain the head when alone
| Capability | Binding | Verdict |
|---|---|---|
| Guard to relax | `worker.py::_deep_chain_placement` guard 1 (`not item.speculative`) | PASS |
| Head adoption path | `worker.py::_adopt_head_on_tip_authority`, `_land_chain_prefix` | PASS |
| Link reservation | task 7720 (pending, upstream of ε). The lane-leak fix 7721 and the off-loop build 7824 were upstream of ε until 2026-11-10 and now follow it (PRD §8, note under ε) | PASS (producer upstream) |

## ζ — bisect, depth table, kill switch
| Capability | Binding | Verdict |
|---|---|---|
| Halving state and its two writers | `worker.py::_chain_halving_state`, `next_halving_state`, `_note_chain_outcome` | PASS |
| Config class, green-tier | `config.py::MergeDeepConfig` (`chain_cap`), `RELOADABLE_FIELDS` | PASS |
| `merge_deep.depth_table` is delivered | built here; mechanical grep, absent today | PASS |

## η — hook parity with the merge gate
| Capability | Binding | Verdict |
|---|---|---|
| Module registry the gate reads | `verify_plan.py::effective_merge_module_configs` | PASS |
| The hook's main-only tier and its drift test | `hooks/project-checks` (`PYRIGHT_PACKAGES`); `tests/scripts/test_hook_pyright_packages_drift.py` | PASS |
| Screen list shape | `plans/merge-pre-verify-screen-prd.md` §3 decision 3 (`scripts/merge_screen.yaml`, absent today; η creates if absent, 7919 consumes) | PASS |
| Rejection fires | boundary rows 10–11: the hook exits non-zero naming the check (observed in η's throwaway-repo test) | PASS (rejection-check) |
| `scripts/hook_module_checks.py` is delivered | built here; mechanical grep, absent today | PASS |

## ι — 7-day canary readout
| Capability | Binding | Verdict |
|---|---|---|
| Every predicate reads a field α produces | `runner_reason`, `base_relation`, `merge_verify_cancelled`, `verdict_voided.item_commit`, `queue_to_land_secs`: producer α upstream | PASS |
| Reader proves its producer | the readout refuses to run when α's fields are absent from the window (INV-13) | PASS |
| Thresholds have a basis | PRD §2 last paragraph (model vs measured per predicate) | PASS (floor stated) |
