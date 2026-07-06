# Capability Manifest — merge-queue-reliability PRD (W1)

Mechanizes G3 (assumed-substrate) + G6 (premise validity) per leaf. Each task's observable
signal is decomposed into the capabilities it asserts; each capability is bound to evidence.
All `grep:` anchors verified 2026-07-06 against HEAD `365e63b9` (main moves fast — re-verify at dispatch).
Evidence vocabulary: `grep:file:line wired` (exists+wired on main) · `producer:task-<label>` (delivered by an **upstream** task in the dependency closure) · `stdlib` · `self-produced` (the task's own deliverable) · `rejection-check:<X> fires` (task builds the guard AND its signal demonstrates it firing).

**Verdict: no FAIL bindings.** No numeric bounds (no floor checks). No novel grammar/syntax (not a language project). All negative assertions are self-produced rejection mechanisms whose signals demonstrate them firing. `OutcomeKind` is NOT asserted anywhere in this batch (out of scope, M3-owned) — deliberately.

---

## α — LandedOutbox durable store + MergeProvenance.lookup(task_id)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| A durable file store keyed by task_id (`record`/`lookup`/`all`/`consume`) | self-produced | PASS |
| fsync-to-disk durability (row survives simulated restart) | `stdlib` (`os.fsync` + dir fsync); atomic-write precedent `grep:orchestrator/src/orchestrator/merge_queue_store.py:182 (_save_raw tmp+os.replace)` | PASS |
| Worker holds a LandedOutbox instance | self-produced (worker `__init__`, precedent `grep:orchestrator/src/orchestrator/merge_queue.py:4116 (SuffixConflictTracker field)`) | PASS |

DAG: root (no upstream). No inversion.

## β — Write-ahead wiring at both CAS advance sites

| Capability | Evidence | Verdict |
|---|---|---|
| `record` API to call write-ahead | `producer:α` (upstream) | PASS |
| CAS advance site #1 (single-branch) | `grep:orchestrator/src/orchestrator/merge_queue.py:8783 (_finalize_inflight advance_main)` wired | PASS |
| CAS advance site #2 (train) | `grep:orchestrator/src/orchestrator/merge_queue.py:2961 (train finalize advance_main)` wired | PASS |
| branch_tip_sha available at advance | `grep:orchestrator/src/orchestrator/merge_types.py:800 (item.merged_branch_tip)` wired | PASS |
| advanced_sha / merge commit available pre-advance | `grep:orchestrator/src/orchestrator/merge_queue.py:8763 (merge_commit/current_sha)` wired | PASS |
| Ordering assertion (row present when advance_main invoked) via fake git_ops | self-produced (test) | PASS |

DAG: β depends on α (upstream) ✓ no inversion.

## γ — Startup reconciler + crash-window contract test

| Capability | Evidence | Verdict |
|---|---|---|
| Enumerate unconsumed rows | `producer:α` (`all`/`consume`) upstream | PASS |
| Rows exist to reconcile | `producer:β` (write-ahead wiring) upstream | PASS |
| Mark task done with `merged` provenance | `grep:orchestrator/src/orchestrator/scheduler.py:1704 (mark_done kind='merged')` wired (existing substrate) | PASS |
| **G6 premise: the merged done-write passes the server gate** | `grep:fused-memory/src/fused_memory/middleware/task_interceptor.py:3582 (is-ancestor backstop)` — advanced_sha IS on main by construction ⇒ is-ancestor passes ⇒ premise VALID (not a false premise) | PASS |
| is_ancestor(advanced_sha, main) to distinguish RC-1 vs RC-2 | `grep:orchestrator/src/orchestrator/git_ops.py (is_ancestor)` wired (used by `recover_pending_merges` :318) | PASS |
| Crash simulated by injected fault point (not real kill) | self-produced (test harness) | PASS |

DAG: γ depends on β→α (upstream) ✓ no inversion. **G6:** the RED contract test asserts convergence, not an impossible number — premise VALID.

## δ — Scheduler consult-before-dispatch gate

| Capability | Evidence | Verdict |
|---|---|---|
| `MergeProvenance.lookup(task_id)` | `producer:α` upstream | PASS |
| Shared reconcile-to-done routine | `producer:γ` upstream | PASS |
| Scheduler dispatch decision point to gate | `grep:orchestrator/src/orchestrator/scheduler.py:3890 (_eligible_for_dispatch)`, `:3899 (try_acquire)` wired | PASS |

DAG: δ depends on γ→β→α (upstream) ✓ no inversion.

## ε — Monkeypatch-path migration + grep-guard ratchet

| Capability | Evidence | Verdict |
|---|---|---|
| The reach-back string-path patches exist to migrate | `grep:orchestrator/src/orchestrator/merge_gates.py:361`, `:439`, `:1353`; `merge_drift.py:254`; `merge_shadow.py:1012`; `merge_liveness.py:181,:280,:702` wired | PASS |
| Full suite runs (green after repoint) | `grep:orchestrator/tests/ (64 files import from orchestrator.merge_queue)` — real suite | PASS |
| **Rejection: guard fails on a NEW reach-back patch** | `rejection-check:new-merge_queue-private-string-patch fires` — task builds the grep-guard AND its signal demonstrates a fixture tripping it (per gates.md G6 branch 4: rejection mechanism is the deliverable, observed to fire) | PASS |

DAG: ε depends on δ (linear spine) ✓.

## ζ — SpecPermit token + PermitLedger (single owner of _speculation_slot)

| Capability | Evidence | Verdict |
|---|---|---|
| `_speculation_slot` semaphore to wrap | `grep:orchestrator/src/orchestrator/merge_queue.py:3942` wired | PASS |
| SpeculationController to refactor onto the ledger | `grep:orchestrator/src/orchestrator/merge_speculation_controller.py:51 (verifier-side still raw)` wired | PASS |
| Token stored on the item | self-produced (`SpeculativeItem.permit`/`InflightEntry.permit`) | PASS |
| Conservation identity `slot_available + len(live) == depth` | self-produced (construction invariant); audit precedent `grep:orchestrator/src/orchestrator/merge_queue.py:5006 (speculation_accounting_violations)` | PASS |

DAG: ζ depends on ε (upstream) ✓.

## η — Thread token; delete census + release_resources flags + raw release sites

| Capability | Evidence | Verdict |
|---|---|---|
| `PermitLedger`/`SpecPermit` to thread | `producer:ζ` upstream | PASS |
| 5-location census to delete | `grep:orchestrator/src/orchestrator/merge_queue.py:5006-5084 (_inflight_speculative_count)` wired | PASS |
| ~10 raw release sites to migrate | `grep:orchestrator/src/orchestrator/merge_queue.py:6404,:6508,:7907,:8362,:9015` + `merge_speculation_controller.py:185,:271,:286` wired | PASS |
| `release_resources`/`_entry_released` caller flags to remove | `grep:orchestrator/src/orchestrator/merge_queue.py (_resolve_and_release release_resources param)` wired | PASS |

DAG: η depends on ζ (upstream) ✓ no inversion.

## θ — CapPermit for _merge_ahead_cap + grep-guard ban

| Capability | Evidence | Verdict |
|---|---|---|
| `_merge_ahead_cap` semaphore | `grep:orchestrator/src/orchestrator/merge_queue.py:3960` (acquire :7348; releases :6408,:7383,:9069) wired | PASS |
| Ledger to own the cap | `producer:ζ/η` upstream | PASS |
| **Rejection: guard bans raw semaphore access** | `rejection-check:raw _speculation_slot/_merge_ahead_cap .acquire()/.release() outside PermitLedger fires` — task builds the grep-guard; signal demonstrates it | PASS |

DAG: θ depends on η (upstream) ✓.

## ι — ItemLifecycle registry + state enum + legal-transition table

| Capability | Evidence | Verdict |
|---|---|---|
| Registry keyed by request_id | self-produced | PASS |
| State enum + legal-transition table | self-produced (`merge_types.py`) | PASS |
| **Rejection: illegal transition raises/escalates** | `rejection-check:illegal transition fires` — task builds `transition()`; signal demonstrates an illegal move raising | PASS |

DAG: ι depends on θ (upstream) ✓.

## κ — Wire transition() everywhere; converge snapshot/audit/liveness

| Capability | Evidence | Verdict |
|---|---|---|
| `transition()` + registry | `producer:ι` upstream | PASS |
| 4 transient side-fields to convert | `grep:orchestrator/src/orchestrator/merge_queue.py:3985,:3996,:4003,:4015` wired | PASS |
| snapshot()/audit/liveness to repoint | `grep:orchestrator/src/orchestrator/merge_queue.py (snapshot, speculation_accounting_violations)`, `merge_liveness.py` wired | PASS |
| **G6 premise: the three consumers can be made to agree** | structural — all three derive from one registry read ⇒ agreement by construction (not an impossible assertion) | PASS |

DAG: κ depends on ι (upstream) ✓ no inversion.

## λ — Delete _verify_phase/_verify_item dual-writes + free-form phase:str

| Capability | Evidence | Verdict |
|---|---|---|
| Registry provides phase (phase = state) | `producer:ι/κ` upstream | PASS |
| Vestigial fields + dual-write sites to delete | `grep:orchestrator/src/orchestrator/merge_queue.py:3990,:3991,:3992` (fields); phase writes `:8702,:8801,:8864`; `merge_types.py:954 (InflightEntry.phase)` wired | PASS |

DAG: λ depends on κ (upstream) ✓.

## μ — QueuedBranch value type + parse() classmethod

| Capability | Evidence | Verdict |
|---|---|---|
| `merge_types.py` home for the value type | `grep:orchestrator/src/orchestrator/merge_types.py:609 (MergeRequest lives here)` wired | PASS |
| `branch_prefix` config to parse against | `grep:orchestrator/src/orchestrator/merge_queue_store.py:117 (req.config.git.branch_prefix)` wired | PASS |
| Frozen value type + single parse() | self-produced | PASS |
| **G6 premise: mixed shape becomes unrepresentable** | structural (parse-don't-validate) — the invariant is enforceable, not asserting an impossible number | PASS |

DAG: μ depends on λ (linear spine) ✓.

## ν — MergeRequest.branch → QueuedBranch; delete normalizers

| Capability | Evidence | Verdict |
|---|---|---|
| `QueuedBranch` type + parse() | `producer:μ` upstream | PASS |
| `canonical_queued_branch_name` to delete | `grep:orchestrator/src/orchestrator/git_ops.py:737` wired | PASS |
| try-both `resolve_queued_branch_ref` to delete | `grep:orchestrator/src/orchestrator/git_ops.py:3968` wired | PASS |
| journal strip/re-add pair to delete | `grep:orchestrator/src/orchestrator/merge_queue_store.py:117,:304` wired | PASS |
| ~8 inline `f'{branch_prefix}{...}'` sites | `grep:orchestrator/src/orchestrator/git_ops.py:1246,:1307,:1770,:2649,:3378` wired | PASS |
| **G6 premise: pyright enforces the invariant at drift sites** | `MergeRequest.branch: QueuedBranch` makes a bare-str assignment a type error — pyright is the existing merge-verify substrate (unscoped pyright runs today) | PASS |

DAG: ν depends on μ (upstream) ✓ no inversion.

## ξ — Migrate _verify_and_advance tests to public surface + delete shim

| Capability | Evidence | Verdict |
|---|---|---|
| `_verify_and_advance` shim to delete | `grep:orchestrator/src/orchestrator/merge_queue.py:9312` wired | PASS |
| ~30 direct-call test sites to migrate | `grep:orchestrator/tests/test_merge_queue.py (23 calls)`, `test_merge_queue_invariant_integration_gate.py (5)`, `test_merge_item_union.py (2)` wired | PASS |
| Public surface to drive instead | `grep:orchestrator/src/orchestrator/merge_queue.py (run/_dispatch_item/_finalize_inflight)` wired | PASS |

DAG: ξ depends on ν (linear spine) ✓.

## ο — Freeze _serial_merge_worker.py + strip normative docstrings

| Capability | Evidence | Verdict |
|---|---|---|
| `_serial_merge_worker.py` exists (384 lines) to freeze | `grep:orchestrator/tests/_serial_merge_worker.py:1` wired | PASS |
| ~10 normative "mirrors the test-local reference" docstrings to strip | `grep:orchestrator/src/orchestrator/merge_queue.py:2400 (classify_and_merge docstring), :3833-3910 (counter comments)` wired | PASS |
| **Rejection: ratchet forbids NEW _serial_merge_worker imports** | `rejection-check:new _serial_merge_worker import fires` — task builds the ratchet; signal demonstrates it | PASS |

DAG: ο depends on ξ (upstream) ✓ terminal leaf.

---

## G6 summary (per gates.md branches)

- **Branch 1 (numeric bound/threshold):** none asserted — N/A for every leaf.
- **Branch 2 (closed-form exactness):** none asserted — N/A.
- **Branch 3 (end-to-end capability / field-population):** γ's merged done-write premise traced to the dependency closure (α+β produce the row; `scheduler.mark_done` + server gate are existing upstream substrate) and validated — advanced_sha is on main so the is-ancestor backstop passes. κ's "three consumers agree" is structural. No downstream-owned capability is asserted on an upstream leaf (no inversion anywhere in the linear spine).
- **Branch 4 (negative assertion / rejection):** ε, θ, ι, ο each assert a rejection (guard trips / illegal transition raises). In every case the rejection **mechanism is the task's own deliverable** and the task's signal explicitly demonstrates it firing (fixture trips the guard / illegal move raises) — bound as `rejection-check:… fires`, PASS by construction.
