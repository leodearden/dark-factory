# Capability manifest: a reserve account for urgent work

**PRD:** `plans/allowance-reserve-by-account-prd.md` · **Sidecar:** `plans/allowance-reserve-by-account-prd.capability-manifest.yaml`
**Code anchors** verified against main `8806eb19eb` (2026-11-16). Cited by symbol.

Every binding below is PASS. Each `grep:` pointer was resolved against main with `git cat-file -e main:<path>`
and the name's definition looked up in `git show main:<path>`. Each built-here pattern was confirmed with
`git grep -E` on main: absent for every `present` check, present for the one `absent` check (task 2's private
store write), so each check fails today and passes when its producer lands.

**G5 (B + H).** The boundary tests are the integration gate, carried by the leaves that build each side: rows B1
to B20 by task 1b, C1 and C2 by task 2, C3 to C7 by task 3, C8 to C10 by tasks 2 and 4. The PRD's signal leaf is
task 7: its pass means every running orchestrator and the memory service's curator report an active reserve.

**G7.** All eight tasks were walked against `docs/legibility/design-invariants.md`. No waiver. Points written into
the task briefs:

- `holds-owned-and-bounded`: the reserve wait (task 1b) is owned by the gate's clock and account transitions,
  bounded by the release time and the caller's deadline, visible in its heartbeat and wait events. The dispatch
  hold (task 3) is recomputed every tick and ends by itself at the release time. Reconciliation's deferral (task
  4) ends at the same time, and a judge it leaves pending is fired when the refusal ends, with a restart as
  backstop.
- `readers-prove-their-producer`: the weekly report (task 5) reads two events that task 3 produces. Task 5 now
  depends on task 3 as well as 1b, so its fixture is written by the real producers.
- `no-lockstep-duplication`: the report must run under the system interpreter (task 7 runs it as a predicate), so
  it keeps a local copy of the event names, checked against the shared constants and the orchestrator's
  `EventType` by a test.
- `no-silent-fail-soft`: an invalid reserve is inert but recorded (`reserve_config` with state invalid) and, for an
  unknown tier name, escalated once (task 3).

**Changes to the PRD's decomposition at decompose** (the PRD's section 8 records them):

1. Task 1 split into 1a (the pure rule module and the loader) and 1b (the gate), joined by a real edge: about
   1,900 to 2,100 changed lines with the B1 to B20 tests, over the overlay's band. Tasks 9804 and 9846 depend on 1b.
2. The helper that builds a task's `WorkClass` moved from task 3 to task 2: the task roles (task 2) need it first.
3. Task 5 depends on task 3 as well as 1b (G7 above).
4. Task 4 also moves the curator gate's store to `attach_cost_store` (it is passed to the constructor today, so the
   curator store would never get the `reserve_config` row task 7 checks for). It also treats the backlog
   iterator's final consolidation as a drain point, and has the mid-life judge recovery skip judges in flight
   (PRD section 6.4).
5. Tasks 7 and 8 are deterministic with delayed milestones (open question 5): task 7 a predicate one day after
   tasks 5 and 6, task 8 a pure gate for Leo 20 days after task 7.

**Producers outside this batch.** None. Tasks 9804 and 9846 (usage-gate refactor chain) are consumers, made to
depend on task 1b with a carry-through note appended to each.

## 1a — the reserve rule and the reserve: block
| Capability | Binding | Verdict |
|---|---|---|
| The loader that will read the block, which ignores unknown keys today | `grep:shared/src/shared/config_models.py::UsageCapConfig._load_accounts_file` | PASS |
| The module shared/src/shared/account_reserve.py | built here; path check, absent today | PASS |
| `WorkClass` | built here; mechanical grep, absent today | PASS |
| `admits` | built here; mechanical grep, absent today | PASS |
| The loader reads the block | built here; grep for the module's import in config_models.py, absent today | PASS |

## 1b — the gate applies the reserve
| Capability | Binding | Verdict |
|---|---|---|
| The rule, the config, the class resolver, the event names | producer:task-1a upstream | PASS |
| One selection walk, called under the gate's lock | `grep:shared/src/shared/usage_gate.py::UsageGate.before_invoke` wired: calls `try_lease` | PASS |
| The single writer of an account's phase, where the reserve waiter event is set | `grep:shared/src/shared/usage_gate.py::UsageGate._transition` | PASS |
| The single account-event writer | `grep:shared/src/shared/usage_gate.py::UsageGate._fire_cost_event` | PASS |
| A slot's cost known when it settles | `grep:shared/src/shared/usage_gate.py::InvokeSlot.confirm` | PASS |
| The retry loop that will forward the keyword | `grep:shared/src/shared/cli_invoke.py::invoke_with_cap_retry` wired: calls `usage_gate.invoke_slot` | PASS |
| `reserve_refusal` | built here; mechanical grep, absent today | PASS |
| `attach_cost_store` | built here; mechanical grep, absent today | PASS |
| `PoolFrozen.reserved_account` | built here; mechanical grep, absent today | PASS |
| `work_class` in the retry loop | built here; mechanical grep, absent today | PASS |
| A reserve wait never freezes the pool | built here; manual, boundary rows B2 and B10 | PASS |

## 2 — every orchestrator call site declares its kind of work
| Capability | Binding | Verdict |
|---|---|---|
| The keyword and the attach method | producer:task-1b upstream | PASS |
| The effective tier | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.task_priority` | PASS |
| The task roles' one call site | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._invoke` | PASS |
| The store attached today by writing the gate's private attribute | `grep:orchestrator/src/orchestrator/harness.py::Harness.run` | PASS |
| Task roles, steward and review checkpoint pass a class | built here; three mechanical greps, absent today | PASS |
| The harness attaches the store through the method | built here; mechanical grep present, and the private write absent | PASS |

## 3 — the dispatch hold and the slot-wait signal
| Capability | Binding | Verdict |
|---|---|---|
| The `WorkClass` helper and the gate's query | producer:task-2 upstream; producer:task-1b upstream | PASS |
| One eligibility test behind scored selection, the watchdog and the pin queue | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_build_candidates` wired: calls `_eligible_for_dispatch`; `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._dispatch_candidacy` wired: calls `_eligible_for_dispatch` | PASS |
| The tick's effective tiers computed before candidates are built | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_compute_priorities` (earlier in `_TICK_PHASE_ORDER`) | PASS |
| The put-back's seams | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.release` `grep:orchestrator/src/orchestrator/slot_lease.py::SlotLease.release` | PASS |
| The tier vocabulary | `grep:orchestrator/src/orchestrator/config.py::PRIORITY_TIERS` | PASS |
| `dispatch_reserve_hold` and `reserve_slot_wait` event types | built here; two mechanical greps, absent today | PASS |
| An all-held tick is not a drained queue | built here; manual, boundary row C4b | PASS |

## 4 — memory service and nightly jobs; reconciliation steps aside
| Capability | Binding | Verdict |
|---|---|---|
| The keyword, the query and the attach method | producer:task-1b upstream | PASS |
| The project loop's drain site | `grep:fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._project_loop` | PASS |
| The backlog runner's drain sites (each chunk and the final consolidation) | `grep:fused-memory/src/fused_memory/reconciliation/harness.py::BacklogIterator.run` wired: calls `_run_chunk` and `_run_final_consolidation` | PASS |
| The judge's pending marker and the recovery pass, callable after start-up as a plain async method | `grep:fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._spawn_judge` `grep:fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._recover_pending_judge_reviews` | PASS |
| The curator gate, built today with its store in the constructor | `grep:fused-memory/src/fused_memory/server/main.py::_setup_curator_usage_gate` | PASS |
| Curator, judge and census pass a class; the curator store is attached through the method | built here; four mechanical greps, absent today | PASS |

## 5 — the weekly report and the dashboard's event list
| Capability | Binding | Verdict |
|---|---|---|
| The producers of every event it reads | producer:task-1b upstream; producer:task-3 upstream | PASS |
| The dashboard's recent-account-events reader | `grep:dashboard/src/dashboard/data/costs.py` | PASS |
| A project's store path | `grep:dashboard/src/dashboard/config.py` (`runs_db`: `<root>/data/orchestrator/runs.db`) | PASS |
| scripts/allowance_reserve_report.py | built here; path check, absent today | PASS |
| The dashboard leaves out `reserve_served` | built here; mechanical grep, absent today | PASS |

## 6 — switch it on
| Capability | Binding | Verdict |
|---|---|---|
| Every caller classified and every consumer ready | producer:task-2 upstream; producer:task-3 upstream; producer:task-4 upstream | PASS |
| The shared accounts file, max-h last | `grep:config/usage-accounts.yaml` | PASS |
| The `reserve:` block | built here; mechanical grep, absent today | PASS |
| The operator section names the report | built here; mechanical grep, absent today | PASS |

## 7 — the fleet picked it up
| Capability | Binding | Verdict |
|---|---|---|
| The report's `--state` mode | producer:task-5 upstream | PASS |
| The block in the live file | producer:task-6 upstream | PASS |
| Every running orchestrator and the curator report active | built here; manual, the deterministic runner's predicate verdict | PASS |

## 8 — the two-week reading
| Capability | Binding | Verdict |
|---|---|---|
| Two full reserve weeks of records | producer:task-7 upstream | PASS |
| Leo receives the tables | built here; manual, a pure gate's escalation | PASS |
