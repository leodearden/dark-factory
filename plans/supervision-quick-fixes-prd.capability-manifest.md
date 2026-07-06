# Capability manifest — supervision-quick-fixes PRD

Per-leaf capability→evidence bindings (mechanizing G3 + G6) for
`plans/supervision-quick-fixes-prd.md`. All bindings verified against main on
2026-07-06 in the authoring session. All five tasks are leaves (no intra-batch deps).
No numeric bounds, no grammar fixtures in this batch; the one negative assertion (γ:
"malformed probe is rejected") is bound to an observed-on-main rejection mechanism.

## α — hoist `inspect_systemd_unit`

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Hardened wait_for+kill+bounded-reap+sentinel inspector pattern exists (the code to hoist) | grep:orchestrator/src/orchestrator/deterministic_runner.py:399 (`asyncio.wait_for(proc.communicate(), timeout=self._inspect_timeout_secs)`), :413 (`proc.kill()`), :415 (bounded reap), :429-434 (sentinel dict) | PASS (wired) |
| Harness duplicate with bare communicate (the defect site) | grep:orchestrator/src/orchestrator/harness.py:382 (`stdout, _ = await proc.communicate()` — no wait_for) | PASS (defect confirmed present) |
| Harness inspector is wired into the production recon-sweep entry path | grep:harness.py:7699 (`inspect_fn = self._recon_unit_inspector or _recon_inspect_unit`), :7922 (`_run_deterministic_recon_sweep`), :8098 (sweep invoked from watcher loop) | PASS (wired) |
| Runner injectable seam preserved for delegation | grep:deterministic_runner.py:1195 (`self._unit_inspector or self._default_inspect_unit`) | PASS (wired) |
| WARNING sentinel log line the signal observes | grep:deterministic_runner.py:416-420 (existing pattern to be emitted from the shared helper) | PASS |

## β — agent_role-scope DeterministicRunner escalation queries

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `Escalation` model carries `agent_role` | grep:escalation/src/escalation/models.py:53 (`agent_role: str`) | PASS |
| `get_by_task(task_id, status, level)` signature to extend | grep:escalation/src/escalation/queue.py:309-311 | PASS |
| Runner files ALL its escalations with the sentinel (so scoping keeps resolution proof) | grep:deterministic_runner.py:586, :804 — the only two runner filing sites, both `agent_role='orchestrator-deterministic'` | PASS (wired) |
| The five unscoped query sites to scope | grep:deterministic_runner.py:575, :793, :861, :922, :952 | PASS (defect confirmed present) |
| Aliasing producer exists (starvation watchdog files same-task_id escalations with a different role) | grep:harness.py:~3940-3990 (`agent_role='orchestrator-starvation-watchdog'`) | PASS |
| Archive-inclusive scan the ever_escalated signal relies on | grep:queue.py:336-338 (`status != 'pending'` extends paths with archive) | PASS |

## γ — substrate-probe fail-closed at the dispatch gate

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Key-presence predicate exists in substrate_gate | grep:orchestrator/src/orchestrator/substrate_gate.py:81-112 (`carries_substrate_probe`) | PASS |
| Rejection mechanism exists and produces FLIP on malformed descriptor (G6 branch-4 rejection-check) | grep:substrate_gate.py:265-280 — `if descriptor is None: if carries_substrate_probe(task): … return SubstrateVerdict(verdict=FLIP, … 'declared but malformed — failing closed')`; observed on main | PASS (rejection mechanism present; currently unreachable from production — exactly what this task wires) |
| The fail-open wrapper (defect site) | grep:orchestrator/src/orchestrator/scheduler.py:1412-1428 (`extract_probe_set(task) is not None`) | PASS (defect confirmed present) |
| Production gate call site to rewire | grep:orchestrator/src/orchestrator/harness.py:5124 (`self.scheduler.carries_substrate_probe(assignment.task) and not await self._run_substrate_gate(...)`) | PASS (wired) |
| SKIP path for genuinely-absent key (companion assertion) | grep:substrate_gate.py:281-287 (SKIP verdict when key absent) | PASS |

## δ — module_charter.py derive/sanitize + single-writer cache

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| shared.locking primitives to compose | grep:shared/src/shared/locking.py:101 (`directory_locks`), :120 (`strip_directory_locks`), :151 (`files_to_modules`) | PASS |
| The 4 duplicated derivation/write sites | grep:scheduler.py:4700-4753 (`_get_modules`), :4316-4339 (`_persist_files_metadata`), :4456 (`handle_blast_radius_expansion` cache write); harness.py:1791-1945 (`_tag_task_modules`, direct cache poke at :1939, writeback strip at :1921) | PASS (defect confirmed present) |
| Dual-writer cache / no write-through on derive (defect) | grep:scheduler.py:4742-4744 (derive returns without caching), :990 (`_module_cache` decl); writers at harness.py:1939 and scheduler.py:4456 | PASS (defect confirmed present) |
| Fused-memory lock-charter LOUD-reject the signal's "no rejection warning" observes | task 1833 (done) — submit/update guard in fused-memory tools; 54ec90fefc incident commit documents a rejected payload | PASS (producer upstream, landed) |
| Deterministic short-circuit + task-id fallback semantics preserved | grep:scheduler.py:4713-4715, :4745-4753 | PASS |

## ε — StreakCounter/StreakRegistry migration

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| The five counters at claimed sites | grep:scheduler.py:1072-1109 (declarations); :2104-2160 (`_external_unresolved_counts`); :2001-2060 (`_external_resolver_degraded_counts`); :1913-1922 (`_external_hold_streak`/`_cause`, cause-change reset); :3396-3488 (three inline `_local_backfill_unresolved_counts` loops); :2179-2287 (`_starvation_first_seen`/`_starvation_escalated`) | PASS (defect confirmed present) |
| Manual GC-sweep enumeration the signal asserts is replaced | grep:scheduler.py:3533-3633 (per-dict blocks with carve-out comments at :3571-3574, :3611-3614) | PASS |
| Starvation GC-resolve callback semantics to preserve | grep:scheduler.py:3615-3633 (`_on_starvation_resolve`, `_STARVATION_NON_ELIGIBLE`) | PASS |
| Existing parity test coverage (the signal's "existing tests pass unchanged" is producible) | grep:orchestrator/tests/test_scheduler.py + orchestrator/tests/test_cross_project_dispatch_integration.py reference `_external_hold_streak`/`_external_unresolved_counts`/`_starvation_first_seen` | PASS |

## FAIL bindings

None. No `declared-only`, `test-only`, `producer-absent`, `producer-downstream`,
`producer-extent-short`, `fixture-ERROR`, `bound≤floor`, or `rejection-absent`
bindings — the batch is clear to file.
