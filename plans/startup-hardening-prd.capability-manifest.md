# Capability manifest — `plans/startup-hardening-prd.md`

Per-leaf capability → evidence bindings (G3 + G6 mechanized). Machine-readable twin:
`plans/startup-hardening-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).
Verified against main `922d9d3baa` (2026-11-02): every `expect: present` pattern below is ABSENT today, so
each check gates something.

| leaf | capability | binding | verdict |
|---|---|---|---|
| α | runner module exists | new file `orchestrator/src/orchestrator/startup_steps.py`; grep `class StartupStep` | PASS |
| α | `startup_step_failed` event type | grep in `orchestrator/src/orchestrator/event_store.py` (`EventType` StrEnum exists; additive member) | PASS |
| α | L1 per failed step, deduped | wired via `EscalationQueue.get_by_task`/`make_id`/`submit` (the `_file_dirty_tree_escalation` idiom, present on main) | PASS |
| α | ad-hoc blocks removed | rejection-style: grep `Failed to recover pending merges` absent in `harness.py` after α (present today, 1c0a's block; 2c2's `startup continues` string is split across two lines and never matched) | PASS |
| β | loop launched after `finish_startup` | grep `_launch_merge_worker_loop` in `harness.py` | PASS |
| β | enqueue without a consumer | `self._merge_queue: asyncio.Queue` in `Harness.__init__`; `_recover_pending_merges` enqueues via `enqueue_merge_request` — present on main | PASS |
| β | stop-guard invariant | `_stop_merge_worker` already gates on `_merge_worker_task is not None` — present; β pins it with a test | PASS |
| γ | journal + quarantine | grep `orchestrator-startup-crash-loop` in `orchestrator/src/orchestrator/` | PASS |
| γ | runs columns | grep `startup_completed_at` and `exit_reason` in `run_store.py`; `add_column_if_missing_sync` present on main | PASS |
| γ | born-at-L2 from the harness | `Escalation(level=2, severity='critical')` via the queue — the dirty-tree guard files exactly this today | PASS |
| δ | NRestarts read | grep `NRestarts` in `scripts/orchestrator-watchdog.py` (0 today) | PASS |
| δ | page into dark-factory's queue | `escalation submit` CLI (`escalation/src/escalation/submit.py`) with `--queue-dir/--task/--severity/--category/--summary/--detail/--agent-role` — present; the dashboard watchdog calls it | PASS |
| δ | liveness min-interval | grep `ORCH_LIVENESS_RESTART_MIN_INTERVAL_SECS` in the watchdog | PASS |
| ε | units resized | grep `StartLimitIntervalSec=1800` in `scripts/orchestrator-*.service` (0 today, 600 on all nine) | PASS |
| ζ | units installed + fleet restarted | manual — human deploy gate; `check_orchestrator_unit_parity.py` is the post-condition | PASS |
| η | producers wrote in production | manual — delayed milestone 7 days after ζ, record appended to PRD §9 | PASS |

No binding resolves to declared-only, test-only, producer-downstream or producer-absent: γ depends on α and
β, ζ on every code leaf, η on ζ; δ and ε are independent producers consumed by ζ.
