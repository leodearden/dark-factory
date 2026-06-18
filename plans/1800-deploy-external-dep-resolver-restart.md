# Deploy Record: cross-project external-dep resolver fix (task 1800)

**Date:** 2026-06-18  
**Task:** dark_factory:1800 — Deploy external-dep resolver fix by restarting orchestrator-reify.service  
**Gated on:** dark_factory:1799 (LANDED)

---

## Deployed Commit

| Field | Value |
|---|---|
| Merge SHA | `90c9be96b3d7b9577b330af21759036a51cb05a3` |
| Merge message | `Merge task/1799 into main` |
| Merge landed | `2026-06-18 12:11:16 +0100 (BST)` |
| Key changed symbol | `Scheduler.get_external_statuses` — `orchestrator/src/orchestrator/scheduler.py:1543` |
| Key changed symbol | `external_deps dispatch gate` — `orchestrator/src/orchestrator/scheduler.py:1694–1719` |
| Key changed symbol | `_mark_blocked routing for cancelled/unresolvable external deps` — `scheduler.py` |
| Key changed symbol | `fused-memory get_external_statuses` — `fused-memory/src/fused_memory/server/tools.py:2152` |
| Impl commits | `019b77206c` (`get_external_statuses` + loud-fail), `34d8c1be11` (gate_held event), `494517606b` (held_live tracking), `ae15f6e055` (blocked_this_tick + RecordingEventStore) |

---

## RED Baseline (pre-restart)

The running orchestrator-reify.service process PREDATED the 1799 merge and therefore executed stale code with the silent external-dep gate bug (cross-project deps always evaluated as not-satisfied):

| Field | Value |
|---|---|
| Main PID | 154121 |
| ActiveEnterTimestamp | Thu 2026-06-18 10:29:26 BST |
| 1799 merge landed at | 2026-06-18 12:11:16 +0100 (BST) |
| Delta | Service started **1 h 41 m 50 s BEFORE** the merge — loaded pre-#1799 code |
| ActiveState | active (but running stale resolver) |

**RED assertion (TRUE = deploy needed):** `ActiveEnterTimestamp (10:29:26 BST) < 1799 merge timestamp (12:11:16 BST)` ✅

---

## Restart Action

*(To be appended in step-2 after the detached restart is scheduled.)*

---

## Verification

*(To be appended in step-3 out-of-band, after the transient unit fires and the service restarts.)*
