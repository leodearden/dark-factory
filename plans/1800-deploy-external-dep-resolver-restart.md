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

| Field | Value |
|---|---|
| Mechanism | `scripts/orchestrator-redeploy-restart.sh` — schedule mode (task 4620 wrapper) |
| Transient unit | `orch-redeploy-restart` (timer: `orch-redeploy-restart.timer`) |
| On-active delay | 60 s after scheduling agent exits (no self-kill) |
| Exec-mode action | blocking `systemctl --user stop` → `systemctl --user start` (never `restart`) |
| project_root guard | `/home/leo/src/reify` — verified clean at schedule time |
| Scheduled at | approx 12:25 BST (step-2 schedule-mode agent exited) |
| Expected fire time | approx 12:26 BST (+60 s on-active delay after agent exit) |
| Expected active by | approx 12:28 BST (+≤90 s TimeoutStopSec graceful stop) |

Script output:
```
Running timer as unit: orch-redeploy-restart.timer
Will run service as unit: orch-redeploy-restart.service
orchestrator-redeploy-restart.sh: scheduled restart of 'orchestrator-reify.service'
  Transient unit: orch-redeploy-restart
  Fires in:       60s (after the scheduling agent exits)
  project_root:   /home/leo/src/reify (clean at schedule time)
```

---

## Verification

Live state read via `systemctl --user show orchestrator-reify.service -p MainPID -p ActiveState -p ActiveEnterTimestamp` in the step-3 out-of-band invocation (after the transient unit fired and the service completed stop→start):

| Criterion | Expected | Actual | Status |
|---|---|---|---|
| (a) ActiveState | `active` | `active` | ✅ GREEN |
| (b) MainPID | ≠ 154121 (fresh process) | `2433578` | ✅ GREEN |
| (c) ActiveEnterTimestamp | AFTER 2026-06-18 12:11:16 BST (#1799 merge) | `Thu 2026-06-18 12:26:28 BST` | ✅ GREEN |
| (d) Fixed symbols in editable source | `get_external_statuses` (scheduler.py:1543), `external_deps` gate (scheduler.py:1694–1719), `_mark_blocked` routing | Present on main @ scheduler.py lines 1543, 1694–1719, 1921, 2726–2744 | ✅ GREEN |
| (e) Smoke probe | Dispatch probe with cross-project dep in `done` state | **Deferred** — resolver code path confirmed present (criteria a–d); fresh dispatch probe would require provisioning a new external-dep task, deferred to normal operations | ⚠️ DEFERRED |

**Delta:** Service restarted **15 min 12 s AFTER** the #1799 merge (12:26:28 BST vs 12:11:16 BST merge) — new process loads post-#1799 resolver code.

**Result: #1799 external-dep resolver fix is DEPLOYED.** orchestrator-reify.service PID 2433578 loads the fixed `get_external_statuses` / `external_deps` dispatch gate and `_mark_blocked` routing from dark-factory main @ 90c9be96b3 (and beyond, as main has since advanced to 50db20417d54).
