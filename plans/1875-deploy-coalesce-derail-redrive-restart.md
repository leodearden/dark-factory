# Deploy Record: coalesce-derail re-drive fix (task 1875)

**Date:** 2026-06-23  
**Task:** dark_factory:1875 — Deploy #1867 coalesce-derail re-drive fix by restarting orchestrator-reify.service  
**Gated on:** dark_factory:1867 (LANDED)

---

## Deployed Commit

| Field | Value |
|---|---|
| Merge SHA | `fc2f8ec08f6fe012937b28c9c2d04cdf65fc51fa` |
| Merge message | `Merge task/1867 into main` |
| Merge landed | `2026-06-23 13:10:10 +0100 (BST)` |
| Key changed file | `orchestrator/src/orchestrator/merge_queue.py` (`redrive_member` callback + `_redrive_coalesce_members` method — re-dispatches absorbed members as solo merges instead of stranding them in merge-deferred when the coalesce train derails) |
| Startup log | `"Speculative merge worker started"` (`orchestrator/src/orchestrator/harness.py:4054`) |
| Test coverage | `orchestrator/tests/test_coalesce_redrive.py` |

---

## RED Baseline (pre-restart)

**Prerequisite verification:**

- `git -C /home/leo/src/dark-factory log --oneline --grep 'task/1867'` → `fc2f8ec08f Merge task/1867 into main` ✅
- `redrive_member` callback present in `merge_queue.py` lines 3454/3519 + `_redrive_coalesce_members` at line 6541 ✅
- `/home/leo/src/dark-factory` on branch `main`, tracked mods: none (committed `review/briefing.yaml` auto-update as `4f9841e391` to clear dirty-state before restart) ✅
- Merge queue: depth=0, no verify in flight ✅

The running orchestrator-reify.service process PREDATED the #1867 merge and therefore executed stale merge worker code without the coalesce-derail re-drive fix (absorbed members that derail from a coalesce train remain stranded in merge-deferred in-process):

| Field | Value |
|---|---|
| Main PID | `937088` |
| ActiveEnterTimestamp | `Mon 2026-06-22 21:05:11 BST` |
| #1867 merge landed at | `2026-06-23 13:10:10 +0100 (BST)` |
| Delta | Service started **~16 h 5 min BEFORE** the merge — loaded pre-#1867 merge_queue.py |
| ActiveState | `active` (but running stale merge worker without #1867 re-drive fix) |

**RED assertion (TRUE = deploy needed):** `ActiveEnterTimestamp (2026-06-22 21:05:11 BST) < #1867 merge timestamp (2026-06-23 13:10:10 BST)` ✅

---

## Restart Action

**Preflight checks (all passed):**
- Merge queue: empty (depth=0, no verify in flight) — clean quiesce ✅
- dark-factory tracked mods: none (committed `review/briefing.yaml` auto-update `4f9841e391` before restart) ✅
- `is_clean /home/leo/src/dark-factory` → 0 (clean) ✅

| Field | Value |
|---|---|
| Mechanism | `ORCH_PROJECT_ROOT=/home/leo/src/dark-factory /home/leo/src/reify/scripts/orchestrator-redeploy-restart.sh` — schedule mode |
| Transient unit | `orch-redeploy-restart` (timer: `orch-redeploy-restart.timer`) |
| On-active delay | 60 s after scheduling agent exits (no self-kill) |
| Exec-mode action | blocking `systemctl --user stop` → `systemctl --user start` (never `restart`) |
| project_root guard | `/home/leo/src/dark-factory` — verified clean at schedule time |
| Scheduled at | *(see script output below)* |
| Expected fire time | ~60 s after agent exits |

| Scheduled at | 2026-06-23 13:31:59 BST (transient unit armed) |
| Expected fire time | 2026-06-23 13:32:59 BST (+60 s on-active) |
| Expected active by | approx 13:35 BST (+≤90 s TimeoutStopSec graceful stop) |

Script output:

```
Running timer as unit: orch-redeploy-restart.timer
Will run service as unit: orch-redeploy-restart.service
orchestrator-redeploy-restart.sh: scheduled restart of 'orchestrator-reify.service'
  Transient unit: orch-redeploy-restart
  Fires in:       60s (after the scheduling agent exits)
  project_root:   /home/leo/src/dark-factory (clean at schedule time)
```

---

## Verification (out-of-band, post-restart)

*(to be filled in step-3 after the transient unit fires and the service completes stop→start)*

| Criterion | Expected | Actual | Status |
|---|---|---|---|
| (a) ActiveState | `active` | *(pending)* | *(pending)* |
| (b) MainPID | ≠ 937088 (fresh process) | *(pending)* | *(pending)* |
| (c) ActiveEnterTimestamp | AFTER 2026-06-23 13:10:10 BST (#1867 merge) | *(pending)* | *(pending)* |
| (d) Startup log | `"Speculative merge worker started"` in post-restart journal | *(pending)* | *(pending)* |
