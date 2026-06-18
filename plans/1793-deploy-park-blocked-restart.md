# Deploy Record: park→blocked / open-L2 fix (task 1793)

**Date:** 2026-06-18  
**Incident:** 4641  
**Task:** dark_factory:1793 — Deploy park→blocked/open-L2 fix by restarting orchestrator-reify.service  
**Gated on:** dark_factory:1792 (LANDED)

---

## Deployed Commit

| Field | Value |
|---|---|
| Merge SHA | `3b3730a2e34f5af119c793952281dc1dbb0a3801` |
| Merge message | `Merge task/1792 into main` |
| Impl commit | `5bc2dabace` — `impl(1792 step-2): GREEN — _ACTION_TARGETS['park']='blocked' (version-a)` |
| Key change | `harness.py:4795`: `_ACTION_TARGETS = {'restart':'pending','park':'blocked','abandon':'cancelled'}` |
| Test coverage | Commits `b6d2360776`, `d1edc41bc3`, `27a4b7e6aa` — assert park→blocked write, L2 stays open, cluster sweep quiescent |

---

## RED Baseline (pre-restart)

The running orchestrator-reify.service process PREDATED the 1792 merge and therefore executed stale code where `park` resolved to `'deferred'`:

| Field | Value |
|---|---|
| Main PID | 349637 |
| ActiveEnterTimestamp | Thu 2026-06-18 09:36:09 BST |
| 1792 merge landed at | 2026-06-18 10:09:40 +0100 (BST) |
| Delta | Service started **33 min 31 s BEFORE** the merge — loaded pre-1792 code |
| ActiveState | active (but running stale harness) |

---

## Restart Action

| Field | Value |
|---|---|
| Mechanism | `scripts/orchestrator-redeploy-restart.sh` — schedule mode (task 4620 wrapper) |
| Transient unit | `orch-redeploy-restart` (timer: `orch-redeploy-restart.timer`) |
| On-active delay | 60 s after scheduling agent exits (no self-kill) |
| Exec-mode action | blocking `systemctl --user stop` → `systemctl --user start` (never `restart`) |
| project_root guard | `/home/leo/src/reify` — verified clean at schedule time |
| Scheduled at | approx 10:10 BST (schedule-mode agent exited) |
| Expected fire time | approx 10:11 BST (+60 s on-active delay after agent exit) |
| Expected active by | approx 10:13 BST (+≤90 s TimeoutStopSec graceful stop) |

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

## Verification (out-of-band, post-restart)

Performed in step-3 (separate post-restart invocation — see below).

**GREEN criteria:**
- `systemctl --user show orchestrator-reify.service -p ActiveState` → `active`
- Main PID **differs** from 349637 (fresh process)
- `ActiveEnterTimestamp` is **after** `2026-06-18 10:09:40 +0100`
- `_ACTION_TARGETS['park'] == 'blocked'` in the editable harness.py loaded by the new process

**Outcome:** _TBD — to be filled by step-3 out-of-band check_
