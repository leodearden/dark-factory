# Deploy Record: in-flight-tip speculation fix (task 1863)

**Date:** 2026-06-22  
**Task:** dark_factory:1863 — Deploy #1862 in-flight-tip speculation fix by restarting orchestrator-reify.service  
**Gated on:** dark_factory:1862 (LANDED)

---

## Deployed Commit

| Field | Value |
|---|---|
| Merge SHA | `b41d665826050c8a9541501702de19fe19d930b8` |
| Merge message | `Merge task/1862 into main` |
| Merge landed | `2026-06-22 20:51:23 +0100 (BST)` |
| Key changed file | `orchestrator/src/orchestrator/merge_queue.py` (+133 — in-flight-tip / pending-spec-base lifecycle + disjoint-skip carve-out) |
| Startup log | `"Speculative merge worker started"` (`orchestrator/src/orchestrator/harness.py:4006`) |
| Test coverage | `orchestrator/tests/test_merge_speculation.py` (+1498 lines) |

---

## RED Baseline (pre-restart)

Verified prerequisite: `git -C /home/leo/src/dark-factory log --oneline main | grep 1862` → `b41d665826 Merge task/1862 into main` ✅

The running orchestrator-reify.service process PREDATED the #1862 merge and therefore executed stale merge worker code without the in-flight-tip speculation fix (disjoint-skip semantic-conflict hole remains open in-process):

| Field | Value |
|---|---|
| Main PID | `3454423` |
| ActiveEnterTimestamp | `Mon 2026-06-22 19:30:29 BST` |
| #1862 merge landed at | `2026-06-22 20:51:23 +0100 (BST)` |
| Delta | Service started **1 h 20 m 54 s BEFORE** the merge — loaded pre-#1862 merge_queue.py |
| ActiveState | `active` (but running stale merge worker without #1862 fix) |

**RED assertion (TRUE = deploy needed):** `ActiveEnterTimestamp (19:30:29 BST) < #1862 merge timestamp (20:51:23 BST)` ✅

---

## Restart Action

**Preflight checks (all passed):**
- Merge queue: empty (depth=0, no verify in flight) — clean quiesce ✅
- leo-laptop reachability: `ssh leo-laptop true` exit 0 — SSH-reachable, but K=1 expected (verify_runners not yet enabled in reify config per task 1716 gate; SSH reachability is necessary but not sufficient) ✅
- dark-factory tracked mods: none (only untracked `??` files) — dirty-tree guard passes ✅

| Field | Value |
|---|---|
| Mechanism | `ORCH_PROJECT_ROOT=/home/leo/src/dark-factory /home/leo/src/reify/scripts/orchestrator-redeploy-restart.sh` — schedule mode |
| Transient unit | `orch-redeploy-restart` (timer: `orch-redeploy-restart.timer`) |
| On-active delay | 60 s after scheduling agent exits (no self-kill) |
| Exec-mode action | blocking `systemctl --user stop` → `systemctl --user start` (never `restart`) |
| project_root guard | `/home/leo/src/dark-factory` — verified clean at schedule time |
| Scheduled at | approx 21:03 BST (step-2 schedule-mode agent exited) |
| Expected fire time | approx 21:04 BST (+60 s on-active delay after agent exit) |
| Expected active by | approx 21:06 BST (+≤90 s TimeoutStopSec graceful stop) |
| K expectation | K=1 (verify_runners not yet enabled in reify config per task 1716 gate; leo-laptop SSH-reachable but runner config not enabled — SSH reachability alone is not the gating condition) |

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

Live state read via `systemctl --user show orchestrator-reify.service -p MainPID -p ActiveState -p ActiveEnterTimestamp` after the transient unit fired and the service completed stop→start:

| Criterion | Expected | Actual | Status |
|---|---|---|---|
| (a) ActiveState | `active` | `active` | ✅ GREEN |
| (b) MainPID | ≠ 3454423 (fresh process) | `937088` | ✅ GREEN |
| (c) ActiveEnterTimestamp | AFTER 2026-06-22 20:51:23 BST (#1862 merge) | `Mon 2026-06-22 21:05:11 BST` | ✅ GREEN |
| (d) Startup log | `"Speculative merge worker started"` in post-restart journal | Present at `21:05:13 BST` | ✅ GREEN |
| (d2) K | K=1 (verify_runners not yet enabled per task 1716 gate) | `hosts_total=1` (K=1 serial-local; verify_runners not yet enabled in reify config per task 1716 gate — expected and actual agree; leo-laptop SSH-reachable but not the gating condition) | ✅ GREEN |
| (e) #1862 fix in loaded source | `pending_spec_base`, disjoint-skip carve-out in `merge_queue.py` | Present: `pending_spec_base` at lines 6841/6854/6857/6858/6878; disjoint gate at lines 1829/1924/1972 | ✅ GREEN |
| (f) Behavioral (deferred) | Follower arriving mid-predecessor-verify emits `speculative_merge` with `base_sha == predecessor merge SHA`; predecessor landing shows no `_reverify_rebased_tree` / `rebased_pending_reverify` | **Deferred** — structural criteria (a)–(e) confirm fresh process running with #1862 source on disk; behavioral path requires a natural mid-flight predecessor/follower cascade; confirmed opportunistically on next real cascade | ⚠️ DEFERRED |

**Delta:** Service restarted **13 min 48 s AFTER** the #1862 merge (21:05:11 BST vs 20:51:23 BST merge) — new process loads post-#1862 merge_queue.py.

**Result: #1862 in-flight-tip speculation fix is DEPLOYED (structural criteria a–e GREEN).** orchestrator-reify.service PID 937088 loads the fixed `pending_spec_base` lifecycle and disjoint-skip carve-out from dark-factory main @ b41d665826.

---

### Open Follow-up: Behavioral Verification (criterion f)

- [ ] **Owner:** Reify orchestrator on-call / first agent observing a mid-flight cascade post-deploy
- **Trigger:** Next natural mid-flight predecessor/follower merge — a follower task arrives while a predecessor is in the `verify` phase
- **What to confirm:**
  - Follower emits `speculative_merge` with `base_sha == predecessor_merge_sha` (not plain `main` HEAD)
  - Predecessor's landing journal shows **no** `_reverify_rebased_tree` / `rebased_pending_reverify` event
- **Resolution:** Check the box, record the qualifying cascade SHA and timestamp in this file, and close the follow-up. If a natural cascade does not occur within a reasonable window, a synthetic predecessor/follower race can be provisioned as a follow-up task.
