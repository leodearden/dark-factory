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

## Verification
