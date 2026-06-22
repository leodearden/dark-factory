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

## Restart Action

## Verification
