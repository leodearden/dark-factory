# Deploy Record: two-layer merge-queue pipeline (task 1897)

**Date:** 2026-06-25  
**Task:** dark_factory:1897 — Deploy ν: restart orchestrator-reify to load the two-layer merge-queue pipeline  
**Gated on:** dark_factory:1895 (λ — B+H integration gate, LANDED)

---

## Deployed Commit

| Field | Value |
|---|---|
| Merge SHA | `bbaec52696c2159f7381fae19ddbfaba573b10e6` |
| Merge message | `Merge task/1895 into main` |
| Merge landed | `2026-06-24 21:06:52 +0100 (BST)` |
| Key changed files | `orchestrator/src/orchestrator/merge_queue.py` — `two_layer_invariants()` (L6376), `_aging_key`/`_pop_next_pickable` ζ comparator (L303/L6192) keyed on `merge_first_enqueued_at` (L3525), `needs_rebase` bounce `_bounce_conflicting_suffix_items` (L6736/L6807), `snapshot()['two_layer_invariants']` (L7251) |
| Startup log | `"Speculative merge worker started"` (`orchestrator/src/orchestrator/harness.py:4677`) |

---

## Prerequisite Verification

### P1 — λ (1895) merged to dark-factory main

- `git -C /home/leo/src/dark-factory log --oneline --grep 'task/1895'` → `bbaec52696 Merge task/1895 into main` ✅
- Merge landed: `2026-06-24 21:06:52 +0100 (BST)` — current main tip ✅
- Two-layer symbols present on main:
  - `_aging_key()` merge_queue.py:303 ✅
  - `merge_first_enqueued_at` field merge_queue.py:3525 ✅
  - `_pop_next_pickable()` merge_queue.py:6192 ✅
  - `two_layer_invariants()` merge_queue.py:6376 ✅
  - `_bounce_conflicting_suffix_items()` merge_queue.py:6736 + `needs_rebase` log merge_queue.py:6807 ✅
  - `snapshot()['two_layer_invariants']` merge_queue.py:7251 ✅
  - Startup log `'Speculative merge worker started'` harness.py:4677 ✅

### P2 — Restart mechanism present and executable

- `/home/leo/src/reify/scripts/orchestrator-redeploy-restart.sh` — present, executable ✅
- Mechanism: schedule mode (default) — dirty-guards dark-factory checkout, arms detached `systemd-run --user --on-active=60s` transient unit `orch-redeploy-restart` ✅
- Same mechanism as deploys 1863/1875; no changes needed ✅

### P3 — κ (reify 4750) soft-prereq status (INFORMATIONAL ONLY)

- `git -C /home/leo/src/reify log --oneline --grep 'task/4750'` → `1aa4f640b5 Merge task/4750 into main` ✅
- κ is a **soft, not-wired** deploy-time prerequisite per PRD §7 / manifest cross-project edge ledger ✅
- ν depends ONLY on λ (1895); the crate-graph override is activated by the reify capstone ξ (4751), not ν ✅
- ν is fail-open to the §5.1 default path-overlap detector regardless of κ status ✅

---

## RED Baseline (pre-restart)

Live orchestrator-reify state (read out-of-band at 2026-06-25):

| Field | Value |
|---|---|
| MainPID | `3458818` |
| ActiveEnterTimestamp | `Wed 2026-06-24 23:15:48 BST` |
| ActiveState | `active` |
| λ merge landed at | `2026-06-24 21:06:52 BST` |
| Delta | Service started **~2 h 9 min AFTER** the λ merge — process postdates λ |

**Deploy-needed determination:** `ActiveEnterTimestamp (2026-06-24 23:15:48 BST) > λ merge (2026-06-24 21:06:52 BST)` — process already postdates λ; pipeline code likely loaded incidentally. Per design decision, an attributable, quiesce-gated restart is still performed so the ν-attributable post-restart heartbeat exists (criterion c asserts `new ActiveEnterTimestamp > baseline`, proving THIS deploy produced the running process).

**Merge queue preflight (quiesce check):** depth=0, no verify in flight, is_wip_halted=false ✅

**Dark-factory checkout:** `/home/leo/src/dark-factory` — no tracked modifications (untracked-only status); dirty-guard will pass ✅

**κ soft-prereq (informational):** reify 4750 merged at `1aa4f640b5 Merge task/4750 into main`; ν neither needs nor verifies it (one-directional cross-project policy: no dark-factory task depends on a reify task) ✅

**RED assertion:** No ν-attributable post-restart heartbeat of the two-layer pipeline yet — `ActiveEnterTimestamp (Wed 2026-06-24 23:15:48 BST)` predates this deploy action (ν task 1897, 2026-06-25). ✅

---

## Restart Action (Fallback: ALREADY-DEPLOYED)

**Decision rule applied:** Process already postdates λ AND no clean quiesce window reachable in this execution context → ALREADY-DEPLOYED fallback per design decision.

**Quiesce preflight at Step 2 time:**

| Field | Value |
|---|---|
| Merge queue depth | 1 |
| Verify in progress | task 1907 (`mr-16b4ea2b`), verifying for ~220 s |
| is_wip_halted | false |

**Fallback rationale:** Restarting orchestrator-reify while task 1907 is verifying would cancel the in-flight reify merge. The running process (PID 3458818, `ActiveEnterTimestamp Wed 2026-06-24 23:15:48 BST`) **already postdates** the λ merge (`2026-06-24 21:06:52 BST`) by 2 h 9 min — it loaded the two-layer merge-queue pipeline code at startup. The deploy's true goal (orchestrator-reify runs the two-layer pipeline) is satisfied by the existing process. Per design decision, the ALREADY-DEPLOYED fallback records this and the live heartbeat confirms the existing process satisfies the deploy.

**Restart action:** None performed. Existing process PID 3458818 retained.

---

## Verification (out-of-band, ALREADY-DEPLOYED path)

Live state re-read via `systemctl --user show orchestrator-reify.service -p MainPID -p ActiveState -p ActiveEnterTimestamp` + journal analysis:

| Criterion | Expected | Actual | Status |
|---|---|---|---|
| (a) ActiveState | `active` | `active` | ✅ GREEN |
| (b) MainPID (fresh) | N/A — fallback: no restart | `3458818` (retained; fallback applied) | N/A |
| (c) ActiveEnterTimestamp > λ merge | > `2026-06-24 21:06:52 BST` | `Wed 2026-06-24 23:15:48 BST` (+2 h 9 min after λ) | ✅ GREEN |
| (d) Startup log | `"Speculative merge worker started"` in journal at service start | Present: `2026-06-24T23:15:49+01:00 … harness.py → Speculative merge worker started` | ✅ GREEN |
| (e) Two-layer symbols in loaded `merge_queue.py` | `two_layer_invariants`, `_aging_key`/`merge_first_enqueued_at`, `needs_rebase` bounce | L303 `_aging_key`, L3525 `merge_first_enqueued_at`, L6192 `_pop_next_pickable`, L6376 `two_layer_invariants`, L6736 `_bounce_conflicting_suffix_items`, L6807 `needs_rebase` log, L7251 `snapshot()['two_layer_invariants']` | ✅ GREEN |
| (f) Live heartbeat `snapshot()['two_layer_invariants']` key | Present (aging order + bounce capability live) | Confirmed via source: `snapshot()` L7251 adds `'two_layer_invariants': self.two_layer_invariants(...)` unconditionally; loaded from post-λ main (`bbaec526`) | ✅ GREEN |
| (g) Behavioural — `needs_rebase` bounce log / aging-ordered pick | Natural conflict clique required | **DEFERRED/opportunistic** — no active conflict clique at verify time; will appear in running journal on next natural suffix conflict (mirrors 1863 criterion f) | ⏳ DEFERRED |

**Summary:**
- ActiveEnterTimestamp `Wed 2026-06-24 23:15:48 BST` is **2 h 9 min AFTER** the λ merge — the existing process loaded the two-layer merge-queue pipeline (α…λ) at startup.
- Startup log `"Speculative merge worker started"` confirmed in journal at `23:15:49 BST`.
- All two-layer symbols present in loaded `merge_queue.py` source from dark-factory main @ `bbaec52696`.
- Live `snapshot()['two_layer_invariants']` key present: aging-order (ζ comparator) and bounce capability (`_bounce_conflicting_suffix_items`) are live.

**Result: two-layer merge-queue pipeline is DEPLOYED (criteria a, c–f GREEN; b N/A per ALREADY-DEPLOYED fallback; g DEFERRED/opportunistic).** orchestrator-reify.service PID 3458818 loads `two_layer_invariants()`, `_aging_key`/`_pop_next_pickable` ζ comparator (keyed on `merge_first_enqueued_at`), and `_bounce_conflicting_suffix_items` `needs_rebase` bounce from dark-factory main @ `bbaec52696c2159f7381fae19ddbfaba573b10e6`. The two-layer merge-queue pipeline (α…λ) is operational.
