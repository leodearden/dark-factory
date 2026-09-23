# Capability manifest — claimant-invariant-detection

PRD: `docs/prds/claimant-invariant-detection.md` · verified against main
`2ae349284e` (2026-08-24) · machine-readable twin:
`docs/prds/claimant-invariant-detection.capability-manifest.yaml`

Mechanises G3 + G6 per leaf. **13 capabilities across 5 tasks — 13 PASS, 0 FAIL,
0 OPEN.** 8 mechanical `delivered_check`s, 5 `manual`. Every `grep` pattern was
measured on this HEAD with a known-positive control: all 8 mechanical patterns
return **0 hits today** (controls: `escalation-analytics` → 12 in `dashboard/`,
`reap-decisions` → 1 in the watcher skill, `register_metadata_submodel('milestone'`
→ 1 in `shared/`, `claimant` → 1+ in OPERATIONS.md), so each is
false-before/true-after, not the neighbouring-name trap.

## Per-task bindings

### d1 — dashboard gauge + invariants endpoint (intermediate; unlocks d2, d5, recurrence r4)
- Predicates consumed from enforcement α: **producer:task-4618 upstream**, dep
  wired at decompose (DAG-direction PASS). α's own delivered_checks gate this.
- `DbPool` per-root idiom + read-only `?mode=ro` pool: verified present
  (`dashboard/data/db.py::DbPool`; opener idiom in `app.py`) — substrate PASS.
- Endpoint delivery: grep `dashboard/invariants` in `dashboard/` (0 today).
- Per-root tasks.db pool path: grep `tasks/tasks\.db` in `dashboard/` (0 today) —
  also pins the doubled-`tasks`-dir path against the known 0-byte decoy.
- Blind-state rendering: `manual` — judged by boundary test B2, not greppable.

### d2 — watcher consumption step (leaf, non-code)
- Both skills exist **only** in-repo (no `~/.claude` copies — verified), each with
  a numbered "## The Main Loop": substrate PASS.
- Delivery: grep `dashboard/invariants` across both skill dirs (0 today).

### d3 — C4-E2 ledger stamp (leaf)
- Explicit supply detectable: `requested_claimant_write` /
  `_CLAIMANT_WIRE_UNSET` tri-state verified in `task_interceptor.py` — PASS.
- Atomic ride: `audit_fields` → `set_status_and_stamp_audit` single-txn
  sibling-preserving metadata merge verified (`sqlite_task_backend.py`) —
  `manual` (atomicity is a test property, B5).
- Ledger delivery: grep `claimant_exception` in `fused-memory/src/fused_memory/`
  (0 today).
- Key blessed: grep `claimant_exception` in `shared/src/shared/` (0 today).

### d4 — set_task_claimant terminal-observation ERROR (leaf)
- In-txn SELECT already present in `set_task_claimant` (extending it to `status`
  is zero-round-trip) — substrate PASS, verified.
- Delivery: grep `claimant_stamped_on_terminal` (the prescribed structured-log
  discriminator) in `fused-memory/src/fused_memory/` (0 today).
- Write-still-succeeds (observation, not refusal): `manual` — B8.

### d5 — contract amendment + OPERATIONS.md (leaf, non-code)
- C4-E7 target text located and quoted; task 4626 (η) confirmed scoped to
  `plans/task-status-authority-prd.md` only — no collision. Substrate PASS.
- Delivery: grep `claimant-invariant-detection` in
  `docs/prds/claimant-invariant-enforcement.md` (0 today) and in `OPERATIONS.md`
  (0 today).
