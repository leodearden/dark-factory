# Capability manifest — recurring-deterministic-tasks

PRD: `docs/prds/recurring-deterministic-tasks.md` · verified against main
`2ae349284e` (2026-08-24) · machine-readable twin:
`docs/prds/recurring-deterministic-tasks.capability-manifest.yaml`

Mechanises G3 + G6 per leaf. **15 capabilities across 6 tasks — 15 PASS, 0 FAIL,
0 OPEN.** 7 mechanical `delivered_check`s, 8 `manual`. Every `grep` pattern
returns **0 hits today**, each against a positive control
(`register_metadata_submodel('milestone'` → 1 in `shared/`,
`milestone_check_failed` → present in `deterministic_runner.py`,
`reap-decisions` → 1 in the watcher skill), so each is false-before/true-after.
Note the deliberately avoided trap: `milestone_check_failed` in
`deterministic_runner.py` is **already present** (the predicate leg), so r3's
delivery check binds on `recurrence` in that file (0 today) plus the behavioural
B7, never on the category string itself.

## Per-task bindings

### r1 — Recurrence submodel + blessing + guard rules (intermediate; unlocks r2, r3, r4)
- Pattern to reuse verified present: `Milestone` submodel +
  `register_metadata_submodel('milestone', …)` + `_validate_milestone` guard rule
  (submit-time ValidationError) — substrate PASS; this task clones the pattern.
- Delivery: grep `register_metadata_submodel\(\s*.recurrence.` in `shared/` (0
  today, control `milestone` form → 1); grep `recurrence` in
  `deterministic_task_guard.py` (0 today).
- Rejection quality (B6 shapes incl. deploy-kind rejection): `manual`.

### r2 — mint-on-terminal at the post-lock seam (intermediate; unlocks r6)
- Post-lock seam exists: step-5/6 fire-and-forget region after `_write_lock`
  release, discriminating `STATUS_TRIGGERS` (done/cancelled) — verified;
  **non-reentrant lock verified** (interceptor-level submit from inside the
  transition deadlocks; the seam avoids it). Substrate PASS.
- Backend add path exists: `tm.add_task` (`SqliteTaskBackend.add_task`) — PASS.
- Delivery: grep `recurrence-mint` (C-3's `metadata.source` value) in
  `fused-memory/src/fused_memory/` (0 today).
- Replay dedup (B4) and fail-soft (B5): `manual`.

### r3 — carrier timeout-leg category (leaf)
- Today-value verified: timeout leg files `category='infra_issue'`; predicate
  leg files `milestone_check_failed`; both at `agent_role=
  'orchestrator-deterministic'` — substrate PASS (B7's before-side is real).
- Delivery: grep `recurrence` in `deterministic_runner.py` (0 today — the
  carrier-scoping must name it); category flip itself: `manual` (B7).

### r4 — dashboard chain panel (leaf)
- Metadata reaches the dashboard per-row (`_shape_task` carries `metadata`) —
  substrate PASS. Depends cross-PRD on detection d1's endpoint + pool path
  (dep wired at decompose; DAG-direction correct).
- Delivery: grep `recurrence` in `dashboard/` (0 today).
- State-function totality (B8, broken-vs-absent): `manual`.

### r5 — watcher chain-triage extension (leaf, non-code)
- Delivery: grep `overdue` across both skill SKILL.md files (0 today, control
  `reap-decisions` → 1).

### r6 — seed the first two chains (leaf, integration gate)
- Seed jobs verified real: `reify-closure-staleness-sweep.sh` (exists,
  executable, stdlib-only, always-exit-0 — needs the predicate variant this task
  writes; its `$sweep_rc`/`$consumer_rc` already captured) and the
  transcript-check command (`check_transcript_persistence.py` via uv, unit never
  installed) — substrate PASS.
- Delivery: grep `reify-closure-staleness-predicate` in `scripts/` (0 today).
- End-to-end link→successor demonstration (B10): `manual` — the task IS the
  integration gate; its signal is the product-read-path chain evidence.
