# Capability manifest — merge-verdict-integrity-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), authored at decompose
2026-07-22. Symbol refs as-of DF `main 08925d962e`; every binding below was
confirmed by direct read/probe during the 5260-RCA forensics (this session +
two commissioned agent investigations). YAML sidecar twin carries
delivered_checks.

## α — INV-1 fail-closed trivial pass
- `trivial-pass-branch-exists` → **PASS (wired)** — `run_scoped_verification`
  no-source short-circuit (`orchestrator/src/orchestrator/verify.py`,
  `_trivial_pass`); failure precedent: 0ms local pass landed red `83336a32`
  (reify, 07-20 17:45Z) via empty `existing_files`.
- `guard-consult-substrate` → **PASS (wired)** — task-1774
  `_verify_pipeline_guard_requires_full_gate` + task-2838
  `_merge_config_only_diff_forces_full_gate` consults precede the trivial pass
  (verify.py); α generalizes, does not invent, the escalation path.
- `spec-empty-command-behavior` → **PASS (confirmed defect)** —
  `build_merge_verify_spec` ships `verify_commands=()` for a zero-module-config
  project (verify_runner.py); full-gate command sourcing is the fix surface.

## β — INV-2 RemoteRunner auto-sync
- `remote-invocation-seam` → **PASS (wired)** — `RemoteRunner.run_merge_verify`
  ssh chain incl. best-effort main push (verify_runner.py); defect precedent:
  laptop DF checkout frozen at `bb834dd42a` (06-11), non-FF push failing rc=1
  since ≥07-20 (workstation journal).
- `quarantine-path` → **PASS (wired)** — runner quarantine exists and fired
  2026-07-20 17:45:34Z (cross-check divergence); β's fail-closed bench reuses it.

## γ — INV-3 chain-intact adoption
- `dispatch-gate-exists` → **PASS (confirmed gap)** — dispatch-time
  discard/remerge block gated on global `_has_inflight_verify`
  (merge_queue.py, task-1862 comment block); 5260's 08:23Z dispatch ran
  against a base dead 50 min (event forensics).
- `cascade-and-discard-exist` → **PASS (wired)** — head-failure cascade +
  `speculative_discard` emission (fired correctly on 5260 attempt-2 06:34Z;
  missed attempt-3 — the dangling-edge defect γ fixes).
- `adoption-chokepoint` → **PASS (wired)** — `_finalize_inflight` →
  `_journal_landed_then_advance` → `advance_main` CAS (merge_queue.py); the
  invariant lands at this single chokepoint.
- `invariant-monitor-exists` → **PASS (wired)** — `two_layer_invariants()`
  reporting (task 2357 lineage) — γ upgrades monitor→enforcement.

## δ — INV-5 detector hardening
- `drift-counter-defect` → **PASS (confirmed defect)** — `_drift_land_count`
  in-memory (merge_drift.py); zero drift events in either project's runs.db,
  ever (event-type census).
- `shadow-sampling-gap` → **PASS (confirmed defect)** —
  `_maybe_schedule_shadow_compare` early-returns on empty `warm_results`
  (merge_shadow.py); remote/trivial lands never sampled.
- `cross-check-exists` → **PASS (wired)** — task-2822 cross-check landed
  07-20 03:22Z; divergence currently trails adoption (83336a32: +3s too late).

## ε — foreign-drift disposition
- `classifier-seam` → **PASS (wired)** — `classify_merge_failure_disposition`
  (`orchestrator/src/orchestrator/merge_disposition.py`); task 2869
  (reference-frame fix) is in-progress on the same module — hard dep wired to
  avoid conflicting edits.
- `structured-line-source` → **PASS (external, defensive)** — reify guard rule
  (c) grammar exists on reify main; ε parses opt-in and no-ops on absence, so
  no cross-project hard dependency.
- `task-diff-oracle` → **PASS (wired)** — merge-base three-dot diff is the
  established footprint oracle (used by all four 5260 dry-run proposals).

## ζ — INV-4 same-tip re-roll policy
- `retry-ledger-substrate` → **PASS (wired)** — `retry_ledger`
  (`consecutive_merge_thrash`, `last_merge_outcome_signature`) persists on task
  metadata (observed live on task 5260's record).
- `tip-identity-oracle` → **PASS (probed)** — `merge_sha^2` = branch tip;
  validated across 138 determinate requeue pairs in the mining run.
- `formation-gate-precedent` → **PASS (wired)** — task-1720 coalescer
  confidence gate (formation-only); ζ extends exclusion to spec_base
  selection, which 1720 does not cover (5213 became a train head through it).
- `evidence-numeric` (G6) → **PASS** — cap=1 rationale from measured 38% win /
  1.24 waste ratio (n=34); recorded as config default, not an assertion.

No FAIL bindings; batch clear to queue.
