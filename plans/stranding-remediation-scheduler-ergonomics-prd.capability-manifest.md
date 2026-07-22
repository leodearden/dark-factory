# Capability manifest — stranding-remediation-scheduler-ergonomics-prd

Bindings authored at decompose 2026-07-22; symbol refs as-of DF `main
08925d962e`. All substrate confirmed by direct read/event evidence during the
5260 RCA. YAML sidecar twin beside this file.

## α — merge-queue-direct stranded remediation
- `stranded-reaper-exists` → **PASS (wired)** — harness stranded-blocked
  reconcile filed `esc-5260-9` (`agent_role=harness-stranded-blocked-reaper`,
  07-21 10:04Z journal + escalation archive).
- `verified-green-shape-readable` → **PASS (wired)** — `LaneRecord` durable
  state (`.lane-state/_lane-9.json` read), `task_runtime._derive_phase`
  (plan.json all-done → 'DONE'), `workflow_verify` events carry `tip_sha`.
- `merge-queue-runs-under-pause` → **PASS (probed)** — `mr-973ad563`
  submitted, verified, landed 07-21/22 while `ewa_trip_29.1818` pause active.
- `born-at-L2-path` → **PASS (wired)** — existing born-at-L2 filing
  (severity/agent_role sentinel) per CLAUDE.md deterministic-runner section.

## β — pause-aware re-pend + pause visibility
- `pause-state-durable` → **PASS (wired)** — "Scheduler pause persisted from
  prior run — restoring" (journal 07-21); pause reason readable in-process.
- `resolution-cascade-seam` → **PASS (wired)** — "cascade-unblock: task 5260
  flipped blocked→pending" (harness journal line = the exact hook point).
- `digest-writer` → **PASS (wired)** — `digest_mod.write_digest_entry` +
  `DigestInputs` (harness.py `_maybe_write_digest`).

## γ — terminal-task lane reclamation + stale census
- `lane-lifecycle-transitions` → **PASS (wired)** — `LaneLifecycle.transition`
  + `LEGAL_TRANSITIONS` incl. `ASSIGNED→RELEASED` (lane_lifecycle.py, read).
- `gc-pass-cadence` → **PASS (wired)** — "Warm-lane GC reclaim pass" runs
  ~every 12 min (journal); γ can piggyback or timer (tactical).
- `branch-survives-release` → **PASS (asserted at runtime)** —
  branch-lifecycle decouple PRD landed
  (`plans/warm-lane-branch-lifecycle-decouple-prd.md`); γ additionally asserts
  the branch ref resolves before each release (defensive, per PRD decision 4).
- Failure-mode precedent (G6-adjacent): pool census 07-21 = 41 assigned + 7
  quarantined / 48, zero free; storm = 22+ pool-exhausted retry-caps in 2.5h.

## δ — EWA trip storm annotation
- `trip-writes-digest-first` → **PASS (wired)** — `_maybe_write_digest` step
  (14) pauses AFTER writing the digest (harness.py, read directly); the
  category tally is already computed in the same function (escalation_stats).

No FAIL bindings; batch clear to queue.
