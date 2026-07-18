# PRD: CPU-load-robust merge verify (autonomously-landable batch)

_Author: /deb → /prd, 2026-07-18. Source RCA: memory
`project_cpu_contention_test_flakiness_rootcause_2026_07_18`; verified against
`main` `25e03d4db0`. This PRD scopes ONLY the autonomously-landable (green /
code) fixes. The RED-TIER levers (`merge_verify_breadth` narrow,
`verify_runners`, host-global cross-project verify semaphore) are deliberately
**out of scope** and remain a human decision tracked on cockpit
`host-verify-cpu-oversubscription-df`._

## 1. Consumer + user-observable surface (G1, G2)

**Consumer:** the orchestrator merge pipeline (`role='merge'`
`run_scoped_verification`) and the fleet operator. **Producer of the problem:**
concurrent full `-n auto` verify pools oversubscribe the 32-core host; a
CPU-starved-but-correct test crosses the 60s **wall-clock** pytest-timeout →
`thread` method `os._exit(1)` → xdist `node down` → a *different, innocent* test
is reported FAILED each run → a correct, code-complete task is blocked from
landing and thrashes merge cycles (task 2700 burned 4).

**User-observable surface:** a code-complete task whose own tests are green
lands on `main` without being red-flagged by an unrelated load-induced flake in
another package; and when a flake *is* suppressed, the operator sees a
structured fact (and a loud escalation if suppression becomes chronic), never a
silent green.

## 2. Motivation / premise validation (G6)

Established this session by code + durable verify-logs (see the RCA memory):

- **Mechanism A (primary, node-down):** `os._exit(1)` at
  `pytest_timeout.py:542` under `timeout=60`/`timeout_method="thread"`
  (orchestrator + fused-memory pyproject), `--max-worker-restart=0`
  (orchestrator). Confirmed by the `data/verify-logs/2484` log (`[gw6] node
  down` on `test_coalesce_integration_gate`) and the orchestrator pyproject's
  own `--max-worker-restart=0` comment. Merge-verify per-module commands
  (`pytest tests/ -q`) inherit the **60s** default (unlike the fallback command
  which uses `--timeout=300`).
- **Mechanism B (Qdrant):** `test_mem0_qdrant_integration.py` +
  `test_mem0_client.py::TestMem0BackendAddSystemRecordIntegration` hit the live
  production Qdrant (:6333, confirmed running) with a 10s client timeout and are
  **not** marked `@pytest.mark.integration`, so `-m 'not integration'` fails to
  exclude them; `qdrant_skipif()` (2s reachability probe) never skips a
  reachable-but-slow Qdrant → `ReadTimeout` / stale-collection `409` under load.
- **Mechanism C (timing asserts):** fixed sleeps / `wait_for(small)` / grace
  windows in specific tests (104 sleep-using test files repo-wide;
  ~18 `assert elapsed < N`).

**Existing partial coverage (verified — α must not duplicate):** the *task*
path already has a bounded whole-suite auto-retry for a **bare** xdist worker
crash (`_is_bare_xdist_worker_crash` → `VerifyInfraError` →
`_run_scoped_verification_with_infra_retry`; tasks 2365 + 2619). Two gaps remain
that α closes: (1) that reclassification is **explicitly gated `not
is_merge_verify`** (`verify.py:3822`) — the merge path is deliberately excluded
because "merge_queue.py has no VerifyInfraError handler and an uncaught raise
there would stall the merge queue" (`verify.py:3800`), so the **merge gate gets
no auto-retry at all** today; and (2) the discriminator fires **only for a bare
crash with no co-occurring `FAILED` marker**, so a starvation timeout that
surfaces as `FAILED <nodeid>` (or a Qdrant `ReadTimeout` / timing-assert flake)
is never retried — exactly what blocked task 2700.

Premises are true and backed by existing mechanisms — none assert an impossible
number. The α flake-vs-real classification reuses the premise main-sweep
already relies on (a node-id that passes in isolation was load-induced).

## 3. Approach

Two complementary code/config levers that raise the merge gate's tolerance to
CPU starvation, plus per-test hardening of the confirmed load-fragile tests. No
parallelism cap and no breadth change (those are the out-of-scope RED-TIER
calls).

- **α — Merge-gate single flake-retry (load-bearing).** Give `role='merge'`
  verify the isolated-rerun-confirm gate that `run_main_tip_sweep` already has:
  on a failing merge verify, extract the failing node-ids
  (`_extract_failing_test_ids` — already handles FAILED / ERROR / xdist
  `node down` surfaces), re-run just those **serially** (`-p no:xdist`, the
  `serial_pytest` recovery form) once with a generous per-test timeout (so the
  confirm run cannot itself starve); if they all pass, the failure was
  load-induced → **suppress and let the merge proceed**; if any still fails, the
  merge stays red. **Critically, α resolves to a pass/fail verdict inline — it
  does NOT raise `VerifyInfraError`** (that is the task-path mechanism which
  `verify.py:3800-3822` documents would stall the merge queue). This mirrors
  `run_main_tip_sweep`'s return-a-category pattern, which is why it is safe on
  the merge path where the bare-crash reclassification deliberately isn't. Single
  highest-leverage lever against the churn spiral — makes a correct task immune to
  a starvation flake without touching parallelism, and covers the co-occurring-
  `FAILED` case that 2365/2619's bare-crash discriminator does not.
- **β — Raise the merge-path per-test wall-clock timeout to 300s.** Append
  `--timeout=300` to the per-module merge `test_command`s (mirroring the
  fallback command's existing convention), so a starved-but-correct worker has
  to cross 300s — not 60s — of wall clock before `os._exit`. Removes the
  Mechanism-A trigger for all but genuinely-hung tests, orthogonally to α.
- **γ / δ — Per-test hardening.** Make the six confirmed LIVE load-fragile tests
  robust: opt-up timeouts, poll-for-condition instead of fixed sleeps,
  deterministic signal-then-cancel instead of `wait_for(0.1)`, load-scaled
  subprocess budgets, a numeric mock-config value, and integration-marking +
  idempotent-create for the Qdrant compat tests.

### Load-bearing design decision: reuse, don't duplicate (INV-5)

α **must not** re-implement node-id extraction or the isolated re-run. The
extraction helper (`_extract_failing_test_ids`, `verify.py:544`) and the
serial-recovery command builder (`serial_pytest` / `_serial_pytest_str`) already
exist and already cover the `node down` crash surface. α wires the *existing*
main-sweep confirm-gate path into the merge-role result handler — one call site,
not a copy.

### Rejected alternatives (this batch)

- **Cap `-n` / narrow breadth / add remote runners** — RED-TIER (restart /
  fleet-policy), human-only. Out of scope (§6).
- **Make Qdrant compat tests load-tolerant in-place** (uuid collection + 60s
  client timeout, keep them in merge-verify) — rejected: a merge gate must not
  depend on a live shared production service. Marking them `integration` removes
  the external dependency entirely; the compat coverage moves to an explicit
  `-m integration` run (§9 open question — flagged, not silently dropped).

## 4. Pre-conditions (G3 — verified on `main` `25e03d4db0` this session)

- `_extract_failing_test_ids` handles FAILED/ERROR/`node down` surfaces —
  `verify.py:503-560`. ✔
- `serial_pytest` / `_serial_pytest_str` build the `-p no:xdist -o addopts=''`
  recovery form — `verify_cmd.py:545`, `verify.py:1063`. ✔
- main-sweep isolated-rerun-suppress precedent (`run_main_tip_sweep`,
  retry-on-flake) — `verify.py:5201-5330`. ✔
- Existing task-path bare-crash retry (`_is_bare_xdist_worker_crash` →
  `VerifyInfraError`) is gated **out** of the merge path (`not is_merge_verify`,
  `verify.py:3822`) and covers **only** bare crashes — confirming α's gap is
  real, not a duplicate of tasks 2365/2619. ✔
- Merge verify runs per-module `pytest tests/ -q` with the **60s** pyproject
  default (no `--timeout`) — `orchestrator/orchestrator.yaml:5` et al. ✔
- `@pytest.mark.integration` marker declared — `fused-memory/pyproject.toml`
  markers. ✔ `QdrantClient.collection_exists` available. ✔
- `_load_scaled_grace` helper — `tests/scripts/test_spawn_claude.py:882`. ✔
- Shared config-mock factory — `orchestrator/tests/conftest.py:616
  mock_orch_config`. ✔
- Storm/streak escalation precedent (INV-4 house pattern) —
  `merge_liveness.py` consecutive-streak gate. ✔
- **No** `-m integration` automated lane exists anywhere (scripts/CI/config).
  ✔ (drives §9).

## 5. Resolved design decisions

1. **α suppression is single-shot, serial, same-tree.** One isolated re-run per
   failing node-id (`-p no:xdist`), against the same merge worktree/SHA — a
   corroboration re-check (INV-3), not a blind retry. No multi-attempt loop.
2. **α emits a structured `merge_flake_suppressed` fact** (INV-2) carrying the
   suppressed node-ids, the merge SHA, and `measured_at` — never log-only.
3. **α carries a storm-streak escalation** (INV-4): a per-window counter of
   suppressed merge flakes; crossing a threshold files an escalation so chronic
   suppression is loud, not silent. This is α's *own* immediate counter — it does
   **not** depend on the reify:5142 `flaky-ledger` substrate that the dormant
   `chronic_flake` detector (`enabled:false` here) needs (§7 seam).
4. **α never suppresses a genuine failure**: if the isolated re-run reproduces
   the failure (or the whole module errors at collection), the merge stays red.
   The B+H boundary test asserts both directions.
5. **β = `--timeout=300`** on every per-module merge `test_command` (uniform with
   the fallback), plus a guard test asserting the floor (mirroring
   `test_fallback_verify_config.py::test_fallback_verify_raises_per_test_timeout`).
6. **Qdrant compat tests → `@pytest.mark.integration`** (removes them from every
   `-m 'not integration'` run) + idempotent create (gate on `collection_exists`
   / catch-409). Coverage-lane gap flagged (§9), not silently accepted.
7. **γ/δ grouped by package**, not one-task-per-test — fewer merge cycles under
   the very load contention being fixed.

## 6. Out of scope (RED-TIER — human decision on `host-verify-cpu-oversubscription-df`)

- Narrowing `merge_verify_breadth` `full`→`scoped` (restart-only).
- `verify_runners` remote capacity (restart-only).
- Host-global cross-project verify admission semaphore (shared slots dir).
- Capping `-n` for the merge role, or lowering
  `merge_verify_max_concurrent_modules` (green-tier, but a live-tuning /
  throughput-tradeoff call the operator should make with load data, not bundled
  as a code change here).
- Tightening the PSI dispatch gate `cpu_some_avg10` (green-tier live-tune).

## 7. Cross-PRD / cross-mechanism seams (G4)

- **`chronic_flake` detector** (`ChronicFlakeConfig`, `enabled:false` here —
  needs reify:5142's `flaky-ledger.jsonl` + `run_all.sh` substrate). Seam owner:
  α owns an *independent* immediate storm counter (decision §5.3); if/when
  reify:5142 lands and `chronic_flake` is enabled, α's suppression facts can feed
  the ledger, but α does **not** wait on it. No reciprocal-ownership deadlock.
- **cockpit `host-verify-cpu-oversubscription-df`**: the RED-TIER levers (§6) are
  tracked there for a human. This PRD explicitly does not touch them.

## 8. Decomposition (G5: α is B+H — contract §Appendix A, boundary test §Appendix B)

- **α — Merge-gate single flake-retry** (`task_kind=normal`, architect path;
  high-stakes merge-path change). *Signal:* a merge whose per-module verify reds
  on a node-id that passes in isolation → merge lands + a `merge_flake_suppressed`
  fact is emitted; a merge whose node-id also fails in isolation → merge stays
  red; a chronic-suppression streak → escalation. B+H boundary test asserts all
  three. Files: `orchestrator/src/orchestrator/verify.py` (merge-role result
  handler), event/escalation wiring, `orchestrator/tests/`.
- **β — Raise merge-path per-test timeout to 300s** (`complexity=simple`).
  *Signal:* every per-module merge `test_command` carries `--timeout=300`; a new
  guard test asserts the ≥300 floor and fails if a segment omits it. Files:
  `*/orchestrator.yaml` per-module `test_command`s, `dark-factory-orchestrator.yaml`
  if applicable, `tests/scripts/test_*_verify_config.py`.
- **γ — orchestrator + scripts test-hardening** (`complexity=simple`). *Signal:*
  `test_coalesce_integration_gate` (4 scenario classes) carry
  `@pytest.mark.timeout(300)` and widened internal `wait_for` bounds;
  `test_window_close_129_robust_to_delayed_trap_install` uses load-scaled
  budgets (`_load_scaled_grace`); `test_verify_merge_cancel_end_to_end` timeout
  `90→120`; `mock_orch_config` (conftest) sets a numeric
  `claimant_heartbeat_interval_secs`. All named tests pass; no fixed-window
  regressions. Files: `orchestrator/tests/test_coalesce_integration_gate.py`,
  `tests/scripts/test_spawn_claude.py`, `orchestrator/tests/test_cli.py`,
  `orchestrator/tests/conftest.py`.
- **δ — fused-memory test-hardening** (`task_kind=normal`). *Signal:*
  `test_ticket_worker::test_threshold_parks_oversize_ticket_as_lookahead`
  replaces `asyncio.sleep(0.5)` with a bounded condition-poll;
  `test_harness::test_timeout_marks_run_failed` drives cancellation
  deterministically (signal-then-cancel Event) instead of `wait_for(0.1)`; the
  Qdrant version-compat tests carry the `integration` marker and an idempotent
  collection create. Named tests pass; merge-verify no longer runs the
  Qdrant-backed tests. Files: `fused-memory/tests/test_ticket_worker.py`,
  `fused-memory/tests/test_harness.py`,
  `fused-memory/tests/test_mem0_qdrant_integration.py`,
  `fused-memory/tests/test_mem0_client.py`.

No intra-batch dependencies (four independent file sets). α and β are
complementary and independent; γ and δ are test-only.

## 9. Open questions (tactical / follow-up — not blocking)

- **Qdrant compat coverage lane.** After δ marks the compat tests `integration`,
  **no automated lane runs them** (confirmed: no `-m integration` invocation
  exists). The version-compat coverage (qdrant-client / mem0 upgrades) therefore
  lapses until an explicit `-m integration` lane exists. Deliberately flagged
  loud (INV-4), not silently dropped. Follow-up candidate: a nightly / on-demand
  `-m integration` job on an unloaded host — **not** in this batch (it's infra
  scheduling). Recorded here so the gap is visible.
- α storm-streak threshold + window values: tactical, tune at implementation
  time against `merge_liveness` precedents.

## Appendix A — α contract (B+H)

Inputs: a `role='merge'` `run_scoped_verification` result whose test leg failed.
Behaviour:
1. Extract failing node-ids via the existing `_extract_failing_test_ids`
   (covers FAILED / ERROR / `node down`). If none extractable (opaque / whole
   collection error) → **do not suppress** (fail closed to red).
2. Re-run exactly those node-ids once, serially (`serial_pytest` form,
   `-p no:xdist -o addopts=''`), with a generous per-test timeout (≥300s so the
   confirm run cannot itself starve → no false non-suppression), in the same
   merge worktree/SHA.
3. All pass → suppress: **return a passed verdict** (do NOT raise
   `VerifyInfraError` — the merge path has no handler, `verify.py:3800`); merge
   proceeds; emit `merge_flake_suppressed` (node-ids, sha, measured_at); bump the
   suppression-streak counter.
4. Any fail / re-run errors / confirm-run times out → merge stays red (existing
   behaviour); streak counter untouched by a non-suppressing outcome. A genuine
   hang re-hangs in isolation → not suppressed (correct).
5. Streak ≥ threshold within window → escalation (loud), streak cleared.

Distinction from tasks 2365/2619: those raise `VerifyInfraError` for a **bare**
crash on the **task** path only (`not is_merge_verify`, `verify.py:3822`). α is
the **merge**-path analog and covers the **co-occurring-`FAILED`** case, via a
return-a-verdict (not raise) resolution — the only shape safe on the merge path.

Invariants: INV-2 (structured fact at suppression), INV-3 (same-tree
corroboration), INV-4 (streak escalation on the fail-soft path), INV-5 (reuse
`_extract_failing_test_ids` + `serial_pytest` + main-sweep confirm pattern, no
duplication).

## Appendix B — α boundary-test sketch (B+H observable signal)

- **B1 suppress-real-flake:** stub a merge verify whose first test-leg run
  reports FAILED for node-id X (and a `node down` variant), whose isolated
  re-run of X passes → assert merge proceeds + `merge_flake_suppressed{X}`
  emitted.
- **B2 no-false-suppress:** isolated re-run of X still FAILED → assert merge
  stays red, no suppression fact.
- **B3 collection-error not suppressed:** failing result is a bare
  `ERROR <file.py>` (whole-module collection error) → assert not suppressed.
- **B4 storm:** N suppressions within window → assert escalation filed once,
  streak cleared.
