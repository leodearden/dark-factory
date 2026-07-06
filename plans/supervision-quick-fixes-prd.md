# PRD: Supervision quick fixes — inspector hoist, escalation-query scoping, substrate-probe fail-closed, module-lock charter helper, streak registry

**Status:** active — authored 2026-07-06 (stream M2 of the bug-hotspot remediation
program, `plans/bug-hotspot-remediation-program-2026-07-06.md`).
**Mode:** bare B (G5: five mechanical hardening fixes, no new load-bearing seam design).
**Survey basis:** `plans/bug-hotspot-survey-2026-07-06-full-findings.json` — one harness-cluster
finding + four scheduler-cluster findings, all `verdict: confirmed`, all re-verified against
main on 2026-07-06 in this session (line references below are current-main).

## Goal

Close five confirmed latent-bug / drift hotspots in the orchestrator's supervision and
dispatch layer, each with an operator-observable behaviour change:

1. A wedged `systemctl --user show` can no longer silently hang the deterministic-strand
   recovery sweep forever (the harness's duplicated inspector gains the task-2091
   timeout+kill+sentinel hardening by delegating to ONE shared implementation).
2. An unrelated escalation (e.g. starvation-watchdog) on a deterministic task can no
   longer alias as "human resolved the deploy" and drive the task to a phantom `done` —
   nor suppress the runner's own gate/infra escalation filing.
3. A task with a **malformed** `substrate_probe` descriptor is BLOCKED at dispatch
   (fail-closed FLIP) instead of silently skipping the substrate gate (fail-open).
4. `metadata.files` lock-charter sanitization (Contract 1: file-level always, coarsen at
   READ only) has exactly one orchestrator-side implementation; the module cache has a
   single writer, so cached and derived module sets cannot diverge.
5. The five hand-rolled consecutive-tick streak counters get uniform reset/GC semantics
   through one registry; the per-tick stale-id sweep iterates the registry instead of a
   manually-enumerated dict list (structurally prevents the leak-and-drift class).

## Background

- **Item 1**: `_recon_inspect_unit` (harness.py:367-393) is an admitted "standalone
  duplicate of `DeterministicRunner._default_inspect_unit`" (its own docstring). The twins
  diverged after task 2091: the runner wraps `proc.communicate()` in
  `asyncio.wait_for(timeout=self._inspect_timeout_secs)` with kill + bounded reap +
  sentinel return (deterministic_runner.py:398-434); the harness copy still does a bare
  `stdout, _ = await proc.communicate()` (harness.py:382). A wedged `systemctl show`
  (2087/2091 signature) hangs `_run_deterministic_recon_sweep` (:7922) — the sweep loop
  awaits the pass, so deterministic-strand recovery silently stops forever. Task 2091 is
  `done` but fixed only the runner copy (premise re-verified 2026-07-06).
- **Item 2**: `EscalationQueue.get_by_task(task_id, status, level)` (queue.py:309) has no
  `agent_role` filter, and `status=None` scans the archive. DeterministicRunner queries it
  unscoped at five sites: :575 (infra-escalation dedup guard), :793 (gate-filing dedup
  guard), :861 (section-1 quiescence), :922 (before_done quiescence), :952
  (`ever_escalated = bool(get_by_task(task_id))` → branch (b) drives `done` with note
  "resumed after human resolution"). The starvation watchdog files escalations with the
  same task_id (`agent_role='orchestrator-starvation-watchdog'`), so an unrelated
  escalation aliases as deploy-resolution proof; symmetrically, a pending unrelated
  escalation suppresses the runner's own gate filing at :793 (the gate is then never
  filed, and the unrelated escalation's later resolution phantom-completes the task).
  The runner files all its escalations with the sentinel
  `agent_role='orchestrator-deterministic'`.
- **Item 3**: two same-named predicates disagree. `substrate_gate.carries_substrate_probe`
  (substrate_gate.py:81-112) is key-presence — written (task 1809) so
  `run_substrate_recheck` fails CLOSED (FLIP) on a declared-but-malformed descriptor
  (used at substrate_gate.py:266). `Scheduler.carries_substrate_probe`
  (scheduler.py:1412-1428) instead returns `extract_probe_set(task) is not None` — False
  for a malformed/empty descriptor. The production gate at harness.py:5124 uses the
  Scheduler version to decide whether to run the recheck at all, so a malformed descriptor
  silently SKIPS the gate; the fail-closed branch is unreachable from production.
- **Item 4**: the `strip_directory_locks → files_to_modules(depth)` pipeline is
  independently coded in `Scheduler._get_modules` (scheduler.py:4700-4753),
  `Harness._tag_task_modules` (harness.py:1791-1945, which also writes
  `scheduler._module_cache` directly at :1939), `Scheduler._persist_files_metadata`
  (scheduler.py:4316-4339), and `handle_blast_radius_expansion` (writes `_module_cache`
  at :4456). 54ec90fefc documents the failure shape: the strip was added to the read path
  but a write site missed it and the fused-memory lock-charter guard rejected the entire
  payload. `_module_cache` has two writers (harness startup + blast-radius) while
  `_get_modules` itself never populates it after deriving.
- **Item 5**: "N consecutive ticks then fire, reset on clean tick, GC on terminal" is
  implemented five times with bespoke semantics: `_external_unresolved_counts`
  (scheduler.py:2104-2160, `(task_id, dep)` keys), `_external_resolver_degraded_counts`
  (:2001-2060), `_external_hold_streak`/`_external_hold_cause` (:1913-1922, cause-change
  reset), `_local_backfill_unresolved_counts` (three near-identical inline loops in
  `acquire_next`, :3396-3488), and `_starvation_first_seen`/`_starvation_escalated`
  (:2179-2287, first-seen age + resolve callback). The stale-id GC sweep (:3533-3633)
  manually enumerates every dict with comment-documented carve-outs; every new watchdog
  re-litigates reset-on-recovery and leak-on-terminal by hand (bug arc: 1807, 1855, 1880).

## Sketch of approach

### α — hoist `inspect_systemd_unit` (item 1)

New small module `orchestrator/src/orchestrator/systemd_inspect.py`:

- `async def inspect_systemd_unit(unit: str, *, timeout_secs: float, reap_grace_secs: float = 5.0) -> dict`
  — module-level, the single implementation of the `systemctl --user show <unit>
  -p MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic` pattern with
  the task-2091 hardening exactly once: `asyncio.wait_for` around `communicate()`, direct
  `proc.kill()` on timeout (NOT killpg — the process shares the orchestrator's group; keep
  the deterministic_runner.py:403-411 comment rationale), bounded reap
  (`wait_for(proc.wait(), reap_grace_secs)`), WARNING log, and the sentinel return
  `{'MainPID': 0, 'ActiveState': '', 'ActiveEnterTimestamp': '',
  'ActiveEnterTimestampMonotonic': 0}`. Integer fields coerced with 0-sentinel on parse
  failure. The 10s default timeout constant moves here (from deterministic_runner.py:158).
- Relocate harness's `_deterministic_deploy_health_verdict` (pure classifier of the
  inspector's output, harness.py:~300-364) into this module so verdict semantics live
  beside the data they classify; harness imports it.
- `DeterministicRunner._default_inspect_unit` (deterministic_runner.py:385) becomes a thin
  delegate passing `self._inspect_timeout_secs` / `self._reap_grace_secs`; the injectable
  seam `self._unit_inspector` (:1195) is unchanged.
- `harness._recon_inspect_unit` (:367) becomes a thin delegate (module timeout default);
  the injection seam `self._recon_unit_inspector or _recon_inspect_unit` (:7699, :985) is
  unchanged.

W10's `proc_supervision.py` will import/relocate this helper — it must remain a
standalone module-level function with no Harness/Runner instance dependencies.

### β — scope DeterministicRunner escalation queries by `agent_role` (item 2)

- Add `agent_role: str | None = None` filter parameter to `EscalationQueue.get_by_task`
  (escalation/src/escalation/queue.py:309), applied like the existing `status`/`level`
  filters (None = no filter; full backward compatibility for existing callers, e.g.
  `has_pending_l1` at :398).
- Introduce/reuse one module constant for the sentinel (e.g.
  `DETERMINISTIC_AGENT_ROLE = 'orchestrator-deterministic'`) in deterministic_runner.py
  and pass it at **all five** query sites: :575, :793, :861, :922, :952. The brief names
  :861/:922/:952; the dedup guards at :575/:793 exhibit the same aliasing class in the
  suppress direction (unrelated pending escalation → runner's own gate never filed →
  later phantom-done when the unrelated escalation resolves), and the brief's charter is
  "scope ALL DeterministicRunner escalation-queue queries".
- Resulting semantics: quiescence blocks only on the runner's own pending escalations;
  `ever_escalated` proof requires a runner-filed (resolved) escalation; the runner files
  its gate even when an unrelated escalation is pending.

This is **query-scoping only**. The stamp-combinatorics state reconstruction itself
(the survey proposal's `metadata.deterministic_state` explicit state field) is owned by
W10's `DeployState` — see the seam table. β must not introduce any state enum.

### γ — substrate-probe fail-closed at the real gate (item 3)

- Delete `Scheduler.carries_substrate_probe` (scheduler.py:1412-1428) and fix the
  docstring cross-reference at :1436 (`is_deterministic`).
- harness.py:5124 calls `substrate_gate.carries_substrate_probe` (key-presence) instead,
  letting `run_substrate_recheck`'s own SKIP/FLIP logic decide policy — the module already
  encodes the fail-closed invariant (task 1809); the wrapper defeated it.
- Update any tests that stub `scheduler.carries_substrate_probe` (e.g. the
  test_release_workflow / test_crash_recovery MagicMock pattern from task 1838).
- New test: a task whose `metadata.substrate_probe` is malformed (`'garbage'` string, or
  `{}`/missing `probe_set`) produces a FLIP verdict → dispatch blocked + escalation, not
  a skipped gate. Companion test: metadata without the key at all → gate skipped (SKIP),
  dispatch proceeds.

### δ — one module-lock derivation helper + single-writer cache (item 4)

New module `orchestrator/src/orchestrator/module_charter.py` owning the orchestrator-side
implementation of Lock-charter Contract 1:

- `derive_modules(files: list, depth: int, *, task_id: str = '') -> list[str]` — the
  `directory_locks` diagnostic (log rejected dir entries) + `strip_directory_locks` +
  `files_to_modules(depth)` pipeline, exactly once (delegating to the `shared.locking`
  primitives, which stay where they are).
- `sanitize_files_for_persist(files: list) -> list[str]` — `strip_directory_locks`
  wrapper used by **every** `metadata.files` write path, so a dir-bearing payload can
  never reach the fused-memory lock-charter guard again (the 54ec90fefc class).
- Migrate the four sites: `Scheduler._get_modules` (:4700) uses `derive_modules` and
  becomes **write-through** (populates `_module_cache[task_id]` on derive);
  `Scheduler._persist_files_metadata` (:4316) uses `sanitize_files_for_persist`;
  `handle_blast_radius_expansion` (:4456) routes its cache update through the single
  cache-writing seam; `Harness._tag_task_modules` (:1791-1945) stops poking
  `scheduler._module_cache` directly (:1939) and instead calls a new public
  `Scheduler.seed_modules(task_id, files)` that routes through `derive_modules`, and uses
  `sanitize_files_for_persist` for its writeback payload (:1921).
- Single-writer invariant: exactly one function assigns into `_module_cache`; the
  deterministic short-circuit (`is_deterministic → []`, :4713-4715) and the
  `task-<id>` fallback (:4745-4753) semantics are preserved unchanged.

### ε — StreakCounter / StreakRegistry (item 5)

New module `orchestrator/src/orchestrator/streaks.py`:

- `StreakCounter` — keyed consecutive-tick counter with constructor options for the
  variant semantics that today are re-derived by hand: `threshold`,
  reset-on-clear (`clear(key)`), cause-change reset (the `_external_hold_streak`/`_cause`
  pair collapses into one counter with `touch(key, cause=...)`), and first-seen-age style
  (`_starvation_first_seen`: `touch` records first timestamp; predicate is age, not
  count). Keys are opaque (str or tuple — `(task_id, dep)` supported); each counter
  declares how to extract the task-id component for GC (`key_fn`).
- `StreakRegistry` — `register(name, counter)`; `registry.gc(stale_ids)` sweeps every
  registered counter (replacing the manual enumeration at scheduler.py:3571-3600);
  per-counter optional `on_gc(key)` async callback preserves the starvation-watchdog
  GC-resolve behaviour (:3615-3633), including its non-eligible-status extension —
  that block keeps its extra `_STARVATION_NON_ELIGIBLE` id-collection logic and routes
  only the clearing through the registry.
- Migrate the five counters; collapse the three near-identical backfill-degradation
  loops in `acquire_next` (:3396-3488) into one helper taking the missing-dep set.
- **Behaviour parity is the hard requirement**: thresholds, reset semantics, escalation
  timing, and log/event output are unchanged; the existing scheduler test suite for
  these counters (1807/1855/1880 coverage) must pass without weakening.

## Resolved design decisions

1. **Helper module placement (α)**: `orchestrator/src/orchestrator/systemd_inspect.py`,
   NOT `proc_supervision.py` — W10 owns creating proc_supervision.py and will
   import/relocate M2's function (program seam table). Module-level function, no class.
2. **`_deterministic_deploy_health_verdict` moves with the inspector (α)** — it is a pure
   classifier of the inspector's output dict; colocating follows the survey proposal and
   gives W10 one place to relocate from.
3. **β scopes all five query sites, not just the three the brief enumerates** — the dedup
   guards (:575/:793) are the same aliasing class in the suppress direction; the brief's
   subject line is "scope ALL DeterministicRunner escalation-queue queries". Recorded
   here because it widens the brief's line list.
4. **β adds a filter parameter; no new state field** — the deeper stamp-combinatorics
   replacement is W10's DeployState (seam table: "M2 = query-scoping only").
5. **γ deletes the Scheduler wrapper rather than fixing it** — two same-named predicates
   with divergent semantics is the defect; substrate_gate's version becomes the actual
   single source of truth (matching the Scheduler docstring's own stated intent).
6. **δ lives in a new orchestrator module (`module_charter.py`), not in `shared.locking`**
   — the primitives (`strip_directory_locks`, `files_to_modules`, `directory_locks`) stay
   in `shared.locking` (both sides of the wire use them); `module_charter.py` owns the
   orchestrator-side *composition* (derive/sanitize pipeline + cache discipline). The
   fused-memory-side guard (task 1833) is untouched — "one implementation on each side of
   the wire".
7. **ε is parity-preserving refactor only** — no threshold retuning, no new watchdog
   semantics; any semantics change found necessary must escalate rather than ship silently.

## Pre-conditions

None. All substrate exists on main (G3 verified 2026-07-06):

- Runner's hardened inspector pattern: deterministic_runner.py:392-445 (wait_for at :399,
  kill at :413, sentinel at :429-434). Harness bare communicate: harness.py:382.
  Wiring: `_run_deterministic_recon_sweep` (:7922) → inspect seam (:7699); runner seam
  `self._unit_inspector or self._default_inspect_unit` (:1195).
- `Escalation` model carries `agent_role`; runner files with
  `agent_role='orchestrator-deterministic'` (deterministic_runner.py:586 et al.);
  `get_by_task` signature at queue.py:309-311.
- `substrate_gate.carries_substrate_probe` (:81-112) and `run_substrate_recheck`'s
  fail-closed use of it (:266); production gate call at harness.py:5124.
- `shared.locking.directory_locks`/`strip_directory_locks`/`files_to_modules`
  (shared/src/shared/locking.py:101/:120/:151); the four derivation sites at the lines
  listed in Background.
- The five streak counters at the lines listed in Background; GC sweep at
  scheduler.py:3533-3633.

## Cross-PRD relationship (G4)

Program doc `plans/bug-hotspot-remediation-program-2026-07-06.md` is authoritative.

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W10 harness-supervision | produces → W10 consumes | `inspect_systemd_unit(unit, *, timeout_secs)` single helper (W10's proc_supervision imports/relocates it — never a second copy) | **M2 (this PRD)** | queued here (task α) |
| W10 harness-supervision | boundary (do not cross) | `DeployState` deterministic-deploy phase enum + persisted verify baseline; replaces stamp-combinatorics | **W10** | M2 must NOT introduce a deploy-state enum; β is query-scoping only |
| W2 task-status-authority | adjacent (no seam) | escalation action legality table + role-derived level ceilings | W2 | β adds a read-side query filter to `get_by_task` — no legality/level semantics touched |
| W3 task-metadata-schema | adjacent (no seam) | `shared/task_metadata.py` typed schema | W3 | δ sanitizes existing `metadata.files` writes; defines no new metadata fields |

No other cross-PRD seams; α–ε touch no fused-memory code.

## Decomposition plan

All five tasks are mutually independent (no intra-batch deps). They overlap on
scheduler.py / harness.py / deterministic_runner.py files, so the orchestrator's module
locks serialize them naturally — no artificial ordering deps are declared.

- **α — Hoist `inspect_systemd_unit` into `systemd_inspect.py`; both inspectors delegate**
  (priority high, complexity simple).
  Modules: orchestrator.
  Files: `orchestrator/src/orchestrator/systemd_inspect.py` (new),
  `orchestrator/src/orchestrator/deterministic_runner.py`,
  `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_harness_deterministic_recon_sweep.py`,
  `orchestrator/tests/test_deterministic_runner.py`.
  Observable signal: with a hung fake `systemctl show` injected under the recon sweep's
  production path, the sweep completes the pass within the timeout and the operator sees
  the WARNING "systemctl show <unit> timed out … returning MainPID=0 sentinel" from the
  harness sweep (previously: silent permanent hang). Grep evidence of de-duplication: exactly one
  `create_subprocess_exec('systemctl', '--user', 'show', …)` site in the orchestrator
  package.
  Consumer: harness `_run_deterministic_recon_sweep` + DeterministicRunner
  baseline/verify legs; W10 proc_supervision (cross-PRD).
- **β — Add `agent_role` filter to `EscalationQueue.get_by_task`; scope all five runner
  query sites** (priority high, complexity simple).
  Modules: escalation, orchestrator.
  Files: `escalation/src/escalation/queue.py`,
  `orchestrator/src/orchestrator/deterministic_runner.py`,
  `escalation/tests/test_queue.py`,
  `orchestrator/tests/test_deterministic_runner.py`.
  Observable signal: a deterministic task with `before_done_ran_at` stamped whose only
  queue history is a resolved `agent_role='orchestrator-starvation-watchdog'` escalation
  is RE-ESCALATED (branch (c), WARNING log + new born-at-L2) instead of driven to done
  "resumed after human resolution"; and with an unrelated PENDING escalation, the runner
  still files its own gate (two pending escalations visible via get_pending_escalations).
  A runner-filed resolved escalation still proves resolution (parity).
  Consumer: DeterministicRunner resume/quiescence paths (production dispatch surface).
- **γ — Substrate gate fail-closed: route harness dispatch check through
  `substrate_gate.carries_substrate_probe`** (priority medium, complexity simple).
  Modules: orchestrator.
  Files: `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_substrate_gate.py`,
  `orchestrator/tests/test_release_workflow_substrate.py` (or the existing harness gate
  test file the implementer locates; task-1838 pattern sites).
  Observable signal: dispatching a task with `metadata.substrate_probe='garbage'`
  produces a FLIP verdict → dispatch blocked + escalation record (operator-visible via
  get_pending_escalations / task blocked status), where before the gate silently skipped
  and the agent spun up. A task with no `substrate_probe` key still dispatches (SKIP).
  `Scheduler.carries_substrate_probe` no longer exists (grep).
  Consumer: harness dispatch gate at harness.py:5124 (production surface).
- **δ — `module_charter.py`: one derive/sanitize implementation + single-writer module
  cache** (priority medium, full path — refactor).
  Modules: orchestrator.
  Files: `orchestrator/src/orchestrator/module_charter.py` (new),
  `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/tests/test_scheduler.py`,
  `orchestrator/tests/test_harness_module_tagging.py`.
  Observable signal: a dir-bearing `metadata.files` list passed through EVERY write path
  (module-tagger writeback, `_persist_files_metadata`, blast-radius) persists the
  stripped file-level list — no fused-memory lock-charter rejection warning in the log
  (54ec90fefc class impossible by construction); `_get_modules` derive and `_module_cache`
  agree after a metadata update (parity test); grep shows exactly one
  `strip_directory_locks(`-composition site in the orchestrator package (the charter
  module), with call sites delegating.
  Consumer: Scheduler lock acquisition (`_get_modules` → `acquire_next`), harness startup
  module tagging, blast-radius expansion — all production dispatch surfaces.
- **ε — `streaks.py`: StreakCounter/StreakRegistry; migrate the five counters; registry
  GC sweep** (priority medium, full path — refactor).
  Modules: orchestrator.
  Files: `orchestrator/src/orchestrator/streaks.py` (new),
  `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/tests/test_scheduler.py`,
  `orchestrator/tests/test_scheduler_state.py`.
  Observable signal: behaviour parity — the existing 1807/1855/1880 counter tests pass
  unchanged (thresholds, reset-on-clean-tick, cause-change reset, starvation GC-resolve
  callback timing); a new leak test proves a terminal-before-threshold task leaves ZERO
  entries in ANY registered counter after one sweep; the stale-id sweep no longer
  manually enumerates counter dicts (grep: the :3571-3600 manual blocks are gone,
  replaced by one `registry.gc(stale_ids)` call).
  Consumer: scheduler tick loop + `acquire_next` (production dispatch surface); every
  future watchdog (registration point instead of hand-rolled dict).

## Out of scope

- **DeployState phase enum / any deterministic-runner state-machine change** — W10 owns it
  (seam table). β leaves the stamp-presence reconstruction in place, only scoped. The
  residual "the task's own earlier-life gate escalation still counts as ever_escalated"
  is accepted here and dies with W10's DeployState.
- **proc_supervision.py itself** (W10) — α's helper is standalone; W10 imports/relocates.
- **Retuning any streak threshold or watchdog semantics** (ε is parity-preserving).
- **The fused-memory-side lock-charter guard** (task 1833, done) — δ is orchestrator-side
  only.
- **`_deterministic_deploy_health_verdict` semantics changes** — α relocates it verbatim.

## Open questions (surfaced but not decided in this session)

1. **Exact harness-side timeout value for the recon sweep's inspector calls.** The runner
   uses a configurable `self._inspect_timeout_secs` (default 10s). **Suggested
   resolution:** the sweep uses the module default (10s) without new config plumbing;
   add a config knob only if a real need appears. Decide during α.
2. **Whether `StreakCounter` should own the escalation/fire callback or only the
   counting.** The five sites fire different actions (escalate, block, resolve).
   **Suggested resolution:** registry owns counting + GC only; fire decisions stay at the
   call sites reading `counter.value(key) >= threshold` — smaller blast radius, parity
   easier to prove. Decide during ε.
3. **Where the malformed-probe FLIP escalation's category lands (γ)** — reuse the existing
   substrate-gate block path's category unchanged. Decide during γ (read
   `_run_substrate_gate`'s existing escalation shape and keep it).
