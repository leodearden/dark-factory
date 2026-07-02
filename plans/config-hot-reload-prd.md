# PRD: Orchestrator config hot-reload (narrow / allowlist)

**Status:** active — authored 2026-07-02 (design session; user AFK, recommended
defaults adopted for the four surfaced choices, all noted in §Resolved).
**Project:** dark_factory. **Approach:** light B+H (contract + boundary-test
sketch; G5 heuristic hit: 3 modules, load-bearing shared config object).

## Goal

An operator (or an L2 escalation-watcher session) can apply a **safe,
explicitly-allowlisted subset** of orchestrator config changes to a *running*
orchestrator by calling a new escalation MCP tool `reload_config`, receiving a
synchronous per-field disposition report — instead of paying a full restart
(90 s graceful-stop that SIGKILLs in-flight agents and cargo verify suites,
then a cold start: warm-lane reseed, module-tagger pass, up-to-280 s
fused-memory wait, dirty-tree start-guard hazard).

User-observable surface: `mcp__escalation__reload_config` returns
`{applied, restart_required, error, …}`; a WARNING journal line fires when
changed fields could NOT be hot-applied; a `config_reload` event row lands in
`runs.db` for the audit trail.

## Background — why an allowlist, not a general reload

Codebase survey (2026-07-02, this session) of how `OrchestratorConfig`
(config.py, single mutable pydantic-settings object, `validate_assignment=True`,
NOT frozen) flows through the running process:

- **One instance per process** (`load_config` → `cli.py:193` → `Harness`),
  threaded **by reference** into Scheduler, ModuleLockTable, TaskWorkflow,
  TaskSteward, ReviewCheckpoint, OfflineLaneWorker, VerifyRunner, and the
  in-process escalation server. GitOps holds the `config.git` **submodel**;
  UsageGate holds `config.usage_cap`. Merge requests carry a reference.
- **Green tier** (this PRD's target): fields read fresh at each use with no
  cross-read coupling — per-role agent params read at every spawn
  (workflow.py `_invoke`), `fairness.skip_threshold` (scheduler.py
  `skip_threshold_for` per tick), starvation-watchdog thresholds,
  `idle_poll_secs`, `orphan_l0_timeout_secs`, watcher-rotation params,
  `review.*`, `unblock_auto.*`, `verify_env`, git offline-lane tunables.
- **Yellow tier** (excluded): values captured into loop locals at loop entry
  (every `*_interval_secs`, `*_enabled` loop knob), breaker window/floor
  captured in its constructor, `max_per_module` memoized per module in the
  lock table.
- **Red tier** (excluded, restart-only forever): startup-sized structures —
  `max_concurrent_tasks` (dispatch semaphore + warm-lane pool size),
  `spare_warm_lanes`, `verify_runners` → `_speculation_k` → spec pool +
  merge-worker speculation depth, `escalation.host/port` (uvicorn already
  bound), `sandbox.backend` (module-global), `project_root`, git fields the
  merge queue reads at multiple stages of one merge
  (`branch_prefix`/`main_branch`/`persistent_merge_worktree` — also guarded
  by fail-closed start-time validators `enforce_merge_liveness_margin` /
  `enforce_persistent_worktree_serial_lane` that a live toggle would bypass).
- **Out of reach**: separate processes (fused-memory, dashboard, remote
  verify runners re-read their own config; running agent subprocesses hold a
  frozen env/arg snapshot from spawn).

Known hazards a naive reload would hit (designed out below):
1. Nested submodel assignment **skips validation** (only the top-level model
   has `validate_assignment=True`) → never patch from raw YAML; always
   construct a fresh fully-validated config and copy leaves.
2. Replacing a submodel object strands GitOps/UsageGate on the old instance →
   mutate **leaf fields only**, never swap submodels.
3. A partial (allowlisted-subset) apply can create a **hybrid** live object
   that violates a cross-field model validator (e.g. `timeouts.steward >=
   steward_completion_timeout`) even though the fresh file as a whole is
   valid → post-apply hybrid re-validation with same-turn rollback (I5).
4. SIGHUP is already owned by usage_gate's OAuth reload (usage_gate.py
   `register_signal_handlers`) → trigger is an MCP tool, not a signal.
5. Torn multi-field reads → apply in one event-loop turn, no awaits between
   writes (I4).

This PRD deliberately does **not** make yellow/red fields reloadable and does
**not** hot-deploy code (orchestrators import code at process start —
deterministic deploy tasks + restart remain the deploy mechanism).

## Resolved design decisions

1. **Trigger = escalation MCP tool only.** Async `reload_config` tool on the
   in-process escalation server (per-orchestrator: reify 8100, df 8102),
   mirroring `halt_scheduler`: `await harness.reload_config()`, standalone
   guard (`harness is None` → error dict), **operator-only** (in no agent
   role's allow-list — agents must never reload config themselves). No file
   watcher (fires on partial saves / uncommitted edits; report has no
   reader), no signal (SIGHUP taken; no structured response channel).
2. **Straddle accepted.** In-place mutation means later stages of an
   in-flight task use new per-role params (architect on old model,
   implementer on new). Benign for tuning knobs — same outcome as today's
   restart + `--resume`. No per-task config pinning.
3. **Broad green-tier allowlist** (v1, §Allowlist below). Every entry is
   read-at-use with no startup-baked copy. Shrink/extend is a one-line
   allowlist edit + test.
4. **Validate-then-copy.** Applied values come only from a fully-constructed
   fresh `OrchestratorConfig` via `load_config()` (all field + model
   validators, sccache→verify_env fold, env expansion) run in a worker
   thread; raw-YAML values never touch the live object.
5. **Hybrid re-validation + rollback.** After the leaf-copy loop, re-validate
   the live object's full dump; on failure restore the just-applied old
   values (same loop turn) and return an error. No reader ever observes the
   invalid hybrid.
6. **Reporting = synchronous response + WARNING log + runs.db event.** No
   auto-`escalate_info`: the caller is an operator receiving the report
   synchronously; the WARNING line + event row cover the AFK audit trail.
   (Loud-over-silent is satisfied by the response itself listing
   `restart_required` — the tool never silently half-applies.)
7. **Module configs out of scope.** `_module_configs` (per-module
   orchestrator.yaml discovery) is not diffed and not swapped; per-module
   config changes remain restart-only in v1 (the lock table memoizes
   `max_per_module` per module anyway, which would make a swap half-take).

## Allowlist (v1)

All dotted paths below; evidence = read-fresh-at-use site.

| Group | Paths | Read site |
|---|---|---|
| Agent role params | `models.*`, `budgets.*`, `turns.*` (max_turns), `effort.*`, `timeouts.*` (incl. `startup_grace_secs`), `backends.*` | workflow.py `_invoke` per spawn; steward/harness per spawn |
| Steward grace | `steward_completion_timeout`, `steward_lifetime_budget` | workflow wait / steward per use |
| Scheduler tuning | `fairness.skip_threshold`, `starvation_watchdog.enabled`, `starvation_watchdog.skip_threshold`, `starvation_watchdog.idle_secs` | scheduler.py per tick |
| Loop-pass thresholds | `idle_poll_secs`, `orphan_l0_timeout_secs`, `watcher_rotation_escalations`, `watcher_rotation_hours` (+ crashloop-window params read live per rotation) | harness.py per iteration/pass |
| Review knobs | `review.enabled`, `review.interval`, `review.full_review_on_complete`, `review.full_review_min_interval_secs`, `review.full_review_min_tasks` | review_checkpoint per merge/idle check |
| Unblock-auto | `unblock_auto.*` | dry-run hook per block event (b3 cap re-reads config from disk already) |
| Verify env | `verify_env` | verify.py at execute time (fresh config's value already carries the sccache fold) |
| Offline-lane tunables (leaf fields on the **existing** `git` submodel — leaf-mutation only per I3) | `git.offline_lane_test_threads`, `git.offline_lane_poll_interval_secs`, `git.offline_lane_red_advances_before_blocker` | offline_lane.py per iteration/run (their own field docs say "retunable via orchestrator.yaml without a code change") |

Everything else that differs between live and fresh → `restart_required`
(reported, never mutated). The allowlist is a code-owned constant in
config.py (reload-safety is a code property, not operator-tunable).

## Contract

### Tool schema

```
reload_config() -> {
  reloaded: bool,              # true iff validation passed and apply committed
  config_path: str,            # the ORCH_CONFIG_PATH that was re-read
  applied: {path: {old, new}},          # allowlisted, changed, now live
  restart_required: {path: {old, new}}, # changed but not allowlisted
  unchanged: int,              # count of equal fields (audit convenience)
  error: str | null,           # set iff reloaded=false; nothing was mutated
}
```

`reload_config(config_path=...)` takes **no** path override — it always
re-reads the path the process was started with (`ORCH_CONFIG_PATH`), so a
reload can never retarget an orchestrator at a different project (the
2026-04-06 cross-project-execution safeguard extends to reload).

### Invariants

- **I1 fail-closed validation.** If fresh `load_config()` raises (bad YAML,
  validator failure, missing file), the live config is untouched and the
  tool returns `reloaded=false, error=…`.
- **I2 allowlist exclusivity.** Only allowlisted dotted paths are ever
  written to the live object. Non-allowlisted diffs are reported in
  `restart_required` and never mutated. (Negative assertion; mechanism =
  the allowlist filter built in task α.)
- **I3 leaf mutation only.** Submodel objects (`config.git`,
  `config.usage_cap`, `config.models`, …) are never replaced; only their
  scalar/dict leaf fields are assigned — preserving the submodel identity
  GitOps and UsageGate captured at startup.
- **I4 atomic apply.** All writes (and any rollback) happen in one event-loop
  turn with no awaits interleaved: `load_config` runs in `asyncio.to_thread`,
  then diff+apply+re-validate runs synchronously on the loop. Coroutine
  readers never observe a torn multi-field state. (Thread-pool readers of
  single fields see per-assignment atomicity via the GIL; no allowlisted
  field pair is jointly read by a threaded reader.)
- **I5 hybrid validity.** After apply, the live object's full dump is
  re-validated against `OrchestratorConfig`; on failure every applied field
  is rolled back to its old value in the same loop turn and the tool
  returns `reloaded=false, error=hybrid-invariant: …`.
- **I6 report completeness.** Every dotted path whose live vs fresh values
  differ appears in exactly one of `applied` / `restart_required`. No
  silent categories: "reload succeeded" ≠ "everything took effect" is made
  impossible to misread.
- **I7 audit.** Every call (success or failure) appends a `config_reload`
  event row to runs.db with the full report; a WARNING log line fires
  whenever `restart_required` is non-empty or `reloaded=false`.
- **I8 operator-only.** The tool appears in no agent role's MCP allow-list
  (same convention as `halt_scheduler`).

## Pre-conditions for activating

- None external. All substrate exists on main today (G3 verified this
  session): escalation-server tool registration with harness wiring
  (escalation/server.py `create_server(harness=…)`, `halt_scheduler`
  precedent), in-process `load_config()` re-read precedent (b3_gate.py
  `_resolve_cap`), `validate_assignment=True` on the top model
  (config.py `model_config`), workflow/scheduler read-at-use sites cited in
  §Allowlist. No novel syntax, schema, endpoint, or flag.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-queue-modularization-invariants-prd.md` (df 1985-2002, in flight) | none (adjacency only) | — no shared mechanism; this PRD does not touch merge_queue internals; file-level overlap limited to small harness.py additions, serialized by module locks | n/a | n/a |

No cross-PRD seams. The deploy capstone (ε) reuses
`scripts/restart-all-orchestrators.sh` committed by the modularization batch —
a committed script, not a live seam; decompose-time check confirms it exists
on main (the deterministic-task guard re-validates at `submit_task`).

## Decomposition plan

Linear chain α→β→γ→δ→ε.

- **α — config.py reload machinery** (intermediate → unlocks β).
  `RELOADABLE_FIELDS` (code-owned dotted-path constant), `diff_config(live,
  fresh) -> (applied_candidates, restart_required, unchanged)`,
  `apply_reload(live, fresh) -> report` implementing I2/I3/I5 (leaf-copy +
  hybrid re-validate + rollback) as pure synchronous functions over two
  `OrchestratorConfig` instances. Modules: orchestrator/config.py (+ unit
  tests incl. the hybrid-rollback case).
- **β — Harness.reload_config()** (intermediate → unlocks γ).
  `asyncio.to_thread(load_config)` → same-turn diff/apply on the loop (I1,
  I4) → runs.db `config_reload` event + WARNING when restart_required ≠ ∅ or
  error (I7). Modules: orchestrator/harness.py, event_store usage.
- **γ — escalation `reload_config` tool** (leaf).
  Async tool mirroring `halt_scheduler`: standalone guard, awaits
  `harness.reload_config()`, returns the report verbatim; excluded from all
  agent allow-lists (I8). **Signal:** calling
  `mcp__escalation__reload_config` on a running orchestrator after editing
  its orchestrator.yaml returns the disposition report (API-response
  difference), and journalctl shows the WARNING line for a restart-required
  field. Modules: escalation/src/escalation/server.py, harness wiring
  (+ tests).
- **δ — integration gate + operator docs** (leaf; boundary-test sketch below
  is its observable signal). End-to-end scenarios through a running
  harness-with-fake-agent-runner test rig, plus operator docs (CLAUDE.md
  orchestrator section / skills escalation-watcher reference: when to
  reload vs restart, allowlist summary). Modules: orchestrator/tests,
  escalation/tests, docs.
- **ε — deterministic deploy** (leaf; `task_kind='deterministic'`).
  `before_done.script = scripts/restart-all-orchestrators.sh`,
  `target_unit` = own unit (self-restart → detached `systemd-run` path,
  done = `scheduled`), `always_escalates=false` (auto-deploy preset).
  Restarts both orchestrators so the reload machinery itself becomes live —
  the last restart this class of tuning change should need. **Signal:**
  `done_provenance kind='deterministic-deploy-scheduled'` stamp + fresh
  `ActiveEnterTimestamp` on both orchestrator units. Depends on δ. Mirror
  the modularization batch's capstone (df 2002) `before_done` spec at
  decompose time.

G2 note: α and β are intermediates whose consumers (β, γ) are in-batch; γ and
δ are the user-observable leaves; ε is the deploy gate. G6: the two negative
assertions (bad-YAML rejected; non-allowlisted not applied) are backed by
mechanisms produced upstream in-batch (α's validator path and allowlist
filter) and observed via the tool response — rejection observed to fire in
δ's scenarios 1 and 3.

## Boundary-test sketch (δ's signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Bad YAML / failing validator on disk | running harness; file corrupted | `reloaded=false, error` set; live config dump byte-identical to pre-call; event row logged (I1, I7) |
| 2 | Allowlisted change (`models.implementer`) | running harness, fake agent runner | response `applied` contains the path with old/new; **next** implementer spawn's CLI args carry the new model (read-at-spawn observed) |
| 3 | Non-allowlisted change (`max_concurrent_tasks`) | running harness | `restart_required` contains it; dispatch semaphore size unchanged; WARNING journal line (I2, I7) |
| 4 | Mixed change (2)+(3) in one edit | running harness | both dispositions populated; only the allowlisted path mutated; I6 holds (every diff reported exactly once) |
| 5 | Hybrid-invariant rollback (I5). NOTE: unreachable end-to-end with the v1 allowlist — both sides of the only scalar cross-field validator (`timeouts.steward` / `steward_completion_timeout`) are allowlisted, so the fresh file's own validation covers them. I5 is defensive for future allowlist growth; test it in α at unit level with an injected allowlist containing only ONE side of the pair | α unit test, synthetic allowlist `{timeouts.steward}` | `apply_reload` returns `reloaded=false, error=hybrid-invariant…`; every applied field rolled back; live dump identical (I5) |
| 6 | No-op reload (file unchanged) | running harness | `reloaded=true`, empty `applied`/`restart_required`, `unchanged>0` |
| 7 | Standalone server (`harness=None`) | escalation server without orchestrator | error dict, mirrors `halt_scheduler` standalone behavior |
| 8 | Reload while a task is mid-pipeline | architect stage completed under old `models.*` | implementer stage spawns with new value; no error; straddle is the documented behavior (decision 2) |
| 9 | `git.offline_lane_test_threads` change | offline lane enabled in rig | next lane run invokes run-offline-deep.sh with new `--test-threads` while GitOps still holds the SAME GitConfig object (identity assert; I3) |

## Out of scope

- Yellow-tier loop knobs (`*_interval_secs`, `*_enabled`, breaker
  window/floor, `max_per_module`) — would need per-loop stop/restart
  plumbing; file a follow-up PRD if the need materializes.
- Red-tier structural fields (pool sizes, `verify_runners`, escalation bind,
  `sandbox.backend`, merge-lane git fields) — restart-only by design; the
  honest lever for those is a future "drain then restart" mode, not reload.
- Auto-reload on file change; any signal-based trigger.
- Propagation to other processes (fused-memory, dashboard, remote verify
  runners) or to already-running agent subprocesses.
- Per-module `_module_configs` refresh (decision 7).
- Code hot-deploy (deterministic deploy tasks + restart remain the deploy
  path).

## Open questions (surfaced but not decided in this session)

1. **Allowlist representation.** Explicit dotted-path frozenset vs
   introspection helpers (e.g. "all fields of ModelsConfig"). **Suggested
   resolution:** explicit frozenset generated per submodel via
   `model_fields` at import time, asserted non-empty in a unit test — keeps
   the audit property without 60 hand-typed lines. Decide in α.
2. **Event-row payload size.** Full report JSON vs truncated (report can be
   large if many fields change). **Suggested resolution:** full JSON; a
   reload is rare and the audit value is high. Decide in β.
3. **`load_config` thread-off timeout.** `_discover_module_configs` walks the
   tree; bound the `to_thread` call (~30 s) and treat timeout as I1 failure.
   Decide in β.
4. **Doc placement for operators.** CLAUDE.md orchestrator section vs
   skills/orchestrate reference vs both. Decide in δ.
