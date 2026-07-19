# PRD — Reusable off-hot-path integration-test lane (generalize the offline lane), + dark-factory Qdrant instance

**Status:** author-complete, gates walked (2026-07-19). Decompose-ready.
**Slug:** `integration-test-lane` · **Milestone:** verify-pipeline infra (root `plans/`).
**Authoritative brief:** `~/.claude/spawn-briefs/integration-test-lane-prd-brief-2026-07-19.md` (verified against HEAD `c7507a4997`, 2026-07-19).
**Closes:** `plans/cpu-load-robust-verify-prd.md` §9 (the "Qdrant compat coverage lane" open question) — the gap task **2773** creates.

---

## 0. TL;DR — the reframe that drives this PRD

The brief poses the core design as a choice between *(a)* extending `run_main_tip_sweep`,
*(b)* lifting reify's flock/`run_all` executor into core, or *(c)* a new mechanism. Reading
the code shows the reusable engine **already exists and is landed** in dark-factory core:
`orchestrator/src/orchestrator/offline_lane.py` (`OfflineLaneWorker`). It is single-flight,
coalescing, always-from-head, never-a-gate, with a full confirmed-red → dedup'd autofiled
fix-task → staged-L2 red path. **Its only reify-specificity is its two hard-coded run
seams.** So the actual design is **(d): generalize the offline lane's *runner* from
hard-coded reify scripts to a per-project, config-driven command list**, reusing the entire
engine (INV-5, no-lockstep-duplication), then instantiate that config for dark-factory's own
`pytest -m integration` (Qdrant) suite.

Two independent-verification corrections to the brief, both surfaced loud:
- **Solar Challenge Platform is not a second reference implementation.** Its
  `dark-factory-orchestrator.yaml` / `orchestrator.yaml` carry only a `-m 'not slow'`
  *exclusion* with **no lane anywhere** to run the excluded tests. Solar is a second
  instance of the **gap** (exclude-without-a-lane), which *strengthens* the case for a
  reusable capability — it does not provide a second pattern to generalize from. Reify's
  `run_all --scope host-infra` (consumed via the offline lane's infra sub-run) is the sole
  real reference.
- **The general lane engine is already in core** — so this PRD reuses, it does not rebuild.

## 1. Goal & user-observable surface (G1 / G2)

**Goal.** Any dark-factory-targeted project can declare, in its
`dark-factory-orchestrator.yaml`, one or more off-hot-path test-lane commands that run
**serialized, at idle priority, off the merge hot path, never merge-blocking**, and
**autofile a non-blocking fix task** (not a merge block, not a born-at-L2) when a command
confirms red — via the existing offline-lane engine. Then wire that capability for
dark-factory's own live-Qdrant version-compat tests, which task 2773 is moving behind
`@pytest.mark.integration` (removing them from every merge-verify and main-sweep run).

**User-observable surface:**
- A project sets `git.offline_lane_commands: [{name, command, cwd, fix_task_priority}]` in
  its orchestrator config; on each post-merge advance the orchestrator's offline lane runs
  those commands (serialized, from the current `main` head, at idle nice/ionice) and logs a
  run record per command: `offline-lane: <name> sub-run head=<sha> status=PASS|FAIL
  duration=<s>`.
- **Fault-injection (the leaf signal):** a command whose subprocess exits non-zero, whose
  failure reproduces on the serial confirm re-run, causes the lane to **autofile a fix task**
  at the configured priority (default `medium`) + an L0 INFO escalation — **while every merge
  in flight still lands** (the lane never touches the merge queue). A command that passes (or
  whose failure does not reproduce — a flake) files nothing.
- **Dark-factory instance:** with the config live, `pytest -m integration` runs in
  `fused-memory/` at each advance; the run appears in the orchestrator log; the Qdrant
  qdrant-client/mem0 version-compat coverage that 2773 removes from the merge path is
  restored on this lane instead of lapsing.

**Consumers (G1) — both named, both real:**
1. **Any DF-targeted project** wanting a slow / integration / live-service test lane without
   merge-blocking. Config-driven; no per-project code. (Solar's stranded `-m 'not slow'` and
   any future project inherit this.)
2. **Dark-factory's own Qdrant compat coverage** — the direct follow-up to task **2773**
   (δ of `plans/cpu-load-robust-verify-prd.md`, currently `blocked`), which marks the
   compat tests `integration`. `plans/cpu-load-robust-verify-prd.md` §9 flags this gap loud
   (INV-4) and names this PRD as the follow-up.

## 2. Background — what already exists (do not rebuild)

`offline_lane.py::OfflineLaneWorker` (tasks 1951–1959, landed on `main`) is the engine:

| Property | How it holds today |
|---|---|
| **Single-flight** | one loop coroutine + a per-project lockfile (`data/orchestrator/offline_lane.lock`), even across process instances |
| **Always-from-head** | each run snapshots `git_ops.get_main_sha()` at run-start, not at trigger time |
| **Coalescing** | an advance during a run re-sets a dirty flag → exactly one coalesced re-run at the new head |
| **Never-a-gate** | out-of-band background asyncio task; never touches the merge queue or the merge lane's `target/`; a red run is logged + backed off, never raised into the merge path |
| **Idle priority** | run under `DF_VERIFY_ROLE=offline` (nice/ionice owned by the invoked script) |
| **Poll backstop** | a missed trigger is caught by a periodic `get_main_sha` poll; correctness lives in the run-start snapshot |
| **Autofile on red** | confirm re-run (flake filter) → `compute_failing_test_set_fingerprint` → `build_offline_lane_fix_task_arguments` (a queued, gate-routed fix task) + `_file_info_escalation` (L0 INFO) |
| **Dedup** | an already-open fingerprint appends a suspect range instead of re-filing; N confirmed-red advances without the fix landing → staged **born-at-L2** `escalate_blocker` (the stalled-fix backstop only) |

The lane is enabled by `git.offline_lane_enabled` (+ `git.persistent_offline_deep_worktree`
for its warm worktree), started restart-only by `Harness._start_offline_lane`
(`harness.py:7192`). Config knobs: `offline_lane_test_threads`,
`offline_lane_poll_interval_secs`, `offline_lane_red_advances_before_blocker`.

**The reify-specificity is confined to the run seams:** `_default_run_suite` →
`scripts/run-offline-deep.sh` (reify numeric nextest suite, run **unconditionally** in
`_run_once`), and `_default_run_infra` → `tests/infra/run_all.sh --scope host-infra` (reify's
H9 host-exclusive runner, gated on `offline_lane_infra_enabled`). Both are hard-coded module
constants (`_RUN_OFFLINE_DEEP_SCRIPT`, `_RUN_ALL_INFRA_SCRIPT`, `_INFRA_SCOPE`). **A project
without those scripts cannot use the lane** — turning `offline_lane_enabled` on today would
unconditionally invoke the absent `run-offline-deep.sh` and drive a spurious red path.

A **separate** autofile mechanism, `chronic_flake.py` (`ChronicFlakeConfig`,
`TaskWorkflow._maybe_file_chronic_flakes`), files a `medium` **De-flake** task for
tests that are *flaky-but-passing*. It reads reify's `flaky-ledger.jsonl` + `CHRONIC-FLAKY`
markers (reify task 5142), is `enabled: false`, and is **not** the mechanism this PRD needs
(that is the offline lane's *confirmed-hard-failure* red path). It is explicitly **out of
scope** (§10) — conflating the two would duplicate intent.

## 3. The core design decision — generalize the runner, reuse the engine

**Decision D1: add a config-driven, per-project command list; keep the engine untouched.**

Introduce `git.offline_lane_commands: list[LaneCommand]` (default `[]`). `_run_once` runs
each configured command as a sub-run at the one run-start snapshot head, feeding the
**existing** red path. The reify legacy seams are preserved behind their existing flags
(§ D2) so reify is byte-identical; a project like dark-factory sets `offline_lane_commands`
and suppresses the legacy numeric run (it has no `run-offline-deep.sh`).

**`LaneCommand` contract (light B — this is the load-bearing seam):**

```
LaneCommand:
  name:              str                # stable label; appears in the log record + fix-task title
  command:           str                # shell command run in the worktree (e.g. "pytest -m integration")
  cwd:               str = project_root # worktree-relative dir the command runs in
  fix_task_priority: "low"|"medium"|"high" = "medium"   # priority of the autofiled fix task
  enabled:           bool = True        # per-command kill switch
```

Semantics (all reuse existing engine behaviour — no new mechanism):
- **Run:** the worker launches `command` in `<offline-deep-worktree>/<cwd>` under idle
  nice/ionice with `DF_VERIFY_ROLE=offline`, captures rc + output tail. rc 0 = green,
  logged; the head advances only when **all** executed sub-runs pass (unchanged
  `_last_green_head` semantics).
- **Confirm (flake filter):** on red, re-run the **same** command once serially
  (`serial_pytest` form / `-p no:xdist`) as the flake filter (mirrors the infra seam's
  "re-run the full small scope" approach — integration suites are small). Extract
  still-failing node-ids via the existing `verify._extract_failing_test_ids` (covers
  `FAILED` / `ERROR` / `node down`). Empty ⇒ flake ⇒ log only, no task.
- **File:** non-empty confirmed set → `compute_failing_test_set_fingerprint` →
  `build_offline_lane_fix_task_arguments` (extended to take the command's
  `fix_task_priority` instead of hard-coded `high`) → the existing dedup/append + L0 INFO +
  N-advances staged L2 promotion. Content-agnostic — pytest node-ids dedup exactly like
  reify's test IDs.

**Why not the brief's alternatives (all rejected against the code):**
- *Extend `run_main_tip_sweep`* — the main-tip sweep re-runs the whole verify suite with a
  single-retry flake filter but has **no dedup / no autofiled fix-task / no fingerprint**;
  its failure surfaces as a main-sweep escalation. The offline lane is the more evolved
  "extra off-hot-path pass with autofile" — the sweep would have to grow the exact
  machinery the lane already has.
- *Lift reify's `run_all`/flock/flaky-ledger into core* — a 1152-line classification+flock
  apparatus built for reify's `test_*.sh` host-exclusive burn suite; overkill for a
  `pytest -m integration` invocation and it would duplicate the lane's engine.
- *New cron / `before_done.kind='predicate'` deterministic task* — one-shot (a milestone
  fires once), files a **born-at-L2** (blocks + human) on failure rather than a non-blocking
  autofiled fix task, and has no coalescing/from-head/dedup. Directly violates the G2
  pre-answer ("NOT a born-at-L2").
- *Host-global cross-project verify semaphore* — **tried-and-rejected** (reify's 30-min+
  verifies starved dark-factory almost entirely). The offline lane is inherently
  **per-orchestrator-instance / per-project** (its own lockfile, its own config), and host
  CPU contention is handled by **idle nice/ionice**, not a shared gate. This PRD adds no
  cross-project chokepoint and the lane **never** gates a merge.

## 4. Resolved design decisions

| # | Decision |
|---|---|
| **D1** | Generalize via `git.offline_lane_commands: list[LaneCommand]` consumed by `_run_once`; reuse the entire engine + red path. |
| **D2 — reify byte-identical, additive** | Keep the legacy reify numeric/infra seams and their flags exactly as they are. The unconditional legacy numeric run is gated so it only fires when a project actually has `run-offline-deep.sh` — resolved shape: a `git.offline_lane_legacy_numeric_enabled` bool, **default `True`** (reify unchanged), which dark-factory sets `False`. (Exact flag name is tactical; the *decision* — legacy path preserved and per-project-suppressible — is fixed.) `offline_lane_commands` runs **in addition to** whichever legacy seams are enabled. Migrating reify itself onto the generic list is a **future, reify-side** change, out of scope here. |
| **D3 — fix-task priority is per-command** | `build_offline_lane_fix_task_arguments` gains a `priority` parameter (default `high`, so the legacy reify call site is byte-identical); the generic path passes the command's `fix_task_priority` (default `medium`). Satisfies the brief's "low/medium, not a merge block, not a born-at-L2". |
| **D4 — idle priority for a raw command** | The generic runner applies idle nice/ionice to the subprocess itself (the reify scripts self-nice; a bare `pytest` does not), so "very low priority" holds for any command. Belt: `DF_VERIFY_ROLE=offline` env + an idle-class launch. |
| **D5 — reuse the warm worktree** | Reuse `persistent_offline_deep_worktree` / `reset_persistent_offline_deep_worktree` (a git worktree reset to head, untracked scrubbed) as the run dir. The `_offline-deep` name is cosmetic and per-project; Python projects have no `target/` so the CoW warmth is simply unused, not harmful. |
| **D6 — hot-reload the command list, not the start gate** | `offline_lane_commands` (+ `fix_task_priority`) are **green-tier** hot-reloadable (the worker reads them each `_run_once`), added to `RELOADABLE_FIELDS`. Turning the lane on the first time (`offline_lane_enabled`) stays **restart-only** — starting the worker needs a restart, matching the existing gate — so first activation for dark-factory needs a deploy (§8 γ). |
| **D7 — never merge-blocking, per-project** | Non-negotiable invariant, already structurally true of the engine; this PRD adds nothing that can gate a merge and no host-global coordination. |

## 5. Pre-conditions / substrate (G3 — all verified against HEAD this session)

**Reused capabilities (exist on `main`, confirmed by grep/read):**
- `orchestrator/offline_lane.py::OfflineLaneWorker` — the engine (`_run_once` at
  `offline_lane.py:363`; red path `_handle_red_run`/`_file_new_fix_task`). ✔
- `verify._extract_failing_test_ids` (`verify.py:545`) — pytest FAILED/ERROR/node-down
  node-id extractor. ✔ (the generic confirm seam's parser)
- `workflow.compute_failing_test_set_fingerprint` (`workflow.py:467`). ✔
- `workflow.build_offline_lane_fix_task_arguments` (`workflow.py:494`) — currently hard-codes
  `priority='high'`; D3 parameterizes it. ✔
- `verify_cmd.serial_pytest` (`verify_cmd.py:545`) / `verify._serial_pytest_str`
  (`verify.py:1067`) — serial confirm-run form. ✔
- `git_ops.reset_persistent_offline_deep_worktree` / `persistent_offline_deep_worktree_path`
  — warm worktree reset-to-head. ✔ (tactical check at impl: Python-safe, no reify-only
  assumption breaks — expected fine, it is a plain worktree reset.)
- `config.RELOADABLE_FIELDS` green-tier list (`config.py:~3896` already lists the
  `git.offline_lane_*` scalar knobs). ✔
- `Harness._start_offline_lane` / `_stop_offline_lane` / `_note_offline_lane` wiring
  (`harness.py:7192`+). ✔
- `scripts/restart-all-orchestrators.sh --drain` — the sanctioned fleet-redeploy chokepoint
  (drain-aware, clock-stamped). ✔ (γ activation)

**Substrate for the dark-factory instance:**
- `integration` pytest marker declared in `fused-memory/pyproject.toml` (`markers`), and
  `addopts = "-n auto --dist loadgroup -m 'not integration'"`. ✔
- **`pytest -m integration` overrides the `addopts` marker filter — EMPIRICALLY PROVEN this
  session.** `pytest tests/test_mem0_qdrant_integration.py --collect-only` (default) collected
  5 tests; `pytest ... -m integration --collect-only` deselected all 5 → the command-line
  `-m` wins over the ini `addopts -m`. So the lane's `pytest -m integration` selects exactly
  the integration set. ✔ (This is the load-bearing G3 fact the manifest binds.)
- **Task 2773 is a hard prerequisite for the instance.** On current `main`,
  `test_mem0_qdrant_integration.py` has **zero** integration-marked tests
  (`-m integration` collected nothing); the `@pytest.mark.integration` marks are added by
  **2773** (`blocked`). Until 2773 lands, `pytest -m integration` in fused-memory collects
  nothing (the lane runs green/no-op). ⇒ the instance leaf (β/γ) **depends_on 2773**.
- A live Qdrant on `:6333` is reachable on the host running the dark-factory orchestrator
  (the fused-memory prod instance) — the compat tests need it, as they do today. ✔
  (pre-condition, not new substrate.)

**2773 unblock sequencing (activation dependency, documented):** 2773 is `blocked` by
`esc-2773-3` — the very "marking these `integration` removes coverage with no `-m
integration` lane" concern this PRD resolves. Landing α (the lane) closes that substantive
gap; an operator then resolves `esc-2773-3` (option A: the lane now exists) and re-pends
2773. Once 2773 lands the marks, β/γ activate the DF lane. This is **not** a code cycle — α
has no dependency on 2773; only the instance (β) does.

## 6. Cross-PRD relationship + seam ownership (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/cpu-load-robust-verify-prd.md` (§9 open question; task 2773 = δ) | this PRD **closes** its gap | the `-m integration` lane that §9 says "does not exist" | **this PRD** | queued here |
| task **2773** (marks Qdrant tests `integration`) | this PRD's instance **consumes** its output | the `@pytest.mark.integration` marks the lane runs | **2773** (upstream) | `blocked` → re-pend after α lands |
| **reify** (`run_all --scope host-infra`, offline-deep suite) | this PRD **preserves, does not touch** | the legacy `offline_lane` reify seams | **reify** (unchanged) | untouched (D2) |

No reciprocal "the other owns it" ambiguity. This PRD owns the general capability + the DF
instance; reify's existing seams are left byte-identical; 2773 owns the marks. The reify
generic-list migration is a future reify-side PRD, explicitly deferred.

## 7. Boundary-test sketch (the fault-injection leaf signal)

Faces both the producer (a red command) and the consumer (the autofile + non-blocking
guarantee). Proven **hermetically** at task α with injected fakes (no deliberately-failing
test lands on `main`):

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| **T1 real-red autofile** | one `LaneCommand`; its `command` subprocess exits non-zero; the serial confirm re-run reports the same failing node-id X | fake `SuiteRunner` returns (rc=1, tail), fake confirm returns `[X]`, fake `task_client` | a fix task is filed at the command's `fix_task_priority` (default `medium`) via the fake `task_client`; one L0 INFO escalation; the merge queue is never touched |
| **T2 flake, no file** | same, but the confirm re-run returns `[]` | fake confirm returns `[]` | no task, no escalation, log-only |
| **T3 green, no file** | `command` exits 0 | fake `SuiteRunner` returns (rc=0, "") | no task; `_last_green_head` advances |
| **T4 dedup + stall→L2** | same fingerprint red across N advances without the fix landing | N ≥ `offline_lane_red_advances_before_blocker` | first advance files; subsequent advances append a suspect range (no re-file); at N, one born-at-L2 blocker |
| **T5 reify byte-identical** | legacy flags set, `offline_lane_commands=[]` | reify config shape | `_run_once` invokes exactly the legacy numeric (+ infra) seams, unchanged; no generic sub-run fires |
| **T6 legacy-suppressed** | `offline_lane_commands=[qdrant]`, legacy numeric disabled | DF config shape | only the generic `pytest -m integration` sub-run fires; `run-offline-deep.sh` is never invoked |

## 8. Decomposition plan (leaf signals — G2)

Three tasks; α is the keystone (orchestrator core), β/γ the dark-factory instance. Greek
labels; real IDs assigned at decompose.

- **α — Generalize the offline-lane runner to config-driven per-project commands.**
  Add the `LaneCommand` model + `git.offline_lane_commands: list[LaneCommand]`
  (+ `offline_lane_legacy_numeric_enabled`, D2) to `OrchestratorConfig`; make `_run_once`
  iterate the configured commands as sub-runs at the one snapshot head; add the generic
  run + serial-confirm seams that apply idle nice/ionice and reuse
  `_extract_failing_test_ids` / `compute_failing_test_set_fingerprint` /
  `build_offline_lane_fix_task_arguments` (D3: parameterized priority); add the new list
  fields to `RELOADABLE_FIELDS` (D6). Preserve the legacy reify path byte-identically (D2).
  *Modules:* `orchestrator` (`config.py`, `offline_lane.py`, `workflow.py`).
  *Observable signal:* the T1–T6 boundary tests above run in CI (hermetic, injected
  fakes) — a red generic command autofiles a `medium` fix task + L0 INFO with no merge
  interaction; a flake/green files nothing; the reify legacy path is byte-identical.
  *Task kind:* `normal` (full architect path — touches core machinery). *Depends:* none.
  **Keystone — β/γ consume it.**

- **β — Instantiate the lane config for dark-factory Qdrant.** In
  `dark-factory-orchestrator.yaml`: set `git.offline_lane_enabled: true`,
  `git.persistent_offline_deep_worktree: true`, `git.offline_lane_legacy_numeric_enabled:
  false`, and `git.offline_lane_commands: [{name: qdrant-integration, command: "pytest -m
  integration", cwd: "fused-memory", fix_task_priority: "medium"}]`. Add a config-parse test
  (repo-root `tests/scripts/`) asserting the yaml loads into the α schema with that one
  command.
  *Modules:* config (`dark-factory-orchestrator.yaml`, `tests/scripts/`).
  *Observable signal:* the new config parses into `OrchestratorConfig` with the
  `qdrant-integration` `LaneCommand` present (schema-parse test green).
  *Task kind:* `normal`, `metadata.complexity: "simple"` (single-file config + a parse test,
  once α's schema exists). *Depends:* α; **2773** (the integration marks the lane runs).

- **γ — Deterministic-deploy: activate the dark-factory offline lane.** Restart the
  dark-factory orchestrator onto the β config via `scripts/restart-all-orchestrators.sh
  --drain` (the offline-lane start gate is restart-only, D6). This is the true end-to-end
  leaf.
  *Observable signal:* after the deploy, the running orchestrator logs `Offline-deep lane
  worker started`, and on the next post-merge advance logs `offline-lane: qdrant-integration
  sub-run head=<sha> status=PASS …` — i.e. `pytest -m integration` actually executes in
  fused-memory off the merge hot path, while merges continue to land.
  *Task kind:* `deterministic` (`before_done` = the restart script; no LLM pipeline).
  *Depends:* β.

**Edges:** β→α; β→2773 (cross-batch, existing task); γ→β. α is independent and lands first
(it also auto-fires a fleet restart via the live U2 coordinator, since it touches
`orchestrator/src/` — a watched prefix — so the *engine* deploys automatically; γ handles the
β *config* activation, whose yaml is not a watched prefix).

## 9. Out of scope

- **`chronic_flake` (flaky-but-passing De-flake autofile).** A different mechanism for a
  different signal (chronically flaky yet green tests), reify-ledger-gated, `enabled:false`.
  Not wired or changed here (INV-5 — do not conflate the two autofiles).
- **Migrating reify's legacy seams onto the generic `offline_lane_commands` list.** A
  future, reify-side change; this PRD keeps reify byte-identical (D2).
- **Reify's `run_all`/flock/classification-manifest/flaky-ledger substrate.** Stays in reify;
  consumed unchanged via the existing infra seam. Not lifted into core.
- **RED-tier host-CPU-oversubscription levers** (narrow `merge_verify_breadth`,
  `verify_runners`, cap merge `-n`) — a separate human decision on cockpit
  `host-verify-cpu-oversubscription-df`, not bundled here.
- **Any host-global cross-project verify semaphore** — tried-and-rejected; out of scope by
  construction.
- **A generic scheduled/cron trigger.** The lane is post-merge-advance-triggered + poll
  backstop (the engine's existing model); a pure time-based cron lane is not needed for the
  Qdrant case and is not added.

## 10. Open questions (tactical — surfaced, not blocking)

1. **Exact suppression-flag name/shape for the legacy numeric run (D2).**
   `offline_lane_legacy_numeric_enabled` (default True) is the proposed shape; an architect
   may instead express the legacy seams as default entries of `offline_lane_commands`. Either
   preserves reify byte-identically. Decide at α impl.
2. **Confirm-run granularity.** The generic confirm re-runs the *whole* command serially
   (simple; fine for a small integration suite). If a project's lane command is large, a
   "re-run only the failing node-ids" confirm (like the numeric seam) is a later refinement.
   Default to whole-command re-run.
3. **γ vs. natural redeploy.** γ makes activation prompt via an explicit deterministic
   deploy; alternatively the 8h staleness backstop would eventually pick up the β config.
   γ is preferred (deterministic), but an operator may skip it and let the backstop fire.
4. **2773 re-pend is an operator step.** Landing α closes `esc-2773-3`'s substantive gap but
   does not auto-re-pend 2773 (escalations don't auto-re-pend). Recorded so the sequence is
   visible: α lands → resolve `esc-2773-3` (option A) + re-pend 2773 → 2773 lands marks →
   β/γ activate.
5. **`pytest -m integration` under xdist + live Qdrant.** The Qdrant tests flake under CPU
   load (the reason 2773 removes them from the hot path); on the idle-nice lane this is
   tolerated and the confirm re-run (serial) is the flake filter. Tune
   `offline_lane_red_advances_before_blocker` if the live service proves noisy. Tactical.

## 11. Invariants / do-nots

- **Never merge-blocking; never a host-global chokepoint.** The lane is per-orchestrator-
  instance, out-of-band, idle-class. It must never gate a merge and must add no cross-project
  coordination primitive.
- **Reify byte-identical.** With `offline_lane_commands` unset and legacy flags on, behaviour
  is exactly as today (T5).
- **Reuse, don't duplicate (INV-5).** The generic path is a new *invocation*, not a new
  engine — fingerprint/dedup/escalation/confirm all come from the existing red path.
- **Structured facts + corroborate-before-acting (INV-2/3).** The red path already confirms
  (serial re-run) before filing and emits structured escalations — preserved, not weakened.
- **Loud, non-blocking autofile.** A confirmed red files a queued fix task + L0 INFO; only a
  *stalled* fix (N advances) escalates to L2. No born-at-L2 on first failure.
