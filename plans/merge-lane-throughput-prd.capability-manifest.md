# Capability manifest — merge-lane-throughput-prd

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for the
merge-lane throughput batch (A–H). One block per task; every capability the
task's signal asserts is bound to evidence. A binding resolving to a FAIL value
(`declared-only` / `test-only` / `producer-absent` / `producer-downstream` /
`producer-extent-short` / `bound≤floor` / `rejection-absent`) blocks the batch.

**All bindings PASS.** Three premises in the PRD prose were found stale or
unachievable during this walk and are corrected below (§Corrections); each
correction is carried into the filed task's own text, so no leaf is dispatched
against a fiction. No G7 waiver was needed — every invariant hit resolved by
redesign inside the owning task's declared scope.

Substrate verified live on DF `main` `383620ee5b` (2026-09-03), by symbol:

| Capability | Evidence |
|---|---|
| `VerifyRunnerPool` / `RemoteRunner` / `LocalRunner` / `HostAllocator` / `HostLease` / `DriftDetector` / `RunnerUnavailable` | `orchestrator/src/orchestrator/verify_runner.py::{VerifyRunnerPool,RemoteRunner,LocalRunner,HostAllocator,HostLease,DriftDetector,RunnerUnavailable}` |
| `orchestrator verify-merge` CLI (laptop entry point) | `orchestrator/src/orchestrator/cli.py::verify_merge` |
| `orchestrator check-config` | `orchestrator/src/orchestrator/cli.py::check_config` |
| `result_to_dict` (JSON round-trip for the `RunnerBusy` payload) | `orchestrator/src/orchestrator/verify_runner.py::result_to_dict` |
| flock slot primitive for host admission | `shared/src/shared/verify_admission.py::{acquire_task_slot,_acquire,_try_once}` |
| storm counter (INV-4 escape) | `shared/src/shared/storm_counter.py::StormCounter` |
| `merge_verify` event carries `runner`, `duration_ms`, `depth`, `speculative`, `chain_items` | `orchestrator/src/orchestrator/verify_runner.py::VerifyRunnerPool.dispatch` emit block |
| `merge_heartbeat` carries `hosts[].slot_state` and `depth` | `orchestrator/src/orchestrator/merge_queue.py` heartbeat snapshot builder |
| `merge_queued` / `merge_dequeued` / `merge_attempt` / `merge_finalized` / `speculative_merge` / `verdict_voided` / `verify_host_unreachable` | `orchestrator/src/orchestrator/event_store.py::EventType` |
| `verify_host_busy` | **absent on main** — this is what C builds |
| `MergeDeepConfig.chain_cap`, `select_chain_depth`, `merge_finalized.landed_via_chain` | `orchestrator/src/orchestrator/config.py::MergeDeepConfig`, `orchestrator/src/orchestrator/merge_queue.py::select_chain_depth`, `event_store.py::EventType.merge_finalized` payload note |
| `VerifyRunnerConfig` (`name`/`ssh_host`/`git_remote`/`config_path`/`df_checkout_path`/`enabled`), `verify_drift_check_every_n_lands`, `OrchestratorConfig.enabled_verify_runners` | `orchestrator/src/orchestrator/config.py` |
| `select_probe_depth`, `_chain_dead_link` (G's diagnosis targets) | `orchestrator/src/orchestrator/merge_queue.py` |
| `make_flock_contention_result` / `is_flock_contention_failure` (the superseded L2 path) | `orchestrator/src/orchestrator/cli.py`, `orchestrator/src/orchestrator/merge_queue.py` |
| `scripts/restart-all-orchestrators.sh`, `SELF_UNIT=orchestrator-dark-factory.service` | committed + executable; unit is loaded/active on this host |
| `scripts/merge-deep-set-cap.sh` (F mirrors 3189's pattern) | committed `fbaf4dc526`; asserts the reload's `applied` disposition carries the knob |
| `done_provenance kind='deterministic-deploy-scheduled'` | `docs/task-authoring.md` §5 table; `fused-memory/.../task_interceptor.py` |
| `metadata.milestone {mode:'delayed', after_secs}` | `docs/task-authoring.md` §6; `shared/src/shared/task_metadata.py::Milestone` |
| `ssh leo-laptop`, `~/.config/orchestrator/dark-factory-laptop.yaml`, `~/src/dark-factory` checkout | probed live 2026-09-03: host answers in batch mode; both per-host yamls present; checkout at `5f185fdf00` |
| task **3188** (deep-merge chain telemetry) | exists, `status=pending`, `dependencies=[3186]` — wired as a real `add_dependency` edge on F, not re-filed |

---

## Corrections applied during this walk

**(1) The host-global admission directory no longer exists — C must create its own.**
The PRD §Contract binds the laptop-side merge-verify slot to "the existing
host-global admission directory convention (`shared.verify_admission`, default
`/tmp/df-verify-slots-<uid>`)". That default was **removed by task 2501**
(`5392f4a357`, 8 weeks ago): `OrchestratorConfig._default_verify_admission_slots_dir`
now derives `/tmp/df-verify-slots-<uid>-<sha256(project_root)[:12]>` — deliberately
**per-project**, precisely so co-tenant projects stop contending. Reusing
`config.verify_admission_slots_dir` would therefore give each project its own
directory and the arbitration would be a **silent no-op**. C must introduce its
own explicitly host-global slot path (project-independent) and **create** it —
`verify_admission` never creates `slots_dir` and **fails open** when it is
missing (module docstring, "C-fail-open"). Binding held as PASS because the
primitive C needs (`_acquire`/`_try_once` flock slots) exists; only the path
convention was stale. Carried into C's task text.

**(2) Boundary row 1 as written would not exercise the behaviour it protects (INV-10).**
The §Boundary-test sketch row 1 runs "two `verify-merge` invocations on one host
(**fixture project**, subprocess ×2)". With correction (1) applied, one fixture
project shares one slot dir, so row 1 passes whether or not the lock is
host-global — the exact defect the row exists to catch survives it. Row 1 must
use **two distinct project roots** on the host. Carried into C's task text.

**(3) `check-config` does not report runner counts — D1's signal corrected (G6 branch 3).**
D1's PRD signal asserts `orchestrator check-config` "accepts the file and reports
one enabled verify runner". `cli.py::check_config` is an **unknown-key linter**
(it walks the raw YAML against the schema via `census_config_keys` and exits 1 on
a genuinely-unknown key); it emits no runner inventory. The first clause is kept
and is the load-bearing one — `OrchestratorConfig` uses `extra='ignore'`, so
`check-config` exiting 0 is what proves the new `verify_runners` block and C's
new fields are recognized schema fields rather than silently-dropped phantom keys
(`config.py` `extra='ignore'`; the PRD's own §Background names this hazard). The
second clause is re-homed onto the capability that actually exists:
`OrchestratorConfig.enabled_verify_runners` resolving to exactly one runner named
`laptop`, with the `K = 1 + len(enabled_verify_runners) = 2` startup log line as
the post-D2 confirmation. Carried into D1's task text.

**(4) `landed_via_chain` is an int, not a bool (F, exactness).** `event_store.py`'s
`merge_finalized` payload note: "`landed_via_chain` is 1 on an item landed via
chain". F's signal is bound as *truthy `landed_via_chain` (int ≥ 1)*, not
`== true`.

**(5) B's dated snapshot already drifted — the "3 dirty files" are untracked dirs.**
Probed 2026-09-03: `~/src/dark-factory` on leo-laptop is at `5f185fdf00` with
`?? export-2026-04-09-141621/`, `?? src/`, `?? taskmaster-ai/` — three
**untracked directories**, which `git reset --hard` does not touch (only
`git clean -xfd` would). B must re-read the live state at act time rather than
trusting either the PRD's or this manifest's snapshot (INV-3), and must not
`git clean` those paths without the decision-7 inspection. Carried into B's text.

---

## A — `scripts/merge_lane_throughput.py` + fixture test

- **event schema the queries read** (`merge_queued`, `merge_dequeued`,
  `merge_verify.{runner,duration_ms,depth,speculative,chain_items}`,
  `merge_heartbeat.{depth,hosts[].slot_state}`, `merge_attempt.outcome`,
  `merge_finalized.{state,landed_via_chain}`, `speculative_merge`,
  `verdict_voided`) → capability→producer (wired): every member is on
  `EventType` and emitted from production paths on `main` (see table above).
  **PASS.**
- **a readable `runs.db` per project** → `data/orchestrator/runs.db` present
  (170MB, live); read-only `mode=ro` is the established convention
  (`scripts/analyze_speculation_depth.py`, `scripts/merge-deep-canary-predicate.sh`).
  **PASS.**
- **`scripts/tests/` fixture harness** → directory exists with 20+ sibling
  script tests and a `conftest.py`. **PASS.**
- **"prints the § Background rows within the stated caveats"** → G6 branch 1
  (numeric): **no threshold is asserted**. The row set is reproduced against a
  dated baseline, and the three disagreeing occupancy estimators (LOCF integral,
  verify-duration sum, raw-sample fraction) are printed side by side rather than
  reconciled. No achievability basis is required for a reproduction claim whose
  reference is the same query over the same store. **PASS.**

## B — refresh the laptop checkout + per-host config

- **`ssh leo-laptop` in batch mode** → probed live this session (host answered,
  `BatchMode=yes`, no prompt). **PASS.**
- **`~/.config/orchestrator/dark-factory-laptop.yaml`** → present on the laptop
  alongside `reify-laptop.yaml`; already carries `project_root`, `git.remote:
  workstation`, `git.persistent_merge_worktree: false`, and a header block
  documenting why it deliberately omits the verify commands. **PASS.**
- **`uv run orchestrator verify-merge --help` exits 0 on the laptop** →
  `cli.py::verify_merge` exists on `main`; B's own `uv sync --all-packages` is
  what makes it resolvable there (the workspace-console-script hazard is already
  documented in `VerifyRunnerConfig.df_checkout_path`, task 4539). Produced by B
  itself. **PASS.**
- **"nothing would be lost" gate (decision 7)** → G6 branch 4 (rejection): the
  asserted refusal is B's own — a `git reset --hard` is withheld and escalated
  when the live tree carries anything not reproducible from `main`. The
  rejection mechanism is built by B and exercised against the live tree at act
  time; correction (5) above records that today's residue is three untracked
  directories `reset --hard` would not have touched anyway. **PASS.**

## C — cross-project host arbitration (B+H)

- **flock slot acquisition with bounded wait** → capability→producer (wired):
  `shared/src/shared/verify_admission.py::_acquire(slots_dir, n, wait)` is on
  `main` and used in production by the task-verify admission path. **PASS**,
  with correction (1): the *directory* must be a new host-global path C creates,
  not `config.verify_admission_slots_dir`.
- **`RunnerBusy` JSON payload round-trips laptop→workstation** →
  `verify_runner.py::result_to_dict` establishes the stdout JSON convention;
  `RunnerUnavailable` (`verify_runner.py:701`) is the sibling exception class the
  PRD models `RunnerBusy` on. Produced by C. **PASS.**
- **`holder` field is populated, not merely declared** → G6 field-population
  twin. The `verify_host_busy` event and the `RunnerBusy` payload both carry
  `holder`; nothing on `main` writes an owner identity into a slot file, so C
  must have the *holding* process write `project_id` + pid into its slot file for
  the waiter to read one. Produced by C, bound as an explicit deliverable rather
  than assumed. **PASS.**
- **`EventType.verify_host_busy`** → absent on `main` (verified); produced by C,
  upstream of its consumers D1/D2 and of task A's occupancy accounting.
  DAG-direction correct. **PASS.**
- **storm counter above `verify_host_busy_l1_per_hour`** →
  `shared/src/shared/storm_counter.py::StormCounter` exists and is the house
  pattern for INV-4; C reuses it rather than minting a second streak helper
  (INV-5), which also settles PRD §Open questions #2. **PASS.**
- **local fallback on busy** → `VerifyRunnerPool.dispatch` already falls back to
  `LocalRunner` on `RunnerUnavailable`; C extends that arm. Wired production
  path, not a test-only symbol. **PASS.**
- **`merge_verify.fallback_reason: 'remote_busy'`** → new key on the existing
  emit block in `dispatch` (which already writes `runner: actual_runner.name`).
  Produced by C, consumed by A's occupancy attribution — A is *upstream* of C, so
  the attribution section A ships must tolerate the key's absence until C lands;
  bound as a forward-compatible optional key, not as an A-side requirement.
  **PASS.**
- **boundary rows 1–5 green** → the B+H integration-gate signal. Row 1 amended
  per correction (2) to two distinct project roots. Row 5 is a regression pin on
  the unchanged `RunnerUnavailable` quarantine/reprobe path. **PASS.**

## D1 — enable the laptop for Dark Factory (config)

- **`verify_runners` is a live schema field** → `config.py::VerifyRunnerConfig` +
  `OrchestratorConfig.verify_runners`; the field's own docstring records that a
  `verify_runners:` block was silently inert before the model landed. **PASS.**
- **`verify_drift_check_every_n_lands`** → `config.py`, live field. **PASS.**
- **C's arbitration fields** (`verify_merge_host_max_concurrent`,
  `verify_merge_host_wait_secs`, `verify_host_busy_l1_per_hour`) →
  producer:task-C, **upstream** of D1 (real edge). DAG-direction correct.
  **PASS.**
- **`orchestrator check-config` exits 0 on the edited yaml** →
  `cli.py::check_config` exists; exit-0 means no genuinely-unknown key. This is
  the clause that proves nothing was silently dropped. **PASS.**
- **"reports one enabled verify runner"** → **corrected**, see correction (3):
  re-homed onto `OrchestratorConfig.enabled_verify_runners` (`config.py`), which
  does exist and does answer the question. **PASS after re-homing.**

## D2 — deterministic deploy

- **`scripts/restart-all-orchestrators.sh`** → committed, executable,
  `SELF_UNIT` defaults to `orchestrator-dark-factory.service`, which is a loaded
  and active user unit on this host. **PASS.**
- **`done_provenance kind='deterministic-deploy-scheduled'`** → the documented
  provenance kind for a `DeterministicRunner` self-restart scheduled via detached
  `systemd-run` — exactly D2's shape (the unit being restarted runs the task).
  **PASS.**
- **`verify_runners` needs a restart, not a reload** → `OPERATIONS.md` §"Config
  reload vs restart" lists it red-tier; D2 is why D1 alone is not the deploy.
  **PASS.**
- **"within 24h a `merge_verify` event with `runner: 'laptop'`"** → G6 branch 1:
  the 24h window is not a tuned threshold but a consequence of the measured
  landing rate (dark_factory median 12 landings/day over the 14-day baseline), so
  ≥1 merge verify inside 24h is near-certain; `runner` is written on every
  `merge_verify` emit. `verify_host_unreachable` is an existing `EventType`, so
  "stays at zero" is observable rather than aspirational. **PASS.**

## E — 14-day measurement after the laptop lever

- **`scripts/merge_lane_throughput.py`** → producer:task-A, upstream (A is a
  transitive prerequisite via D1→D2 and also a direct edge). **PASS.**
- **the after-window data exists when E fires** → `metadata.milestone
  {mode:'delayed', after_secs: 1209600}` anchors on E's dependency set going
  `done`; its last dependency to complete is D2, so the 14 days are measured from
  the deploy, which is what makes the window an *after* window. Frozen-once
  anchor semantics per `docs/task-authoring.md` §6. **PASS.**
- **no numeric target** → G6 branch 1: PRD decision 3 forbids asserting a
  percentage; E asserts a *comparison against the dated baseline* only. The
  achievability basis for direction (not magnitude) is reify's measured laptop
  verify p50 17.4 min vs 43.4 local on the same workload class. **PASS.**

## F — deep-chain canary on Dark Factory

- **`MergeDeepConfig.chain_cap` + `select_chain_depth`** → on `main`; the
  config docstring's "nothing reads the knob yet" is stale (the PRD says so and
  `select_chain_depth` proves it). **PASS.**
- **chain telemetry to read the canary by** → **producer:task-3188**
  (out-of-batch, exists, `pending`), wired as a real `add_dependency` edge.
  DAG-direction correct: 3188 is upstream of F. **PASS.**
- **`scripts/merge-deep-set-cap.sh`** → committed; sets the knob, commits only
  that file, hot-reloads via the target's escalation MCP and asserts the knob
  appears in the reload's `applied` disposition, exiting non-zero otherwise —
  which is the PRD's "if the reload says `restart_required`, restart per D2's
  recipe" branch, surfaced as a born-at-L2 rather than a silent skip. DF's
  escalation port is 8102. **PASS.**
- **`merge_finalized.landed_via_chain` observable within 7 days** → G6 branch 1.
  Basis stated by the PRD: dark_factory queue depth p90 = 8 over the 14-day
  baseline, so chains of ≥2 form routinely at cap 6, and the deep-merge PRD's
  own study validated cap 6. Bound is a *direction with a window*, not a rate.
  Exactness corrected per (4): truthy int ≥ 1, not `== true`. **PASS.**

## G — speculation-waste diagnosis

- **`scripts/merge_lane_throughput.py --speculation`** → producer:task-A,
  upstream (real edge A→G). G may extend A's script; both tasks therefore
  declare `scripts/merge_lane_throughput.py` and are serialized by that edge
  (same-file rule). **PASS.**
- **the code symbols the diagnosis must cite** (`select_probe_depth`,
  `_chain_dead_link`, `verdict_voided`/`chain_dead`, K=2 placement, `gate_retry`
  / CAS retry) → all present on `main` (see table). **PASS.**
- **the 58% / 30% / 3.6% / 40% rates** → G6 branch 1: these are *measured*
  baseline observations to be explained, not thresholds to be hit. G's signal
  binds reproduction of the rates by the script, not attainment of a target.
  **PASS.**
- **consumer** → the policy follow-up PRD is unauthored, so G's named consumer
  is **Leo's authoring session** plus task A. Recorded as the PRD's own G1
  resolution (a) — `blocked-on-consumer` in the §Cross-PRD table. G introduces no
  mechanism, only a report, so no producer-orphan is created. **PASS.**

## H — 14-day measurement after the chain lever

- **`scripts/merge_lane_throughput.py` (incl. `--chains`)** → producer:task-A,
  in H's **transitive** dependency closure via F→E→A. H's only direct dependency
  is F, by operator instruction, so the 14-day clock anchors on F rather than on
  E. DAG-direction correct. **PASS.**
- **`--chains` section (`landed_via_chain`, chain length)** → producer:task-A
  (declared in A's spec), upstream. **PASS.**
- **no numeric target** → same as E: comparison against the dated baseline,
  no threshold. **PASS.**

---

## G7 walk (design invariants) — every task, no waivers

Walked against `docs/legibility/design-invariants.md` (INV-1..INV-11) on
`383620ee5b`. Hits and their resolutions, all inside the owning task's declared
scope:

| Invariant | Hit | Resolution (carried into the task text) |
|---|---|---|
| `contracts-machine-checked` | C's `RunnerBusy` wire shape would be a prose convention between two processes (PRD §Open questions #1) | the payload is a shared typed model validated on parse; an unparseable/mismatched payload is a typed error, never a silent `passed=False` |
| `structured-facts-at-failure` | — | already satisfied by design: `verify_host_busy` carries `{host, holder, waited_secs, fallback}`; A reads events, never logs |
| `corroborate-before-acting` | B resets a remote tree read earlier | B re-reads the live tree immediately before acting; the PRD's and this manifest's snapshots are provenance, not premises (correction 5) |
| `storm-escape-required` | C adds a fallback that can fire per merge | `StormCounter` + one L1 above `verify_host_busy_l1_per_hour`, never an L2 — already in §Contract |
| `no-lockstep-duplication` | C could mint a second streak helper; E/H could each re-derive the report queries | reuse `shared.storm_counter.StormCounter`; all query logic lives once, in A's script, which E/H invoke |
| `status-matches-liveness` | a host marked busy could outlive its holder | `busy_until = now + retry_after` expires by construction; the flock slot is released by process exit |
| `holds-owned-and-bounded` | the laptop-side slot is a hold | owner written into the slot file (`project_id` + pid) and surfaced as `holder` on the event; waiter bounded by `verify_merge_host_wait_secs`; holder bounded by process lifetime |
| `loop-thread-occupancy-bounded` | a bounded wait could be implemented as a workstation-side busy-poll | the wait happens on the **laptop** inside the `verify-merge` subprocess; the workstation arm stays a single awaited dispatch |
| `one-fact-one-home` | the PRD's §Background table, E's report and H's report all hold the same measured facts | home is `runs.db` via A's script for a stated window; every table is dated provenance that names its window, and E/H's signals require agreement with a live script run |
| `guards-exercise-behaviour` | boundary row 1 would pass without the behaviour it protects | row 1 uses two distinct project roots (correction 2) |
| `no-silent-fail-soft` | `verify_admission` is documented **C-fail-open**: a missing slots dir silently disables admission | C creates the host-global dir up front and distinguishes three outcomes — acquired / busy (`RunnerBusy`) / **admission-unavailable**, the last emitted as a distinct storm-counted signal rather than proceeding as if acquired |

---

## Filed batch (2026-09-03)

| Label | Task | Depends on | Priority | Kind |
|---|---|---|---|---|
| A | 5050 | — | high | normal |
| B | 5051 | A | high | normal |
| C | 5052 | A | high | normal (`design_first`) |
| D1 | 5053 | A, B, C | high | normal |
| D2 | 5054 | D1 | high | deterministic (`before_done` deploy) |
| E | 5056 | A, D2 | medium | normal + `milestone{delayed, 1209600}` |
| F | 5057 | **3188**, E | medium | deterministic (`before_done` deploy) |
| G | 5058 | A | medium | normal |
| H | 5059 | **F only** | medium | normal + `milestone{delayed, 1209600}` |

Task **3188** (deep merge-ahead ε, chain telemetry) is an existing out-of-batch
task wired as a real `add_dependency` edge on F — not re-filed. H's single edge
to F is deliberate (PRD §Open questions #3): the `delayed` anchor stamps the
first tick *all* dependencies are `done`, so a sole dependency on F starts the
14-day clock at the chain lever rather than at E. A reaches H through the
transitive closure F→E→A. Nothing in this batch is wired behind tasks 3730,
3733, 4755 or 5020.

---

## Post-review findings (2026-09-03) — the "all bindings PASS" claim did not hold

After filing, four independent read-only agents re-walked this batch. The walk
above was made at `383620ee5b`; these findings were verified by hand at
`c0dc5f8926`. **Two bindings above are revised: one to FAIL, one to OPEN.** The
sidecar carries the revised verdicts; the original text is left in place as
dated provenance rather than rewritten.

### Blocking

**P1 — an unreconciled RED-TIER ruling forbids task C's mechanism.**
`plans/cpu-load-robust-verify-prd.md` § 6 ("Out of scope — RED-TIER, human
decision") and `plans/integration-test-lane-prd.md` § 11 both rule out a
host-global cross-project verify admission semaphore by name:
*"TRIED-AND-REJECTED, not merely deferred: under a single fair global semaphore,
Reify's very long (30min+ cargo) verifies starved dark_factory almost entirely
… Do not revisit this as an option."* Neither PRD is cited in the throughput PRD
or in this manifest — the G4 seam check missed them. This is the same rejection
that produced the per-project slots dir of correction (1); they are one fact.
C's shape is plausibly distinguishable (merge-only, capacity 1, bounded wait,
local fallback) but P2 shows the distinguishing property is broken. **Leo's call,
not an implementer's.** Task 5052 carries a STOP-AND-RECONCILE block.

**P2 — `local-fallback-arm-already-wired` revised PASS → FAIL
(`producer-extent-short`).** `merge_queue.py::_run_post_merge_verify` builds the
remote pool as `VerifyRunnerPool([runner], …)` with **no `LocalRunner`** — its
own comment says so — so `dispatch`'s `except RunnerUnavailable` arm hits
`self._local is None` and **re-raises**. The real fallback decision lives in
`merge_queue.py`, which C must not edit (PRD decision 6; quality-PRD Appendix A).
Knock-on: the cross-PRD table's κ row ("the contention branch becomes
unreachable") is false too, so κ must not perform that deletion.

**P3 — `storm-escape-reuses-shared-counter` revised PASS → OPEN.** C's storm
escape cannot fire in production for two independent reasons: the pool is
constructed **per merge verify**, so a pool-instance counter never accumulates a
per-hour rate (contrast `HostAllocator`, which is worker-lifetime); and
`VerifyRunnerPool.__init__` takes **no `escalation_queue`**, so the pool cannot
file the L1 at all. Boundary row 3 would go green on a hand-built pool while the
production path is dead — the same silent-no-op-that-passes-a-naive-test shape as
correction (1), one layer up (INV-10).

### Corrected in the filed tasks

- **A** — the signal named `--window 30d`, but § Background is a **14-day**
  window except four explicitly-30d rows; and `--project-root` must be
  **repeatable**, since the `events` table has no project column and A's own
  `--speculation` section promises "void rate by project" (which G's and E's
  signals then depend on).
- **D1/D2** — "the K value logged at startup will be 2" is unachievable;
  `Harness._speculation_k` is computed but never logged.
- **D2** — "zero `verify_host_unreachable`" is deduped behind an open L1 and
  skipped when the queue is unwired, so zero ≠ reachable.
- **F** — its 7-day signal cannot gate a task that completes on a deploy
  script's exit code; D2 had the de-scoping sentence and F did not.
- **H** — INV-7 was not walked for the milestone tasks: H's sole dependency on F
  means a regressed F leaves H an **unbounded** hold (timer elapsed, deps
  unsatisfied) with no overdue surface. Also gained E's anchor-persistence note.
- **G** — its signal's invocation omitted `--project-root` while quoting
  two-project rates.
- **3188's dependencies are `[3186, 5036]`, not `[3186]`** — 5036 is the quality
  PRD's package-move gating anchor, so F and H are transitively behind it.

### `delivered_checks` audit

Gates are re-evaluated against **current `main` every tick** (only `DELIVERED` is
cached; any commit invalidates), so a check can go red *after* gating begins.
Three were changed: `storm_counter` was **vacuously green** (`harness.py` already
imports `StormCounter`) and is now scoped to C's own production files;
`verify_runners:` is anchored to line start; and F's `chain_cap` grep — which
gated H fourteen days later against a hand-edited ops file — was **removed**
rather than hardened, since F cannot reach `done` without the knob and H is a
measurement task for which a reverted cap is data, not grounds to withhold. All
eight surviving gates verified red.

### Not found

G6 branch 1 (numeric premises) came back clean under adversarial review: D2's
24h and F's 7d are windows on a direction with stated measured bases, C's
`verify_host_busy_l1_per_hour: 20` is a config default rather than an asserted
achievement, and E/H/G assert no thresholds. The `.md` and `.yaml` twins agree
with each other; the divergence was between both of them and the PRD, which the
PRD's own § Corrections block now closes.

---

## RULED 2026-09-03 — task C cancelled; P1 and P2 resolved by removal

Leo's ruling on the P1/P2/P3 findings above: **task C (5052) is cancelled**, and
D1 no longer depends on it. The question that settled it — *why can two projects
verify concurrently on the workstation but not on the laptop?* — exposed that
C's founding premise was false.

**The two projects cannot collide.** The laptop-side merge-verify lock is scoped
to `project_root / config.worktree_dir` (`git_ops.py`), so reify's locks are
under `/home/leo/src/reify/.worktrees/` and dark_factory's under
`/home/leo/src/dark-factory/.worktrees/`. The born-at-L2
`FLOCK_CONTENTION_CATEGORY` branch is a **within-project orphan detector**, not a
co-tenant detector. And nothing else serializes merge verifies across projects on
any host: `verify_admission.py::acquire_task_slot` returns immediately for any
role outside `{task, background}` ("C-merge-priority"), which is exactly why both
projects already share the workstation ungated.

**Consequences for this manifest:**

- Every C capability is now `OPEN` and marked MOOT; C's block is retained as the
  record of why the design could not have worked as specified.
- D1's `arbitration-config-fields-upstream` binding is `OPEN`/MOOT — the three
  arbitration config fields will never exist. D1 adds only `verify_runners` and
  `verify_drift_check_every_n_lands`, both pre-existing schema fields.
- **P1 (the RED-TIER prohibition) is resolved by compliance, not by exception.**
  Nothing host-global is being built.
- **P2's knock-on outlives the cancellation:** the throughput PRD's κ row is
  wrong either way — C would not have made the contention branch unreachable, and
  with C gone the branch is plainly still live. **κ must not delete
  `is_flock_contention_failure`.** This is the one item that needs carrying to the
  quality PRD.
- The decompose session's "1–3 blocked merges per day without arbitration"
  estimate is **retracted** — it was arithmetic on the false premise.

**What replaces the mechanism:** measurement. Task E now carries laptop
oversubscription as a headline section — reify's laptop verify p50/p90 against
its 17.4 / 20.6 min baseline, dark_factory's against its 40.7 / 52.4 min local
baseline, and the overlapping-span contention rate. The residual risk is verify
*duration*, which cannot block a merge.

Revised DAG: **A → B → D1 → D2 → E → F → H**, with **A → G**.
