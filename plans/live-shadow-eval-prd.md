# PRD: Live paired shadow eval — measuring candidate models on the current codebase

**Status:** active — authored 2026-09-10 (interactive design session with Leo; the four
shape/record/slate/supersession decisions and the architect-consequence shape were ruled in
conversation and are honored here, not re-litigated). **Project:** dark_factory (the
mechanism runs in every orchestrator process, so reify is covered by the same code).
**Approach:** **B+H** — G5 heuristic hit on three axes: cross-module blast radius ≥ 3
(`orchestrator/harness.py`, `orchestrator/evals/*`, `orchestrator/config.py`,
`orchestrator/cli.py`, `orchestrator/run_store.py`), a **load-bearing seam** (the harness
dispatch path and the merge-lane landing event), and **two cross-PRD consumers** (the fable
η gate and the adaptive-routing flip).

> **Code anchors** verified against main `32d0f6d1a7` (`2026-09-10`). Main moves fast —
> cite-by-symbol; re-locate at implementation time.

## Goal

Replace the frozen fixture corpus as the **capability instrument** with a live, paired
shadow evaluation: when the scheduler dispatches a sampled production task, a candidate
model is run in the role under test against the **same task at the same base commit**, in a
throwaway worktree that never enters the merge lane, and its outcome is scored against what
production actually landed.

**User-observable surface (G2 top signal):** one CLI invocation —

```
orchestrator eval-shadow report [--shape implementer|architect|architect-consequence|end-to-end] [--candidate NAME]
```

— prints, per candidate and shape, the **paired** delta against the incumbent over the
cells settled so far: pair count, the shape's primary outcome with a bootstrap CI, cost per
usable outcome, the decline split (`terminal_kind`), and the cap-excluded / unreferenced /
expired counts. Every line traces to rows in one record store; nothing is derived from
transcripts or logs.

**Named consumers (G1):**

1. **esc-3637-1 / task 3637** (fable-trial-v2 η, the admission re-ruling gate, pending
   since 2026-08-30). Leo ruled 2026-09-10 that *this PRD is the redesigned eval set* the
   v2 decision record deferred to. The gate's ruling consumes this PRD's report over the
   `architect-consequence` shape for `architect-fable-max` vs `architect-opus-max`.
2. **`plans/adaptive-model-routing-prd.md`** — any production routing change stays a
   separate deterministic-deploy task filed on a favourable report (this PRD measures; it
   never flips config).
3. **`plans/eval-framework-revival-prd.md` decision 9's *confirm* stage** — the
   `end-to-end` shape is its live analogue; the offline `eval-confirm` driver is no longer
   the path to a production change.

## Background — why the frozen corpus cannot answer the question

Measured 2026-09-09 against main:

| Corpus | Fixtures | With frozen plan | Base age | Commits behind HEAD |
|---|---|---|---|---|
| `evals/tasks/` | 22 | **6** (five are April fixtures) | Apr–Jul 2026 | 7k–65k |
| `evals/tasks_hard_v2/` | 39 | 3 | Apr–Jul 2026 | 6k–65k |

- The **implementer** screen needs a frozen plan; five of the six plan-bearing fixtures are
  the April tasks (`df_task_12/13/18`, `reify_task_12/27`) — a codebase 59k–65k commits
  ago. Main moves ~400 commits/day; any frozen fixture is weeks from irrelevance.
- **Historical replay is incoherent by construction.** The eval worktree shares the live
  object DB, refs, task DB and escalation queue, so an honest architect sees the task
  already landed and correctly declines — 47 of 53 cells in fable-trial-v2 tranche 1
  (`plans/fable-architect-trial-v2-decision-2026-08-30.md` §8). Tasks 4758/4844 try to
  patch the replay frame; a live eval makes the frame moot.
- The non-Claude candidate bundles (`evals/configs.py::claude_endpoint_candidates`,
  `codex_pi_candidates`) have **never produced a result**: `eval-ofat` hardcodes
  `ofat_candidates()` and excludes them, the slate is pinned to July-12 model ids, and no
  provider key is present on the host.
- **The v1 fable verdict tied on plan quality** (0.919 vs 0.909, six near-ceiling
  fixtures); the judge scores a plan against the incumbent's landed diff, so a better plan
  that takes a different route is penalised. Leo's live hypothesis — *Opus can plan; Fable
  plans better* — needs an instrument that measures a plan by its **consequences**, not its
  resemblance.

## Premise (G6)

- `run_eval`, `run_architect_eval`, `run_end_to_end` (`evals/runner.py`) accept a fixture
  path and are driven end-to-end by the July revival; their fixture contract is the
  `load_task` dict (`id`, `project_root`, `pre_task_commit`, `task_definition`,
  `verify_commands`, `plan`, `reference.post_task_commit`, timeouts). **Verified.**
- Production persists the accepted plan durably at `.task-meta/<task>/plan.json`
  (`artifacts.py::TaskArtifacts.write_plan`, written by `mcp/plan_tools.py`). **Verified.**
- The task record carries `metadata.branch_base_sha` (the dispatch base). **Verified.**
- `EventType.phase_enter` / `phase_exit` carry `(task_id, phase)`;
  `EventType.merge_finalized` carries `branch`, `state`, `merge_sha`, `snapshot_tip`,
  `superseded_by`; `EventStore.latest_merge_finalized(branch=)` and
  `fetch_events_by_type` exist. **Verified.**
- `EvalMetrics` already carries `terminal_kind` (task 4760), `cap_tainted`,
  `role_under_test`, `cost_source`; `EvalResult` is persisted by `save_result` as
  `<task>__<config>__<run_id>.json`. **Verified.**
- The dispatch load gate is `scheduler.py::_phase_psi_gate` over `shared.psi::saturated`.
  **Verified.**
- The harness's `_maybe_auto_eval` sibling-task redo hook has fired **0 times ever**
  (`plans/author-declared-complexity-prd.md` §Out of scope) and files real tasks; it is
  **not** reused (decision 3).
- Isolation is **not** in place today: eval cells write into live fused-memory,
  escalations and the task DB because `run_architect_eval` leaves
  `strict_mcp_config=False` (task **4757**, pending, high) and eval-lane escalations reach
  the L2 queue (task **3096**, pending). This PRD's dispatching leaves **depend on both**.

## Sketch of approach

A `ShadowCoordinator` lives in each orchestrator process. It watches the event stream the
harness already emits, decides — deterministically, per sampled task and candidate — which
**cells** to open, runs each cell through the existing eval runners against a **live
fixture** built from the production task, and **settles** each cell when production's own
outcome for that task becomes known. Cells are rows in a `shadow_cells` table; results are
`EvalResult` JSONs; branches are `shadow/<task>/<cell_id>`; nothing about a cell is a task,
an escalation, a merge request or a memory write.

### The four shapes

| Shape | Trigger | Candidate runs… | Shared with production | Primary outcome (paired vs production) |
|---|---|---|---|---|
| `implementer` | `phase_exit(PLAN)` on the production task | the implementer, on production's plan.json, at `branch_base_sha` | plan, base | `tests_pass`, `review_blocking_issues`, `iterations`, `cost_usd` to terminal |
| `architect` | `phase_enter(PLAN)` | the architect, live, plan-only | base, briefing | `terminal_kind` split, `plan_quality` (judge, secondary) |
| `architect-consequence` | as `architect`, then on the candidate plan | leg 1: candidate architect; leg 2: the **incumbent** implementer on the candidate's plan | base, implementer | leg-2 `tests_pass`, `review_blocking_issues`, `iterations`, `cost_usd`; judge secondary |
| `end-to-end` | `phase_enter(PLAN)` | candidate architect **and** candidate implementer | base | as `implementer`, plus $/done |

`architect-consequence` is the instrument for Leo's hypothesis: same task, same base, same
implementer; only the plan differs. `architect` alone stays cheap and answers the decline
question. `end-to-end` is the live confirm stage, opened only for a survivor candidate.

### Life of a cell

`sampled → opened → running → awaiting_reference → settled | cap_excluded | expired | failed`

1. **Sample.** On the trigger event, `should_open(task, candidate, shape)` hashes
   `(task_id, candidate, shape, seed)` against `sample_rate` within the task's stratum
   (`task_sampler.py::classify_kind / classify_path / repo_of_project`), then checks the
   daily USD cap, `max_concurrent`, the host `saturated()` gate and the candidate's
   own key presence. Every negative decision is a `shadow_cell_skipped` event with a
   reason (INV-11).
2. **Open.** Write the `shadow_cells` row, build the live fixture, create the
   `shadow/<task>/<cell_id>` worktree at `branch_base_sha` (`snapshots.py::create_eval_worktree`).
3. **Run.** Dispatch through the existing runner for the shape, under the shadow
   invocation profile: `EVAL_PROFILE` + `strict_mcp_config=True` + null-routed memory +
   the landlock sandbox for implementer legs. A usage-cap hit ends the cell as
   `cap_excluded`; there is no retry — production owns the account pool.
4. **Await reference.** On `merge_finalized(state=done)` for `task/<task_id>`, the
   reference is `merge_sha` against `branch_base_sha`. On any other terminal state
   (blocked, cancelled, superseded, expired) the cell settles with `reference_kind=none`.
5. **Settle.** Score per shape (contract C4), write `settled_at`, save the `EvalResult`,
   emit `shadow_cell_settled`. A cell whose production task has no terminal event by
   `settle_deadline` settles as `expired` — never left open (INV-7).

## Resolved design decisions

1. **(Leo) Live paired shadow replaces the frozen corpus as the capability instrument.**
   `evals/tasks/` is retained only as the instrument regression corpus for the runner's own
   tests; `evals/tasks_hard_v2/` is retired as capability evidence. Eval-revival decision 1
   ("fixed target") is amended by the companion task in Phase 4.
2. **(Leo) Four shapes in scope:** `implementer`, `architect`, `architect-consequence`,
   `end-to-end`. `end-to-end` opens only when `shadow_eval.end_to_end_candidates` names
   the candidate — it is the confirm stage, not a screen.
3. **(Leo) Cells are not tasks.** They live in `shadow_cells` (runs.db, owned by
   `run_store.py`) plus `EvalResult` JSONs; they carry no task id of their own, never
   enter the curator, the merge queue, the orphan reapers or the dashboards' task views.
   The `auto_eval` sibling-task path is not reused.
4. **Reference and settle rule.** The reference is production's landed diff. If
   production does not land, the cell is scored on what does not need a reference
   (verify gates, review outcome, plan structure, `terminal_kind`) and tagged
   `reference_kind=none`; it is never scored as a pass or a fail by default.
5. **Pairing is within task.** Every report statistic is a paired difference
   (candidate cell minus production's own metrics for the same task, read from
   `task_results` / the invocation ledger), with a bootstrap CI over pairs. Unpaired
   cells (production unreferenced *and* the shape needs a reference) are counted, not
   averaged in.
6. **Cap policy.** A shadow cell that hits a usage cap is `cap_excluded`, not retried and
   not re-sampled. Production always wins the pool (consistent with the 2026-08-30 ruling
   on task 4741: evals share the fleet pool and have no reserve).
7. **Load policy.** A cell counts as a dispatch for the PSI/runqueue gate: it is opened
   only when `saturated()` is false and `shadow_eval.max_concurrent` has a free slot
   (default 1). It never pre-empts a production dispatch.
8. **Sampling is deterministic and stratified.** `should_open` is a pure function of
   `(task_id, candidate, shape, stratum, seed, sample_rate)`, so a cut is reproducible
   and both projects' coordinators agree without coordination.
9. **Isolation is a hard prerequisite, owned upstream.** Dispatching leaves depend on
   4757 (strict MCP + no live-store writes) and 3096 (eval-lane escalation containment).
   Until both land, the coordinator refuses to open cells and says so
   (`shadow_cell_skipped reason=isolation_unavailable`).
10. **Storm escape (INV-4).** The coordinator pauses itself and files exactly one
    escalation when: the daily USD cap trips, or `failed`+`expired` cells reach a streak of
    `shadow_eval.failure_streak_pause` (default 5), or the cell/production cost ratio for
    a candidate exceeds `shadow_eval.cost_ratio_ceiling` over the trailing window.
11. **One home (INV-9).** The `shadow_cells` row is the authoritative record of a cell;
    the `EvalResult` JSON carries the row id and the report renders from rows. No cell
    fact is copied into task metadata, escalations or memory.
12. **Candidate slate refresh is folded in (Leo).** One leaf updates the endpoint and
    Codex model ids and prices in `evals/configs.py` with strings Leo supplies (Codex
    Sol/Astra, GLM-5.3 and GLM-5.3-flash, the current MiniMax, and whatever else he
    names). The `eval-ofat` candidate-selector gap is **not** fixed: the coordinator is
    the screen.
13. **No production config change** is applied by anything in this PRD (as every eval
    PRD in this lineage).

## Pre-conditions for activating

- Tasks **4757** and **3096** landed (isolation). Phase 1 leaves may proceed; Phase 2's
  dispatching leaf and everything after it depend on both.
- Provider keys present in the orchestrator's environment for any endpoint bundle to be
  sampled (`ZAI_API_KEY`, `MINIMAX_API_KEY`, …); a missing key is a per-candidate
  `shadow_cell_skipped reason=no_credentials`, never a run that 401s.
- Nothing else is novel: every substrate item in §Premise exists on main.

## Cross-PRD relationship

| Other PRD / owner | Direction | Mechanism crossing the seam | Owner of the integration |
|---|---|---|---|
| `eval-framework-revival-prd.md` | **consumes** | `run_eval` / `run_architect_eval` / `run_end_to_end`, `EvalResult`/`EvalMetrics`, `save_result`, the judge, `build_eval_orch_config`, `task_sampler` classifiers | revival owns the runners; **this PRD** owns the live fixture builder and any additive fields on `EvalMetrics` (`shadow_cell_id`, `reference_kind`) |
| task **4757**, task **3096** | **consumes** | strict MCP config on eval invocations; eval-lane escalation containment | those tasks; this PRD's ε2 depends on them and adds the deterministic check that a shadow invocation carries them |
| `fable-architect-trial-v2-prd.md` / esc-3637-1 | **produces** | the `architect-consequence` report for `architect-fable-max` vs `architect-opus-max` | this PRD (κ gate); 3637's ruling consumes it |
| `adaptive-model-routing-prd.md` | **produces** | a favourable report as the evidence a flip task cites | that PRD's flip task, filed by Leo's ruling — never by this PRD |
| `load-throttle-harmonisation-prd.md` | **consumes** | `shared.psi::saturated()` / `tripping_metric()` for the open decision | that PRD; this PRD adds no arm |
| `usage-gate-model-scoped-caps-prd.md` | **consumes** | `UsageGate` via `_build_eval_usage_gate`; `cap_hit` | that PRD; the `cap_excluded` disposition is this PRD's |
| `os-sandbox-worktree-containment-prd.md` | **consumes** | landlock wrapping for implementer legs in `shadow/` worktrees | that PRD; this PRD passes the worktree path, nothing more |
| the harness `_maybe_auto_eval` hook | **none** | not reused (decision 3) | — |

## Contract (H)

### C1 — `ShadowCell` record (`run_store.py`, table `shadow_cells`)

```
shadow_cells(
  cell_id TEXT PRIMARY KEY,            -- ulid
  project_id TEXT NOT NULL,
  task_id TEXT NOT NULL,               -- the production task
  shape TEXT NOT NULL,                 -- implementer|architect|architect-consequence|end-to-end
  candidate TEXT NOT NULL,             -- EvalConfig.name
  incumbent TEXT NOT NULL,             -- the pinned incumbent config name for the role
  stratum TEXT NOT NULL,               -- "<repo>×<kind>×<path>"
  base_sha TEXT NOT NULL,              -- task.metadata.branch_base_sha at open
  plan_source TEXT,                    -- 'production' (implementer) | 'candidate' (consequence leg 2) | NULL
  state TEXT NOT NULL,                 -- sampled|opened|running|awaiting_reference|settled|cap_excluded|expired|failed
  reason TEXT,                         -- populated on cap_excluded|expired|failed
  reference_kind TEXT,                 -- landed|none  (NULL until settled)
  reference_sha TEXT,                  -- merge_sha when landed
  result_path TEXT,                    -- EvalResult JSON once saved
  cost_usd REAL NOT NULL DEFAULT 0,
  opened_at TEXT NOT NULL, running_at TEXT, settled_at TEXT,
  settle_deadline TEXT NOT NULL,       -- opened_at + shadow_eval.settle_deadline
  owner TEXT NOT NULL DEFAULT 'shadow_coordinator'
)
```

Invariants: a `(task_id, shape, candidate)` triple opens at most once per trial index;
`state` transitions are monotone along the lifecycle above; every non-`settled` terminal
state carries a non-empty `reason`; `settle_deadline` is never NULL (INV-7).

### C2 — `ShadowCoordinator` (new module `orchestrator/shadow_eval.py`)

```python
class ShadowCoordinator:
    def __init__(self, config: ShadowEvalConfig, store: RunStore, events: EventStore,
                 runner: ShadowRunner, clock: Callable[[], datetime], seed: int) -> None: ...

    def should_open(self, task: dict, shape: str, candidate: EvalConfig) -> OpenDecision:
        """Pure. Returns OpenDecision(open: bool, reason: str). Reasons are a closed
        vocabulary: sampled_out | no_credentials | isolation_unavailable | budget_exhausted |
        concurrency_full | host_saturated | shape_disabled | already_open."""

    async def on_phase(self, task_id: str, phase: str, entering: bool) -> list[str]:
        """Harness-side hook: called for every phase_enter/phase_exit. Opens cells whose
        trigger matches; returns cell_ids opened. Never raises into the harness."""

    async def on_merge_finalized(self, payload: dict) -> list[str]:
        """Settles awaiting cells for payload['branch']. Returns cell_ids settled."""

    async def expire(self, now: datetime) -> list[str]:
        """Settles every awaiting cell past settle_deadline as expired."""

    def storm_state(self) -> StormState:
        """paused: bool, reason: str | None — read by the harness before each on_phase."""
```

Ordering invariants: `on_phase` for a task is idempotent per `(task, shape, candidate)`;
`on_merge_finalized` and `expire` are the **only** writers of `settled_at`; a cell whose
run fails writes `failed` before any settle can occur. Errors inside a cell never propagate
to the production slot (the coordinator wraps the runner; a wrapped failure is a `failed`
cell with the exception class in `reason`).

### C3 — live fixture builder (`evals/live_fixture.py`)

```python
def build_live_fixture(task: dict, *, base_sha: str, project_root: Path,
                       plan: dict | None, verify_commands: dict[str, str],
                       shape: str, cell_id: str) -> dict
```

Returns a dict accepted unchanged by `runner.load_task`'s consumers: `id =
f"shadow_{task_id}_{cell_id}"`, `pre_task_commit = base_sha`, `task_definition` from the
task record's title/description/details, `plan` as given (production's for `implementer`,
the candidate's for consequence leg 2, `None` for architect legs), **no `reference`** at
build time. Invariant: `build_live_fixture` reads nothing from the live repo beyond the
task record and `plan.json`; the base is the caller's, never "HEAD now".

### C4 — settle and scoring rules

| Shape | `reference_kind=landed` | `reference_kind=none` |
|---|---|---|
| `implementer`, `end-to-end`, consequence leg 2 | verify gates at `base_sha`; judge vs `reference_sha` diff; composite as `compute_composite` | verify gates + review outcome only; `composite_score` left `None`, never 0.0 |
| `architect` | `terminal_kind`; `plan_quality` judged vs reference diff | `terminal_kind`; `score_plan_structure` only; `plan_quality=None` |

Production's paired metrics for the same task are read from `task_results` and the
invocation ledger (`runs.db`), never from logs. A cell settles exactly once.

### C5 — report (`orchestrator eval-shadow report`)

Per `(shape, candidate)`: `pairs`, `unpaired`, `cap_excluded`, `expired`, `failed`; the
shape's primary outcome as paired mean difference with a 95% bootstrap CI; `$ per usable
outcome` for candidate and incumbent; `terminal_kind` split; and `n_min` (contract: a
line whose `pairs < n_min` is printed with an `UNDERPOWERED` tag, not hidden). Machine
form: `--json` emits the same rows.

### C6 — config (`ShadowEvalConfig`, green-tier hot-reloadable, under `shadow_eval:`)

`enabled` (default **false**), `sample_rate` (0..1, per candidate override map),
`shapes` (enabled set), `candidates` (list of `EvalConfig` names), `end_to_end_candidates`,
`max_concurrent` (1), `daily_budget_usd` (50.0, the `auto_eval_redo_budget_usd` pattern),
`settle_deadline_hours` (168), `failure_streak_pause` (5), `cost_ratio_ceiling` (3.0),
`n_min` (12, provisional — see Open questions), `seed`.

## Boundary-test sketch (H) — the integration-gate signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Implementer cell opens on PLAN exit | `enabled`, candidate sampled in, plan.json present, isolation landed | one `shadow_cells` row `opened`; `shadow/<task>/<cell>` worktree at `branch_base_sha`; fixture `plan` byte-equal to plan.json |
| 2 | Sampled-out task opens nothing | hash outside `sample_rate` | no row; one `shadow_cell_skipped(reason=sampled_out)` event |
| 3 | Isolation not landed | 4757 or 3096 absent (probe: invocation kwargs lack `strict_mcp_config=True`) | no cell; `skipped(reason=isolation_unavailable)`; coordinator `storm_state.paused=False` |
| 4 | Production lands | cell `awaiting_reference`; `merge_finalized(state=done, merge_sha=X)` | cell `settled`, `reference_kind=landed`, `reference_sha=X`; result JSON path set; one `shadow_cell_settled` |
| 5 | Production blocks and is cancelled | cell awaiting; `merge_finalized` never fires; task terminal `cancelled` observed | cell `settled`, `reference_kind=none`, `composite_score=None` |
| 6 | Deadline passes | cell awaiting; `now > settle_deadline` | `expire()` → state `expired`, reason `settle_deadline`; never re-opened |
| 7 | Cap hit mid-cell | `UsageGate` reports `cap_hit` for the cell's invocation | state `cap_excluded`; no retry invocation recorded; production task unaffected |
| 8 | Host saturated | `saturated()` true at trigger | `skipped(reason=host_saturated)`; production dispatch proceeds |
| 9 | Consequence leg 2 | leg-1 architect cell produced a plan with `plan_steps>0` | a second cell, `plan_source=candidate`, incumbent implementer config, same `base_sha`; leg 1 and leg 2 rows linked by `cell_id` prefix |
| 10 | Consequence leg 1 declines | leg-1 `terminal_kind` is a decline | **no** leg 2; leg 1 settles with the decline recorded; report counts it in the `terminal_kind` split |
| 11 | Storm escape | five consecutive `failed`/`expired` cells | `storm_state.paused=True`; exactly one escalation filed; subsequent triggers `skipped(reason=paused)` |
| 12 | Report renders from rows only | rows present, result JSON deleted for one cell | report prints the row with `result_missing`, exit non-zero; no log-scrape |
| 13 | Shadow branch never reaches the lane | `shadow/<task>/<cell>` exists in `<root>-eval-worktrees/` (`snapshots.py::create_eval_worktree`), outside `.worktrees/` | no `merge_request` is ever filed for it (the coordinator holds no merge client); the worktree reapers' enumeration is asserted to exclude the eval-worktree root or ε2 adds the exclusion; the assertion is executed, not read from prose |
| 14 | Runner failure is contained | the shape runner raises | cell `failed` with exception class in `reason`; the production slot's `TaskReport` is unchanged |

## Decomposition plan

Leaf sizing follows the overlay bands (300–1,500 LOC, ≤10–12 files). Greek labels; real ids
at decompose. **G7** notes are inline. Every leaf cites `docs/code-quality.md` heuristics
in its brief where they bind (small function scopes and stateless interactions for the
coordinator; SPOT for the reason vocabulary).

**Phase 1 — foundation**

- **α — cell store, config and events** *(leaf)*. `shadow_cells` table and
  `ShadowCell` dataclass in `run_store.py`; `ShadowEvalConfig` in `config.py` (green
  tier, registered with reload); `EventType.shadow_cell_opened/skipped/settled` with the
  closed reason vocabulary as an enum (INV-1: the vocabulary is a schema, not prose).
  **Signal:** `orchestrator check-config` accepts a `shadow_eval:` block; a hermetic test
  inserts a row through `RunStore` and reads it back; `reload_config` reports
  `shadow_eval.*` under `applied`. Modules: `orchestrator`. Prereqs: none.
- **β — live fixture builder** *(leaf)*. `evals/live_fixture.py::build_live_fixture`
  reusing `task_sampler`'s verify-command derivation and stratum classifiers. **Signal:**
  for a synthetic task record + plan.json, the emitted dict round-trips through
  `runner.load_task` and `build_eval_orch_config` without error; the builder is proven
  to read only its arguments (a test passes a temp dir with no git repo). Modules:
  `orchestrator/evals`. Prereqs: none.
- **γ — candidate slate refresh** *(leaf, independent)*. Update `evals/configs.py`
  model ids, base URLs where changed, `CANDIDATE_ENDPOINT_PRICES`, `CODEX_RUST_MODEL`
  (and a second Codex candidate if Leo names two), the reviewer-trial cross-family
  variant, and retire the April `codex-gpt54*` / `gemini-*` entries from `EVAL_CONFIGS`.
  Leo supplies the exact provider strings and prices at filing (open question 1).
  **Signal:** `test_eval_candidate_bundles.py` and `test_eval_codex_pi_bundles.py`
  green against the new constants; `get_config_by_name` resolves every new name;
  `claude_endpoint_price_table()` has an entry for every non-incumbent model.
  Modules: `orchestrator/evals`, `orchestrator/tests`. Prereqs: none.
- **δ — shadow invocation profile** *(leaf)*. `build_shadow_orch_config` =
  `build_eval_orch_config` + `strict_mcp_config=True` + null memory endpoint + shadow
  branch naming; plus the **deterministic isolation probe** the coordinator consults
  (row 3): a check that the invocation kwargs a shadow cell would send carry strict MCP
  and that the eval-lane escalation containment from 3096 is active. **Signal:** a test
  builds the profile and asserts the kwargs; with 4757/3096 absent the probe returns
  `isolation_unavailable`. G7: `guards-exercise-behaviour` — the probe inspects the
  built invocation, not a docstring. Modules: `orchestrator/evals`. Prereqs: α.
  Depends (out-of-batch): **4757**, **3096**.

**Phase 2 — vertical slice (the integration gate)**

- **ε1 — coordinator core** *(leaf)*. `orchestrator/shadow_eval.py`: `should_open`
  (pure), the cell state machine, `expire`, storm state, settle rules (C4) as pure
  functions over `EvalMetrics` and a reference. Hermetic tests cover rows 2, 5, 6, 10,
  11 with an injected clock and a fake runner. G7: `holds-owned-and-bounded` — every
  awaiting cell has `owner` and `settle_deadline`; `no-silent-fail-soft` — every
  non-open is a reason. **Signal:** the state-machine tests; a property test that
  `should_open` is deterministic in its inputs. Modules: `orchestrator`. Prereqs: α.
- **ε2 — harness wiring + implementer shape, end to end** *(integration gate)*. Hook
  `on_phase` into the phase-event emission path in `workflow.py`/`harness.py` (the only
  existing hook is `on_soft_cancel`; this adds a coordinator observer), `on_merge_finalized`
  into the merge-lane landing path, the `shadow/` worktree creation via
  `snapshots.py::create_eval_worktree`, the `implementer` shape through `run_eval`, the
  PSI gate and `max_concurrent` checks, `cap_excluded` on `cap_hit`, and the row-13
  binding: verify (or add) the worktree reapers' exclusion of the eval-worktree root —
  G6 branch 4: a "never reaches the lane" claim must name the mechanism that refuses. **Signal: boundary
  rows 1, 3, 4, 7, 8, 13, 14 executed against a real `TaskWorkflow` in eval mode** (the
  revival's `build_workflow` factory), with the production slot's `TaskReport` asserted
  unchanged. G7: `status-matches-liveness` — a cell whose runner dies writes `failed`
  through the coordinator before the slot returns; `loop-thread-occupancy-bounded` —
  the runner is awaited as a separate task, never inline in the slot. Modules:
  `orchestrator`, `orchestrator/evals`. Prereqs: β, δ, ε1.
- **θ1 — minimal report** *(leaf)*. `eval-shadow report` over the `implementer` shape:
  pairs, paired mean difference with bootstrap CI, `$ per usable`, counts, `UNDERPOWERED`
  tag, `--json`. Row 12. **Signal:** against a seeded `shadow_cells` fixture the report
  prints the contracted columns and exits non-zero on a missing result JSON. Modules:
  `orchestrator/evals`, `orchestrator/cli`. Prereqs: α (rows), ε1 (settle rules).

**Phase 3 — the remaining shapes**

- **ζ — architect and architect-consequence shapes** *(leaf)*. Trigger on
  `phase_enter(PLAN)`; leg 1 through `run_architect_eval` with `terminal_kind`; leg 2
  opened only on `plan_steps>0`, through `run_eval` with `plan_source=candidate` and the
  incumbent implementer; rows 9 and 10. **Signal:** rows 9–10 executed end to end; the
  report shows the consequence shape with leg-2 outcomes paired against production's
  implementer metrics for the same task. Modules: `orchestrator`, `orchestrator/evals`.
  Prereqs: ε2, θ1.
- **η — end-to-end shape** *(leaf)*. Through `run_end_to_end`, gated on
  `end_to_end_candidates`. **Signal:** a cell for a listed candidate runs both roles live
  and settles with `$/done` paired; an unlisted candidate is `skipped(reason=shape_disabled)`.
  Prereqs: ζ.
- **θ2 — full report and the η-gate view** *(leaf)*. All shapes; `terminal_kind`
  split; per-stratum breakdown; the `architect-consequence` view for
  `architect-fable-max` vs `architect-opus-max` that esc-3637-1 consumes. **Signal:** the
  report over a seeded multi-shape store matches a committed golden output; `--json`
  validates against a committed schema. Prereqs: ζ, η.

**Phase 4 — operator gate and companion corrections**

- **κ — first live campaign gate** *(deterministic milestone gate; born-at-L2 for
  Leo)*. Enable `shadow_eval` on df and reify with `candidates=[architect-fable-max,
  <one endpoint bundle>]`, `shapes=[architect-consequence, implementer]`, at the ruled
  `sample_rate`; the gate's predicate is `pairs ≥ n_min` for the fable consequence shape
  in the `--json` report. **Signal:** the escalation filed with the report attached;
  Leo rules esc-3637-1 on it. No config flip. Prereqs: θ2, and 4757/3096 landed.
- **λ — companion corrections** *(leaf, docs)*. Amend `eval-framework-revival-prd.md`
  decision 1 and status; append a pointer note to `fable-architect-trial-v2-prd.md`
  naming this PRD as the redesign; add an `OPERATIONS.md` §"Shadow eval" (config keys,
  the report, the storm escape, how to read `UNDERPOWERED`); mark `tasks_hard_v2` as
  retired-for-capability in `evals/README` or the pool's own header. **Signal:** the
  three documents carry the dated pointers; no restated contract (INV-9). Prereqs: ε2.

## Out of scope

- **Any production routing change.** A favourable report is evidence for a flip task
  under `adaptive-model-routing-prd.md`, filed by Leo's ruling.
- **Fixing `eval-ofat`'s candidate selector** (decision 12) and any further offline
  campaign; the offline runners are kept as the instrument this PRD drives, not as a
  screen.
- **The replay-frame tasks 4758 / 4844.** They concern the offline instrument; this PRD
  neither depends on nor closes them (Leo may re-disposition them once κ runs).
- **vLLM / self-hosted serving**, the LME programme, the memory-eval programme.
- **Dashboard panels** for shadow cells — a follow-up once θ2's `--json` exists.
- **A shadow reviewer or judge shape.** The judge OFAT axis stays offline.

## Open questions (surfaced but not decided in this session)

1. **Exact provider model strings and list prices for γ.** Leo names them at filing
   (Codex Sol / Astra ids, `glm-5.3`, `glm-5.3-flash`, current MiniMax; anything else).
   Until then γ is authored with placeholders that the slate tests reject, so it cannot
   be marked done on stale ids.
2. **`n_min` calibration.** 12 pairs is a screen floor chosen for legibility, not a
   power calculation. **Suggested resolution:** θ2 prints the observed pair variance so
   κ's ruling can state the power it had; recalibrate in a follow-up.
3. **Where the coordinator reads production's paired metrics.** `task_results` vs the
   invocation ledger vs `merge_finalized` payload for cost. **Suggested resolution:**
   the invocation ledger for cost, `task_results` for outcome; decide in ε1.
4. **Settle-deadline default.** 168 h assumes production lands within a week; long
   `merge-deferred` holds may exceed it. **Suggested resolution:** keep 168 h, count
   `expired` in the report, revisit after κ.
5. **Cross-project report aggregation.** Each orchestrator writes its own runs.db;
   `eval-shadow report --project-root` per project vs a merged view. Decide in θ1.
6. **Consequence leg 2 reviewer.** Whether leg 2 runs the reviewer (adds cost, gives
   `review_blocking_issues`) or stops at verify. **Suggested resolution:** run it —
   review outcome is half the signal for "better plan". Decide in ζ.
