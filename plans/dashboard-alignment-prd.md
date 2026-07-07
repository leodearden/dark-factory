# PRD: Dashboard alignment — stop silent drift from orchestrator internals

**Status**: active — authored 2026-07-06 (stream M3 of the bug-hotspot remediation
program, `plans/bug-hotspot-remediation-program-2026-07-06.md`).
**Mode**: bare B (per program brief; boundary test on the one owned seam).
**Survey findings**: dashboard cluster of
`plans/bug-hotspot-survey-2026-07-06-full-findings.json`.

## Goal

The dashboard stops silently drifting from orchestrator internals in four ways:

1. **Merge-outcome vocabulary** becomes an owned, enumerated contract
   (`OutcomeKind` StrEnum in orchestrator `merge_types.py`, M3-owned per the
   program G4 seam table) and the dashboard consumes it **fail-safe**: the
   active-merges classifier is inverted to *terminal-unless-listed* with an
   explicit ACTIVE_ONLY allowlist, so newly added terminal outcomes fail safe
   (drop off the active panel) instead of failing stale (30-minute phantom
   "in flight" rows — the bug that exists on main today for ~11 outcome
   strings added after commit `8553b388ad` froze the dashboard list).
2. **Request-scoped `now`** is threaded through the burndown and costs
   aggregators exactly as `dashboard/data/merge_queue.py` already does (the
   task-692 `effective_now` pattern), via a shared `resolve_now` helper, with a
   grep-guard test forbidding bare `datetime.now(UTC)` inside
   `dashboard/src/dashboard/data/` so the fixed-once-latent-in-siblings class
   cannot recur.
3. **MCP fan-out-with-failover and TTL caching** — currently reimplemented
   independently in 5 modules (memory.py `_first_success`, tasks.py ×3 loops +
   `_fetch_tasks_cache`, scheduler.py `_one_project` + `_scheduler_cache`,
   metrics.py, app.py ×2 loops + `_task_cards_cache`, merge_queue.py
   `_task_titles_cache`) — are extracted into one `dashboard/data/mcp_fanout.py`
   helper module and the copies converted.
4. **Re-derived orchestrator formats** (ps-scan launch patterns, `.task/`
   artifact layout in `dashboard/data/orchestrator.py`) get an explicit
   format-coupling contract comment naming each orchestrator source of truth.
   No import unification: the dashboard package has **no dependency on the
   orchestrator package** (verified: `dashboard/pyproject.toml` depends only on
   `escalation` + `dark-factory-shared`), the orchestrator side's `run`
   subcommand name is a click-decorator literal (not cheap to constant-ify),
   and `.task/` path derivation is owned by stream **W11** (`TaskArtifacts`) —
   M3 must not create a competing owner. Per the brief: leave and document; do
   not build a new API layer.

User-observable surface: the dashboard merge-queue panel (active list + outcome
doughnut), burndown panel, and costs panel.

## Background

- Survey findings (dashboard cluster): outcome-vocabulary drift; per-DB
  clock-skew race fixed once (task 692 / `0020b1d0ac`) but latent in sibling
  aggregators; MCP fan-out + TTL caching duplicated 4-5×; re-derived
  process-launch/file-artifact formats.
- Program resolved decision #5 (authoritative, do not relitigate):
  `OutcomeKind` is owned by orchestrator `merge_types.py`; dashboard consumes
  it fail-safe; if a hard import is undesirable, invert to
  terminal-unless-listed with an explicit ACTIVE_ONLY allowlist.
- Verified 2026-07-06 against main (`e19aeea088`): all file:line references
  below re-checked.

## The outcome vocabulary (enumerated from code, 2026-07-06)

All 21 strings passed to `_emit_merge_attempt` today (source of truth for the
`OutcomeKind` member set; implementer must re-enumerate at implementation time
— main moves fast):

| Member | Emit site(s) | terminal? |
|---|---|---|
| `done` | merge_gates.py:644 | yes |
| `already_merged` | merge_queue.py:1466, 2552 | yes |
| `unknown_branch` | merge_queue.py:1507 | yes |
| `conflict` | merge_queue.py:2580, 2890 | yes |
| `merge_failed` | merge_queue.py:2890 | yes |
| `verify_failed` | merge_queue.py:2958 | yes |
| `advance_failed` | merge_queue.py:2974 | yes |
| `dropped_plan_targets` | merge_queue.py:2638 | yes |
| `abandoned_verify_timeouts` | merge_queue.py:2819, 7271 | yes |
| `train_incomplete` | merge_queue.py:2853 | yes |
| `train_rebase_conflict` | merge_queue.py:2871 | yes |
| `train_partial_flip` | merge_queue.py:3092 | yes |
| `cas_exhausted` | merge_queue.py:8911, 8984 | yes |
| `main_health_red` | merge_queue.py:749 | yes |
| `post_merge_equivalence_failed` | merge_gates.py:387 (GateVerdict.emit_subtype) | yes |
| `post_merge_pyright_broken` | merge_gates.py:462 (GateVerdict.emit_subtype) | yes |
| `plan_files_not_touched` | workflow.py:5364 | yes |
| `plan_files_narrowed` | workflow.py:5376 | **no** — mid-submission waypoint; enqueue → `merge_queued` follows in the same flow |
| `cas_retry` | merge_queue.py:9004 | **no** — attempt continues |
| `gate_retry` | merge_queue.py:8941 | **no** — attempt continues |
| `post_merge_generation_chained` | merge_gates.py:421 | **no** — gen-(n+1) request already enqueued (`_maybe_auto_chain_generation` calls `enqueue_merge_request` *before* this emit, so this is the latest event for a task whose merge is still live) |

Non-terminal set (mirrored by the dashboard ACTIVE_ONLY allowlist):
`{cas_retry, gate_retry, post_merge_generation_chained, plan_files_narrowed}`.
The brief named `cas_retry`/`gate_retry`; the two additions come from tracing
the emit sites (above) — both denote a merge lifecycle that is still live.

`blocked` is deliberately NOT a member: it appears in the old dashboard list
and in historical event rows, but `_emit_merge_attempt`'s own docstring
documents that bare infrastructure `blocked` outcomes are intentionally not
emitted; only specific diagnostic codes are. Historical `blocked` rows
classify as terminal under inversion — correct.

## Sketch of approach

### 1. `OutcomeKind` (orchestrator, M3-owned seam)

- `class OutcomeKind(StrEnum)` in `orchestrator/src/orchestrator/merge_types.py`
  with the 21 members above and an `is_terminal` property
  (`self not in _NON_TERMINAL_OUTCOMES`).
- `_emit_merge_attempt(..., outcome: OutcomeKind, ...)` — signature narrowed
  from `str`. StrEnum members serialize identically (`str(OutcomeKind.DONE) ==
  'done'`; `json.dumps` via the emit path is byte-identical), so event-store
  rows and every downstream reader are unaffected. A test asserts payload
  string identity.
- All call sites converted from string literals to members
  (merge_queue.py, merge_gates.py — including `GateVerdict.emit_subtype:
  OutcomeKind | None` — and workflow.py's two sites).
- Boundary test (the G5 two-way test for this seam): an AST-based test walks
  merge_queue.py / merge_gates.py / workflow.py, finds every
  `_emit_merge_attempt(` call, and asserts the outcome argument is an
  `OutcomeKind` attribute access (or a conditional expression over them) —
  never a bare string. Pyright enforces the same at type-check (pre-commit).
- A frozen-contract test asserts the exact non-terminal member set, with a
  comment stating that changing it requires updating the dashboard
  ACTIVE_ONLY allowlist (named by path) in the same change.

### 2. Fail-safe dashboard classification (inversion)

In `dashboard/src/dashboard/data/merge_queue.py`:

- Delete `_TERMINAL_MERGE_OUTCOMES` (line 53) and `_CANONICAL_OUTCOMES`
  (line 51) — the two divergent hardcoded lists.
- `active_queued_merges` inverts: a latest `merge_attempt` event is
  **terminal unless** its outcome is in
  `_ACTIVE_ONLY: frozenset = {'cas_retry', 'gate_retry',
  'post_merge_generation_chained', 'plan_files_narrowed'}`, documented as a
  mirror of `orchestrator merge_types.OutcomeKind` non-terminal members (no
  import — the dashboard package has no orchestrator dependency; the
  orchestrator-side frozen-contract test is the drift tripwire).
- `outcome_distribution` loses `_CANONICAL_OUTCOMES` ordering: order slices by
  count descending, alphabetical tiebreak (purely presentational; the doughnut
  colors already degrade gracefully via `outcome_colors.py`'s unknown ramp).

### 3. Request-scoped `now` (the task-692 pattern, generalized)

- `resolve_now(now: datetime | None) -> datetime` in
  `dashboard/src/dashboard/data/utils.py` (returns `now` or
  `datetime.now(UTC)`).
- costs.py: `_cutoff(days, *, now=None)`; `now: datetime | None = None`
  keyword threaded through every `get_cost_*` and `aggregate_cost_*`; app.py's
  costs route (app.py:714-719) captures `now` once and passes it to all six
  aggregates.
- burndown.py: `get_burndown_series` / `aggregate_burndown_series` /
  `aggregate_window_completion` gain `now=`; app.py:1301 captures once.
  Writer paths (`collect_snapshot`, `downsample`) capture once per call
  already — they get an exemption tag, not threading.
- Existing `now if now is not None else datetime.now(UTC)` expressions in
  dashboard merge_queue.py migrate to `resolve_now`.
- Grep-guard test (new `dashboard/tests/test_clock_discipline.py`): scans
  `dashboard/src/dashboard/data/*.py` for `datetime.now(` occurrences; allowed
  only in `utils.py`'s `resolve_now` definition or on lines carrying an
  explicit `# clock-exempt:` justification tag. This is the CI check the brief
  asks for, expressed as a pytest (runs in the same merge verify).
- **Grandfathered pre-existing debt** (added 2026-07-07, /unblock 2192 —
  option C). The full-tree scan is retained on purpose — it protects *all* data
  modules against the next latent-in-siblings clock read; narrowing it would
  recreate exactly the 692→726 blind spot. But 24 pre-existing bare
  `datetime.now(` CODE reads live in 7 modules this PRD never converts
  (metrics.py, scheduler.py, reconciliation.py, write_journal.py,
  active_tasks.py, redux_api.py, cap_history.py) — and **none is a cross-DB
  fan-out aggregator** (so none is the task-692 skew shape; they are writers /
  single-DB cutoffs / age reads / fallbacks). δ tags each with a **distinct**
  debt marker `# clock-exempt: deferred-consolidation (task 2281)` — still
  inside the guard's `# clock-exempt:` allow-list, but semantically separate
  from a correctness exemption (`# clock-exempt: single-capture writer`): it
  asserts *tracked debt, not safety*, and names the owning task. Task **2281**
  owns the real conversion (deps 2192, 2218 — runs after ε2 settles the fan-out
  structure) and is done when
  `grep -rn 'deferred-consolidation' dashboard/src/dashboard/data/` is empty
  with the guard still green. This keeps the acceptance criterion greenable in
  δ's scope without dragging repo-wide consolidation (explicitly deferred by
  γ's `resolve_now` docstring) into δ.

### 4. MCP fan-out + TTL cache helper

New module `dashboard/src/dashboard/data/mcp_fanout.py`:

- `async def first_success(client, urls, call, *, log_label, offline_result)` —
  sequential first-success failover over `urls`; `call(url)` is an async
  callable (supports both the single-tool shape and scheduler.py's
  paired-calls-per-URL shape); on the transport exception set
  (`httpx.ConnectError/TimeoutException/HTTPStatusError`, `ValueError`) it
  logs at debug, `invalidate_session(url)`, and continues; all-fail returns
  `offline_result`.
- `class TTLCache` — monotonic-clock TTL cache with per-key async refresh
  lock (generalizing scheduler.py's `_scheduler_cache` + lock discipline) and
  a `clear()` test hook.

Conversions (behavior-preserving): memory.py `_first_success` + singleton
getters; tasks.py's three URL loops + `_fetch_tasks_cache`; scheduler.py
`_one_project` + `_scheduler_cache`; metrics.py:105 loop; app.py:820/1043
loops + `_task_cards_cache`; dashboard merge_queue.py `_task_titles_cache`.
NOT converted: aggregate-across-all-URLs loops (`get_queue_stats`,
`get_wal_status`) and merge_halt.py's concurrent probe-all — different idiom
(sum/probe every URL, not failover), documented as such.

### 5. Format-coupling contract (document-only)

`dashboard/src/dashboard/data/orchestrator.py` gets a FORMAT COUPLING doc
block naming, for each re-derived format, the orchestrator source of truth:
ps-scan launch patterns (`orchestrator run --prd/--config`) →
`orchestrator/src/orchestrator/cli.py` `run` command; `.task/` layout
(`metadata.json`, `plan.json` steps/files, `iterations.jsonl`,
`reviews/*.json` verdicts) → `orchestrator/src/orchestrator/artifacts.py`,
future single owner: stream W11's `TaskArtifacts` (which also plans to
relocate `.task/` out of the git tree — this doc block is what W11 greps for
to find the dashboard reader).

## Resolved design decisions

1. **Enum home + consumption mode**: program decision #5 verbatim. Hard import
   confirmed unavailable (no dashboard→orchestrator dependency), so the
   inversion path is used; adding an orchestrator dependency to the dashboard
   is rejected (couples deploy units for one frozenset).
2. **ACTIVE_ONLY membership**: exactly the `is_terminal == False` members —
   4, not the brief's 2; rationale traced per-site in the vocabulary table.
3. **Alignment guard across the package boundary**: orchestrator-side
   frozen-contract test on the non-terminal set + reciprocal path-naming
   comments on both sides. A cross-package import in tests is rejected
   (separate uv projects/venvs).
4. **`blocked` excluded from the enum** (not emitted today; historical rows
   fail safe under inversion).
5. **`outcome_distribution` ordering** becomes count-descending (was:
   canonical-first). Presentational only.
6. **Grep-guard as pytest**, not a lint plugin or pre-commit hook — runs in
   merge verify with zero infra; `# clock-exempt:` tag for justified sites.
7. **Item 4 is document-only**: no shared constants module for launch
   markers (click-decorator literal on the producer side makes the import
   non-cheap; a shared constant consumed only by the dashboard is a third
   copy, not an alignment).
8. **`resolve_now` lives in `dashboard/data/utils.py`**, not
   `shared.timestamps` — it is dashboard-request-scoping, not a cross-process
   contract; no second consumer exists.
9. **Full-tree guard + grandfathered debt** (2026-07-07, /unblock 2192): the
   grep-guard keeps its full `data/*.py` scan (retro-narrowing it would gut its
   latent-in-siblings purpose). Pre-existing bare reads in the 7 non-converted
   modules are grandfathered with a distinct `# clock-exempt:
   deferred-consolidation (task 2281)` marker (tracked debt ≠ correctness
   exemption); task 2281 owns the real conversion after ε2. Rejected: (A)
   narrowing the scan (recreates the blind spot the guard exists to close),
   (B) expanding δ to convert all 7 now (scope creep + δ↔ε2 tangle in
   scheduler/metrics; contradicts γ's documented repo-wide-consolidation
   deferral).

## Pre-conditions

None. All substrate exists on main (verified 2026-07-06): `_emit_merge_attempt`
and all 21 call-site strings; `merge_types.py` already hosts StrEnums
(`InflightStatus`); dashboard merge_queue.py's `now=`-threading exemplar;
`config.fused_memory_urls`; `invalidate_session`; all named caches and loops.
No novel substrate — G3 verified by direct grep (see capability manifest).

## Cross-PRD relationship (G4)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W1 (merge-queue-reliability) | produces → W1 consumes | `OutcomeKind` in `merge_types.py` | **M3 (this PRD)** — W1 must not introduce a competing outcome enum (program seam table) | queued (task α) |
| W1 | file-contention only | α edits emit call sites inside merge_queue.py / merge_gates.py which W1 refactors | n/a — additive literal→member conversion; scheduler module locks serialize | noted |
| W11 (worktree-lane-lifecycle) | consumes ← W11 will own | `.task/` path derivation (`TaskArtifacts`) | **W11** — M3 documents the dashboard reader, moves no derivation logic | queued (task ζ, doc-only) |
| M2 / W10 | none | no shared files | — | — |

## Decomposition plan

Labels α…ζ; deps in parentheses. Signals are the G2 user-observable signals.

- **α — Introduce OutcomeKind in merge_types.py and type the emit chokepoint** (high)
  Modules: orchestrator. Files: `orchestrator/src/orchestrator/merge_types.py`,
  `orchestrator/src/orchestrator/merge_queue.py`,
  `orchestrator/src/orchestrator/merge_gates.py`,
  `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/tests/test_outcome_kind.py` (new).
  Signal: `merge_attempt` event rows emitted via `OutcomeKind` members are
  byte-identical to today's string payloads (test asserts
  `data['outcome'] == 'done'` etc. through a real EventStore emit); the AST
  boundary test proves every `_emit_merge_attempt` call site passes an
  `OutcomeKind` member; the frozen-contract test pins the non-terminal set.
  Consumer: β's ACTIVE_ONLY mirror + W1 (program seam table) + every existing
  event-store reader (unchanged payloads).

- **β — Invert dashboard active-merge classification to terminal-unless-listed** (high, deps: α)
  Files: `dashboard/src/dashboard/data/merge_queue.py`,
  `dashboard/tests/test_merge_queue_data.py`.
  Signal: with an event fixture whose latest merge_attempt outcome is
  `verify_failed` (or ANY string absent from ACTIVE_ONLY, including a made-up
  future outcome `frobnicated`), `active_queued_merges` excludes the task —
  the merge-queue panel no longer shows the 30-minute phantom; `cas_retry` /
  `gate_retry` / `post_merge_generation_chained` fixtures remain listed as
  `in_flight`. `outcome_distribution` orders by count descending.
  Consumer: merge-queue dashboard panel (user surface).

- **γ — resolve_now helper + request-scoped now through costs aggregators** (medium)
  Files: `dashboard/src/dashboard/data/utils.py`,
  `dashboard/src/dashboard/data/costs.py`, `dashboard/src/dashboard/app.py`,
  `dashboard/tests/test_costs_data.py`, `dashboard/tests/test_data_utils.py`.
  Signal: a test drives `aggregate_cost_summary`/`aggregate_cost_trend` over
  two DBs with an injected `now` and asserts both per-DB queries used the
  identical cutoff (deterministic window boundaries — the task-692 skew class
  is closed for costs); app.py passes one captured `now` to all six cost
  aggregates.
  Consumer: costs dashboard panel; δ (helper + pattern).

- **δ — now-threading for burndown + clock-discipline grep-guard** (medium, deps: γ, β)
  Files: `dashboard/src/dashboard/data/burndown.py`,
  `dashboard/src/dashboard/data/merge_queue.py`,
  `dashboard/src/dashboard/app.py`, `dashboard/tests/test_burndown_data.py`,
  `dashboard/tests/test_clock_discipline.py` (new), plus a mechanical
  comment-only tag pass over the 7 grandfathered modules (metrics.py,
  scheduler.py, reconciliation.py, write_journal.py, active_tasks.py,
  redux_api.py, cap_history.py — see §3 "Grandfathered pre-existing debt";
  behavior-preserving, tags only).
  Signal: `aggregate_burndown_series` over two DBs with injected `now` uses one
  shared cutoff; the guard test FAILS if a bare `datetime.now(` is introduced
  anywhere in `dashboard/src/dashboard/data/*.py` outside `resolve_now` /
  `# clock-exempt:`-tagged lines (verified by the test's own negative fixture),
  and PASSES on the converted-and-grandfather-tagged tree.
  Consumer: burndown dashboard panel; the guard protects all data modules.

- **ε1 — Extract mcp_fanout helper (first_success + TTLCache); convert memory.py + tasks.py** (medium)
  Files: `dashboard/src/dashboard/data/mcp_fanout.py` (new),
  `dashboard/src/dashboard/data/memory.py`,
  `dashboard/src/dashboard/data/tasks.py`,
  `dashboard/tests/test_mcp_fanout.py` (new),
  `dashboard/tests/test_memory.py`.
  Signal: existing memory/tasks panel tests stay green with the loops deleted;
  new helper tests prove failover order, session invalidation on transport
  error, all-fail offline sentinel, and TTL refresh-lock single-flight;
  `grep -c 'for url in config.fused_memory_urls' memory.py tasks.py` drops to 0.
  Consumer: ε2 (remaining conversions); memory + tasks panels.

- **ε2 — Convert scheduler/metrics/app/merge_queue onto mcp_fanout** (medium, deps: ε1, δ)
  Files: `dashboard/src/dashboard/data/scheduler.py`,
  `dashboard/src/dashboard/data/metrics.py`,
  `dashboard/src/dashboard/app.py`,
  `dashboard/src/dashboard/data/merge_queue.py`,
  `dashboard/tests/test_scheduler_page.py`,
  `dashboard/tests/test_metrics_data.py`.
  Signal: scheduler/metrics/app panels stay green on the shared helper;
  repo-wide `for url in config.fused_memory_urls` survives only inside
  mcp_fanout.py and the documented aggregate-across-all loops
  (`get_queue_stats`/`get_wal_status`); the four ad-hoc TTL caches
  (`_scheduler_cache`, `_fetch_tasks_cache`, `_task_titles_cache`,
  `_task_cards_cache`) are TTLCache instances with one shared implementation.
  Consumer: scheduler/metrics dashboard panels.

- **ζ — Format-coupling contract doc in dashboard/data/orchestrator.py** (medium, complexity=simple)
  Files: `dashboard/src/dashboard/data/orchestrator.py`.
  Signal: the FORMAT COUPLING doc block exists on main naming each re-derived
  format and its orchestrator source-of-truth file (cli.py `run`,
  artifacts.py) and the W11 `TaskArtifacts` future ownership — the block W11
  greps for when relocating `.task/`.
  Consumer: stream W11 (documented hand-off point) + operators reading the
  module. Document-only by explicit brief instruction ("otherwise leave and
  document — do not build a new API layer").

Dependency edges: β→α, δ→γ, δ→β, ε2→ε1, ε2→δ. No out-of-batch or
cross-project deps.

## Out of scope

- Any merge-queue behavior change (W1 owns merge_queue.py internals; α is an
  additive enum + literal→member conversion at emit sites only).
- Frontend/JS work.
- `outcome_colors.py` rework — already fail-safe by design (unknown codes get
  the neutral ramp); its `_CODE_FAMILY` superset is documented as
  non-authoritative in-file.
- Converting merge_halt.py / `get_queue_stats` / `get_wal_status`
  (probe-all/sum-all idioms, not failover).
- Moving `.task/` path derivation or building a dashboard API layer (W11).
- A `shared/` constants module for launch markers (rejected, decision 7).

## Open questions (tactical)

1. **Does `emit_subtype` stay `str | None` at the `GateVerdict` layer or
   become `OutcomeKind | None`?** Suggested: `OutcomeKind | None` (pyright
   then covers the gate path too). Decide during α.
2. **TTLCache key typing** (str keys everywhere today vs Generic). Suggested:
   plain `str` keys, values typed per-instance via Generic[V]. Decide during ε1.
3. **Whether `aggregate_window_completion` needs `now=`** (it consumes
   already-fetched series; likely no DB clock read). Verify during δ; thread
   only if it reads the clock.
