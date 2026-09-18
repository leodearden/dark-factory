# PRD: dashboard-one-datum-one-path — every dashboard number has one access path and carries its own provenance

**Status:** active — authored 2026-09-18. **Approach B + H** (contract + two-way
boundary tests): blast radius is `dashboard/` server + SPA + `shared/` + `scripts/`,
the load-bearing seam is the server→SPA wire contract, and the mechanism count is
well past eight. **Type:** greenfield discipline PRD that also resolves the census
seam the sibling dashboard PRDs left unowned.

**Code anchors** are cited by symbol (`path::symbol`), never by line. Main moves
fast — re-locate at implementation time.

## Goal

The dashboard never shows two different answers to one question, and never shows a
confident number for a fact it did not measure.

User-observable end state:

- Orchestrators › any project: the Progress card, its pips, the Active/Pending/Complete
  filter bar, the four top tiles, the Overview "Active tasks" tile, the topbar
  "tasks active" chip and the rail "Tasks" badge all show numbers derived from **one**
  per-project census taken at **one** instant, each labelled with the **named view**
  it shows (`running`, `in-flight`, `backlog`, `terminal`). "0/1" cannot occur.
- When a project's census could not be measured, every one of those surfaces shows
  the same thing: the last known value with its age (`stale · 3h`) or an em-dash with
  the reason (`unknown · budget expired`). Never `0`.
- Burndown's aggregate "Backlog" equals the sum of the per-project rows beneath it;
  a project whose sampler cycle failed shows a gap and an age, not a stale row served
  as current.
- Every stat tile's sparkline is the history of the same datum its headline shows.
- The window chip is honoured or visibly declined on every windowed panel; no panel
  header says "30d" while serving 7d.
- Merge, escalation and memory headline pairs that today disagree (167 vs 246 merges,
  38 vs 252 pending escalations, 319 011 vs 320 670 ops) are one query each, with any
  narrower view labelled by its filter.

## Background

`plans/dashboard-contradiction-sweep-2026-09-14.md` (team sweep, 46 findings, 39
confirmed) established seven root causes. The three that account for the reported
instance and most findings:

- **RC1** three independent censuses of one task tree (`orchestrator.py::discover_orchestrators`
  whole-tree fetch; `active_tasks.py::collect_tasks_with_counts` narrowed fetch +
  status map with two caches; `burndown.py::collect_snapshot` whole-tree sampler),
  with no shared cache, budget, instant or failure encoding.
- **RC2** `shared/src/shared/task_statuses.py::TaskStatus` (nine members) is imported by
  no dashboard module; at least seven hand-rolled bucketers disagree.
- **RC3** UNKNOWN has no wire representation; the orchestrators entry's `offline`
  flag is read by no JSX; `tabs.jsx::OrchTab` renders `total || 1`.

Plus **RC4** tile value and spark from unrelated provenance, **RC5** the burndown
aggregate zero-fills ragged snapshots, **RC6** window semantics not threaded, **RC7**
two queries per headline.

**Measured 2026-09-18** (dashboard's own `tasks.py::fetch_statuses` /
`tasks.py::fetch_tasks`, caches cleared, two runs each): the compact status map for
dark-factory (5539 tasks) returns in 0.11 s and for reify (7623) in 0.07 s; the
active-row fetch in 0.8–1.5 s. Only the whole-tree fetches time out. The "census
can't complete" premise was self-inflicted by the two whole-tree fetch families this
PRD removes.

**Ruled by Leo 2026-09-18**: treat divergence as a smell; for each datum, move every
access onto one shared access implementation and branch the datapath as far up the
stack as possible; provenance (B) is a consequence of that, not an add-on, because a
single path that fails silently turns contradictions into uniform lies.

Prior slice fixes (3516, 1564, 1601, 1814, 3517, 3857) each hardened one surface;
this PRD is the class fix. Task **4795** (deferred: tasks budget cannot cover nine
roots) is superseded by the acquisition change and its acceptance is inherited by ι.
Task **5320** (done 2026-09-17) set the vocabulary-module precedent for recon status.

## Sketch of approach — the discipline

1. **A datum is a named fact with one access implementation.** `dashboard/data/` gets
   one module per datum family; nothing else may query that fact's source. Consumers
   receive the datum and derive views; they never re-count, re-bucket or re-fetch.
2. **Every datum is a `Datum` envelope on the wire**: `{value, as_of, state, reason,
   age_seconds}` with `state ∈ {fresh, stale, unknown, lower_bound}`. The envelope is a
   declared schema (INV-1), produced only by the access implementation, and the shared
   rendering components refuse to paint a bare number for `unknown`.
3. **The task census** is one datum: a nine-member count vector over
   `shared.task_statuses.TaskStatus`, its total, and four named views, built from the
   compact status map, snapshotted together with the active rows under one `as_of`,
   carried in `/api/v2/dashboard/tasks`. `/api/v2/dashboard/orchestrators` becomes
   process discovery only. The burndown sampler persists this same datum.
4. **Views are defined once, in Python**, and shipped; the SPA gets the vocabulary as a
   file **generated** from `shared.task_statuses` with a parity test (INV-5), so no JS
   ever buckets a status string for counting.
5. **A stat tile takes a datum**, and its spark is that datum's persisted history, so
   the headline and the trend line agree at the last point by construction.
6. **A sampler persists a datum**, writes an explicit gap row on failure, and the
   aggregate carries each project's last value forward with its age rather than
   zero-filling absence.
7. **The window is a datum too**: every windowed payload echoes `{requested, served,
   days}`; labels render from it; the chip is validated against the destination tab.
8. **Old paths are deleted, not deprecated**, behind grep-guards paired with executed
   boundary tests (INV-10).

## Resolved design decisions

1. **Scope: all seven root causes in one PRD** (Leo, 2026-09-18). The discipline is
   established once and applied to every datum family the sweep found, so the
   one-surface-hardened pattern does not recur.
2. **Census wire home: inside `/tasks`, one snapshot with the rows.** Counts and rows
   share one `as_of` per project. `/orchestrators` drops `summary`; the burndown
   endpoint is the persisted history of the same datum. Rejected: a separate
   `/census` endpoint (counts and rows from different poll cycles again) and
   embedding in both endpoints (two wire copies — RC1's shape).
3. **Named views (the only composites any surface may show)**: `running` =
   in-progress; `in-flight` = in-progress + blocked + merge-deferred + review;
   `backlog` = pending + deferred + infra-hold; `terminal` = done + cancelled. Disjoint
   by construction, summing to `total`. Every surface labels the view it shows. The
   cap-comparable question (task 3516) is `running`; the PRD-finished question
   (`prd_grouping.js`) is `terminal` vs the rest. "Active" as a label is retired.
4. **Vocabulary reaches JS by generation**: `scripts/gen_dashboard_task_vocab.py`
   renders `dashboard/src/dashboard/static/redux/task_vocab.js` (members, view
   membership, display tones) from `shared.task_statuses`; a test regenerates and
   diffs. Rejected: the hand-maintained-twin pattern of `recon_status.js` (two copies)
   — retained there because it is landed and carries its own runtime `unknown`
   bucket; and views-only (per-status pips and colours would move server-side for no
   gain).
5. **Census source is the compact status map**, not the tree. The live/stranded split
   of `in-progress` (task 3543, parity alarm) needs claimant columns the map lacks
   (memory `5cccec6c`), so it is derived from the active rows in the same snapshot via
   `shared.task_claimant.is_stranded`. The terminal-row window (`_MAX_DONE_PER_PROJECT`)
   stays for the row listing only; **no count is ever derived from a capped row set**
   except the PRD box's `n/m` (decision 8).
6. **Datum states**: `fresh` (measured within the datum's freshness bound); `stale`
   (last good value served, `age_seconds` set, reason = why the refresh failed);
   `unknown` (no value ever measured or the last value is older than the datum's
   retention bound; reason set); `lower_bound` (a tally over a disclosed window).
   Freshness bounds are per datum family, declared beside the access implementation,
   default 2× the acquisition cadence. The access implementation holds the last good
   value; the SPA never synthesises a state.
7. **Rendering is uniform and lives in the shared components only**: `charts.jsx::StatTile`,
   `tabs.jsx::ST`, `shell.jsx::ProjectGroup` pips and the row cells accept a datum.
   `unknown` → em-dash with the reason as title; `stale` → value plus an age badge;
   `lower_bound` → `≥` prefix. Tab code never branches on state.
8. **PRD box `n/m done` is a `lower_bound` datum** over the fetched rows, rendered as
   `≥ n/m`, until a per-PRD server census exists (out of scope: the status map carries
   no PRD field).
9. **Burndown**: the sampler consumes the same snapshot builder as `/tasks` (map + active
   rows); a failed cycle writes a row with `state='gap'` (new nullable column via
   `burndown.py::ensure_snapshot_columns`); all nine members are stored and shipped
   (`_ZONE_KEYS`/`_BURNDOWN_KEYS`/`burndown_bands.js` collapse to the generated
   vocabulary); `redux_api.py::shape_burndown` carries each project's last value
   forward with its age when building the aggregate and emits per-project `as_of`,
   `age_seconds`, `state`. The client-side `shell.jsx::dailyDeltas` is deleted; the
   OrchTab "Completed / day" spark consumes the server's per-project `completed`
   series. Retires `test_shape_burndown_ragged_input_is_passed_through_unnormalized`
   (it pins the zero-fill).
10. **Window**: `app.py::_parse_window` and the burndown parser return `{requested,
    served, days}` and every windowed payload echoes it; `app.jsx` re-validates the
    chip against the destination tab's set on tab switch and resets to that tab's
    default when invalid; hardcoded "30d" labels render from the payload.
    **Recent merges follows the chip**, newest N within the window, labelled
    "showing N of M in <window>" (Leo, 2026-09-18) — retires the `test_app.py` pin
    on `recent_window_minutes=1440`. **Performance anchors to wall-clock** with an
    idle state: a project with no completions in the window renders its cards `stale`
    with last-completion age (Leo, 2026-09-18) — `performance.py::_project_cutoffs`
    goes. `model_role.py::_cutoff` adopts the digest's bound semantics.
11. **Merge attempts are one datum**: one query over `merge_attempt` events in the
    window feeds both the outcome distribution and the latency stats; the latency
    tile is labelled "of N with recorded duration". "In queue now" is a live datum
    whose history is `merge_snapshots`; when the live probe fails the tile is `stale`
    with the last sample's age, never the events-table approximation shown as live.
12. **Escalations are one corpus walk** (root + archive via
    `escalation.queue.iter_all_escalation_paths`) with a `location` field, one cache,
    consumed by both `escalations.py` and `escalation_analytics.py`; "pending in live
    queue" and "open in history" are views; the tab shows the datum's age (Leo,
    2026-09-18). The benign/actionable split enumerates every member of
    `escalation.models.RESOLUTION_CLASSES` so `stale-strand` and
    `moot-terminal-subject` are visible.
13. **Memory ops are one query** grouped by both `kind` and `operation`; reads/writes
    and the breakdown donut derive from it. `redux_api.py::shape_memory` normalises the
    queue block in one place; `queue.offline` becomes the datum state and is rendered.
14. **Locks**: `scheduler_utils.jsx::lockChipState` is the single classifier;
    `scheduler_heatmap.jsx::cellStateFor` calls it; a project in the scheduler
    snapshot's `offline_projects` yields an `unknown` Locks datum, not an empty list.
15. **Dead second implementations are deleted**: `active_tasks.py::collect_done_counts`,
    `task_status_counts.js` (its two questions become view lookups), the bucketers in
    `prd_grouping.js`, `orchestrator.py`'s summary, `app.jsx`'s two reductions.
16. **Guards**: a pytest grep-guard (the `dashboard-alignment-prd.md` clock-guard
    pattern) forbids in `static/redux/`: `|| 1`, `?? 0`, `|| []` on datum fields;
    status-literal comparisons for counting outside the generated vocabulary; and in
    `data/`: a second producer of any datum listed in the registry. Each guard is
    paired with an executed boundary test (INV-10) — the grep is the lockstep
    backstop, not the proof.
17. **`app.py` is already past the size alarm** (`docs/code-quality.md` heuristic 14).
    No leaf may grow it; a leaf that must change a handler extracts that handler
    into a module that passes heuristic 13. `tabs.jsx` is at the soft ceiling; leaves
    touching it prefer deletion over addition.
18. **No dashboard→orchestrator import** (dashboard-alignment decision 1 stands).
    `shared` is already a runtime dependency and is the only cross-package import
    this PRD adds.

## Verified substrate (G3)

| Capability | Evidence |
|---|---|
| `shared.task_statuses.TaskStatus`, `ACTIVE`, `TERMINAL` | read 2026-09-18; nine members; `ACTIVE = frozenset(TaskStatus) - TERMINAL` |
| dashboard imports `shared` at runtime | `dashboard/pyproject.toml` depends on `dark-factory-shared`; timing script imported it under `uv run --project dashboard` |
| fused-memory `get_statuses` paging, `page_size ≤ 2000`, `has_more` | called live: dark-factory 5539 statuses in three pages |
| `tasks.py::fetch_statuses` cost | measured 0.07–0.12 s on 5.5k–7.6k trees |
| `tasks.py::fetch_tasks(statuses=…)` is a real server-side filter | memory `fa730a43`; measured ~1 s for ACTIVE on both big trees |
| `shared.task_claimant.is_stranded` | present; consumed by `tasks.py::task_is_stranded` |
| `burndown.py::ensure_snapshot_columns` probe-then-add migration | present; used for the live/stranded columns |
| `metrics.py` sampler tables `merge_snapshots`, `orchestrator_snapshots`; `get_merge_active_series` | present |
| `merge_queue.py::resolve_active`, `::_get_durations`, `::outcome_distribution` | present |
| `escalation.queue.iter_all_escalation_paths`; `escalation.classify.effective_benign`; `escalation.models.RESOLUTION_CLASSES` | imported by `escalation_analytics.py` today |
| `write_journal.py::get_memory_timeseries`, `::get_operations_breakdown` | present, both over `write_ops` |
| `performance.py::_project_cutoffs` / `_WINDOW_SQL` | present (to be removed) |
| `app.py::_parse_window`, `_WINDOW_DAYS`; `app.jsx` window state + `Toolbar` | present |
| `scheduler_utils.jsx::lockChipState`, `scheduler_heatmap.jsx::cellStateFor`, `tabs.jsx::LocksCell` | present |
| `data.js::applyKey` + `__loaded` marker; `DEFAULT_POLL_DEPS` | present — the registry extension point |
| node `--test` harness for classic scripts (`tests/js/`, `classic_script_scope.test.mjs`) | present |
| root `.venv` carries every workspace member (generator + parity test can import `shared`) | `CLAUDE.md` § Locating installed code |

No novel substrate beyond these; the generator and the `Datum` type are new code in
this PRD, not assumed capabilities.

## Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-status-authority-prd.md` | consumes | `shared.task_statuses` vocabulary + `ACTIVE` | that PRD (landed) | wired |
| `plans/dashboard-alignment-prd.md` | extends | grep-guard-as-pytest pattern; decision 1 no-import-unification | this PRD reuses, does not edit | wired |
| `plans/dashboard-availability-prd.md` | consumes | request budgets (`_TASKS_TOTAL_BUDGET` family) | this PRD keeps the bounds, changes what is fetched under them | wired |
| `plans/dashboard-taskgraph-legibility-prd.md` | supersedes | `prd_grouping.js` bucket rules (its decision 8 "different questions") | this PRD (views) | queued (γ3) |
| `plans/dashboard-task-runtime-endpoint-prd.md` | consumes | per-task runtime rows on `/tasks` | unchanged | wired |
| task 4795 (deferred) | supersedes | tasks fan-out budget starvation | this PRD (β) ; ι inherits its acceptance | close as superseded at decompose |
| task 5320 (done) | adjacent | recon vocabulary module | left as landed | n/a |
| orchestrator `digest.py::model_role_rollup` | consumes | window-bound semantics | orchestrator; dashboard mirrors (decision 10) | wired |
| `escalation` package | consumes | `iter_all_escalation_paths`, `RESOLUTION_CLASSES` | escalation | wired |

No reciprocal-ownership ambiguity: every seam above is either landed or owned here.

## Contract (B + H)

### The `Datum` envelope (declared schema, `dashboard/data/datum.py`)

```
Datum[T] = {
  "value":       T | null,            # null iff state == "unknown"
  "as_of":       str | null,          # ISO-8601 UTC of the measurement that produced value
  "state":       "fresh" | "stale" | "lower_bound" | "unknown",
  "reason":      str | null,          # required when state != "fresh"; verbatim producer message
  "age_seconds": int | null,          # now - as_of at shaping time; null iff as_of null
}
```

Invariants (machine-checked by a validator every shape function runs):
- `state == "unknown"` ⇔ `value is None`.
- `state != "fresh"` ⇒ `reason` non-empty.
- `state == "stale"` ⇒ `age_seconds > freshness_bound` for that datum family.
- The SPA never constructs a `Datum`; `data.js` only stores them.

### The task census (`dashboard/data/census.py`)

```
TaskCensus = {
  "counts": {<TaskStatus member>: int, ... all nine keys always present ...},
  "total":  int,                                  # == sum(counts.values())
  "views":  {"running": int, "in_flight": int, "backlog": int, "terminal": int},
  "in_progress_live": int, "in_progress_stranded": int,   # partition of counts["in-progress"]
}
```
- `views` is computed by `census.py::VIEWS`, a mapping `view → frozenset[TaskStatus]`,
  the only definition; a test asserts the four sets partition `TaskStatus`.
- `build_census(status_map, active_rows, now) -> TaskCensus` is pure.

### `/api/v2/dashboard/tasks` payload (additions and removals)

```
TASKS_SNAPSHOT: { <project>: {
    "census": Datum[TaskCensus],
    "rows":   Datum[list[TaskRow]],        # lower_bound when the terminal window was cut
} }
```
- `DONE_COUNTS` removed. `TASKS_OFFLINE_PROJECTS` / `DEGRADED` / `COUNT_UNKNOWN` lists
  remain for the banner but are derived from the `state`/`reason` of the snapshot
  entries, not computed separately.
- `/api/v2/dashboard/orchestrators` entries lose `summary`; keep `pids`, `running`,
  `started`, `last_update`, `offline`, `error`.
- `BURNDOWN_BY_PROJECT[p]` gains `as_of`, `age_seconds`, `state`; series carry all nine
  members plus the live/stranded split; `BURNDOWN` aggregate is a carry-last sum.
- Every windowed payload gains `WINDOW: {requested, served, days}`.

### The generated vocabulary (`static/redux/task_vocab.js`, classic script)

Exports `window.DF_TASK_VOCAB = {MEMBERS, VIEWS, TONES}`. Regenerated by
`scripts/gen_dashboard_task_vocab.py`; `dashboard/tests/test_task_vocab_generated.py`
regenerates to a temp file and asserts byte equality with the committed file.

### Shared rendering components

`StatTile({label, datum, history, …})`, `ST(...)` likewise, `ProjectGroup` pips and
`LocksCell` take a `Datum`. `history` is the persisted series of the same datum key.
A component receiving a bare number for `datum` throws in tests (the grep-guard is
the backstop in production code review).

### Sampler contract (`burndown.py::collect_snapshot`, `metrics.py::collect_metrics_snapshot`)

- Consumes the access implementation's snapshot builder; never re-fetches by its own
  route.
- Writes one row per project per tick: a value row, or a `state='gap'` row carrying
  `reason` when acquisition failed.
- Aggregation over projects at label `t` uses each project's last value row at or
  before `t`, with `age` = `t - as_of`.

### `data.js` datum registry

The endpoint→keys map becomes `endpoint → {key: {kind: "datum" | "plain"}}`. A datum
key is applied only when its payload validates as a `Datum`; a failed fetch leaves the
previous `Datum` in place and the registry stamps `state="stale"` on it with the fetch
error as `reason` (the one place the client may alter a state, and only in that
direction).

## Boundary-test sketch (B + H) — the integration gate ι's signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Census agrees everywhere | fixture `/tasks` payload with dark-factory census (running 25, in-flight 43, backlog 1310, terminal 4106), fresh | OrchTab card, pips, filter bar, four tiles, Overview tile, topbar chip, rail badge render numbers equal to the named view they label; no surface renders a number absent from `views`/`counts` |
| 2 | Unknown census renders uniformly | payload with dark-factory `census.state="unknown"`, reason "budget expired" | every surface in #1 renders `—` with that reason as title; no `0`, no `0/1`; banner names the project |
| 3 | Stale census carries value and age | `state="stale"`, `age_seconds=10800`, value present | every surface renders the value plus `3h` badge; the Progress bar widths use the stale value |
| 4 | Parts sum to whole | any fresh payload | `sum(counts) == total` and `sum(views) == total`, asserted server-side and in the SPA test |
| 5 | Nine statuses, nine buckets | a fixture tree containing one task in each `TaskStatus` member | each member counted exactly once in `counts`; `review` and `infra-hold` rows present in `rows` |
| 6 | Tile spark matches headline | burndown history whose last point is `t`, live census `as_of == t` | spark's last value equals the tile value; with `as_of > t` the tile shows the live value and the spark's last point is unchanged |
| 7 | Burndown gap and carry-last | project A samples at t1,t2; project B samples at t1 only, gap at t2 | aggregate at t2 = A(t2) + B(t1); B's row shows age = t2 − t1 and `state="stale"`; a `gap` row exists for B at t2 |
| 8 | Window echoed and declined | request `?window=90d` on `/costs` | payload `WINDOW = {requested:"90d", served:"30d", days:30}`; header renders "30d (90d not available)"; chip highlights none and resets on next tab switch |
| 9 | Recent merges follow the window | events over 3 days, chip 7d then 24h | table rows and its "showing N of M in 7d/24h" caption change with the chip; the outcome donut total M equals the latency tile's N plus its "without duration" remainder |
| 10 | One escalation corpus | root has 2 pending, archive has 3 pending for one project | pill "queue pending 2", strip "open in history 5", both with the same `as_of`; changing only the cache TTL never changes either number |
| 11 | Memory ops reconcile | `write_ops` rows with a `kind` outside read/write | reads + writes + other == donut total, on both Overview and Memory tab |
| 12 | Locks unknown when scheduler offline | scheduler snapshot lists project P in `offline_projects` | every in-progress row of P shows Locks `—` with reason; the heatmap and the chip classify a parked-and-held cell identically |
| 13 | Old paths gone | repository state | grep-guards pass: no `|| 1`/`?? 0`/`|| []` on datum fields in `static/redux/`; no status-literal counting outside `task_vocab.js`; `collect_done_counts`, `task_status_counts.js`, `dailyDeltas`, `_STATUS_MAP`, `orchestrator.py` summary absent |
| 14 | Cold render over nine roots (4795 acceptance) | caches cleared, all nine configured roots reachable | no root marked `unknown`/`stale` by budget; total handler time and payload size recorded in the completion note against 4795's 20 s / 11 MB baseline |

## Decomposition plan

Sizing per the overlay bands (300–1500 LOC, ≤10–12 files); every pair of leaves that
touch `tabs.jsx`, `redux_api.py`, `app.py` or `data.js` is serialised by a real
dependency edge, so the plan is a chain with two short side branches. Greek labels
become task ids at decompose.

| # | Title | Modules | Observable signal / unlock | Prereqs |
|---|---|---|---|---|
| α | `Datum` envelope, `TaskCensus` + views, generated task vocabulary + parity test | `dashboard/data/datum.py`, `dashboard/data/census.py`, `scripts/gen_dashboard_task_vocab.py`, `static/redux/task_vocab.js`, `index.html`, tests | **Intermediate** — unlocks β, γ1. `build_census` over a nine-member fixture returns a partition (sketch #4, #5); `task_vocab.js` byte-equal to regeneration | — |
| β | One task snapshot on `/tasks`; `/orchestrators` loses `summary`; dead census code deleted | `data/active_tasks.py`, `data/tasks.py`, `data/orchestrator.py`, `data/redux_api.py`, `app.py` (handler extracted, not grown), tests | **Intermediate** — unlocks γ2/γ3/δ. `GET /api/v2/dashboard/tasks` carries `TASKS_SNAPSHOT[p].census` as a `Datum`; `/orchestrators` has no `summary`; `_ACTIVE_STATUSES` derives from `shared.task_statuses.ACTIVE`; `collect_done_counts` gone | α |
| γ1 | Shared components render a `Datum`; `data.js` datum registry | `charts.jsx`, `tabs.jsx` (`ST`, `LocksCell`), `shell.jsx`, `task_row_cells.js`, `data.js`, `tests/js/` | **Intermediate** — unlocks γ2, γ3, δ, ζ, η, θ. Node tests: unknown → em-dash+reason, stale → age badge, lower_bound → `≥`; a bare number throws; a failed fetch stamps `stale` | α |
| γ2 | Orchestrators tab, Overview and app chrome consume the census views | `tabs.jsx` (OrchTab), `tab_overview.jsx`, `app.jsx`, `shell.jsx`, tests | **Leaf** — sketch #1–#3 on the live dashboard: the reported "0/1 vs Active 33" cannot occur; topbar and rail show the same `in_flight` number | β, γ1 |
| γ3 | Tasks tab and PRD boxes consume the census; client bucketers deleted | `tab_tasks.jsx`, `prd_grouping.js`, `task_status_counts.js` (deleted), `orch_filter.js`, tests incl. retiring `test_status_matches_keeps_the_three_status_disjunction` | **Leaf** — Tasks header pips are view lookups labelled `running`/`in-flight`; an offline or degraded project shows `—`; PRD box shows `≥ n/m` | γ2 |
| δ | Burndown persists the census datum; gap rows; carry-last aggregate; staleness on the wire; client `dailyDeltas` deleted | `data/burndown.py`, `data/redux_api.py`, `app.py::_burndown_loop` (extracted), `burndown_bands.js`, `tabs.jsx` (BurnTab, OrchTab spark), `shell.jsx`, tests incl. retiring the ragged-passthrough pin | **Leaf** — sketch #6, #7 on live data: aggregate Backlog equals the per-project sum; dark-factory and reify produce a value or gap row every tick; "Status mix" legend has nine entries | γ2 |
| ε | Window threading: echo, per-tab validation, labels from payload, recent merges follow chip, performance wall-clock + idle, model-role bounds | `app.py` (window parsing extracted), `app.jsx`, `tabs.jsx` (Costs/Burn/Merge/Perf headers), `data/merge_queue.py`, `data/performance.py`, `data/model_role.py`, tests incl. retiring the 1440 pin | **Leaf** — sketch #8, #9 first half; no header literal "30d" remains | δ |
| ζ | Merge attempts as one datum; "In queue now" as live datum with sampled history | `data/merge_queue.py`, `data/metrics.py`, `data/redux_api.py`, `app.py::api_merge_queue` (extracted), `tabs.jsx` (MergeTab), tests | **Leaf** — sketch #9 second half: donut total = latency N + "without duration" remainder on the same card; probe failure renders the tile `stale` with sample age | ε |
| η | Escalations as one corpus walk with location views and one cache; complete resolution classes | `data/escalations.py`, `data/escalation_analytics.py`, `app.py` handlers (extracted), `tab_escalations.jsx`, `tab_escalation_analytics.jsx`, tests | **Leaf** — sketch #10 on live reify: pill and strip carry one `as_of`; `stale-strand`/`moot-terminal-subject` appear in the split | ζ |
| θ | Memory ops one query; queue normaliser + offline state; single lock classifier; Locks unknown when scheduler offline | `data/write_journal.py`, `data/redux_api.py::shape_memory`, `data/scheduler.py`, `scheduler_utils.jsx`, `scheduler_heatmap.jsx`, `tabs.jsx` (MemoryTab, LocksCell), `tab_overview.jsx`, tests | **Leaf** — sketch #11, #12 | η |
| ι | Integration gate: grep-guards, old-path census, cold-render acceptance, full boundary suite | `dashboard/tests/test_datum_guards.py`, `tests/js/boundary_*.test.mjs`, completion note | **Leaf (gate)** — sketch #13, #14; closes 4795 as superseded with the measured before/after | θ |

Companion at decompose: close 4795 as superseded by ι with a pointer; no other PRD
prose needs editing.

## Out of scope

- A per-PRD server census (needs a PRD field on the status map — a fused-memory
  change); PRD boxes stay `lower_bound`.
- Field projection on `get_tasks` (4795 direction d) — the row fetch is already ~1 s.
- Regenerating `recon_status.js` from `journal.py` (cross-package; 5320's runtime
  unknown bucket already catches drift).
- Dashboard supervision/watchdog (`dashboard-availability-prd.md`).
- Any change to fused-memory, escalation or orchestrator packages.
- Raising or retuning the request budgets; ι measures, the next capacity change decides.

## Open questions (tactical — decide at implementation)

1. **Freshness bounds per datum family.** Suggested: live census 2× poll interval
   (6 s ⇒ 10 s); sampled data 2× sample interval (20 min); escalations 2× TTL (120 s).
   Decide in α/γ1.
2. **`Datum` implementation type.** TypedDict + validator vs a frozen dataclass with
   `to_wire()`. Suggested: frozen dataclass (heuristic 8), validated once at shaping.
   Decide in α.
3. **Gap-row representation.** Nullable `state` column with `reason` vs a separate
   `snapshot_gaps` table. Suggested: column, via the existing migration idiom. Decide
   in δ.
4. **Stale-age badge format** (`3h` / `3h 12m` / ISO). Suggested: coarse humanised age
   via `window.DF_SHELL.timeago`. Decide in γ1.
5. **Recent-merges cap N.** Suggested 200. Decide in ε.
6. **Generated file header** and whether the generator runs in pre-commit or only in the
   parity test. Suggested: test only. Decide in α.
