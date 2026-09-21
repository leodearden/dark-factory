# PRD: dashboard-one-datum-one-path — every dashboard number has one access path and carries its own provenance

**Status:** active — authored 2026-09-18, amended the same day after an adversarial
review (eight blocking findings, all folded in; see §Adversarial review). **Approach
B + H** (contract + two-way boundary tests): blast radius is `dashboard/` server + SPA
+ `shared/` + `scripts/`, the load-bearing seam is the server→SPA wire contract, and
the mechanism count is well past eight. **Type:** greenfield discipline PRD that also
resolves the census seam the sibling dashboard PRDs left unowned.

**Code anchors** are cited by symbol (`path::symbol`), never by line. Main moves
fast — re-locate at implementation time.

## Goal

The dashboard never shows two different answers to one question, and never shows a
confident number for a fact it did not measure.

User-observable end state:

- Orchestrators › any project: the Progress card, its pips, the filter bar, the four
  top tiles, the Overview "tasks" tile, the topbar chip and the rail badge all show
  numbers derived from **one** per-project census taken at **one** acquisition, each
  labelled with the **named view** it shows (`in-flight`, `backlog`, `terminal`, and
  `running` as the disclosed sub-view of `in-flight`). "0/1" cannot occur.
- When a project's census could not be refreshed, every one of those surfaces shows
  the same thing: the last measured value with its age (`stale · 3h`), or an em-dash
  with the reason when nothing was ever measured (`unknown · not yet fetched`).
  Never `0`.
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
`tasks.py::fetch_tasks`, caches cleared; re-measured independently by the adversarial
reviewer):

| Call | dark-factory (5544) | reify (7624) | Nine roots cold |
|---|---|---|---|
| compact status map | 0.11–0.15 s / 91 KB | 0.07–0.09 s / 125 KB | 0.38 s / 0.2 MB |
| active rows (`statuses=ACTIVE`) | 1.0–1.5 s / 1361 rows | 0.8–0.95 s / 1485 rows | 2.49 s / 16.6 MB |
| terminal window (400 done/cancelled rows) | — | — | 2.25 s / 9.4 MB |
| whole tree (`orchestrator.py`, `burndown.py`, `app.py::_load_task_cards`, `merge_queue.py::load_task_titles`) | 3.4 s / 29 MB | 3.65 s / 30 MB | — |
| `GET /api/v2/dashboard/tasks` today | | | 6.8 s / 16.4 MB |

The whole-tree fetches are the heavy calls and this PRD removes all four from the
request path. The compact map is cheap **but not immune**: the dashboard journal for
07:00–09:00 on 2026-09-18 records 124 `fetch_statuses` failures across all nine
roots (dark-factory 26, reify 30, and 10 each for `autotrade` and `mission-control`,
which have zero tasks) beside 238 whole-tree `fetch_tasks` failures in the same two
hours — contention on the single fused-memory instance, not payload size, and much
of that contention is the whole-tree traffic this PRD removes. A single authority therefore has to hold its last good
value and serve it as `stale`, or a transient becomes a synchronised blackout. That
requirement shapes decisions 5 and 6.

**Ruled by Leo 2026-09-18**: treat divergence as a smell; for each datum, move every
access onto one shared access implementation and branch the datapath as far up the
stack as possible; provenance (B) is a consequence of that, not an add-on, because a
single path that fails silently turns contradictions into uniform lies.

Prior slice fixes (3516, 1564, 1601, 1814, 3517, 3857, 4884) each hardened one
surface; this PRD is the class fix. Task **4795** (deferred: tasks budget cannot
cover nine roots) is **not** superseded: its remaining lever after this PRD is field
projection on `get_tasks` (task 4390, out of scope here); ι records the before/after
cold-render measurement 4795's acceptance 4 asks for, and 4795 carries a dated note
pointing at ι (a dependency edge on a `deferred` task would be inert). Task
**5320** (done 2026-09-17) set the vocabulary-module precedent for recon status.

## Sketch of approach — the discipline

1. **A datum is a named fact with one access implementation.** `dashboard/data/` gets
   one module per datum family; nothing else may query that fact's source. Consumers
   receive the datum and derive views; they never re-count, re-bucket or re-fetch.
2. **Every datum is a `Datum` envelope on the wire**: `{value, as_of, state, reason,
   freshness_bound_seconds}` with `state ∈ {fresh, stale, unknown, lower_bound}`,
   produced only server-side by the access implementation, validated at shaping, and
   rendered only by the shared components, which refuse to paint a bare number for
   `unknown`.
3. **The task census** is one datum: a nine-member count vector over
   `shared.task_statuses.TaskStatus`, its total, and the named views, built from the
   compact status map, acquired together with the active rows as one snapshot unit
   with one cache, carried in `/api/v2/dashboard/tasks`. `/api/v2/dashboard/orchestrators`
   becomes process discovery only. The burndown sampler persists this same datum.
   Terminal rows are fetched on demand, never on the default render.
4. **Views are defined once, in Python**, and shipped; the SPA gets the vocabulary as a
   file **generated** from `shared.task_statuses` with a parity test (INV-5), so no JS
   ever buckets a status string for counting.
5. **A stat tile takes a datum**, and its spark is that datum's persisted history, so
   the headline and the trend line agree at the last point by construction.
6. **A sampler persists a datum**, writes an explicit gap row on failure, and the
   aggregate carries each project's last value forward with its age rather than
   zero-filling absence; safety signals evaluate measured rows only.
7. **The window is a datum too**: every windowed payload echoes `{requested, served,
   days}`; labels render from it; the chip is validated against the destination tab.
8. **Two freshness facts, two homes.** "Has this endpoint refreshed" stays with the
   existing `static/redux/endpoint_staleness.js` (three consecutive failures,
   `DF_DATA.__stale`, `app.jsx::staleNoticesForTab`, task 4884). "How old is this
   measurement" is the Datum's `as_of`. The client never stamps a Datum state.
9. **Old paths are deleted, not deprecated**, behind executed boundary tests and one
   AST-based access-path check that reuses the clock-discipline guard's apparatus.

## Resolved design decisions

1. **Scope: all seven root causes in one PRD** (Leo, 2026-09-18).
2. **Census wire home: inside `/tasks`, one snapshot unit with the rows.** `/orchestrators`
   drops `summary`; the burndown endpoint is the persisted history of the same datum.
   Rejected: a separate `/census` endpoint (counts and rows from different poll cycles
   again) and embedding in both endpoints (two wire copies — RC1's shape).
3. **Named views.** The partition of `TaskStatus` is three views: `in_flight` =
   in-progress + blocked + merge-deferred + review + infra-hold; `backlog` = pending
   + deferred; `terminal` = done + cancelled. `running` = in-progress is a **sub-view**
   of `in_flight`, always rendered with its superset ("25 running of 43 in-flight"),
   and is the only number an operator compares against `max_concurrent_tasks`. A
   test asserts the three views partition `TaskStatus` and `running ⊆ in_flight`.
   **Amendment vs the first draft**: `infra-hold` moved from `backlog` to `in_flight`
   on the reviewer's evidence — it is entered from in-progress via
   `orchestrator/src/orchestrator/workflow.py::_mark_blocked(block_status='infra-hold')`
   on a task that still holds a worktree and resumes to `pending`, so it is a hold on
   dispatched work, like `blocked`, not never-started work. `review` is in `in_flight`
   (past dispatch, under verification). "Active" as a label is retired. The
   PRD-finished question (`prd_grouping.js`) is `terminal` vs the rest.
4. **Vocabulary reaches JS by generation**: `scripts/gen_dashboard_task_vocab.py`
   renders `dashboard/src/dashboard/static/redux/task_vocab.js` (members, view
   membership, display tones) from `shared.task_statuses`; the parity test lives in
   root `tests/scripts/` beside the existing generator precedent
   (`scripts/render_dashboard_unit.py` + `tests/scripts/test_dashboard_service_template.py`),
   where `scripts/` is on `sys.path` by design. Rejected: the hand-maintained-twin
   pattern of `recon_status.js` (kept there as landed) and views-only.
5. **The task snapshot is one acquisition unit.** `dashboard/data/task_snapshot.py`
   acquires, per project and under one budget share: the compact status map (paged,
   `page_size ≤ 2000`, `has_more` honoured — today's `fetch_statuses` makes one
   unpaginated call whose 91–125 KB responses already exceed the server's documented
   ~62 KB safe envelope) and the active rows (`statuses = shared.task_statuses.ACTIVE`,
   which fused-memory's `_VALID_TASK_STATUSES` accepts). The two reads carry their own
   `as_of`; the unit is cached **as one entry with one TTL**, replacing today's 5 s
   status-map cache and 20 s row cache. The live/stranded split partitions the
   **rows'** in-progress count (via `shared.task_claimant.is_stranded`) and carries the
   rows' `as_of`; the census's in-progress count may differ from the rows' by the
   intra-unit skew, which is emitted as `skew_seconds` and rendered when non-zero,
   never hidden; the rendered skew badge has a 2 s threshold, since two reads in one
   `gather` always differ by milliseconds. **Terminal rows are on demand**:
   `?terminal=<project>` fetches the done/cancelled window (400 rows, descending id,
   `state="lower_bound"`) only when the Complete filter **or the PRD grouping
   toggle** is active for that project, removing the 2.25 s / 9.4 MB call from the
   default render. The client side of that request is γ3's (`data.js` gains a
   per-project on-demand key; γ1 gives the registry parameterised-key support). This
   **resolves task 4416** (pending): its option (a) — the live-PRD terminal-member
   contract reads "live members inside the fetched window" — with the under-count
   disclosed as `lower_bound` rather than sanctioned silently; γ3 amends
   `plans/dashboard-taskgraph-legibility-prd.md` accordingly and 4416 is wired to
   depend on γ3 so the two cannot be dispatched over the same files. No count is
   ever derived from a row set except the PRD box's `≥ n/m` (decision 8).
6. **Datum states are decided server-side only.** `fresh`: refreshed within
   `freshness_bound_seconds`. `stale`: refresh failed or the value is older than the
   bound; the access implementation serves its last good value with `reason` (the
   verbatim failure). `unknown`: no value has ever been measured in this process, or
   the last value is older than the retention bound; `reason` set. `lower_bound`: a
   tally over a disclosed window. The client never synthesises or alters `value`,
   `as_of`, `state` or `reason`: a failed fetch leaves the previous payload in place
   and is reported by the existing endpoint-staleness mechanism (sketch item 8). The
   registry **does** attach receipt metadata when it applies a payload — each stored
   `Datum` gets `_served_at` (copied from the payload) and `_received_at` (client
   clock) — and the shared components compute displayed age as
   `(served_at − as_of) + (now − received_at)`, immune to clock skew, so a payload
   that stops refreshing keeps ageing on screen; `age_seconds` is not on the wire.
   Bounds are declared beside each access implementation and shipped on the
   envelope. When the computed age passes the bound while `state` still reads
   `fresh` (the server's verdict at shaping), the **age badge wins** on screen: the
   operator reads the age, and the state is what the server knew when it answered.
7. **Rendering is uniform and lives in the shared components only**: `charts.jsx::StatTile`,
   `tabs.jsx::ST`, a new shared `Pip` and the row cells accept a `Datum`.
   `unknown` → em-dash with the reason as title; `stale` → value plus age badge;
   `lower_bound` → `≥` prefix. Tab code never branches on state. Every call site
   migrates in γ1 — **43 tile sites, measured 2026-09-21 (task 5706)**: 32 in
   `tabs.jsx`, 4 in `tab_overview.jsx`, 5 in `tab_escalations.jsx`, 2 in
   `tab_escalation_analytics.jsx`. The component is reached by **three** syntactic
   forms and only one of them carries its own name, so enumerate by grep at
   implementation across all three: `<StatTile` (plain destructure —
   `tab_overview.jsx`), `<ST` (the `StatTile: ST` alias in `tabs.jsx`'s
   `window.DF_CHARTS` destructure) and `<C.StatTile` (both escalations tabs bind
   `const C = window.DF_CHARTS` and call through the namespace object;
   `esc_flow_diagram.jsx` binds the same `C` but uses only `C.PALETTE`, so it holds no
   tile — measured 2026-09-21). A `<StatTile` grep alone matches 4 of the 43 and
   silently misses the other 39 — which is why the two escalations tabs were absent
   from this decision's earlier enumeration. Those three forms are a **discovery aid,
   not the completeness guarantee**: a fourth binding form would miss again just as
   silently, and decision 17 rejects a substring matcher trusted as a guard. The
   guarantee is executed, per decision 17(c) — the migrated components throw on a
   non-`Datum` argument, and because the legacy shape is mechanically distinguishable
   (`charts.jsx::StatTile` takes `value`/`unit`/`delta` today, never `datum`), a
   `value`-with-no-`datum` call throws too; γ1's node tests render each of the four
   tile-bearing tabs against a fixture payload, so an unmigrated site fails at render
   whichever binding form reached it. Retired from that earlier list:
   `scheduler_drawer.jsx`, `tab_curator.jsx`, `tab_memory_evals.jsx` and
   `tab_scheduler.jsx` carry **zero** tile call sites and bind no `StatTile` at all. `shell.jsx::ProjectGroup` is **not** the pips' owner
   either: it takes `summary` as an opaque node and renders no pip itself (the token
   does not occur in the file). Every pip today is a caller-side JSX fragment sharing
   only the `.proj-head .summary .pip` CSS — **23 fragments, measured 2026-09-21
   (task 5706)** by `className="pip"`, which carries none of the tiles' alias hazard:
   16 in `tabs.jsx`, 4 in `tab_escalations.jsx`, 3 in `tab_tasks.jsx` (one of those
   three a `.map()` over `activityPips`, so it emits one pip per active status at
   runtime). So γ1's shared `Pip` is **new code**, not a migration of an existing
   component. The pips also split across phases where the tiles do not: γ1 ships `Pip`
   and migrates the `tabs.jsx` and `tab_escalations.jsx` fragments, while
   `tab_tasks.jsx` — which appears in no γ1 module list, only γ3's — migrates its three
   alongside γ3's census rework. Shipping `Pip` with `tab_tasks.jsx` untouched is
   γ1 complete, not a γ1 miss. Values not yet served as a `Datum` are
   wrapped by `plainDatum(value, endpointKey)` with the endpoint's `served_at` as
   `as_of`, so every tile has provenance at least at endpoint granularity. `data.js`
   pre-fetch defaults for datum-kinded keys become `unknown` Datums with reason
   "not yet fetched".
8. **PRD box `n/m done` is a `lower_bound` datum** over the fetched rows, rendered as
   `≥ n/m`, until a per-PRD server census exists (out of scope: the status map carries
   no PRD field).
9. **Burndown**: the sampler consumes the snapshot unit from decision 5; a failed cycle
   writes a row with `state='gap'` and `reason`; all nine members are stored — the
   three columns that do not exist today (`review`, `merge_deferred`, `infra_hold`)
   are added **nullable**, NULL meaning "not measured before this migration" and
   rendered as gap, never as a `DEFAULT 0` fabricated zero across history;
   `_STATUS_MAP`/`_ZONE_KEYS`/`_BURNDOWN_KEYS`/`burndown_bands.js` collapse to the
   generated vocabulary. **Two sources, disclosed in the row**: the nine member
   columns come from the census (status map) and sum to `total`; the live/stranded
   split and a new `in_progress_rows` column come from the rows and carry the rows'
   `as_of`, so `_SPLIT_KEYS`' invariant is restated as `live + stranded ==
   in_progress_rows`, not `== in_progress`. `running` everywhere is the census's
   in-progress; the task-3543 parity alarm compares the rows-derived
   `in_progress_live` against the cap, as it does today, and the two numbers sit in
   one row with their skew visible. `redux_api.py::shape_burndown` builds the aggregate by
   carrying each project's last **measured** row forward with its age, and emits
   per-project `as_of`, `state`. `burndown.py::compute_parity_alarm`,
   `redux_api.py::_aggregate_parity` and `compute_forecast_confidence` evaluate
   measured rows only — a carried-forward or gap row never contributes to a breach
   count or a forecast. The client-side `shell.jsx::dailyDeltas` is deleted; the
   OrchTab "Completed / day" spark consumes the server's per-project `completed`
   series. `test_shape_burndown_ragged_input_is_passed_through_unnormalized` is
   **updated** to the carry-last contract as its own docstring instructs; the
   per-project ragged pass-through it also protects is retained.
10. **Window**: `app.py::_parse_window` and the burndown parser return `{requested,
    served, days}` and every windowed payload echoes it; `app.jsx` re-validates the
    chip against the destination tab's set on tab switch and resets to that tab's
    default when invalid; hardcoded "30d" labels render from the payload.
    **Recent merges follows the chip**, newest N within the window, labelled
    "showing N of M in <window>" (Leo, 2026-09-18) — `test_app.py`'s
    `recent_window_minutes=1440` pin is retired (the smallest chip is 24h, so the new
    behaviour can only widen). **Performance anchors to wall-clock** with an idle
    state: a project with no completions in the window renders its cards `stale`
    with last-completion age (Leo, 2026-09-18) — `performance.py::_project_cutoffs`
    goes. `model_role.py::_cutoff` adopts the digest's bound semantics.
11. **Merge attempts are one datum**: one query over `merge_attempt` events in the
    window feeds both the outcome distribution and the latency stats; the latency
    tile is labelled "of N with recorded duration". "In queue now" is a live datum
    whose history is `merge_snapshots`; when the live probe fails the tile is `stale`
    with the last sample's age, never the events-table approximation shown as live.
12. **Task rows for arbitrary ids come from the snapshot unit plus a per-id miss
    path — never a whole tree.** Today `app.py::_load_task_cards` (escalation cards,
    which carry the **whole task row**, not a title — `redux_api.py::shape_escalations`
    emits `task: dict`) and `merge_queue.py::load_task_titles` each fetch the whole
    tree on request under a 10 s cache. `get_tasks` has no id filter (verified), but
    `get_task` does exist per id. `dashboard/data/task_lookup.py` resolves an id set
    by reading the snapshot unit's active rows first and fetching each miss with
    `get_task` into a per-id cache (terminal rows change rarely; long TTL keyed on
    status). **Measured 2026-09-18** on the live corpus: 149 pending escalation rows
    carry a task id, 104 distinct ids, 53 active (served from rows), 51 misses; a
    cold `get_task` costs 0.1–4.4 s (mean 1.85 s, cold MCP sessions dominate). The
    miss path is therefore **budgeted in the same idiom as today's
    `app.py::_load_task_cards`** (which is wrapped in `asyncio.wait_for(_TASK_CARDS_BUDGET)`
    because an unbounded path there once wedged `/escalations` for 19.8 h): a
    whole-operation deadline per request, a concurrency width (semaphore, suggested
    4) over the miss set, and a per-request miss cap (suggested 64, newest ids
    first); ids beyond the cap or the deadline are `unknown` with reason
    "lookup budget", and the per-id cache makes the second render cheap. An id absent
    from the store is `unknown` with reason, not stale. Freshness is the snapshot's
    for active ids and per-read for cached misses — no 10-minute regression. Both
    consumers read this datum.
13. **Escalations are one corpus walk** (root + archive via
    `escalation.queue.iter_all_escalation_paths`) with a `location` field, one cache,
    consumed by both `escalations.py` and `escalation_analytics.py`; "pending in live
    queue" and "open in history" are views; the tab shows the datum's age (Leo,
    2026-09-18). The benign/actionable split enumerates every member of
    `escalation.models.RESOLUTION_CLASSES` so `stale-strand` and
    `moot-terminal-subject` are visible.
14. **Memory ops are one query** grouped by both `kind` and `operation`; reads/writes
    and the breakdown donut derive from it. `redux_api.py::shape_memory` normalises the
    queue block in one place; `queue.offline` becomes the datum state and is rendered.
    The recon "Active agents" tile and its hint use one activity rule (sweep
    memory-recon#5).
15. **Locks**: `scheduler_utils.jsx::lockChipState` is the single classifier;
    `scheduler_heatmap.jsx::cellStateFor` calls it; a project in the scheduler
    snapshot's `offline_projects` yields an `unknown` Locks datum, not an empty list.
16. **Dead second implementations are deleted**: `active_tasks.py::collect_done_counts`,
    `task_status_counts.js` and its four pins (`index.html` script tag,
    `test_index_html.py`, `test_cache_buster_freshness.py::_REQUIRED_ASSETS`,
    `tests/js/classic_script_scope.test.mjs::EXPECTED_WINDOW_GLOBALS`), the bucketers
    in `prd_grouping.js`, `orchestrator.py`'s summary, `app.jsx`'s two reductions,
    the two request-path whole-tree fetches (decision 12).
17. **Guards are executed, not grepped.** The first draft's token greps (`|| 1`,
    `?? 0`, `|| []`, status literals) were measured unsatisfiable — the only `|| 1`
    in `static/redux/` is a chart-range fix, `|| []` is the optional-collection idiom
    123 times over, and 62 % of status literals are filters or styling — and INV-10
    rejects a substring guard whose matcher is untested. Replaced by: (a) the
    boundary suite below, executed in the merge gate; (b) one AST-based access-path
    check in `dashboard/tests/test_datum_access_paths.py`, built on
    `test_clock_discipline.py::find_clock_violations`'s exemption apparatus, asserting
    that `fetch_tasks`/`fetch_task_page`/`fetch_statuses` are called only from
    `data/task_snapshot.py` and `data/task_lookup.py`, with one named exemption:
    `app.py::_fanout_probe_completion` (the `/healthz` MCP-fanout liveness probe)
    calls the raw, **uncached** fetch by design — a cache hit would mask exactly the
    wedge it exists to detect — and keeps doing so after β removes the module-level
    caches; (c) the shared components throwing on a non-`Datum` argument under the
    node test harness.
18. **`app.py` is already past the size alarm** (`docs/code-quality.md` heuristic 14,
    2511 lines). Extraction is one mechanical leaf (α2) at the start, not a side
    effect of six semantic leaves: the handlers this PRD touches **and the two
    background loops** (`app.py::_burndown_loop`, `app.py::_metrics_loop`, which δ1
    and ζ must change) move to
    `dashboard/src/dashboard/api/{tasks,orchestrators,burndown,merge_queue,escalations,memory,window}.py`
    and `dashboard/src/dashboard/loops.py` with no behaviour change; `test_app.py`
    is retargeted mechanically (import and patch paths) with every assertion kept.
    The measure is heuristic 13 per new file and "no handler or loop body remains in
    `app.py`", never a line count (`docs/code-quality.md` § What not to steer by).
    Later leaves touch those modules and never grow `app.py`. `tabs.jsx` is at the
    soft ceiling; leaves touching it prefer deletion over addition.
19. **No dashboard→orchestrator import** (dashboard-alignment decision 1 stands).
    `shared` is already a runtime dependency and is the only cross-package import
    this PRD adds (four dashboard tests already import `shared.*`).
20. **One cache per datum.** The snapshot unit's cache (decision 5) and each access
    implementation's last-good value are the same entry; the 5 s and 20 s TTL caches
    are removed, not layered under. The unit TTL is 15 s: against today that raises
    the active-row fetch rate by a third (2.49 s / 16.6 MB per cold nine-root cycle,
    now at most every 15 s instead of 20 s) while removing four whole-tree fetch
    families (3.4 s / 29 MB each per request or tick) and two request-path
    whole-tree reads — a large net reduction of the contention the background table
    records. After a dashboard restart every live datum is `unknown` until its first
    acquisition (seconds); the persisted burndown history is a different datum and
    does not back-fill live values.

## Verified substrate (G3)

| Capability | Evidence |
|---|---|
| `shared.task_statuses.TaskStatus`, `ACTIVE`, `TERMINAL` | read 2026-09-18; nine members; `ACTIVE = frozenset(TaskStatus) - TERMINAL` |
| fused-memory accepts `review`/`infra-hold` in a `statuses` filter | `fused_memory/server/tools.py::_VALID_TASK_STATUSES = ACTIVE_TASK_STATUSES \| TERMINAL_STATUSES` imports from `shared` |
| dashboard imports `shared` at runtime | `dashboard/pyproject.toml` depends on `dark-factory-shared`; timing script imported it under `uv run --project dashboard` |
| fused-memory `get_statuses` paging (`page_size`, `offset`, `has_more`, ≤ 2000) | called live: dark-factory 5539 statuses in three pages |
| `tasks.py::fetch_statuses` cost / `fetch_tasks(statuses=…)` real server-side filter | measured table above; memory `fa730a43` |
| `get_tasks` has **no** id filter; `get_task` per id exists | grep of `fused_memory/server/tools.py` 2026-09-18 (`::get_task`) — hence decision 12's miss path |
| `app.py::_fanout_probe_completion` is a raw `fetch_tasks` caller by design (healthz) | present; exempted in decision 17 |
| `shared.task_claimant.is_stranded` | present; consumed by `tasks.py::task_is_stranded` |
| `burndown.py::ensure_snapshot_columns` probe-then-add migration | present; nullable columns expressible (`concurrency_cap INTEGER` precedent) |
| `static/redux/endpoint_staleness.js` (`STALE_FAILURE_THRESHOLD = 3`), `data.js::DF_DATA.__stale`, `app.jsx::staleNoticesForTab` | present (task 4884) — the endpoint-freshness home this PRD defers to |
| `metrics.py` sampler tables `merge_snapshots`, `orchestrator_snapshots`; `get_merge_active_series`; `_metrics_loop` 10-min cadence | present |
| `merge_queue.py::resolve_active`, `::_get_durations`, `::outcome_distribution`, `::load_task_titles`; `app.py::_load_task_cards`; `redux_api.py::shape_escalations` emits the whole task row per card | present |
| `escalation.queue.iter_all_escalation_paths`; `escalation.classify.effective_benign`; `escalation.models.RESOLUTION_CLASSES` | imported by `escalation_analytics.py` today |
| `write_journal.py::get_memory_timeseries`, `::get_operations_breakdown` | present, both over `write_ops` |
| `performance.py::_project_cutoffs` / `_WINDOW_SQL` | present (to be removed) |
| `app.py::_parse_window`, `_WINDOW_DAYS`; `app.jsx` window state + `Toolbar` | present |
| `scheduler_utils.jsx::lockChipState`, `scheduler_heatmap.jsx::cellStateFor`, `tabs.jsx::LocksCell` | present |
| `data.js::applyKey` + `__loaded` marker; `DEFAULT_POLL_DEPS` | present — the registry extension point |
| node `--test` harness in the merge gate | `dashboard/tests/test_graph_layout_js.py::test_graph_layout_js_suite_passes` (subprocess + TAP count), collected by `cd dashboard && uv run pytest tests/` |
| generator + parity-test precedent | `scripts/render_dashboard_unit.py` + `tests/scripts/test_dashboard_service_template.py::test_template_renders_to_hardcoded_file` |
| AST guard precedent with exemption apparatus | `dashboard/tests/test_clock_discipline.py::find_clock_violations` |
| root `.venv` carries every workspace member | `CLAUDE.md` § Locating installed code |

No novel substrate beyond these; the generator, the `Datum` type, the snapshot unit
and the `task_lookup` datum are new code in this PRD, not assumed capabilities.

## Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-status-authority-prd.md` | consumes | `shared.task_statuses` vocabulary + `ACTIVE` | that PRD (landed) | wired |
| `plans/dashboard-alignment-prd.md` | extends | AST-guard-as-pytest pattern; decision 1 no-import-unification | this PRD reuses, does not edit | wired |
| `plans/dashboard-availability-prd.md` + task 4884 | consumes | request budgets; `endpoint_staleness.js` as the endpoint-freshness home | unchanged; this PRD defers to it (sketch item 8) | wired |
| `plans/dashboard-taskgraph-legibility-prd.md` | supersedes | `prd_grouping.js` bucket rules (its decision 8); its §Contract line "`DONE_COUNTS` semantics unchanged" | this PRD (views; `DONE_COUNTS` retired) — γ3 adds a dated retirement note to that PRD | queued (γ3) |
| `plans/dashboard-task-runtime-endpoint-prd.md` | consumes | per-task runtime rows on `/tasks` | unchanged | wired |
| task 4795 (deferred) | related | tasks fan-out budget starvation | stays open; reads ι's measurement; remaining lever is 4390 (field projection) | dated note on 4795 pointing at ι at decompose (a dep edge on a `deferred` task is inert) |
| task 4416 (pending, low) | resolves | live-PRD terminal-member contract vs the windowed terminal fetch | this PRD (decision 5: option (a) + `lower_bound` disclosure + on-demand fetch); γ3 amends the taskgraph-legibility PRD | wire `4416 depends_on γ3` at decompose; 4416 closes when γ3 lands |
| task 4390 | out of scope | `get_tasks` field projection | fused-memory | untouched |
| task 5320 (done) | adjacent | recon vocabulary module | left as landed | n/a |
| orchestrator `digest.py::model_role_rollup` | consumes | window-bound semantics | orchestrator; dashboard mirrors (decision 10) | wired |
| `escalation` package | consumes | `iter_all_escalation_paths`, `RESOLUTION_CLASSES` | escalation | wired |

No reciprocal-ownership ambiguity: every seam above is landed, owned here, or
explicitly left with its owner.

## Contract (B + H)

### The `Datum` envelope (declared schema, `dashboard/data/datum.py`)

```
Datum[T] = {
  "value":  T | null,                       # null iff state == "unknown"
  "as_of":  str | null,                     # ISO-8601 UTC of the measurement; null iff unknown
  "state":  "fresh" | "stale" | "lower_bound" | "unknown",
  "reason": str | null,                     # required when state != "fresh"; verbatim producer message
  "freshness_bound_seconds": int,           # declared beside the access implementation
}
```
Every payload carries top-level `served_at` (server clock at shaping).

Invariants (machine-checked by a validator every shape function runs):
- `state == "unknown"` ⇔ `value is None` ⇔ `as_of is None`.
- `state != "fresh"` ⇒ `reason` non-empty.
- `state == "fresh"` ⇒ `served_at − as_of ≤ freshness_bound_seconds`.
- Produced only server-side; the SPA stores and renders, never constructs or mutates.

### The task census (`dashboard/data/census.py`)

```
TaskCensus = {
  "counts": {<TaskStatus member>: int, ... all nine keys always present ...},
  "total":  int,                                  # == sum(counts.values())
  "views":     {"in_flight": int, "backlog": int, "terminal": int},   # a partition: sums to total
  "sub_views": {"running": int},                                       # running ⊆ in_flight
}
```
- `census.py::VIEWS` maps `in_flight`/`backlog`/`terminal` → `frozenset[TaskStatus]`
  and `SUB_VIEWS` maps `running` → `{IN_PROGRESS}`; the wire mirrors that split so
  `sum(views.values()) == total` holds and no consumer can re-add the sub-view. A
  test asserts the three views partition `TaskStatus` and `running ≤ in_flight`.
- `build_census(status_map) -> TaskCensus` is pure.

### The task snapshot unit (`dashboard/data/task_snapshot.py`)

```
TaskSnapshot = {
  "census": Datum[TaskCensus],                   # from the paged status map
  "rows":   Datum[list[TaskRow]],                # active rows; lower_bound never (terminal rows are on demand)
  "in_progress_live": int, "in_progress_stranded": int,   # partition of the ROWS' in-progress count
  "skew_seconds": int,                           # |census.as_of − rows.as_of|
}
```
- One acquisition per project under one budget share; one cache entry; the last good
  unit is retained and served `stale` when a refresh fails.

### `/api/v2/dashboard/tasks` payload (additions and removals)

- `TASKS_SNAPSHOT: {<project>: TaskSnapshot}`; `DONE_COUNTS` removed;
  `TASKS_OFFLINE_PROJECTS` / `DEGRADED` / `COUNT_UNKNOWN` derived from the snapshot
  entries' `state`/`reason`, not computed separately.
- `?terminal=<project>` returns that project's terminal-row window as `Datum[list]`
  with `state="lower_bound"` and the window size in `reason`.
- `/api/v2/dashboard/orchestrators` entries lose `summary`; keep `pids`, `running`,
  `started`, `last_update`, `offline`, `error`.
- `BURNDOWN_BY_PROJECT[p]` gains `as_of`, `state`; series carry all nine members plus
  the live/stranded split; `BURNDOWN` aggregate is a carry-last sum over measured rows.
- Every windowed payload gains `WINDOW: {requested, served, days}`.

### The generated vocabulary (`static/redux/task_vocab.js`, classic script)

Exports `window.DF_TASK_VOCAB = {MEMBERS, VIEWS, SUB_VIEWS, TONES}`. Regenerated by
`scripts/gen_dashboard_task_vocab.py`; `tests/scripts/test_dashboard_task_vocab.py`
regenerates to a temp file and asserts byte equality with the committed file.

### Shared rendering components

`StatTile({label, datum, history, …})`, `ST(...)` likewise, the new shared `Pip`
(γ1 — see decision 7) and `LocksCell` take a `Datum`. `shell.jsx::ProjectGroup` is
not itself a `Datum` consumer: it keeps taking `summary` as an opaque node.
`history` is the persisted series of the same datum key.
`plainDatum(value, endpointKey)` wraps a not-yet-migrated value with the endpoint's
`served_at`. Under the node test harness a component receiving a non-`Datum` throws.

### Sampler contract (`burndown.py::collect_snapshot`, `metrics.py::collect_metrics_snapshot`)

- Consumes the access implementation's snapshot unit; never re-fetches by its own route.
- Writes one row per project per tick: a value row, or a `state='gap'` row carrying
  `reason` when acquisition failed.
- Aggregation over projects at label `t` uses each project's last **measured** row at
  or before `t`, with `age = t − as_of`; parity and forecast read measured rows only.

### `data.js` datum registry

The endpoint→keys map becomes `endpoint → {key: {kind: "datum" | "plain"}}`. A datum
key is applied only when its payload validates as a `Datum`; a failed fetch leaves the
previous value in place untouched — endpoint freshness is `endpoint_staleness.js`'s
fact, not the registry's. Datum-kinded keys default to an `unknown` Datum with reason
"not yet fetched" before the first response.

## Boundary-test sketch (B + H) — the integration gate ι's signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Census agrees everywhere | fixture `/tasks` payload with dark-factory census (in_flight 43 of which running 25, backlog 1310, terminal 4106, total 5459), fresh | OrchTab card, pips, filter bar, tiles, Overview tile, topbar chip, rail badge render numbers equal to the named view they label; no surface renders a number absent from `views`/`counts` |
| 2 | Unknown census renders uniformly | payload with `census.state="unknown"`, reason "not yet fetched" | every surface in #1 renders `—` with that reason as title; no `0`, no `0/1`; banner names the project |
| 3 | Stale census carries value and age | `state="stale"`, `as_of` three hours before `served_at`, reason "ReadTimeout" | every surface renders the value plus `3h` badge; the Progress bar widths use the stale value; the badge keeps ageing with local time if no new payload arrives |
| 4 | Parts sum to whole | any fresh payload | `sum(counts) == total`, `in_flight + backlog + terminal == total`, `running ≤ in_flight`, asserted server-side and in the SPA test |
| 5 | Nine statuses, nine buckets (synthetic — no live `review`/`infra-hold` rows exist today) | a fixture status map with one id in each `TaskStatus` member | each member counted exactly once; `review` and `infra-hold` land in `in_flight` |
| 6 | Tile spark matches headline | burndown history whose last point is `t`, live census `as_of == t` | spark's last value equals the tile value; with `as_of > t` the tile shows the live value and the spark's last point is unchanged |
| 7 | Burndown gap and carry-last | project A samples at t1,t2; project B samples at t1 only, gap at t2 | aggregate at t2 = A(t2) + B(t1); B shows age = t2 − t1 and `state="stale"`; a `gap` row exists for B at t2; parity breach count and forecast are computed from A(t2) and B(t1) measured rows only |
| 8 | Window echoed and declined | request `?window=90d` on `/costs` | payload `WINDOW = {requested:"90d", served:"30d", days:30}`; header renders "30d (90d not available)"; chip highlights none and resets on next tab switch |
| 9 | Recent merges follow the window | events over 3 days, chip 7d then 24h | table rows and the "showing N of M in <window>" caption change with the chip; the outcome donut total M equals the latency tile's N plus its "without duration" remainder |
| 10 | One escalation corpus | root has 2 pending, archive has 3 pending for one project | pill "queue pending 2", strip "open in history 5", both with the same `as_of`; changing only the cache TTL never changes either number |
| 11 | Memory ops reconcile | `write_ops` rows with a `kind` outside read/write | reads + writes + other == donut total, on both Overview and Memory tab |
| 12 | Locks unknown when scheduler offline | scheduler snapshot lists project P in `offline_projects` | every in-progress row of P shows Locks `—` with reason; the heatmap and the chip classify a parked-and-held cell identically |
| 13 | Snapshot unit survives a transient | status-map read raises `ReadTimeout` once after a good unit | `/tasks` serves the previous unit with `census.state="stale"`, reason "ReadTimeout"; the next good read returns it to `fresh` |
| 14 | Old paths gone | repository state | AST access-path check passes; `collect_done_counts`, `task_status_counts.js` and its four pins, `dailyDeltas`, `_STATUS_MAP`, `orchestrator.py` summary, request-path whole-tree fetches absent |
| 15 | Cold render over nine roots — **recorded, not gated** | caches cleared, all nine roots reachable | handler time, payload size and per-root state recorded in ι's completion note against the 2026-09-18 baseline (6.8 s / 16.4 MB; cold acquisition 0.38 s + 2.49 s + terminal 2.25 s); 4795 depends on ι and decides the next lever from this |

## Decomposition plan

Sizing per the overlay bands (300–1500 LOC, ≤10–12 files); every pair of leaves that
touch `tabs.jsx`, `redux_api.py`, `data.js` or the same `api/*.py` module is serialised
by a real dependency edge. α and α2 are independent roots; the rest is a chain.

| # | Title | Modules | Observable signal / unlock | Prereqs |
|---|---|---|---|---|
| α | `Datum` envelope, `TaskCensus` + views, generated task vocabulary + parity test | `dashboard/data/datum.py`, `dashboard/data/census.py`, `scripts/gen_dashboard_task_vocab.py`, `static/redux/task_vocab.js`, `index.html`, `tests/scripts/test_dashboard_task_vocab.py`, dashboard tests | **Intermediate** — unlocks β, γ1. `build_census` over a nine-member fixture yields the three-view partition with `running ≤ in_flight` (sketch #4, #5); `task_vocab.js` byte-equal to regeneration | — |
| α2 | Extract the touched handlers and the two background loops from `app.py`, no behaviour change | `app.py`, `dashboard/src/dashboard/api/{tasks,orchestrators,burndown,merge_queue,escalations,memory,window}.py`, `dashboard/src/dashboard/loops.py`, `test_app.py` (patch paths retargeted, assertions kept) | **Intermediate** — unlocks β. No handler or loop body remains in `app.py`; each new file passes heuristic 13; every `test_app.py` assertion still passes after retargeting; `_fanout_probe_completion` still calls the raw fetch | — |
| β | One task snapshot unit on `/tasks` (paged status map + active rows, one cache, last-good stale, terminal rows on demand); `/orchestrators` loses `summary`; `collect_done_counts` deleted | `data/task_snapshot.py`, `data/active_tasks.py`, `data/tasks.py`, `data/orchestrator.py`, `data/redux_api.py`, `api/tasks.py`, `api/orchestrators.py`, tests | **Intermediate** — unlocks γ2/γ3/δ1. `GET /api/v2/dashboard/tasks` carries `TASKS_SNAPSHOT[p].census` as a `Datum`; a forced `ReadTimeout` after a good unit serves `stale` (sketch #13); `/orchestrators` has no `summary`; `_ACTIVE_STATUSES` derives from `shared.task_statuses.ACTIVE`; the 5 s/20 s caches are gone | α, α2 |
| γ1 | Shared components render a `Datum`; every call site migrated; `data.js` datum registry with receipt metadata, parameterised on-demand keys and unknown defaults | `charts.jsx`, `tabs.jsx` (`ST`, `LocksCell`), `tab_overview.jsx`, `tab_escalations.jsx`, `tab_escalation_analytics.jsx`, `shell.jsx` (the new shared `Pip` only — no tile, `ProjectGroup` unchanged per the Contract, and its lone `window.DF_CHARTS` read `SP_SHELL` is referenced nowhere, measured 2026-09-21), `task_row_cells.js`, `tab_curator.jsx` (Sparkline/StepSpark only — no tile), `tab_memory_evals.jsx` (Sparkline/StepSpark only — no tile), `data.js`, `tests/js/` | **Intermediate** — unlocks γ2, γ3, δ2, ζ, η, θ. Node tests: unknown → em-dash+reason, stale → age badge computed from `_served_at`/`_received_at` that grows with a mocked local clock, lower_bound → `≥`; a non-`Datum` throws; pre-fetch render shows `unknown` tiles, not zeros; a per-project parameterised key can be requested and applied; each of the four tile-bearing tabs (decision 7) renders against a fixture payload, so any site left in the legacy `value` shape throws regardless of which binding form reached it | α |
| γ2 | Orchestrators tab, Overview and app chrome consume the census views | `tabs.jsx` (OrchTab), `tab_overview.jsx`, `app.jsx`, `shell.jsx`, tests | **Leaf** — sketch #1–#3 against fixture payloads under the node harness: the reported "0/1 vs Active 33" shape cannot render; topbar and rail show the same `in_flight` number with `running` disclosed | β, γ1 |
| γ3 | Tasks tab and PRD boxes consume the census; on-demand terminal fetch wired; client bucketers and `task_status_counts.js` (with its four pins) deleted; 4416 contract landed in the taskgraph-legibility PRD | `tab_tasks.jsx`, `prd_grouping.js`, `task_status_counts.js` (deleted), `orch_filter.js`, `data.js`, `index.html`, `test_index_html.py`, `test_cache_buster_freshness.py`, `tests/js/classic_script_scope.test.mjs`, `test_tab_tasks_status_counts.py` (retire the disjunction pin), `plans/dashboard-taskgraph-legibility-prd.md` | **Leaf** — Tasks header pips use the shared `Pip` (γ1) and are view lookups labelled `running of in-flight`; an offline or degraded project shows `—`; toggling Complete or PRD grouping issues `?terminal=<project>` and the PRD box then shows `≥ n/m` with n from the window; a synthetic nine-status fixture renders nine pips | γ2 |
| δ1 | Burndown sampler consumes the snapshot unit; gap rows; nullable member columns; `in_progress_rows` column | `data/burndown.py`, `loops.py`, tests | **Intermediate** — unlocks δ2. Against a fixture store: a forced acquisition failure writes a `gap` row with reason and a success writes a value row whose nine members sum to `total` and whose `live + stranded == in_progress_rows`; pre-migration rows read NULL, not 0, for the three new members | β |
| δ2 | Carry-last aggregate, per-project staleness on the wire, parity/forecast over measured rows only, BurnTab nine zones, `dailyDeltas` deleted | `data/redux_api.py`, `burndown_bands.js`, `tabs.jsx` (BurnTab, OrchTab spark), `shell.jsx`, `test_redux_api.py` (update the ragged test), tests | **Leaf** — sketch #6, #7 against a fixture store with a ragged two-project history: aggregate Backlog equals the per-project sum; "Status mix" legend has nine entries; parity breach count unchanged by a gap row | δ1, γ1 |
| ε1 | Window echo, per-tab chip validation, labels from payload, recent merges follow the chip | `api/window.py`, `app.jsx`, `tabs.jsx` (Costs/Burn/Merge headers), `data/merge_queue.py`, `test_app.py` (retire the 1440 pin), tests | **Leaf** — sketch #8, #9 first half; no header literal "30d" remains | δ2 |
| ε2 | Performance wall-clock cutoff + idle state; model-role bound semantics | `data/performance.py`, `data/model_role.py`, `tabs.jsx` (PerfTab), tests | **Leaf** — a project with no completions in the window renders `stale` with last-completion age; hourly sparklines and cards share one cutoff | ε1 |
| ζ | Merge attempts as one datum; "In queue now" as live datum with sampled history; `task_lookup` datum (snapshot rows + per-id miss path) replaces `load_task_titles` | `data/merge_queue.py`, `data/metrics.py`, `data/task_lookup.py`, `data/tasks.py` (per-id `get_task` wrapper), `data/redux_api.py`, `api/merge_queue.py`, `loops.py`, `tabs.jsx` (MergeTab), tests | **Leaf** — sketch #9 second half; probe failure renders the tile `stale`; against a fixture snapshot with one active and one terminal id, `task_lookup` serves the first from rows and the second via one `get_task` call, and `load_task_titles` is gone | ε2 |
| η | Escalations as one corpus walk with location views and one cache; complete resolution classes; escalation cards read `task_lookup` | `data/escalations.py`, `data/escalation_analytics.py`, `api/escalations.py`, `tab_escalations.jsx`, `tab_escalation_analytics.jsx`, tests | **Leaf** — sketch #10 against a fixture corpus (2 root + 3 archive pending); `stale-strand`/`moot-terminal-subject` appear in the split; `_load_task_cards` is gone and a card for a terminal task id still renders its row | ζ |
| θ | Memory ops one query; queue normaliser + offline state; recon Active-agents one rule; single lock classifier; Locks unknown when scheduler offline | `data/write_journal.py`, `data/redux_api.py::shape_memory`, `data/scheduler.py`, `scheduler_utils.jsx`, `scheduler_heatmap.jsx`, `tabs.jsx` (MemoryTab, ReconTab, LocksCell), `tab_overview.jsx`, tests | **Leaf** — sketch #11, #12 | η |
| ι | Integration gate: AST access-path check (with the healthz-probe exemption), old-path census, full boundary suite, cold-render measurement recorded | `dashboard/tests/test_datum_access_paths.py`, `tests/js/boundary_*.test.mjs`, completion note | **Leaf (gate)** — sketch #13–#15; the completion note carries the measurement 4795 reads | θ |

Companion at decompose: append a dated note to 4795 pointing at ι (its `deferred`
status makes a dep edge inert); wire `4416 depends_on γ3` with a dated note; no other
PRD prose needs editing beyond γ3's amendment of the taskgraph-legibility PRD.

## Adversarial review (2026-09-18, fresh opus reviewer at xhigh)

Verdict FIX FIRST; eight blocking findings, all folded in above: B1 view arithmetic
(decision 3, sketch #4); B2 status-map timeouts under contention (background table,
decisions 5/6/20, sketch #13); B3 one `as_of` over two reads (decision 5, snapshot
contract); B4/B5 duplicate client staleness authority (sketch item 8, decision 6,
registry contract); B6 unsatisfiable greps (decision 17); B7 4795 acceptance and the
un-costed terminal window (background table, decision 5 on-demand terminal rows,
sketch #15, G4 row); B8 `DEFAULT 0` migration (decision 9). Non-blocking N1
(infra-hold placement — adopted, flagged to Leo), N2 (synthetic #5), N3 (sizing —
δ/ε split, α2 added), N4 (parity test location), N5 (cache layering — decision 20),
N6 (state entry conditions — decision 6). Coverage gaps: the four
`task_status_counts.js` pins, the remaining `StatTile` call sites, the two surviving
whole-tree fetches, `fetch_statuses` paging, parity/forecast over carried rows, the
taskgraph-legibility premise, `data.js` defaults, sweep memory-recon#5 — each now
named in a leaf.

Second pass (same reviewer, on the amendment): B1–B8 confirmed closed; eight new
local findings folded in: C1/C2 escalation cards need whole rows and a 10-minute
titles datum would regress status freshness (decision 12 → `task_lookup`: snapshot
rows + per-id `get_task` miss path); C3 task 4416 owns the terminal-member contract
(decision 5 resolves it, G4 row, `4416 depends_on γ3`); C4 `?terminal=` had no client
owner (γ3 + γ1); C5 α2 omitted the two background loops and cited a line-count
threshold this repo forbids steering by (decision 18, α2 row); C6 the healthz probe
is a legitimate raw `fetch_tasks` caller (decision 17 exemption); C7 the burndown
row's two in-progress sources (decision 9 `in_progress_rows`); C8 no receipt anchor
for the age computation (decision 6 registry metadata). Non-blocking: wire `views`
vs `sub_views` split, TTL arithmetic in decision 20, a 2 s skew threshold, four leaf
signals rewritten as fixture-driven, the inert 4795 edge dropped for a note, and the
age-badge-wins rule.

Third pass: C1–C8 confirmed closed; one new finding D1 — the `task_lookup` miss path
was 2.5× the stated premise (104 ids / 53 active / 51 misses) and carried no budget
(decision 12 now budgets it in the `_load_task_cards` idiom with a concurrency width
and a miss cap); δ2's signal reworded as fixture-driven. META answered yes.

## Out of scope

- A per-PRD server census (needs a PRD field on the status map — a fused-memory
  change); PRD boxes stay `lower_bound`.
- Field projection on `get_tasks` (task 4390; 4795's remaining lever).
- Regenerating `recon_status.js` from `journal.py` (5320's runtime unknown bucket
  already catches drift).
- Dashboard supervision/watchdog and endpoint-freshness thresholds
  (`dashboard-availability-prd.md`, task 4884).
- Any change to fused-memory, escalation or orchestrator packages.
- Raising or retuning the request budgets; ι measures, 4795 decides.

## Open questions (tactical — decide at implementation)

1. **Freshness bounds per datum family.** Suggested: live snapshot 2× poll interval
   (6 s ⇒ 12 s); sampled data 2× sample interval (20 min); escalations 2× TTL (120 s);
   titles 2× metrics cadence (20 min). Retention bound (stale → unknown): 24 h.
   Decide in α/β.
2. **`Datum` implementation type.** Frozen dataclass with `to_wire()` (heuristic 8),
   validated once at shaping. Decide in α.
3. **`task_lookup` tuning**: per-id miss cache TTL (suggested 10 min for terminal
   ids), concurrency width (suggested 4), per-request miss cap (suggested 64) and
   the whole-operation deadline (suggested: reuse `_TASK_CARDS_BUDGET`). Decide in ζ
   against the measured 51-miss cold render.
4. **Stale-age badge format** (`3h` / `3h 12m`). Suggested: coarse humanised age via
   `window.DF_SHELL.timeago`. Decide in γ1.
5. **Recent-merges cap N.** Suggested 200. Decide in ε1.
6. **Whether the generator runs in pre-commit or only in the parity test.** Suggested:
   test only. Decide in α.
7. **`api/*.py` module boundaries** — one per endpoint family as listed, or fewer.
   Decide in α2 against heuristic 13.
