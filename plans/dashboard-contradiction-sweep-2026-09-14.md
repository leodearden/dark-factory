# Dashboard contradiction sweep — 2026-09-14

**Status**: findings + root causes, resolution NOT yet decided (discussion pending).
**Method**: 7 area-reader seats (1 opus/high, 6 sonnet) → 18 skeptic seats (sonnet/high,
3 findings each, prompted to refute) → lead spot-checks against code and a live API
capture. 46 findings: 39 confirmed, 5 amended, 2 refuted. Live JSON capture of every
`/api/v2/dashboard/*` endpoint taken 10:49–10:52 BST from the running dashboard on
port 8080; the seats' full structured output is in the session scratchpad
(`sweep.json`, `findings.md`).

**Reported instance** (confirmed live): Orchestrators › dark-factory Progress card
"0/1 · 0 done 0 active 0 blocked 0 pending" beside a filter bar "Active · 33 ·
Pending · 958 · Complete · 3861". The card reads `o.summary` from
`/api/v2/dashboard/orchestrators` (`data/orchestrator.py::discover_orchestrators`,
its own whole-tree `fetch_tasks` under its own 20 s budget; on `ReadTimeout` it emits
`offline=True` with an ALL-ZERO summary, which `tabs.jsx::OrchTab` renders through
`total || 1` as "0/1"). The bar counts `DF.ACTIVE_TASKS` + `DF.DONE_COUNTS` from
`/api/v2/dashboard/tasks` (`data/active_tasks.py::collect_tasks_with_counts`, a
different fetch with different budgets and two caches). The two endpoints are polled,
cached and backed off independently (`static/redux/data.js::refreshOne`), so one
render routinely mixes their answers. The "blocked in active" split is a fifth
bucket rule (see RC2).

## Seven root causes

Findings are cited as `area#n` (ids in `findings.md`). Heuristics are cited by name
from `docs/code-quality.md`.

### RC1 — Three independent censuses of one task tree (and the same shape elsewhere)

"How many tasks does project P have, by status" is answered by three mechanisms that
share no cache, budget, instant or failure encoding:

| Mechanism | Fetch | Budget | Failure encoding | Consumers |
|---|---|---|---|---|
| `orchestrator.py::discover_orchestrators` | live whole-tree `fetch_tasks` per running process | 20 s total / 7 s per root | `offline=True` + zero summary (HTTP 200) | OrchTab card, pips, 4 ST tiles; Overview tiles + "Task pipeline"; topbar "tasks active" |
| `active_tasks.py::collect_tasks_with_counts` | status-narrowed `fetch_tasks` + `fetch_statuses` (two caches, up to ~15 s apart, plumbing#2) | 20 s total / per-root | four disjoint lists (`OFFLINE`, `DEGRADED`, `COUNT_UNKNOWN`, `PROJECT_COUNT`), no `DONE_COUNTS` entry for a degraded root | OrchTab filter bar + rows; Tasks tab; rail "Tasks" badge |
| `burndown.py::collect_snapshot` (background, 10 min) | whole-tree `fetch_tasks` into SQLite | 75 s fan-out backstop only (plumbing#3 amended) | **none** — a failed cycle writes no row (`burndown.py:471 continue`), so a stale row is served as current (plumbing#7) | BurnTab, every sparkline on OrchTab/Overview tiles |

Live: orchestrators says dark-factory and reify are offline/zero; tasks says
dark-factory offline but reify has 1415 rows and 5528 done; burndown's reify row is two
days old (done 5484). The sampler almost never succeeds for the two big trees (16 and 7
rows in a month vs ~373 for every small project) because it fetches the whole tree
instead of the ~95 %-smaller status map.

Same class elsewhere: merge "In queue now" value (live MCP) vs its spark (10-min
`merge_snapshots` sampler) — 8 vs 2 live (merge-scheduler#2); escalations tab
(request-fresh) vs escalation-analytics (60 s TTL) on one tab (escalations#2);
Orchestrators "offline" vs Merge tab "Running" for the same project, which are in fact
two unrelated backends — the shared fused-memory task tree vs the project's own
escalation MCP — with nothing on screen saying so (merge-scheduler#4 amended).
Findings: plumbing#1, #2, #8, task-census#2, #11, costs-perf-burndown#1,
overview-shell#1. Heuristics: **SPOT**, **prefer stateless interactions**
(three private caches for one fact), **well-defined purpose**.

### RC2 — No owned status→bucket mapping

`shared/src/shared/task_statuses.py::TaskStatus` is the closed nine-member vocabulary
(pending, in-progress, blocked, deferred, review, merge-deferred, infra-hold, done,
cancelled). **No dashboard module imports it.** Instead there are at least seven
hand-rolled bucketers, each answering a slightly different question:

| Site | Rule | Consequence |
|---|---|---|
| `orchestrator.py` summary | four literals + `total = len(tasks)` | denominator counts 9 statuses, segments cover 4: every live project's bar has a bare gap (autopilot-video 37/653; Overview percentages sum to 96 %) (task-census#5) |
| `active_tasks.py::_ACTIVE_STATUSES` | 5 literals, no `review`/`infra-hold` | forwarded as the server-side filter → those statuses are **unfetchable**, invisible on the Tasks tab (task-census#4) |
| `burndown.py::_STATUS_MAP` | partial dict, `.get(…, 'pending')` default | `merge-deferred` silently counted as pending: 960 + 10 = 970 live (task-census#3) |
| `burndown._ZONE_KEYS` (6) → `redux_api._BURNDOWN_KEYS` (a different 6) → `burndown_bands.js` (5) | three key tuples | "Status mix" chart drops cancelled+deferred: 590 of dark-factory's 5454 tasks (task-census#6) |
| `task_status_counts.js` | running/blocked/mergeDeferred/pending/done | cap-comparable by design (task 3516) |
| `prd_grouping.js` | merge-deferred→in-progress, deferred→pending | PRD-finished question, defended in comments as a different question |
| `tabs.jsx::OrchTab` filter | active = in-progress+blocked; pending; done | merge-deferred / deferred / cancelled countable by nothing and listable by nothing |
| `app.jsx` | topbar: in_progress+blocked from ORCHESTRATORS; rail badge: in-progress+blocked+pending from ACTIVE_TASKS | 22 vs 2181 on screen together on every tab (task-census#2, overview-shell#1) |

"Active" therefore names three populations across five surfaces and "in-flight" a
fourth (task-census#10). The same pattern outside tasks: recon run status passed
through raw and guessed three ways (memory-recon#1, owned by task 5320); scheduler
heatmap vs Locks chip precedence (merge-scheduler#6); escalation terminal classes
(`stale-strand`, `moot-terminal-subject`) counted in totals but never in the
benign/actionable split (escalations#3 amended). Heuristics: **SPOT**,
**structured data instead of meaningful strings**, **clear invariants uniformly
enforced** (nothing asserts parts sum to whole).

### RC3 — UNKNOWN has no representation on the wire or in the UI

A count is a bare number, so "unmeasured" can only be encoded as 0, as absence, or as a
side-channel flag. Every producer chose differently and most consumers read none of them:

- `discover_orchestrators` emits `offline` + a verbatim "UNKNOWN (not zero)" `error`;
  **no JSX reads either field** for an orchestrator entry (grep confirmed). All-zero
  summary rendered as fact; `total || 1` manufactures "0/1" (task-census#1). A project
  with no running process gets no entry at all — a third convention (plumbing#5).
- `api_tasks` publishes four disjoint failure lists; `tab_tasks.jsx` uses three for
  banner copy and only `TASKS_COUNT_UNKNOWN_PROJECTS` to suppress the number, so seven
  of nine projects on the capture rendered "0 in-progress · 0 pending · 0 done" under a
  banner saying the view was partial (task-census#9). OrchTab's "Complete" silently
  falls back to the capped row tally with no marker at all (task-census#8).
- Burndown has no staleness/degradation vocabulary of any kind (plumbing#7).
- Scheduler tab shows an offline banner, but the Locks chip on OrchTab/Tasks does
  `rows || []` and renders "no locks" for the same offline project (merge-scheduler#3).
- `shape_memory`'s offline branch drops the queue key-defaulting the online branch
  applies; `queue.offline` is emitted and never read (memory-recon#2, #3).
- Overview never checks `o.offline`; "7 / 7 running" while two are unreachable; divides
  by an unguarded `tasksTotal` → `NaN%` when all are offline (overview-shell#2).
- `collect_done_counts` is a dead second implementation without the degradation
  channel, with the more inviting name (task-census#13).

Heuristics: **clear invariants informatively enforced**, **structured data**.

### RC4 — Stat tile = value + spark of unrelated provenance

`ST`/`StatTile` take `value` and `spark` as independent props. Callers wire a live
cross-project aggregate to one and a 30 d burndown series (different endpoint,
population, instant, and ignoring the project filter) to the other: OrchTab's three
tiles, Overview "Active tasks", BurnTab "Completed", merge "In queue now", recon
"Active agents" hint vs value (task-census#11, costs-perf-burndown#1,
merge-scheduler#2, memory-recon#5). Heuristic: **coherent narrow interfaces**.

### RC5 — Aggregate over ragged snapshots

`redux_api.py::shape_burndown` sums per-project series onto the union of all labels,
0-initialised, without carrying a project's last value forward — deliberately, pinned by
a test, to avoid a worse cross-project timestamp conflation. Because the two big
projects rarely snapshot, the aggregate is a sawtooth: live `BURNDOWN.pending[-1] = 1`
directly above a per-project table reading 963 and 1097; `done` ranges 1462..10767
in the window. The "Backlog", "Forecast clear", "Net velocity" tiles all read
`series[-1]` of this (task-census#7, costs-perf-burndown#2).

### RC6 — Window and time semantics not threaded through

- The window chip is one app-level state shared by four endpoints with different valid
  sets (`_WINDOW_DAYS` lacks `90d`; burndown lacks `all`): switch tabs and no chip is
  highlighted while the server silently serves its default (costs-perf-burndown#3).
- Panel headers hardcode "30d" on Costs, Burndown and OrchTab regardless of the served
  window (costs-perf-burndown#4).
- Merge "Recent merges" is pinned to 24 h beside windowed stats, by test
  (merge-scheduler#5).
- Performance cards anchor "7-day" to the project's own last `completed_at`; the
  adjacent sparklines anchor to wall clock (costs-perf-burndown#5).
- model×role rollup: dashboard uses open bounds, orchestrator digest closed
  (costs-perf-burndown#6, acknowledged in its docstring).

### RC7 — Two queries for one headline

- Merge "Merges (window)" = `_get_durations` (duration > 0 only) vs donut total =
  `outcome_distribution` (all attempts): 167 vs 246 live for dark-factory
  (merge-scheduler#1).
- Overview "reads + writes" (`GROUP BY kind`, filtered to two kinds) vs Memory donut
  (`GROUP BY operation`): 319 011 vs 320 670 live (overview-shell#4).
- Escalations "pending": root-only live scan (38 for reify) vs archive-inclusive
  analytics walk (252) on the same tab (escalations#1).
- OrchTab "Completed / day" divides by day-to-day transitions (n−1) while the server's
  velocity divides by days (n): 6 vs 7 (overview-shell#3 amended).
- PrdBox "n/m done" re-derived over the capped row window beside the project header's
  authoritative `DONE_COUNTS` (task-census#12).

## Refuted / amended (for the record)

- **Refuted** merge-scheduler#7 (outcome policy split): `train_throughput` is wire-only,
  no surface renders it. **Refuted** memory-recon#4 ("pending" collision): the Memory
  tile is labelled "Write queue"; the word never renders.
- **Amended** overview-shell#3 (fencepost, not unused server computation),
  merge-scheduler#4 (two unrelated backends, not two probes of one target),
  escalations#3 (extra terminal classes + `resolved_at` filter, not a dropped-None
  branch), plumbing#3 (75 s backstop bounds it; staleness is from silently skipped
  cycles), plumbing#6 (real boot-time `escalation_urls` staleness, but no same-value
  sibling).

## Prior owners

3516 (split the merged "active" pip), 1564/1601 (done and deferred rows dropped at
collection), 1814 (signal layer: faults not zeros), 3517 (runtime probe offline
distinction), 4795 (deferred: tasks fetch budget cannot cover 9 roots), 5320
(in-progress: recon status vocabulary). Each fixed one slice of one surface; none
addressed the class, and several of the fixes are the "one surface hardened, siblings
not" pattern the sweep keeps finding (task-census#8, #9, overview-shell#2).
`plans/dashboard-alignment-prd.md` (2026-07-06) unified the MCP fan-out helper and the
outcome vocabulary but did not touch censuses or status buckets.

## Resolution options (to discuss)

Ordered by how many root causes each retires. A and B together cover RC1–RC3, which
account for ~60 % of findings and the reported instance.

**A. One server-side census, one bucket table.** A single `TaskCensus` per project
{counts over all nine `TaskStatus` members, total, as_of, provenance/degradation state}
produced once (from the cheap `get_statuses` map, not the whole tree) and consumed by
the orchestrators shape, the tasks shape, the burndown sampler and the app chrome.
`orchestrator.py`'s private summary, `app.jsx`'s two reductions and
`collect_done_counts` go away. The status→bucket rule lives once in Python, derived from
`shared.task_statuses` (import it), and either ships to the client as pre-bucketed
counts or is generated into one JS module that replaces the three bucketers. The two
"different questions" (cap-comparable vs PRD-finished) become two named views of the
same table, not two tables.

**B. Unknown as a type.** Census fields on the wire carry
`{value, as_of, state: fresh | stale | lower_bound | unknown}`; the tile/card
components refuse to render a bare number for `unknown` (em-dash + reason) and mark
`lower_bound`/`stale`. A grep-guard test forbids `|| 1`, `|| []`, `?? 0` fallbacks on
census fields, the way the alignment PRD's grep-guard forbids bare `datetime.now`.
Retires the entire "confident zero" family and the `0/1`.

**C. Provenance-bound stat tile.** `ST`/`StatTile` take one series object carrying
value, spark and as_of from the same key, or assert the two props share provenance.
Retires RC4 mechanically.

**D. Coherent poll.** Either one combined census endpoint (removes the largest
instance of RC1 outright) or an `as_of` stamp on every payload with the max skew shown
in the topbar. The independent per-endpoint backoff is defensible; the invisibility of
the skew is not.

**E. Burndown sampler + aggregate.** Sample from `get_statuses` so the big trees
actually snapshot; write an explicit gap row on failure instead of nothing; carry-last
per project in the aggregate (or drop the aggregate tiles in favour of per-project);
emit per-project staleness age. Retires RC5 and burndown's share of RC3.

**F. Window threading.** Validate the chip against the destination tab's set on tab
switch; labels read the served window from the payload; "Recent merges" and the
performance anchors either follow the window or say on screen that they don't.
Retires RC6.

**G. Query reconciliation.** For each RC7 pair pick one population and assert the other
reconciles to it (or label the denominator). Small, independent, low risk.

Quick wins that fall out regardless of the big decision: import `TaskStatus` and derive
`_ACTIVE_STATUSES` from `shared.task_statuses.ACTIVE`; give `_STATUS_MAP` no default;
read `o.offline` in OrchTab/Overview and drop `|| 1`; gate the Tasks-tab em-dash on all
three degradation lists; delete `collect_done_counts`.

## Not covered

Seats time-boxed: `memory_evals.py` escalation-fingerprint join (~L1150–1340),
`spark_path.js`/`runtime_format.js` internals, `esc_flow_layout.js` in full, and
several test files scanned by grep rather than read (`test_scheduler_page.py`,
`test_merge_queue_data.py`, `test_escalations_data.py`). Playwright was disconnected
for the whole sweep, so no rendered-DOM confirmation; every rendering claim is from
JSX reading plus the live JSON.
