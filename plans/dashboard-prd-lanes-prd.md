# Dashboard PRD view: lanes by what each PRD needs, honest progress, unowned work

**Status:** authored 2026-10-09 (`/team` + `/prd` author mode). **Decomposed 2026-10-09:**
α 6581, β 6582, γ1 6583, γ2 6584, δ 6585, ε1 6586, ε2 6588, ε3 6590, ε4 6591, ζ 6592; follow-up
6593 (repoint `Scheduler._milestone_time_gated`). Bindings, decompose decisions and the fresh
review: `dashboard-prd-lanes-prd.capability-manifest.md` and its YAML sidecar.
**Type:** extension of two shipped dashboard PRDs. It replaces the Tasks tab's PRD box grid
with one server-computed view, adds one fused-memory read parameter, one optional
capability-manifest field and one method on the shared milestone model.
**Approach:** B+H. G5 applies on several counts: four packages (fused-memory, shared,
dashboard, the in-repo `/prd` skill), about ten mechanisms, and three cross-PRD seams.
**Code anchors** verified against main `99ad2d7ddb` (2026-10-09). Cite-by-symbol; re-locate
at implementation time.
**Approval (Leo, 2026-10-07):** "All your other proposals, LGTM", given after reading the
2026-10-06 status review (artifact `2YipixJ2gEUCMoKQxfa8p9`, whose top section prototypes
this view). The authoring brief is `~/.claude/spawn-briefs/prd-dashboard-prd-lanes-2026-10-09.md`.
§3 records where this PRD departs from the brief's wording and why.

## 1. Goal

Leo, on 2026-10-06: "the dashboard view is fairly overwhelming. There are so many part-done
PRDs that they don't all fit on the screen and I'm struggling to understand which ones are
important or how far through they actually are in practice."

When this lands, opening the Tasks tab shows, for each project:

- **One contention line** at the top. It names the most-wanted held files among ready tasks,
  who holds each one, and how many ready tasks want it. It also counts the ready high-priority
  tasks that have never started.
- **Every PRD with open work, in exactly one of five lanes**, ordered by what the PRD needs:
  *Needs a ruling*, *Moving*, *Stuck*, *Parked*, *Tail*. Tail is collapsed to one line with a
  count and the PRD names.
- **One row per PRD.** Each row has a segmented bar over all its attributed tasks, an exact
  `done/total`, a signal mark (delivered / partial / not yet / unknown), `ready n · oldest Nd`,
  and a next-action chip. A row expands to its open tasks, drawn by the existing task graph.
- **A dark-matter strip** for the open tasks that no PRD owns: one line of counts, which
  expands to a list.

The server computes every figure on the page, as part of a `Datum` the page renders. Each
figure carries its rule text on hover. The browser never counts, buckets or classifies.

The view answers "which PRDs need me, and how far through is each" from recorded facts. Where
a fact is not recorded, the view says so. It never guesses.

## 2. Background (measured 2026-10-09, read-only, live stores)

Population: the dark_factory store at about 12:20Z had 6,530 tasks. Of these, 1,659 were
non-terminal: pending 1,292, deferred 336, in-progress 25, blocked 6, review 0, plus 2
merge-deferred outside that rule. The store moves under a reader: a later read about 20
minutes on found 52 deferred tasks cancelled. Every figure here is "at authoring" and carries
its rule.

### 2.1 Attribution: the PRD boxes see a sixth of the work

- **What the boxes group by today.** The Tasks tab groups rows by each task's *own* PRD field.
  The server coalesces that field (`dashboard/src/dashboard/data/active_tasks.py::_coalesce_prd`:
  `prd_path`, then `prd`, then `prd_ref`, with `#…`/`§…` stripped) and the browser buckets
  rows on it (`dashboard/src/dashboard/static/redux/prd_grouping.js::groupTasksByPrd`).
- **Coverage.** Of the 1,659 non-terminal tasks, 255 carry their own field.
  - Another 313 reach a PRD only by following `spawned_from`, then `x_recovered_from_task`,
    then `cross_repo_refile_of`, across all tasks.
  - Hops: 190 at 1, 69 at 2, 36 at 3, 14 at 4, 2 at 5, 2 at 6.
  - Over the whole store: no cycles, no task with more than one link key, no task with more
    than one PRD field.
  - The other 1,091 reach no PRD. The 2026-10-06 count was 266 / 293 / 1,104 of 1,663.
- **Chains run through finished tasks.** 290 of the 313 chains pass through a done or
  cancelled task, and 289 end at a terminal PRD holder.
  - The default render fetches metadata only for active rows
    (`dashboard/src/dashboard/data/task_snapshot.py::_read_unit`: `get_tasks` over
    `shared.task_statuses.ACTIVE`, plus the status-only `get_statuses` map).
  - A walk limited to that set keeps only 23 of 294 chain attributions. It loses 271 (92%),
    which is half of all 547 attributions. That is a seat's later read of 1,609 active tasks:
    253 direct and 294 chain.
  - The on-demand 400-row terminal window (`task_snapshot.py::_TERMINAL_FETCH_WINDOW`) rescues
    52 of the 271.
  - A full terminal read is about 40 MB per refresh for this project. It also breaks the
    per-root budget arithmetic in `dashboard/tests/test_tasks_budget.py::test_tasks_budget_is_structurally_deliverable`
    (4.4 s × 4 calls > 14.0 s).
- **Why the 1,091 reach nothing.** 363 carry no link at all, and 682 follow links to a root
  with no PRD. 46 hit a link that cannot be followed:
  - 28 point into another project (`'reify:5569'`, `cross_repo_refile_of`);
  - 11 are unparseable (`'df_task_2921'`, dict-shaped `spawned_from`, `'cand-NNNN'`);
  - 7 point at an id missing from the store.

  Link values are digit strings in 99% of cases: `spawned_from` is a digit string on 1,885
  of 1,918 tasks and an int on 2.
- **PRD keys.** 77 distinct PRD keys have open work: 50 reached by a direct member, 27 only
  through chains. None is spelled two ways, and every one is a tracked file on main.

### 2.2 Progress: the box number answers the wrong question

- **What the box shows.** It shows `≥ n/m terminal`, a `lower_bound` over the rows the
  browser holds (`plans/dashboard-one-datum-one-path-prd.md` decision 8). Every PRD gets the
  same box, ordered by a cross-PRD dependency mini-DAG (`prd_grouping.js::orderPrdGroups`).
- **The prototype's verdicts were judgments.** The 2026-10-06 study's lanes came from LLM
  seats that judged each PRD's `state` and `action`; the prototype's `lane()` reads those
  labels. This PRD has to turn each lane into a rule over recorded facts.
- **Ready.** No ready flag is stored. With the approximation of §4.5, 1,062 of 1,292 pending
  tasks are ready, 127 of them high priority.
  - `metadata.pending_since` anchors 1,058 of the 1,062.
  - All 266 ready tasks pending for 30 days or more have `pending_since_backfilled` set: the
    v4→v5 migration copied `updated_at`, so each such age is a lower bound.
- **Moving.** `tasks.updated_at` is not a completion time: of 523 done tasks with a recent
  `updated_at`, 367 have a completion event in the window. `data/orchestrator/runs.db::events`
  is the reliable source:
  - `task_started` (9,954 rows over 4,357 tasks since 2026-04-09);
  - `task_completed`;
  - `task_results.completed_at`.
- **Needs a ruling.** 27 open tasks are human gates (`task_kind='deterministic'` with
  `always_escalates`). 12 pending level-2 escalations name 10 distinct tasks.
  `metadata.escalation_id` is the origin escalation, not a live pointer.
- **Parked.** No deferred task carries `metadata.deferral` today (0 of 336). The record
  arrives with `plans/deferral-flip-condition-prd.md` (6524 in progress, 6525–6529 pending).
- **Signal.** Nothing machine-readable records which task carries a PRD's user-observable
  signal.
  - The capability-manifest sidecar schema has no such field
    (`shared/src/shared/capability_manifest.py::CapabilityManifestDoc`, `extra='forbid'`).
  - All 50 directly-attributed PRDs with open work have a sidecar with stamped ids; 72
    sidecars are tracked in all.
  - Deriving "the leaf" from the dependency graph gives a unique answer for only 19 of the
    72.
- **The five lanes as briefed leave 10 of the 77 PRDs in no lane.** These PRDs have fresh
  ready work, nothing started in 14 days, and no gate.
- **The rules of §4.8 fit the live data.** The critic seat simulated the draft rules at about 14:30Z,
  over 74 open-work PRDs (the set had moved), before any signal backfill. It found Moving 45,
  Stuck 26, Ruling 3, Parked 0, Tail 0.
  - 13 of the Stuck PRDs have a ready member pending 30 days or more, all on backfilled
    anchors.
  - Moving is large because lineage follow-ups keep landed PRDs moving. Tail fills as ε names
    signal leaves.

### 2.3 Dispatch: lock contention, not priority

- **Hub files.** Among 1,072 ready tasks (local-dependency rule), the most-wanted files are:
  - `orchestrator/src/orchestrator/workflow.py`, by 173 (held by 5730);
  - `harness.py`, by 148 (held by 4618);
  - `fused-memory/src/fused_memory/server/tools.py`, by 94 (held by 5276);
  - `OPERATIONS.md`, by 82 (no holder).

  On 2026-10-06 the first two were 167 and 143.
- **Who names a held file.** All 127 ready high tasks name at least one currently held file,
  against 56% of ready low tasks. 58 of the 127 have never produced a `task_started` event.
- **Locks.** The lock is per file at this project's `lock_depth: 12`
  (`shared/src/shared/locking.py::files_to_modules`, `modules_conflict`). Holders are in
  `scheduler_state.json::current_holders`. The dashboard reads that file through MCP
  `get_scheduler_state` (`dashboard/src/dashboard/data/scheduler.py::collect_scheduler_state`).
- **The Scheduler tab already counts contention**
  (`data/scheduler.py::_module_contention_counts`), but over all active rows, not the ready
  set.

## 3. Rulings executed, and departures from the brief's wording

Leo accepted the brief's design 1–6 on 2026-10-07. This PRD executes them. Where the brief's
wording would contradict a measured fact or a design invariant, the PRD departs as follows.

| # | Brief said | This PRD | Why |
|---|---|---|---|
| R1 | `signal_delivered` "written by the done-callback of the leaf task(s) the manifest marks" | The sidecar gains an optional `signal_leaves` (labels). The dashboard **derives** the mark at read time from those leaves' task status. No runtime writer. | INV-9 `one-fact-one-home`. A written copy of the leaf's status would need invalidating on reopen, cancel and curator combine. Its producer would also fire only on future transitions, so the 23 sidecars whose leaves are already done would read "no record" forever (INV-13). The fact being recorded is *which leaf proves the signal*. Its home is the sidecar, authored with the PRD. "Delivered" is then rendered from the store, never inferred from done counts. This keeps the brief's intent: recorded, never a guess, minimal write side. |
| R2 | Stuck = "ready work untouched ≥30 days, or everything remaining gated" | Stuck is the **residual lane**: open work, nothing started or finished in 14 days, no ruling pending, not all held. The 30-day bar becomes the red age highlight inside the row. | The briefed rule leaves 10 of 77 PRDs in no lane (§2.2). The approved prototype's own `lane()` already made Stuck the fallthrough. A partition is checkable (§5.5); a lane set with holes is not. |
| R3 | Lane order as listed: ruling, moving, stuck, parked, tail | Precedence for assignment: **ruling > tail > parked > moving > stuck**. Display order stays ruling, moving, stuck, parked, tail. | A lane says what the PRD needs from Leo. A delivered PRD whose follow-ups are busy needs nothing, so it belongs in Tail. The skeptic's caveat (lineage is not topic) is met by §4.8's "high work under collapsed Tail PRDs" line, not by keeping tail PRDs out of Tail. |
| R4 | Attention score uses "stability relevance" | Order within a lane by open high/critical count, then oldest-ready age, then ready count, then key. | Stability relevance was an LLM label in the 2026-10-06 scratchpad (`classify_out_*.jsonl`); the store records nothing like it. Heuristic 12: no figure from a guess. |
| R5 | Dark-matter strip counts "how many sit in a ranked briefing concern" | Dropped. The strip counts by status, priority, source and **why attribution failed**. | `review/briefing.yaml::stability_concerns` has list position only: no ids, patterns or task mapping. The 2026-10-06 figure was an LLM label. G3 resolution (a): rewrite to an existing capability. A recorded concern id on tasks is out of scope (§10). |
| R6 | Row has a "theme/state line" | A deterministic **state line**: the lane's own reason, with figures, e.g. "3 ready, oldest pending ≥43 d; nothing started in 14 d". | Theme was an LLM label too. |
| R7 | Palette `#2a78d6,#eb6834,…` with light and dark steps | The existing dashboard tones (`charts.jsx::PALETTE`; `styles.css` `:root`). | The dashboard has one dark theme and no light theme. One palette home (INV-5). |
| R8 | "done/cancelled rows are bounded buckets (≤50/project)" | Premise retired. Terminal rows are a 400-row on-demand window today (§2.1). | Measured. |
| R9 | Needs a ruling includes "disposition needs_ruling" | Dropped. Ruling is an actionable human gate or a pending level-2 escalation (§4.8). | `needs_ruling` was a seat's disposition label in the 2026-10-06 study; nothing records it. |
| R10 | "ready high tasks never scheduled (no runs.db event)" | "never started": no `task_started` event (58 of 127 at authoring, against 54 with no event of any type). | "Any event" needs a full scan of a 458k-row table on every refresh. `task_started` is served by the `event_type` index, and the figure says exactly what it measures: these tasks never started (some were skipped, i.e. considered). The rule text says which. |
| R11 | Row expands to "purpose" | The PRD's H1 title, shown on the row, not in the expansion. | Anything past the title means parsing PRD prose (heuristic 12). |

None of these needs a fresh ruling: R1–R3 serve the approved intent, and R4–R11 are forced
by G3/G6 or by cost.

## 4. Resolved design decisions

### 4.1 Attribution is computed in the dashboard backend over the whole store

- **The rule.** A task's PRD is its own PRD field. Failing that, it is the first PRD field
  reached by following its link. A link is the first present of `spawned_from`,
  `x_recovered_from_task` or `cross_repo_refile_of`, within the same project.
- **Bounds.** At most `MAX_HOPS = 8` (measured max 6) and cycle-safe.
- **Serving.** For each open task the server serves its attribution through the view:
  - a PRD row's `direct_open_ids` and `followup_open` (`[{task_id, hops}]`);
  - the dark-matter strip's `members` lists, keyed by an `unresolved` reason drawn from a
    closed enum (§5.4), never a bare `None` (INV-11).

  Open tasks are all in the active rows the client already holds, so the client renders by id
  and never re-attributes.
- **One parser.** Link values are parsed into a typed `Link` once, at the read boundary
  (heuristic 12). A digit string or an int becomes a local id. `'project:id'` becomes
  `cross_project_link`. Anything else becomes `unparseable_link`.
- **Cross-project links are not followed.** The 28 such tasks land in the dark-matter strip
  under that reason. Following them would attribute a dark_factory task to a reify PRD inside
  dark_factory's group, a cross-project join with no consumer.
- **Precedence.** The orders `prd_path > prd > prd_ref` and
  `spawned_from > x_recovered_from_task > cross_repo_refile_of` are inert today (no task has
  two keys). They stay deterministic, and the view serves a `multi_link` count so the day the
  order starts to matter is visible.
- **Lineage, not topic.** Attribution follows lineage. The skeptic found unrelated high bugs
  (3962, 5527, 5529) under unrelated PRDs. So every row shows direct and follow-up counts
  separately, and §4.8's line keeps high work visible.

### 4.2 The lineage index: one compact projection read, its own datum

- **A new datum family.** Attribution needs the link and PRD fields of every task, terminal
  ones included. They come from a new datum, the task lineage index
  (`dashboard/src/dashboard/data/task_lineage.py`).
- **The read.** One **unpaged** `get_tasks(fields=[…])` call per project root returns only
  `id`, the three PRD fields, the three link fields and `done_provenance`.
  - That is about 150 bytes a row: about 1 MB for dark_factory and about 1.3 MB for reify
    (8,383 tasks).
  - The server builds the whole list in memory either way (`server/tools.py::get_tasks`
    slices pages after the read). The dashboard reads it over plain HTTP, which has no
    response cap. So one call replaces a paged walk, and the dead `fetch_tasks` page walk that
    task 5924 deletes is not needed.
  - The read is typed: projected rows never pass through `data/tasks.py::_shape_task`, which
    would fabricate `title=''` and `status=None`.
- **Cadence.** It has its own cache and last-good value, with a TTL of minutes, not seconds
  (open question 1). Link and PRD fields on finished tasks almost never change, and the rows
  below cover fresh ones.
- **Never blocks a request past its deadline.** The `/prds` handler has a whole-handler
  deadline and a per-root concurrency cap, mirroring `/tasks`
  (`active_tasks.py::_serve_roots_within_budget`, `_TASKS_ROOT_CONCURRENCY`; open question 1
  sets the figures).
  - A lineage entry that is cold or expired is refreshed by a single-flight background task.
  - A request that reaches its deadline first serves the last-good value as `stale`, or
    `unknown` before any success, and the refresh carries on for the next poll.
  - The 15 s snapshot unit's budget and `test_tasks_budget.py` do not change. γ1 adds a
    budget test for `/prds` of the same shape.
- **Population and status.** The view's population is the keys of the snapshot unit's
  `status_map`, the read the census uses, and every per-PRD count reads status from it.
  - Lineage entries and active rows are used only as attribution facts and as chain hops.
  - At view time the raw active rows (15 s, full metadata) override lineage entries for the
    same ids, so a new task is attributed from its own row at once.
  - A status-map id in neither the lineage index nor the active rows is `lineage_pending`.
    It is counted in the dark-matter strip if open, and in `unattributed_done` if done, never
    dropped.
- **The census identity.** The per-project identity of §5.5 holds by construction against
  `census.build_census(status_map)` computed over that same map:
  `Σ PRD totals + closed + unattributed = census total − cancelled`.
- **The parameter.** fused-memory's `get_tasks` gains `fields` (α). It is the "field
  projection" part of pending task 4390, built here against a named consumer. 4390 keeps its
  other parts (recency `order_by`/`limit`, the `ids` filter) and is re-wired to depend on α
  (§8).

### 4.3 The view is one pure derivation over six inputs

- **Placement.** `dashboard/src/dashboard/data/prd_view.py` is pure: no I/O, no clock reads
  except the `now` it is given. It builds a per-project `PrdView` from six inputs, each a
  `Datum` with one access implementation:
  1. the snapshot unit (status map and raw active rows). It is **required**: an empty or
     failed status-map half makes the view `unknown` with that half's own reason;
  2. the lineage index (§4.2);
  3. the escalation corpus (`data/escalation_corpus.py::acquire_corpus`, the one corpus walk
     of one-datum decision 13), **scoped to this project's queue** with
     `escalation_corpus.py::_scope_provenance`, as `views_over` does. The corpus as a whole is
     `lower_bound` whenever any configured root lacks `data/escalations/`, and two live roots
     do;
  4. the task activity datum (§4.6);
  5. the signal records (§4.7);
  6. deferral holder liveness (§4.8, Parked).
- **Contention.** A second Datum, `Contention`, is built from the snapshot unit, the activity
  datum and the **scheduler state datum**.
  - γ2 extracts `get_scheduler_state`'s read out of
    `data/scheduler.py::collect_scheduler_state` into one access datum per project:
    `acquire_scheduler_state(root) -> Datum[SchedulerState]`, carrying `lock_depth`,
    `current_holders` and the file's own `snapshot_at`, with its own cache.
  - The Scheduler tab's collector and Contention both consume it: one access path, two
    readers (one-datum sketch 1).
  - Today the per-project `lock_depth`, raw holders and file `snapshot_at` are discarded
    inside `_one_project`. The cached `snapshot_at` there is the dashboard's fill time.
  - A Tasks-tab poll therefore never triggers the Scheduler tab's event fan-out.
- **Where it sits.** The pure module is layered under its readers. `api/prds.py` imports it.
  It imports `census`, `datum`, `prd_attribution`, `shared.task_statuses`, `shared.locking`,
  `shared.task_metadata` and `shared.task_deferral`. It never imports `active_tasks`
  (heuristic 13).
- **Composite provenance.** For each input, the view uses that input's state **in this
  project's scope**, judged against the input's own bound.
  - The composite is `fresh` only when every input is fresh.
  - Otherwise its state is the worst (unknown > stale > lower_bound > fresh), and `reason`
    names each non-fresh input with that input's own reason.
  - `as_of` is the oldest input's `as_of`, and `freshness_bound_seconds` is that same input's
    bound, so `validate_datum`'s fresh check (`served_at − as_of ≤ bound`) agrees with the
    per-input verdict.
- **Inputs are listed.** Each input's `{name, as_of, state, reason}` is listed in the value
  and drawn in the project footer (§4.11), so a hover can say *which* input is stale.
- **Failure modes.** A required input that has never been measured makes the view `unknown`
  with that reason. The tab then shows the reason in place of lanes and keeps the list mode
  working. No input failure becomes a 500 (`api/tasks.py::_unknown_after_contract_break`
  pattern).

### 4.4 One endpoint, two datums per project

- **The endpoint.** `GET /api/v2/dashboard/prds` (router `dashboard/src/dashboard/api/prds.py`)
  returns `{PRD_VIEW: {project: Datum[PrdView]}, PRD_CONTENTION: {project: Datum[Contention]}, served_at}`.
  - It is registered for the tasks tab only (`data.js` endpoint map; `endpoint_staleness.js::TAB_ENDPOINTS`).
  - The rest of the app is untouched: `/tasks` keeps its key set (`tests/test_app.py::_TASKS_KEYS`),
    and other tabs never poll `/prds`.
- **Why a separate endpoint.** A failure of the scheduler state datum degrades only
  `PRD_CONTENTION`.
  A lineage failure degrades only `PRD_VIEW`. The tasks payload's existing contract and
  five-site key pin stay intact.
- **Retired client code.** The by-PRD toggle no longer fires `?terminal=<project>`, because
  the server's counts are exact. The Complete filter still uses it. This supersedes one-datum
  decision 8 (`≥ n/m` lower bound) and that PRD's out-of-scope bullet "a per-PRD server
  census".

### 4.5 Segments and the ready predicate are defined once, in Python

- **Segments.** A PRD's members are its attributed tasks, excluding cancelled. Each member
  falls in exactly one segment:
  - `done`;
  - `in_progress`: in-progress, review, merge-deferred, infra-hold, i.e. the census
    `in_flight` view minus blocked;
  - `blocked`;
  - `deferred`;
  - `ready`;
  - `waiting`.
- **`ready`** means all of:
  - status pending;
  - every local dependency done or cancelled in the status map;
  - no `metadata.external_deps`;
  - no **unfired** milestone;
  - not a human gate.
- **External dependencies.** Any external dependency makes a task `waiting`. External
  statuses are resolved only inside `/tasks` row shaping (`fetch_external_statuses`). Five
  ready-looking tasks carried one at authoring. The rule text says the case is not
  evaluated.
- **Milestones.** A milestone stays in metadata after it fires (`docs/task-authoring.md` §6),
  so "has a milestone" is the wrong test. Firing is decided by a new pure method on the shared
  model, `shared.task_metadata.Milestone.fired(anchor, now)`:
  - `dated` fires when `now ≥ at`;
  - `delayed` fires when `metadata.milestone_deps_satisfied_at + after_secs ≤ now`, and has
    not fired while that anchor is unset;
  - a malformed spec never fires, as the scheduler fails safe.

  The model is the home of milestone semantics. `Scheduler._milestone_time_gated` keeps its
  own copy until a follow-up repoints it (decompose companion, §9). That copy is a duplication
  owned by that follow-up (INV-12), not by this PRD.
- **`waiting`** is pending and not ready: an unmet dependency, an external dependency, an
  unfired milestone, or a human gate. The word "waiting" is chosen over "waiting on deps"
  because milestones and gates are in it too (heuristic 1).
- **Disclosed limits.** The rule text says what `ready` does not evaluate: `delivered_checks`
  on dependencies, pins, parks, requeue cool-downs, external statuses. So it is an upper bound
  on what the scheduler will dispatch. The dashboard may not import the orchestrator
  (one-datum decision 19), so this predicate is a stated approximation, not a copy of
  `Scheduler._deps_satisfied`.
- **One home.** The predicate has one home, `prd_view.py::is_ready`. The contention line uses
  the same function.
- **Ready age.** It is the oldest ready member's pending age, from `metadata.pending_since`,
  shown as `oldest Nd`.
  - It is time pending, not time ready, and the rule text says so.
  - A backfilled anchor (`pending_since_backfilled`) makes the age a lower bound, shown `≥Nd`.
  - The age turns red at 30 days.
- **Tones.** Segment tones reuse existing palette keys (`done` ok, `in_progress` accent,
  `blocked` bad, `deferred` fg3, `ready` warn, `waiting` info). No new colour enters the
  palette.

### 4.6 Moving reads runs.db, not `updated_at`

- **The datum.** A new access module, `dashboard/src/dashboard/data/task_activity.py`, gives
  per project, from that project's `data/orchestrator/runs.db` (via
  `project_dbs.py::_project_scoped_dbs_labeled`, the existing roots walk, so task 5613's
  count of walks does not grow; read-only, off the event loop):
  - the latest `task_started` timestamp per task in the window;
  - the latest completion per task (`task_completed` with outcome done, or
    `task_results.completed_at`);
  - the set of task ids with any `task_started` ever, plus the table's first event timestamp
    (the window start).
- **Cost.** Each query is bounded by the `event_type` index. A full scan of the 458k-row table
  is never needed: "never started" means no `task_started`, not "no event of any kind".
- **Fallback.** A done task with no completion event (hand-carried, gate-closed) counts as
  recent only through `done_provenance.stamped_at`, from the lineage projection. Without that
  stamp it does not count, and the rule text says so.

### 4.7 The signal mark is read from the sidecar and rendered from the store

- **The field.** `CapabilityManifestDoc` gains `signal_leaves: list[str] | None`: labels of
  the sidecar's own `tasks`, naming the task or tasks whose completion delivers the PRD's
  G1 signal, usually the integration gate.
  - A model validator requires every label to exist and the list, when present, to be
    non-empty.
  - `/prd` decompose authors it (decompose-mode Step 2.5).
  - The `commit_planning` stamper rewrites the raw dict, so a declared field survives
    re-stamping.
- **One path rule.** `shared.capability_manifest.sidecar_path(prd_key)` becomes the single
  home of the `.md` → `.capability-manifest.yaml` rule for the stamper and the dashboard. The
  four other restatements (`scripts/audit_*`, `delivered_check_polarity.py`) are left alone,
  per §10.
- **The reader.** `dashboard/src/dashboard/data/prd_signal.py` loads each open-work PRD's
  sidecar with `load_capability_manifest`.
  - It reads off the event loop, cached by path and mtime, bounded by the number of PRDs with
    open work (77 here).
  - It resolves each label to its bound task, and **cross-checks that the bound task's own PRD
    field is this PRD**. That catches the foreign ids in `task_id` measured on
    `plans/load-throttle-harmonisation-prd` ρ1–ρ3 and `plans/os-sandbox-worktree-containment-prd`
    γ7a–γ7e.
- **The mark.**

  | Mark | When |
  |---|---|
  | `delivered` | every signal leaf is done; the hover shows each leaf's `done_provenance.kind` |
  | `partial` | some, not all, are done, and none is cancelled |
  | `not_yet` | none is done, and none is cancelled |
  | `unknown` | no sidecar; a sidecar that fails to load (reason = the validation error); **"PRD names no signal leaf"** (the INV-13 state, distinct from every other); an unbound label; a bound task absent from the store; a stale binding (bound task's PRD ≠ this PRD); an `external_task_id` leaf; a cancelled signal leaf ("signal leaf cancelled", never `delivered`) |

- **Coverage.** The view also serves `signal_coverage = {named, of}`: how many open-work PRDs
  name a signal leaf. So "unknown everywhere" is visibly a missing-producer state, not "not
  yet".
- **Backfill (ε1–ε4, "ε" below).** Until ε lands, every PRD reads unknown, and so Tail is
  empty. ε adds `signal_leaves` to dark_factory's open-work PRDs' sidecars, under the rule in
  §9: name a leaf only where the PRD's own text names it.

### 4.8 Lanes: a partition served with rules

Lanes are assigned per PRD by first match, in this order. The order is served in the payload,
and each rule string has one home (`prd_view.py::LANE_RULES`).

1. **Needs a ruling (`ruling`).** Some open member is either:
   - an **actionable human gate**: `task_kind='deterministic'` with `always_escalates`, no
     unfired milestone, and either blocked, or pending with every dependency met; or
   - the subject of a **pending level-2 escalation** in the project's queue (escalation corpus,
     queue-pending, level 2, numeric task id matching).

   A gate whose dependencies are unmet, or whose milestone has not fired, does not count.
   Neither does a level-1 escalation.
2. **Tail (`tail`).** The signal mark is `delivered`.
3. **Parked (`parked`).** Every open member is held by a recorded trigger:
   - deferred, with a valid `metadata.deferral` of a caller kind (not `planning`) whose
     `shared.task_deferral.needs_human` is `None`; or
   - pending, with an unfired milestone.

   `needs_human` takes the holder's liveness for a `held_by_session` record. Without
   liveness, that record reads `liveness_unknown`, which needs a human. So liveness is the
   view's sixth input. It is produced once per snapshot, off the event loop, by one
   access-layer function shared with the deferral PRD's row summary (6527). γ2 extracts that
   function if 6527 placed it inside `active_tasks.py`, which `prd_view` may not import.

   A `planning` record is never held: a planning birth is transient by design, and a PRD
   mid-decompose must not read as parked. A deferred member with no record, or one needing a
   human, keeps the PRD out of Parked.
4. **Moving (`moving`).** Some member, open or done, started or finished within 14 days (§4.6),
   or an open member is in progress now.
5. **Stuck (`stuck`).** Everything else: open work remains, nothing started or finished in 14
   days, no ruling is pending, and not everything is held.

Each row's **state line** is its lane's reason with figures, built server-side. For example:

- "awaits ruling on esc-3546-2 (#3546)";
- "signal delivered; 4 follow-ups open";
- "moving: 1 direct, 3 follow-ups started or finished in 14 d". Moving always splits direct
  from follow-up activity, because lineage follow-ups keep landed PRDs in Moving (§2.2);
- "3 ready, oldest pending ≥43 d; nothing started in 14 d";
- "nothing ready: waiting on #4618 (no PRD)";
- "2 deferred with no recorded reason";
- "2 planning births not yet committed".

**Next-action chip.** One enum kind with a target:

| Kind | Target | Used when |
|---|---|---|
| `rule_on` | escalation id or gate task id | ruling |
| `dispatch` | oldest ready task | stuck with ready work |
| `unblock_dependency` | the earliest non-terminal dependency outside the PRD, with its own PRD or "no PRD" | stuck with nothing ready |
| `record_deferral` | a deferred member with no record | parked-blocking |
| `none` | — | moving and tail rows, which show their counts |

**Within a lane**, rows are ordered by attention: open high/critical count desc, oldest-ready
age desc, ready count desc, key asc.

**Tail never hides high work.** The dark-matter strip carries a second line, "high work under
collapsed Tail PRDs": every open high or critical member of a Tail PRD, by id. It meets the
brief's "never hide a high-priority task behind a collapsed tail". The brief also asked to
cover "stability-class" tasks; the store has no stability class, so that part is dropped (R4,
R5).

### 4.9 Dark matter and contention

- **Dark matter** (per project, open unattributed tasks):
  - total;
  - counts by status (the nine census members);
  - counts by priority;
  - counts by `metadata.source` (top five plus "other");
  - counts by `unresolved` reason;
  - the ids of ready high unattributed tasks;
  - `members: {reason: [task_id]}`, the drill-down.

  It expands to a list grouped by reason. Every listed task is open, so its row is already in
  the client's active rows and the client draws it by id. About 1,091 ids is a few KB on the
  wire. The client renders served membership and never groups by itself, so no further
  endpoint is needed. The drawn list is lazy (open question 4).
- **Contention** (per project):
  - The scheduler state datum's `lock_depth` and the file's own `snapshot_at` (the time of
    the last change, not a heartbeat: the file is content-deduplicated).
  - The top `N` (default 5) lock keys by the number of ready tasks whose
    `files_to_modules(metadata.files, lock_depth)` include the key. Each task counts once per
    key. Each key carries its holder (task id and that task's store status, which can be
    `pending`; 4828 was), or "no holder".
  - The share of ready tasks naming any held key, by priority band.
  - `never_started`: ready tasks at high or critical priority with no `task_started` in that
    project's runs.db since its first event, with count and ids, and the window start in the
    rule.
- **Shared counting rule.** The counting rule is lifted out of
  `data/scheduler.py::_module_contention_counts` into one public pure function parameterised
  by population. The Scheduler tab keeps counting over active rows; this line counts over the
  ready set. The two numbers differ by design, and each one's rule says which population it
  counts.

### 4.10 Rule text is payload, not envelope

- **Where rules live.** Every figure in `PrdView` and `Contention` names a `FigureId`.
  `prd_view.py::FIGURE_RULES` maps every `FigureId` to its rule text, and the map is shipped in
  each value as `rules`.
- **Why not the envelope.** The `Datum` envelope keeps its five fields: a rule is a
  definition, static per deploy, and `reason` is a per-measurement disclosure. Mixing them
  would put two axes of variability in one field (heuristic 3) and touch every Datum call
  site.
- **Rendering.** `shell.jsx::DatumReading` and `Pip` gain an optional `rule` prop. Their
  `title` is the non-fresh `reason` when there is one, otherwise the rule. Figures inside a
  view render through one small component that takes the parent Datum, so `≥`, staleness and
  em-dash-for-unknown apply uniformly (one-datum decision 7).
- **Test.** A test asserts every `FigureId` has a non-empty rule and every figure in a shaped
  payload names one.

### 4.11 The page

- **The PRD mode.** The per-project `list | by PRD` toggle stays. By PRD becomes the default
  when the preference is unset, and a persisted `list` is respected. In PRD mode a project
  group shows the contention line, the five lane sections in display order, then the
  dark-matter strip.
- **Tail.** Tail is collapsed to one line (count and names) and expands to rows.
- **Rows.** A row shows:
  - the PRD's H1 title (key on hover);
  - the state line;
  - the bar: six segments with 2 px gaps and a legend per project, never colour alone (each
    segment's title carries its label and count);
  - `done/total`, plus direct and follow-up counts on hover;
  - the signal mark;
  - `ready n · oldest Nd`;
  - the next-action chip.
- **Expanded row.** Clicking a row expands its open members through the existing
  `tab_tasks.jsx::TaskGraph` over the active rows already held, so taskgraph-legibility's
  layout and focus mode are kept. It also lists blockers, the oldest ready high members, and
  deferred members with no recorded reason.
- **Retired.** `ProjectPrdGroups`, `PrdBox` and `prd_grouping.js` are deleted. The client
  gains no counting code: one-datum sketch items 1 and 4 hold.
- **The contention line** shows in both modes, at the top of each project group.
- **A project footer** in PRD mode draws the remaining served figures, so none is produced
  without a reader (G1):
  - "n PRDs closed (m tasks)";
  - "k of n open-work PRDs name a signal leaf" (`signal_coverage`);
  - `multi_link` when non-zero;
  - each input's state as a `Pip`, with its reason on hover.

### 4.12 Placement (heuristics 6, 9, 13, 14)

- **Backend, new modules:** `data/task_lineage.py` (access),
  `data/prd_attribution.py` (pure; `_coalesce_prd` moves here from `active_tasks.py`, so
  coalescing has one home), `data/prd_view.py` (pure), `data/prd_signal.py` (access),
  `data/task_activity.py` (access), `api/prds.py` (router).
- **Backend, extractions in existing modules:**
  - `data/scheduler.py` gains the scheduler state access datum, and its collector consumes it.
  - The deferral-liveness access function moves to its own small module if 6527 left it in
    `active_tasks.py`.
  - `shared/src/shared/task_metadata.py::Milestone` gains `fired`.
- **Backend, existing files that do not grow:** `data/active_tasks.py` (1,232 lines) and
  `data/redux_api.py` (1,476) take no new logic. `app.py` (1,704) gains only the router
  include (one-datum decision 18).
- **Frontend:** components go in a new `static/redux/prd_lanes.jsx`. Any pure client helper
  (legend order, age formatting) goes in a classic script under `node --test`. `tab_tasks.jsx`
  loses the box code.
- **Access-path guard.** `tests/test_datum_access_paths.py::_GRANTS` gains one grant:
  `data/task_lineage.py` may call the typed projected read γ1 adds to `data/tasks.py`. It is
  owned by γ1 and ratified here (INV-12).

## 5. Contract

### 5.1 fused-memory `get_tasks(fields=…)` (α)

```
get_tasks(project_root, tag=None, page_size=None, offset=0, statuses=None,
          fields: list[str] | None = None)
```
- `fields=None` returns full rows, byte-identical to today.
- **Valid entries.** Each entry is either a top-level task key from a fixed allow-list, or
  `metadata.<key>` with `<key>` matching `[A-Za-z_][A-Za-z0-9_]*` (one level).
  - The allow-list is exactly the keys
    `backends/sqlite_task_backend.py::_row_to_task` emits, in its spelling (`updatedAt`,
    `testStrategy`; `id` is a string).
- **Coercion is kept.** Projection applies after `_row_to_task`'s coercion, so a row with
  malformed metadata JSON still projects (as `{}`) and never aborts the call. A SQL-side
  `json_extract` would abort the whole query on one bad row. There were 0 such rows across
  six stores at authoring.
- **Projected rows.** `id` is always returned. A projected row carries only the requested
  keys. `metadata` is present only when at least one `metadata.*` entry was asked for, and it
  holds only the requested keys **present on that task**. An absent key is omitted, not
  `null`.
- **Errors.** An invalid entry returns `{'error': …, 'error_type': 'ValidationError', 'field': <entry>, 'allowed': [...]}`
  (INV-2).
- **Composition.** The parameter composes with `statuses` and `page_size`/`offset`. The
  `pagination` envelope is unchanged, so pages tile the population.
- **One query path.** Projection is applied in one query-building path through
  `server/tools.py::get_tasks` → `TaskInterceptor.get_tasks` → `backends/task_backend_protocol.py`
  → `backends/sqlite_task_backend.py::SqliteTaskBackend.get_tasks`, the path 4390's other
  parts will reuse.

### 5.2 Sidecar `signal_leaves` and `sidecar_path` (β)

```
CapabilityManifestDoc:
  prd: str
  schema_version: Literal[1]
  tasks: list[ManifestTask]
  signal_leaves: list[str] | None = None   # labels in `tasks`; non-empty when present

sidecar_path(prd_key: str) -> str          # 'plans/x-prd.md' -> 'plans/x-prd.capability-manifest.yaml'
```
- **Validation.** An unknown label raises, naming the label and the known labels. An empty
  list raises. A minimal sidecar (only the signal leaf's block, with `capabilities: []`) is
  valid: `ManifestTask.capabilities` has no minimum length.
- **Stamper.** `manifest_stamping.py::_stamp_capability_manifests_impl` uses `sidecar_path`
  and preserves `signal_leaves` across a stamp.
- **Drift audit.** `scripts/audit_manifest_descriptor_drift.py` reports a signal label whose
  bound task is missing or whose PRD field disagrees with the sidecar's `prd`. It reports only,
  like the rest of that audit.

### 5.3 Lineage datum (γ1, `data/task_lineage.py`)

```
LineageEntry  = {own_prd: str | None, link: Link | None, done_stamped_at: datetime | None}
TaskLineage   = Mapping[int, LineageEntry]          # every task id in the project
acquire_lineage(client, config, project_root, *, now) -> Datum[TaskLineage]
```
- **Read.** One unpaged `get_tasks(fields=LINEAGE_FIELDS)` call through a typed projected read
  in `data/tasks.py` that bypasses `_shape_task`.
  - One cache entry and one last-good value per project, served `stale` past its bound and
    `unknown` before the first success.
  - A single-flight background refresh never holds a request past the handler deadline
    (§4.2).
- **Parsing.** Link and PRD values are parsed by `prd_attribution` (the one parser).

### 5.4 Attribution (γ1, `data/prd_attribution.py`, pure)

```
Link = LocalLink(field, target: int) | UnresolvableLink(field, reason, raw: str)
UnresolvedReason = no_link | root_without_prd | cross_project_link | unparseable_link
                 | missing_target | cycle | depth_exceeded | foreign_prd | lineage_pending
Attribution = {prd: str | None, direct: bool, hops: int, unresolved: UnresolvedReason | None}

coalesce_prd(metadata) -> str | None          # moved from active_tasks._coalesce_prd, same rule
PrdKeyResult = PrdKey(key: str) | Unresolved(reason: UnresolvedReason)   # tagged, never a bare str
normalise_prd_key(raw, project_root) -> PrdKeyResult   # strips './' and the root prefix; foreign → foreign_prd
parse_link(metadata) -> Link | None
attribute(entries: Mapping[int, LineageEntry]) -> tuple[Mapping[int, Attribution], int]  # (attribution, multi_link)
```
- **Invariants.** `prd is None` ⇔ `unresolved is not None`. `direct` ⇔ `hops == 0`.
  `hops ≤ MAX_HOPS`.
- **Memoisation.** The walk memoises per id, so the cost is O(tasks × MAX_HOPS) at worst, pure
  CPU. At about 6.5k tasks that is milliseconds, inside INV-8's bound for one coroutine; the
  shaping runs once per request, not per row.

### 5.5 `PrdView` and `Contention` (γ1 + γ2, `data/prd_view.py`)

```
Segment   = done | in_progress | blocked | deferred | ready | waiting
Lane      = ruling | tail | parked | moving | stuck
SignalMark= delivered | partial | not_yet | unknown
FigureId  = (closed enum; every figure below names one)

PrdRow = {
  key, title, lane, state_line,
  segments: {Segment: int}, total: int,             # total == sum(segments); cancelled excluded
  direct_open_ids: [int], followup_open: [{task_id, hops}],
  ready: {count, oldest_task_id | null, oldest_pending_days | null, oldest_is_lower_bound},
  signal: {mark, reason | null, leaves: [{label, task_id | null, status | null, done_kind | null}]},
  next_action: {kind, target | null, text},
  detail: {blocker_refs: [{task_id, prd | null}], ruling_refs,
           oldest_ready_high_ids, deferred_unrecorded_ids, high_open_ids},
  activity: {direct: int, followup: int},           # members started/finished in the window
}
PrdView = {
  prds: [PrdRow],                        # every key with >= 1 open attributed member
  lanes: [{lane, label, prd_keys}],      # display order; a partition of prds
  closed: {prds: int, tasks: int},       # attributed PRDs with no open member
  unattributed_done: int,                # done tasks no PRD owns (not drawn; closes the sum)
  dark_matter: {total, by_status, by_priority, by_source, by_reason, ready_high_ids,
                members: {UnresolvedReason: [int]},
                hidden_high: {count, refs: [{task_id, prd}]}},   # total = open unattributed
  signal_coverage: {named: int, of: int},
  multi_link: int,
  inputs: [{name, as_of, state, reason}],
  rules: {FigureId: str},
}
Contention = {
  lock_depth: int, snapshot_at: str,
  top: [{key, wanted_by_ready: int, holder: {task_id, status} | null}],
  held_share: {band: {ready: int, naming_held_key: int}},
  never_started: {count, of, ids, window_start},
  inputs: [...], rules: {FigureId: str},
}
```
**Shaping validators** (`validate_prd_view`, run at shaping like `validate_datum`; a breach
raises `DatumContractError` and the key degrades to `unknown`, the endpoint still 200):
- `lanes` partition `prds`. Every key appears in exactly one lane, and that lane equals the
  row's `lane`.
- For every row, `sum(segments) == total` and `total − segments['done'] ≥ 1`: a PRD has a row
  exactly when it has an open member. Otherwise it is counted in `closed`.
- `Σ total(prds) + closed.tasks + unattributed_done + dark_matter.total == c.total − c.counts['cancelled']`,
  where `c = census.build_census(status_map)` is computed inside `prd_view` over the very map
  the members were drawn from. It is never compared with `snapshot.census`, which may be a
  last-good value while the map half failed.
- `dark_matter.total == Σ len(members[r])`, and every id appears once across all rows'
  `direct_open_ids`/`followup_open` and the dark-matter `members`.
- Every figure's `FigureId` is in `rules`.

### 5.6 Wire

`GET /api/v2/dashboard/prds` → `{"PRD_VIEW": {label: Datum}, "PRD_CONTENTION": {label: Datum}, "served_at": str}`.
`data.js` registers both as maps of nested Datums stamped with the response receipt, like
`TASKS_SNAPSHOT`. The SPA never constructs, recounts or reclassifies (one-datum sketch 1,
decision 6).

## 6. Boundary-test sketch

Producer and consumer sides of each seam, over real fixtures. Each test drives a real store,
a fixture runs.db, fixture sidecars or a fixture corpus. No test patches a private name
(Tests stance).

| # | Scenario [task] | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Projection shape [α] | sqlite store with tasks carrying some/none of the requested metadata keys | `fields=['id','metadata.prd_path','metadata.spawned_from']` rows carry only `id` + present keys; `fields=None` byte-identical to today |
| 2 | Projection errors [α] | — | `fields=['foo']` and `fields=['metadata.a-b']` → `ValidationError` naming the entry and the allowed set |
| 3 | Projection paging [α] | 4,500 tasks | `fields` + `statuses` + `page_size=2000` pages tile the population; `pagination` keys unchanged |
| 4 | `signal_leaves` validation [β] | sidecar fixtures | known labels load; unknown label and `[]` raise naming the label; minimal sidecar with `capabilities: []` loads |
| 5 | Stamp preserves the field [β] | sidecar with `signal_leaves`, planning batch | `commit_planning` stamps ids, field intact; stamper resolves the path via `sidecar_path` |
| 6 | Drift audit [β] | signal label bound to a task of another PRD | audit reports it, naming label, task and both PRD values |
| 7 | Attribution cases [γ1] | lineage fixture: direct; chains of 1–6 hops through done tasks; a cycle; 9 hops; `'reify:5569'`; dict link; `'cand-12'`; missing target; two link keys | each gets the expected `prd`/`hops`/`unresolved`; `multi_link == 1` |
| 8 | Overlay and pending [γ1] | lineage older than a new active row; a status-map id in neither | active row's own PRD wins; the orphan id is `lineage_pending` in dark matter |
| 9 | The census invariant [γ1] | snapshot fixture; a second fixture whose status-map half failed while its census is last-good | Σ equation of §5.5 holds against `build_census(status_map)`; a failed map half yields `unknown` with **the map's own reason** (not a validator reason); a forced breach yields `unknown` with the validator's reason; endpoint 200 throughout |
| 10 | Segments and ready [γ1] | members: done, review, blocked, deferred, pending with a pending dep, with an external dep, with an unfired dated milestone, with a fired dated milestone, with a delayed milestone whose anchor is unset, a human gate, a clean pending | each lands in its segment (fired milestone → `ready`, unset delayed anchor → `waiting`); `sum == total`; cancelled member excluded |
| 11 | Lineage failure and deadline [γ1] | lineage read fails before / after a first success; a lineage read slower than the handler deadline | `unknown` with reason / `stale` with last-good; the slow read does not hold the response past the deadline and lands for the next request; `PRD_CONTENTION` unaffected; the composite is `fresh` only when every input is fresh in this project's scope |
| 12 | Rules complete [γ1] | shaped payload | every `FigureId` has rule text; every figure names one; every open id appears exactly once across PRD rows and dark-matter `members` |
| 12a | `/prds` budget [γ1] | the handler's budget constants | a structural test, like `test_tasks_budget.py`, that the per-root reads fit the whole-handler deadline and the deadline fits the browser abort |
| 13 | Lane precedence [γ2] | one fixture PRD per precedence pair | each lands in the first matching lane; lanes partition |
| 14 | Ruling inputs [γ2] | corpus fixture with L2 pending on a member, L1 pending on another PRD; a second configured root with no `data/escalations/`; gates with deps met / unmet / blocked / deps met but milestone unfired | only the L2 and the actionable gates put a PRD in ruling (not the unfired-milestone gate); `rule_on` names the escalation; the missing queue of the other root leaves this project's corpus input `fresh` |
| 15 | Signal marks [γ2] | sidecar fixtures for each row of §4.7's table | each mark and reason as tabled; `signal_coverage` counts named PRDs |
| 16 | Moving inputs [γ2] | runs.db fixture: starts 13 d and 15 d ago; done with event; done without event with/without `stamped_at` | 13 d and in-window done move the PRD; 15 d does not; unreadable runs.db → activity `unknown` and the view's state says so |
| 17 | Parked [γ2] | all-deferred PRD with valid caller-kind records; one with an unrecorded deferral; one all-`planning`; a `held_by_session` record with a live holder (a real spawned process) and one with a dead holder; a pending member with an unfired milestone and one with a fired milestone | first parked; second stuck with "1 deferred with no recorded reason" and `record_deferral`; all-planning not parked ("planning births not yet committed"); live holder held, dead holder not; unfired milestone held, fired not |
| 18 | Stuck line and age [γ2] | ready members with backfilled and real anchors | `oldest_pending_days` from the oldest; `oldest_is_lower_bound` true for a backfilled anchor; `dispatch` names it |
| 19 | Contention [γ2] | scheduler state fixture with `lock_depth`, holders (one `pending` holder), a wanted key with no holder, a file `snapshot_at`; runs.db with/without `task_started` | top keys counted once per task; holder status from the status map; served `snapshot_at` is the file's, not the cache fill time; `never_started` from `task_started` only; scheduler offline → `PRD_CONTENTION` unknown, `PRD_VIEW` unaffected; the Scheduler tab's payload is unchanged (it now reads the same access datum) |
| 20 | Render [δ] | boundary payload fixture family `prd_view` | five sections in display order, Tail collapsed with count and names; each row's bar has six titled segments and a legend; hover titles equal the served rules; an `unknown` view renders its reason and the list mode still works; `prd_grouping.js` is gone and `classic_script_scope` agrees |
| 21 | Mode default [δ] | no persisted preference / persisted `list` | by-PRD / list |

## 7. Pre-conditions (G3): substrate verified on `99ad2d7ddb`

| Capability assumed | Evidence |
|---|---|
| Snapshot unit exposes raw active rows (full metadata) and the full status map | `dashboard/src/dashboard/data/task_snapshot.py::TaskSnapshot` (`rows: Datum[list[dict]]`, `status_map`) |
| Census has nine members incl. `cancelled` | `dashboard/src/dashboard/data/census.py::build_census`; generated `task_vocab.js` |
| `get_tasks` paging + `statuses`; **no** `fields` today | `fused-memory/src/fused_memory/server/tools.py::get_tasks` — queued as α |
| Field-projection owner exists, pending, same files | task 4390 (pending, medium) — α delivers its part (2), §8 |
| Escalation corpus with level, status, task_id, queue location | `dashboard/src/dashboard/data/escalation_corpus.py::acquire_corpus`, `CorpusRecord`; a level-2 task-id view is new (γ2) |
| runs.db per project, `events(event_type, task_id TEXT, timestamp)`, `event_type` index; `task_results.completed_at` | `data/orchestrator/runs.db`; `dashboard/src/dashboard/project_dbs.py::_project_scoped_dbs_labeled` |
| Scheduler snapshot with `current_holders`, `lock_depth`, `snapshot_at` through MCP | `fused-memory/src/fused_memory/mcp_tools/scheduler_state.py::read_scheduler_state`. The dashboard's `data/scheduler.py::collect_scheduler_state` reads it but **discards** the raw holders, `lock_depth` and the file's `snapshot_at` inside `_one_project` — γ2 extracts the access datum (§4.3) |
| Escalation corpus per-project scoping | `dashboard/src/dashboard/data/escalation_corpus.py::_scope_provenance`, `views_over`; records keyed per queue (`QueueRef.id`), so empty `project_id` on records does not matter |
| Milestone model; no `fired` method today | `shared/src/shared/task_metadata.py::Milestone`; the scheduler's own predicate `orchestrator/src/orchestrator/scheduler.py::Scheduler._milestone_time_gated` — γ1 adds `Milestone.fired` |
| Full-list build in `get_tasks` (so one unpaged projected call costs the same server work as a walk) | `fused-memory/src/fused_memory/server/tools.py::get_tasks` slices after building the list |
| Typed projected read must bypass row shaping | `dashboard/src/dashboard/data/tasks.py::_shape_task` fabricates `title=''` / `status=None` for missing keys |
| Lock normalisation helpers | `shared/src/shared/locking.py::files_to_modules`, `modules_conflict` |
| Sidecar loader; empty `capabilities` allowed; `extra='forbid'` (so β precedes any use) | `shared/src/shared/capability_manifest.py::load_capability_manifest`, `ManifestTask`, `CapabilityManifestDoc` |
| Stamper rewrites the raw dict | `fused-memory/src/fused_memory/server/manifest_stamping.py::_stamp_capability_manifests_impl` |
| Deferral record model and `needs_human` | **not on main**: `shared/src/shared/task_deferral.py` is task 6524 (in progress). γ2 depends on it |
| Human-gate shape | `docs/task-authoring.md` (execution_class `operational`/`decision` → `task_kind='deterministic'` + `always_escalates`) |
| `pending_since` / `pending_since_backfilled` | `docs/task-authoring.md` (pending_since paragraph) |
| `done_provenance.stamped_at` (subset of done tasks) | `docs/task-authoring.md` §2; measured on 345 tasks |
| Shared rendering components and `datumView` | `dashboard/src/dashboard/static/redux/shell.jsx::DatumReading`, `Pip`; `datum.js::datumView` |
| Task graph reusable without PRD code | `dashboard/src/dashboard/static/redux/tab_tasks.jsx::TaskGraph`, `graph_layout.js::taskGraphLayout` |
| Access-path guard and its grant list | `dashboard/tests/test_datum_access_paths.py::_GRANTS` |
| Cache-buster guard | `dashboard/tests/test_index_html.py::test_redux_cache_buster_is_newer_than_merge_base` |

## 8. Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/dashboard-one-datum-one-path-prd.md` | consumes + supersedes | `Datum` envelope, snapshot unit, census, corpus walk, access-path guard. Supersedes its decision 8 (`≥ n/m`) and out-of-scope "per-PRD server census" | **this PRD**; decompose appends a dated note to that PRD | landed (5585–5598 done) |
| `plans/dashboard-taskgraph-legibility-prd.md` | replaces part 2, keeps part 1 | `ProjectPrdGroups`, `PrdBox`, `prd_grouping.js` deleted; `TaskGraph`, `TaskGraphEdges`, `graph_layout.js`, focus mode kept as the row body | **this PRD**; dated note at decompose | landed |
| `plans/deferral-flip-condition-prd.md` (6524–6529) | consumes | `shared.task_deferral` record + `needs_human` decide Parked; 6527's per-snapshot holder-liveness read is reused, not duplicated | that PRD owns the record, `needs_human` and its row label; **this PRD** owns the lane rule and, in γ2, extracting the liveness function if 6527 left it inside `active_tasks.py`. γ2 depends on 6524 (the model) and 6527 (the liveness read) | 6524 in progress, rest pending |
| task 4390 (fused-memory `get_tasks` query surface) | shares files | α delivers 4390's part (2) "field projection"; 4390 keeps (1) and (3) | **this PRD** owns `fields`; decompose wires 4390 → α and appends a dated note | pending |
| task 5276 (stamper: preserve sidecar comments, skip signals) | premise | it replaces the stamper's `yaml.safe_dump(raw)` write-back that "a declared field survives a stamp" rests on | 5276 owns the write-back; **β depends on 5276** and re-verifies the premise against what landed | in progress |
| `/prd` skill + `plans/capability-delivered-checks-prd.md` §Contract | produces | `signal_leaves`, `sidecar_path`; decompose-mode Step 2.5 and gates.md manifest text | **this PRD** (β) | — |
| `plans/scheduler-dispatch-scoring-and-lock-layer-prd.md`, `plans/dashboard-park-stack-visibility-prd.md` | consumes read-only | scheduler state through one access datum extracted in `data/scheduler.py`; the contention counting rule made public and parameterised by population | **this PRD** for both dashboard extractions; no scheduler change | landed |
| Tasks superseded by δ: 6336 (`ProjectPrdGroups` `filteredSig` fix), 6013 (`TestThePrdBoxCount` pin rewrite) | obsoleted | both edit code δ deletes | decompose cancels both, citing δ | pending |
| Status review 2026-10-06 (artifact `2YipixJ2gEUCMoKQxfa8p9`) | source | the prototype and measurements | — | proposal |

**File-order seams.** Other open tasks share files with this batch. The critic's overlap query
at authoring, over open tasks' `metadata.files`, found:

- α: 165 (`server/tools.py` is a hub);
- β: 12, including 4700, 6423, 6525 and 6558;
- γ1: 4618 in progress on `data/tasks.py` and `active_tasks.py`, and 6309 on `app.py` and
  `_boundary_payloads.py`;
- γ2: 4889 and 5760 on `scheduler.py`;
- δ: 21, including 6309, 6304 (moves `TAB_ENDPOINTS` to a new `poll_scope.js`), 6220 and 6253.

The scheduler's file locks already stop two of these from running at once. A dependency edge
is added only where this batch's **premise** depends on the other task's outcome (5276, 6524,
6527). Each affected brief names its overlaps, so the agent re-reads the landed state; for
example, δ edits `TAB_ENDPOINTS` wherever it lives when δ runs. Task 5924 (delete the dead
`fetch_tasks` page walk) is unaffected, because γ1's lineage read is one unpaged call.

No reciprocal ambiguity: each row names one owner. The deferral PRD's own §8 row lists "a later
lanes redesign" as the consumer of its record. This PRD is that consumer.

## 9. Decomposition plan

Ten tasks: α, β, γ1, γ2, δ, ε1–ε4, ζ. Sizes follow the overlay bands
(`.claude/skills/prd/project.md`). Within the batch, two tasks never edit the same file
without a dependency edge between them. Edges to tasks outside the batch exist only where a
premise depends on them (§8).

No task in the batch delivers a service restart. So every live check that needs fused-memory
or the dashboard restarted onto this code belongs to ζ (the deferral PRD's decision 12
precedent).

- **α — fused-memory: `get_tasks(fields=…)` projection.** [medium; normal; ~400–700 LOC;
  ~6 files] no dependency. Its overlaps (6525, 5276 and others on `server/tools.py`) are
  serialised by the scheduler's locks. Intermediate: unlocks γ1; 4390 then depends on it.
  - Files: `fused-memory/src/fused_memory/server/tools.py`,
    `middleware/task_interceptor.py`, `backends/task_backend_protocol.py`,
    `backends/sqlite_task_backend.py`, a new `fused-memory/tests/test_get_tasks_projection.py`.
  - Signal: boundary rows 1–3 green against a real sqlite store.
- **β — `signal_leaves`, `sidecar_path`, skill text.** [medium; normal; ~350–600 LOC;
  ~8 files] depends on 5276: its stamper write-back is the premise of "the field survives a
  stamp", and β's test re-verifies it against what landed. Intermediate: unlocks γ2 and ε1–ε4.
  - Files: `shared/src/shared/capability_manifest.py` and its tests;
    `fused-memory/src/fused_memory/server/manifest_stamping.py` (use `sidecar_path`) and the
    stamping test; `scripts/audit_manifest_descriptor_drift.py` and its test;
    `skills/prd/references/decompose-mode.md` Step 2.5; `skills/prd/references/gates.md`
    (Capability Manifest output contract).
  - Signal: boundary rows 4–6.
- **γ1 — Lineage, attribution, per-PRD census, dark matter, `/prds`.** [medium; normal;
  ~1,100–1,500 LOC; ~13 files] depends on α. Intermediate: unlocks γ2.
  - Files: `data/tasks.py` (typed projected read); new `data/task_lineage.py`,
    `data/prd_attribution.py`, `data/prd_view.py` (segments, ready, census join, dark matter,
    rules, validators), `api/prds.py` (handler deadline, per-root concurrency);
    `app.py` (router include); `data/active_tasks.py` (`_coalesce_prd` →
    `prd_attribution.coalesce_prd`); `shared/src/shared/task_metadata.py` (`Milestone.fired`).
  - Tests: new `test_prd_attribution.py` (the 12 `test_build_task_row_prd_*` cases move here),
    `test_task_lineage.py`, `test_prd_view.py`, `test_prds_budget.py`;
    `test_datum_access_paths.py` (the grant); `_boundary_payloads.py` (`prd_view` family);
    `shared/tests/` (`Milestone.fired`).
  - Overlaps: 4618 (in progress) on `data/tasks.py` and `active_tasks.py`; 6309 on `app.py`
    and `_boundary_payloads.py`; 6558 on `task_metadata.py`.
  - **Payload subset.** γ1 serves the subset of §5.5 it can fill from recorded inputs:
    segments, totals, ready, dark matter (without `hidden_high`), `closed`, `multi_link`,
    inputs and rules.
    - The lane, signal and next-action fields and `PRD_CONTENTION` are **absent**, never
      placeholders (INV-13). γ2 adds them, and its validators then require them.
    - δ does not start before γ2, so no page ever renders the subset.
  - Signal: boundary rows 7–12. `GET /api/v2/dashboard/prds` against the fixture substrate
    returns exact per-PRD `done/total` and a dark-matter block satisfying §5.5's invariant.
- **γ2 — Lanes, signal mark, activity, ruling, contention.** [medium; normal;
  ~1,100–1,500 LOC; ~13 files] depends on γ1, β, 6524 and 6527. Intermediate: unlocks δ.
  - Files: `data/prd_view.py` (lanes, state line, next action, attention order, hidden high,
    `Contention`); new `data/prd_signal.py`, `data/task_activity.py`;
    `data/escalation_corpus.py` (a project-scoped pending-level-2 task-id view);
    `data/scheduler.py` (extract `acquire_scheduler_state` as the one access datum and make
    the collector consume it; public contention counter by population); the deferral-liveness
    access function (extracted to its own module if 6527 left it in `active_tasks.py`);
    `api/prds.py`.
  - Tests: new `test_prd_lanes.py`, `test_prd_signal.py`, `test_task_activity.py`;
    `test_scheduler*.py`; `_boundary_payloads.py`.
  - Overlaps: 4889 and 5760 on `scheduler.py`.
  - Signal: boundary rows 13–19.
- **δ — The page.** [medium; normal; ~1,000–1,400 LOC; ~17 files] depends on γ2.
  Intermediate: unlocks ζ.
  - Files: new `static/redux/prd_lanes.jsx`; `tab_tasks.jsx` (delete `ProjectPrdGroups`/`PrdBox`,
    mode default, drop the by-PRD `?terminal=` trigger); delete `prd_grouping.js`;
    `shell.jsx` (`rule` prop); `data.js`; `endpoint_staleness.js`; `styles.css`;
    `index.html` (script tags, `?v=` bump).
  - Tests: delete `tests/js/prd_grouping.test.mjs`; new `tests/js/boundary_prd_view.test.mjs`;
    rewrite the PRD-box source pins in `test_tab_tasks_terminal_window.py`,
    `test_tab_tasks_focus_header.py` and `test_tab_tasks_status_counts.py`; update
    `test_index_html.py`, `test_cache_buster_freshness.py`, `test_tab_tasks_offline_banner.py`
    (it lists `prd_grouping.js` among its plain-JS siblings) and
    `tests/js/classic_script_scope.test.mjs`.
  - **The file count crosses the 15-file review trigger,** at about 17. That is accepted after
    re-examination. Nine of the files are pins and guards that must move in the same change
    as the code they pin: `prd_grouping.js` cannot be deleted in one commit and its pins in
    another without a red tree. The new code is one file, `prd_lanes.jsx`.
  - Overlaps: 6309 (`tab_tasks.jsx`, `shell.jsx`, `data.js`, `index.html`); 6304 (moves
    `TAB_ENDPOINTS` out of `endpoint_staleness.js`); 6220 and 6253 (`tab_tasks.jsx`).
  - Signal: boundary rows 20–21.
- **ε1–ε4 — Backfill `signal_leaves` for dark_factory's PRDs with open work.** [medium;
  normal; four tranches of about 19 PRDs each; data edits plus a completion note] each
  depends on β. Intermediate: each unlocks ζ.
  - **Why four tasks.** At authoring the 74 open-work PRD files totalled about 2.6 MB of text
    (median 30 KB, max 113 KB). One agent cannot read that inside the architect's turn cap.
    Decompose fixes each tranche's PRD keys from the live open-work set, ordered by key.
  - **What to read.** A tranche reads each PRD's goal and decomposition-plan sections, and its
    `.capability-manifest.md` where one exists. Those are where an integration gate or signal
    leaf is named. It does not read whole files.
  - **Scope.** dark_factory only. Other projects' PRDs render `unknown` with "PRD names no
    signal leaf" until their own backfill. That is not filed here: reify's sidecars live in
    another repository.
  - **This PRD's own sidecar** cannot carry `signal_leaves` at decompose, because
    `extra='forbid'` would make the stamper skip it until β lands. The tranche holding this
    PRD's key names ζ as its signal leaf.
  - **The rule.** Name a label only where the PRD's own text names the task(s) whose
    completion delivers its goal signal: its integration gate, or the leaf its decomposition
    plan marks as carrying the user-observable signal.
  - **Where the text names none,** or the label is unbound or bound to a foreign id, leave the
    field absent and list the PRD in the completion note with why.
  - **A PRD with open work and no YAML sidecar** (the chain-only, legacy ones) gets a minimal
    sidecar only when the leaf's task id is unambiguous: its own PRD field is this PRD, and the
    PRD's text names that task. Otherwise it stays unknown.
  - **Never** infer from the dependency graph alone (19 of 72 unique, §2.2).
  - The file count per tranche is high by nature (one small edit per sidecar, about 19). It is
    a mechanical-shape edit under one rule.
  - Signal: every touched sidecar loads, and the drift audit (β) reports no stale signal
    binding. The completion note's table lists each PRD in the tranche as named (label, task)
    or unnamed (why).
- **ζ — Integration gate: restart, then live checks.** [medium; deterministic,
  `execution_class='operational'`, human gate] depends on δ and ε1–ε4 (and so on
  everything). It closes the batch.
  1. Confirm no merge verify is in flight (`get_merge_queue`). Restart fused-memory per
     `OPERATIONS.md`, then the dashboard (`scripts/restart-dashboard.sh`).
  2. **(a) Coverage.** The Tasks tab shows every PRD with open work in exactly one lane, with
     Tail collapsed.
     - The set of PRD keys equals an independent read-only sqlite recount under §4.1's rule,
       taken within a minute of the payload's `as_of`. Record both.
     - The dark-matter total equals the recount's unattributed count, within the churn between
       the two reads. Record both. At authoring this was 1,091 of 1,659: the "truly unowned
       share" the brief asked for.
  3. **(b) Stuck with red age.** Some PRD with no member started or finished in 14 days and a
     ready member pending 30 days or more appears in Stuck, with that age in red (`≥` when
     backfilled). Cross-check that PRD's members in sqlite and runs.db.
  4. **(c) Contention.** The contention line names the top held files with their holders.
     They equal `scheduler_state.json::current_holders` at the same `snapshot_at`.
  5. **(d) Rules on hover.** Hovering a lane header, a bar segment, `ready n`, a signal mark
     and a contention count shows rule text.
     - Any input that is non-fresh at check time (an offline project, a stale scheduler
       snapshot) shows its reason in place of a number.
     - If every input is fresh during the check, record that and rely on boundary rows 11 and
       19. Do not break a live service to provoke it.
  6. **(e) Signal.** At least one PRD that ε named shows `delivered`, `partial` or `not_yet`,
     with leaf hover. A PRD ε left unnamed shows "PRD names no signal leaf".
     `signal_coverage.named` is at least ε's named count for PRDs still open. List any
     difference: PRDs leave the open-work set, and later decomposes add names.
  7. **(f) Live grant.** `GET /api/v2/dashboard/prds` returns 200 with both keys for every
     configured project, and `PRD_VIEW[dark_factory].state` is `fresh`.
  8. Record results in the task's completion note. A failed check is filed as a task against
     the owning leaf, not fixed in the gate.

### Capability bindings (draft for the decompose manifest)

| Leaf | Capability | Evidence today | Producer |
|---|---|---|---|
| ζ (a) | per-PRD exact counts | none (`≥ n/m` only) | γ1, upstream |
| ζ (a) | terminal-task link fields readable in bulk | `get_tasks` has no `fields` | α, upstream of γ1 |
| ζ (b) | ready age, lower-bound marker | `metadata.pending_since`, `pending_since_backfilled` present (1,058/1,062) | γ1 reads |
| ζ (b) | activity in 14 days | runs.db `task_started` live (9,954 rows) | γ2 reads |
| ζ (c) | holders, lock depth, file `snapshot_at` | `scheduler_state.json` keys live; the dashboard discards them today | γ2 extracts the scheduler state datum |
| ζ (b), Parked, Ruling | milestone fired-ness | `Milestone` model exists; no `fired` | γ1 adds `Milestone.fired` |
| Parked | holder liveness for `held_by_session` | 6527 computes it per snapshot (pending) | 6527, reused by γ2 |
| ζ (d) | rule per figure | no rule slot today | γ1 (`FIGURE_RULES`), δ renders |
| ζ (e) | which leaf proves the signal | no field today | β schema, ε data |
| ζ (e), Parked | deferral record + `needs_human` | not on main | 6524 (deferral α), upstream of γ2 |
| ruling | pending L2 by task | corpus has `level`, `status`, `task_id` | γ2 view |

### G7 walk (advisory at author; re-walk at decompose)

- **INV-13 `readers-prove-their-producer`.**
  - The signal reader's producer (`signal_leaves`) does not exist until β and ε. The view's
    "PRD names no signal leaf" state and `signal_coverage` keep that distinct from "not yet".
  - Parked reads `metadata.deferral`, which has no rows until the deferral migration. A deferred
    member without a record keeps its PRD out of Parked, with a state line saying so. It is
    never treated as parked.
  - The lineage read's producer (α's parameter) is proven live in ζ (f).
- **INV-11 `no-silent-fail-soft`.** Every unattributed task carries a reason, and every input
  carries its state. A partial input degrades the Datum's state; it never shortens a count
  silently.
- **INV-9 `one-fact-one-home`.** The signal-leaf fact lives in the sidecar. "Delivered" is
  rendered from the store. Rules, lanes, segments and the ready predicate live once in
  `prd_view.py`, and PRD coalescing lives once in `prd_attribution.py`.
- **INV-5 `no-lockstep-duplication`.** The client gets vocabulary and rules from the payload.
  `prd_grouping.js` goes away. The contention counter and the scheduler state datum are
  shared with the Scheduler tab, and the deferral-liveness read with 6527.
  - The ready predicate is a stated approximation of the scheduler's, not a copy: it may not
    be imported (one-datum decision 19).
  - Its milestone limb uses the shared `Milestone.fired`. The scheduler's own copy is owned by
    the follow-up that decompose files (§9).
- **INV-8 `loop-thread-occupancy-bounded`.** Attribution is O(tasks × 8) memoised pure CPU,
  about 6.5k tasks. Sidecar loads and runs.db queries run off-loop. Sidecar loads are bounded
  by the number of open-work PRDs and cached by mtime. runs.db queries are bounded by the
  `event_type` index. The `/prds` handler has a whole-handler deadline, and a slow lineage
  read refreshes behind it (§4.2).
- **INV-1 `contracts-machine-checked`.** §5.5's validators run at shaping.
- **INV-12 `exceptions-owned-or-ratified`.** The one new `_GRANTS` entry is owned by γ1 and
  ratified in §4.12.
- **INV-10 `guards-exercise-behaviour`.** The PRD-box source-regex pins are retired. δ's tests
  render fixture payloads through the boundary suite.
- **INV-3, INV-4, INV-6, INV-7.** Not applicable: the dashboard reads, holds nothing, and
  acts on nothing.

No hits; no waivers.

### Decompose companions (the decompose session's own docs commit)

- A dated note on `plans/dashboard-one-datum-one-path-prd.md` decision 8 and its out-of-scope
  bullet, pointing at γ1.
- A dated note on `plans/dashboard-taskgraph-legibility-prd.md` part 2, pointing at δ.
- On task 4390: a dated note that part (2) is delivered by α, plus `add_dependency(4390 → α)`.
- Cancel 6336 and 6013 as superseded by δ, each with a note naming δ.
- File one low follow-up: repoint `Scheduler._milestone_time_gated` at `Milestone.fired`
  (INV-12 owner of the remaining duplicate), depending on γ1.
- Add `metadata.files` overlaps to each brief as listed in §8.

## 10. Out of scope

- Any scheduler change: priority, scoring, locks, the hub-file problem itself. This PRD shows
  contention; it does not relieve it.
- Re-attributing tasks in the store, or stamping `prd_path` on follow-ups. Attribution is
  computed at read time.
- A recorded "stability class" or briefing-concern id on tasks. That is a classification
  scheme with its own producer question; file it separately if wanted.
- Following cross-project links.
- Migrating the four other restatements of the sidecar suffix to `sidecar_path`.
- 4390's recency window and `ids` filter.
- Resolving `external_task_id` signal leaves. They render unknown with that reason.
- The memory, escalation and scheduler tabs, apart from the shared contention counter
  refactor, which keeps the Scheduler tab's output unchanged.
- Re-checking a delivered signal leaf's `delivered_checks` on main ("delivered then
  superseded"). The mark is status-derived.
- Repointing `Scheduler._milestone_time_gated` at `Milestone.fired`. Decompose files it as a
  low follow-up (§9 companions).
- `signal_leaves` backfill outside dark_factory.

## 11. Open questions (tactical)

1. **Lineage TTL, freshness bound and the `/prds` handler figures.** Suggested:
   - lineage TTL 300 s and bound 900 s;
   - handler deadline 20 s, below the 30 s browser abort, as `/tasks`;
   - per-root concurrency 4, as `/tasks`;
   - per-call timeout sized from a measured unpaged projected read on reify (8.4k tasks).

   Decide in γ1 and pin the figures in `test_prds_budget.py`.
2. **Contention `N`.** Suggested 5. Decide in δ from screen fit.
3. **Moving window.** 14 days, as briefed. Keep it a named constant beside its rule.
4. **The dark-matter drill-down's page size.** Suggested 100 rows per reason group with "more".
   Decide in δ.
5. **Whether `/prds` polls on the 3 s Tasks-tab cadence or slower.** The inputs are cached,
   so the poll is cheap. Suggested: same cadence. Decide in δ.
6. **The H1 title read.** Suggested: the first `# ` line of the PRD file at the project root,
   cached by mtime, falling back to the key. Decide in γ2.
