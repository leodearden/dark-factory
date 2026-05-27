# Escalation queues dashboard section

## Mission
Add a top-level **Escalations** tab to the redux dashboard. It shows one foldable
subsection per orchestrator plus one for fused-memory (reconciliation), each
listing that queue's escalations. The operator can filter by level and status,
sort by task, and click any escalation to open a sidebar showing its full detail
and the task it applies to.

## Consumer & user-observable surface
- **Consumer:** the human operator (the L2 of the escalation ladder). This is a
  leaf observability surface — no other PRD or mechanism consumes its output.
- **User-observable surface:** a new **Escalations** entry on the dashboard rail.
  Opening it shows foldable per-queue subsections, level/status filter chips, a
  task sort toggle, and a detail sidebar on selection.

## Why
Escalations are the factory's exception channel (L0 agent→steward, L1
steward→auto-watcher, L2 auto-watcher→human). Today the only way to see what's
pending across orchestrators and the reconciliation queue is to read JSON files
under `data/escalations/` and `data/reconciliation/escalations/` or tail logs.
The escalation-watcher skill consumes L2s one at a time; there is no at-a-glance
view of the whole backlog, its distribution across queues, or the task context
behind a given escalation. This tab makes the escalation backlog a first-class
visual surface, mirroring the Curator and Merge Queue tabs.

## Scope decisions (locked)

1. **Read-only v1.** The tab reads escalation JSON files and renders them. No
   write path to any escalation MCP — no resolve/dismiss/promote buttons.
   Resolution continues through `/unblock`, the escalation-watcher, and the
   escalation MCP as today. *Rationale: matches the literal ask, introduces no
   write seam to the escalation servers, and avoids encoding the subtle
   `resolve_issue(terminate=…)` / `promote_to_l2` semantics in the UI.*
   Resolve/dismiss/promote is a candidate follow-up PRD.

2. **Data source = live queue root only.** Read each `<root>/data/escalations/*.json`
   (the queue root only — **skip** the `archive/YYYY-MM-DD/` subtree). The root
   holds all `pending` escalations plus recently-resolved ones not yet archived,
   which is enough to make the status filter meaningful. No time-window param.
   *Rationale: archive can be large and unbounded; a live-queue view is always
   fresh and cheap, and the status filter still has resolved/dismissed rows to
   act on.* (`escalation.queue.iter_all_escalation_paths` includes archive, so a
   root-only loader variant is required — see Components.)

3. **Sidebar shows the full task card.** On selecting an escalation, render the
   full escalation record plus the task's full card (title, status, description,
   details, dependencies, metadata) sourced from the owning project's
   `fetch_tasks` map. No new MCP tool — `fetch_tasks` already returns this shape.

4. **Filters are global, fold/unfold is per-subsection.** Level and status are
   cross-cutting properties, so their filter chips live in the tab toolbar and
   apply to every subsection at once. Fold/unfold state is per-subsection
   (matching the other per-orchestrator tabs). Sort direction is global.
   *(If per-subsection filters are preferred this is a small change — see Open
   questions.)*

5. **Queue enumeration.** One subsection per known project root
   (`config.project_root` + `config.known_project_roots`), each reading
   `<root>/data/escalations`. Plus exactly one **fused-memory** subsection
   reading `config.project_root / data/reconciliation/escalations` (the shared
   reconciliation queue, the "8103" write-only queue). The primary root's own
   `data/escalations` and the reconciliation dir are distinct subsections.

## Architecture overview

```
<root>/data/escalations/*.json      ─┐  (per orchestrator: pending + recent)
  (one dir per known project root)    │
                                      ├─→ load_queue_escalations (root-only)
<primary>/data/reconciliation/        │        │
  escalations/*.json  ───────────────┘        ├─→ group by queue
  (fused-memory subsection)                    │   + summary counts (level/status)
                                               │
fused-memory MCP get_tasks ──→ per-project ────┤─→ /api/v2/dashboard/escalations
  (fetch_tasks, ~10s cached)    {id: task}     │        │
                                owning-project  │   shape_escalations
                                resolution      │        │
                                               └─→ window.DF_DATA.ESCALATIONS
                                                        │
                                            tab_escalations.jsx
                                       (subsections / filters / sort / sidebar)
```

## Data model

### Reused (no schema changes)
- **Escalation JSON** — `escalation.models.Escalation` (`escalation/src/escalation/models.py`).
  Fields the UI uses: `id`, `task_id`, `agent_role`, `severity`, `category`,
  `summary`, `detail`, `suggested_action`, `timestamp`, `status`
  (`pending`/`resolved`/`dismissed`), `resolution`, `worktree`, `workflow_state`,
  `level` (0/1/2), `resolved_at`, `resolved_by`, `members`, `root_cause`,
  `options`.
- **`fetch_tasks(client, config, project_root)`** (`dashboard/data/tasks.py`) —
  returns dashboard-shaped task rows `{id, title, description, details, status,
  priority, dependencies, metadata}`. Already cached ~10s per root by the
  merge-queue `load_task_titles` pattern; the sidebar reuses the same per-root
  task map.

### New
- No DB tables. The escalation queues are filesystem-sourced and the task cards
  come from existing MCP reads. The only new persistent state is client-side UI
  state in localStorage (`df.open.esc`, `df.esc.filter`, `df.esc.sort`).

## Definitions

### Queue subsection
A foldable group bound to one escalation directory. Subsection id =
`str(project_root)` for orchestrators, plus the literal `reconciliation` for the
fused-memory queue. Label = project basename (or "fused-memory").

### Owning-project resolution (for the sidebar task card)
Orchestrator escalations belong to their subsection's project root, so the task
card is fetched from that root's `fetch_tasks` map. Reconciliation escalations
reference `task_id`s that belong to some project but the recon queue is global,
so resolve the owning project by:
1. `worktree` path prefix if present (e.g. `/home/leo/src/reify/.worktrees/123`
   → `reify`), else
2. probe each known root's task map for the `task_id`, first hit wins.

Label the task card with its resolved project so the operator sees which project
the task lives in. Task ids can in principle collide across projects; the
worktree hint and project label make the ambiguity visible rather than silent.

### Summary counts
Per subsection and overall: counts bucketed by `level` (0/1/2) and `status`
(pending/resolved/dismissed), shown in the `ProjectGroup` summary line and the
rail count (rail count = total pending across all queues).

## Components

### Backend (`dashboard/src/dashboard/`)
- **`data/escalations.py`** (new) —
  - `load_queue_escalations(esc_dir: Path) -> list[dict]`: read `*.json` from the
    queue **root only** (not archive). Mirror `performance._load_escalations`'s
    error tolerance (missing dir → `[]`, bad JSON → skip+log) but **without**
    `iter_all_escalation_paths` (which walks archive). A root-only `glob('*.json')`.
  - `build_escalation_queues(config) -> dict`: enumerate per-root `data/escalations`
    dirs (reuse the `_performance_resources` enumeration shape) + the single
    `data/reconciliation/escalations` dir; return
    `{subsections: [{id, label, kind, escalations: [...]}], }` plus summary counts.
  - `resolve_owning_project(esc, roots) -> str | None`: worktree-prefix then
    task-map probe (see Definitions).
- **`config.py`** — add a `reconciliation_escalations_dir` property
  (`project_root / 'data' / 'reconciliation' / 'escalations'`) for clarity, or
  construct inline in the builder.
- **`app.py`** — `GET /api/v2/dashboard/escalations`: build queues, fetch each
  project's task map (`fetch_tasks`, concurrent + cached), shape via
  `redux_api.shape_escalations`, return `JSONResponse`. No query params (no
  window; filtering/sorting are client-side).
- **`data/redux_api.py`** — `shape_escalations(queues, task_maps) -> {ESCALATIONS:
  {subsections, summary}}`. Each subsection row carries the escalation fields the
  UI renders plus a resolved `project` label and the task card (or a
  `task_unresolved` marker). Follow the existing `shape_*` / DF_DATA-key
  convention.

### Frontend (`dashboard/src/dashboard/static/redux/`)
- **`tab_escalations.jsx`** (new) — exports `window.DF_TABS.EscalationsTab`.
  - Fold/unfold: `useOpenSet(subsectionIds, true, 'df.open.esc')` + `ProjectGroup`
    per subsection (same pattern as `OrchTab`/`MergeTab`).
  - Global filters: `usePersistedState('df.esc.filter', {levels:{0,1,2}, statuses:
    {pending}})` rendered as toolbar chips; applied across all subsections.
  - Sort: `usePersistedState('df.esc.sort', {key:'task', dir:'asc'})`; numeric-aware
    sort on `task_id`, direction toggle. Secondary sort by `timestamp`.
  - Selection + sidebar: a slide-in detail panel modeled on
    `scheduler_drawer.jsx` (`role="dialog"`, `onClose`), showing the escalation
    record (summary/detail/suggested_action/level/category/severity/agent_role/
    workflow_state/worktree/resolution) and the linked task card.
- **`app.jsx`** — add `{id:'esc', label:'Escalations'}` to `tabs`, import the
  component, add a `renderTab` case, add toolbar config (filter chips + sort +
  expand/collapse-all).
- **`shell.jsx`** — add a Rail item + `Glyph` SVG; rail count = total pending.
- **DF_DATA reference-capture:** if subsection escalation arrays are captured at
  module load (the `SHELL_PROJECTS`/`SHELL_AGENTS` pattern), register the relevant
  keys in `STABLE_ARRAY_KEYS` so the live loader mutates in place rather than
  replacing references.
- **`styles.css`** — reuse existing `ProjectGroup`, table, chip, and drawer
  styles; add only escalation-specific tweaks (level/severity color pills).

## Cross-PRD relationship & seam ownership

Sibling to `curator-dashboard-section.md` (an independent tab; no shared
mechanism). This PRD is a pure **reader** of existing escalation artifacts.

| Seam | Owner (producer) | This PRD's role |
|---|---|---|
| `<root>/data/escalations/*.json` | orchestrator escalation queue (`escalation/queue.py`) | read-only consumer; no changes |
| `data/reconciliation/escalations/*.json` | reconciliation writer | read-only consumer; no changes |
| task cards (`get_tasks`) | fused-memory MCP | call existing `fetch_tasks`; no new tool |
| `ESCALATIONS` DF_DATA key + `/api/v2/dashboard/escalations` | **this PRD** | new |

No reciprocal "the other owns it" patterns. Because v1 is read-only, there is no
write seam to any escalation server.

## Decomposition plan (one task per leaf; each names its observable signal)

1. **`data/escalations.py` reader + queue builder + project resolution.**
   Signal: a unit test feeding fixture escalation dirs (orchestrator + recon)
   returns subsections grouped correctly, root-only (archive files excluded), with
   `resolve_owning_project` picking the worktree-hinted project then falling back
   to the task-map probe.
2. **`/api/v2/dashboard/escalations` endpoint + `shape_escalations`.**
   Signal: `curl /api/v2/dashboard/escalations` returns one subsection per known
   project root + a `reconciliation` subsection, each escalation carrying its
   fields + resolved `project` + summary counts by level/status that match the
   files on disk.
3. **Tab scaffold + registration (`tab_escalations.jsx`, `app.jsx`, `shell.jsx`).**
   Signal: an **Escalations** entry appears on the rail with a pending count;
   opening it shows foldable subsections — one per orchestrator and one
   fused-memory — rendering the live escalation rows.
4. **Per-subsection fold/unfold + task-sorted table.**
   Signal: each subsection folds/unfolds independently and persists across reload;
   rows sort by `task_id` with a working ascending/descending toggle.
5. **Global level + status filter chips.**
   Signal: toggling level=2 hides non-L2 rows in every subsection; toggling
   status=resolved reveals resolved rows; selections persist across reload.
6. **Detail sidebar with escalation record + full task card.**
   Signal: clicking an escalation opens the sidebar showing its
   detail/suggested_action/level/category and the linked task's
   title/status/description; for a reconciliation escalation, the card shows the
   correctly resolved owning project; close button dismisses.

## Out of scope (v1)
- Any write action: resolve, dismiss, promote-to-L2, terminate. (Read-only.)
- Archived/historical escalations (`archive/YYYY-MM-DD/`) and any time-window
  selector.
- Per-subsection (rather than global) filters.
- Live push/websocket updates beyond the dashboard's existing poll cadence.
- Cluster/L2 member drill-down beyond showing `members`/`root_cause`/`options`
  fields in the sidebar.
- Cross-project task-id collision disambiguation beyond the worktree hint +
  first-hit-with-project-label.

## Acceptance criteria
- The **Escalations** tab appears on the rail with a pending count and renders a
  foldable subsection for every known project root plus one for fused-memory.
- Each subsection's rows match the `*.json` files in that queue's root directory
  (archive excluded), and fold/unfold persists across reload.
- Level and status filter chips restrict the visible rows across all subsections;
  selections persist across reload.
- Rows sort by task id with a working direction toggle.
- Selecting an escalation opens a sidebar with its full record and the task it
  applies to, including the correct owning project for reconciliation
  escalations.
- No escalation files are written or mutated by the dashboard.
- `pyright` and `pytest` clean for every touched package; dashboard pytest run
  from the `dashboard/` subdir.

## Open questions (tactical, implementation-time)
- **Filter granularity:** global (locked default) vs. per-subsection. Trivial to
  flip to per-subsection `filterMap` (the `OrchTab` pattern) if preferred.
- **Sidebar style:** reuse the `scheduler_drawer.jsx` slide-in, or a static
  right-rail panel. Drawer is the closer precedent; confirm at build time.
- **Recon owning-project on cache miss:** if a recon escalation's `task_id`
  isn't in any cached task map (task deleted/cancelled), show `task_id` +
  `worktree` + an "unresolved task" note rather than erroring.
- **Rail count semantics:** total pending across all queues vs. L2-only. Default
  to total pending; revisit if it's too noisy.
