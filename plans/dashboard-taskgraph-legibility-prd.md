# PRD: Dashboard task-graph legibility — crossing-minimized ordering + PRD progress view

**Status**: active — authored 2026-07-13 (interactive design session; no program stream).
**Mode**: bare B (self-contained display feature; one intra-PRD seam, contracted inline in §Contract).

## Goal

Two user-observable improvements to the dashboard Tasks tab
(`dashboard/src/dashboard/static/redux/tab_tasks.jsx`):

1. **The dependency graph becomes readable.** Connected components render as
   contiguous blocks, tasks with no dependencies in either direction sit in a
   compact trailing strip instead of interleaving with graph roots, and within
   each component the horizontal order minimizes dependency-arrow crossings
   (children sit near their parents). A **focus mode** filters the graph to a
   selected task's ancestor+descendant subgraph on demand.
2. **PRD progress becomes visible.** A per-project "group by PRD" toggle
   renders each PRD's tasks inside a titled cluster box carrying the PRD's
   aggregate status (same color vocabulary as task nodes), a "n/m done" count,
   and a stacked status progress bar. Boxes are ordered by their own
   dependency mini-DAG (a PRD consuming another PRD's tasks sits below it) and
   collapse to just the title+progress bar — collapsed boxes double as a PRD
   burndown summary.

## Background

Current layout (all in `tab_tasks.jsx`): vertical tiers are longest-path
(`computeTiers`, :19 — `tier = 1 + max(deps' tiers)`), but horizontal order
within a tier is a bare status-priority sort (`rows` memo, :132–140) with
fetch order as tiebreak — the graph structure plays no part, so edges cross
arbitrarily and unrelated singleton tasks interleave with real graph roots.
Nodes are flexbox divs; edges are cubic Béziers in an SVG overlay computed
from `getBoundingClientRect()` (`TaskGraphEdges`, :41–125), so the edge
renderer is layout-agnostic and needs no changes for any of this. A
click-to-highlight neighborhood feature already exists (`neighborhood` memo,
:143–164; CSS class toggles only).

PRD provenance already exists on tasks: the `/prd` decompose skill writes
`metadata.source = "prd-decomposition"` + `metadata.prd_path` +
`metadata.prd_task_label` (`skills/prd/references/decompose-mode.md:66–74`).
Store census 2026-07-13: 376 tasks carry `prd_path`; every
`source='prd-decomposition'` task has one; ~90 older tasks use legacy keys
`prd` / `prd_ref` (sometimes with `#anchor`/`§` suffixes); ~60% of currently
pending tasks carry `prd_path` (the rest were filed by escalation / review /
recon flows and genuinely belong to no PRD). The dashboard backend keeps the
metadata dict internally (`dashboard/src/dashboard/data/tasks.py:109`) but
`_build_task_row` (`data/active_tasks.py:105–147`) forwards only
`files`/`train`/`external_deps` — the client never sees PRD provenance.
Done/cancelled rows are bounded buckets (≤50/project, `app.py:544`) shipped
with `deps: []` (`active_tasks.py:246–247`), so PRD-member completion state
is invisible client-side today.

Prior art to avoid: `data/orchestrator.py:203–208` derives a `prd` by
regex-parsing live orchestrator process command lines — that channel misses
the modern fused-memory-scheduled flow and is not used here.

## Sketch of approach

### Part 1 — ordering (Sugiyama ordering phase, hand-rolled)

New **plain-JS** (no JSX) module
`dashboard/src/dashboard/static/redux/graph_layout.js`, loaded via script tag
(exported as `window.DF_GRAPH_LAYOUT`) and imported by a node test harness.
`computeTiers` moves there unchanged. New pure functions:

- `partitionComponents(tasks)` → weakly-connected components (over the
  currently filtered task set, same as tiers today) + a `singletons` list
  (tasks with no in-set deps in either direction).
- `orderRows(component, tiers, statusOrder)` → per-tier arrays ordered by
  **barycenter sweeps**: initial permutation = the existing status-priority
  sort (preserving today's "blocked leftmost" instinct as tiebreak); then
  3–4 alternating sweeps (top-down keyed on mean of dep positions in rows
  above, bottom-up keyed on dependent positions below), positions normalized
  to [0,1] per row before averaging; finish with a greedy adjacent-swap
  (transpose) pass that keeps any swap strictly reducing the crossing count.
  Deterministic (stable sorts, fixed sweep count). **No dummy/virtual nodes
  in v1** (see §Out of scope).
- `countCrossings(rows, edges)` — used by the transpose pass and the tests.

`TaskGraph` renders components as contiguous blocks (each component's tiers
stacked, components separated), then the singleton strip. `TaskGraphEdges`
is untouched — it reads rendered positions.

**Focus mode** (explicit action, NOT reorder-in-place — re-sorting the whole
graph on selection destroys the spatial map and moves the clicked node out
from under the cursor): with a task selected, a "focus" affordance filters
the graph to the selected neighborhood (the existing `neighborhood` set);
tiers/ordering recompute over the filtered set exactly as they already do
for status filters. Esc or "clear" restores the full view.

**Testing**: node v22 is on the host. A pytest wrapper
(`dashboard/tests/test_graph_layout_js.py`) subprocess-runs
`node --test dashboard/tests/js/` against fixture graphs; this makes the
layout logic the first JS in the dashboard with CI coverage. Fixtures are
constructed so the status-sort baseline provably has crossings that
barycenter removes (e.g. two parents whose children are filed in inverted
order).

### Part 2 — PRD progress view

**Backend** (`data/active_tasks.py` + `data/tasks.py`):
- Every task row gains a **`prd`** field: coalesce
  `metadata.prd_path → metadata.prd → metadata.prd_ref`, strip `#anchor` and
  `§…` suffixes, else `null`. (See §Contract.)
- **Live-PRD member completion**: for each PRD value that has ≥1 member in
  `_ACTIVE_STATUSES`, its done/cancelled members are included as rows even
  beyond the bounded 50-cap, **with real `deps` populated** (they keep
  `started: 0` + `completed`). This makes "n/m done" counts and intra-box
  edges truthful. `DONE_COUNTS` semantics unchanged.

**Frontend** (`tab_tasks.jsx` + `styles.css`):
- A per-project **"group by PRD"** toggle (persisted via the existing
  `tasksPersistedState` hook, :9–16).
- When on: tasks bucket by `prd` (null → a trailing untitled "no PRD" group).
  Each PRD renders as a titled outlined box; the Part-1 layout
  (components + barycenter) runs **per box** on the induced subgraph — the
  same "tiers over a filtered list" mechanism used per project today.
  Cross-box dependency edges keep drawing correctly for free (the SVG
  overlay reads global rendered positions).
- **Box ordering**: condense cross-box edges into a PRD-level mini-DAG and
  tier it with the same `computeTiers`; tiebreak by activity
  (blocked+in-progress count desc, then pending count).
- **Box chrome**: title = `prd` basename with `-prd.md` stripped (full path
  in tooltip); outline + title pip use the aggregate status —
  any member `blocked` → blocked; else any `in-progress`/`merge-deferred` →
  in-progress; else any `pending`/`deferred` → pending; else all `done` →
  done; else cancelled. "n/m done" counts **all** members present in the
  payload regardless of the status display filter (nodes shown still respect
  the filter). A thin stacked bar (existing `window.DF_CHARTS.PALETTE`
  colors: done/in-progress/blocked/pending segments) sits in the title bar.
  Boxes collapse to title bar + bar; all-done PRDs start collapsed.

## Resolved design decisions

1. **Hand-rolled barycenter, no graph library.** dagre/ELK compute absolute
   coordinates, which fights the responsive flexbox rendering and adds a CDN
   dependency; the ordering phase alone is ~50–80 lines and tiers already
   exist.
2. **Component grouping + singleton strip before any barycenter work** —
   biggest single legibility win; singletons stop stretching rows and
   forcing wrap.
3. **Focus mode instead of reorder-on-highlight** (spatial-stability
   rationale above). Reorder-in-place with FLIP animation is explicitly out
   of scope for v1.
4. **Layout logic extracted to plain JS with node+pytest CI coverage** — the
   in-browser Babel JSX is untestable today; pure functions move to a module
   both the page and `node --test` can load. node v22.22.3 verified on host.
5. **PRD grouping key is coalesced server-side** (`prd_path → prd → prd_ref`,
   anchors stripped) so the client sees one canonical `prd` string.
6. **Schema promotion of `prd_path` is NOT this PRD** — task 2330 (Tier A of
   its vocabulary reconciliation) already owns promoting
   `prd_path`/`prd_task_label`/`source` to typed fields or a blessed
   allowlist. The dashboard reads the raw metadata dict via `get_tasks`, so
   there is no runtime dependency on 2330 in either direction.
7. **Done-member inclusion is always-on** (no query param) — payload is
   bounded by live-PRD size; revisit only if `/api/v2/dashboard/tasks`
   payload becomes a measured problem (§Open questions).
8. **ε serializes after γ** (both edit `tab_tasks.jsx`/`styles.css`) —
   single-writer ordering rather than lock-contention churn.

## Contract: task-row `prd` field (the δ→ε seam)

- **Field**: `prd: string | null` on every row emitted by
  `collect_tasks_with_counts` (active AND bounded/exempted terminal rows).
- **Value**: `metadata.prd_path` if a non-empty string; else `metadata.prd`;
  else `metadata.prd_ref`; anchors stripped (`#…` and `§…` suffixes,
  whitespace-trimmed); else `null`. Never absent, never `''`.
- **Invariant**: two tasks from the same PRD yield byte-identical `prd`
  strings (the coalescer is the single normalization point; the client never
  re-normalizes).
- **Terminal-member exemption**: a done/cancelled task whose `prd` matches a
  PRD with ≥1 member in `_ACTIVE_STATUSES` (`active_tasks.py:44`) is emitted
  with populated `deps`, exempt from the 50-cap; other terminal rows are
  unchanged (cap applies, `deps: []`).

## Pre-conditions for activating

None — all substrate exists on main (deps edge list client-side,
`metadata.prd_path` in store, metadata dict available server-side, node on
host, persisted-state hook and palette already wired into this file).

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task 2330 (task-metadata census / vocabulary reconciliation) | none at runtime | `prd_path` typed-schema promotion | task 2330 | excluded here by design (decision 6) |
| `plans/dashboard-alignment-prd.md` (M3) | builds on | `data/active_tasks.py` shapes it refactored | landed (no pending tasks touch these files, checked 2026-07-13) | none |
| `plans/fleet-cockpit-prd.md` | none | different surface (terminal cockpit) | — | none |

No contested seams; no reciprocal-ownership statements.

## Decomposition plan

| # | Task | Modules | Prereqs | Observable signal |
|---|---|---|---|---|
| α | Extract `graph_layout.js` (tiers + components + singletons + barycenter + crossing count) with node+pytest CI harness | `dashboard/.../static/redux/graph_layout.js`, `dashboard/tests/js/`, `dashboard/tests/test_graph_layout_js.py` | — | **Intermediate** — unlocks β. CI: pytest-wrapped `node --test` asserts crossings(barycenter) < crossings(status-sort) on the tangled fixture, ≤ on all fixtures; components contiguous; singleton partition exact; deterministic across runs |
| β | Wire component/barycenter ordering into `TaskGraph` (component blocks + singleton strip; status sort demoted to tiebreak) | `tab_tasks.jsx`, `styles.css`, `index.html` (script tag + `?v=` bump) | α | **Leaf** — live dashboard Tasks tab renders each connected component as a contiguous block with children horizontally near their parents, singletons in a labeled trailing strip; existing `test_index_html.py` `?v=` guard passes |
| γ | Focus mode (filter graph to selected neighborhood; Esc/clear restores) | `tab_tasks.jsx`, `styles.css` | β | **Leaf** — with a task selected, activating "focus" re-renders the graph containing exactly its ancestors+descendants (tiers recomputed over the subset); Esc restores the full graph |
| δ | Backend `prd` field + live-PRD terminal-member rows with deps (per §Contract) | `data/active_tasks.py`, `dashboard/tests/test_active_tasks.py` | — | **Intermediate** — unlocks ε. API: `GET /api/v2/dashboard/tasks` rows carry `prd` per §Contract; pytest (existing mock patterns) asserts a done member of a live PRD appears beyond the cap with populated `deps`, legacy `prd`/`prd_ref` keys coalesce, anchors stripped |
| ε | PRD grouping view (toggle, titled status-aggregated boxes, per-box layout, mini-DAG box order, stacked progress bar, collapse) | `tab_tasks.jsx`, `styles.css` | β, δ, γ (single-writer ordering, decision 8) | **Leaf** — toggling "group by PRD" on the live dashboard shows each PRD's tasks inside a titled box with status-colored outline, accurate "n/m done" (counting payload members regardless of display filter), stacked status bar, collapsible; a PRD consuming another PRD's tasks renders below it; "no PRD" group trails |

Signal notes: β/γ/ε are UI-state-change signals observable through the
product (dashboard is the product surface); α/δ are intermediates with named
in-batch consumers plus CI/API-observable verification. G6: α's
strictly-fewer-crossings premise is achievable by fixture construction
(baseline crossing count is fixed and known; the transpose pass only accepts
strictly-reducing swaps); ε's "n/m done" premise requires terminal member
rows, delivered upstream by δ.

## Out of scope

- **Dummy/virtual nodes** for multi-tier edge ordering (full Sugiyama) —
  revisit only if long edges remain tangled after v1.
- **Reorder-in-place on selection** (FLIP-animated or otherwise).
- **`prd_path` schema promotion** and any `shared/task_metadata.py` change —
  owned by task 2330.
- **Legacy-key data migration** (`prd`/`prd_ref` → `prd_path` rewrites in the
  store) — the coalescer absorbs them read-side; 2330 Tier B keeps aliases
  warned.
- **Browser-driving JS UI test harness** — CI coverage here is the pure
  layout module only; UI signals stay human-verified.
- **Cross-project PRD boxes** — each project section groups only its own
  tasks (a `reify:docs/prds/...` prd value groups within its own project's
  section).
- Graph-library adoption (dagre/ELK/cytoscape).

## Open questions (surfaced but not decided in this session)

1. **Sweep count / transpose on-off tuning.** 3–4 sweeps + transpose is the
   default; tune against real snapshots. **Suggested resolution:** keep
   whatever the fixture suite shows converged. Decide during α.
2. **Payload growth from done-member inclusion.** Always-on per decision 7.
   **Suggested resolution:** if `/api/v2/dashboard/tasks` exceeds ~1 MB in
   practice, add a `?prd_members=` opt-out then. Decide if observed.
3. **Focus-mode affordance placement** (button in detail panel vs
   double-click vs both). **Suggested resolution:** button in the detail
   panel + double-click shortcut. Decide during γ.
4. **Singleton-strip collapsibility.** **Suggested resolution:**
   implementer's call during β.
