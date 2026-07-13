# Capability manifest — plans/dashboard-taskgraph-legibility-prd.md

Bindings audited 2026-07-13 against main @ 705c01da70 (PRD commit). Evidence
verified by direct read/grep in the authoring session; store census via
read-only inspection of the task registry the same day.

## Leaf β — wire component/barycenter ordering into TaskGraph

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Per-task `deps` edge list reaches the client | `dashboard/src/dashboard/data/active_tasks.py:208-217` builds `deps`; served via `app.py:530` `api_tasks` → `data.js` → `DF_DATA.ACTIVE_TASKS` → consumed `tab_tasks.jsx:172` (production path) | PASS wired |
| Replaceable ordering point exists | `tab_tasks.jsx:132-140` — `rows` memo, status-priority sort is the only ordering | PASS wired |
| `graph_layout.js` (tiers, components, singletons, barycenter, crossing count) | producer: task α, upstream of β; extent = the five named functions incl. `countCrossings` | PASS producer-upstream |
| CI harness for the module | node v22.22.3 verified on host 2026-07-13; pytest subprocess wrapper delivered by α (same producer, same extent) | PASS producer-upstream |
| `?v=` cache-bust guard | `dashboard/tests/test_index_html.py` exists (test listing verified) | PASS wired |

G6 note (numeric premise): α's CI signal asserts crossings(barycenter) <
crossings(status-sort) on the tangled fixture. Achievability basis: the
fixture is constructed with a known baseline crossing count (inverted
parent/child filing order ⇒ ≥1 removable crossing) and the transpose pass
accepts only strictly-reducing swaps, so strict improvement on that fixture
is guaranteed by construction; the global assertion is ≤, not <.

## Leaf γ — focus mode

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Neighborhood (ancestors+descendants) set | `tab_tasks.jsx:143-164`, consumed in render `:172-183` (production path) | PASS wired |
| Tiers/ordering recompute over filtered subsets | `computeTiers` is list-parametric (`tab_tasks.jsx:19,131`); TasksTab already renders filtered lists (`:434,460`) | PASS wired |
| Selection state to hang the mode on | `selectedId`/`onSelect` wired (`tab_tasks.jsx:362,396,460`) | PASS wired |

## Leaf ε — PRD grouping view

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `prd: string\|null` on every row (§Contract) | producer: task δ, upstream of ε; extent = coalesce `prd_path→prd→prd_ref`, anchor-strip, null-never-absent, on active AND terminal rows | PASS producer-upstream |
| Terminal members of live PRDs present with populated `deps`, cap-exempt | producer: task δ, upstream; extent = §Contract terminal-member-exemption clause | PASS producer-upstream |
| `metadata.prd_path` populated in the store (field-population check) | census 2026-07-13: 376 tasks carry `prd_path`; every `source='prd-decomposition'` row has one; values are real non-sentinel paths (sampled task 2499: `plans/tier1-prompt-optimization-prd.md`); legacy `prd` (62) / `prd_ref` (30) absorbed by δ's coalescer | PASS populated |
| Metadata dict available server-side to δ | `dashboard/src/dashboard/data/tasks.py:97-109` `_shape_task` retains `metadata`; consumed by `_shape_one_project` (`active_tasks.py:181`) | PASS wired |
| Persisted toggle state hook | `tasksPersistedState` `tab_tasks.jsx:9-16`, already used for filters `:365` | PASS wired |
| Status palette for stacked bar | `window.DF_CHARTS.PALETTE` consumed at `tab_tasks.jsx:4,84` today | PASS wired |
| Per-group component/barycenter layout | producer: tasks α (module) + β (render path), both upstream of ε | PASS producer-upstream |

## Intermediates (verification evidence, not leaf bindings)

- **α**: pure-JS module + fixtures; consumer β named in-batch. node harness
  premise verified above.
- **δ**: API-observable via existing mock patterns
  (`dashboard/tests/test_active_tasks.py` exists); `_ACTIVE_STATUSES` at
  `active_tasks.py:44`; bounded-bucket cap at `app.py:544`; consumer ε named
  in-batch.

No grammar fixtures (no DSL surface), no numeric floors (no absolute-accuracy
bounds), no rejection assertions. No FAIL bindings — batch clear to queue.
