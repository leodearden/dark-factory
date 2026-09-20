/* scheduler_heatmap.jsx — heatmap grid showing lock-module contention per task.
   Behavioral invariants (no JS test runner; verified manually per task spec).

   The grid renders a BOUNDED selection of the rows x modules cross-product,
   never the raw props: unbounded, the 2026-09-20 live snapshot is 2,991 x
   4,302 = 12,867,282 cells and the browser renderer dies before it finishes.
   Which rows and columns survive is decided by scheduler_heatmap_bounds.js,
   a plain classic script so the bound can be proven executably against a
   production-scale fixture (dashboard/tests/js/scheduler_heatmap_bounds.test.mjs);
   that this component actually CONSUMES it — what makes the cap structural
   rather than advisory — is pinned in dashboard/tests/test_tab_scheduler.py.

   Exports: window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor }
*/

// Module-scope destructure with no `|| {}` fallback, matching tab_scheduler.jsx:15.
// index.html loads scheduler_heatmap_bounds.js as a classic script, so it runs
// before every Babel-transformed tag; the ordering is enforced by
// test_index_html.py::test_scheduler_heatmap_bounds_js_loads_before_scheduler_heatmap.
const { boundHeatmapAxes, rowTouchesModule } = window.DF_SCHED_HEATMAP_BOUNDS;

// ── Pure cell-state classifier (no React deps) ──
//
// Returns one of:
//   'free'            module is in the task's lock set and currently unblocked
//   'held-by-other'   another task is running with this module held
//   'parked-by-me'    this task is parked waiting on this module
//   'parked-by-other' another task is parked waiting on this module
//   'not-in-set'      module is not in this task's declared lock set
//
// `module` may carry an optional `parked_by` field (task_id string) that
// SchedulerHeatmap pre-computes from the full rows list before calling here.
function cellStateFor(row, module) {
  // Membership — the project-scope guard AND the lock_set check — is owned by
  // rowTouchesModule, which the axis filter also reaches.  One source, so the
  // filter cannot drop a row whose cells this renderer would have coloured.
  if (!rowTouchesModule(row, module)) return 'not-in-set';

  // This task is parked waiting on this specific module.  park_state.modules
  // is a list of parked module keys (server snapshot shape), not a scalar.
  const ps = row.park_state;
  if (ps && (ps.modules || []).includes(module.path)) return 'parked-by-me';

  // Another task is currently running with this module held
  if (module.holder && module.holder !== row.task_id) return 'held-by-other';

  // Another task is parked waiting on this module
  if (module.parked_by && module.parked_by !== row.task_id) return 'parked-by-other';

  return 'free';
}

// ── Single heatmap cell ──
//
// Props:
//   state  — one of the five state strings above
//   holder — task_id to show in the tooltip (for held-by-other / parked-by-other)
function HeatmapCell({ state, holder }) {
  const label = {
    'free':            'free',
    'held-by-other':   holder ? `held by ${holder}` : 'held by another task',
    'parked-by-me':    'parked (me)',
    'parked-by-other': holder ? `parked by ${holder}` : 'parked by another task',
    'not-in-set':      'not in lock set',
  }[state] || state;

  return <div className={`sched-cell ${state}`} title={label} />;
}

// ── Full heatmap grid ──
//
// Props:
//   rows          list[dict]  — composed task rows (from SCHEDULER.rows)
//   modules       list[dict]  — sorted module-contention list (from SCHEDULER.modules)
//   onRowClick    fn(row)     — called when a task row is clicked
//   selectedTaskId string     — task_id of the currently-selected row (or null)
//
// Memoised because app.jsx ticks a 1 Hz clock (`setInterval(() => setNow(...), 1000)`)
// that re-renders the active tab's subtree whether or not data changed. This
// component takes its data as PROPS, so between ticks they are referentially
// identical and the whole grid is skipped; on a real 5s refresh
// window.DF_DATA.SCHEDULER yields new array identities and it re-renders.
//
// The inner function stays NAMED — React DevTools keeps a useful label, and
// the source-structure probes in test_tab_scheduler.py resolve it by name via
// extract_function_body, which raises rather than passing vacuously on a miss.
const SchedulerHeatmap = React.memo(function SchedulerHeatmap({ rows, modules, onRowClick, selectedTaskId }) {
  const { useState, useMemo } = React;

  // Choose the axes that will actually render.  Memoised on the two props so
  // the selection is not recomputed on a re-render driven by anything else
  // (row selection, or App's 1 Hz clock tick).
  const bounded = useMemo(() => boundHeatmapAxes({ rows, modules }), [rows, modules]);

  // Pre-compute parked rows keyed by `(project, module)` so cellStateFor
  // can classify 'parked-by-other' without leaking parks across projects
  // (two projects sharing a file path must not appear to park each other).
  // park_state.modules is a list — register one entry per parked module.
  //
  // Scanned over ALL rows, not the bounded ones, and deliberately so: a module
  // may be parked by a task whose own row did not survive selection, and that
  // cell must still read 'parked-by-other' rather than 'free'.
  const parkedByModule = {};
  for (const row of (rows || [])) {
    const ps = row.park_state;
    for (const m of (ps && ps.modules) || []) {
      parkedByModule[`${row.project || ''}/${m}`] = row.task_id;
    }
  }

  // Enrich each RENDERED module with `parked_by` before passing to cellStateFor.
  // Use the module's owning project to look up the project-scoped park map.
  const enrichedColumns = bounded.modules.map(m => ({
    ...m,
    parked_by: parkedByModule[`${m.project || ''}/${m.path}`] || null,
  }));

  // Memoised against `bounded.modules` (stable while the props are) to avoid
  // the O(n^2 * segments) scan on every re-render not triggered by a change in
  // module data (e.g. row selection).  Keying on `enrichedColumns` would not
  // work because it is a new array reference on every render — the memo would
  // never hit.  The path list is identical: enrichment only adds `parked_by`
  // metadata, which doesn't affect the labels.
  // Computing it over the BOUNDED paths rather than all of them is most of the
  // win: at most MAX_HEATMAP_COLS paths instead of the full module list, on
  // every 5s poll.
  // Hook MUST run on every render path, so it precedes the early-return guard
  // below — `rows` legitimately toggles empty/non-empty on a live dashboard,
  // and a conditionally-called hook would change the hook count and crash.
  const labelMap = useMemo(
    () => (window.DF_SCHED_UTILS || {}).disambiguateLabels
      ? window.DF_SCHED_UTILS.disambiguateLabels(bounded.modules.map(m => m.path))
      : null,
    [bounded.modules]
  );

  // Zero COLUMNS is as empty as zero rows: a table with a Task column and
  // nothing to show against it is not a heatmap.  The existing copy is
  // literally true in that state — every surviving task's lock set is free.
  if (bounded.rows.length === 0 || bounded.modules.length === 0) {
    return (
      <div className="sched-empty">
        No contention right now — every pending task has its lock set free.
      </div>
    );
  }

  return (
    <div className="sched-heatmap-wrap">
      {(bounded.rowsTruncated || bounded.modulesTruncated) && (
        <div className="sched-heatmap-cap">
          Showing the {bounded.rows.length} most contended of {bounded.rowsTotal} tasks
          {' '}and {bounded.modules.length} of {bounded.modulesTotal} modules.
        </div>
      )}
      <table className="sched-heatmap">
        <thead>
          <tr>
            <th className="sched-row-label-hd">Task</th>
            <th className="sched-skip-hd" title="Times skipped in last hour">Skip</th>
            {enrichedColumns.map(m => (
              <th
                key={`${m.project || ''}/${m.path}`}
                className="sched-col-label"
                title={m.project ? `${m.path} · ${m.project}` : m.path}
              >
                <span className="sched-col-path">{labelMap ? labelMap.get(m.path) : m.path.split('/').pop()}</span>
                {m.contention > 1 && (
                  <span className="sched-col-count">{m.contention}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bounded.rows.map(row => {
            const isSelected = row.task_id === selectedTaskId;
            return (
              <tr
                key={`${row.project || ''}/${row.task_id}`}
                className={'sched-row' + (isSelected ? ' selected' : '')}
                onClick={() => onRowClick && onRowClick(row)}
                style={{ cursor: 'pointer' }}
              >
                <td className="sched-row-label">
                  <div className="sched-row-meta">
                    <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>
                      T-{row.task_id}
                    </span>
                    {row.priority_differs && (
                      <span
                        className="badge warn sched-badge"
                        title={`Priority override: declared=${row.declared_priority} effective=${row.effective_priority}`}
                      >
                        ↑{row.effective_priority}
                      </span>
                    )}
                    {row.pinned && (
                      <span className="badge ok sched-badge" title="Pinned">pin</span>
                    )}
                    {row.reserve_now && (
                      <span className="badge bad sched-badge" title="Reserve-now active">rsv</span>
                    )}
                  </div>
                  <div className="sched-row-title" title={row.title}>
                    {row.title || '—'}
                  </div>
                </td>
                <td className="sched-skip">
                  <span className="mono" style={{ fontSize: 10 }}>{row.skip_count || 0}</span>
                </td>
                {enrichedColumns.map(m => {
                  const state = cellStateFor(row, m);
                  const holder =
                    state === 'held-by-other'   ? m.holder :
                    state === 'parked-by-other' ? m.parked_by : null;
                  return (
                    <td key={`${m.project || ''}/${m.path}`} className="sched-cell-td">
                      <HeatmapCell state={state} holder={holder} />
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
});

window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor };
