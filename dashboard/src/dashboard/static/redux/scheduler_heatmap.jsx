/* scheduler_heatmap.jsx — heatmap grid showing lock-module contention per task.
   cellStateFor is executed by dashboard/tests/test_lock_chip_state.py; the
   grid itself is verified by source-structure probes in test_tab_scheduler.py.

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

// ── Pure cell classifier (no React deps) ──
//
// Returns lockChipState's { cls, hint, ownerLabel }. Membership is
// rowTouchesModule's — the same predicate the axis filter reaches, so the
// filter cannot drop a row whose cells this would colour. The lock's state is
// lockChipStateFor's, the one lock classifier the task-row chips also read, so
// a lock reads the same in the heatmap as on its chip.
function cellStateFor(row, module) {
  if (!rowTouchesModule(row, module)) return { cls: 'not-in-set', hint: 'not in lock set', ownerLabel: null };
  return window.DF_SCHED_UTILS.lockChipStateFor(module, row.task_id, row.project);
}

// ── Single heatmap cell ── `state` is cellStateFor's answer.
function HeatmapCell({ state }) {
  return <div className={`sched-cell ${state.cls}`} title={state.hint} />;
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

  // Memoised against `bounded.modules` (stable while the props are) to avoid
  // the O(n^2 * segments) scan on every re-render not triggered by a change in
  // module data (e.g. row selection).
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
        // Only the COLUMN superlative is earned, and the asymmetry is the
        // point: the server returns modules sorted `(-contention, path)` and
        // the selection takes an order-preserving prefix of that, so those
        // really are the most contended.  Nothing orders rows by contention —
        // they arrive in composition order, parked ones pulled to the front —
        // so the row axis gets a plain count rather than a claim the
        // selection cannot keep.
        <div className="sched-heatmap-cap">
          Showing {bounded.rows.length} of {bounded.rowsTotal} tasks
          {' '}and the {bounded.modules.length} most contended of {bounded.modulesTotal} modules.
        </div>
      )}
      <table className="sched-heatmap">
        <thead>
          <tr>
            <th className="sched-row-label-hd">Task</th>
            <th className="sched-skip-hd" title="Times skipped in last hour">Skip</th>
            {bounded.modules.map(m => (
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
                {bounded.modules.map(m => (
                  <td key={`${m.project || ''}/${m.path}`} className="sched-cell-td">
                    <HeatmapCell state={cellStateFor(row, m)} />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
});

window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor };
