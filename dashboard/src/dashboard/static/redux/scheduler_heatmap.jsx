/* scheduler_heatmap.jsx — heatmap grid showing lock-module contention per task.
   Behavioral invariants (no JS test runner; verified manually per task spec).

   Exports: window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor }
*/

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
  // Modules are project-scoped (keyed by `(project, path)` on the server).
  // A row from project B sharing a file path with project A's module entry
  // is NOT contending for the same lock, so the cell must remain blank.
  // Falsy `module.project` (legacy/single-project mode) skips the check.
  if (module.project && row.project && module.project !== row.project) {
    return 'not-in-set';
  }
  const lockSet = row.lock_set || [];
  if (!lockSet.includes(module.path)) return 'not-in-set';

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
function SchedulerHeatmap({ rows, modules, onRowClick, selectedTaskId }) {
  const { useState } = React;

  // Pre-compute parked rows keyed by `(project, module)` so cellStateFor
  // can classify 'parked-by-other' without leaking parks across projects
  // (two projects sharing a file path must not appear to park each other).
  // park_state.modules is a list — register one entry per parked module.
  const parkedByModule = {};
  for (const row of (rows || [])) {
    const ps = row.park_state;
    for (const m of (ps && ps.modules) || []) {
      parkedByModule[`${row.project || ''}/${m}`] = row.task_id;
    }
  }

  // Enrich each module with `parked_by` before passing to cellStateFor.
  // Use the module's owning project to look up the project-scoped park map.
  const enriched = (modules || []).map(m => ({
    ...m,
    parked_by: parkedByModule[`${m.project || ''}/${m.path}`] || null,
  }));

  if (!rows || rows.length === 0) {
    return (
      <div className="sched-empty">
        No contention right now — every pending task has its lock set free.
      </div>
    );
  }

  const labelMap = (window.DF_SCHED_UTILS || {}).disambiguateLabels
    ? window.DF_SCHED_UTILS.disambiguateLabels(enriched.map(m => m.path))
    : null;

  return (
    <div className="sched-heatmap-wrap">
      <table className="sched-heatmap">
        <thead>
          <tr>
            <th className="sched-row-label-hd">Task</th>
            <th className="sched-skip-hd" title="Times skipped in last hour">Skip</th>
            {enriched.map(m => (
              <th
                key={`${m.project || ''}/${m.path}`}
                className="sched-col-label"
                title={m.project ? `${m.path} · ${m.project}` : m.path}
              >
                <span className="sched-col-path">{labelMap ? labelMap.get(m.path) : m.path}</span>
                {m.contention > 1 && (
                  <span className="sched-col-count">{m.contention}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => {
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
                {enriched.map(m => {
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
}

window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor };
