/* Scheduler heatmap — pure presentational components.
   Renders a grid of cells showing module lock contention per task.

   Exports: window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor }
*/
const { useRef } = React;

// ── Cell state classifier (no React deps — pure function) ──

function cellStateFor(row, module) {
  /* Classify the relationship between a task row and a module column.
     Returns one of: 'free' | 'held-by-me' | 'held-by-other' | 'parked-me' | 'parked-other' | 'not-in-set'
  */
  if (!row.lock_set || !row.lock_set.includes(module.path)) return 'not-in-set';
  if (row.park_state && row.park_state.modules && row.park_state.modules.includes(module.path)) {
    return 'parked-me';
  }
  if (module.holder) {
    return module.holder === row.task_id ? 'held-by-me' : 'held-by-other';
  }
  return 'free';
}

// ── HeatmapCell ──

function HeatmapCell({ state, holder }) {
  const title = holder ? `holder: ${holder}` : state;
  return (
    <td
      className={`sched-cell ${state}`}
      title={title}
    />
  );
}

// ── SchedulerHeatmap ──

function SchedulerHeatmap({ rows, modules, onRowClick, selectedTaskId }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="sched-empty">
        No contention right now — every pending task has its lock set free.
      </div>
    );
  }

  return (
    <div className="sched-heatmap-wrap">
      <table className="sched-heatmap">
        <thead>
          <tr>
            <th className="sched-task-col">Task</th>
            <th className="sched-task-col">Skip</th>
            {modules.map(m => (
              <th key={m.path} className="sched-mod-col" title={m.path}>
                {m.path.split('/').pop()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr
              key={row.task_id}
              className={`sched-row${selectedTaskId === row.task_id ? ' selected' : ''}`}
              onClick={() => onRowClick && onRowClick(row)}
            >
              <td className="sched-task-id">
                {row.priority_differs && (
                  <span className="priority-badge" title={`effective: ${row.effective_priority}`}>!</span>
                )}
                T-{row.task_id}
              </td>
              <td className="sched-skip">{row.skip_count || 0}</td>
              {modules.map(m => (
                <HeatmapCell
                  key={m.path}
                  state={cellStateFor(row, m)}
                  holder={m.holder}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

window.DF_SCHED_HEATMAP = { SchedulerHeatmap, HeatmapCell, cellStateFor };
