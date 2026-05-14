/* tab_scheduler.jsx — Scheduler tab: lock-contention map + priority override controls.
   Behavioral invariants (no JS test runner; verified manually per task spec).

   Layout:
     1. Active-Pins strip (drag-to-reorder, X-to-unpin)
     2. Segmented Tasks / Modules toggle
     3. Tasks view: SchedulerHeatmap grid
     4. Modules view: per-module cards
     5. Right-edge SchedulerDrawer (opens on row click)

   Exports: window.DF_SCHEDULER = { SchedulerTab }
*/
const { useState: stUseState, useCallback: stUseCallback, useEffect: stUseEffect, useRef: stUseRef } = React;
const { Segmented, timeago, fmtDateTime } = window.DF_SHELL;
const { SchedulerHeatmap, cellStateFor } = window.DF_SCHED_HEATMAP;
const { SchedulerDrawer } = window.DF_SCHED_DRAWER;

const D = window.DF_DATA;

// ── Helpers ──

function fmtAge(seconds) {
  if (seconds == null || isNaN(seconds)) return '—';
  if (seconds < 60)   return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

function totalEvents(spark) {
  if (!spark || !spark.values) return 0;
  return spark.values.reduce((a, b) => a + b, 0);
}

function p50WaitSeconds(spark) {
  const total = totalEvents(spark);
  if (total < 10) return null;
  const n = (spark.labels || []).length;
  if (n === 0) return null;
  return Math.round((n * 300) / total);
}

// ── Active-Pins strip ──
// Horizontal list of pinned tasks with drag-to-reorder and X-to-unpin.
function ActivePinsStrip({ pinQueue, rows, onReorder, onUnpin }) {
  const [dragging, setDragging] = stUseState(null);
  const [dragOver, setDragOver] = stUseState(null);

  // Build ordered pin items enriched with task title
  const rowById = {};
  for (const r of (rows || [])) rowById[r.task_id] = r;

  const pins = (pinQueue || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));

  if (pins.length === 0) {
    return (
      <div className="sched-pins-empty">No pinned tasks</div>
    );
  }

  function handleDragStart(e, taskId) {
    setDragging(taskId);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', taskId);
  }

  function handleDragOver(e, taskId) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (taskId !== dragging) setDragOver(taskId);
  }

  function handleDrop(e, targetId) {
    e.preventDefault();
    if (!dragging || dragging === targetId) { setDragging(null); setDragOver(null); return; }
    // Build new ordered list by inserting dragged item before target
    const ids = pins.map(p => p.task_id).filter(id => id !== dragging);
    const idx = ids.indexOf(targetId);
    if (idx >= 0) ids.splice(idx, 0, dragging);
    else ids.push(dragging);
    onReorder(ids);
    setDragging(null);
    setDragOver(null);
  }

  function handleDragEnd() {
    setDragging(null);
    setDragOver(null);
  }

  return (
    <div className="sched-pins-strip">
      {pins.map(pin => {
        const row = rowById[pin.task_id];
        const isDragging = dragging === pin.task_id;
        const isOver = dragOver === pin.task_id;
        return (
          <div
            key={pin.task_id}
            className={'sched-pin-chip' + (isDragging ? ' dragging' : '') + (isOver ? ' drag-over' : '')}
            draggable
            onDragStart={e => handleDragStart(e, pin.task_id)}
            onDragOver={e => handleDragOver(e, pin.task_id)}
            onDrop={e => handleDrop(e, pin.task_id)}
            onDragEnd={handleDragEnd}
            title={row ? row.title : `T-${pin.task_id}`}
          >
            <span className="sched-pin-handle" title="Drag to reorder">⠿</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--accent)' }}>T-{pin.task_id}</span>
            {row && (
              <span style={{ fontSize: 11, color: 'var(--fg-2)', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {row.title}
              </span>
            )}
            <button
              className="sched-pin-remove"
              onClick={() => onUnpin(pin.task_id, row ? row.project_root : '')}
              title={`Unpin T-${pin.task_id}`}
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ── Modules view ──
// Per-module cards showing holder + waiting tasks.
function ModulesView({ modules, rows, eventsMap }) {
  if (!modules || modules.length === 0) {
    return (
      <div className="sched-empty">No module contention data available.</div>
    );
  }
  const rowById = {};
  for (const r of (rows || [])) rowById[r.task_id] = r;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {modules.map(m => {
        // Tasks waiting on this module (have it in their lock_set but don't hold it)
        const waiters = (rows || []).filter(r =>
          (r.lock_set || []).includes(m.path) && m.holder !== r.task_id
        );
        const holder = m.holder ? rowById[m.holder] : null;
        const holderAge = holder ? fmtAge(holder.age_seconds) : null;

        return (
          <div key={m.path} className="panel" style={{ padding: '10px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--fg-0)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.path}>
                  {m.path.split('/').pop()}
                </div>
                <div style={{ fontSize: 10, color: 'var(--fg-3)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.path}>
                  {m.path}
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                {m.contention > 0 && (
                  <span className="badge warn" style={{ fontSize: 10 }}>{m.contention} tasks</span>
                )}
              </div>
            </div>

            {/* Holder info */}
            {m.holder ? (
              <div style={{ fontSize: 11, marginBottom: 8, padding: '6px 8px', background: 'var(--bg-2)', borderRadius: 4 }}>
                <span style={{ color: 'var(--fg-3)' }}>held by </span>
                <span className="mono" style={{ color: 'var(--accent)' }}>T-{m.holder}</span>
                {holderAge && (
                  <span style={{ color: 'var(--fg-3)', fontSize: 10, marginLeft: 8 }}>({holderAge})</span>
                )}
                {(() => {
                  const spark = eventsMap && m.holder ? eventsMap[m.holder] : null;
                  const p50 = p50WaitSeconds(spark);
                  if (p50 == null) return null;
                  return <span style={{ color: 'var(--fg-3)', fontSize: 10, marginLeft: 8 }}>p50 ~{fmtAge(p50)}</span>;
                })()}
              </div>
            ) : (
              <div style={{ fontSize: 11, marginBottom: 8, color: 'var(--ok)' }}>free</div>
            )}

            {/* Waiters */}
            {waiters.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {waiters.slice(0, 5).map(r => (
                  <div key={r.task_id} style={{ display: 'flex', gap: 8, fontSize: 11, alignItems: 'baseline' }}>
                    <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>T-{r.task_id}</span>
                    <span style={{ color: 'var(--fg-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.title}>
                      {r.title || '—'}
                    </span>
                    {r.park_state && r.park_state.module_path === m.path && (
                      <span style={{ color: 'var(--warn)', fontSize: 10, flexShrink: 0 }}>parked</span>
                    )}
                  </div>
                ))}
                {waiters.length > 5 && (
                  <div style={{ color: 'var(--fg-3)', fontSize: 10 }}>+{waiters.length - 5} more…</div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Main SchedulerTab component ──
function SchedulerTab({ projectFilter = [] }) {
  const sched = D.SCHEDULER || {};
  const {
    rows = [],
    modules = [],
    pin_queue = [],
    events_by_task = {},
    snapshot_at = null,
    offline = false,
    offline_projects = [],
  } = sched;

  // Sub-tab: 'tasks' or 'modules'
  const [subTab, setSubTab] = stUseState('tasks');

  // Selected task for the drawer
  const [selectedTask, setSelectedTask] = stUseState(null);

  // Toast notifications
  const [toasts, setToasts] = stUseState([]);
  const toastTimers = stUseRef([]);

  stUseEffect(() => () => toastTimers.current.forEach(clearTimeout), []);

  const addToast = stUseCallback((msg, kind = 'bad') => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, msg, kind }]);
    const timer = setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
      const idx = toastTimers.current.indexOf(timer);
      if (idx >= 0) toastTimers.current.splice(idx, 1);
    }, 6000);
    toastTimers.current.push(timer);
  }, []);

  // Filter rows by projectFilter
  const visibleRows = projectFilter.length === 0
    ? rows
    : rows.filter(r => !r.project || projectFilter.includes(r.project));

  // ── Override submit ──
  const handleSubmitOverride = stUseCallback(async (body) => {
    try {
      const resp = await fetch('/api/v2/dashboard/scheduler/override', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        addToast(`Override failed (${resp.status}): ${text.slice(0, 80)}`);
      } else {
        // Close drawer on success
        setSelectedTask(null);
      }
    } catch (err) {
      addToast(`Override error: ${err.message || String(err)}`);
    }
  }, [addToast]);

  // ── Reorder pin queue ──
  const handleReorder = stUseCallback(async (taskIds) => {
    // Find project_root from any pinned task
    const firstRow = rows.find(r => taskIds.includes(r.task_id));
    const projectRoot = firstRow ? (firstRow.project_root || '') : '';
    try {
      const resp = await fetch('/api/v2/dashboard/scheduler/reorder-pin-queue', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ task_ids: taskIds, project_root: projectRoot }),
      });
      if (!resp.ok) {
        addToast(`Reorder failed (${resp.status})`);
      }
    } catch (err) {
      addToast(`Reorder error: ${err.message || String(err)}`);
    }
  }, [rows, addToast]);

  // ── Unpin (clear pinned override) ──
  const handleUnpin = stUseCallback(async (taskId, projectRoot) => {
    try {
      const resp = await fetch('/api/v2/dashboard/scheduler/clear-override', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ task_id: taskId, project_root: projectRoot || '', fields: ['pinned'] }),
      });
      if (!resp.ok) {
        addToast(`Unpin failed (${resp.status})`);
      }
    } catch (err) {
      addToast(`Unpin error: ${err.message || String(err)}`);
    }
  }, [addToast]);

  // ── Row click ──
  const handleRowClick = stUseCallback((row) => {
    setSelectedTask(prev => prev && prev.task_id === row.task_id ? null : row);
  }, []);

  const snapshotLabel = snapshot_at ? `snapshot ${fmtDateTime(snapshot_at)}` : 'no snapshot';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', position: 'relative' }}>

      {/* Offline banner */}
      {offline && (
        <div className="badge bad" style={{ padding: '6px 12px', fontSize: 11 }}>
          ⚠ Scheduler offline
          {offline_projects.length > 0 && `: ${offline_projects.join(', ')}`}
        </div>
      )}

      {/* Active-Pins strip */}
      <div className="panel" style={{ padding: '8px 12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <span style={{ fontSize: 10, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Pinned tasks
          </span>
          <span style={{ fontSize: 10, color: 'var(--fg-3)', marginLeft: 'auto' }} className="mono">
            {snapshotLabel}
          </span>
        </div>
        <ActivePinsStrip
          pinQueue={pin_queue}
          rows={visibleRows}
          onReorder={handleReorder}
          onUnpin={handleUnpin}
        />
      </div>

      {/* Sub-tab switcher */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Segmented
          options={[{ value: 'tasks', label: 'Tasks' }, { value: 'modules', label: 'Modules' }]}
          value={subTab}
          onChange={setSubTab}
        />
        <span style={{ fontSize: 11, color: 'var(--fg-3)' }}>
          {visibleRows.length} task{visibleRows.length !== 1 ? 's' : ''} · {modules.length} module{modules.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Main view + optional drawer */}
      <div style={{ flex: 1, display: 'flex', gap: 12, minHeight: 0, overflow: 'hidden' }}>
        {/* Content area */}
        <div style={{ flex: 1, overflow: 'auto', minWidth: 0 }}>
          {subTab === 'tasks' ? (
            <SchedulerHeatmap
              rows={visibleRows}
              modules={modules}
              onRowClick={handleRowClick}
              selectedTaskId={selectedTask ? selectedTask.task_id : null}
            />
          ) : (
            <ModulesView
              modules={modules}
              rows={visibleRows}
              eventsMap={events_by_task}
            />
          )}
        </div>

        {/* Drawer (slides in from the right) */}
        {selectedTask && (
          <SchedulerDrawer
            task={selectedTask}
            modules={modules}
            eventsForTask={events_by_task[selectedTask.task_id] || null}
            allRows={visibleRows}
            onClose={() => setSelectedTask(null)}
            onSubmitOverride={handleSubmitOverride}
          />
        )}
      </div>

      {/* Toast container */}
      {toasts.length > 0 && (
        <div className="curator-toasts">
          {toasts.map(t => (
            <div key={t.id} className={`curator-toast${t.kind === 'ok' ? ' ok' : ''}`}>
              {t.msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

window.DF_SCHEDULER = { SchedulerTab };
