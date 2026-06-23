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
const { useState: stUseState, useCallback: stUseCallback, useEffect: stUseEffect, useRef: stUseRef, useMemo: stUseMemo } = React;
const { Segmented, ProjectChips, timeago, fmtDateTime } = window.DF_SHELL;
const { SchedulerHeatmap, cellStateFor } = window.DF_SCHED_HEATMAP;
const { SchedulerDrawer } = window.DF_SCHED_DRAWER;
// Shared helpers loaded by scheduler_utils.jsx (must come before this script).
const { fmtAge, totalEvents, avgWaitSeconds, labelFor } = window.DF_SCHED_UTILS;

const D = window.DF_DATA;

// Composite key for a pin entry — taskmaster task_ids are project-scoped,
// so the same numeric id can appear in two projects' pin queues.  Keying
// pins by '${project}/${task_id}' lets the React strip render duplicates
// without collapsing rows and lets the multi-project reorder handler
// group entries by their owning project_root.
function pinKey(pin) {
  return `${pin.project || ''}/${pin.task_id}`;
}

// ── Active-Pins strip ──
// Horizontal list of pinned tasks with drag-to-reorder and X-to-unpin.
function ActivePinsStrip({ pinQueue, rows, onReorder, onUnpin }) {
  const [dragging, setDragging] = stUseState(null);
  const [dragOver, setDragOver] = stUseState(null);

  // Build ordered pin items enriched with task title.  We key by composite
  // '${project}/${task_id}' to prevent cross-project collisions when two
  // projects each have a row with task_id='1'.  Memoised against `rows` so
  // we don't rebuild the O(n) index on every parent re-render of the live
  // 5s poll cycle.
  const rowByCompositeKey = stUseMemo(() => {
    const idx = {};
    for (const r of (rows || [])) {
      idx[`${r.project || ''}/${r.task_id}`] = r;
    }
    return idx;
  }, [rows]);

  const pins = (pinQueue || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));

  if (pins.length === 0) {
    return (
      <div className="sched-pins-empty">No pinned tasks</div>
    );
  }

  function handleDragStart(e, key) {
    setDragging(key);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', key);
  }

  function handleDragOver(e, key) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (key !== dragging) setDragOver(key);
  }

  function handleDrop(e, targetKey) {
    e.preventDefault();
    if (!dragging || dragging === targetKey) { setDragging(null); setDragOver(null); return; }
    // Build new ordered list of composite keys by inserting dragged item
    // before target.  Composite keys are project-qualified, so the upstream
    // reorder handler can route per-project without ambiguity.
    const keys = pins.map(pinKey).filter(k => k !== dragging);
    const idx = keys.indexOf(targetKey);
    if (idx >= 0) keys.splice(idx, 0, dragging);
    else keys.push(dragging);
    onReorder(keys);
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
        const key = pinKey(pin);
        const row = rowByCompositeKey[key];
        const isDragging = dragging === key;
        const isOver = dragOver === key;
        // Prefer the pin's own project_root (server-tagged) over the row's —
        // the row may be absent on the first poll after a remote pin, but
        // pin.project_root is always set by collect_scheduler_state.
        const pinProjectRoot = pin.project_root || (row ? row.project_root : '') || '';
        return (
          <div
            key={key}
            className={'sched-pin-chip' + (isDragging ? ' dragging' : '') + (isOver ? ' drag-over' : '')}
            draggable
            onDragStart={e => handleDragStart(e, key)}
            onDragOver={e => handleDragOver(e, key)}
            onDrop={e => handleDrop(e, key)}
            onDragEnd={handleDragEnd}
            title={row ? row.title : `T-${pin.task_id}`}
          >
            <span className="sched-pin-handle" title="Drag to reorder">⠿</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--accent)' }}>T-{pin.task_id}</span>
            {pin.project && (
              <span style={{ fontSize: 9, color: 'var(--fg-3)' }} title={pin.project}>·{pin.project}</span>
            )}
            {row && (
              <span style={{ fontSize: 11, color: 'var(--fg-2)', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {row.title}
              </span>
            )}
            <button
              className="sched-pin-remove"
              onClick={() => onUnpin(pin.task_id, pinProjectRoot)}
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

// ── Park Stacks section ──
// Rendered in the Modules sub-tab below ModulesView.
// For each module with a non-empty park_stack, shows the full LIFO stack
// TOP→BOTTOM (reverse of the bottom→top snapshot order) with owner, tier,
// age, active-top vs shadowed, and live/dead indicator.
// `rows` (all visible rows, including synthetic stranded rows from the server)
// is used to resolve each owner's project_root via the composite-key index —
// stranded rows carry project_root keyed by ${project}/${task_id}.
// `onEvict(taskId, projectRoot)` is called when the operator clicks evict.
function ParkStacksSection({ modules, rows, onEvict }) {
  const parked = (modules || []).filter(m => m.park_stack && m.park_stack.length > 0);
  if (parked.length === 0) return null;

  // Build a project-qualified row index to resolve each owner's project_root.
  // Synthetic stranded rows (_stranded_park_rows, task γ) carry project_root +
  // project + task_id (== owner), so the composite key resolves it without
  // modifying the data layer.  Mirrors the rowByCompositeKey pattern in ModulesView.
  const rowByCompositeKey = stUseMemo(() => {
    const idx = {};
    for (const r of (rows || [])) {
      idx[`${r.project || ''}/${r.task_id}`] = r;
    }
    return idx;
  }, [rows]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 16 }}>
      <div style={{ fontSize: 10, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
        Module Park Stacks
      </div>
      {parked.map(m => {
        // Render TOP→BOTTOM (reverse of bottom→top snapshot order: stack[-1] is active top)
        const stack = [...(m.park_stack || [])].reverse();
        return (
          <div key={`${m.project || ''}/${m.path}`} className="panel" style={{ padding: '10px 14px' }}>
            <div className="mono" style={{ fontSize: 11, color: 'var(--fg-0)', marginBottom: 6 }} title={m.path}>
              {m.path}
              {m.project && <span style={{ marginLeft: 6, color: 'var(--fg-3)', fontSize: 10 }}>·{m.project}</span>}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {stack.map((entry, idx) => {
                const isActive = !entry.shadowed;
                const isDead = !entry.live;
                const ageStr = entry.installed_at ? timeago(entry.installed_at) : null;
                // Resolve owner's project_root from the row index (stranded rows carry it).
                const ownerRow = rowByCompositeKey[`${m.project || ''}/${entry.owner}`] || {};
                const ownerProjectRoot = ownerRow.project_root || '';
                return (
                  <div
                    key={`${entry.owner}-${idx}`}
                    style={{
                      display: 'flex', alignItems: 'baseline', gap: 8, fontSize: 11,
                      padding: '4px 8px',
                      background: isActive ? 'var(--bg-2)' : 'transparent',
                      borderRadius: 4,
                      opacity: isDead ? 0.6 : 1,
                    }}
                  >
                    <span className="mono" style={{ color: isDead ? 'var(--warn)' : 'var(--accent)', fontSize: 10 }}>
                      T-{entry.owner}
                    </span>
                    <span style={{ color: 'var(--fg-3)', fontSize: 10 }}>
                      tier {entry.rank}
                    </span>
                    {ageStr && (
                      <span style={{ color: 'var(--fg-3)', fontSize: 10 }}>{ageStr}</span>
                    )}
                    {isActive && (
                      <span className="badge ok" style={{ fontSize: 9, padding: '1px 5px' }}>active</span>
                    )}
                    {entry.shadowed && (
                      <span style={{ color: 'var(--fg-3)', fontSize: 10 }}>shadowed</span>
                    )}
                    {isDead && (
                      <span className="badge bad" style={{ fontSize: 9, padding: '1px 5px' }}>dead</span>
                    )}
                    {/* Evict button: present on every entry, disabled when owner is live.
                        Per-entry entry.live is the correct signal (a live top can shadow
                        a dead owner — the has_dead_park case).  Server-side task δ is the
                        authoritative guard; this is defense-in-depth only. */}
                    {onEvict && (
                      <button
                        disabled={entry.live}
                        onClick={() => onEvict(entry.owner, ownerProjectRoot)}
                        style={{ marginLeft: 'auto', fontSize: 9, padding: '1px 6px', cursor: entry.live ? 'default' : 'pointer' }}
                        title={entry.live ? 'Owner is live — eviction blocked' : 'Evict this park owner'}
                      >evict</button>
                    )}
                  </div>
                );
              })}
            </div>
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
  // Build a project-qualified row index.  Taskmaster task_ids are
  // project-scoped, so two projects can each have T-1 — keying by raw
  // task_id would pick whichever project the iterator hit last.  Memoised
  // against `rows` to avoid rebuilding on every poll-driven re-render.
  const rowByCompositeKey = stUseMemo(() => {
    const idx = {};
    for (const r of (rows || [])) {
      idx[`${r.project || ''}/${r.task_id}`] = r;
    }
    return idx;
  }, [rows]);

  // Memoised against `modules` (same pattern as rowByCompositeKey above) to
  // avoid the O(n^2 * segments) disambiguation scan on every poll-driven re-render.
  const labelMap = stUseMemo(() => labelFor(modules.map(m => m.path)), [modules]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {modules.map(m => {
        // Waiters are scoped to the module's owning project: a row from
        // project B that happens to include 'src/utils.py' in its lock_set
        // is NOT waiting on project A's 'src/utils.py' module.
        const waiters = (rows || []).filter(r =>
          (!m.project || r.project === m.project) &&
          (r.lock_set || []).includes(m.path) &&
          m.holder !== r.task_id
        );
        // The holder task_id is project-scoped; pair it with `holder_project`
        // (set by the server when `current_holders` is non-empty) so the
        // joined row lookup hits the correct project's row.
        const holderProject = m.holder_project || m.project || '';
        const holder = m.holder ? rowByCompositeKey[`${holderProject}/${m.holder}`] : null;
        const holderAge = holder ? fmtAge(holder.age_seconds) : null;

        return (
          <div key={`${m.project || ''}/${m.path}`} className="panel" style={{ padding: '10px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--fg-0)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.path}>
                  {labelMap.get(m.path)}
                </div>
                <div style={{ fontSize: 10, color: 'var(--fg-3)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.path}>
                  {m.path}
                  {m.project && (
                    <span style={{ marginLeft: 6, color: 'var(--fg-3)' }}>·{m.project}</span>
                  )}
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
                  // events_by_task is keyed by '${project}/${task_id}' (see
                  // collect_scheduler_state); the previous raw-task_id lookup
                  // here always returned null, silently dropping p50.
                  const spark = eventsMap && m.holder
                    ? eventsMap[`${holderProject}/${m.holder}`]
                    : null;
                  const p50 = avgWaitSeconds(spark);
                  if (p50 == null) return null;
                  return <span style={{ color: 'var(--fg-3)', fontSize: 10, marginLeft: 8 }}>avg wait ~{fmtAge(p50)}</span>;
                })()}
              </div>
            ) : (
              <div style={{ fontSize: 11, marginBottom: 8, color: 'var(--ok)' }}>free</div>
            )}

            {/* Waiters */}
            {waiters.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {waiters.slice(0, 5).map(r => (
                  <div key={`${r.project || ''}/${r.task_id}`} style={{ display: 'flex', gap: 8, fontSize: 11, alignItems: 'baseline' }}>
                    <span className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>T-{r.task_id}</span>
                    <span style={{ color: 'var(--fg-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.title}>
                      {r.title || '—'}
                    </span>
                    {r.park_state && (r.park_state.modules || []).includes(m.path) && (
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
// Note: the global `projects` filter is intentionally not consumed here — the
// Scheduler tab owns its own per-project chip filter (ProjectChips) derived from
// SCHEDULER rows/modules. The global Toolbar project dropdown is hidden on this
// tab via app.jsx toolbarConfig.scheduler.showProjects: false.
function SchedulerTab() {
  const sched = D.SCHEDULER || {};
  const {
    rows = [],
    modules = [],
    pin_queue = [],
    events_by_task = {},
    snapshot_at = null,
    offline = false,
    offline_projects = [],
    paused = false,
    paused_projects = [],
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

  // Derive project list from SCHEDULER data (rows + modules .project field)
  const chipOptions = stUseMemo(() => {
    const set = new Set();
    rows.map(r => r.project).filter(Boolean).forEach(p => set.add(p));
    modules.map(m => m.project).filter(Boolean).forEach(p => set.add(p));
    return [...set].sort();
  }, [rows, modules]);

  // Local chip selection state: null = all (default); [] = nothing; [...] = explicit set
  const [chipSelected, setChipSelected] = stUseState(null);
  const effectiveSelected = chipSelected !== null ? chipSelected : chipOptions;

  // Filter rows by chip selection (explicit: row visible iff !r.project or selection includes r.project)
  const visibleRows = effectiveSelected.length === chipOptions.length || chipOptions.length === 0
    ? rows
    : rows.filter(r => !r.project || effectiveSelected.includes(r.project));

  // Filter modules by chip selection — same strict per-project predicate as visibleRows.
  // A module is visible iff:
  //   • no project tag (legacy/single-project path, !m.project guard mirrors !r.project)
  //   • the module's own project is selected (m.project)
  // The old holder_project OR-branch is intentionally removed: the data layer sets
  // holder_project == project whenever a holder exists, making it redundant.
  // cellStateFor already returns 'not-in-set' for cross-project cells, so no genuine
  // cross-project conflict is hidden by this strictness.
  const visibleModules = effectiveSelected.length === chipOptions.length || chipOptions.length === 0
    ? modules
    : modules.filter(m => !m.project || effectiveSelected.includes(m.project));

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
  // Receives composite keys ('${project}/${task_id}') from ActivePinsStrip and
  // groups them by project_root.  Each project's pin queue is independent on
  // the server (reorder_pin_queue is project-scoped), so a multi-project
  // pin-strip reorder must dispatch one MCP call per project, preserving the
  // relative order of each project's task_ids in the user's drag result.
  const handleReorder = stUseCallback(async (compositeKeys) => {
    // Index pins by composite key so we can recover task_id + project_root.
    const pinByKey = {};
    for (const p of (pin_queue || [])) {
      pinByKey[`${p.project || ''}/${p.task_id}`] = p;
    }
    // Group task_ids by project_root in the order they appear in compositeKeys.
    const groups = new Map();
    for (const key of compositeKeys) {
      const pin = pinByKey[key];
      if (!pin) continue;
      const root = pin.project_root || '';
      if (!groups.has(root)) groups.set(root, []);
      groups.get(root).push(pin.task_id);
    }
    if (groups.size === 0) return;
    // Fire all per-project reorder calls in parallel; surface a toast per
    // failing project so the user can identify which one broke.
    const calls = [];
    for (const [projectRoot, taskIds] of groups.entries()) {
      calls.push(
        fetch('/api/v2/dashboard/scheduler/reorder-pin-queue', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ task_ids: taskIds, project_root: projectRoot }),
        })
          .then(resp => ({ projectRoot, resp, err: null }))
          .catch(err => ({ projectRoot, resp: null, err })),
      );
    }
    const results = await Promise.all(calls);
    for (const { projectRoot, resp, err } of results) {
      if (err) {
        addToast(`Reorder error (${projectRoot || '∅'}): ${err.message || String(err)}`);
      } else if (resp && !resp.ok) {
        addToast(`Reorder failed for ${projectRoot || '∅'} (${resp.status})`);
      }
    }
  }, [pin_queue, addToast]);

  // ── Evict park owner ──
  // Mirrors handleUnpin: POST to the evict-park endpoint and toast on failure.
  // The server-side task δ guard is authoritative; this call is the UI lever.
  const handleEvict = stUseCallback(async (taskId, projectRoot) => {
    try {
      const resp = await fetch('/api/v2/dashboard/scheduler/evict-park', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ task_id: taskId, project_root: projectRoot || '' }),
      });
      if (!resp.ok) {
        addToast(`Evict failed (${resp.status})`);
      }
    } catch (err) {
      addToast(`Evict error: ${err.message || String(err)}`);
    }
  }, [addToast]);

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
  // Toggle by (project, task_id) composite identity — clicking a row in
  // project B that shares a numeric task_id with the selected row in
  // project A should switch selection, not deselect.  Matches the
  // composite identity used everywhere else in this file.
  const handleRowClick = stUseCallback((row) => {
    setSelectedTask(prev => (
      prev && prev.task_id === row.task_id && prev.project === row.project
        ? null
        : row
    ));
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

      {/* Paused banner */}
      {paused && (
        <div className="badge bad" style={{ padding: '6px 12px', fontSize: 11 }}>
          ⏸ Paused
          {paused_projects.length > 0 && `: ${paused_projects.map(p => p.reason ? `${p.project} (${p.reason})` : p.project).join(', ')}`}
        </div>
      )}

      {/* Stranded-parks banner — modules with any dead park owner.
           `has_dead_park` is True if any stack entry (including shadowed) is
           dead, so a live active-top with a dead shadowed owner is also caught.
           This keeps the banner in agreement with the per-entry dead/live
           indicators in ParkStacksSection and with stranded rows in the task list. */}
      {(() => {
        const strandedMods = visibleModules.filter(m => m.has_dead_park || (m.parked_by && !m.parked_owner_live));
        if (strandedMods.length === 0) return null;
        const paths = strandedMods.map(m => m.path).join(', ');
        return (
          <div className="badge bad" style={{ padding: '6px 12px', fontSize: 11 }}>
            ⚠ {strandedMods.length} stranded park{strandedMods.length !== 1 ? 's' : ''}: {paths}
          </div>
        );
      })()}

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

      {/* Sub-tab switcher + per-project chip filter */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <Segmented
          options={[{ value: 'tasks', label: 'Tasks' }, { value: 'modules', label: 'Modules' }]}
          value={subTab}
          onChange={setSubTab}
        />
        {chipOptions.length > 0 && (
          <ProjectChips
            options={chipOptions}
            selected={effectiveSelected}
            onChange={setChipSelected}
          />
        )}
        <span style={{ fontSize: 11, color: 'var(--fg-3)' }}>
          {visibleRows.length} task{visibleRows.length !== 1 ? 's' : ''} · {visibleModules.length} module{visibleModules.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Main view + optional drawer */}
      <div style={{ flex: 1, display: 'flex', gap: 12, minHeight: 0, overflow: 'hidden' }}>
        {/* Content area */}
        <div style={{ flex: 1, overflow: 'auto', minWidth: 0 }}>
          {subTab === 'tasks' ? (
            <SchedulerHeatmap
              rows={visibleRows}
              modules={visibleModules}
              onRowClick={handleRowClick}
              selectedTaskId={selectedTask ? selectedTask.task_id : null}
            />
          ) : (
            <div>
              <ModulesView
                modules={visibleModules}
                rows={visibleRows}
                eventsMap={events_by_task}
              />
              <ParkStacksSection modules={visibleModules} rows={visibleRows} onEvict={handleEvict} />
            </div>
          )}
        </div>

        {/* Drawer (slides in from the right) */}
        {selectedTask && (
          <SchedulerDrawer
            task={selectedTask}
            modules={visibleModules}
            eventsForTask={events_by_task[`${selectedTask.project}/${selectedTask.task_id}`] || null}
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
