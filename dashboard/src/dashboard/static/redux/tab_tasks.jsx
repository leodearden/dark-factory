/* Tasks tab: per-project dependency graph + filters + detail panel.
   Least-dependent tasks at top; tasks depending on them stack below. */
const { ProjectGroup: PG_T, Segmented: SEG_T } = window.DF_SHELL;
const { PALETTE: CP_T } = window.DF_CHARTS;
const DF_T = window.DF_DATA;
const { useState: uS_T, useEffect: uE_T, useRef: uR_T, useLayoutEffect: uLE_T, useMemo: uM_T } = React;

// Persisted-state hook (same shape as elsewhere)
function tasksPersistedState(key, def) {
  const [v, setV] = uS_T(() => {
    try { const r = localStorage.getItem(key); return r != null ? JSON.parse(r) : def; }
    catch { return def; }
  });
  uE_T(() => { try { localStorage.setItem(key, JSON.stringify(v)); } catch {} }, [key, v]);
  return [v, setV];
}

// ── Compute dep tiers for a task list (Kahn's algorithm style; tier = max(deps' tier)+1) ──
function computeTiers(tasks) {
  const byId = new Map(tasks.map(t => [t.id, t]));
  const tiers = new Map(); // id -> tier
  const visiting = new Set();

  function tierOf(id) {
    if (tiers.has(id)) return tiers.get(id);
    const t = byId.get(id);
    if (!t) return 0;             // dep references a task outside the project list — ignore
    if (visiting.has(id)) return 0; // cycle guard
    visiting.add(id);
    const inProject = (t.deps || []).filter(d => byId.has(d.id));
    const tier = inProject.length === 0 ? 0 : 1 + Math.max(...inProject.map(d => tierOf(d.id)));
    tiers.set(id, tier);
    visiting.delete(id);
    return tier;
  }
  for (const t of tasks) tierOf(t.id);
  return tiers;
}

// ── Edge router: draw curves from each parent (dep) to each child node ──
function TaskGraphEdges({ containerRef, nodeRefs, tasks, selectedId, neighborhood }) {
  const [paths, setPaths] = uS_T([]);

  // Stable signature: edges flicker because the parent's `tasks` is a new array
  // every render (overview clock ticks force re-renders). Only re-run when the
  // *content* changes — task ids, statuses, dep edges, selection.
  // neighborhood is derived from selectedId+tasks so it is already covered.
  const signature = uM_T(() => {
    const parts = tasks.map(t => `${t.id}:${t.status}:${(t.deps||[]).map(d=>d.id+(d.done?'1':'0')).join(',')}`);
    return parts.join('|') + '|sel=' + (selectedId || '');
  }, [tasks, selectedId]);

  // Capture latest tasks/selection in a ref so recompute() always reads fresh
  // values without us having to put them in the effect deps array.
  const latest = uR_T({ tasks, selectedId, neighborhood });
  latest.current = { tasks, selectedId, neighborhood };

  uLE_T(() => {
    function recompute() {
      const cont = containerRef.current;
      if (!cont) return;
      const { tasks: ts, selectedId: sel, neighborhood: nb } = latest.current;
      const visible = new Set(ts.map(t => t.id));
      const cb = cont.getBoundingClientRect();
      const out = [];
      for (const t of ts) {
        const childEl = nodeRefs.current[t.id];
        if (!childEl) continue;
        for (const d of (t.deps || [])) {
          if (!visible.has(d.id)) continue;
          const parentEl = nodeRefs.current[d.id];
          if (!parentEl) continue;
          const pb = parentEl.getBoundingClientRect();
          const cBox = childEl.getBoundingClientRect();
          const x1 = pb.left - cb.left + pb.width / 2;
          const y1 = pb.top  - cb.top  + pb.height;
          const x2 = cBox.left - cb.left + cBox.width / 2;
          const y2 = cBox.top  - cb.top;
          const dy = Math.max(12, (y2 - y1) / 2);
          const involved = nb && nb.has(t.id) && nb.has(d.id);
          const blocking = !d.done;
          out.push({
            d: `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`,
            stroke: blocking ? CP_T.warn : 'var(--line-strong)',
            opacity: involved ? 1 : (sel ? 0.25 : 0.7),
            width: involved ? 1.6 : 1,
            key: `${d.id}->${t.id}`,
          });
        }
      }
      // Only update if changed (cheap shallow compare on key+d) — avoids flicker.
      setPaths(prev => {
        if (prev.length !== out.length) return out;
        for (let i = 0; i < out.length; i++) {
          if (prev[i].key !== out[i].key || prev[i].d !== out[i].d ||
              prev[i].stroke !== out[i].stroke || prev[i].opacity !== out[i].opacity ||
              prev[i].width !== out[i].width) return out;
        }
        return prev;
      });
    }
    recompute();
    const ro = new ResizeObserver(recompute);
    if (containerRef.current) ro.observe(containerRef.current);
    window.addEventListener('resize', recompute);
    return () => { ro.disconnect(); window.removeEventListener('resize', recompute); };
  }, [signature]);

  return (
    <svg className="edges">
      <defs>
        <marker id="tg-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-strong)" />
        </marker>
        <marker id="tg-arrow-warn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={CP_T.warn} />
        </marker>
      </defs>
      {paths.map(p => (
        <path key={p.key} d={p.d} fill="none" stroke={p.stroke} strokeWidth={p.width} opacity={p.opacity}
              markerEnd={p.stroke === CP_T.warn ? 'url(#tg-arrow-warn)' : 'url(#tg-arrow)'} />
      ))}
    </svg>
  );
}

function TaskGraph({ tasks, selectedId, onSelect }) {
  const containerRef = uR_T(null);
  const nodeRefs = uR_T({});

  const tiers = uM_T(() => computeTiers(tasks), [tasks]);
  const rows = uM_T(() => {
    const max = Math.max(0, ...Array.from(tiers.values()));
    const arr = Array.from({ length: max + 1 }, () => []);
    for (const t of tasks) arr[tiers.get(t.id) || 0].push(t);
    // Within a tier, sort by status priority: blocked → in-progress → pending → done
    const order = { blocked: 0, 'in-progress': 1, pending: 2, deferred: 3, done: 4 };
    arr.forEach(row => row.sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9)));
    return arr;
  }, [tasks, tiers]);

  // Highlight neighborhood when something is selected
  const neighborhood = uM_T(() => {
    if (!selectedId) return null;
    const set = new Set([selectedId]);
    const byId = new Map(tasks.map(t => [t.id, t]));
    // ancestors
    const stackUp = [selectedId];
    while (stackUp.length) {
      const cur = stackUp.pop();
      const t = byId.get(cur);
      if (!t) continue;
      for (const d of (t.deps || [])) if (byId.has(d.id) && !set.has(d.id)) { set.add(d.id); stackUp.push(d.id); }
    }
    // descendants
    const stackDown = [selectedId];
    while (stackDown.length) {
      const cur = stackDown.pop();
      for (const t of tasks) {
        if ((t.deps || []).some(d => d.id === cur) && !set.has(t.id)) { set.add(t.id); stackDown.push(t.id); }
      }
    }
    return set;
  }, [selectedId, tasks]);

  if (tasks.length === 0) {
    return <div className="empty">no tasks match the current filter</div>;
  }

  return (
    <div className="taskgraph" ref={containerRef}>
      <TaskGraphEdges containerRef={containerRef} nodeRefs={nodeRefs} tasks={tasks}
                      selectedId={selectedId} neighborhood={neighborhood} />
      {rows.map((row, ri) => (
        <div key={ri} className="row">
          {row.length === 0 ? <div className="empty-tier">—</div> : row.map(t => {
            const isSel = selectedId === t.id;
            const inTree = neighborhood && neighborhood.has(t.id) && !isSel;
            const dim = neighborhood && !neighborhood.has(t.id);
            return (
              <div key={t.id}
                   ref={el => { if (el) nodeRefs.current[t.id] = el; else delete nodeRefs.current[t.id]; }}
                   className={`node s-${t.status} ${isSel ? 'selected' : ''} ${inTree ? 'in-tree' : ''} ${dim ? 'dim' : ''}`}
                   onClick={() => onSelect(isSel ? null : t.id)}>
                <div className="meta">
                  <span className="status-pip"></span>
                  <span className="id">{window.DF_SHELL.taskId(t.id)}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--fg-3)', fontFamily: 'var(--mono)' }}>
                    {t.status === 'in-progress' ? `${t.started}m` : t.status === 'done' ? (t.completed || 'done') : t.status}
                  </span>
                </div>
                <div className="ttl">{t.title}</div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function fmtAge(t) {
  if (t.status === 'done')        return t.completed || '—';
  if (t.status === 'pending')     return 'unstarted';
  return `${t.started}m running`;
}

function MarkdownText({ text, className }) {
  if (!text) return null;
  const html = DOMPurify.sanitize(marked.parse(text));
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

function TaskDetail({ task, allTasks }) {
  if (!task) {
    return <div className="placeholder">Click a task in the graph to see its full description, dependencies, and locks.</div>;
  }
  const deps = task.deps || [];
  const dependents = allTasks.filter(t => (t.deps || []).some(d => d.id === task.id));
  return (
    <div className="task-detail">
      <h4>{task.title}</h4>
      <div className="sub">{window.DF_SHELL.taskId(task.id)} · {task.project} · {fmtAge(task)}</div>

      <div className="kv">
        <span className="k">status</span>
        <span><span className={`badge ${task.status === 'blocked' ? 'bad' : task.status === 'done' ? 'ok' : task.status === 'deferred' ? 'muted' : task.status === 'pending' ? 'warn' : 'accent'}`}>{task.status}</span></span>
        <span className="k">agent</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.agent || <span style={{ color: 'var(--fg-3)' }}>unassigned</span>}</span>
        <span className="k">loops</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.loops}</span>
        <span className="k">attempts</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.attempts}</span>
      </div>

      <div className="section-lbl">Description</div>
      {task.description
        ? <MarkdownText text={task.description} className="desc" />
        : <div className="desc">—</div>}

      {task.details && (<>
        <div className="section-lbl">Details</div>
        <MarkdownText text={task.details} className="desc" />
      </>)}

      <div className="section-lbl">Depends on ({deps.length})</div>
      {deps.length === 0
        ? <span className="chip-empty">no upstream dependencies</span>
        : <div className="chips multiline">{deps.map(d =>
            <span key={d.id} className={`chip ${d.done ? 'dep-done' : 'dep-pending'}`} title={d.title}>{window.DF_SHELL.taskId(d.id)} · {d.title}</span>)}
          </div>}

      <div className="section-lbl">Blocks ({dependents.length})</div>
      {dependents.length === 0
        ? <span className="chip-empty">nothing depends on this</span>
        : <div className="chips multiline">{dependents.map(d =>
            <span key={d.id} className="chip" title={d.title}>{window.DF_SHELL.taskId(d.id)} · {d.title}</span>)}
          </div>}

      {task.locks && task.locks.length > 0 && (<>
        <div className="section-lbl">File locks ({task.locks.length})</div>
        <div className="chips column">
          {task.locks.map(p => {
            const fl = (DF_T.FILE_LOCKS && DF_T.FILE_LOCKS[task.project]) || {};
            const holder = fl[p]?.holder;
            const cls = !holder ? 'lock-free' : holder === task.id ? 'lock-mine' : 'lock-taken';
            return (
              <span key={p} className={`chip ${cls}`} title={p}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{p}</span>
                {cls === 'lock-taken' && <span className="holder">⊘ {holder}</span>}
              </span>
            );
          })}
        </div>
      </>)}
    </div>
  );
}

function TasksTab({ projectFilter, search }) {
  // Open/closed per project, persisted
  const [openMap, setOpenMap] = tasksPersistedState('df.tasksOpen', {});
  const toggle = (id) => setOpenMap(m => ({ ...m, [id]: !m[id] }));

  // Selected task (single, across all projects)
  const [selectedId, setSelectedId] = uS_T(null);

  // Filter, default {active, pending} on
  const [filters, setFilters] = tasksPersistedState('df.tasksFilters',
    { active: true, pending: true, complete: false, deferred: false });
  const flipFilter = (k) => setFilters(f => ({ ...f, [k]: !f[k] }));

  const allTasks = DF_T.ACTIVE_TASKS;
  const projects = DF_T.PROJECTS.filter(p =>
    projectFilter.length === 0 || projectFilter.includes(p.id)
  );

  // Surface a single-line banner when fused-memory is unreachable for any
  // discovered project — without it the Tasks tab renders an empty grid that
  // looks indistinguishable from "no active work".
  const tasksOffline = !!DF_T.TASKS_OFFLINE;
  const offlineProjects = DF_T.TASKS_OFFLINE_PROJECTS || [];

  function statusMatches(s) {
    if (filters.active   && (s === 'in-progress' || s === 'blocked')) return true;
    if (filters.pending  && s === 'pending')  return true;
    if (filters.complete && s === 'done')     return true;
    if (filters.deferred && s === 'deferred') return true;
    return false;
  }
  function searchMatches(t) {
    if (!search) return true;
    const terms = search.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return true;
    const haystack = `${t.id} ${t.title}`.toLowerCase();
    return terms.every(term => haystack.includes(term));
  }

  const selectedTask = selectedId ? allTasks.find(t => t.id === selectedId) : null;

  return (
    <div className="grid cols-12" style={{ gap: 16 }}>
      {tasksOffline && (
        <div className="col-span-12" data-testid="tasks-offline-banner"
             style={{
               padding: '8px 12px',
               border: '1px solid var(--line)',
               borderRadius: 4,
               background: 'var(--bg-2)',
               color: 'var(--fg-3)',
               fontFamily: 'var(--mono)',
               fontSize: 11,
             }}>
          fused-memory offline — task data unavailable
          {offlineProjects.length > 0 && ` (${offlineProjects.join(', ')})`}
        </div>
      )}
      {/* Filter bar */}
      <div className="col-span-12" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="lbl" style={{ color: 'var(--fg-3)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>show</span>
        <div className="seg">
          <button className={filters.active   ? 'on' : ''} onClick={() => flipFilter('active')}>active</button>
          <button className={filters.pending  ? 'on' : ''} onClick={() => flipFilter('pending')}>pending</button>
          <button className={filters.complete ? 'on' : ''} onClick={() => flipFilter('complete')}>complete</button>
          <button className={filters.deferred ? 'on' : ''} onClick={() => flipFilter('deferred')}>deferred</button>
        </div>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--fg-3)', fontFamily: 'var(--mono)' }}>
          click a task to inspect →
        </span>
      </div>

      <div className="col-span-12" style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 16, minHeight: 0 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {projects.map(p => {
            const projTasks = allTasks.filter(t => t.project === p.id);
            const filtered = projTasks.filter(t => statusMatches(t.status) && searchMatches(t));
            const counts = {
              total:    projTasks.length,
              active:   projTasks.filter(t => t.status === 'in-progress' || t.status === 'blocked').length,
              pending:  projTasks.filter(t => t.status === 'pending').length,
              complete: projTasks.filter(t => t.status === 'done').length,
            };
            const isOpen = openMap[p.id] !== false; // default-open
            const summary = (
              <>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.accent }}></span>{counts.active} active</span>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.warn }}></span>{counts.pending} pending</span>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.ok }}></span>{counts.complete} done</span>
                <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>{filtered.length}/{counts.total} shown</span>
              </>
            );

            return (
              <PG_T key={p.id} id={p.id} label={p.id} open={isOpen} onToggle={() => toggle(p.id)} summary={summary}>
                <TaskGraph tasks={filtered} selectedId={selectedId} onSelect={setSelectedId} />
              </PG_T>
            );
          })}
        </div>

        <div className="panel" style={{ position: 'sticky', top: 12, alignSelf: 'start', maxHeight: 'calc(100vh - 200px)', overflow: 'auto' }}>
          <div className="panel-head">
            <span className="title">Task detail</span>
            {selectedTask && (
              <button onClick={() => setSelectedId(null)}
                      style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--fg-3)', fontFamily: 'var(--mono)', padding: '2px 6px', borderRadius: 3, border: '1px solid var(--line)', background: 'transparent', cursor: 'pointer' }}>
                clear
              </button>
            )}
          </div>
          <TaskDetail task={selectedTask} allTasks={allTasks} />
        </div>
      </div>
    </div>
  );
}

window.DF_TASKS = { TasksTab };
