/* Tasks tab: per-project dependency graph + filters + detail panel.
   Least-dependent tasks at top; tasks depending on them stack below. */
const { ProjectGroup: PG_T, Segmented: SEG_T } = window.DF_SHELL;
const { PALETTE: CP_T } = window.DF_CHARTS;
const DF_T = window.DF_DATA;
const { useState: uS_T, useEffect: uE_T, useRef: uR_T, useLayoutEffect: uLE_T, useMemo: uM_T } = React;
const { computeTiers, partitionComponents, orderRows, computeNeighborhood, focusSubset } = window.DF_GRAPH_LAYOUT;
const { prdTitle, aggregatePrdStatus, summarizePrdMembers, groupTasksByPrd, orderPrdGroups } = window.DF_PRD_GROUPING;

// Persisted-state hook (same shape as elsewhere)
function tasksPersistedState(key, def) {
  const [v, setV] = uS_T(() => {
    try { const r = localStorage.getItem(key); return r != null ? JSON.parse(r) : def; }
    catch { return def; }
  });
  uE_T(() => { try { localStorage.setItem(key, JSON.stringify(v)); } catch {} }, [key, v]);
  return [v, setV];
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

function TaskGraph({ tasks, selectedId, onSelect, onEnterFocus }) {
  const containerRef = uR_T(null);
  const nodeRefs = uR_T({});

  // Partition into weakly-connected components + singletons, then order each
  // component's tiers via barycenter/transpose (STATUS_ORDER baked in as the
  // initial permutation/tiebreak — see graph_layout.js). Tiers are computed
  // once over the full filtered set: a weakly-connected component has no
  // edges to other components, so per-component tiers equal global tiers.
  const { blocks, singletons } = uM_T(() => {
    const tiers = computeTiers(tasks);
    const { components, singletons: singles } = partitionComponents(tasks);
    return { blocks: components.map(c => orderRows(c, tiers)), singletons: singles };
  }, [tasks]);

  // Highlight neighborhood when something is selected
  const neighborhood = uM_T(() => computeNeighborhood(tasks, selectedId), [selectedId, tasks]);

  // Node card JSX, shared verbatim between component-block tier rows and the
  // singleton strip so TaskGraphEdges (which resolves positions via
  // nodeRefs) keeps working identically in both.
  function renderNode(t) {
    const isSel = selectedId === t.id;
    const inTree = neighborhood && neighborhood.has(t.id) && !isSel;
    const dim = neighborhood && !neighborhood.has(t.id);
    return (
      <div key={t.id}
           ref={el => { if (el) nodeRefs.current[t.id] = el; else delete nodeRefs.current[t.id]; }}
           className={`node s-${t.status} ${isSel ? 'selected' : ''} ${inTree ? 'in-tree' : ''} ${dim ? 'dim' : ''}`}
           onClick={() => onSelect(isSel ? null : t.id)}
           onDoubleClick={() => onEnterFocus && onEnterFocus(t.id)}>
        <div className="meta">
          <span className="status-pip"></span>
          <span className="id">{window.DF_SHELL.taskId(t.id)}</span>
          {t.train && <span className="train-badge" title={`train ${t.train.id} · order ${t.train.order}`}>🚂 {t.train.id}</span>}
          <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--fg-3)', fontFamily: 'var(--mono)' }}>
            {t.status === 'in-progress' ? `${t.started}m` : t.status === 'done' ? (t.completed ? window.DF_SHELL.timeago(t.completed) : 'done') : t.status}
          </span>
        </div>
        <div className="ttl">{t.title}</div>
      </div>
    );
  }

  if (tasks.length === 0) {
    return <div className="empty">no tasks match the current filter</div>;
  }

  return (
    <div className="taskgraph" ref={containerRef}>
      <TaskGraphEdges containerRef={containerRef} nodeRefs={nodeRefs} tasks={tasks}
                      selectedId={selectedId} neighborhood={neighborhood} />
      {blocks.map((block, bi) => (
        <div key={bi} className="component-block">
          {block.map((row, ri) => (
            <div key={ri} className="row">
              {/* Defensive fallback only: orderRows buckets every component member into
                  tiers 0..maxTier with no gaps (tier = 1 + max(in-component deps' tier)),
                  so a row shouldn't be empty here in practice. Kept in case that
                  invariant is ever violated by a future graph_layout.js change. */}
              {row.length === 0 ? <div className="empty-tier">—</div> : row.map(renderNode)}
            </div>
          ))}
        </div>
      ))}
      {singletons.length > 0 && (
        <div className="singleton-strip">
          <div className="strip-label">unconnected</div>
          <div className="row">
            {singletons.map(renderNode)}
          </div>
        </div>
      )}
    </div>
  );
}

// Per-project wrapper around TaskGraph: memoizes the focus-mode subset so
// it isn't recomputed (rebuilding focusSubset's Map + re-walking
// computeNeighborhood) on every TasksTab render that doesn't actually
// change this project's filtered tasks, selection, or focus state — e.g.
// re-renders driven by unrelated live-data ticks elsewhere on the page.
// This has to be a real component (not an inline computation inside the
// projects.map() callback below) so the useMemo call follows the Rules of
// Hooks: one consistent hook per mounted project instance, not a
// variable-count hook call inside a loop.
//
// Guarding on selectedId != null (not just focusMode) keeps the existing
// immediate-passthrough behavior when a node is deselected by clicking it
// again — that only clears selectedId, and without this guard the subset
// would still reflect the stale focusAnchorId for one render until the
// effect above catches focusMode up to it.
function ProjectTaskGraph({ filtered, selectedId, focusMode, focusAnchorId, onSelect, onEnterFocus }) {
  const graphTasks = uM_T(
    () => (focusMode && selectedId != null ? focusSubset(filtered, focusAnchorId) : filtered),
    [filtered, focusMode, selectedId, focusAnchorId]
  );
  return <TaskGraph tasks={graphTasks} selectedId={selectedId} onSelect={onSelect} onEnterFocus={onEnterFocus} />;
}

// Per-project "group by PRD" render: buckets the project's filtered tasks
// into PRD boxes (groupTasksByPrd) and lays out each box's induced subgraph
// via the existing per-box TaskGraph machinery (which itself calls
// computeTiers/partitionComponents/orderRows). Kept as its own component
// (not an inline computation inside the projects.map() callback below) for
// the same Rules-of-Hooks reason as ProjectTaskGraph above — one consistent
// useMemo per mounted project instance.
function ProjectPrdGroups({ filtered, selectedId, onSelect, onEnterFocus }) {
  const groups = uM_T(() => groupTasksByPrd(filtered), [filtered]);
  return (
    <div className="prd-groups">
      {groups.map(g => (
        <div key={g.noPrd ? '__no_prd__' : g.prd} className="prd-box">
          <div className="prd-box-head">
            <span className="prd-box-title" title={g.noPrd ? undefined : g.prd}>
              {g.noPrd ? 'No PRD' : prdTitle(g.prd)}
            </span>
          </div>
          <div className="prd-box-body">
            <TaskGraph tasks={g.tasks} selectedId={selectedId} onSelect={onSelect} onEnterFocus={onEnterFocus} />
          </div>
        </div>
      ))}
    </div>
  );
}

function fmtAge(t) {
  if (t.status === 'done')             return t.completed ? window.DF_SHELL.timeago(t.completed) : '—';
  if (t.status === 'pending')          return 'unstarted';
  if (t.status === 'merge-deferred')   return 'parked for train';
  if (t.status === 'deferred')         return 'deferred';
  if (t.status === 'cancelled')        return t.completed ? window.DF_SHELL.timeago(t.completed) : 'cancelled';
  return `${t.started}m running`;
}

function MarkdownText({ text, className, empty }) {
  // Memoize parse+sanitize so it only reruns when text changes, not on every
  // parent re-render (e.g. streaming status updates from another tab).
  // breaks:true converts single newlines to <br>, matching the pre-line semantics
  // that task descriptions were originally authored against; \n\n still produces
  // paragraph breaks. gfm:true enables GitHub-Flavored Markdown (the dominant
  // dialect in pasted task descriptions).
  const html = uM_T(() => {
    if (!text || typeof marked === 'undefined' || typeof DOMPurify === 'undefined') return null;
    try {
      const sanitized = DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }));
      // Short-circuit: skip the <template> reparse on the common no-link path.
      // /<a[\s>]/i matches '<a ' (attributed) and bare '<a>' so no anchor is missed.
      if (!/<a[\s>]/i.test(sanitized)) return sanitized;
      // Scoped anchor rewrite: mutate <a> attributes after sanitizing rather than
      // registering a global DOMPurify hook. No future DOMPurify caller in the app
      // silently inherits target=_blank, and dev-reload re-evaluation is harmless.
      // <template> is inert (no script execution or resource loading), so walking
      // already-sanitized HTML is safe; tpl.innerHTML re-serializes the post-mutation
      // DOM (the value React injects), preserving the sanitizer's guarantees.
      // Behavioral invariants (no JS test runner; verified manually per task spec):
      //   a) no-anchor input short-circuits above, returning sanitized unchanged.
      //   b) https://… external links get target=_blank + rel=noopener noreferrer.
      //   c) #fragment and /path links stay in the same tab (in-page / SPA-internal).
      //   d) //host protocol-relative links are treated as external — see (b).
      const tpl = document.createElement('template');
      tpl.innerHTML = sanitized;
      tpl.content.querySelectorAll('a').forEach(a => {
        // In-page fragment and SPA-relative links stay in the same tab.
        // '//' is protocol-relative (external) and must NOT be skipped.
        const href = a.getAttribute('href') || '';
        if (href.startsWith('#') || (href.startsWith('/') && !href.startsWith('//'))) return;
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
      return tpl.innerHTML;
    } catch (err) {
      // On any parse/sanitize error return null; the html===null branch below
      // degrades to plain pre-line text rather than unmounting TaskDetail.
      console.warn('MarkdownText render failed', err);
      return null;
    }
  }, [text]);
  // No content: render empty-prop placeholder or nothing.
  if (!text) return empty !== undefined ? <div className={className}>{empty}</div> : null;
  // CDN libraries unavailable (network blip, ad-blocker, integrity-hash mismatch):
  // degrade gracefully to plain text instead of throwing a ReferenceError.
  if (html === null) return <div className={className} style={{whiteSpace: 'pre-line'}}>{text}</div>;
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
        <span><span className={`badge ${task.status === 'blocked' ? 'bad' : task.status === 'done' ? 'ok' : task.status === 'deferred' ? 'muted' : task.status === 'cancelled' ? 'muted' : task.status === 'pending' ? 'warn' : task.status === 'merge-deferred' ? 'merge-deferred' : 'accent'}`}>{task.status}</span></span>
        <span className="k">agent</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.agent || <span style={{ color: 'var(--fg-3)' }}>unassigned</span>}</span>
        <span className="k">loops</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.loops}</span>
        <span className="k">attempts</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.attempts}</span>
      </div>

      <div className="section-lbl">Description</div>
      <MarkdownText text={task.description} className="desc" empty="—" />

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

      {(() => {
        const extDeps = task.external_deps || [];
        if (!extDeps.length) return null;
        // Any status that begins with 'unknown' (covers future unknown_* sentinels)
        // or equals the 'malformed' sentinel renders as dep-unknown (muted / warning).
        const isUnknown = s => s.startsWith('unknown') || s === 'malformed';
        return (<>
          <div className="section-lbl">External dependencies ({extDeps.length})</div>
          <div className="chips multiline">
            {extDeps.map(d => {
              const cls = d.status === 'done' ? 'dep-done' : isUnknown(d.status) ? 'dep-unknown' : 'dep-pending';
              return (
                <span key={d.id} className={`chip ${cls}`} title={d.status}>{d.id} · {d.status}</span>
              );
            })}
          </div>
        </>);
      })()}

      {(() => {
        const { buildSchedLockInfo, disambiguateLabels } = window.DF_SCHED_UTILS || {};
        const { rawTaskId, lockSet, moduleByPath } = buildSchedLockInfo
          ? buildSchedLockInfo(task, DF_T.SCHEDULER)
          : { rawTaskId: String(task.id).split('/T-').pop(), lockSet: [], moduleByPath: new Map() };
        if (!lockSet.length) return null;
        const labelMap = disambiguateLabels ? disambiguateLabels(lockSet) : null;
        return (<>
          <div className="section-lbl">Module locks ({lockSet.length})</div>
          <div className="chips column">
            {lockSet.map(modPath => {
              const m = moduleByPath.get(modPath);
              const holder = m && m.holder;
              const holderProject = m && m.holder_project;
              const parkedBy = m && m.parked_by;
              const parkedOwnerLive = m && m.parked_owner_live;
              const isMine = holder === rawTaskId && (holderProject || task.project) === task.project;
              const { cls, ownerLabel } = window.DF_SCHED_UTILS.lockChipState({ holder, isMine, parkedBy, parkedOwnerLive });
              return (
                <span key={modPath} className={`chip ${cls}`} title={modPath}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{labelMap ? labelMap.get(modPath) : modPath}</span>
                  {cls === 'lock-parked' && <span className="holder">⏸ {ownerLabel}</span>}
                  {cls === 'lock-taken' && <span className="holder">⊘ {ownerLabel}</span>}
                </span>
              );
            })}
          </div>
        </>);
      })()}
    </div>
  );
}

function TasksTab({ projectFilter, search }) {
  // Open/closed per project, persisted
  const [openMap, setOpenMap] = tasksPersistedState('df.tasksOpen', {});
  const toggle = (id) => setOpenMap(m => ({ ...m, [id]: !m[id] }));

  // Per-project "group by PRD" view toggle, persisted (same per-project-id
  // map shape as openMap). Key absent/'flat' -> today's ungrouped view;
  // 'prd' -> the grouped-by-PRD view.
  const [groupByPrdMap, setGroupByPrdMap] = tasksPersistedState('df.tasksGroupByPrd', {});
  const setGroupByPrd = (id, v) => setGroupByPrdMap(m => ({ ...m, [id]: v }));

  // Selected task (single, across all projects)
  const [selectedId, setSelectedId] = uS_T(null);

  // Focus mode: ephemeral (NOT persisted) — filters the graph down to the
  // focus anchor's ancestors+descendants. Never survives a reload.
  // focusAnchorId is tracked separately from selectedId: single-clicking a
  // different in-view node while focused only updates the detail-panel
  // selection (matching the plain onSelect wiring on node clicks below) —
  // it does NOT silently re-narrow the graph out from under the user.
  // (Re-)entering focus on a node is the deliberate onEnterFocus action
  // (double-click / the "focus" button), which sets both together.
  const [focusMode, setFocusMode] = uS_T(false);
  const [focusAnchorId, setFocusAnchorId] = uS_T(null);
  function enterFocus(id) { setSelectedId(id); setFocusMode(true); setFocusAnchorId(id); }
  function exitFocus() { setFocusMode(false); }

  // Deselecting (via the graph, "clear", or otherwise) always drops focus —
  // there's no such thing as a focused-but-unselected state.
  uE_T(() => { if (selectedId == null) setFocusMode(false); }, [selectedId]);

  // Esc exits focus mode from anywhere on the page.
  uE_T(() => {
    function onKeyDown(e) { if (e.key === 'Escape') exitFocus(); }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  // Filter, default {active, pending} on
  const [filters, setFilters] = tasksPersistedState('df.tasksFilters',
    { active: true, pending: true, complete: false, deferred: false, cancelled: false });
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
    if (filters.active    && (s === 'in-progress' || s === 'blocked' || s === 'merge-deferred')) return true;
    if (filters.pending   && s === 'pending')    return true;
    if (filters.complete  && s === 'done')       return true;
    if (filters.deferred  && s === 'deferred')   return true;
    if (filters.cancelled && s === 'cancelled')  return true;
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
          <button className={filters.active    ? 'on' : ''} onClick={() => flipFilter('active')}>active</button>
          <button className={filters.pending   ? 'on' : ''} onClick={() => flipFilter('pending')}>pending</button>
          <button className={filters.complete  ? 'on' : ''} onClick={() => flipFilter('complete')}>complete</button>
          <button className={filters.deferred  ? 'on' : ''} onClick={() => flipFilter('deferred')}>deferred</button>
          <button className={filters.cancelled ? 'on' : ''} onClick={() => flipFilter('cancelled')}>cancelled</button>
        </div>
        <span className={focusMode && focusAnchorId ? 'focus-chip' : ''}
              style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--fg-3)', fontFamily: 'var(--mono)' }}>
          {focusMode && focusAnchorId
            ? `focused on ${window.DF_SHELL.taskId(focusAnchorId)} — Esc to exit`
            : 'click a task to inspect →'}
        </span>
      </div>

      <div className="col-span-12" style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 16, minHeight: 0 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {projects.map(p => {
            const projTasks = allTasks.filter(t => t.project === p.id);
            const filtered = projTasks.filter(t => statusMatches(t.status) && searchMatches(t));
            const _fallbackDone = projTasks.filter(t => t.status === 'done').length;
            const counts = {
              total:    projTasks.length,
              active:   projTasks.filter(t => t.status === 'in-progress' || t.status === 'blocked' || t.status === 'merge-deferred').length,
              pending:  projTasks.filter(t => t.status === 'pending').length,
              // DONE_COUNTS carries the authoritative full count from the server.
              // The fallback counts only the bounded done rows loaded into ACTIVE_TASKS
              // (≤50 per project). If we hit that cap without a server count, show
              // "50+" so the user knows the displayed number is a lower bound.
              complete: (DF_T.DONE_COUNTS && DF_T.DONE_COUNTS[p.id] != null)
                ? DF_T.DONE_COUNTS[p.id]
                : (_fallbackDone >= 50 ? '50+' : _fallbackDone),
            };
            const isOpen = openMap[p.id] !== false; // default-open
            const groupByPrd = groupByPrdMap[p.id] === 'prd';
            const summary = (
              <>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.accent }}></span>{counts.active} active</span>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.warn }}></span>{counts.pending} pending</span>
                <span className="pip"><span className="pip-dot" style={{ background: CP_T.ok }}></span>{counts.complete} done</span>
                <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>{filtered.length}/{counts.total} shown</span>
              </>
            );
            const summaryRight = (
              <span onClick={(e) => e.stopPropagation()}>
                <SEG_T options={[{ value: 'flat', label: 'list' }, { value: 'prd', label: 'by PRD' }]}
                       value={groupByPrd ? 'prd' : 'flat'}
                       onChange={(v) => setGroupByPrd(p.id, v)} />
              </span>
            );

            return (
              <PG_T key={p.id} id={p.id} label={p.id} open={isOpen} onToggle={() => toggle(p.id)}
                    summary={summary} summaryRight={summaryRight}>
                {groupByPrd
                  ? <ProjectPrdGroups filtered={filtered} selectedId={selectedId}
                                      onSelect={setSelectedId} onEnterFocus={enterFocus} />
                  : <ProjectTaskGraph filtered={filtered} selectedId={selectedId} focusMode={focusMode}
                                      focusAnchorId={focusAnchorId} onSelect={setSelectedId} onEnterFocus={enterFocus} />}
              </PG_T>
            );
          })}
        </div>

        <div className="panel" style={{ position: 'sticky', top: 12, alignSelf: 'start', maxHeight: 'calc(100vh - 200px)', overflow: 'auto' }}>
          <div className="panel-head">
            <span className="title">Task detail</span>
            {selectedTask && (
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                {!focusMode && (
                  <button className="focus-btn" onClick={() => enterFocus(selectedId)}>
                    focus
                  </button>
                )}
                <button onClick={() => { setSelectedId(null); setFocusMode(false); }}
                        style={{ fontSize: 10, color: 'var(--fg-3)', fontFamily: 'var(--mono)', padding: '2px 6px', borderRadius: 3, border: '1px solid var(--line)', background: 'transparent', cursor: 'pointer' }}>
                  clear
                </button>
              </div>
            )}
          </div>
          <TaskDetail task={selectedTask} allTasks={allTasks} />
        </div>
      </div>
    </div>
  );
}

window.DF_TASKS = { TasksTab };
