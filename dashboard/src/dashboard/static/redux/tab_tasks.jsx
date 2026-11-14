/* Tasks tab: per-project dependency graph + filters + detail panel.
   Least-dependent tasks at top; tasks depending on them stack below. */
const { ProjectGroup: PG_T, Segmented: SEG_T, Pip } = window.DF_SHELL;
const { PALETTE: CP_T } = window.DF_CHARTS;
const DF_T = window.DF_DATA;
// One namespace alias, so no data.js const is rebound under its own name —
// see the SCOPE note in dashboard/tests/js/classic_script_scope.test.mjs.
const DF_LOADER_T = window.DF_DATA_LOADER;
const { useState: uS_T, useEffect: uE_T, useRef: uR_T, useLayoutEffect: uLE_T, useMemo: uM_T } = React;
const { computeNeighborhood, focusGroupView, layoutSignature, taskGraphLayout } = window.DF_GRAPH_LAYOUT;
const { ProjectPrdLanes, PrdContentionLine } = window.DF_PRD_LANES;
const { projectPrdView, projectPrdContention } = window.DF_PRD_LANES_VIEW;
const { strandBadgeState, agentCellState, locksCellState, schedulerLocksDatum, filedCellState } = window.DF_TASK_ROW_CELLS;
const { rtCell, rtAge, rtProbe, rtProbeSummary } = window.DF_RUNTIME_FMT;
const { tasksBannerNoticesFor } = window.DF_TASKS_OFFLINE_BANNER;
const { invariantChipFor, chainChipFor } = window.DF_INVARIANT_CHIP;
const { deferralCellState, deferralNeedsHumanChip } = window.DF_DEFERRAL_CELL;
// Const exports renamed, function exports not: test_tab_tasks_prose.py::_LOAD_SAFETY_MECHANISM.
const {
  projectCensus, projectRows, viewRows, snapshotRowsOver, unrequestedTerminalRows, terminalWindowProjects,
  terminalWindowEntrants,
  CENSUS_VIEWS: CENSUS_VIEWS_T, EVERY_VIEW: EVERY_VIEW_T,
} = window.DF_TASK_SNAPSHOT;
const { createPersistedHooks } = window.DF_PERSISTED_STATE;
const { usePersistedState } = createPersistedHooks(React);

// A per-project preference map holds only the projects that deviate from the
// default (an undefined deviation), so a map with every project back at its
// default is {} and usePersistedState removes its key.
function withProjectDeviation(map, id, deviation) {
  const { [id]: _previous, ...rest } = map;
  return deviation === undefined ? rest : { ...rest, [id]: deviation };
}

// CSS accent per probe-status tone (runtime_format.js owns which tone each
// status gets; this only translates a tone into a colour for the banner's
// left rule). 'muted' means "expected, not a fault" — no accent at all.
const PROBE_TONE_ACCENT_T = {
  muted: 'var(--line)',
  warn: 'var(--warn)',
  bad: 'var(--bad)',
};

// Text colour per invariant chip tone, claimant or chain (invariant_chip.js owns the tone).
const INVARIANT_CHIP_TONE_T = {
  muted: 'var(--fg-3)',
  warn: 'var(--warn)',
  bad: 'var(--bad)',
};

// A deferred row's hold, drawn the same on its graph node and in the detail pane.
function DeferralHold({ task }) {
  const cell = deferralCellState(task);
  if (!cell) return null;
  return (
    <div className="deferral" data-state={cell.state} title={cell.title}>
      <span className="deferral-text">{cell.text}</span>
      {cell.age && <span className="deferral-age">{cell.age}</span>}
      {cell.needsHuman && <span className="deferral-human">needs human</span>}
    </div>
  );
}

// The detail pane's lock chips. Whether anything is known about the project's
// locks is locksCellState's answer, drawn as tabs.jsx::LocksCell draws it.
function ModuleLocksSection({ task }) {
  const { buildSchedLockInfo, disambiguateLabels, lockChipStateFor } = window.DF_SCHED_UTILS;
  const lockInfo = buildSchedLockInfo(task, DF_T.SCHEDULER);
  const cell = locksCellState(schedulerLocksDatum(DF_T.SCHEDULER, task.project, DF_T.__receipt), lockInfo);
  if (cell.placeholder) {
    return (<>
      <div className="section-lbl">Module locks</div>
      <span className="chip-empty" title={cell.title || undefined}>{cell.placeholder}</span>
    </>);
  }
  const { rawTaskId, lockSet, moduleByPath } = lockInfo;
  if (!lockSet.length) return null;
  const labelMap = disambiguateLabels(lockSet);
  return (<>
    <div className="section-lbl">Module locks ({lockSet.length})</div>
    <div className="chips column">
      {lockSet.map(modPath => {
        const { cls, ownerLabel } = lockChipStateFor(moduleByPath.get(modPath), rawTaskId, task.project);
        return (
          <span key={modPath} className={`chip ${cls}`} title={modPath}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{labelMap.get(modPath)}</span>
            {cls === 'lock-parked' && <span className="holder">⏸ {ownerLabel}</span>}
            {cls === 'lock-taken' && <span className="holder">⊘ {ownerLabel}</span>}
          </span>
        );
      })}
    </div>
  </>);
}

// ── Edge router: draw curves from each parent (dep) to each child node ──
function TaskGraphEdges({ containerRef, nodeRefs, tasks, selectedId, neighborhood }) {
  const [paths, setPaths] = uS_T([]);

  // Edges flicker if redrawn on the `tasks` reference, a new array every render
  // (each data poll and each interaction), so the effect below is keyed on
  // content: the layout's key, each dep's done bit (it picks the stroke), and
  // the selection. neighborhood derives from selectedId+tasks, so it is covered.
  const layoutKey = layoutSignature(tasks);
  const doneKey = JSON.stringify(tasks.map(t => (t.deps || []).map(d => !!d.done)));

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
    let raf = null;
    if (containerRef.current) {
      ro.observe(containerRef.current);
    } else {
      // First mount only: containerRef is owned by an ANCESTOR of this
      // component (TaskGraph's own wrapper div) —
      // React attaches refs bottom-up during commit, so a still-mounting
      // ancestor's own ref can be null at the exact moment THIS layout
      // effect body runs, even though the whole commit (every ref in the
      // tree) finishes before the browser's next paint. Deferring one
      // frame reliably sees it populated without touching the fast path
      // above, which every later (already-populated) run still takes.
      raf = requestAnimationFrame(() => {
        recompute();
        if (containerRef.current) ro.observe(containerRef.current);
      });
    }
    window.addEventListener('resize', recompute);
    return () => {
      if (raf != null) cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', recompute);
    };
  }, [layoutKey, doneKey, selectedId]);

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

  // Keyed on content, not on `tasks`: callers pass a fresh array every render
  // (each poll, each click). The layout holds ids only (see
  // graph_layout.js::taskGraphLayout), so nodes render the current task objects.
  const layoutKey = layoutSignature(tasks);
  const { blocks, singletons } = uM_T(() => taskGraphLayout(tasks), [layoutKey]);
  const neighborhood = uM_T(() => computeNeighborhood(tasks, selectedId), [layoutKey, selectedId]);
  const taskById = new Map(tasks.map(t => [t.id, t]));

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
          {(() => { const sb = strandBadgeState(t, { compact: true }); return sb && <span className={sb.cls} title={sb.title}>{sb.label}</span>; })()}
          <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--fg-3)', fontFamily: 'var(--mono)' }}>
            {t.status === 'in-progress' ? rtAge(t.started) : t.status === 'done' ? (t.completed ? window.DF_SHELL.timeago(t.completed) : 'done') : t.status}
          </span>
        </div>
        <div className="ttl">{t.title}</div>
        <DeferralHold task={t} />
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
              {row.length === 0 ? <div className="empty-tier">—</div> : row.map(id => renderNode(taskById.get(id)))}
            </div>
          ))}
        </div>
      ))}
      {singletons.length > 0 && (
        <div className="singleton-strip">
          <div className="strip-label">unconnected</div>
          <div className="row">
            {singletons.map(id => renderNode(taskById.get(id)))}
          </div>
        </div>
      )}
    </div>
  );
}

function fmtAge(t) {
  if (t.status === 'done')             return t.completed ? window.DF_SHELL.timeago(t.completed) : '—';
  if (t.status === 'pending')          return 'unstarted';
  if (t.status === 'merge-deferred')   return 'parked for train';
  if (t.status === 'deferred')         return 'deferred';
  if (t.status === 'cancelled')        return t.completed ? window.DF_SHELL.timeago(t.completed) : 'cancelled';
  const age = rtAge(t.started);
  return age === '—' ? age : `${age} running`;
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

// The selected task's description/details, which the list rows do not carry.
// A child of TaskDetail so that TaskDetail's early `if (!task)` return cannot
// change the hook order.
function TaskProse({ uid }) {
  const [outcome, setOutcome] = uS_T(null);
  uE_T(() => {
    let current = true;
    setOutcome(null);
    DF_LOADER_T.requestOnDemand('taskProse', uid).then(o => { if (current) setOutcome(o); });
    return () => { current = false; };
  }, [uid]);
  const prose = DF_T[DF_LOADER_T.ON_DEMAND_KEYS.taskProse.key(uid)];
  switch (DF_LOADER_T.onDemandView(prose, outcome)) {
    case DF_LOADER_T.ON_DEMAND_VIEWS.ready:
      return (<>
        <div className="section-lbl">Description</div>
        <MarkdownText text={prose.description} className="desc" empty="—" />
        {prose.details && (<>
          <div className="section-lbl">Details</div>
          <MarkdownText text={prose.details} className="desc" />
        </>)}
      </>);
    case DF_LOADER_T.ON_DEMAND_VIEWS.loading:
      return (<>
        <div className="section-lbl">Description</div>
        <div className="desc" style={{ color: 'var(--fg-3)' }}>loading…</div>
      </>);
    default:
      return (<>
        <div className="section-lbl">Description</div>
        <div className="desc" style={{ color: 'var(--warn)' }}>The description and details could not be fetched.</div>
      </>);
  }
}

function TaskDetail({ task, allTasks }) {
  if (!task) {
    return <div className="placeholder">Click a task in the graph to see its full description, dependencies, and locks.</div>;
  }
  const deps = task.deps || [];
  const dependents = allTasks.filter(t => (t.deps || []).some(d => d.id === task.id));
  // Why the runtime cells below are dashed. Three separable cases the operator
  // must not confuse: the orchestrator is unreachable, OUR probe deadline fired
  // (a starved dashboard event loop looks exactly like this), or no runtime
  // endpoint is configured at all. Produced by task_runtime._probe_one ->
  // active_tasks._probe_status. null on the healthy path — nothing renders.
  const probe = rtProbe(task.runtime_status);
  return (
    <div className="task-detail">
      <h4>{task.title}</h4>
      <div className="sub">{window.DF_SHELL.taskId(task.id)} · {task.project} · {fmtAge(task)}</div>

      <div className="kv">
        <span className="k">status</span>
        <span><span className={`badge ${task.status === 'blocked' ? 'bad' : task.status === 'done' ? 'ok' : task.status === 'deferred' ? 'muted' : task.status === 'cancelled' ? 'muted' : task.status === 'pending' ? 'warn' : task.status === 'merge-deferred' ? 'merge-deferred' : 'accent'}`}>{task.status}</span>
          {/* Strand verdict (task 3543). Computed server-side from the claim
              columns; deliberately INDEPENDENT of `agent` below, which is only
              worktree presence and stays truthy after the agent dies. */}
          {(() => { const sb = strandBadgeState(task); return sb && <span className={sb.cls} style={{ marginLeft: sb.marginLeft }} title={sb.title}>{sb.label}</span>; })()}</span>
        <span className="k">filed</span>{(() => { const fc = filedCellState(task); return <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }} title={fc.title}>{fc.text}</span>; })()}
        {deferralCellState(task) && (
          <React.Fragment>
            <span className="k">hold</span>
            <DeferralHold task={task} />
          </React.Fragment>
        )}
        <span className="k">agent</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{(() => { const ac = agentCellState(task); return ac.color ? <span style={{ color: ac.color }}>{ac.text}</span> : ac.text; })()}</span>
        <span className="k">loops</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{rtCell(task.loops)}</span>
        <span className="k">attempts</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{rtCell(task.attempts)}</span>
        <span className="k">lane</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{rtCell(task.lane)}</span>
        <span className="k">phase</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{rtCell(task.phase)}</span>
        <span className="k">state</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{rtCell(task.lane_state)}</span>
        {probe && (
          <React.Fragment>
            <span className="k">runtime</span>
            <span data-testid="task-runtime-probe" title={probe.hint}>
              <span className={`badge ${probe.tone}`}>{probe.label}</span>
              <span style={{ color: 'var(--fg-3)', fontSize: 10, marginLeft: 6 }}>{probe.hint}</span>
            </span>
          </React.Fragment>
        )}
      </div>

      <TaskProse uid={task.id} />

      <div className="section-lbl">Depends on ({deps.length})</div>
      {deps.length === 0
        ? <span className="chip-empty">no upstream dependencies</span>
        : <div className="chips multiline">{deps.map(d =>
            <span key={d.id} className={`chip ${d.done ? 'dep-done' : 'dep-pending'}`} title={d.title}>{window.DF_SHELL.taskId(d.id)}{d.title ? ` · ${d.title}` : ''}</span>)}
          </div>}

      <div className="section-lbl">Blocks ({dependents.length})</div>
      {dependents.length === 0
        ? <span className="chip-empty">nothing depends on this</span>
        : <div className="chips multiline">{dependents.map(d =>
            <span key={d.id} className="chip" title={d.title}>{window.DF_SHELL.taskId(d.id)}{d.title ? ` · ${d.title}` : ''}</span>)}
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

      <ModuleLocksSection task={task} />
    </div>
  );
}

function TasksTab({ projectFilter, search }) {
  // Open/closed per project, persisted
  const [openMap, setOpenMap] = usePersistedState('df.tasksOpen', {});
  const toggle = (id) => setOpenMap(m => withProjectDeviation(m, id, m[id] === false ? undefined : false));

  // Per-project view, persisted (same per-project-id map shape as openMap):
  // the PRD lanes by default, 'flat' for the task graph listed flat.
  const [groupByPrdMap, setGroupByPrdMap] = usePersistedState('df.tasksGroupByPrd', {});
  const setGroupByPrd = (id, v) => setGroupByPrdMap(m => withProjectDeviation(m, id, v === 'flat' ? 'flat' : undefined));

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

  // The census views, in-flight and backlog on by default. A new key, not
  // 'df.tasksFilters': an old {active, pending, ...} object would read as
  // "nothing selected" under the view keys.
  const [filterMap, setFilterMap] = usePersistedState('df.tasks.views', { in_flight: true, backlog: true, terminal: false });
  const filter = Object.fromEntries(CENSUS_VIEWS_T.map(v => [v.key, !!(filterMap || {})[v.key]]));
  const flipFilter = (key) => setFilterMap({ ...filter, [key]: !filter[key] });

  const allTasks = snapshotRowsOver(DF_T);
  const projects = DF_T.PROJECTS.filter(p =>
    projectFilter.length === 0 || projectFilter.includes(p.id)
  );

  // PRD decision 5: a project's terminal window is fetched on demand only, once
  // as the project enters the wanted set and never per poll. It is always
  // lower_bound, so its age badge says how old it is.
  const projectIds = projects.map(p => p.id);
  const wanted = terminalWindowProjects(projectIds, filter);
  const wantedKey = wanted.join('\n');
  const [terminalOutcomes, setTerminalOutcomes] = uS_T({});
  // No cleanup cancels a pending outcome: a later run does not re-request a
  // project still in the set, so its outcome arrives only via the run that asked.
  const wantedBefore = uR_T([]);
  uE_T(() => {
    const entrants = terminalWindowEntrants(wantedBefore.current, wanted);
    wantedBefore.current = wanted;
    setTerminalOutcomes(o => ({ ...o, ...Object.fromEntries(entrants.map(project => [project, null])) }));
    for (const project of entrants) {
      DF_LOADER_T.requestOnDemand('terminal', project).then(outcome => {
        setTerminalOutcomes(o => ({ ...o, [project]: outcome }));
      });
    }
  }, [wantedKey]);
  const terminalOf = (id) => (wanted.includes(id)
    ? DF_LOADER_T.onDemandDatum('terminal', id, terminalOutcomes[id] ?? null)
    : unrequestedTerminalRows(DF_T[DF_LOADER_T.ON_DEMAND_KEYS.terminal.key(id)]));

  // Surface a banner when task data is missing or incomplete — without it the
  // Tasks tab renders an empty grid that looks indistinguishable from "no
  // active work". That rationale still motivates the total-outage case, and it
  // is exactly why the partial and degraded cases exist too: after the
  // whole-handler budget expires, a silently-missing project would otherwise
  // read as a project with nothing to do.
  //
  // The distinction between those cases is decided SERVER-SIDE (app.api_tasks)
  // and turned into copy by tasks_offline_banner.js. Nothing here re-derives
  // it: a `k of N` judgement made in the JSX would drift from the one the
  // handler made. N likewise comes from the server (TASKS_PROJECT_COUNT): the
  // numerator is a count of task project roots, so the denominator must be
  // one too — PROJECTS is the orchestrator-derived list, a different
  // population that diverges whenever a root has no orchestrator (or the
  // reverse), which made the notice understate how many projects failed.
  const bannerNotices = tasksBannerNoticesFor(DF_T);
  const bannerTestIds = {
    global: 'tasks-offline-banner',
    partial: 'tasks-partial-banner',
    degraded: 'tasks-degraded-banner',
    'count-unknown': 'tasks-count-unknown-banner',
    'count-stale': 'tasks-count-stale-banner',
  };

  // Runtime-probe health, derived frontend-side from the snapshot rows
  // (deliberately not a new top-level payload key — the per-project fact is
  // already fully recoverable from `runtime_status`, so a new key would carry
  // zero extra information). `runtime_status` is produced by
  // task_runtime._probe_one (which names the fault domain) and mapped to the
  // row vocabulary by active_tasks._probe_status — grep either end to find
  // the other.
  //
  // A SIBLING of bannerNotices above, not another kind inside it. The two
  // answer different questions: tasksBannerNotices reports whether TASK DATA
  // is available, a distinction decided server-side in app.api_tasks, while
  // rtProbeSummary is a client-side derivation over the snapshot rows
  // reporting whether we could reach the ORCHESTRATORS. Those are independent
  // — fused-memory being down says nothing about orchestrator reachability —
  // so both banners can show at once, and neither gates the other.
  //
  // Folding this into tasksBannerNotices would not merely blur that: it would
  // SUPPRESS the probe verdict. That function short-circuits
  // (tasks_offline_banner.js:112-115) — when `offline` is true it returns a
  // single global notice and never evaluates the other kinds — so a probe
  // notice routed through it would vanish during precisely the fused-memory
  // outage we must not gate on, which is when an operator is already
  // mid-triage and least able to afford a missing signal. Pinned by
  // test_tab_tasks_runtime.py::test_probe_banner_is_not_a_tasksbannernotices_kind.
  //
  // Computed over the UNFILTERED rows, and deliberately NOT narrowable.
  // `selfInflicted` is an assertion about the DASHBOARD's own health — that it
  // finished none of the probes it started — so its denominator must be every
  // probed project. This was previously scoped to the projectFilter-visible
  // rows, which let an operator narrowed to two timed-out projects manufacture
  // an all-at-once verdict and blame the dashboard for what was actually a
  // per-project outage: the precise misdiagnosis this banner exists to prevent.
  // The banner is therefore a global fact.
  //
  // Accepted tradeoff, so a future reader does not "fix" this back: an
  // operator filtered down to one healthy project may still see a banner about
  // a project they cannot see. That is strictly less harmful than a false "the
  // orchestrators may be healthy — check the dashboard first" shown mid-triage.
  const probeSummary = rtProbeSummary(allTasks);

  function searchMatches(t) {
    if (!search) return true;
    const terms = search.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return true;
    const haystack = `${t.id} ${t.title}`.toLowerCase();
    return terms.every(term => haystack.includes(term));
  }

  const knownTasks = [...allTasks, ...wanted.flatMap(id => terminalOf(id).value || [])];
  const selectedTask = selectedId ? knownTasks.find(t => t.id === selectedId) : null;

  return (
    <div className="grid cols-12" style={{ gap: 16 }}>
      {bannerNotices.map(notice => (
        <div key={notice.kind} className="col-span-12"
             data-testid={bannerTestIds[notice.kind]}
             style={{
               padding: '8px 12px',
               border: '1px solid var(--line)',
               borderRadius: 4,
               background: 'var(--bg-2)',
               color: 'var(--fg-3)',
               fontFamily: 'var(--mono)',
               fontSize: 11,
             }}>
          {notice.text}
        </div>
      ))}
      {probeSummary && (
        <div className="col-span-12" data-testid="tasks-runtime-probe-banner"
             style={{
               padding: '8px 12px',
               border: '1px solid var(--line)',
               // Accent by the WORST tone present, not by selfInflicted: a
               // lone timed-out project is a 'warn' (quite possibly our own
               // starved loop) and must not wear the same alarm colour as a
               // confirmed orchestrator outage. rtProbe owns the mapping.
               borderLeft: `3px solid ${PROBE_TONE_ACCENT_T[probeSummary.tone] || 'var(--warn)'}`,
               borderRadius: 4,
               background: 'var(--bg-2)',
               color: 'var(--fg-2)',
               fontFamily: 'var(--mono)',
               fontSize: 11,
             }}>
          {probeSummary.text}
        </div>
      )}
      {/* Filter bar */}
      <div className="col-span-12" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="lbl" style={{ color: 'var(--fg-3)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>show</span>
        <div className="seg">
          {CENSUS_VIEWS_T.map(v => (
            <button key={v.key} className={filter[v.key] ? 'on' : ''} onClick={() => flipFilter(v.key)}>{v.label}</button>
          ))}
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
            const census = projectCensus(DF_T, p.id);
            const rows = projectRows(DF_T, p.id);
            const terminal = terminalOf(p.id);
            const listed = viewRows(rows, terminal, filter);
            const held = viewRows(rows, terminal, EVERY_VIEW_T).rows;
            const filtered = listed.rows.filter(searchMatches);
            const onlyThePlaceholder = listed.placeholder && filtered.length === 0;
            // The ONE focus-narrowing site. The header count below and the
            // group body — the flat graph, or the lanes page's expanded rows
            // drawn from `pool` — read this single result, so they cannot be
            // fed different arrays (task 4137: narrowing applied per group
            // while the header counted the PRE-focus `filtered`). A plain
            // call, not a hook, so it is legal here inside map().
            //
            // It runs for a COLLAPSED group too, deliberately: shell.jsx's
            // ProjectGroup renders `summary` unconditionally but `children`
            // only when open, yet the header still shows the narrowed count.
            // So this runs for every project on every render. The bound
            // is computeNeighborhood's O(|neighborhood| x |tasks|) descendants
            // scan over ONE project's filtered list (hundreds at most today),
            // and it is free in the two common cases: focus off is a
            // passthrough, and an already-empty `filtered` gives the walk
            // nothing to scan.
            const groupView = focusGroupView(filtered, { focusMode, selectedId, focusAnchorId });
            const isOpen = openMap[p.id] !== false; // default-open
            const groupByPrd = groupByPrdMap[p.id] !== 'flat';
            const invariantChip = invariantChipFor(DF_T.INVARIANTS, p.id);
            const chainChip = chainChipFor(DF_T.INVARIANTS, p.id);
            const deferralChip = deferralNeedsHumanChip(rows.value || []);
            const summary = (
              <>
                {CENSUS_VIEWS_T.map(v => <Pip key={v.key} datum={census} color={CP_T[v.tone]} format={v.reading} />)}
                {deferralChip && (
                  <span className="mono deferral-chip" data-testid="tasks-deferral-needs-human" title={deferralChip.title}>
                    {deferralChip.text}
                  </span>
                )}
                {invariantChip && (
                  <span className="mono" data-testid="tasks-invariant-chip" title={invariantChip.title}
                        style={{ color: INVARIANT_CHIP_TONE_T[invariantChip.tone], fontSize: 10 }}>
                    {invariantChip.text}
                  </span>
                )}
                {chainChip && (
                  <span className="mono" data-testid="tasks-chain-chip" title={chainChip.title}
                        style={{ color: INVARIANT_CHIP_TONE_T[chainChip.tone], fontSize: 10 }}>
                    {chainChip.text}
                  </span>
                )}
                <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>
                  {groupView.shownCount}/{held.length} shown{groupView.emptiedByFocus ? ' — none in focus' : ''}
                </span>
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
                <PrdContentionLine contention={projectPrdContention(DF_T, p.id)} />
                {listed.placeholder && <div className="empty" title={listed.placeholder.title}>{listed.placeholder.text}</div>}
                {listed.notes.map(note => (
                  <div key={note} className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>{note}</div>
                ))}
                {groupByPrd
                  ? <ProjectPrdLanes project={p.id} view={projectPrdView(DF_T, p.id)} pool={groupView.shown}
                                     renderMembers={tasks => <TaskGraph tasks={tasks} selectedId={selectedId}
                                                                        onSelect={setSelectedId} onEnterFocus={enterFocus} />}
                                     onSelect={setSelectedId} />
                  : !onlyThePlaceholder && (groupView.emptiedByFocus
                    ? <div className="empty">no tasks in the focused neighborhood — Esc to exit focus</div>
                    : <TaskGraph tasks={groupView.shown} selectedId={selectedId}
                                 onSelect={setSelectedId} onEnterFocus={enterFocus} />)}
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
