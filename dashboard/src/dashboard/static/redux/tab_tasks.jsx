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
function TaskGraphEdges({ containerRef, nodeRefs, tasks, selectedId }) {
  const [paths, setPaths] = uS_T([]);

  // Stable signature: edges flicker because the parent's `tasks` is a new array
  // every render (overview clock ticks force re-renders). Only re-run when the
  // *content* changes — task ids, statuses, dep edges, selection.
  const signature = uM_T(() => {
    const parts = tasks.map(t => `${t.id}:${t.status}:${(t.deps||[]).map(d=>d.id+(d.done?'1':'0')).join(',')}`);
    return parts.join('|') + '|sel=' + (selectedId || '');
  }, [tasks, selectedId]);

  // Capture latest tasks/selection in a ref so recompute() always reads fresh
  // values without us having to put them in the effect deps array.
  const latest = uR_T({ tasks, selectedId });
  latest.current = { tasks, selectedId };

  uLE_T(() => {
    function recompute() {
      const cont = containerRef.current;
      if (!cont) return;
      const { tasks: ts, selectedId: sel } = latest.current;
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
          const involved = sel && (sel === t.id || sel === d.id);
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
    const order = { blocked: 0, 'in-progress': 1, pending: 2, done: 3 };
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
      <TaskGraphEdges containerRef={containerRef} nodeRefs={nodeRefs} tasks={tasks} selectedId={selectedId} />
      {rows.map((row, ri) => (
        <div key={ri} className="row">
          {row.length === 0 ? <div className="empty-tier">—</div> : row.map(t => {
            const isSel = selectedId === t.id;
            const dim = neighborhood && !neighborhood.has(t.id);
            return (
              <div key={t.id}
                   ref={el => { if (el) nodeRefs.current[t.id] = el; else delete nodeRefs.current[t.id]; }}
                   className={`node s-${t.status} ${isSel ? 'selected' : ''} ${dim ? 'dim' : ''}`}
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

// ── Hard-coded long descriptions per task (so the detail panel has real content) ──
const TASK_DESCRIPTIONS = {
  'T-19': 'Implement an exponential-backoff retry policy for the consolidation agent. The agent currently throws on transient store errors and the orchestrator restarts it cold, losing in-flight context. Add a retry decorator that distinguishes transient (network, 5xx, lock contention) from terminal (validation, auth) errors. Cap at 5 retries with jitter; emit OTel spans per attempt; persist attempt count in the task journal so restarts resume mid-policy.',
  'T-12': 'Backfill burndown snapshots from the orchestrator journal so we can show 30-day history. The collector only persists current state, so anything before deployment day is missing. Walk the journal in 15-minute buckets, reconstruct task counts at each timestamp, and write to the burndown table. Idempotent: rerunning over the same range must produce identical rows.',
  'T-7':  'Add cross-store dedup at consolidation time. Today the consolidation agent writes the same memory to both Mem0 and Graphiti when classifications disagree, producing duplicate rows on every reconciliation pass. Block on T-21 (dedup index) and T-23 (hash collision handling); emit a metric for dedup hits so we can tell whether duplication is dropping post-deploy.',
  'T-21': 'Build a content-addressed dedup index for the Graphiti store. Hash should incorporate normalized text + entity set + timestamp bucket. Index must support O(1) lookups during write and survive store-side compaction. Output: a `dedup_idx` table + the read/write helpers. Migration script to backfill from existing rows is in scope.',
  'T-23': 'Hash collision handling for the dedup index. With ~10M memories projected and a 64-bit content hash we expect rare collisions; current behavior is silent overwrite. Add a fallback path that reads candidate row, byte-compares, and only treats as dedup if equal. On mismatch, fall back to a second hash function and log a collision event. Depends on T-21.',
  'T-24': 'Scheduler quiescence detector. The orchestrator currently has no way to know when "all useful work has been scheduled" — it busy-loops waking up every second. Detect quiescence by watching the task journal write rate and the in-flight count; emit a `quiescent` event so downstream tools (e.g. background reconciliation, snapshot) can run during idle periods. Depends on T-12 to read the journal.',
  'T-25': 'Burndown forecast model. Take the BURNDOWN_BY_PROJECT timeseries and fit a simple regression to project tasks-remaining out 7 days. Render a confidence band on the burndown chart. Stretch goal: switch to a state-space model that handles the discontinuity from new task additions. Blocked on T-12 (data) and T-24 (we want forecasts to update only during quiescent windows).',
  'T-17': 'Heuristic pre-filter for the consolidation classifier. Skip the LLM call entirely for memories matching simple regexes (timestamps, structured logs, repeated phrases). Drops classifier cost by ~40% on observed traffic with no measurable accuracy loss.',
  'T-16': 'Cross-project dedup. Same memory observed by two different projects should not produce two graph nodes. Uses the same hash from T-21 but queries across project namespaces.',
  'T-15': 'Mem0 collection partitioning. Split the single mega-collection into per-project collections so Mem0 query latency stops scaling with total memory count.',
  'T-14': 'Restart-safe burndown collector. Persist collector state every flush so an orchestrator restart does not lose the in-flight bucket.',
  'T-13': 'Watchdog timer tuning. The watchdog was killing tasks at the 90s mark even when they were making progress; raise to 180s and add a heartbeat check.',

  'T-8':  'Parser error recovery for partial DSL. Today a single syntax error aborts the whole parse. Implement panic-mode recovery at statement boundaries so we can show all errors in one pass and still produce a usable AST for downstream tools (LSP, linter). Synchronization tokens: `;`, `}`, `def`, `let`.',
  'T-11': 'DSL lexer rewrite for nested macros. The current lexer cannot handle `macro!{ macro!{ ... } }` because it greedily matches the outer `}`. Rewrite as a state-machine lexer that tracks brace depth per macro context.',
  'T-18': 'Macro expansion error messages. When expansion fails inside a nested macro, the error currently points at the outermost macro call — useless for debugging. Thread span info through expansion so the error points at the actual failing token.',
  'T-6':  'AST node visitor refactor. Migrate from inheritance-based visitors to a trait-object dispatch so plugins can register visitors without forking the AST crate.',

  'T-4':  'Scene boundary detection v2. Replace the histogram-diff detector with a learned model. Same input/output contract; should reduce false positives at scene transitions with motion blur. Reuse the optimized frame sampler from T-2.',
  'T-9':  'Subtitle alignment. SRT timestamps drift by up to 400ms over a 30-minute clip. Re-align using audio-track resampling (T-22) plus dynamic-time-warping against a phoneme transcript. Currently blocked on T-22.',
  'T-22': 'Audio track resampling. Pull-up/down arbitrary audio sample rates to the canonical 48kHz used by the rest of the pipeline. Important: preserve sample alignment so subtitle and scene boundaries stay in sync.',
  'T-26': 'Multi-pass encoder pipeline. Two-pass x265 with rate-control stats from pass 1. Wire into the existing single-pass entrypoint; pass 1 runs at 4x speed with quality disabled.',
  'T-2':  'Frame sampler optimization. The sampler was decoding every frame and discarding 90%; switch to keyframe-only seeking with selective fine-decode around scene-detection candidate timestamps. ~6x speedup on 4K clips.',
};

function fmtAge(t) {
  if (t.status === 'done')        return t.completed || '—';
  if (t.status === 'pending')     return 'unstarted';
  return `${t.started}m running`;
}

function TaskDetail({ task, allTasks }) {
  if (!task) {
    return <div className="placeholder">Click a task in the graph to see its full description, dependencies, and locks.</div>;
  }
  const deps = task.deps || [];
  const dependents = allTasks.filter(t => (t.deps || []).some(d => d.id === task.id));
  const desc = TASK_DESCRIPTIONS[task.id] || task.title;
  return (
    <div className="task-detail">
      <h4>{task.title}</h4>
      <div className="sub">{window.DF_SHELL.taskId(task.id)} · {task.project} · {fmtAge(task)}</div>

      <div className="kv">
        <span className="k">status</span>
        <span><span className={`badge ${task.status === 'blocked' ? 'bad' : task.status === 'done' ? 'ok' : task.status === 'pending' ? 'warn' : 'accent'}`}>{task.status}</span></span>
        <span className="k">agent</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.agent || <span style={{ color: 'var(--fg-3)' }}>unassigned</span>}</span>
        <span className="k">loops</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.loops}</span>
        <span className="k">attempts</span><span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{task.attempts}</span>
      </div>

      <div className="section-lbl">Description</div>
      <div className="desc">{desc}</div>

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
  const [filters, setFilters] = tasksPersistedState('df.tasksFilters', { active: true, pending: true, complete: false });
  const flipFilter = (k) => setFilters(f => ({ ...f, [k]: !f[k] }));

  const allTasks = DF_T.ACTIVE_TASKS;
  const projects = DF_T.PROJECTS.filter(p =>
    projectFilter.length === 0 || projectFilter.includes(p.id)
  );

  function statusMatches(s) {
    if (filters.active   && (s === 'in-progress' || s === 'blocked')) return true;
    if (filters.pending  && s === 'pending') return true;
    if (filters.complete && s === 'done') return true;
    return false;
  }
  function searchMatches(t) {
    if (!search) return true;
    const q = search.toLowerCase();
    return t.id.toLowerCase().includes(q) || t.title.toLowerCase().includes(q);
  }

  const selectedTask = selectedId ? allTasks.find(t => t.id === selectedId) : null;

  return (
    <div className="grid cols-12" style={{ gap: 16 }}>
      {/* Filter bar */}
      <div className="col-span-12" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="lbl" style={{ color: 'var(--fg-3)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>show</span>
        <div className="seg">
          <button className={filters.active   ? 'on' : ''} onClick={() => flipFilter('active')}>active</button>
          <button className={filters.pending  ? 'on' : ''} onClick={() => flipFilter('pending')}>pending</button>
          <button className={filters.complete ? 'on' : ''} onClick={() => flipFilter('complete')}>complete</button>
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
