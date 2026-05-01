/* Remaining tabs: orchestrators, performance, memory, recon, merge, costs, burndown */
const { Sparkline: SP, LineChart: LC, StackedAreaChart: SA, BarChart: BC, HBarChart: HBC, Donut: DN, StatTile: ST, HistBar: HB, PALETTE: CP } = window.DF_CHARTS;
const { Glyph: GL, ProjectGroup, Segmented } = window.DF_SHELL;
const DF = window.DF_DATA;
const { useState: uS, useEffect: uE } = React;

// shared open-state helper for furl/unfurl, persisted to localStorage by key
function useOpenSet(ids, defaultOpen = true, storageKey = null) {
  const [openMap, setOpenMap] = uS(() => {
    let stored = {};
    if (storageKey) {
      try { stored = JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; } catch (e) {}
    }
    const init = {};
    for (const id of ids) init[id] = id in stored ? !!stored[id] : defaultOpen;
    return init;
  });
  uE(() => {
    if (storageKey) {
      try { localStorage.setItem(storageKey, JSON.stringify(openMap)); } catch (e) {}
    }
  }, [storageKey, openMap]);
  const toggle = id => setOpenMap(m => ({ ...m, [id]: !m[id] }));
  const setAll = v => setOpenMap(Object.fromEntries(ids.map(id => [id, v])));
  return [openMap, toggle, setAll];
}

// generic persisted state hook
function usePersistedState(storageKey, defaultValue) {
  const [v, setV] = uS(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? defaultValue : JSON.parse(raw);
    } catch (e) { return defaultValue; }
  });
  uE(() => { try { localStorage.setItem(storageKey, JSON.stringify(v)); } catch (e) {} }, [storageKey, v]);
  return [v, setV];
}

function GroupAllToggle({ allOpen, onSetAll }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
      <button className="seg" onClick={() => onSetAll(!allOpen)} style={{ cursor: 'pointer', padding: '4px 10px', fontSize: 11, color: 'var(--fg-2)' }}>
        {allOpen ? '⌃ collapse all' : '⌄ expand all'}
      </button>
    </div>
  );
}

const fmtMs = ms => {
  if (!ms || ms <= 0) return '—';
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = s / 60;
  if (m < 60) return `${m.toFixed(0)}m`;
  return `${(m/60).toFixed(1)}h`;
};

// ── Cross-project aggregation helpers (no synthetic fallbacks) ──
// PERFORMANCE is keyed by project; each entry has paths/escalation/hist/ttc.
// These helpers return null when there's no data so the UI can render '—'.
function _weightedMean(samples) {
  // samples: [[value, weight], ...].  Returns null when total weight is 0.
  let num = 0, den = 0;
  for (const [v, w] of samples) {
    if (v == null || !w) continue;
    num += v * w;
    den += w;
  }
  return den > 0 ? num / den : null;
}

function aggTtcMs(perf, percentile) {
  // Weighted by per-project task count so big projects dominate.
  const samples = Object.values(perf || {})
    .map(p => [p.ttc?.[percentile], p.ttc?.count || 0]);
  return _weightedMean(samples);
}

function aggOnePassPct(perf) {
  let onePass = 0, total = 0;
  for (const p of Object.values(perf || {})) {
    for (const path of (p.paths || [])) {
      total += path.count || 0;
      if (path.path === 'one-pass') onePass += path.count || 0;
    }
  }
  return total > 0 ? (onePass / total) * 100 : null;
}

function aggEscalationRate(perf, kind /* 'steward_rate' | 'interactive_rate' */) {
  // Each project's escalation block carries a *_count and total_tasks; we don't
  // get total_tasks back in the redux shape, so weight by ttc.count instead.
  const samples = Object.values(perf || {})
    .map(p => [p.escalation?.[kind], p.ttc?.count || 0]);
  return _weightedMean(samples);
}

// ── Deps + locks chip lists ──
function shortPath(p) {
  // strip leading dirs, keep filename + parent
  const parts = p.split('/');
  return parts.length <= 2 ? p : parts.slice(-2).join('/');
}

function DepChip({ dep }) {
  return (
    <span className={`chip ${dep.done ? 'dep-done' : 'dep-pending'}`} title={`${dep.id}: ${dep.title}${dep.done ? ' (complete)' : ' (incomplete)'}`}>
      {window.DF_SHELL.taskId(dep.id)}
    </span>
  );
}

function LockChip({ path, holder, currentTaskId }) {
  let cls, hint;
  const holderDisplay = window.DF_SHELL.taskId(holder);
  if (!holder) { cls = 'lock-free'; hint = 'available'; }
  else if (holder === currentTaskId) { cls = 'lock-mine'; hint = 'held by this task'; }
  else { cls = 'lock-taken'; hint = `held by ${holderDisplay}`; }
  return (
    <span className={`chip ${cls}`} title={`${path} · ${hint}`}>
      {shortPath(path)}
      {cls === 'lock-taken' && <span className="holder">⊘ {holderDisplay}</span>}
    </span>
  );
}

// Generic chips list with truncate-or-expand. blockers (incomplete deps / taken locks) sort first.
function ChipList({ items, renderChip, maxInline = 2, persistKey, expandLayout = 'multiline', alwaysToggle = false }) {
  const [expanded, setExpanded] = persistKey ? usePersistedState(persistKey, false) : uS(false);
  if (!items || items.length === 0) return <span className="chip-empty">—</span>;

  if (expanded) {
    return (
      <div className={`chips ${expandLayout}`}>
        {items.map((it, i) => renderChip(it, i))}
        <button className="chip-toggle" onClick={() => setExpanded(false)}>collapse</button>
      </div>
    );
  }
  const visible = items.slice(0, maxInline);
  const hidden = items.length - visible.length;
  return (
    <div className="chips">
      {visible.map((it, i) => renderChip(it, i))}
      {hidden > 0
        ? <button className="chip-toggle" onClick={() => setExpanded(true)}>+{hidden}…</button>
        : alwaysToggle && items.length > 1
          ? <button className="chip-toggle" onClick={() => setExpanded(true)} title="expand to one per line">⇲</button>
          : null}
    </div>
  );
}

function DepsCell({ task }) {
  const sorted = [...(task.deps || [])].sort((a, b) => Number(a.done) - Number(b.done)); // incomplete first
  return <ChipList items={sorted} renderChip={(d) => <DepChip key={d.id} dep={d} />} maxInline={2} persistKey={`df.deps.${task.id}`} />;
}

function LocksCell({ task }) {
  const fileLocks = (DF.FILE_LOCKS && DF.FILE_LOCKS[task.project]) || {};
  const sorted = [...(task.locks || [])].sort((a, b) => {
    const ha = fileLocks[a]?.holder, hb = fileLocks[b]?.holder;
    const blockedA = ha && ha !== task.id ? 0 : 1;
    const blockedB = hb && hb !== task.id ? 0 : 1;
    return blockedA - blockedB; // blocked-by-other first
  });
  return <ChipList items={sorted} renderChip={(p) => <LockChip key={p} path={p} holder={fileLocks[p]?.holder} currentTaskId={task.id} />} maxInline={2} persistKey={`df.locks.${task.id}`} expandLayout="column" alwaysToggle={true} />;
}

// ── Orchestrators ──
function OrchTab({ projectFilter, search }) {
  const matches = DF.ORCHESTRATORS.filter(o => projectFilter.length === 0 || projectFilter.includes(o.project));
  const tasks = DF.ACTIVE_TASKS.filter(t => (projectFilter.length === 0 || projectFilter.includes(t.project))
    && (!search || (t.title + t.id).toLowerCase().includes(search.toLowerCase())));
  const orchIds = matches.map(o => o.pid);
  const [openMap, toggle, setAll] = useOpenSet(orchIds.map(String), true, 'df.open.orch');
  const allOpen = orchIds.every(p => openMap[String(p)]);
  const [filterMap, setFilterMap] = usePersistedState('df.orch.filter', {}); // { [pid]: { active, pending, complete } }
  const DEFAULT_FILTER = { active: true, pending: false, complete: false };
  const getFilter = (pid) => {
    const f = filterMap[pid];
    if (!f || typeof f !== 'object') return { ...DEFAULT_FILTER }; // back-compat: ignore old string values
    return { active: !!f.active, pending: !!f.pending, complete: !!f.complete };
  };
  const flipFilter = (pid, key) => {
    const cur = getFilter(pid);
    setFilterMap({ ...filterMap, [pid]: { ...cur, [key]: !cur[key] } });
  };

  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12 grid cols-4">
        <ST label="Orchestrators" value={matches.length} hint={`${matches.filter(o=>o.running).length} running`} spark={(DF.ORCHESTRATORS_SPARK?.values || []).slice(-30)} sparkColor={CP.accent} />
        <ST label="Tasks in flight" value={matches.reduce((s,o)=>s+o.summary.in_progress,0)} spark={DF.BURNDOWN.in_progress} sparkColor={CP.accent} hint="30d" />
        <ST label="Blocked" value={matches.reduce((s,o)=>s+o.summary.blocked,0)} spark={DF.BURNDOWN.blocked} sparkColor={CP.bad} hint="30d" />
        <ST label="Pending" value={matches.reduce((s,o)=>s+o.summary.pending,0)} spark={DF.BURNDOWN.pending} sparkColor={CP.warn} hint="30d" />
      </div>

      <div className="col-span-12"><GroupAllToggle allOpen={allOpen} onSetAll={setAll} /></div>

      {matches.map(o => {
        const total = o.summary.total || 1;
        const projTasks = tasks.filter(t => t.project === o.project);
        const filter = getFilter(o.pid);
        // partition by filter (multi-select)
        const filtered = projTasks.filter(t => {
          if (filter.active   && (t.status === 'in-progress' || t.status === 'blocked')) return true;
          if (filter.pending  && t.status === 'pending') return true;
          if (filter.complete && t.status === 'done')    return true;
          return false;
        });
        const counts = {
          active:   projTasks.filter(t => t.status === 'in-progress' || t.status === 'blocked').length,
          pending:  projTasks.filter(t => t.status === 'pending').length,
          complete: projTasks.filter(t => t.status === 'done').length,
        };

        const summary = (
          <>
            <span className="pip"><span className={`status-dot ${o.running ? 'running' : 'completed'}`} style={{ marginRight: 0 }}></span>{o.running ? 'running' : 'completed'}</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.ok }}></span>{o.summary.done}/{total}</span>
            {o.summary.in_progress > 0 && <span className="pip"><span className="pip-dot" style={{ background: CP.accent }}></span>{o.summary.in_progress} active</span>}
            {o.summary.blocked > 0 && <span className="pip"><span className="pip-dot" style={{ background: CP.bad }}></span>{o.summary.blocked} blocked</span>}
            <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>PID {o.pid}</span>
          </>
        );

        return (
          <div key={o.pid} className="col-span-12">
            <ProjectGroup id={String(o.pid)} label={o.project} open={openMap[String(o.pid)]} onToggle={() => toggle(String(o.pid))} summary={summary}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, gap: 12, flexWrap: 'wrap' }}>
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 2 }}>Current focus</div>
                      <div style={{ fontSize: 13, color: 'var(--fg-0)' }}>{o.current_task}</div>
                    </div>
                    <div className="seg" role="group" aria-label="Task filter">
                      <button className={filter.active   ? 'on' : ''} onClick={() => flipFilter(o.pid, 'active')}>Active · {counts.active}</button>
                      <button className={filter.pending  ? 'on' : ''} onClick={() => flipFilter(o.pid, 'pending')}>Pending · {counts.pending}</button>
                      <button className={filter.complete ? 'on' : ''} onClick={() => flipFilter(o.pid, 'complete')}>Complete · {counts.complete}</button>
                    </div>
                  </div>

                  <table className="tbl" style={{ marginTop: 4, tableLayout: 'fixed', width: '100%' }}>
                    <colgroup>
                      <col style={{ width: 80 }} />
                      <col />
                      <col style={{ width: 130 }} />
                      <col style={{ width: 60 }} />
                      <col style={{ width: 60 }} />
                      <col style={{ width: 100 }} />
                      <col style={{ width: 200 }} />
                      <col style={{ width: 240 }} />
                      <col style={{ width: 90 }} />
                    </colgroup>
                    <thead><tr>
                      <th>ID</th><th>Title</th><th>Agent</th>
                      <th className="num">Loops</th><th className="num">Tries</th>
                      <th className="num">Age</th>
                      <th>Deps</th><th>Locks</th>
                      <th>State</th>
                    </tr></thead>
                    <tbody>
                      {filtered.length === 0 && <tr><td colSpan={9} className="empty" style={{ padding: 20 }}>No {filter === 'all' ? '' : filter + ' '}tasks</td></tr>}
                      {filtered.map(t => {
                        const isDone = t.status === 'done';
                        const isPending = t.status === 'pending';
                        return (
                          <tr key={t.id}>
                            <td className="mono" style={{ color: 'var(--fg-1)', whiteSpace: 'nowrap' }}>{window.DF_SHELL.taskId(t.id)}</td>
                            <td style={{ color: isDone ? 'var(--fg-2)' : 'var(--fg-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={t.title}>{t.title}</td>
                            <td className="mono" style={{ color: 'var(--fg-2)', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.agent || '—'}</td>
                            <td className="num">{t.loops}</td>
                            <td className="num">{t.attempts}</td>
                            <td className="num" style={{ color: 'var(--fg-3)' }}>{isDone ? t.completed : isPending ? '—' : `${t.started}m`}</td>
                            <td><DepsCell task={t} /></td>
                            <td><LocksCell task={t} /></td>
                            <td>
                              <span className={`badge ${
                                t.status === 'blocked' ? 'bad' :
                                t.status === 'done'    ? 'ok'  :
                                t.status === 'pending' ? 'warn' : 'accent'}`}>{t.status}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--fg-3)', marginBottom: 4 }}>
                      <span>Progress</span>
                      <span className="mono" style={{ color: 'var(--fg-1)' }}>{o.summary.done}/{total}</span>
                    </div>
                    <div className="stack-bar" style={{ height: 12 }}>
                      <span style={{ width: `${o.summary.done/total*100}%`, background: CP.ok }} />
                      <span style={{ width: `${o.summary.in_progress/total*100}%`, background: CP.accent }} />
                      <span style={{ width: `${o.summary.blocked/total*100}%`, background: CP.bad }} />
                      <span style={{ width: `${o.summary.pending/total*100}%`, background: CP.warn }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--fg-3)', marginTop: 4 }}>
                      <span style={{ color: CP.ok }}>{o.summary.done} done</span>
                      <span style={{ color: CP.accent }}>{o.summary.in_progress} active</span>
                      <span style={{ color: CP.bad }}>{o.summary.blocked} blocked</span>
                      <span style={{ color: CP.warn }}>{o.summary.pending} pending</span>
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 4 }}>Started · {o.started}</div>
                    <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 4 }}>Completed / day · 30d</div>
                    {(() => {
                      const pb = DF.BURNDOWN_BY_PROJECT[o.project];
                      const rates = window.DF_SHELL.dailyDeltas(pb?.labels, pb?.done);
                      return <div style={{ height: 50 }}><SP values={rates} color={CP.accent} /></div>;
                    })()}
                  </div>
                </div>
              </div>
            </ProjectGroup>
          </div>
        );
      })}
    </div>
  );
}

// ── Performance ──
function PerfTab({ projectFilter }) {
  const projects = Object.keys(DF.PERFORMANCE).filter(p => projectFilter.length === 0 || projectFilter.includes(p));
  const [openMap, toggle, setAll] = useOpenSet(projects, true, 'df.open.perf');
  const allOpen = projects.every(p => openMap[p]);
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12 grid cols-4">
        {(() => {
          const subset = projectFilter.length === 0
            ? DF.PERFORMANCE
            : Object.fromEntries(Object.entries(DF.PERFORMANCE).filter(([pid]) => projectFilter.includes(pid)));
          const p50 = aggTtcMs(subset, 'p50');
          const p95 = aggTtcMs(subset, 'p95');
          const onePass = aggOnePassPct(subset);
          const escalation = aggEscalationRate(subset, 'interactive_rate');
          const totalTasks = Object.values(subset).reduce((s, p) => s + (p.ttc?.count || 0), 0);
          const fmtPct = v => v == null ? '—' : `${v.toFixed(1)}`;
          // Aggregate historical sparks across the in-scope projects.
          // Per-project hour buckets must be aligned by label before summing.
          const histKeys = Object.keys(subset);
          const buildAggSpark = field => {
            const bucketMap = {};
            histKeys.forEach(pid => {
              const h = subset[pid][field] || { labels: [], values: [], p50: [], p95: [] };
              const labels = h.labels || [];
              const values = field === 'time_centiles_history' ? (h.p95 || []) : (h.values || []);
              labels.forEach((lbl, i) => {
                bucketMap[lbl] = (bucketMap[lbl] || 0) + (values[i] || 0);
              });
            });
            return Object.keys(bucketMap).sort().map(k => bucketMap[k]);
          };
          const buildP50Spark = () => {
            const bucketMap = {};
            histKeys.forEach(pid => {
              const h = subset[pid].time_centiles_history || { labels: [], p50: [] };
              (h.labels || []).forEach((lbl, i) => {
                // For ttc: average across projects per bucket (medians don't sum).
                bucketMap[lbl] = bucketMap[lbl] || { sum: 0, n: 0 };
                bucketMap[lbl].sum += (h.p50 || [])[i] || 0;
                bucketMap[lbl].n += 1;
              });
            });
            return Object.keys(bucketMap).sort()
              .map(k => Math.round(bucketMap[k].sum / Math.max(1, bucketMap[k].n)));
          };
          const buildP95Spark = () => {
            const bucketMap = {};
            histKeys.forEach(pid => {
              const h = subset[pid].time_centiles_history || { labels: [], p95: [] };
              (h.labels || []).forEach((lbl, i) => {
                bucketMap[lbl] = bucketMap[lbl] || { sum: 0, n: 0 };
                bucketMap[lbl].sum += (h.p95 || [])[i] || 0;
                bucketMap[lbl].n += 1;
              });
            });
            return Object.keys(bucketMap).sort()
              .map(k => Math.round(bucketMap[k].sum / Math.max(1, bucketMap[k].n)));
          };
          const buildPctSpark = field => {
            const bucketMap = {};
            histKeys.forEach(pid => {
              const h = subset[pid][field] || { labels: [], values: [] };
              (h.labels || []).forEach((lbl, i) => {
                bucketMap[lbl] = bucketMap[lbl] || { sum: 0, n: 0 };
                bucketMap[lbl].sum += (h.values || [])[i] || 0;
                bucketMap[lbl].n += 1;
              });
            });
            return Object.keys(bucketMap).sort()
              .map(k => Math.round(bucketMap[k].sum / Math.max(1, bucketMap[k].n) * 10) / 10);
          };
          const p50Spark = buildP50Spark();
          const p95Spark = buildP95Spark();
          const onePassSpark = buildPctSpark('one_pass_history');
          const escalationSpark = buildPctSpark('escalation_history');
          return (
            <>
              <ST label="p50 time-to-completion"
                value={p50 == null ? '—' : fmtMs(p50)}
                hint={`${totalTasks} tasks (window)`} spark={p50Spark} sparkColor={CP.accent} />
              <ST label="p95 time-to-completion"
                value={p95 == null ? '—' : fmtMs(p95)}
                hint={`${totalTasks} tasks (window)`} spark={p95Spark} sparkColor={CP.warn} />
              <ST label="One-pass success" value={fmtPct(onePass)} unit={onePass == null ? '' : '%'}
                hint={onePass == null ? 'no tasks' : 'across all paths'} spark={onePassSpark} sparkColor={CP.ok} />
              <ST label="Human escalation rate" value={fmtPct(escalation)} unit={escalation == null ? '' : '%'}
                hint="interactive" spark={escalationSpark} sparkColor={CP.warn} />
            </>
          );
        })()}
      </div>

      <div className="col-span-12"><GroupAllToggle allOpen={allOpen} onSetAll={setAll} /></div>

      {projects.map(pid => {
        const p = DF.PERFORMANCE[pid];
        const donutData = p.paths.map(x => ({ label: x.path, value: x.count, color: CP.paths[x.path] }));
        const onePass = p.paths.find(x => x.path === 'one-pass');
        const onePassPct = onePass ? onePass.pct : 0;
        const summary = (
          <>
            <span className="pip"><span className="pip-dot" style={{ background: CP.accent }}></span>{fmtMs(p.ttc.p50)} p50</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.warn }}></span>{fmtMs(p.ttc.p95)} p95</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.ok }}></span>{onePassPct}% 1-pass</span>
            <span style={{ color: 'var(--fg-3)' }}>· {p.ttc.count} tasks</span>
          </>
        );
        return (
          <div key={pid} className="col-span-12">
            <ProjectGroup id={pid} label={pid} open={openMap[pid]} onToggle={() => toggle(pid)} summary={summary}>
              <div className="grid cols-12" style={{ gap: 12 }}>
                <div className="col-span-4 panel">
                  <div className="panel-head"><span className="title">Completion paths</span></div>
                  <div className="panel-body" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <DN data={donutData} size={120} thickness={20} centerValue={String(p.paths.reduce((s,x)=>s+x.count,0))} centerLabel="tasks" />
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {p.paths.map(x => (
                        <div key={x.path} style={{ display: 'grid', gridTemplateColumns: '8px 1fr auto auto', gap: 6, alignItems: 'center', fontSize: 11 }}>
                          <span style={{ width: 8, height: 8, background: CP.paths[x.path], borderRadius: 2 }}></span>
                          <span style={{ color: 'var(--fg-2)' }}>{x.path}</span>
                          <span className="mono" style={{ color: 'var(--fg-1)' }}>{x.count}</span>
                          <span className="mono" style={{ color: 'var(--fg-3)', width: 38, textAlign: 'right', fontSize: 10 }}>{x.pct}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="col-span-3 panel">
                  <div className="panel-head"><span className="title">Escalations</span></div>
                  <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div><div className="mono" style={{ fontSize: 22, color: 'var(--fg-0)' }}>{p.escalation.steward_rate}<span style={{ fontSize: 11, color: 'var(--fg-3)' }}>%</span></div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>steward</div></div>
                      <div><div className="mono" style={{ fontSize: 22, color: 'var(--warn,#fbbf24)' }}>{p.escalation.interactive_rate}<span style={{ fontSize: 11, color: 'var(--fg-3)' }}>%</span></div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>interactive</div></div>
                    </div>
                    <div style={{ height: 1, background: 'var(--line)' }}></div>
                    <div>
                      <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Human attention ({p.escalation.interactive_count} interactive)</div>
                      <div style={{ display: 'flex', gap: 8, fontSize: 11 }}>
                        {p.escalation.human_attention.zero > 0 && <span className="badge muted">{p.escalation.human_attention.zero} zero</span>}
                        {p.escalation.human_attention.minimal > 0 && <span className="badge warn">{p.escalation.human_attention.minimal} minimal</span>}
                        {p.escalation.human_attention.significant > 0 && <span className="badge bad">{p.escalation.human_attention.significant} significant</span>}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="col-span-5 panel">
                  <div className="panel-head"><span className="title">Time to completion · centiles</span></div>
                  <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
                      {[['p50',p.ttc.p50],['p75',p.ttc.p75],['p90',p.ttc.p90],['p95',p.ttc.p95]].map(([l,v]) => (
                        <div key={l}>
                          <div className="mono" style={{ fontSize: 18, color: 'var(--fg-0)' }}>{fmtMs(v)}</div>
                          <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>{l}</div>
                        </div>
                      ))}
                    </div>
                    {(() => {
                      const h = p.time_centiles_history || { labels: [], p50: [], p95: [] };
                      if (!h.labels || h.labels.length < 2) {
                        return (
                          <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>
                            ttc trend — not enough history yet
                          </div>
                        );
                      }
                      return (
                        <div>
                          <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                            ttc trend · per-hour
                          </div>
                          <LC
                            labels={h.labels}
                            series={[
                              { values: h.p50, color: CP.accent },
                              { values: h.p95, color: CP.warn },
                            ]}
                            height={70}
                            formatY={fmtMs}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>

                <div className="col-span-6 panel">
                  <div className="panel-head"><span className="title">Review cycles · outer loop</span></div>
                  <div className="panel-body"><BC labels={p.hist_outer.labels} values={p.hist_outer.values} height={140} /></div>
                </div>
                <div className="col-span-6 panel">
                  <div className="panel-head"><span className="title">Verify attempts · inner loop</span></div>
                  <div className="panel-body"><BC labels={p.hist_inner.labels} values={p.hist_inner.values} height={140} color={CP.info} /></div>
                </div>
              </div>
            </ProjectGroup>
          </div>
        );
      })}
    </div>
  );
}

// ── Memory ──
function MemoryTab({ projectFilter }) {
  const projects = Object.entries(DF.MEMORY_STATUS.projects).filter(([pid]) => projectFilter.length === 0 || projectFilter.includes(pid));
  const ts = DF.MEMORY_TIMESERIES;
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12 grid cols-4">
        <ST label="Graphiti nodes" value={DF.MEMORY_STATUS.graphiti.node_count.toLocaleString()}
            hint={`${DF.MEMORY_STATUS.graphiti.edge_count.toLocaleString()} edges`}
            spark={(DF.MEMORY_STATUS.graphiti.spark?.values || []).slice(-30)} sparkColor={CP.accent} />
        <ST label="Mem0 memories" value={DF.MEMORY_STATUS.mem0.memory_count.toLocaleString()}
            hint={`${DF.MEMORY_STATUS.graphiti.episode_count.toLocaleString()} episodes`}
            spark={(DF.MEMORY_STATUS.mem0.spark?.values || []).slice(-30)} sparkColor={CP.info} />
        <ST label="Write queue" value={DF.MEMORY_STATUS.queue.counts.pending}
            hint={DF.MEMORY_STATUS.queue.oldest_pending_age_seconds != null
              ? `${DF.MEMORY_STATUS.queue.oldest_pending_age_seconds}s oldest`
              : 'idle'}
            spark={(DF.MEMORY_STATUS.queue.spark?.values || []).slice(-30)} sparkColor={CP.warn} />
        {(() => {
          // Combined ops (read+write) — last hour bucket, plus a per-hour spark.
          const combined = ts.reads.map((r, i) => r + (ts.writes[i] || 0));
          const last = combined.length ? combined[combined.length - 1] : 0;
          return (
            <ST label="Ops / hr" value={last.toLocaleString()}
                spark={combined} sparkColor={CP.accent} hint="last 24h" />
          );
        })()}
      </div>

      <div className="col-span-8 panel">
        <div className="panel-head"><span className="title">Reads vs writes · last 24h</span></div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
            <span style={{ color: 'var(--fg-2)' }}><span style={{ display: 'inline-block', width: 10, height: 2, background: CP.accent, marginRight: 5, verticalAlign: 'middle' }}></span>reads</span>
            <span style={{ color: 'var(--fg-2)' }}><span style={{ display: 'inline-block', width: 10, height: 2, background: CP.ok, marginRight: 5, verticalAlign: 'middle' }}></span>writes</span>
          </div>
          <div style={{ flex: 1, minHeight: 220 }}>
            <LC labels={ts.labels} series={[
              { values: ts.reads, color: CP.accent },
              { values: ts.writes, color: CP.ok },
            ]} height={240} />
          </div>
        </div>
      </div>

      <div className="col-span-4 panel">
        <div className="panel-head"><span className="title">Operations · 24h</span></div>
        <div className="panel-body" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <DN data={DF.MEMORY_OPS_BREAKDOWN.map((d, i) => ({ ...d, color: [CP.accent, CP.ok, CP.warn, CP.info, CP.accent2, CP.bad][i % 6] }))} size={130} thickness={18} centerValue={DF.MEMORY_OPS_BREAKDOWN.reduce((s,d)=>s+d.value,0).toLocaleString()} centerLabel="ops" />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 5 }}>
            {DF.MEMORY_OPS_BREAKDOWN.map((d, i) => (
              <div key={d.label} style={{ display: 'grid', gridTemplateColumns: '8px 1fr auto', gap: 6, alignItems: 'center', fontSize: 11 }}>
                <span style={{ width: 8, height: 8, background: [CP.accent, CP.ok, CP.warn, CP.info, CP.accent2, CP.bad][i % 6], borderRadius: 2 }}></span>
                <span className="mono" style={{ color: 'var(--fg-2)' }}>{d.label}</span>
                <span className="mono" style={{ color: 'var(--fg-1)' }}>{d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="col-span-12 panel">
        <div className="panel-head"><span className="title">Per-project memory</span></div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead><tr><th>Project</th><th className="num">Graph nodes</th><th className="num">Vector memories</th><th className="num">Δ-24h</th><th>Mix</th></tr></thead>
            <tbody>
              {projects.map(([pid, s]) => {
                const total = (s.graphiti_nodes || 0) + (s.mem0_memories || 0);
                const gPct = total > 0 ? (s.graphiti_nodes / total) * 100 : 0;
                const fmtDelta = (cur, before) => {
                  if (before == null || cur == null) return null;
                  const d = cur - before;
                  if (d === 0) return '0';
                  const sign = d > 0 ? '+' : '';
                  return `${sign}${d.toLocaleString()}`;
                };
                const dG = fmtDelta(s.graphiti_nodes, s.graphiti_nodes_24h_ago);
                const dM = fmtDelta(s.mem0_memories, s.mem0_memories_24h_ago);
                return (
                  <tr key={pid}>
                    <td className="mono" style={{ color: 'var(--fg-1)' }}>{pid}</td>
                    <td className="num">{(s.graphiti_nodes || 0).toLocaleString()}</td>
                    <td className="num">{(s.mem0_memories || 0).toLocaleString()}</td>
                    <td className="num mono" style={{ fontSize: 11, color: 'var(--fg-2)' }}>
                      {dG == null && dM == null ? '—' : (
                        <span>
                          <span style={{ color: dG && dG.startsWith('-') ? 'var(--bad,#f87171)' : 'var(--fg-2)' }}>{dG ?? '—'}</span>
                          <span style={{ color: 'var(--fg-3)' }}> / </span>
                          <span style={{ color: dM && dM.startsWith('-') ? 'var(--bad,#f87171)' : 'var(--fg-2)' }}>{dM ?? '—'}</span>
                        </span>
                      )}
                    </td>
                    <td style={{ width: 240 }}>
                      <div className="stack-bar"><span style={{ width: `${gPct}%`, background: CP.accent }} /><span style={{ width: `${100-gPct}%`, background: CP.info }} /></div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Reconciliation ──
function ReconTab({ projectFilter, search }) {
  const r = DF.RECON_STATE;
  const runs = r.runs.filter(x => {
    const pid = x.project_id || x.project;
    return (projectFilter.length === 0 || projectFilter.includes(pid))
      && (!search || x.id.toLowerCase().includes(search.toLowerCase()));
  });
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      {(() => {
        // Derive "Last full run" from per-project watermarks: pick the most
        // recent last_full_run_completed across all projects.
        let lastFullProject = null, lastFullIso = null;
        for (const [pid, w] of Object.entries(r.watermarks || {})) {
          const ts = w.last_full_run_completed;
          if (ts && (!lastFullIso || ts > lastFullIso)) {
            lastFullIso = ts; lastFullProject = pid;
          }
        }
        const totalRuns = r.runs.length;
        const successCount = r.runs.filter(x => x.status === 'success').length;
        const successPct = totalRuns ? Math.round(successCount / totalRuns * 100) : null;
        // Sparkline of recent run durations (oldest first).
        const durSpark = r.runs
          .filter(x => x.duration_seconds != null)
          .slice(0, 40)
          .map(x => x.duration_seconds)
          .reverse();
        return (
          <div className="col-span-12 grid cols-4">
            <ST label="Buffered events" value={r.buffer.buffered_count}
                hint={r.buffer.oldest_event_age_seconds != null
                  ? `oldest ${r.buffer.oldest_event_age_seconds}s`
                  : 'idle'}
                spark={(r.buffer.spark?.values || []).slice(-30)} sparkColor={CP.warn} />
            <ST label="Active agents" value={r.burst_state.length}
                hint={`${r.burst_state.filter(b=>b.state!=='idle').length} non-idle`}
                spark={(r.agents_spark?.values || []).slice(-30)} sparkColor={CP.accent} />
            <ST label="Last full run"
                value={lastFullIso ? window.DF_SHELL.timeago(lastFullIso) : '—'}
                hint={lastFullProject || 'no completed run'}
                spark={[]} />
            <ST label="Run success rate"
                value={successPct != null ? successPct : '—'}
                unit={successPct != null ? '%' : ''}
                hint={`${totalRuns} runs · last ${durSpark.length} durations`}
                spark={durSpark} sparkColor={CP.ok} />
          </div>
        );
      })()}

      <div className="col-span-4 panel">
        <div className="panel-head"><span className="title">Burst state</span></div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead><tr><th>Agent</th><th>State</th><th>Last write</th></tr></thead>
            <tbody>
              {r.burst_state.map(b => (
                <tr key={b.agent_id}>
                  <td className="mono" style={{ fontSize: 11 }}>{b.agent_id}</td>
                  <td><span className={`badge ${b.state === 'bursting' ? 'accent' : b.state === 'cooling' ? 'warn' : b.state === 'running' ? 'info' : 'muted'}`}>{b.state}</span></td>
                  <td style={{ color: 'var(--fg-3)' }}>{window.DF_SHELL.timeago(b.last_write_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="col-span-8 panel">
        <div className="panel-head"><span className="title">Watermarks · per project</span></div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead><tr><th>Project</th><th>Last full</th><th>Last episode</th><th>Last memory</th><th>Last task Δ</th></tr></thead>
            <tbody>
              {Object.entries(r.watermarks).filter(([pid]) => projectFilter.length === 0 || projectFilter.includes(pid)).map(([pid, w]) => (
                <tr key={pid}>
                  <td className="mono" style={{ color: 'var(--fg-1)' }}>{pid}</td>
                  <td style={{ color: 'var(--fg-2)' }}>{window.DF_SHELL.timeago(w.last_full_run_completed)}</td>
                  <td style={{ color: 'var(--fg-2)' }}>{window.DF_SHELL.timeago(w.last_episode_timestamp)}</td>
                  <td style={{ color: 'var(--fg-2)' }}>{window.DF_SHELL.timeago(w.last_memory_timestamp)}</td>
                  <td style={{ color: 'var(--fg-2)' }}>{window.DF_SHELL.timeago(w.last_task_change_timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="col-span-12 panel">
        <div className="panel-head">
          <span className="title">Recent runs</span>
          <span className="meta">{runs.length} matching</span>
        </div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead><tr><th>Run</th><th>Project</th><th>Trigger</th><th>Type</th><th>Status</th><th className="num">Events</th><th className="num">Duration</th><th>When</th><th className="num">Journal</th></tr></thead>
            <tbody>
              {runs.map(rn => (
                <tr key={rn.id}>
                  <td className="mono" style={{ color: 'var(--accent)' }}>{rn.id}</td>
                  <td className="mono">{rn.project_id || rn.project}</td>
                  <td style={{ color: 'var(--fg-2)' }}>{rn.trigger_reason || rn.trigger}</td>
                  <td>{rn.run_type || rn.type}</td>
                  <td><span className={`badge ${rn.status === 'success' || rn.status === 'completed' ? 'ok' : rn.status === 'failed' ? 'bad' : 'warn'}`}>{rn.status}</span></td>
                  <td className="num">{rn.events_processed ?? rn.events ?? 0}</td>
                  <td className="num">{rn.duration_seconds != null ? `${rn.duration_seconds.toFixed(1)}s` : '—'}</td>
                  <td style={{ color: 'var(--fg-3)' }}>{window.DF_SHELL.timeago(rn.started_at)}</td>
                  <td className="num"><span className="badge accent">{rn.journal_entry_count ?? 0}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Merge Queue ──
function MergeTab({ projectFilter }) {
  const projects = Object.entries(DF.MERGE_QUEUE).filter(([pid]) => projectFilter.length === 0 || projectFilter.includes(pid));
  const projIds = projects.map(([pid]) => pid);
  const [openMap, toggle, setAll] = useOpenSet(projIds, true, 'df.open.merge');
  const allOpen = projIds.every(p => openMap[p]);
  const totals = projects.reduce((acc, [_, d]) => ({
    count: acc.count + d.latency.count,
    hits: acc.hits + d.speculative.hit_count,
    discards: acc.discards + d.speculative.discard_count,
    active: acc.active + d.active.length,
  }), { count: 0, hits: 0, discards: 0, active: 0 });
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      {(() => {
        // Aggregate the per-project queue-depth time-series by index. Different
        // projects can have different bucket counts; we line them up at the
        // tail so the most-recent buckets stay aligned.
        const allDepths = projects.map(([, d]) => d.depth?.values || []);
        const maxLen = allDepths.reduce((m, a) => Math.max(m, a.length), 0);
        const aggDepth = Array.from({ length: maxLen }, (_, i) =>
          allDepths.reduce((s, a) => {
            const off = a.length - maxLen;
            const val = a[i + off];
            return s + (val || 0);
          }, 0),
        );
        // Worst-case p95 across projects (max — informative for SLO).
        const p95s = projects.map(([, d]) => d.latency?.p95).filter(v => v != null && v > 0);
        const p95 = p95s.length ? Math.max(...p95s) : null;
        const hitPct = totals.hits + totals.discards > 0
          ? Math.round(totals.hits / (totals.hits + totals.discards) * 100)
          : null;
        return (
          <div className="col-span-12 grid cols-4">
            <ST label="Merges (window)" value={totals.count}
                spark={aggDepth} sparkColor={CP.accent} />
            {(() => {
              // Aggregate the per-project active_spark series by label.
              const labelMap = {};
              projects.forEach(([, d]) => {
                const sp = d.active_spark || { labels: [], values: [] };
                (sp.labels || []).forEach((lbl, i) => {
                  labelMap[lbl] = (labelMap[lbl] || 0) + ((sp.values || [])[i] || 0);
                });
              });
              const activeSpark = Object.keys(labelMap).sort().map(k => labelMap[k]).slice(-30);
              return (
                <ST label="In queue now" value={totals.active}
                    hint={`${projects.filter(([_,d])=>d.active.length>0).length} projects`}
                    spark={activeSpark} sparkColor={CP.warn} />
              );
            })()}
            <ST label="Speculative hit rate"
                value={hitPct != null ? hitPct : '—'} unit={hitPct != null ? '%' : ''}
                hint={`${totals.hits}/${totals.hits + totals.discards} attempts`}
                spark={[]} sparkColor={CP.ok} />
            <ST label="p95 latency · worst project"
                value={p95 != null ? fmtMs(p95) : '—'}
                hint={p95s.length ? `${p95s.length} projects` : 'no merges'}
                spark={[]} sparkColor={CP.warn} />
          </div>
        );
      })()}

      <div className="col-span-12"><GroupAllToggle allOpen={allOpen} onSetAll={setAll} /></div>

      {projects.map(([pid, d]) => {
        const hitPct = Math.round(d.speculative.hit_rate * 100);
        const summary = (
          <>
            <span className="pip"><span className="pip-dot" style={{ background: CP.accent }}></span>{d.latency.count} attempts</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.warn }}></span>{d.active.length} queued</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.ok }}></span>{fmtMs(d.latency.p50)} p50</span>
            <span style={{ color: 'var(--fg-3)' }}>· {hitPct}% spec hit</span>
          </>
        );
        return (
          <div key={pid} className="col-span-12">
            <ProjectGroup id={pid} label={pid} open={openMap[pid]} onToggle={() => toggle(pid)} summary={summary}>
              <div className="grid cols-12" style={{ gap: 12 }}>
                <div className="col-span-7 panel">
                  <div className="panel-head"><span className="title">Merge attempts · 15-min buckets</span></div>
                  <div className="panel-body"><LC labels={d.depth.labels.map(String)} series={[{ values: d.depth.values, color: CP.accent }]} height={180} /></div>
                </div>

                <div className="col-span-5 panel">
                  <div className="panel-head"><span className="title">Outcomes</span></div>
                  <div className="panel-body" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <DN data={d.outcomes.labels.map((l, i) => ({ label: l, value: d.outcomes.values[i], color: CP.status[l] || CP.accent }))} size={120} thickness={18} centerValue={d.outcomes.values.reduce((s,v)=>s+v,0)} centerLabel="merges" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, flex: 1 }}>
                      {d.outcomes.labels.map((l, i) => (
                        <div key={l} style={{ display: 'grid', gridTemplateColumns: '8px 1fr auto', gap: 6, alignItems: 'center', fontSize: 11 }}>
                          <span style={{ width: 8, height: 8, background: CP.status[l] || CP.accent, borderRadius: 2 }}></span>
                          <span style={{ color: 'var(--fg-2)' }}>{l}</span>
                          <span className="mono">{d.outcomes.values[i]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="col-span-6 panel">
                  <div className="panel-head"><span className="title">Latency centiles</span></div>
                  <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
                    {[['p50',d.latency.p50],['p95',d.latency.p95],['p99',d.latency.p99],['mean',d.latency.mean_ms]].map(([l,v]) => (
                      <div key={l}><div className="mono" style={{ fontSize: 18 }}>{fmtMs(v)}</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>{l}</div></div>
                    ))}
                  </div>
                </div>
                <div className="col-span-6 panel">
                  <div className="panel-head"><span className="title">Speculative merge</span></div>
                  <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
                    <div><div className="mono" style={{ fontSize: 18, color: CP.ok }}>{d.speculative.hit_count}</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>hits</div></div>
                    <div><div className="mono" style={{ fontSize: 18, color: CP.warn }}>{d.speculative.discard_count}</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>discards</div></div>
                    <div><div className="mono" style={{ fontSize: 18 }}>{Math.round(d.speculative.hit_rate*100)}%</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>hit rate</div></div>
                  </div>
                </div>

                {d.active.length > 0 && (
                  <div className="col-span-6 panel">
                    <div className="panel-head"><span className="title">Currently queued</span></div>
                    <div className="panel-body flush">
                      <table className="tbl"><thead><tr><th>Task</th><th>Title</th><th>State</th><th>Branch</th><th className="num">When</th></tr></thead>
                        <tbody>
                          {d.active.map((row, i) => (
                            <tr key={i}>
                              <td className="mono">{row.task_id}</td>
                              <td>{row.title}</td>
                              <td><span className={`badge ${row.state === 'in_flight' ? 'warn' : 'info'}`}>{row.state}</span></td>
                              <td className="mono" style={{ color: 'var(--fg-3)', fontSize: 11 }}>{row.branch}</td>
                              <td className="num" style={{ color: 'var(--fg-3)' }}>{row.timestamp}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                <div className={d.active.length > 0 ? 'col-span-6 panel' : 'col-span-12 panel'}>
                  <div className="panel-head"><span className="title">Recent merges</span></div>
                  <div className="panel-body flush">
                    <table className="tbl"><thead><tr><th>Task</th><th>Title</th><th>Outcome</th><th className="num">Duration</th><th className="num">When</th></tr></thead>
                      <tbody>
                        {d.recent.map((row, i) => (
                          <tr key={i}>
                            <td className="mono">{row.task_id}</td>
                            <td>{row.title}</td>
                            <td><span className={`badge ${row.outcome === 'done' ? 'ok' : row.outcome === 'conflict' ? 'warn' : row.outcome === 'blocked' ? 'bad' : 'info'}`}>{row.outcome}</span></td>
                            <td className="num">{fmtMs(row.duration_ms)}</td>
                            <td className="num" style={{ color: 'var(--fg-3)' }}>{row.timestamp}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </ProjectGroup>
          </div>
        );
      })}
    </div>
  );
}

// ── Costs ──
function CostsTab({ projectFilter }) {
  const c = DF.COSTS;
  const projects = c.by_project.filter(p => projectFilter.length === 0 || projectFilter.includes(p.project));
  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      <div className="col-span-12 grid cols-4">
        <ST label="Total spend" value={`$${c.summary.total.toFixed(2)}`}
            delta={c.summary.delta_pct != null ? `${c.summary.delta_pct}%` : null}
            deltaDir={c.summary.delta_pct == null ? null : (c.summary.delta_pct < 0 ? 'down' : 'up')}
            spark={c.trend.values} sparkColor={CP.accent}
            hint={c.summary.delta_pct == null && c.summary.delta_hint
              ? c.summary.delta_hint
              : 'window total'} />
        <ST label="Runs"
            value={c.summary.runs ? c.summary.runs.toLocaleString() : '—'}
            hint={c.summary.runs
              ? `avg $${(c.summary.total / c.summary.runs).toFixed(3)}/run`
              : 'no runs in window'}
            spark={[]} sparkColor={CP.info} />
        {(() => {
          const t = c.summary.tokens;
          const tot = t && typeof t === 'object' ? t.total : t;
          const display = tot != null ? `${(tot/1e6).toFixed(2)}M` : '—';
          const breakdown = (t && typeof t === 'object' && tot != null)
            ? `in ${(t.input/1e6).toFixed(1)}M · out ${(t.output/1e6).toFixed(1)}M · cache ${((t.cache_read+t.cache_create)/1e6).toFixed(1)}M`
            : 'no token data';
          return (
            <ST label="Tokens" value={display} hint={breakdown}
                spark={[]} sparkColor={CP.ok} />
          );
        })()}
        <ST label="p95 run cost"
            value={c.summary.p95_run_cost != null ? `$${c.summary.p95_run_cost.toFixed(2)}` : '—'}
            hint={c.summary.p95_run_cost != null ? 'across all runs' : 'no runs in window'}
            spark={[]} sparkColor={CP.warn} />
      </div>

      <div className="col-span-7 panel">
        <div className="panel-head"><span className="title">Spend trend · 30d</span></div>
        <div className="panel-body"><LC labels={c.trend.labels} series={[{ values: c.trend.values, color: CP.accent }]} height={200} formatY={v => `$${v.toFixed(0)}`} /></div>
      </div>

      <div className="col-span-5 panel">
        <div className="panel-head"><span className="title">By account</span></div>
        <div className="panel-body" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <DN data={c.by_account.map((a, i) => ({ label: a.account, value: a.total, color: [CP.accent, CP.warn, CP.ok, CP.info][i % 4] }))} size={130} thickness={18} centerValue={`$${c.summary.total.toFixed(0)}`} centerLabel="total" />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {c.by_account.map((a, i) => (
              <div key={a.account} style={{ display: 'grid', gridTemplateColumns: '8px 1fr auto auto', gap: 6, alignItems: 'center', fontSize: 11 }}>
                <span style={{ width: 8, height: 8, background: [CP.accent, CP.warn, CP.ok, CP.info][i % 4], borderRadius: 2 }}></span>
                <span className="mono" style={{ color: 'var(--fg-1)' }}>{a.account}</span>
                <span className={`badge ${a.status === 'capped' ? 'warn' : 'muted'}`} style={{ fontSize: 9 }}>{a.status === 'capped' ? a.resets_at : a.status}</span>
                <span className="mono" style={{ color: 'var(--fg-1)' }}>${a.total.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="col-span-6 panel">
        <div className="panel-head"><span className="title">By project · stacked by model</span></div>
        <div className="panel-body">
          {(() => {
            // Discover the set of model keys present in the data; reserved
            // keys (project / total) are excluded.  Stable color cycle.
            const reserved = new Set(['project', 'total']);
            const modelKeys = new Set();
            for (const row of projects) {
              for (const k of Object.keys(row)) if (!reserved.has(k)) modelKeys.add(k);
            }
            const palette = [CP.accent, CP.info, CP.ok, CP.warn, CP.bad, CP.accent2 || CP.accent];
            const segments = [...modelKeys].map((k, i) => ({
              key: k, color: palette[i % palette.length], label: k,
            }));
            return (
              <HBC rows={projects} valueKey="total" labelKey="project"
                   segments={segments}
                   formatVal={v => `$${v.toFixed(2)}`} />
            );
          })()}
        </div>
      </div>

      <div className="col-span-6 panel">
        <div className="panel-head"><span className="title">By role</span></div>
        <div className="panel-body">
          <HBC rows={c.by_role} valueKey="total" labelKey="role" formatVal={v => `$${v.toFixed(2)}`} />
        </div>
      </div>

      <div className="col-span-12 panel">
        <div className="panel-head"><span className="title">Account events</span></div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead><tr><th>When</th><th>Account</th><th>Event</th><th>Detail</th></tr></thead>
            <tbody>
              {c.events.map((e, i) => (
                <tr key={i}>
                  <td className="mono" style={{ color: 'var(--fg-3)' }}>{window.DF_SHELL.timeago(e.ts)}</td>
                  <td className="mono">{e.account}</td>
                  <td><span className={`badge ${e.event === 'rate_limited' || e.event === 'cap_hit' ? 'warn' : e.event === 'cap_reset' || e.event === 'resumed' ? 'ok' : e.event === 'auth_failed' ? 'bad' : 'info'}`}>{e.event}</span></td>
                  <td style={{ color: 'var(--fg-2)' }}>{e.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Burndown ──
function BurnTab({ projectFilter }) {
  const b = DF.BURNDOWN;
  const [view, setView] = uS('aggregate'); // 'aggregate' | 'per-project'
  const projects = DF.PROJECTS.filter(p => projectFilter.length === 0 || projectFilter.includes(p.id));
  const projIds = projects.map(p => p.id);
  const [openMap, toggle, setAll] = useOpenSet(projIds, true, 'df.open.burn');
  const allOpen = projIds.every(p => openMap[p]);

  return (
    <div className="grid cols-12" style={{ gap: 12 }}>
      {(() => {
        const totalDone = b.done.reduce((s,x)=>s+x,0);
        const days = b.labels.length || 1;
        const velocity = totalDone / days;  // tasks/day average over the window
        const lastPending = b.pending.length ? b.pending[b.pending.length-1] : 0;
        const firstPending = b.pending.length ? b.pending[0] : 0;
        const forecastDays = velocity > 0 ? Math.round(lastPending / velocity) : null;
        return (
          <div className="col-span-12 grid cols-4">
            <ST label="Net velocity" value={velocity.toFixed(1)} unit="/day"
                hint={`window avg · ${days}d`}
                spark={b.done} sparkColor={CP.ok} />
            <ST label="Completed (window)" value={totalDone}
                spark={b.done} sparkColor={CP.ok} />
            <ST label="Backlog" value={lastPending}
                delta={`${lastPending - firstPending}`}
                deltaDir={lastPending < firstPending ? 'down' : 'up'}
                spark={b.pending} sparkColor={CP.warn} />
            {(() => {
              // Server-computed forecast confidence (recent 7d vs lifetime
              // velocity) is null when <7 days of history. Fall back to the
              // simple point estimate above so the tile still shows a value
              // once any history exists.
              const lo = b.forecast_low;
              const hi = b.forecast_high;
              const haveRange = lo != null && hi != null;
              const display = haveRange
                ? (lo === hi ? `${lo}d` : `${lo}–${hi}d`)
                : (forecastDays != null ? `${forecastDays}d` : '—');
              const hint = haveRange
                ? `${lastPending} pending · 7d vs lifetime velocity`
                : (forecastDays != null
                    ? `${lastPending} / ${velocity.toFixed(1)} per day · need 7d for range`
                    : (velocity === 0 ? 'velocity is zero' : 'no data'));
              return (
                <ST label="Forecast clear" value={display} hint={hint}
                    spark={b.pending} sparkColor={CP.ok} />
              );
            })()}
          </div>
        );
      })()}

      <div className="col-span-12" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Segmented options={[{ value: 'aggregate', label: 'Aggregate' }, { value: 'per-project', label: 'Per project' }]} value={view} onChange={setView} />
        {view === 'per-project' && <GroupAllToggle allOpen={allOpen} onSetAll={setAll} />}
      </div>

      {view === 'aggregate' && (
        <div className="col-span-12 panel">
          <div className="panel-head">
            <span className="title">Status mix · 30d</span>
            <span className="meta">aggregate · all projects</span>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              {[['done',CP.ok],['in-progress',CP.accent],['blocked',CP.bad],['pending',CP.warn]].map(([l,c]) => (
                <span key={l} style={{ color: 'var(--fg-2)' }}><span style={{ display: 'inline-block', width: 10, height: 10, background: c, marginRight: 5, verticalAlign: 'middle', borderRadius: 2 }}></span>{l}</span>
              ))}
            </div>
            <SA labels={b.labels} stacks={[
              { key: 'done',        color: CP.ok,     values: b.done },
              { key: 'in_progress', color: CP.accent, values: b.in_progress },
              { key: 'blocked',     color: CP.bad,    values: b.blocked },
              { key: 'pending',     color: CP.warn,   values: b.pending },
            ]} height={300} />
          </div>
        </div>
      )}

      {view === 'aggregate' && (
        <div className="col-span-12 panel">
          <div className="panel-head"><span className="title">Per project · summary</span></div>
          <div className="panel-body flush">
            <table className="tbl">
              <thead><tr><th>Project</th><th className="num">Completed</th><th className="num">Active</th><th className="num">Blocked</th><th className="num">Pending</th><th>Trend</th></tr></thead>
              <tbody>
                {projects.map(p => {
                  const pb = DF.BURNDOWN_BY_PROJECT[p.id];
                  if (!pb) return null;
                  const done = pb.done.reduce((s,x)=>s+x,0);
                  const active = pb.in_progress[pb.in_progress.length-1];
                  const blocked = pb.blocked[pb.blocked.length-1];
                  const pending = pb.pending[pb.pending.length-1];
                  return (
                    <tr key={p.id}>
                      <td className="mono">{p.id}</td>
                      <td className="num" style={{ color: CP.ok }}>{done}</td>
                      <td className="num" style={{ color: CP.accent }}>{active}</td>
                      <td className="num" style={{ color: CP.bad }}>{blocked}</td>
                      <td className="num" style={{ color: CP.warn }}>{pending}</td>
                      <td style={{ width: 200 }}><div style={{ height: 22 }}><SP values={pb.done} color={CP.accent} /></div></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {view === 'per-project' && projects.map(p => {
        const pb = DF.BURNDOWN_BY_PROJECT[p.id];
        if (!pb) return null;
        const done = pb.done.reduce((s,x)=>s+x,0);
        const last = i => pb[i][pb[i].length-1];
        const summary = (
          <>
            <span className="pip"><span className="pip-dot" style={{ background: CP.ok }}></span>{done} done</span>
            <span className="pip"><span className="pip-dot" style={{ background: CP.accent }}></span>{last('in_progress')} active</span>
            {last('blocked') > 0 && <span className="pip"><span className="pip-dot" style={{ background: CP.bad }}></span>{last('blocked')} blocked</span>}
            <span className="pip"><span className="pip-dot" style={{ background: CP.warn }}></span>{last('pending')} pending</span>
            <span style={{ color: 'var(--fg-3)' }}>· {(done/30).toFixed(1)}/day</span>
          </>
        );
        return (
          <div key={p.id} className="col-span-12">
            <ProjectGroup id={p.id} label={p.id} open={openMap[p.id]} onToggle={() => toggle(p.id)} summary={summary}>
              <div className="grid cols-12" style={{ gap: 12 }}>
                <div className="col-span-8 panel">
                  <div className="panel-head">
                    <span className="title">Status mix · 30d</span>
                  </div>
                  <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
                      {[['done',CP.ok],['in-progress',CP.accent],['blocked',CP.bad],['pending',CP.warn]].map(([l,c]) => (
                        <span key={l} style={{ color: 'var(--fg-2)' }}><span style={{ display: 'inline-block', width: 10, height: 10, background: c, marginRight: 5, verticalAlign: 'middle', borderRadius: 2 }}></span>{l}</span>
                      ))}
                    </div>
                    <SA labels={b.labels} stacks={[
                      { key: 'done',        color: CP.ok,     values: pb.done },
                      { key: 'in_progress', color: CP.accent, values: pb.in_progress },
                      { key: 'blocked',     color: CP.bad,    values: pb.blocked },
                      { key: 'pending',     color: CP.warn,   values: pb.pending },
                    ]} height={220} />
                  </div>
                </div>

                <div className="col-span-4 panel">
                  <div className="panel-head"><span className="title">Velocity</span></div>
                  <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      <div><div className="mono" style={{ fontSize: 22, color: CP.ok }}>{(done/30).toFixed(1)}</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>completed/day</div></div>
                      <div><div className="mono" style={{ fontSize: 22, color: CP.warn }}>{last('pending')}</div><div style={{ fontSize: 10, color: 'var(--fg-3)' }}>backlog now</div></div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Completion trend</div>
                      <div style={{ height: 36 }}><SP values={pb.done} color={CP.ok} /></div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Backlog trend</div>
                      <div style={{ height: 36 }}><SP values={pb.pending} color={CP.warn} /></div>
                    </div>
                  </div>
                </div>
              </div>
            </ProjectGroup>
          </div>
        );
      })}
    </div>
  );
}

window.DF_TABS = { OrchTab, PerfTab, MemoryTab, ReconTab, MergeTab, CostsTab, BurnTab };
