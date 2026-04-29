/* Overview tab — command-center grid */
const { Sparkline, LineChart, StackedAreaChart, BarChart, HBarChart, Donut, StatTile, HistBar, PALETTE: P } = window.DF_CHARTS;
const { Glyph, LiveFeed } = window.DF_SHELL;
const D = window.DF_DATA;

function StatusDot({ kind }) { return <span className={`status-dot ${kind}`}></span>; }

function OverviewTab({ paused }) {
  // Compute live numbers
  const orchRunning = D.ORCHESTRATORS.filter(o => o.running).length;
  const tasksTotal = D.ORCHESTRATORS.reduce((s, o) => s + o.summary.total, 0);
  const tasksDone = D.ORCHESTRATORS.reduce((s, o) => s + o.summary.done, 0);
  const tasksInP = D.ORCHESTRATORS.reduce((s, o) => s + o.summary.in_progress, 0);
  const tasksBlocked = D.ORCHESTRATORS.reduce((s, o) => s + o.summary.blocked, 0);
  const tasksPending = D.ORCHESTRATORS.reduce((s, o) => s + o.summary.pending, 0);
  const memTotal = Object.values(D.MEMORY_STATUS.projects).reduce((s, p) => s + p.graphiti_nodes + p.mem0_memories, 0);
  const queue = D.MEMORY_STATUS.queue.counts;
  const queueDepth = queue.pending + queue.retry + queue.dead;

  // Combined memory throughput sparkline: per-hour read+write counts (last 24h).
  const memOpsSpark = D.MEMORY_TIMESERIES.reads.map(
    (r, i) => r + (D.MEMORY_TIMESERIES.writes[i] || 0),
  );
  // ops/min in the most recent hour bucket.
  const opsLast = memOpsSpark.length ? memOpsSpark[memOpsSpark.length - 1] : 0;
  const opsPerMin = (opsLast / 60).toFixed(1);
  // Real recon-latency sparkline: most-recent N run durations, oldest first.
  const reconRuns = D.RECON_STATE.runs || [];
  const reconLatencySpark = reconRuns
    .filter(r => r.duration_seconds != null)
    .slice(0, 40)
    .map(r => r.duration_seconds)
    .reverse();
  const costSpark = (D.COSTS.trend.values || []).slice(-30);
  const todaySpend = D.COSTS.summary?.today ?? 0;
  const deltaPct = D.COSTS.summary?.delta_pct;

  return (
    <div className="grid cols-12" style={{ gridTemplateRows: 'auto auto 1fr', gap: 12, height: '100%' }}>

      {/* Row 1: KPI tiles */}
      <div className="col-span-12 grid cols-4">
        <StatTile label="Orchestrators running" value={orchRunning} unit={`/ ${D.ORCHESTRATORS.length}`}
          spark={[]} sparkColor={P.accent} hint="live" />
        <StatTile label="Active tasks" value={tasksInP + tasksBlocked} unit={`/ ${tasksTotal}`}
          spark={D.BURNDOWN.in_progress} sparkColor={P.accent} hint={`${tasksDone} done`} />
        <StatTile label="Memory ops / min" value={opsPerMin} unit="ops"
          spark={memOpsSpark} sparkColor={P.ok} hint="last 24h hourly" />
        <StatTile label="Spend (today)" value={`$${todaySpend.toFixed(2)}`}
          delta={deltaPct != null ? `${deltaPct}%` : null}
          deltaDir={deltaPct != null ? (deltaPct < 0 ? 'down' : 'up') : null}
          spark={costSpark} sparkColor={P.warn}
          hint={D.COSTS.summary?.runs ? `${D.COSTS.summary.runs} runs (window)` : 'no cost data'} />
      </div>

      {/* Row 2: Wide chart + side panels */}
      <div className="col-span-8 panel">
        <div className="panel-head">
          <span className="title">Activity timeline</span>
          <span style={{ color: 'var(--fg-3)' }}>· last 24h · 1h buckets</span>
          <span className="meta">{Math.round(D.MEMORY_TIMESERIES.reads.reduce((a,b)=>a+b,0))} reads · {Math.round(D.MEMORY_TIMESERIES.writes.reduce((a,b)=>a+b,0))} writes</span>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--fg-2)' }}>
            <span><span style={{ display: 'inline-block', width: 10, height: 2, background: P.accent, marginRight: 5, verticalAlign: 'middle' }}></span>memory reads</span>
            <span><span style={{ display: 'inline-block', width: 10, height: 2, background: P.ok, marginRight: 5, verticalAlign: 'middle' }}></span>memory writes</span>
          </div>
          <div style={{ flex: 1, minHeight: 200 }}>
            <LineChart
              labels={D.MEMORY_TIMESERIES.labels}
              series={[
                { values: D.MEMORY_TIMESERIES.reads,  color: P.accent },
                { values: D.MEMORY_TIMESERIES.writes, color: P.ok, fill: false },
              ]}
              height={210}
              formatY={v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : Math.round(v)}
            />
          </div>
        </div>
      </div>

      <div className="col-span-4 panel">
        <div className="panel-head">
          <span className="title">Task pipeline</span>
          <span className="meta">{tasksTotal} total</span>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="stack-bar" style={{ height: 18, borderRadius: 4 }}>
            <span style={{ width: `${tasksDone/tasksTotal*100}%`, background: P.ok }} title={`done ${tasksDone}`} />
            <span style={{ width: `${tasksInP/tasksTotal*100}%`, background: P.accent }} title={`in-progress ${tasksInP}`} />
            <span style={{ width: `${tasksBlocked/tasksTotal*100}%`, background: P.bad }} title={`blocked ${tasksBlocked}`} />
            <span style={{ width: `${tasksPending/tasksTotal*100}%`, background: P.warn }} title={`pending ${tasksPending}`} />
          </div>
          {[
            { l: 'done',        v: tasksDone,    c: P.ok },
            { l: 'in-progress', v: tasksInP,     c: P.accent },
            { l: 'blocked',     v: tasksBlocked, c: P.bad },
            { l: 'pending',     v: tasksPending, c: P.warn },
          ].map(r => (
            <div key={r.l} style={{ display: 'grid', gridTemplateColumns: '12px 1fr auto auto', gap: 8, alignItems: 'center', fontSize: 12 }}>
              <span style={{ width: 8, height: 8, background: r.c, borderRadius: 2 }}></span>
              <span style={{ color: 'var(--fg-2)' }}>{r.l}</span>
              <span className="mono" style={{ color: 'var(--fg-0)' }}>{r.v}</span>
              <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10, width: 36, textAlign: 'right' }}>{(r.v/tasksTotal*100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Row 3: lower band — orchestrators table + system health + live feed */}
      <div className="col-span-5 panel">
        <div className="panel-head">
          <span className="title">Orchestrators · current work</span>
          <span className="meta">{orchRunning} running</span>
        </div>
        <div className="panel-body flush">
          <table className="tbl">
            <thead>
              <tr><th>Orch</th><th>Project</th><th>Current task</th><th className="num">Done</th><th className="num">⏱</th></tr>
            </thead>
            <tbody>
              {D.ORCHESTRATORS.map(o => (
                <tr key={o.pid}>
                  <td>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <StatusDot kind={o.running ? 'running' : 'completed'} />
                      <span className="mono" style={{ fontSize: 11 }}>{o.pid}</span>
                    </span>
                  </td>
                  <td className="mono" style={{ color: 'var(--fg-1)' }}>{o.project}</td>
                  <td style={{ color: 'var(--fg-2)', maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {o.current_task}
                  </td>
                  <td className="num"><span className="mono">{o.summary.done}/{o.summary.total}</span></td>
                  <td className="num" style={{ color: 'var(--fg-3)', fontSize: 11 }}>{o.started}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="col-span-3 panel">
        <div className="panel-head">
          <span className="title">System health</span>
          <span className="meta">all ok</span>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[
            { l: 'Graphiti', sub: `${D.MEMORY_STATUS.graphiti.node_count.toLocaleString()} nodes · ${D.MEMORY_STATUS.graphiti.edge_count.toLocaleString()} edges`, ok: true },
            { l: 'Mem0',     sub: `${D.MEMORY_STATUS.mem0.memory_count.toLocaleString()} memories`, ok: true },
            { l: 'Taskmaster', sub: 'mcp v0.18 · responsive', ok: true },
            { l: 'Write queue', sub: `${queue.pending} pending · ${queue.retry} retry · ${queue.dead} dead`, ok: queue.dead === 0, warn: queue.pending > 5 || queue.retry > 0 },
            (() => {
              const v = D.RECON_STATE.verdict;
              const sev = v?.severity || 'none';
              const action = v?.action_taken || 'none';
              return { l: 'Reconciliation', sub: `verdict: ${sev} · ${action}`, ok: sev !== 'serious', warn: sev === 'minor' };
            })(),
          ].map(s => (
            <div key={s.l} style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: 8, alignItems: 'center' }}>
              <span className={`dot ${s.ok ? (s.warn ? 'warn' : 'ok') : 'bad'}`}></span>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{s.l}</div>
                <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>{s.sub}</div>
              </div>
              <span className={`badge ${s.ok ? (s.warn ? 'warn' : 'ok') : 'bad'}`}>{s.ok ? (s.warn ? 'warn' : 'ok') : 'bad'}</span>
            </div>
          ))}
          <div style={{ marginTop: 4, paddingTop: 8, borderTop: '1px solid var(--line)' }}>
            <div style={{ fontSize: 10, color: 'var(--fg-3)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Recon latency · last 40 runs</div>
            <div style={{ height: 30 }}><Sparkline values={reconLatencySpark} color={P.warn} /></div>
          </div>
        </div>
      </div>

      <div className="col-span-4 panel">
        <div className="panel-head">
          <span className="title">Live event stream</span>
          <span className="dot live" style={{ marginLeft: 4 }}></span>
          <span className="meta">{paused ? 'paused' : 'streaming'}</span>
        </div>
        <div className="panel-body flush" style={{ overflow: 'hidden' }}>
          <div style={{ height: '100%', overflow: 'auto' }}>
            <LiveFeed paused={paused} />
          </div>
        </div>
      </div>
    </div>
  );
}

window.DF_OVERVIEW = { OverviewTab };
