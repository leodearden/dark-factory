/* App shell: rail nav, topbar, toolbar (filters), live feed */
const { useState, useEffect, useMemo, useRef } = React;
const SP_SHELL = window.DF_CHARTS.Sparkline;
const SHELL_PROJECTS = window.DF_DATA.PROJECTS;
const SHELL_AGENTS = window.DF_DATA.AGENTS;

// Strip the "T-" prefix from a task identifier so the bare numeric portion shows.
// Accepts plain ids ("T-19") and project-scoped uids ("dark_factory/T-19").
function taskId(id) {
  return id == null ? '' : String(id).replace(/T-/g, '');
}

// Convert a series of cumulative-state snapshots into per-day deltas.
// `labels` are ISO timestamps (ascending); `values` are the cumulative count
// at each timestamp. Buckets by date prefix and returns the day-to-day diff,
// floored at 0. Length is (#distinct days - 1); empty for sparse history.
function dailyDeltas(labels, values) {
  if (!labels || !values || labels.length === 0 || values.length === 0) return [];
  const byDay = {};
  const order = [];
  const n = Math.min(labels.length, values.length);
  for (let i = 0; i < n; i++) {
    const lbl = labels[i];
    const day = (typeof lbl === 'string' && lbl.length >= 10) ? lbl.slice(0, 10) : null;
    if (!day) continue;
    if (!(day in byDay)) order.push(day);
    byDay[day] = values[i];
  }
  order.sort();
  const out = [];
  for (let i = 1; i < order.length; i++) {
    out.push(Math.max(0, (byDay[order[i]] || 0) - (byDay[order[i - 1]] || 0)));
  }
  return out;
}

// Format a UTC ISO8601 timestamp as a relative string ("now", "12s", "4m", "2h", "1d").
// Returns "—" for null/undefined/unparseable input.
function timeago(iso) {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (isNaN(t)) return '—';
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 5) return 'now';
  if (sec < 60) return `${sec}s`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h`;
  const d = Math.round(hr / 24);
  return `${d}d`;
}

// Replace any ISO8601 timestamps embedded inside a free-form string with the
// minute-precision UTC form. Used for fields like cost-event `detail` where
// the API returns a stringified JSON blob containing raw timestamps.
function scrubIsos(s) {
  if (typeof s !== 'string' || !s) return s;
  return s.replace(
    /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?/g,
    m => fmtDateTime(m),
  );
}

// Format an ISO8601 timestamp as "YYYY-MM-DD HH:MM" in UTC — minute precision,
// space instead of T, no seconds/subseconds/timezone suffix. Date-only inputs
// ("YYYY-MM-DD") pass through unchanged. Returns "—" on null/undefined/unparseable.
//
// ISO strings without an explicit timezone (e.g. "2026-04-26T12:00") are
// interpreted as UTC — Python serialisers in this app uniformly emit UTC, but
// ECMAScript Date.parse would default unzoned values to local time.
function fmtDateTime(iso) {
  if (iso == null || iso === '') return '—';
  if (typeof iso !== 'string') return String(iso);
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso;
  const tzPart = iso.split('T')[1] || '';
  const naive = iso.includes('T') && !/[Zz]|[+-]\d{2}:?\d{2}$/.test(tzPart);
  const t = Date.parse(naive ? iso + 'Z' : iso);
  if (isNaN(t)) return iso;
  const d = new Date(t);
  const Y = d.getUTCFullYear();
  const M = String(d.getUTCMonth() + 1).padStart(2, '0');
  const D = String(d.getUTCDate()).padStart(2, '0');
  const h = String(d.getUTCHours()).padStart(2, '0');
  const m = String(d.getUTCMinutes()).padStart(2, '0');
  return `${Y}-${M}-${D} ${h}:${m}`;
}

// ── Glyphs (simple shapes only — no complex SVG) ──
const Glyph = ({ kind }) => {
  const props = { width: 14, height: 14, viewBox: '0 0 14 14', fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinecap: 'round', strokeLinejoin: 'round' };
  switch (kind) {
    case 'overview': return <svg {...props}><rect x="1.5" y="1.5" width="4.5" height="4.5" rx="1"/><rect x="8" y="1.5" width="4.5" height="4.5" rx="1"/><rect x="1.5" y="8" width="4.5" height="4.5" rx="1"/><rect x="8" y="8" width="4.5" height="4.5" rx="1"/></svg>;
    case 'orch':     return <svg {...props}><circle cx="7" cy="3" r="1.5"/><circle cx="3" cy="11" r="1.5"/><circle cx="11" cy="11" r="1.5"/><path d="M7 4.5 L3.6 9.7 M7 4.5 L10.4 9.7"/></svg>;
    case 'tasks':    return <svg {...props}><circle cx="3" cy="3" r="1"/><circle cx="3" cy="7" r="1"/><circle cx="3" cy="11" r="1"/><path d="M5 3h7 M5 7h6 M5 11h4"/></svg>;
    case 'perf':     return <svg {...props}><path d="M2 11 L5 7 L8 9 L12 3"/><path d="M2 12.5 L12 12.5"/></svg>;
    case 'memory':   return <svg {...props}><rect x="1.5" y="3" width="11" height="8" rx="1"/><path d="M4 3v8 M7 3v8 M10 3v8"/></svg>;
    case 'recon':    return <svg {...props}><path d="M2 7 a5 5 0 0 1 9-3"/><path d="M12 7 a5 5 0 0 1 -9 3"/><path d="M11 2v2.5h-2.5 M3 12v-2.5h2.5"/></svg>;
    case 'merge':    return <svg {...props}><circle cx="3" cy="3" r="1.5"/><circle cx="3" cy="11" r="1.5"/><circle cx="11" cy="7" r="1.5"/><path d="M3 4.5v5 M3 4.5 a 5 2.5 0 0 0 7.5 2.5 M3 9.5 a 5 2.5 0 0 1 7.5 -2.5"/></svg>;
    case 'cost':     return <svg {...props}><circle cx="7" cy="7" r="5"/><path d="M7 4v6 M5.5 5.5h2.5 a1.2 1.2 0 0 1 0 2.5 h-2 a1.2 1.2 0 0 0 0 2.5 h3"/></svg>;
    case 'burn':     return <svg {...props}><path d="M2 11 L5 8 L8 10 L12 4"/><path d="M2 12.5 L12 12.5"/><circle cx="12" cy="4" r="0.6" fill="currentColor"/></svg>;
    case 'search':   return <svg {...props}><circle cx="6" cy="6" r="3.5"/><path d="M8.5 8.5 L12 12"/></svg>;
    case 'filter':   return <svg {...props}><path d="M2 3h10 M3.5 7h7 M5 11h4"/></svg>;
    case 'chev':     return <svg {...props}><path d="M4 5l3 3 3-3"/></svg>;
    case 'live':     return <svg {...props}><circle cx="7" cy="7" r="2" fill="currentColor"/><circle cx="7" cy="7" r="5"/></svg>;
    case 'pause':    return <svg {...props}><rect x="3" y="3" width="3" height="8"/><rect x="8" y="3" width="3" height="8"/></svg>;
    default: return null;
  }
};

// ── Status pill in top bar ──
function StatStrip({ live, lastUpdate, summary }) {
  return (
    <div className="stat-strip">
      <span className="stat-pill">
        <span className={`dot ${live ? 'live' : 'warn'}`}></span>
        <span className="lbl">{live ? 'live' : 'paused'}</span>
        <span className="val">{lastUpdate}</span>
      </span>
      <span className="stat-pill">
        <span className="lbl">orch</span>
        <span className="val" style={{ color: 'var(--accent)' }}>{summary.orchRunning}</span>
        <span style={{ color: 'var(--fg-3)' }}>/</span>
        <span className="val">{summary.orchTotal}</span>
      </span>
      <span className="stat-pill">
        <span className="lbl">tasks active</span>
        <span className="val">{summary.tasksActive}</span>
      </span>
      <span className="stat-pill">
        <span className="lbl">queue</span>
        <span className="val">{summary.queue}</span>
      </span>
      <span className="stat-pill">
        <span className="lbl">spend 24h</span>
        <span className="val">${summary.spend24h.toFixed(2)}</span>
      </span>
    </div>
  );
}

// ── Filter chips ──
function ChipGroup({ options, value, onChange }) {
  return (
    <div className="chips">
      {options.map(o => (
        <button key={o} className={value === o ? 'on' : ''} onClick={() => onChange(o)}>{o}</button>
      ))}
    </div>
  );
}

function MultiSelect({ label, options, selected, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    function onClick(e) { if (ref.current && !ref.current.contains(e.target)) setOpen(false); }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);
  const allSelected = selected.length === 0 || selected.length === options.length;
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <div className="multi" onClick={() => setOpen(o => !o)}>
        <span className="lbl">{label}</span>
        <span style={{ color: 'var(--fg-1)' }}>
          {allSelected ? 'all' : selected.length === 1 ? selected[0] : `${selected.length} selected`}
        </span>
        <span className="cnt">{allSelected ? options.length : selected.length}</span>
        <Glyph kind="chev" />
      </div>
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 4px)', left: 0, zIndex: 50,
          background: 'var(--bg-2)', border: '1px solid var(--line)', borderRadius: 5,
          padding: 6, minWidth: 200, boxShadow: '0 12px 30px rgba(0,0,0,0.5)',
        }}>
          <button
            onClick={() => onChange(allSelected ? [options[0]] : [])}
            style={{ display: 'block', width: '100%', textAlign: 'left', padding: '4px 8px', fontSize: 11, color: 'var(--accent)', borderRadius: 3 }}>
            {allSelected ? 'select none' : 'select all'}
          </button>
          <div style={{ height: 1, background: 'var(--line)', margin: '4px 0' }} />
          {options.map(o => {
            const on = allSelected || selected.includes(o);
            return (
              <label key={o} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', fontSize: 12, cursor: 'pointer', color: on ? 'var(--fg-0)' : 'var(--fg-2)', borderRadius: 3 }}
                     onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-3)'}
                     onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <input type="checkbox" checked={on} onChange={() => {
                  let next;
                  if (allSelected) next = options.filter(x => x !== o);
                  else if (selected.includes(o)) next = selected.filter(x => x !== o);
                  else next = [...selected, o];
                  onChange(next);
                }} style={{ accentColor: 'var(--accent)' }} />
                <span className="mono" style={{ fontSize: 11 }}>{o}</span>
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Toolbar (per-tab filter row) ──
function Toolbar({
  window: win, onWindow, windows = ['1h', '24h', '7d', '30d', '90d', 'all'],
  projects, onProjects,
  agents, onAgents, showAgents = false,
  search, onSearch, searchPlaceholder = 'Search…',
  extra,
}) {
  return (
    <div className="toolbar">
      <span className="lbl">Window</span>
      <ChipGroup options={windows} value={win} onChange={onWindow} />
      <MultiSelect label="project" options={SHELL_PROJECTS.map(p => p.name)} selected={projects} onChange={onProjects} />
      {showAgents && <MultiSelect label="agent" options={SHELL_AGENTS} selected={agents} onChange={onAgents} />}
      {onSearch && (
        <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
          <span style={{ position: 'absolute', left: 8, color: 'var(--fg-3)', pointerEvents: 'none', display: 'inline-flex' }}>
            <Glyph kind="search" />
          </span>
          <input className="search-input" style={{ paddingLeft: 28 }}
                 placeholder={searchPlaceholder} value={search || ''}
                 onChange={e => onSearch(e.target.value)} />
        </span>
      )}
      <span className="grow"></span>
      {extra}
    </div>
  );
}

// ── Live feed (pinned bottom of overview) ──
//
// Builds a unified stream from real events already present in DF_DATA:
//   - reconciliation runs (RECON_STATE.runs) — id, status, project, events
//   - merge queue activity (MERGE_QUEUE[pid].recent + .active) — task, outcome
//   - cost / account events (COSTS.events) — account, event_type, detail
// Each source is normalised to {ts, src, msg, key}, sorted by timestamp,
// trimmed to the most recent 60.
function buildFeedEntries(D) {
  const rows = [];

  for (const r of (D.RECON_STATE.runs || [])) {
    if (!r.started_at) continue;
    const dur = r.duration_seconds != null ? `${r.duration_seconds.toFixed(1)}s` : '—';
    rows.push({
      key: `recon:${r.id}`,
      iso: r.started_at,
      src: 'recon',
      msg: `run ${r.id} ${r.status} · ${r.events_processed ?? 0} events · ${dur}`,
    });
  }

  for (const [pid, mq] of Object.entries(D.MERGE_QUEUE || {})) {
    for (const m of (mq.recent || [])) {
      if (!m.timestamp) continue;
      const dur = m.duration_ms != null ? `${(m.duration_ms / 1000).toFixed(1)}s` : '—';
      rows.push({
        key: `merge:${pid}:${m.task_id}:${m.timestamp}`,
        iso: m.timestamp,
        src: 'merge',
        msg: `${pid} · ${m.task_id} · ${m.outcome} · ${dur}`,
      });
    }
    for (const a of (mq.active || [])) {
      if (!a.timestamp) continue;
      rows.push({
        key: `mq-active:${pid}:${a.task_id}:${a.timestamp}`,
        iso: a.timestamp,
        src: 'queue',
        msg: `${pid} · ${a.task_id} ${a.state} · ${a.branch || ''}`.trim(),
      });
    }
  }

  for (const e of (D.COSTS.events || [])) {
    if (!e.ts) continue;
    rows.push({
      key: `cost:${e.account}:${e.ts}`,
      iso: e.ts,
      src: 'cost',
      msg: `${e.account} ${e.event}${e.detail ? ' · ' + scrubIsos(e.detail) : ''}`,
    });
  }

  rows.sort((a, b) => (a.iso < b.iso ? 1 : a.iso > b.iso ? -1 : 0));
  return rows.slice(0, 60).map(r => ({ ...r, ts: timeago(r.iso) }));
}

function LiveFeed({ paused }) {
  const [, tick] = useState(0);
  // Re-tick every 2s so the relative timeago strings refresh even when
  // DF_DATA hasn't changed.  Suspends while paused.
  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => tick(n => n + 1), 2000);
    return () => clearInterval(t);
  }, [paused]);

  const entries = buildFeedEntries(window.DF_DATA);
  if (entries.length === 0) {
    return (
      <div className="feed">
        <div className="row"><span className="msg" style={{ color: 'var(--fg-3)' }}>no recent events</span></div>
      </div>
    );
  }
  return (
    <div className="feed">
      {entries.map(e => (
        <div key={e.key} className="row">
          <span className="ts">{e.ts}</span>
          <span className="src">{e.src}</span>
          <span className="msg">{e.msg}</span>
        </div>
      ))}
    </div>
  );
}

// ── Rail ──
function Rail({ active, onSelect, counts }) {
  const items = [
    { id: 'overview',  label: 'Overview',     glyph: 'overview', count: '' },
    { id: 'orch',      label: 'Orchestrators', glyph: 'orch',     count: counts.orch },
    { id: 'tasks',     label: 'Tasks',        glyph: 'tasks',    count: counts.tasks },
    { id: 'perf',      label: 'Performance',  glyph: 'perf',     count: '' },
    { id: 'memory',    label: 'Memory',       glyph: 'memory',   count: '' },
    { id: 'recon',     label: 'Reconciliation', glyph: 'recon',  count: counts.recon },
    { id: 'merge',     label: 'Merge Queue',  glyph: 'merge',    count: counts.merge },
    { id: 'cost',      label: 'Costs',        glyph: 'cost',     count: '' },
    { id: 'burn',      label: 'Burndown',     glyph: 'burn',     count: '' },
  ];
  return (
    <div className="rail">
      <div className="rail-brand">
        <div className="mark"></div>
        <div>
          <div className="name">DARK FACTORY</div>
        </div>
        <span className="env">v1.7 · prod</span>
      </div>
      <div className="rail-section">SECTIONS</div>
      <div className="rail-nav">
        {items.map(it => (
          <button key={it.id} className={active === it.id ? 'active' : ''} onClick={() => onSelect(it.id)}>
            <span className="glyph"><Glyph kind={it.glyph} /></span>
            <span>{it.label}</span>
            {it.count !== '' && <span className="count">{it.count}</span>}
          </button>
        ))}
      </div>
      <div className="rail-foot">
        <div className="row"><span>Graphiti</span><span className="mono" style={{ color: 'var(--ok)' }}>● online</span></div>
        <div className="row"><span>Mem0</span><span className="mono" style={{ color: 'var(--ok)' }}>● online</span></div>
        <div className="row"><span>Taskmaster</span><span className="mono" style={{ color: 'var(--ok)' }}>● online</span></div>
        <div className="row" style={{ paddingTop: 6, borderTop: '1px solid var(--line)' }}>
          <span>Uptime</span>
          <span className="mono">3d 14h 22m</span>
        </div>
      </div>
    </div>
  );
}

// ── Project group (furl/unfurl) ──
function ProjectGroup({ id, label, open, onToggle, summary, summaryRight, children }) {
  return (
    <div className="panel" style={{ overflow: 'hidden' }}>
      <div className="proj-head" data-open={open ? 'true' : 'false'} onClick={onToggle}
           style={{ padding: '10px 14px', borderBottom: open ? '1px solid var(--line)' : '1px solid transparent' }}>
        <span className="twirl">
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M3.5 2L6.5 5L3.5 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </span>
        <span className="name">{label}</span>
        {summary && <span className="summary">{summary}</span>}
        {summaryRight}
      </div>
      {open && <div style={{ padding: '12px 14px 14px' }}>{children}</div>}
    </div>
  );
}

// ── Segmented control ──
function Segmented({ options, value, onChange }) {
  return (
    <div className="seg">
      {options.map(o => {
        const v = typeof o === 'string' ? o : o.value;
        const lbl = typeof o === 'string' ? o : o.label;
        return (
          <button key={v} className={value === v ? 'on' : ''} onClick={() => onChange(v)}>{lbl}</button>
        );
      })}
    </div>
  );
}

window.DF_SHELL = { Glyph, StatStrip, ChipGroup, MultiSelect, Toolbar, LiveFeed, Rail, ProjectGroup, Segmented, timeago, fmtDateTime, scrubIsos, taskId, dailyDeltas };
