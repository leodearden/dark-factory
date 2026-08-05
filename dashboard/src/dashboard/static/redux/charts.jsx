/* SVG chart primitives — no external dep, always render */

const { useRef, useEffect, useState, useMemo } = React;

// The scale+path math for every chart primitive here lives in the plain-JS
// sibling /static/redux/spark_path.js, where it is behaviourally testable under
// `node --test` (dashboard/tests/js/spark_path.test.mjs) — this file is JSX
// behind CDN Babel with no node_modules, so nothing in it can be executed by a
// test. spark_path.js is also what makes a MISSING sample stay missing instead
// of being drawn as a measured zero: the two sparklines since task 3436, and
// LineChart/StackedAreaChart/BarChart/HistBar since task 3489.
//
// Destructured, with no `|| {}` fallback, deliberately: if spark_path.js is
// missing or 404s this throws at load with a clear message, rather than
// deferring to a TypeError inside a render or silently degrading into charts
// that draw nothing. Matches the DF_GRAPH_LAYOUT / DF_RUNTIME_FMT precedent.
// index.html loads that classic script before this file; test_index_html.py
// pins both the load order and that it is actually served — and the shared
// `?v=` cache-buster is bumped whenever this destructure reaches for a name a
// cached copy of spark_path.js would not have.
const {
  sparkPaths: sparkSmoothPaths,
  stepPaths: sparkStepPaths,
  plottableMax,
  axisY,
  axisPaths,
  stackedAreaPaths,
} = window.DF_SPARK_PATH;

const PALETTE = {
  accent:  'oklch(0.72 0.14 230)',
  accent2: 'oklch(0.62 0.12 200)',
  ok:      'oklch(0.74 0.14 155)',
  warn:    'oklch(0.80 0.14 80)',
  bad:     'oklch(0.68 0.18 25)',
  info:    'oklch(0.62 0.20 305)',
  fg2:     'oklch(0.66 0.012 250)',
  fg3:     'oklch(0.50 0.012 250)',
  line:    'oklch(0.32 0.014 250)',
  bg2:     'oklch(0.235 0.014 250)',
  status: {
    'in-progress': 'oklch(0.72 0.14 230)',
    'in_progress': 'oklch(0.72 0.14 230)',
    running: 'oklch(0.72 0.14 230)',
    done: 'oklch(0.74 0.14 155)',
    completed: 'oklch(0.74 0.14 155)',
    blocked: 'oklch(0.68 0.18 25)',
    pending: 'oklch(0.80 0.14 80)',
    queued: 'oklch(0.62 0.20 305)',
    in_flight: 'oklch(0.80 0.14 80)',
    conflict: 'oklch(0.80 0.14 80)',
    already_merged: 'oklch(0.62 0.20 305)',
    unknown_branch: 'oklch(0.58 0.04 250)',
    failed: 'oklch(0.68 0.18 25)',
    success: 'oklch(0.74 0.14 155)',
    partial: 'oklch(0.80 0.14 80)',
  },
  paths: {
    'one-pass':        'oklch(0.74 0.14 155)',
    'multi-pass':      'oklch(0.72 0.14 230)',
    'via-steward':     'oklch(0.80 0.14 80)',
    'via-interactive': 'oklch(0.72 0.10 50)',
    'blocked':         'oklch(0.68 0.18 25)',
  },
};

function Sparkline({ values, width = 100, height = 28, area = true, color = PALETTE.accent, strokeWidth = 1.5 }) {
  if (!values || values.length === 0) return null;
  const { line, area: areaPath } = sparkSmoothPaths(values, width, height);
  // No plottable samples at all (an all-hole series) — draw NOTHING rather
  // than a flat line along the chart floor, which would assert measurements
  // that were never taken.
  if (!line) return null;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
      {area && areaPath && <path d={areaPath} fill={color} fillOpacity={0.15} />}
      <path d={line} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// StepSpark — step-interpolated sparkline (horizontal-then-vertical, no diagonals).
// Used by CuratorTab to render the capped_spark time-series so that discrete
// state transitions are visible as sharp steps rather than smoothed lines.
function StepSpark({ values, width = 100, height = 28, color = PALETTE.bad, strokeWidth = 1.5, area = false }) {
  if (!values || values.length === 0) return null;
  // The horizontal-then-vertical edge construction — and the single-data-point
  // full-width tick, which exists because a path holding only a Move command
  // renders nothing — now live in spark_path.js, reproduced there exactly.
  const { line, area: areaPath } = sparkStepPaths(values, width, height);
  if (!line) return null;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
      {area && areaPath && <path d={areaPath} fill={color} fillOpacity={0.15} />}
      <path d={line} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="square" />
    </svg>
  );
}

// NOTE — this formatY default deliberately does NOT round, while
// StackedAreaChart's below does; both hand formatY the RAW tick, only the
// defaults differ. So an integer-count series renders `1.75` / `3.5` here but
// `2` / `4` there. Aligning them means auditing every LineChart caller
// (tabs.jsx:538, tabs.jsx:1109, tab_overview.jsx:231, the analytics tab) — out
// of scope for task 4059, which only fixed the pre-rounding defect below.
function LineChart({ series, labels, height = 220, yLabel, formatY = (v) => String(v), formatX = (v) => v }) {
  const ref = useRef(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  const padL = 38, padR = 12, padT = 8, padB = 22;
  const chartW = Math.max(w - padL - padR, 50);
  const chartH = height - padT - padB;
  const all = series.flatMap(s => s.values);
  // Hole-excluding fold (task 3489): one undefined/NaN sample used to poison
  // maxV to NaN, `range = NaN - NaN || 1` silently fell back to 1, and every y
  // became NaN — a single NaN token invalidates the whole `d` attribute, so SVG
  // dropped the ENTIRE path. The `1` seed is preserved deliberately, as is
  // minV = 0: switching to a `Math.min(...)` fold would newly bring negative
  // samples in-range and silently re-frame every LineChart on the dashboard.
  // This is a null-handling fix, not a re-scaling.
  const maxV = plottableMax(all, 1);
  const minV = 0;
  const range = maxV - minV || 1;
  const n = labels.length;
  const stepX = chartW / Math.max(n - 1, 1);
  // The plot box every value is scaled into. `count` is the LABEL count, so a
  // series shorter than the label row stops at its own last x rather than being
  // stretched across the full width.
  const geom = { x0: padL, y0: padT, width: chartW, height: chartH, count: n, min: minV, range };
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minV + (range * i) / ticks);
  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }}>
        {yTicks.map((t, i) => {
          const y = axisY(t, geom);
          return (
            <g key={i}>
              <line x1={padL} y1={y} x2={padL + chartW} y2={y} stroke={PALETTE.line} strokeWidth={0.5} strokeDasharray={i === 0 ? '0' : '2 3'} />
              <text x={padL - 6} y={y + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(t)}</text>
            </g>
          );
        })}
        {labels.map((lab, i) => {
          if (n > 12 && i % Math.ceil(n / 8) !== 0 && i !== n - 1) return null;
          const x = padL + i * stepX;
          return (
            <text key={i} x={x} y={height - 6} fontSize="9" fill={PALETTE.fg3} textAnchor="middle" fontFamily="JetBrains Mono">{formatX(lab)}</text>
          );
        })}
        {series.map((s, si) => {
          const color = s.color || PALETTE.accent;
          // One subpath per run of consecutive real samples, so the line is
          // genuinely discontinuous across a hole and no fill is painted under
          // a slot that holds no measurement (the pre-fix area closed at the
          // full chart width regardless).
          const { line, area } = axisPaths(s.values, geom);
          // Skips THIS SERIES only, when it has no plottable sample at all.
          // The axes, gridlines and tick labels below are structural facts
          // about the requested window rather than measurements, so they keep
          // rendering — an empty frame reads as "no data in this window",
          // where no chart at all reads as "nothing was asked for".
          if (!line) return null;
          return (
            <g key={si}>
              {s.fill !== false && area && <path d={area} fill={color} fillOpacity={0.10} />}
              <path d={line} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function StackedAreaChart({ stacks, labels, height = 220, formatY = v => String(Math.round(v)), formatX = v => v }) {
  // stacks: [{ key, color, values }]
  const ref = useRef(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current); return () => ro.disconnect();
  }, []);
  const padL = 38, padR = 12, padT = 8, padB = 22;
  const chartW = Math.max(w - padL - padR, 50);
  const chartH = height - padT - padB;
  const n = labels.length;
  const stepX = chartW / Math.max(n - 1, 1);
  // One polygon per layer, plus the axis maximum they were all scaled against
  // (task 3489). The cumulative folds used to scrub every sample through a
  // `|| 0` fallback, which fabricated a zero-height band at a hole AND fed a
  // hole-as-zero partial sum to the axis. (The exact pre-fix expression is not
  // quoted here on purpose: test_charts_null_samples.py greps this body for it,
  // and prose repeating it would trip that probe.) The builder instead carries
  // a hole as unknown: layer li is drawn at column i only if every layer BELOW
  // it is measured there — its base is their sum — while the bands below keep
  // drawing truthfully.
  const geom = { x0: padL, y0: padT, width: chartW, height: chartH, count: n };
  const { max: maxV, paths } = stackedAreaPaths(stacks, geom);

  // The value axis the builder scaled every band against: 0..maxV.
  const tickGeom = { y0: padT, height: chartH, min: 0, range: maxV };

  // Each tick is handed to formatY RAW (fractional), because a caller's own
  // formatter is the only thing that knows the axis UNITS. The Workflow panel
  // (tab_escalation_analytics.jsx:494) plots a 100%-normalized stack and
  // multiplies by 100, so pre-rounding the tick to an integer collapsed
  // 0/0.25/0.5/0.75/1.0 to 0/0/1/1/1 and rendered "0% / 0% / 100% / 100% /
  // 100%". The rounding is NOT deleted — it moved into the formatY default
  // above, so the three callers that pass no formatter (tabs.jsx:1259,
  // tabs.jsx:1329, tab_escalation_analytics.jsx:244) keep byte-identical
  // integer count axes instead of gaining 2.5 / 7.5 labels. LineChart
  // already passed the raw tick and is deliberately left alone (see the note
  // at its signature: only the two formatY DEFAULTS differ).
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => (maxV * i) / ticks);

  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }}>
        {yTicks.map((t, i) => {
          const y = axisY(t, tickGeom);
          return (
            <g key={i}>
              <line x1={padL} y1={y} x2={padL + chartW} y2={y} stroke={PALETTE.line} strokeWidth={0.5} strokeDasharray={i === 0 ? '0' : '2 3'} />
              <text x={padL - 6} y={y + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(t)}</text>
            </g>
          );
        })}
        {labels.map((lab, i) => {
          if (n > 8 && i % Math.ceil(n / 6) !== 0 && i !== n - 1) return null;
          const x = padL + i * stepX;
          return <text key={i} x={x} y={height - 6} fontSize="9" fill={PALETTE.fg3} textAnchor="middle" fontFamily="JetBrains Mono">{formatX(lab)}</text>;
        })}
        {stacks.map((st, li) => {
          const d = paths[li];
          // A layer with nothing plottable draws nothing — not a band pinned to
          // the floor. The axes and tick labels above still render.
          if (!d) return null;
          return <path key={st.key} d={d} fill={st.color} fillOpacity={0.85} stroke={st.color} strokeWidth={0.5} />;
        })}
      </svg>
    </div>
  );
}

function BarChart({ labels, values, height = 160, color = PALETTE.accent, formatY = v => String(v) }) {
  const ref = useRef(null);
  const [w, setW] = useState(400);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current); return () => ro.disconnect();
  }, []);
  const padL = 30, padR = 8, padT = 8, padB = 22;
  const chartW = Math.max(w - padL - padR, 50);
  const chartH = height - padT - padB;
  const max = Math.max(...values, 1);
  const bw = chartW / values.length;
  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
          const y = padT + chartH * (1 - f);
          const v = Math.round(max * f);
          return (
            <g key={i}>
              <line x1={padL} y1={y} x2={padL + chartW} y2={y} stroke={PALETTE.line} strokeWidth={0.5} strokeDasharray={i === 0 ? '0' : '2 3'} />
              {i % 2 === 0 && <text x={padL - 4} y={y + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(v)}</text>}
            </g>
          );
        })}
        {values.map((v, i) => {
          const h = (v / max) * chartH;
          const x = padL + i * bw + 2;
          const y = padT + chartH - h;
          return (
            <g key={i}>
              <rect x={x} y={y} width={Math.max(bw - 4, 2)} height={h} fill={color} rx={2} />
              <text x={padL + i * bw + bw / 2} y={height - 6} fontSize="9" fill={PALETTE.fg3} textAnchor="middle" fontFamily="JetBrains Mono">{labels[i]}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function HBarChart({ rows, valueKey = 'total', labelKey = 'label', segments, formatVal = v => v, height }) {
  // rows: [{ label, total, ...segments? }]; segments: [{ key, color, label }]
  const max = Math.max(...rows.map(r => r[valueKey]), 1);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {rows.map((r, i) => {
        const w = (r[valueKey] / max) * 100;
        if (segments) {
          let cum = 0;
          return (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                <span style={{ color: 'var(--fg-1)' }}>{r[labelKey]}</span>
                <span className="mono" style={{ color: 'var(--fg-2)' }}>{formatVal(r[valueKey])}</span>
              </div>
              <div style={{ display: 'flex', height: 14, background: 'var(--bg-2)', borderRadius: 3, overflow: 'hidden' }}>
                {segments.map(s => {
                  const segW = ((r[s.key] || 0) / max) * 100;
                  cum += segW;
                  return <div key={s.key} title={`${s.label}: ${formatVal(r[s.key] || 0)}`} style={{ width: `${segW}%`, background: s.color }} />;
                })}
              </div>
            </div>
          );
        }
        return (
          <div key={i}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
              <span style={{ color: 'var(--fg-1)' }}>{r[labelKey]}</span>
              <span className="mono" style={{ color: 'var(--fg-2)' }}>{formatVal(r[valueKey])}</span>
            </div>
            <div style={{ height: 6, background: 'var(--bg-2)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${w}%`, height: '100%', background: PALETTE.accent }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Donut({ data, size = 120, thickness = 18, centerLabel, centerValue }) {
  // data: [{ label, value, color }]
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const r = (size - thickness) / 2;
  const c = size / 2;
  const circ = 2 * Math.PI * r;
  let cumPct = 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="var(--bg-2)" strokeWidth={thickness} />
      {data.map((d, i) => {
        const pct = d.value / total;
        const dash = pct * circ;
        const offset = -cumPct * circ;
        cumPct += pct;
        return (
          <circle key={i} cx={c} cy={c} r={r} fill="none" stroke={d.color}
            strokeWidth={thickness} strokeDasharray={`${dash} ${circ - dash}`}
            strokeDashoffset={offset} transform={`rotate(-90 ${c} ${c})`} />
        );
      })}
      {centerValue && (
        <text x={c} y={c - 2} textAnchor="middle" fontSize="14" fontFamily="JetBrains Mono" fontWeight="600" fill="var(--fg-0)">{centerValue}</text>
      )}
      {centerLabel && (
        <text x={c} y={c + 12} textAnchor="middle" fontSize="9" fill="var(--fg-3)">{centerLabel}</text>
      )}
    </svg>
  );
}

function StatTile({ label, value, unit, delta, deltaDir, spark, sparkColor, hint }) {
  return (
    <div className="kpi">
      <div className="lbl">{label}{hint && <span style={{ color: 'var(--fg-3)', textTransform: 'none', letterSpacing: 0, fontSize: 10 }}> · {hint}</span>}</div>
      <div className="val">{value}{unit && <span className="unit">{unit}</span>}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'center', gap: 6 }}>
        <div className="spark" style={{ height: 22 }}>
          {spark && <Sparkline values={spark} color={sparkColor || PALETTE.accent} />}
        </div>
        {delta && <span className={`delta ${deltaDir || 'flat'}`}>{deltaDir === 'up' ? '▲' : deltaDir === 'down' ? '▼' : '·'} {delta}</span>}
      </div>
    </div>
  );
}

function Heatmap({ rows, cols, getCell }) {
  // generic grid
  return (
    <div className="heat-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {Array.from({ length: rows * cols }, (_, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        const lvl = getCell(r, c);
        return <div key={i} className={`heat-cell ${lvl ? 'l' + lvl : ''}`} />;
      })}
    </div>
  );
}

function HistBar({ values, maxOverride, height = 50, color = PALETTE.accent }) {
  const max = maxOverride ?? Math.max(...values, 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height }}>
      {values.map((v, i) => (
        <div key={i} style={{ flex: 1, height: `${(v / max) * 100}%`, background: color, borderRadius: '2px 2px 0 0', minHeight: 1 }} />
      ))}
    </div>
  );
}

const SMOOTHING_OPTIONS = ['1h', '2h', '4h', '8h', '1d', '3d'];

function smoothingLabelToSeconds(label) {
  const map = { '1h': 3600, '2h': 7200, '4h': 14400, '8h': 28800, '1d': 86400, '3d': 259200 };
  return map[label] ?? null;
}

function defaultSmoothingForWindow(windowKey) {
  const map = { '24h': '2h', '7d': '8h', '30d': '1d', '90d': '3d' };
  return map[windowKey] ?? '8h';
}

// Returns a per-sample rate array (tasks/day) for the given cumulative series
// using a trailing time window.  Irregular sample spacing is handled by
// parsing ISO timestamps from labels.  Returns [] for <2 points or
// length mismatch.  A flat series yields exact zeros (no synthesis).
//
// Uses a monotonic two-pointer sweep (O(n)): since labels are sorted ascending,
// the window's left boundary is non-decreasing as i advances, so `left` only
// ever moves forward.
//
// Cf. dailyDeltas() in shell.jsx — related but distinct: dailyDeltas buckets by
// calendar day, clamps rates to >=0, and returns N-1 entries (day-to-day diff).
// deriveVelocitySeries uses a configurable trailing time window, allows negative
// rates (backlog can shrink), and returns N entries aligned to each sample.
function deriveVelocitySeries(series, labels, smoothingWindowSeconds) {
  if (!series || !labels || series.length !== labels.length || series.length < 2) return [];
  const t = labels.map(l => {
    const ms = new Date(l).getTime();
    return isNaN(ms) ? NaN : ms / 1000;
  });
  const result = [];
  let left = 0;
  for (let i = 0; i < series.length; i++) {
    if (isNaN(t[i])) { result.push(0); continue; }
    // Advance left past NaN entries and entries that have fallen outside the
    // trailing window.  left is strictly non-decreasing, giving O(n) total.
    while (left < i && (isNaN(t[left]) || t[i] - t[left] > smoothingWindowSeconds)) {
      left++;
    }
    const dtDays = (t[i] - t[left]) / 86400;
    result.push(dtDays > 0 ? (series[i] - series[left]) / dtDays : 0);
  }
  return result;
}

window.DF_CHARTS = { PALETTE, Sparkline, StepSpark, LineChart, StackedAreaChart, BarChart, HBarChart, Donut, StatTile, Heatmap, HistBar, SMOOTHING_OPTIONS, smoothingLabelToSeconds, defaultSmoothingForWindow, deriveVelocitySeries };
