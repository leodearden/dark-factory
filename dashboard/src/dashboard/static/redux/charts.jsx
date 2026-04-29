/* SVG chart primitives — no external dep, always render */

const { useRef, useEffect, useState, useMemo } = React;

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
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = width / Math.max(values.length - 1, 1);
  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return [x, y];
  });
  const linePath = points.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
      {area && <path d={areaPath} fill={color} fillOpacity={0.15} />}
      <path d={linePath} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

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
  const maxV = Math.max(...all, 1);
  const minV = 0;
  const range = maxV - minV || 1;
  const n = labels.length;
  const stepX = chartW / Math.max(n - 1, 1);
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minV + (range * i) / ticks);
  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }}>
        {yTicks.map((t, i) => {
          const y = padT + chartH - ((t - minV) / range) * chartH;
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
          const pts = s.values.map((v, i) => {
            const x = padL + i * stepX;
            const y = padT + chartH - ((v - minV) / range) * chartH;
            return [x, y];
          });
          const linePath = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
          const areaPath = `${linePath} L${padL + chartW},${padT + chartH} L${padL},${padT + chartH} Z`;
          return (
            <g key={si}>
              {s.fill !== false && <path d={areaPath} fill={color} fillOpacity={0.10} />}
              <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function StackedAreaChart({ stacks, labels, height = 220, formatY = v => String(v) }) {
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
  const totals = labels.map((_, i) => stacks.reduce((s, st) => s + (st.values[i] || 0), 0));
  const maxV = Math.max(...totals, 1);

  // build cumulative stacks
  const cumLayers = stacks.map((_, li) =>
    labels.map((_, i) => stacks.slice(0, li + 1).reduce((s, st) => s + (st.values[i] || 0), 0))
  );
  const baseLayers = stacks.map((_, li) =>
    labels.map((_, i) => stacks.slice(0, li).reduce((s, st) => s + (st.values[i] || 0), 0))
  );

  const yToPx = v => padT + chartH - (v / maxV) * chartH;

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => (maxV * i) / ticks);

  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height: '100%', display: 'block' }}>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={padL} y1={yToPx(t)} x2={padL + chartW} y2={yToPx(t)} stroke={PALETTE.line} strokeWidth={0.5} strokeDasharray={i === 0 ? '0' : '2 3'} />
            <text x={padL - 6} y={yToPx(t) + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(Math.round(t))}</text>
          </g>
        ))}
        {labels.map((lab, i) => {
          if (n > 8 && i % Math.ceil(n / 6) !== 0 && i !== n - 1) return null;
          const x = padL + i * stepX;
          return <text key={i} x={x} y={height - 6} fontSize="9" fill={PALETTE.fg3} textAnchor="middle" fontFamily="JetBrains Mono">{lab}</text>;
        })}
        {stacks.map((st, li) => {
          const top = cumLayers[li];
          const base = baseLayers[li];
          const points = [];
          for (let i = 0; i < n; i++) points.push([padL + i * stepX, yToPx(top[i])]);
          for (let i = n - 1; i >= 0; i--) points.push([padL + i * stepX, yToPx(base[i])]);
          const d = points.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ') + ' Z';
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

window.DF_CHARTS = { PALETTE, Sparkline, LineChart, StackedAreaChart, BarChart, HBarChart, Donut, StatTile, Heatmap, HistBar };
