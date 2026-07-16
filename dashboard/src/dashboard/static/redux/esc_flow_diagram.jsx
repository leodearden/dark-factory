/* esc_flow_diagram.jsx — lifecycle flow diagram (hand-rolled mini-Sankey) for
 * the escalation-analytics Workflow panel.
 *
 * No JS test runner for .jsx in this project (see scheduler_drawer.jsx /
 * tab_escalation_analytics.jsx comments) — all aggregation/geometry
 * correctness lives in esc_flow_layout.js's node:test suite
 * (dashboard/tests/js/esc_flow_layout.test.mjs). This file is a THIN
 * React/SVG render shell with no aggregation/geometry logic of its own,
 * verified only by the Python source-assertion tests in
 * dashboard/tests/test_esc_flow_diagram.py.
 *
 * Load order: esc_flow_layout.js (classic script) -> charts.jsx ->
 * esc_flow_diagram.jsx -> tab_escalation_analytics.jsx (see index.html) —
 * both `window.DF_ESC_FLOW_LAYOUT` and `window.DF_CHARTS` must exist before
 * this file's top-level destructure runs.
 *
 * Export: sets window.DF_ESC_FLOW to an object holding LifecycleFlowDiagram,
 * consumed by tab_escalation_analytics.jsx's WorkflowPanel via a top-level
 * destructure — mirrors the scheduler_heatmap.jsx -> window.DF_SCHED_HEATMAP
 * precedent. See the single assignment at the bottom of this file.
 */
const L = window.DF_ESC_FLOW_LAYOUT;
const C = window.DF_CHARTS;

const _COLUMN_LABELS = ['Origin', 'Level', 'Tier', 'Class'];
const _HEADER_H = 14;

// Four columns -> four palette colors, reused for both a column's node rects
// and the ribbons LEAVING that column (a ribbon's color follows its source
// column, same idiom as charts.jsx's StackedAreaChart band coloring).
function columnColors() {
  return [C.PALETTE.accent, C.PALETTE.accent2, C.PALETTE.info, C.PALETTE.ok];
}

// Props:
//   flowDaily  list[dict]  — already-windowed workflow.flow_daily rows
//                            ({date, source, level, tier, class, n}); the
//                            caller (WorkflowPanel) owns window slicing —
//                            this component just sums whatever it's given.
function LifecycleFlowDiagram({ flowDaily, width = 640, height = 260 }) {
  const [hoveredKey, setHoveredKey] = React.useState(null);

  if (!flowDaily || flowDaily.length === 0) {
    return (
      <div style={{ fontSize: 11, color: 'var(--fg-3)', padding: '8px 0' }}>
        No lifecycle flow in this window.
      </div>
    );
  }

  const chartHeight = height - _HEADER_H;
  const model = L.aggregateFlow(flowDaily);
  const geo = L.layoutFlow(model, { width, height: chartHeight });
  const colors = columnColors();
  const colorForCol = col => colors[col % colors.length];
  const ribbonKey = rb => `${rb.col}:${rb.from}->${rb.to}`;

  return (
    <svg viewBox={`0 0 ${geo.width} ${height}`} style={{ width: '100%', height, display: 'block' }}>
      {_COLUMN_LABELS.map((label, i) => (
        <text
          key={label}
          x={geo.width * (i / (_COLUMN_LABELS.length - 1))}
          y={10}
          fontSize="9"
          fill={C.PALETTE.fg3}
          textAnchor={i === 0 ? 'start' : i === _COLUMN_LABELS.length - 1 ? 'end' : 'middle'}
        >
          {label}
        </text>
      ))}
      <g transform={`translate(0, ${_HEADER_H})`}>
        {geo.ribbons.map(rb => {
          const key = ribbonKey(rb);
          const isHovered = hoveredKey === key;
          return (
            <path
              key={key}
              d={rb.d}
              fill={colorForCol(rb.col)}
              fillOpacity={isHovered ? 0.8 : 0.28}
              stroke={isHovered ? colorForCol(rb.col) : 'none'}
              strokeWidth={isHovered ? 0.5 : 0}
              onMouseEnter={() => setHoveredKey(key)}
              onMouseLeave={() => setHoveredKey(null)}
            >
              <title>{`${rb.from} → ${rb.to}: ${rb.count}`}</title>
            </path>
          );
        })}
        {geo.nodes.map(n => {
          // Nodes span the full [0,width] range with no outer margin (see
          // layoutFlow), so a label placed to the right of the LAST column's
          // nodes would render past `width`. Every other column's label sits
          // to its right (in the gap before the next column); only the last
          // column's sits to its left (in the gap before it) instead.
          const isLastCol = n.col === _COLUMN_LABELS.length - 1;
          return (
            <g key={`${n.col}:${n.id}`}>
              <rect
                x={n.x}
                y={n.y}
                width={n.w}
                height={Math.max(n.h, 0.5)}
                fill={colorForCol(n.col)}
                fillOpacity={0.9}
              >
                <title>{`${n.label}: ${n.count}`}</title>
              </rect>
              <text
                x={isLastCol ? n.x - 4 : n.x + n.w + 4}
                y={n.y + n.h / 2}
                dy="0.32em"
                fontSize="9"
                fill={C.PALETTE.fg2}
                textAnchor={isLastCol ? 'end' : 'start'}
              >
                {`${n.label} (${n.count})`}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

window.DF_ESC_FLOW = { LifecycleFlowDiagram };
