"""Structural contract: charts.jsx's chart primitives delegate to spark_path.js.

WHY THIS FILE IS SOURCE-ASSERTION AND NOT BEHAVIOURAL — charts.jsx is JSX
transformed by CDN Babel at runtime, and this repo has no node_modules, so its
component bodies cannot be parsed by node or rendered by React in any harness
here.  The BEHAVIOURAL coverage for the scale/path math lives in
``dashboard/tests/js/spark_path.test.mjs``, which executes the real module
under ``node --test``.  This file exists solely to prove the components are
actually WIRED to it — that the defective arithmetic did not quietly survive
alongside the delegation.  Same precedent as
``test_tab_curator.py::test_charts_jsx_exports_step_spark`` and the body-slicing
assertions in ``test_tab_escalation_analytics.py``.

THE DEFECT (task 3436) — ``Sparkline`` and ``StepSpark`` each did their own
scale/path arithmetic inline, with no null handling at all::

    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const y = height - ((v - min) / range) * height;

A ``null`` coerces to 0 in that ``y`` expression, so a MISSING sample was drawn
as a real point at the value-0 baseline, joined by line segments to both
neighbours and indistinguishable from a measured regression to zero.  An
``undefined``/``NaN`` poisoned both extrema to NaN and blanked the whole chart.
An all-hole series drew a fully synthetic flat line along the chart floor.

THE SAME DEFECT IN THE PADDED CHARTS (task 3489) — ``LineChart``,
``StackedAreaChart``, ``BarChart`` and ``HistBar`` carried the identical defect
class with the same failure modes plus two of their own: StackedAreaChart's
``(st.values[i] || 0)`` conflated a measured 0 with a missing sample AND fed a
hole-as-zero partial sum to the axis maximum, and HistBar's ``minHeight: 1``
rendered a hole as a visible 1px stub.  They now delegate to the padded
primitives in the same module.

WHY THE PROBES ARE PER COMPONENT — the components' folds differ textually
(LineChart's poisoning fold is ``Math.max(...all``, BarChart's is
``Math.max(...values``, StackedAreaChart's is ``Math.max(...totals``), so a
single module-level probe tuple could not express what each component must no
longer contain.  Each entry in ``_CONTRACTS`` therefore carries its own probe
set, its own required builders, and — via ``_PRE_FIX_BODIES`` — its own
verbatim pre-fix body for the negative control at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest
from starlette.testclient import TestClient


@dataclass(frozen=True)
class _Contract:
    """What one charts.jsx component must delegate to, and must no longer contain.

    ``builders`` are the spark_path.js functions (under the names charts.jsx
    binds them by) that must appear in the module-top-level
    ``window.DF_SPARK_PATH`` destructure AND be called in the component's body.
    ``banned`` is that component's own defective-arithmetic probe set: each
    entry is ``(probe substring, what its presence would mean)``.
    """

    builders: tuple[str, ...]
    banned: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# The defective-arithmetic probes, per component.  The negative control at the
# bottom runs each set over that component's verbatim pre-fix body — without
# it, a stale probe (a reformat that inserts a space, a variable rename) would
# make every absence assertion here a permanent false GREEN.
# ---------------------------------------------------------------------------

# Shared by Sparkline and StepSpark: both scaled the RAW series into a bare
# 0..height viewport with the same four expressions (task 3436).
_SPARKLINE_BANNED = (
    (
        'Math.max(...values',
        'folds the extrema over the RAW series, so an undefined/NaN hole '
        'poisons max to NaN and (with `range = NaN - NaN || 1` silently '
        'falling back to 1) blanks the entire chart',
    ),
    (
        'Math.min(...values',
        'folds the extrema over the RAW series — same NaN-poisoning as the '
        'max above',
    ),
    (
        '(v - min) / range',
        'scales each RAW value directly, so a `null` coerces to 0 and a '
        'MISSING sample is plotted as a real point at the value-0 baseline',
    ),
    (
        'L0,${height} Z',
        'closes the filled area across the FULL width regardless of holes, '
        'painting fill under slots that hold no measurement',
    ),
)

_LINE_CHART_BANNED = (
    (
        'Math.max(...all',
        'folds the axis maximum over the RAW flattened series, so one '
        'undefined/NaN sample poisons maxV to NaN, `range = NaN - NaN || 1` '
        'silently falls back to 1, and every y becomes NaN — a single NaN '
        'token invalidates the whole `d` attribute, so SVG drops the ENTIRE '
        'path and the series vanishes',
    ),
    (
        '(v - minV) / range',
        'scales each RAW sample directly, so a `null` coerces to 0 and a '
        'MISSING sample is plotted as a real point at the chart floor, joined '
        'by line segments to both neighbours',
    ),
    (
        'L${padL},${padT + chartH} Z',
        'closes the filled area at the FULL chart width regardless of where '
        'the measurements actually stop, painting fill across every gap',
    ),
)

_STACKED_AREA_BANNED = (
    (
        'st.values[i] || 0',
        'scrubs a MISSING sample to a measured zero: it fabricates a '
        'zero-height band at the hole AND feeds a hole-as-zero PARTIAL SUM to '
        'the column totals, understating the axis every other band is scaled '
        'against',
    ),
    (
        'Math.max(...totals',
        'folds the axis maximum over those scrubbed totals, so the axis is '
        'set by sums no column ever measured (and a surviving NaN poisons it '
        'outright)',
    ),
    (
        '(v / maxV) * chartH',
        'scales raw cumulative values inline, with no way to express "no '
        'measurement here" — a hole and a real zero land on the same pixel',
    ),
)

_BAR_CHART_BANNED = (
    (
        'Math.max(...values',
        'folds the axis maximum over the RAW series, so one undefined/NaN '
        'sample poisons max to NaN and EVERY bar is emitted as '
        'height="NaN" — not just the missing one',
    ),
    (
        '(v / max) * chartH',
        'scales each RAW value directly, so a `null` becomes a zero-height bar '
        'indistinguishable from a measured zero',
    ),
)

_HIST_BAR_BANNED = (
    (
        'Math.max(...values',
        'folds the axis maximum over the RAW series — same NaN-poisoning as '
        'BarChart, straight into a CSS height',
    ),
    (
        '(v / max) * 100',
        'scales each RAW value directly into a height percentage, so a `null` '
        'is indistinguishable from a measured zero',
    ),
    (
        'minHeight: 1',
        'floors every bar at 1px, so a MISSING sample renders as a visible '
        'stub — a fabricated measurement, drawn at a fixed size no data '
        'produced. A hole must render no bar at all, and a measured zero its '
        'honest zero-height one',
    ),
)

_CONTRACTS: dict[str, _Contract] = {
    'Sparkline': _Contract(builders=('sparkSmoothPaths',), banned=_SPARKLINE_BANNED),
    'StepSpark': _Contract(builders=('sparkStepPaths',), banned=_SPARKLINE_BANNED),
    # LineChart routes its gridline y through axisY too, so no sample-scaling
    # arithmetic is left inline anywhere in the component.
    'LineChart': _Contract(
        builders=('plottableMax', 'axisPaths', 'axisY'),
        banned=_LINE_CHART_BANNED,
    ),
    # StackedAreaChart gets its axis maximum back FROM the builder (a stack's
    # height is a property of the whole stack), then derives its y-ticks from
    # it through axisY.
    'StackedAreaChart': _Contract(
        builders=('stackedAreaPaths', 'axisY'),
        banned=_STACKED_AREA_BANNED,
    ),
    # The two bar primitives share one helper: barFractions hands back `null`
    # at a hole and a real fraction everywhere else, which BarChart multiplies
    # by chartH pixels and HistBar by 100 percent.
    'BarChart': _Contract(
        builders=('plottableMax', 'barFractions'),
        banned=_BAR_CHART_BANNED,
    ),
    # HistBar's `maxOverride ?? ...` precedence survives by passing the
    # override through as barFractions' explicit max.
    'HistBar': _Contract(
        builders=('plottableMax', 'barFractions'),
        banned=_HIST_BAR_BANNED,
    ),
}

_DELEGATING_COMPONENTS = tuple(_CONTRACTS)

# The subset that must decline to render ENTIRELY when nothing is plottable.
# A sparkline is nothing but its line, so an all-hole series must draw nothing.
# LineChart/StackedAreaChart are deliberately NOT in this list: their axes,
# gridlines and tick labels are structural facts about the requested window
# rather than measurements, and blanking them would hide from the operator that
# the chart was asked for at all.  Keeping the frame while dropping the marks is
# also what makes a hole legible AS a hole.
_WHOLE_COMPONENT_GUARD_COMPONENTS = ('Sparkline', 'StepSpark')


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def charts_jsx(_client) -> str:
    return _client.get('/static/redux/charts.jsx').text


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Copied from test_tab_escalation_analytics.py.  Paren-depth walks past the
    parameter list before looking for the body's opening ``{`` — these
    components take a destructured parameter (``function Sparkline({ values,
    ... }) {``) whose own ``{``/``}`` pair sits INSIDE the parameter list, so
    naively taking the first ``{`` after the opening ``(`` would return just
    the destructuring pattern instead of the function body.
    """
    m = re.search(rf'\bfunction\s+{re.escape(fn_name)}\s*\(', src)
    if m is None:
        return ''
    paren_depth = 1
    i = m.end()
    while i < len(src) and paren_depth > 0:
        if src[i] == '(':
            paren_depth += 1
        elif src[i] == ')':
            paren_depth -= 1
        i += 1
    if paren_depth != 0:
        return ''
    start = src.find('{', i)
    if start == -1:
        return ''
    depth = 0
    for j in range(start, len(src)):
        c = src[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    return ''


def _banned_arithmetic_hits(body: str, banned: tuple[tuple[str, str], ...]) -> list[str]:
    """Return the probes from ``banned`` that are present in ``body``."""
    return [probe for probe, _ in banned if probe in body]


def _component_body(charts_jsx: str, name: str) -> str:
    body = _extract_function_body(charts_jsx, name)
    assert body, (
        f'Could not locate the `function {name}(` body in charts.jsx — either '
        f'the component was removed/renamed, or it was rewritten as an arrow '
        f'function/class method, which _extract_function_body does not match. '
        f'This test cannot silently skip: without a body to inspect it proves '
        f'nothing.'
    )
    return body


# ---------------------------------------------------------------------------
# Delegation contract
# ---------------------------------------------------------------------------


def test_charts_jsx_destructures_the_builders_at_module_top_level(charts_jsx: str) -> None:
    """charts.jsx reads window.DF_SPARK_PATH once, at load, above the first function.

    Destructuring (rather than an object read or a ``|| {}`` fallback) is
    deliberate: a missing/404 spark_path.js then fails LOUDLY at load with a
    clear message, instead of deferring to a TypeError inside a render or
    silently degrading into charts that draw nothing.  Same contract
    test_index_html.py documents for DF_GRAPH_LAYOUT / DF_RUNTIME_FMT /
    DF_ORCH_FILTER, and the project's loud-over-silent-degradation norm.
    """
    m = re.search(r'window\.DF_SPARK_PATH', charts_jsx)
    assert m is not None, (
        'charts.jsx does not reference `window.DF_SPARK_PATH` — the chart '
        'scale/path math lives in /static/redux/spark_path.js (tasks 3436, '
        '3489) and must be reached through that global.'
    )

    destructure = re.search(
        r'const\s*\{[^}]*\}\s*=\s*window\.DF_SPARK_PATH\s*;', charts_jsx
    )
    assert destructure is not None, (
        'charts.jsx references window.DF_SPARK_PATH but does not DESTRUCTURE '
        'it (`const { sparkPaths: ..., stepPaths: ... } = window.DF_SPARK_PATH;`). '
        'An object read defers a missing-module failure to a TypeError inside a '
        'render; a `|| {}` fallback degrades silently into charts that draw '
        'nothing. The destructure throws at load, loudly.'
    )

    first_function = charts_jsx.find('function ')
    assert first_function != -1, 'charts.jsx contains no `function ` declaration at all.'
    assert destructure.start() < first_function, (
        'the window.DF_SPARK_PATH destructure sits BELOW the first function '
        'declaration in charts.jsx. It must be at module top level so the '
        'global is read once at load — the same top-level-destructure contract '
        'the other DF_* consumers follow.'
    )

    for name, contract in _CONTRACTS.items():
        for builder in contract.builders:
            assert builder in destructure.group(0), (
                f'the window.DF_SPARK_PATH destructure does not bind `{builder}`, '
                f'which {name} delegates to.'
            )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_no_longer_carries_the_defective_arithmetic(
    charts_jsx: str, name: str
) -> None:
    """No primitive may still scale raw samples itself.

    This is the assertion that fails against the pre-fix source.  Leaving the
    old arithmetic in place alongside the delegation would reintroduce the
    defect on whichever path actually renders.
    """
    contract = _CONTRACTS[name]
    body = _component_body(charts_jsx, name)
    hits = _banned_arithmetic_hits(body, contract.banned)

    assert not hits, (
        f'{name} still contains hole-blind scale/path arithmetic: {hits}. '
        + ' '.join(
            f'`{probe}` {why}.' for probe, why in contract.banned if probe in hits
        )
        + ' All of it now lives in /static/redux/spark_path.js, which excludes '
        'non-finite samples and breaks the path at every hole (tasks 3436, 3489).'
    )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_calls_all_of_its_required_builders(charts_jsx: str, name: str) -> None:
    """Every builder a component is supposed to delegate to is actually called.

    Binding a builder in the destructure but never calling it would leave the
    component computing something else — the absence assertions above are
    satisfied by ANY rewrite, including a wrong one.
    """
    contract = _CONTRACTS[name]
    body = _component_body(charts_jsx, name)

    for builder in contract.builders:
        assert re.search(rf'\b{re.escape(builder)}\s*\(', body), (
            f'{name} never calls `{builder}(`. It is bound at module top level '
            f'but unused here, so this component is not actually delegating its '
            f'scale/path math to spark_path.js.'
        )


@pytest.mark.parametrize('name', _WHOLE_COMPONENT_GUARD_COMPONENTS)
def test_component_delegates_to_its_builder_exactly_once(charts_jsx: str, name: str) -> None:
    """Each sparkline primitive calls its own builder, once."""
    body = _component_body(charts_jsx, name)
    builder = _CONTRACTS[name].builders[0]

    calls = len(re.findall(rf'\b{re.escape(builder)}\s*\(', body))
    assert calls == 1, (
        f'{name} calls `{builder}(` {calls} times; expected exactly 1. The '
        f'component should build its paths once and render them — a second '
        f'call means duplicated (and potentially divergent) geometry.'
    )

    other = _CONTRACTS['StepSpark' if name == 'Sparkline' else 'Sparkline'].builders[0]
    assert other not in body, (
        f'{name} references `{other}`, the OTHER builder. Sparkline must use '
        f'the smooth builder and StepSpark the step builder; crossing them '
        f'silently changes the interpolation of every chart on the tab.'
    )


@pytest.mark.parametrize('name', _WHOLE_COMPONENT_GUARD_COMPONENTS)
def test_component_renders_nothing_when_the_builder_yields_no_line(
    charts_jsx: str, name: str
) -> None:
    """An all-hole series must draw nothing, not a synthetic floor line.

    Pre-fix, ``[null, null, null]`` produced max=1/min=0 from the extrema seeds
    alone and three real points at ``y = height`` — a flat line along the chart
    floor asserting three measurements that were never taken.  The builder now
    returns an empty ``line`` for that input, and the component must decline to
    render on it.

    Deliberately scoped to the two SPARKLINES (see
    ``_WHOLE_COMPONENT_GUARD_COMPONENTS``): a LineChart/StackedAreaChart must
    keep drawing its axes, gridlines and tick labels even when no sample is
    plottable, because that frame describes the requested window rather than
    any measurement.

    Matched structurally (a falsy-check early return mentioning the
    destructured ``line``) rather than by exact wording, so a reworded guard
    does not fail this test spuriously.
    """
    body = _component_body(charts_jsx, name)

    assert re.search(r'\bline\b', body), (
        f'{name} does not reference a `line` binding — it should destructure '
        f'the builder result (`const {{ line, area: ... }} = ...`).'
    )

    guard = re.search(r'if\s*\(\s*!\s*line\s*\)\s*return\s+null\s*;', body)
    assert guard is not None, (
        f'{name} has no early return on an empty line path. Add a guard '
        f'equivalent to `if (!line) return null;` directly after the builder '
        f'call, so a series with no plottable samples renders nothing at all '
        f'rather than an <svg> containing a synthetic flat line along the '
        f'chart floor.'
    )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_still_exists_and_is_still_exported(charts_jsx: str, name: str) -> None:
    """Every pinned primitive survives the rewrite and stays on window.DF_CHARTS.

    Restated here (test_tab_curator.py already holds it for StepSpark) so this
    file fails LOUDLY rather than passing vacuously if a rewrite drops one —
    every other assertion in this module is about the ABSENCE of something, and
    absence is trivially satisfied by a deleted component.
    """
    assert f'function {name}(' in charts_jsx, (
        f'charts.jsx no longer defines `function {name}(`.'
    )

    parts = charts_jsx.split('window.DF_CHARTS')
    assert len(parts) > 1, 'charts.jsx has no `window.DF_CHARTS` export block.'
    assert name in parts[1], (
        f'{name} is not registered in the `window.DF_CHARTS = {{ ... }}` export '
        f'object at the bottom of charts.jsx — its consumers reach it through '
        f'that global.'
    )


# ---------------------------------------------------------------------------
# Negative control
#
# Each entry is a VERBATIM copy of that component's body as it stood before its
# fix — Sparkline's from before task 3436 (trimmed only of its JSX return, which
# carries no probe), LineChart's and StackedAreaChart's captured whole from
# charts.jsx before task 3489 touched it.  They are frozen: charts.jsx no longer
# contains this code at all, and these copies exist purely so the probes above
# can be proven to fire on the real defect.
# ---------------------------------------------------------------------------

_PRE_FIX_SPARKLINE = """{
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
  return null;
}"""

_PRE_FIX_LINE_CHART = """{
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
}"""

_PRE_FIX_STACKED_AREA = """{
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
          return <text key={i} x={x} y={height - 6} fontSize="9" fill={PALETTE.fg3} textAnchor="middle" fontFamily="JetBrains Mono">{formatX(lab)}</text>;
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
}"""

_PRE_FIX_BAR_CHART = """{
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
}"""

_PRE_FIX_HIST_BAR = """{
  const max = maxOverride ?? Math.max(...values, 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height }}>
      {values.map((v, i) => (
        <div key={i} style={{ flex: 1, height: `${(v / max) * 100}%`, background: color, borderRadius: '2px 2px 0 0', minHeight: 1 }} />
      ))}
    </div>
  );
}"""

_PRE_FIX_BODIES: dict[str, str] = {
    'Sparkline': _PRE_FIX_SPARKLINE,
    'LineChart': _PRE_FIX_LINE_CHART,
    'StackedAreaChart': _PRE_FIX_STACKED_AREA,
    'BarChart': _PRE_FIX_BAR_CHART,
    'HistBar': _PRE_FIX_HIST_BAR,
}


@pytest.mark.parametrize('name', sorted(_PRE_FIX_BODIES))
def test_banned_arithmetic_probes_actually_fire_on_pre_fix_source(name: str) -> None:
    """Each component's probes detect that component's real pre-fix code.

    Mirrors test_index_html.py's control.  Without this, a stale probe — a
    reformat that inserts a space, a variable rename — would make
    test_component_no_longer_carries_the_defective_arithmetic pass permanently
    while the defect sat untouched in charts.jsx.
    """
    banned = _CONTRACTS[name].banned
    hits = _banned_arithmetic_hits(_PRE_FIX_BODIES[name], banned)

    assert sorted(hits) == sorted(probe for probe, _ in banned), (
        f'the banned-arithmetic probes for {name} did NOT all fire on a '
        f'verbatim copy of its pre-fix body — only matched {hits}. At least '
        f'one probe has gone stale, which would make '
        f'test_component_no_longer_carries_the_defective_arithmetic a '
        f'permanent false GREEN for {name}.'
    )


def test_every_probe_set_has_a_pre_fix_control() -> None:
    """No component's probe set may go unproven.

    A component without its own captured pre-fix body is only acceptable if it
    SHARES a probe set with one that has been proven above — StepSpark shares
    Sparkline's four expressions verbatim.  Any other unproven set means a
    component is pinned by probes nothing has ever demonstrated can fire.
    """
    proven = [_CONTRACTS[name].banned for name in _PRE_FIX_BODIES]

    for name, contract in _CONTRACTS.items():
        assert contract.banned in proven, (
            f'{name}\'s probe set is never exercised by the negative control: '
            f'it has no verbatim pre-fix body in _PRE_FIX_BODIES and does not '
            f'share a probe set with a component that does. Add its pre-fix '
            f'body, or its absence assertions prove nothing.'
        )
