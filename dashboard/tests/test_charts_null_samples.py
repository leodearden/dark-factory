"""Structural contract: charts.jsx's sparkline primitives delegate to spark_path.js.

WHY THIS FILE IS SOURCE-ASSERTION AND NOT BEHAVIOURAL — charts.jsx is JSX
transformed by CDN Babel at runtime, and this repo has no node_modules, so its
component bodies cannot be parsed by node or rendered by React in any harness
here.  The BEHAVIOURAL coverage for the scale/path math lives in
``dashboard/tests/js/spark_path.test.mjs``, which executes the real module
under ``node --test``.  This file exists solely to prove the two components are
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
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

# The two primitives this task fixes.  LineChart / StackedAreaChart / BarChart /
# HistBar carry the same defect but have different scale signatures (padding
# offsets, a hard-coded minV = 0, stacking) and are deliberately out of scope —
# tracked as separate follow-up work.
_DELEGATING_COMPONENTS = ('Sparkline', 'StepSpark')

# Which builder each component must delegate to, keyed by component name.
_EXPECTED_BUILDER = {
    'Sparkline': 'sparkSmoothPaths',
    'StepSpark': 'sparkStepPaths',
}


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


# ---------------------------------------------------------------------------
# The defective-arithmetic probes.  Defined as module-level functions so the
# negative control at the bottom can run the EXACT same code over a synthetic
# pre-fix snippet — without that, a stale probe would give a permanent false
# GREEN.
# ---------------------------------------------------------------------------

# Each entry: (probe substring, what its presence would mean).
_BANNED_ARITHMETIC = (
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


def _banned_arithmetic_hits(body: str) -> list[str]:
    """Return the banned arithmetic probes present in ``body``."""
    return [probe for probe, _ in _BANNED_ARITHMETIC if probe in body]


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
        'charts.jsx does not reference `window.DF_SPARK_PATH` — the sparkline '
        'scale/path math lives in /static/redux/spark_path.js (task 3436) and '
        'must be reached through that global.'
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

    for name in _DELEGATING_COMPONENTS:
        builder = _EXPECTED_BUILDER[name]
        assert builder in destructure.group(0), (
            f'the window.DF_SPARK_PATH destructure does not bind `{builder}`, '
            f'which {name} delegates to.'
        )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_no_longer_carries_the_defective_arithmetic(
    charts_jsx: str, name: str
) -> None:
    """Neither primitive may still scale raw samples itself.

    This is the assertion that fails against the pre-fix source.  Leaving the
    old arithmetic in place alongside the delegation would reintroduce the
    defect on whichever path actually renders.
    """
    body = _component_body(charts_jsx, name)
    hits = _banned_arithmetic_hits(body)

    assert not hits, (
        f'{name} still contains hole-blind scale/path arithmetic: {hits}. '
        + ' '.join(
            f'`{probe}` {why}.' for probe, why in _BANNED_ARITHMETIC if probe in hits
        )
        + ' All of it now lives in /static/redux/spark_path.js, which excludes '
        'non-finite samples and breaks the path at every hole (task 3436).'
    )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_delegates_to_its_builder_exactly_once(charts_jsx: str, name: str) -> None:
    """Each primitive calls its own builder, once."""
    body = _component_body(charts_jsx, name)
    builder = _EXPECTED_BUILDER[name]

    calls = len(re.findall(rf'\b{re.escape(builder)}\s*\(', body))
    assert calls == 1, (
        f'{name} calls `{builder}(` {calls} times; expected exactly 1. The '
        f'component should build its paths once and render them — a second '
        f'call means duplicated (and potentially divergent) geometry.'
    )

    other = _EXPECTED_BUILDER['StepSpark' if name == 'Sparkline' else 'Sparkline']
    assert other not in body, (
        f'{name} references `{other}`, the OTHER builder. Sparkline must use '
        f'the smooth builder and StepSpark the step builder; crossing them '
        f'silently changes the interpolation of every chart on the tab.'
    )


@pytest.mark.parametrize('name', _DELEGATING_COMPONENTS)
def test_component_renders_nothing_when_the_builder_yields_no_line(
    charts_jsx: str, name: str
) -> None:
    """An all-hole series must draw nothing, not a synthetic floor line.

    Pre-fix, ``[null, null, null]`` produced max=1/min=0 from the extrema seeds
    alone and three real points at ``y = height`` — a flat line along the chart
    floor asserting three measurements that were never taken.  The builder now
    returns an empty ``line`` for that input, and the component must decline to
    render on it.

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
    """Both primitives survive the rewrite and stay on window.DF_CHARTS.

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
# ---------------------------------------------------------------------------


def test_banned_arithmetic_probes_actually_fire_on_pre_fix_source() -> None:
    """The probes detect the real pre-fix code (mirrors test_index_html.py's control).

    Without this, a stale probe — a reformat that inserts a space, a variable
    rename — would make every assertion above pass permanently while the defect
    sat untouched in charts.jsx.  The snippet below is the Sparkline body as it
    stood before task 3436, verbatim.
    """
    pre_fix_body = """{
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

    hits = _banned_arithmetic_hits(pre_fix_body)

    assert sorted(hits) == sorted(probe for probe, _ in _BANNED_ARITHMETIC), (
        'the banned-arithmetic probes did NOT all fire on a verbatim copy of '
        f'the pre-fix Sparkline body — only matched {hits}. At least one probe '
        'has gone stale, which would make '
        'test_component_no_longer_carries_the_defective_arithmetic a permanent '
        'false GREEN.'
    )
