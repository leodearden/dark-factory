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
applied to every slot, rendering a hole as a visible 1px stub.  They now
delegate to the padded primitives in the same module.  (HistBar keeps that floor
on its MEASURED branch, where it is what stops a real 0 — ``height: 0%``, which
paints nothing — from rendering exactly like a hole.  What was wrong was
flooring the slot rather than the measurement.)

WHY THE PROBES ARE PER COMPONENT — the components' folds differ textually
(LineChart's poisoning fold is ``Math.max(...all``, BarChart's is
``Math.max(...values``, StackedAreaChart's is ``Math.max(...totals``), so a
single module-level probe tuple could not express what each component must no
longer contain.  Each entry in ``_CONTRACTS`` therefore carries its own probe
set, its own required builders, and — via ``_PRE_FIX_BODIES`` — its own
verbatim pre-fix body for the negative control at the bottom.

WHY THE PROBES SEE CODE ONLY — every probe here is a plain substring/regex
search, so ``_component_body`` blanks comments (``_strip_comments``) before any
of them run, and the negative control strips its frozen pre-fix bodies
identically.  Without that, a maintainer documenting the defect they just fixed,
in the component they just fixed, gets a CI failure whose message says the
component "still contains hole-blind scale/path arithmetic" — pointing at a
comment.  charts.jsx carried a parenthetical apologising for exactly that until
the stripper landed.
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
    # NOT probed here: `minHeight: 1`.  A blunt substring ban on the floor would
    # also ban it on the MEASURED branch, where it is what keeps a real zero
    # (`height: 0%`, which paints nothing) visible as a drawn-but-empty bar.
    # What was actually wrong pre-fix is that the floor applied to every slot,
    # hole included — a branch-shaped fact a substring cannot express.  It is
    # pinned by test_hist_bar_draws_nothing_for_a_hole_but_floors_a_measurement
    # below instead, which reads the two style branches and is proven to fail on
    # the pre-fix body by its own control.
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


def _strip_comments(src: str) -> str:
    """Return ``src`` with every JS/JSX comment blanked, string literals intact.

    EVERY probe in this file is a plain substring/regex search, so without this
    they also match PROSE.  That coupling deformed the production source once
    already: charts.jsx carried a comment whose content was an apology for what
    it could not say, because naming the very expression the component had just
    stopped using would fail CI with a message claiming the component "still
    contains hole-blind scale/path arithmetic" — pointing at a comment.  A
    comment is exactly where that expression SHOULD be quotable, so the probes
    are scoped to code and the comment now names ``(st.values[i] || 0)``
    directly.

    Quote-aware rather than a bare regex: blanking from a ``//`` inside a string
    literal (a URL, say) to end-of-line would delete real CODE, and an absence
    assertion over deleted code is a permanent false GREEN — the exact failure
    mode the negative control at the bottom of this file exists to prevent.
    Each comment is replaced by a single space rather than removed outright, so
    two previously separated tokens can never be spliced into a new match.

    Deliberately not a full JS lexer: a regex literal containing ``//``, or a
    quote nested inside a template's ``${...}``, would confuse it.  Neither
    occurs in these component bodies, and the negative control would catch the
    silent-GREEN direction if one ever did.
    """
    out: list[str] = []
    quote: str | None = None
    i, n = 0, len(src)

    while i < n:
        ch = src[i]

        if quote is not None:
            out.append(ch)
            if ch == '\\' and i + 1 < n:  # an escaped char cannot close the string
                out.append(src[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in '\'"`':
            quote = ch
            out.append(ch)
            i += 1
            continue

        if ch == '/' and src[i : i + 2] == '//':
            end = src.find('\n', i)
            out.append(' ')
            i = n if end == -1 else end
            continue

        if ch == '/' and src[i : i + 2] == '/*':
            end = src.find('*/', i + 2)
            out.append(' ')
            i = n if end == -1 else end + 2
            continue

        out.append(ch)
        i += 1

    return ''.join(out)


def _banned_arithmetic_hits(body: str, banned: tuple[tuple[str, str], ...]) -> list[str]:
    """Return the probes from ``banned`` that are present in ``body``.

    ``body`` must already be comment-stripped — ``_component_body`` does that for
    live source, and the negative control strips its frozen pre-fix bodies the
    same way, so both sides of that control see the identical treatment.
    """
    return [probe for probe, _ in banned if probe in body]


def _component_body(charts_jsx: str, name: str) -> str:
    """The named component's body, with comments blanked (see ``_strip_comments``).

    Every caller here asks a question about CODE — "does this component still
    scale raw samples itself", "does it actually call this builder" — so the
    single choke point strips prose once rather than leaving each probe to
    accidentally match a comment.
    """
    body = _extract_function_body(charts_jsx, name)
    assert body, (
        f'Could not locate the `function {name}(` body in charts.jsx — either '
        f'the component was removed/renamed, or it was rewritten as an arrow '
        f'function/class method, which _extract_function_body does not match. '
        f'This test cannot silently skip: without a body to inspect it proves '
        f'nothing.'
    )
    return _strip_comments(body)


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


def _balanced_brace_block(src: str, start: int) -> str:
    """Return the balanced ``{...}`` block beginning at ``src[start]``, or ''."""
    if start < 0 or start >= len(src) or src[start] != '{':
        return ''
    depth = 0
    for j in range(start, len(src)):
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    return ''


def _hole_and_measured_style_branches(body: str) -> tuple[str, str] | None:
    """Split a ``<cond> === null ? {hole} : {measured}`` style into its two branches.

    Returns ``None`` when the body has no such conditional at all — which is
    precisely what the PRE-FIX HistBar looks like, since it styled every slot
    the same way.  The negative control below depends on that distinction, so a
    missing conditional must come back as None rather than as an assertion here.
    """
    m = re.search(r'===\s*null\s*\?', body)
    if m is None:
        return None

    hole_start = body.find('{', m.end())
    hole = _balanced_brace_block(body, hole_start)
    if not hole:
        return None

    rest = body[hole_start + len(hole) :]
    colon = rest.find(':')
    if colon == -1:
        return None
    measured = _balanced_brace_block(rest, rest.find('{', colon))
    if not measured:
        return None

    return hole, measured


def test_hist_bar_draws_nothing_for_a_hole_but_floors_a_measurement(charts_jsx: str) -> None:
    """A hole and a measured zero must not render identically.

    Two failures are possible here and they pull in opposite directions.  The
    pre-fix one: ``minHeight: 1`` on every slot turned a MISSING sample into a
    visible 1px stub — a fabricated measurement drawn at a fixed size no data
    produced.  The mirror one: dropping the floor entirely leaves a measured
    zero at ``height: 0%``, which paints nothing at all, so a real zero and a
    hole are once again indistinguishable — the conflation this component was
    fixed to end — and any sub-pixel nonzero value disappears with them.

    So the floor is a property of the MEASURED branch, not of the slot.  That is
    branch-shaped, which is why it is asserted structurally rather than by a
    substring probe (see the note in ``_HIST_BAR_BANNED``).
    """
    body = _component_body(charts_jsx, 'HistBar')
    branches = _hole_and_measured_style_branches(body)

    assert branches is not None, (
        'HistBar no longer styles its slots through a `f === null ? ... : ...` '
        'conditional. A single unconditional style cannot distinguish a hole '
        'from a measurement — that is exactly the pre-fix shape.'
    )
    hole, measured = branches

    assert hole != measured, 'the hole and measured styles must differ'

    for painted in ('height', 'background', 'minHeight', 'borderRadius'):
        assert painted not in hole, (
            f'HistBar\'s HOLE branch sets `{painted}`: {hole}. A slot with no '
            f'measurement must keep its flex position and draw nothing — no '
            f'height, no fill, and above all no floor, which is what rendered '
            f'a hole as a fabricated 1px stub pre-fix.'
        )

    assert 'height' in measured and 'background' in measured, (
        f'HistBar\'s MEASURED branch does not paint a bar: {measured}'
    )
    assert 'minHeight' in measured, (
        f'HistBar\'s MEASURED branch has no visible floor: {measured}. A '
        f'measured 0 scales to `height: 0%` and paints no pixels, so without '
        f'one a real zero renders exactly like a hole — the conflation this '
        f'component was fixed to end. Floor the MEASURED branch only (or mark '
        f'the zero some other visible way); do not floor the slot.'
    )


def test_hist_bar_branch_contract_fails_on_pre_fix_source() -> None:
    """The negative control for the assertion above.

    The pre-fix HistBar styled every slot with one unconditional object carrying
    ``minHeight: 1``, so it has no hole branch to find.  If the extractor ever
    started returning branches for that body, the contract above would be
    asserting something the defect also satisfies.
    """
    body = _strip_comments(_PRE_FIX_HIST_BAR)

    assert _hole_and_measured_style_branches(body) is None, (
        'the pre-fix HistBar body parsed as having a hole/measured style split. '
        'It has none — it floors every slot at 1px unconditionally — so '
        'test_hist_bar_draws_nothing_for_a_hole_but_floors_a_measurement would '
        'be a false GREEN.'
    )
    assert 'minHeight: 1' in body, 'the pre-fix floor is still in the frozen control body'


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
    # Stripped exactly as the live source is, so this control proves the probes
    # fire on the pre-fix CODE — not on the `// build cumulative stacks` prose
    # that came with it, and not on a treatment the live side never gets.
    hits = _banned_arithmetic_hits(_strip_comments(_PRE_FIX_BODIES[name]), banned)

    assert sorted(hits) == sorted(probe for probe, _ in banned), (
        f'the banned-arithmetic probes for {name} did NOT all fire on a '
        f'verbatim copy of its pre-fix body — only matched {hits}. At least '
        f'one probe has gone stale, which would make '
        f'test_component_no_longer_carries_the_defective_arithmetic a '
        f'permanent false GREEN for {name}.'
    )


def test_strip_comments_removes_prose_without_touching_code_or_strings() -> None:
    """The probes must see code only — and must still see ALL of the code.

    Both directions matter.  If prose were probed, documenting the defect in the
    component that fixed it would fail CI (the reason this stripper exists).  If
    the stripper over-reached — blanking from a ``//`` inside a string literal to
    end-of-line — it would delete real code, and an absence assertion over
    deleted code is a permanent false GREEN.
    """
    prose = _strip_comments('const a = 1; // st.values[i] || 0 is the old scrub\nconst b = 2;')
    assert 'st.values[i] || 0' not in prose, 'a line comment must not be probed'
    assert 'const a = 1;' in prose and 'const b = 2;' in prose, 'code either side survives'

    block = _strip_comments('const a = 1; /* Math.max(...values, 1) */ const b = 2;')
    assert 'Math.max(...values' not in block, 'a block comment must not be probed'
    assert 'const a = 1;' in block and 'const b = 2;' in block

    jsx = _strip_comments('<g>{/* (v / max) * 100 */}<rect /></g>')
    assert '(v / max) * 100' not in jsx, 'a JSX comment must not be probed'
    assert '<rect />' in jsx

    # The over-reach direction: a `//` inside a string is NOT a comment, so
    # nothing after it may be blanked.
    for literal in ('"https://x/y"', "'https://x/y'", '`https://x/y`'):
        kept = _strip_comments(f'const u = {literal}; const h = (v / max) * chartH;')
        assert '(v / max) * chartH' in kept, (
            f'a `//` inside the string literal {literal} was treated as a comment, '
            f'blanking the code after it — that silently disarms every probe on '
            f'the rest of the line'
        )

    # Two tokens separated only by a comment must not be spliced into a match.
    assert 'ab' not in _strip_comments('a/* */b')


def test_prose_quoting_the_defect_does_not_trip_its_own_probe(charts_jsx: str) -> None:
    """A comment may name the expression the component stopped using.

    This is the contract that lets charts.jsx's StackedAreaChart comment say
    ``(st.values[i] || 0)`` outright instead of apologising for not being able
    to.  Asserted against a synthetic body AND against the real one, so it holds
    even if that comment is later reworded.
    """
    synthetic = _strip_comments(
        '{\n'
        '  // The cumulative folds used to scrub every sample through\n'
        '  // `(st.values[i] || 0)`, and the axis through `Math.max(...totals, 1)`.\n'
        '  const { max: maxV, paths } = stackedAreaPaths(stacks, geom);\n'
        '}'
    )
    assert _banned_arithmetic_hits(synthetic, _STACKED_AREA_BANNED) == [], (
        'a comment NAMING the pre-fix expression tripped the banned-arithmetic '
        'probe, which is what forced charts.jsx to document its own fix by '
        'refusing to quote it'
    )
    assert 'stackedAreaPaths(stacks, geom)' in synthetic, 'the code around it survives'

    body = _component_body(charts_jsx, 'StackedAreaChart')
    assert 'stackedAreaPaths' in body, 'the delegation is still visible after stripping'


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
