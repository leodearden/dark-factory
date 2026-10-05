"""Behavioural contract: StackedAreaChart hands formatY the RAW y-axis tick.

WHY THIS FILE MIXES SOURCE-EXTRACTION WITH A NODE SUBPROCESS — charts.jsx is
JSX transformed by CDN Babel at runtime, and this repo has no node_modules, so
its component bodies cannot be imported by node or rendered by React in any
harness here (same rationale block as ``test_charts_null_samples.py``:3-12 and
``test_tab_memory_evals.py``:8).  The repo's usual answer for executable chart
math is extraction into a plain-JS classic script (``spark_path.js`` +
``dashboard/tests/js/spark_path.test.mjs``), but that costs a new ``<script>``
tag, a ``?v=`` registration, load-order pins in ``test_index_html.py`` and
registration in ``classic_script_scope.test.mjs`` — disproportionate for a
two-token WIRING fix.  Pure source assertions, though, could only ever prove the
TEXT changed, never that the axis actually reads 0/25/50/75/100%.

So this file extracts the REAL committed source text of the default
``formatY``, the argument expression handed to ``formatY`` at the y-tick
``<text>`` element, the tick generator, and WorkflowPanel's own ``formatY``,
then EXECUTES that composed pipeline under ``node -e`` and asserts on the
rendered label strings.  The extractors assert loudly on a miss (never silently
returning '') and the negative control at the bottom proves they still fire on
verbatim pre-fix source, so a rename cannot turn this file into a false GREEN.
Assertions are on rendered LABELS, not on source spelling, so renaming the tick
variable or swapping in an equivalent rounding default stays green.

THE DEFECT (task 4059) — the y-tick label was rendered as
``{formatY(Math.round(t))}``, snapping the raw fractional tick to an integer
BEFORE the caller's own formatter ever saw it.  ``WorkflowPanel``
(tab_escalation_analytics.jsx:494) plots a 100%-normalized stack whose bands sum
to exactly 1.0, so ``maxV`` is 1.0 and the ticks are 0/0.25/0.5/0.75/1.0.
``Math.round`` collapsed those to 0/0/1/1/1, and the panel's
``v => `${Math.round(v * 100)}%``` rendered the axis as
"0% / 0% / 100% / 100% / 100%" — every intermediate gridline mislabelled.

THE FIX — the rounding is not deleted, it MOVES into the ``formatY`` default
(``v => String(Math.round(v))``) and ``formatY`` receives the raw tick.  That
composition is byte-identical for the three callers that pass no formatter at
all (tabs.jsx:1259, tabs.jsx:1329, tab_escalation_analytics.jsx:244 — all
integer counts), which ``test_default_format_y_callers_keep_integer_count_axes``
pins executably rather than by inspection.

ALSO HERE: THE LINECHART CALLER AUDIT (task 4232) — task 4059 deferred the
audit of LineChart's non-rounding default, and this file now carries its
outcome.  Of the four call sites that inherit that default, three plot integer
COUNTS (memory reads/writes, merge attempts per bucket, escalation churn) and
have gained ``formatY={formatCountTick}``; the fourth plots
``escalation_analytics.py::_esc_per_done``'s ``filings / done``, a genuine
fraction, and must keep the raw default.  That asymmetry is what makes a
rounding default unsafe — it would collapse the ratio axis exactly as
pre-4059 pre-rounding collapsed WorkflowPanel's percent axis — so the ratio
site is pinned as an ANTI-regression alongside the three fixes, with a frozen
rounding-default control proving the guard actually fires.  The helper itself
lives in spark_path.js and is behaviourally tested under ``node --test``; what
is measured HERE is the wiring only charts.jsx and the tabs can express.

ALSO HERE: COUNT-AXIS SNAPPING (task 5121) — LineChart and StackedAreaChart
take an opt-in ``snapMax`` that maps the folded data maximum to the axis
maximum, and count callers pass spark_path.js's ``niceCountMax``.  The
scaled-axis programs below execute each component's REAL ``const`` statements
that carry a data maximum to its raw ticks (LineChart's own scale lines;
StackedAreaChart's call into the real ``stackedAreaPaths``) against the real
spark_path.js.  They emit those statements in SOURCE ORDER, so a snap line that
reads ``ticks`` before its declaration throws the same TDZ ReferenceError the
browser would, and they return RAW ticks rather than labels, so one expectation
table holds across callers whose formatters differ.  The caller audit is pinned
in both directions: the seven COUNT call sites must pass ``snapMax``, and the
three FRACTION sites (the esc-per-done ratio, the 100%-normalized stack and the
ECDF percent axis) must not, because a snapped 1 reads as 400%.  The duration
and dollar charts are deliberately left unpinned either way.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _dashboard_helpers import (
    DF_CHARTS_DESTRUCTURE_RE,
    DF_CHARTS_EXPORT_RE,
    destructure_bindings,
    extract_function_body,
    find_function_params,
)

# ---------------------------------------------------------------------------
# The served-asset fixtures (`charts_jsx_body`, `tab_analytics_jsx_body`,
# `index_html_body`) and the `_client` they read through now live in
# conftest.py (task 3549).  Reading through the app's own static route rather
# than the filesystem still proves the asset is actually SERVED, which is the
# property the cache-buster assertion below depends on.
# ---------------------------------------------------------------------------


def _extract_signature(src: str, fn_name: str) -> str:
    """Return the parameter-list text of ``function <fn_name>(...)``, parens excluded.

    A thin projection over the shared `find_function_params` paren-depth walk.
    It stays module-local rather than being hoisted because it has exactly one
    consumer (`_signature_default` below); what was worth sharing was the WALK,
    not this slice of its result.  It cannot be replaced by
    `extract_function_body` either: that returns the BODY, and the two slices
    are disjoint — `_signature_default` regexes `<prop> = <default>` out of the
    PARAMETER LIST, which the body excludes.

    The miss message is kept file-specific: naming the vacuous-GREEN
    consequence for THIS file is more use at this call site than the shared
    helper's four-way wording.
    """
    def _miss(what: str) -> BaseException:
        return AssertionError(
            f'no `function {fn_name}(` declaration found in the source ({what}) '
            '— the component was renamed or converted to another declaration '
            'form, and every assertion in this file would go vacuously GREEN.'
        )

    # This caller wants the params, not the body that follows them, so the
    # returned mask is unused here.
    _masked, params_start, params_end = find_function_params(
        src, fn_name, miss=_miss,
    )
    return src[params_start:params_end]


# ---------------------------------------------------------------------------
# Extractors.  Each fails LOUDLY on a miss — a silent '' would make a rename
# turn this whole file into a permanent false GREEN.  Neither needs a guard of
# its own any more: both are projections over shared `_dashboard_helpers`
# helpers that RAISE rather than return '' (task 3549, task 4881).
# `_component_body` takes the BODY via `extract_function_body`;
# `_extract_signature` above takes the PARAMETER LIST via the same
# `find_function_params` paren walk that helper is built on.
# ---------------------------------------------------------------------------


def _component_body(src: str, name: str) -> str:
    return extract_function_body(src, name)


def _signature_default(src: str, component: str, prop: str) -> str:
    """The ``<prop> = <default>`` expression from ``component``'s signature.

    Every default read here is an arrow containing neither a comma nor a brace
    (``v => String(Math.round(v))``, ``(dataMax) => dataMax``), so a ``[^,}]+``
    run over the signature slice captures exactly it.
    """
    signature = _extract_signature(src, component)
    m = re.search(rf'\b{prop}\s*=\s*([^,}}]+)', signature)
    assert m is not None, (
        f'{component} no longer declares a `{prop} = <default>` in its '
        f'signature. Signature was: {signature!r}'
    )
    return m.group(1).strip()


def _default_format_y(charts_jsx_body: str, component: str = 'StackedAreaChart') -> str:
    """The ``formatY`` default: StackedAreaChart's (task 4059) unless told otherwise."""
    return _signature_default(charts_jsx_body, component, 'formatY')


def _tick_label_arg(body: str) -> str:
    """The argument expression actually handed to ``formatY`` at the y-tick ``<text>``.

    Takes an already-sliced COMPONENT BODY, not the whole file: LineChart and
    StackedAreaChart carry textually identical ``formatY(t)}</text>`` markup, so
    an unscoped search would read the wrong component's wiring.
    """
    m = re.search(r'formatY\((.*?)\)\}</text>', body)
    assert m is not None, (
        'could not find the y-tick `{formatY(...)}</text>` label element in the '
        'component body — the axis label markup changed shape and this file no '
        'longer measures what reaches formatY.'
    )
    return m.group(1).strip()


def _tick_map_param(body: str) -> str:
    """The name bound to each tick by ``yTicks.map((<param>, i) => ...)``.

    Extracted rather than hardcoded as ``t`` so that renaming the tick variable
    is a behaviour-preserving refactor here, not a test failure: the extracted
    label-argument expression is evaluated in a scope where this name is bound.
    """
    m = re.search(r'yTicks\.map\(\(\s*(\w+)', body)
    assert m is not None, (
        'could not find the `yTicks.map((<tick>, i) => ...)` y-axis loop in the '
        'component body — the extracted label expression has no scope to run in.'
    )
    return m.group(1)


def _tick_generator(body: str) -> str:
    """The verbatim ``const ticks = ...;`` / ``const yTicks = Array.from(...);`` lines."""
    ticks = re.search(r'^\s*(const ticks\s*=\s*[^;]+;)', body, re.MULTILINE)
    y_ticks = re.search(r'^\s*(const yTicks\s*=\s*Array\.from\(.*?\);)', body, re.MULTILINE)
    assert ticks is not None, 'the component body no longer declares `const ticks = ...;`'
    assert y_ticks is not None, (
        'the component body no longer declares `const yTicks = Array.from(...);` — '
        'the tick generator changed shape, so the executable assertions below '
        'would no longer run the real generator.'
    )
    return f'{ticks.group(1)}\n  {y_ticks.group(1)}'


def _const_statements_in_source_order(body: str, names) -> str:
    """The one single-line ``const`` binding each of ``names``, joined in SOURCE order.

    A name is bound either plainly (``const maxV = ...;``) or inside a
    destructure pattern (``const { max: maxV, paths, stepX } = ...;``).  Source
    order is the point: a statement that reads a name before its ``const`` is
    declared then throws the same TDZ ReferenceError the browser would.
    """
    found = []
    for name in names:
        pattern = (
            rf'^[ \t]*(const\s+(?:{name}\b|\{{[^}}\n]*\b{name}\b[^}}\n]*\}})\s*=.*;)[ \t]*$'
        )
        matches = list(re.finditer(pattern, body, re.MULTILINE))
        assert len(matches) == 1, (
            f'expected exactly one single-line `const` binding `{name}` in the '
            f'component body, found {len(matches)}. The axis scale arithmetic '
            'changed shape, so the scaled-axis program would no longer run the '
            'real committed statements.'
        )
        found.append(matches[0])
    return '\n  '.join(m.group(1) for m in sorted(found, key=lambda m: m.start()))


def _workflow_panel_format_y(tab_analytics_jsx_body: str) -> str:
    """WorkflowPanel's own percent formatter, from its ``<C.StackedAreaChart>`` call.

    Non-greedy up to ``} />``, which correctly steps over the ``${...}`` brace
    pair inside the template literal.
    """
    body = _component_body(tab_analytics_jsx_body, 'WorkflowPanel')
    m = re.search(r'<C\.StackedAreaChart\b[^>]*?formatY=\{(.*?)\}\s*/>', body)
    assert m is not None, (
        'WorkflowPanel no longer passes a `formatY={...}` to <C.StackedAreaChart> '
        '— the 100%-normalized resolver-mix axis this task fixes has moved or '
        'changed shape.'
    )
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# API-surface extractors (task 4232) — the DF_SPARK_PATH -> DF_CHARTS route.
#
# `formatCountTick` is defined in spark_path.js and CONSUMED in tabs.jsx and
# tab_escalation_analytics.jsx, which reach it only through `window.DF_CHARTS`.
# charts.jsx is the one hop in between, and that hop is pure wiring — exactly
# the kind of two-token join that goes silently missing in a rename.
# ---------------------------------------------------------------------------

_SPARK_PATH_DESTRUCTURE_RE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_SPARK_PATH')
# `_SPARK_PATH_DESTRUCTURE_RE` above stays LOCAL: it is a different namespace
# and this is its only consumer.  The DF_CHARTS pair it shadows does NOT — both
# are imported from `_dashboard_helpers` (see that module's banner for why the
# brace-hostile `[^{}]*` must stay).  All three feed the same shared
# `destructure_bindings`; only the projection below is this module's own.
# The wrappers still assert on a miss explicitly, so that a nested `{}` fails
# HERE, naming the coupling, instead of downstream as an opaque "could not
# parse the DF_CHARTS exports".

_SPARK_PATH_JS = (
    Path(__file__).resolve().parent.parent
    / 'src' / 'dashboard' / 'static' / 'redux' / 'spark_path.js'
)


def _binding_names(brace_body: str) -> set:
    """The SOURCE property names in a destructure or object-literal brace body.

    `sparkPaths: sparkSmoothPaths` -> `sparkPaths` — the name read OFF the
    namespace, which is the one that has to actually exist on it. A bare
    `axisY` is both source and local.
    """
    return {canonical for canonical, _local in destructure_bindings(brace_body)}


def _spark_path_destructure(charts_jsx_body: str) -> set:
    """Names charts.jsx destructures off ``window.DF_SPARK_PATH`` at module top level."""
    m = _SPARK_PATH_DESTRUCTURE_RE.search(charts_jsx_body)
    assert m is not None, (
        'charts.jsx no longer opens with a `const { ... } = window.DF_SPARK_PATH;` '
        'destructure — either the spark_path.js dependency was rewired (in which '
        'case the load-order and cache-buster reasoning in this file and in '
        'charts.jsx:13-20 no longer applies) or the binding list grew a nested '
        'brace this extractor cannot read. Not returning an empty set, because '
        'that would make every routing assertion below vacuously GREEN.'
    )
    return _binding_names(m.group(1))


def _df_charts_export_names(charts_jsx_body: str) -> set:
    """Key names in the ``window.DF_CHARTS = { ... }`` export literal."""
    m = DF_CHARTS_EXPORT_RE.search(charts_jsx_body)
    assert m is not None, (
        'could not read the `window.DF_CHARTS = { ... }` export literal in '
        'charts.jsx. The overwhelmingly likely cause is a NESTED BRACE inside '
        'that literal: `_dashboard_helpers.py::DF_CHARTS_EXPORT_RE` is '
        '`[^{}]*` by design. That is ONE shared object every DF_CHARTS '
        'consumer imports, so a nested brace does not break this module alone '
        '— test_tab_burndown.py reads the same pattern, where the miss is '
        'silent and surfaces as an empty export set failing '
        'test_every_labels_prop_sits_on_a_chart_component with an '
        'unrelated-looking message. Keep every export a bare identifier.'
    )
    return _binding_names(m.group(1))


def _df_charts_destructure(src: str) -> set:
    """Names ``src`` pulls off ``window.DF_CHARTS`` by destructure, across all of them."""
    matches = list(DF_CHARTS_DESTRUCTURE_RE.finditer(src))
    assert matches, (
        'this source no longer has a `const { ... } = window.DF_CHARTS` destructure '
        'at all — either it was rewired onto a namespace binding (`const C = '
        'window.DF_CHARTS`, which needs no per-name entry and would make the '
        'assertion below inapplicable rather than failing) or the binding list grew '
        'a nested brace this extractor cannot read. Not returning an empty set, '
        'because that would make the binding assertion vacuously GREEN.'
    )
    return {name for m in matches for name in _binding_names(m.group(1))}


def _spark_path_module_surface() -> dict:
    """``{name: typeof}`` for the REAL shipped spark_path.js, executed under node.

    Proves the name charts.jsx binds actually EXISTS at the source, rather than
    merely being spelled the same in two files — the failure mode a pure
    source-text pairing cannot see.
    """
    return _run_node(
        f'const api = require({json.dumps(str(_SPARK_PATH_JS))});\n'
        'console.log(JSON.stringify(Object.fromEntries('
        'Object.keys(api).map(k => [k, typeof api[k]]))));'
    )


# ---------------------------------------------------------------------------
# Caller-audit extractors (task 4232) — what each LineChart CALL SITE passes.
#
# The audit's outcome is a per-caller fact, not a property of charts.jsx, so it
# has to be read out of the tab sources. `_line_chart_format_y` returns None for
# a site that passes no formatter — a real, load-bearing answer here (three
# sites must stop relying on the default and one must keep relying on it), which
# is why a MISSING ELEMENT asserts instead of also returning None.
# ---------------------------------------------------------------------------

# Every spelling the tabs use: tabs.jsx aliases `LineChart: LC` (and
# `StackedAreaChart: SA`) in its line-2 DF_CHARTS destructure,
# tab_escalation_analytics.jsx goes through its `const C = window.DF_CHARTS`
# namespace, and tab_overview.jsx binds `LineChart` unaliased.
_LINE_CHART_TAG_RE = re.compile(r'<(?:LC|LineChart|C\.LineChart)\b')
_STACKED_AREA_TAG_RE = re.compile(r'<(?:SA|C\.StackedAreaChart)\b')


def _jsx_elements(body: str, tag_re: re.Pattern) -> list:
    """Every self-closing JSX element in `body` whose tag matches `tag_re`.

    Walks to the element's own `/>` at brace depth 0, so a `series={[{...}]}`
    prop containing braces (or a `${...}` inside a template literal) cannot end
    the element early. A regex cannot do this: these props nest.
    """
    out = []
    for m in tag_re.finditer(body):
        depth = 0
        i = m.end()
        while i < len(body):
            ch = body[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif depth == 0 and ch == '/' and body[i + 1 : i + 2] == '>':
                out.append(body[m.start() : i + 2])
                break
            i += 1
    return out


def _chart_prop(body: str, tag_re: re.Pattern, anchor: str, prop: str):
    """The `<prop>={...}` expression of the chart call site containing `anchor`.

    Returns the expression text, or ``None`` when the site does not pass `prop`
    and therefore inherits the component's default. `body` is an already-sliced
    COMPONENT body: several tabs render more than one chart, and the anchor
    (a series expression such as ``MEMORY_OPS.reads``) is what names the axis.
    """
    matches = [el for el in _jsx_elements(body, tag_re) if anchor in el]
    assert len(matches) == 1, (
        f'expected exactly one chart element matching {tag_re.pattern!r} whose '
        f'props mention {anchor!r}, found {len(matches)}. The chart was renamed, '
        'removed, or its series expression changed — either way the caller-audit '
        'assertion that depends on it must FAIL rather than quietly measure a '
        'different chart (or nothing at all).'
    )
    element = matches[0]
    m = re.search(rf'\b{prop}=\{{', element)
    if m is None:
        return None
    depth = 1
    i = m.end()
    while i < len(element) and depth > 0:
        if element[i] == '{':
            depth += 1
        elif element[i] == '}':
            depth -= 1
        i += 1
    assert depth == 0, f'unbalanced {prop}={{...}} in the element anchored at {anchor!r}'
    return element[m.end() : i - 1].strip()


def _line_chart_format_y(body: str, anchor: str):
    """The `formatY={...}` of the LineChart call site at `anchor`, or None (task 4232)."""
    return _chart_prop(body, _LINE_CHART_TAG_RE, anchor, 'formatY')


def _line_chart_axis_scalars(body: str) -> str:
    """LineChart's verbatim ``const minV = ...;`` / ``const range = ...;`` lines.

    LineChart's tick generator is written in terms of `minV` and `range` (where
    StackedAreaChart's is written in terms of `maxV`), so a program that runs
    the real generator has to carry LineChart's real scale arithmetic too.
    Extracted rather than hardcoded, for the same reason as everything else in
    this file: changing `minV = 0` or dropping the `|| 1` degenerate-range guard
    must move these assertions rather than slip past them.
    """
    min_v = re.search(r'^\s*(const minV\s*=\s*[^;]+;)', body, re.MULTILINE)
    rng = re.search(r'^\s*(const range\s*=\s*[^;]+;)', body, re.MULTILINE)
    assert min_v is not None, (
        'the component body no longer declares `const minV = ...;` — LineChart '
        "'s scale arithmetic changed shape and the extracted tick generator has "
        'nothing to run against.'
    )
    assert rng is not None, (
        'the component body no longer declares `const range = ...;` — see above.'
    )
    return f'{min_v.group(1)}\n  {rng.group(1)}'


def _line_chart_axis_program(body: str, format_y: str, max_vs) -> str:
    """Render a LineChart y-axis by executing its REAL committed pipeline.

    Sibling of `_axis_labels_program` above, differing in two ways: it also
    carries LineChart's own `minV`/`range` lines (see `_line_chart_axis_scalars`),
    and it binds the real spark_path.js module first, so a caller expression such
    as `formatCountTick` or `C.formatCountTick` resolves to the genuinely SHIPPED
    function rather than to more extracted text.

    `C` is bound to the spark_path API rather than to the real DF_CHARTS (which
    is charts.jsx, unparseable by node). That is sound precisely because
    test_charts_jsx_routes_count_axis_helper_from_spark_path_to_df_charts pins
    the DF_SPARK_PATH -> DF_CHARTS hop separately: this program measures what the
    caller's expression RENDERS, that one measures that the name is routed.
    """
    return f"""
const DF_SPARK_PATH = require({json.dumps(str(_SPARK_PATH_JS))});
const {{ formatCountTick }} = DF_SPARK_PATH;
const C = Object.assign({{}}, DF_SPARK_PATH);
const formatY = {format_y};
function yTicksFor(maxV) {{
  {_line_chart_axis_scalars(body)}
  {_tick_generator(body)}
  return yTicks;
}}
const out = {{}};
for (const maxV of {json.dumps(list(max_vs))}) {{
  out[maxV] = yTicksFor(maxV).map(({_tick_map_param(body)}, i) => formatY({_tick_label_arg(body)}));
}}
console.log(JSON.stringify(out));
"""


def _render_caller_axis(charts_jsx_body: str, caller_body: str, anchor: str, max_vs) -> dict:
    """Render the axis the caller at `anchor` actually draws, defaults included."""
    line_chart = _component_body(charts_jsx_body, 'LineChart')
    format_y = _line_chart_format_y(caller_body, anchor)
    if format_y is None:
        format_y = _default_format_y(charts_jsx_body, 'LineChart')
    return _run_node(_line_chart_axis_program(line_chart, format_y, max_vs))


def _scaled_axis_program(snap_max: str, data_maxes, fixture: str, statements: str) -> str:
    """Run real axis statements per data maximum and print ``{dataMax: rawTicks}``.

    Binds the real spark_path.js, and `C` to its API, for the same reason as
    `_line_chart_axis_program`: a caller's `niceCountMax` / `C.niceCountMax`
    then resolves to the genuinely SHIPPED function.
    """
    return f"""
const DF_SPARK_PATH = require({json.dumps(str(_SPARK_PATH_JS))});
const {{ plottableMax, niceCountMax, stackedAreaPaths }} = DF_SPARK_PATH;
const C = Object.assign({{}}, DF_SPARK_PATH);
const snapMax = {snap_max};
const out = {{}};
for (const dataMax of {json.dumps(list(data_maxes))}) {{
  {fixture}
  {statements}
  out[dataMax] = yTicks;
}}
console.log(JSON.stringify(out));
"""


def _line_chart_scaled_axis_program(body: str, snap_max: str, data_maxes) -> str:
    """LineChart's real data-max -> raw-ticks statements, under `snap_max`."""
    return _scaled_axis_program(
        snap_max,
        data_maxes,
        'const series = [{ values: [0, dataMax] }];',
        _const_statements_in_source_order(body, ['all', 'ticks', 'maxV', 'minV', 'range', 'yTicks']),
    )


def _stacked_area_scaled_axis_program(body: str, snap_max: str, data_maxes) -> str:
    """StackedAreaChart's real statements, through the real ``stackedAreaPaths``."""
    return _scaled_axis_program(
        snap_max,
        data_maxes,
        "const stacks = [{ key: 'a', values: [dataMax] }];\n"
        '  const geom = { x0: 38, y0: 8, width: 100, height: 190, count: 1 };',
        _const_statements_in_source_order(body, ['ticks', 'maxV', 'yTicks']),
    )


_SCALED_AXIS_PROGRAMS = {
    'LineChart': _line_chart_scaled_axis_program,
    'StackedAreaChart': _stacked_area_scaled_axis_program,
}


def _render_snapped_caller_axis(
    charts_jsx_body: str, component: str, caller_body: str, tag_re: re.Pattern, anchor: str, data_maxes
) -> dict:
    """The raw ticks the caller at `anchor` actually draws, its snapMax default included."""
    snap_max = _chart_prop(caller_body, tag_re, anchor, 'snapMax')
    if snap_max is None:
        snap_max = _signature_default(charts_jsx_body, component, 'snapMax')
    program = _SCALED_AXIS_PROGRAMS[component]
    return _run_node(program(_component_body(charts_jsx_body, component), snap_max, data_maxes))


# ---------------------------------------------------------------------------
# node -e harness.  Same invocation shape as test_graph_layout_js.py:37-63,
# including its hard-assert-not-skip policy: node v22.22.3 is a verified part
# of the host/CI toolchain, so an absent node is an environment regression that
# must not silently drop this task's only behavioural coverage.
# ---------------------------------------------------------------------------


def _run_node(program: str):
    node = shutil.which('node')
    assert node is not None, (
        'node executable not found on PATH — node v22.22.3 is required to '
        'execute the extracted charts.jsx axis-label expressions. This is a '
        'hard failure, not a skip: node is a verified part of the host/CI '
        'toolchain, so its absence is a regression that must not be hidden '
        "behind a skip (which would silently drop this file's only "
        'behavioural coverage).'
    )
    # The program text is COMPOSED FROM REGEX-EXTRACTED SOURCE, so a future
    # extraction that captures a partial expression could yield a program that
    # spins or blocks. Bound it and detach stdin, so an extraction failure
    # surfaces as a readable assertion instead of hanging the pytest run.
    try:
        result = subprocess.run(
            [node, '-e', program],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            'node -e did not terminate within 60s evaluating the extracted '
            'axis-label pipeline. That is an EXTRACTION failure, not a chart '
            'defect: one of the regexes above captured a partial or unbalanced '
            'expression and composed a program that never exits.\n'
            f'--- program ---\n{program}'
        ) from exc
    assert result.returncode == 0, (
        f'node -e exited {result.returncode} evaluating the extracted axis-label '
        f'pipeline\n--- program ---\n{program}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    return json.loads(result.stdout)


def _axis_labels_program(label_arg: str, tick_map_param: str, tick_gen: str, format_y: str, max_vs) -> str:
    """Compose the real extracted expressions into a runnable axis-label pipeline."""
    return f"""
const formatY = {format_y};
function yTicksFor(maxV) {{
  {tick_gen}
  return yTicks;
}}
const out = {{}};
for (const maxV of {json.dumps(list(max_vs))}) {{
  out[maxV] = yTicksFor(maxV).map(({tick_map_param}, i) => formatY({label_arg}));
}}
console.log(JSON.stringify(out));
"""


# ---------------------------------------------------------------------------
# Anti-regression probe — where the rounding must NOT be.
# ---------------------------------------------------------------------------


def test_stacked_area_chart_does_not_pre_round_the_tick_before_format_y(charts_jsx_body: str) -> None:
    """StackedAreaChart must not snap the tick to an integer before formatY sees it.

    Pre-rounding destroys the fractional information every percent/decimal
    formatter needs — the caller's own formatter is the only thing that knows
    the axis UNITS. FAILED before the fix at charts.jsx:186.
    """
    body = _component_body(charts_jsx_body, 'StackedAreaChart')

    assert 'formatY(Math.round(' not in body, (
        "StackedAreaChart's body still calls `formatY(Math.round(...))`, "
        "snapping each y-tick to an integer BEFORE handing it to the caller's "
        'formatter. On the 100%-normalized Workflow panel that collapses the '
        'ticks 0/0.25/0.5/0.75/1.0 to 0/0/1/1/1 and renders the axis as '
        '"0% / 0% / 100% / 100% / 100%". Pass the raw tick and let the '
        'formatY DEFAULT do the rounding instead.'
    )


# ---------------------------------------------------------------------------
# The heart of the fix — executed against the real committed source.
# ---------------------------------------------------------------------------


def test_workflow_panel_percent_axis_reads_0_25_50_75_100(charts_jsx_body: str, tab_analytics_jsx_body: str) -> None:
    """The 100%-normalized Workflow axis renders every intermediate tick correctly.

    Runs the REAL extracted expressions — StackedAreaChart's tick generator at
    ``maxV = 1`` (the normalized stack sums to exactly 1.0), the argument
    expression actually handed to formatY, and WorkflowPanel's own
    ``v => `${Math.round(v * 100)}%``` — under node.

    Before the fix this produced ``['0%', '0%', '100%', '100%', '100%']``:
    the user-visible defect, executed against shipped source. The ticks
    0/0.25/0.5/0.75/1 are exact binary fractions, so ``Math.round(0.25 * 100)``
    is exactly 25 and no tolerance is needed.
    """
    body = _component_body(charts_jsx_body, 'StackedAreaChart')

    labels = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_workflow_panel_format_y(tab_analytics_jsx_body),
            max_vs=[1],
        )
    )['1']

    assert labels == ['0%', '25%', '50%', '75%', '100%'], (
        "the Workflow panel's 100%-normalized y-axis renders as "
        f'{labels} instead of the expected 0%/25%/50%/75%/100%. The tick '
        'values reaching formatY have been rounded, scaled, or generated '
        'differently than the extracted source implies.'
    )


def test_default_format_y_callers_keep_integer_count_axes(charts_jsx_body: str) -> None:
    """The three no-formatY callers still render whole-number axes.

    tabs.jsx:1259, tabs.jsx:1329 and tab_escalation_analytics.jsx:244 pass no
    ``formatY`` at all and all plot integer counts. Their axes were
    integer-labelled only because of the old ``Math.round`` at the label site,
    so deleting that rounding outright would render ``2.5`` / ``7.5`` on all
    three — trading one mislabelled axis for three. The rounding therefore MOVED
    into the default, and these are the exact labels the pre-fix composition
    produced, pinned so the equivalence is checked rather than argued.
    """
    body = _component_body(charts_jsx_body, 'StackedAreaChart')

    axes = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_default_format_y(charts_jsx_body),
            max_vs=[7, 10],
        )
    )

    # maxV = 7  -> raw ticks 0 / 1.75 / 3.5 / 5.25 / 7
    # maxV = 10 -> raw ticks 0 / 2.5  / 5   / 7.5  / 10
    expected = {'7': ['0', '2', '4', '5', '7'], '10': ['0', '3', '5', '8', '10']}
    assert axes == expected, (
        f'the default-formatY integer-count axes render as {axes}, expected '
        f'{expected}. Either the default stopped rounding (fractional labels '
        'like `2.5` would now reach tabs.jsx:1259, tabs.jsx:1329 and '
        'tab_escalation_analytics.jsx:244) or the tick generator changed.'
    )


def test_line_chart_hands_format_y_the_raw_tick(charts_jsx_body: str) -> None:
    """LineChart already handed formatY the raw tick — pin that, change nothing else.

    LineChart's non-rounding default is DELIBERATELY left alone by task 4059;
    the divergence and the caller audit it would need are documented at its
    signature in charts.jsx. This pins only the invariant the two primitives
    share — the value REACHING formatY is the unmodified tick — by executing
    LineChart's own label expression against a recording formatter rather than
    by matching its source text.
    """
    body = _component_body(charts_jsx_body, 'LineChart')

    received = _run_node(f"""
const received = [];
const formatY = v => {{ received.push(v); return String(v); }};
[0, 1.75, 3.5, 5.25, 7].map(({_tick_map_param(body)}, i) => formatY({_tick_label_arg(body)}));
console.log(JSON.stringify(received));
""")

    assert received == [0, 1.75, 3.5, 5.25, 7], (
        f'LineChart handed formatY {received} instead of the raw ticks '
        '[0, 1.75, 3.5, 5.25, 7] — it has regressed into the same pre-rounding '
        'defect StackedAreaChart was just fixed for, and any caller supplying a '
        'percent or decimal formatter would see collapsed gridline labels.'
    )


# ---------------------------------------------------------------------------
# API surface — routing formatCountTick from DF_SPARK_PATH to DF_CHARTS.
# ---------------------------------------------------------------------------


# What a missing hop costs, per routed helper. The two differ in kind:
# formatCountTick fails loudly at its first label, niceCountMax fails SILENTLY,
# because `snapMax={undefined}` falls back to the identity default.
_MISSING_ROUTE_CONSEQUENCE = {
    'formatCountTick': (
        'the count callers render `formatY={undefined}`, LineChart silently '
        'falls back to its own default, and the fractional labels come straight back'
    ),
    'niceCountMax': (
        'the count callers render `snapMax={undefined}`, the identity default '
        'kicks in, and every count axis silently stops snapping'
    ),
}


@pytest.mark.parametrize('helper', ['formatCountTick', 'niceCountMax'])
def test_charts_jsx_routes_count_axis_helper_from_spark_path_to_df_charts(
    charts_jsx_body: str, helper: str
) -> None:
    """charts.jsx must import each count-axis helper and re-export it on DF_CHARTS.

    This is the wiring the caller fix depends on, and it is invisible to every
    other test in the repo: the tabs reach these helpers only through
    `window.DF_CHARTS`, so a missing hop here raises no error anywhere — see
    `_MISSING_ROUTE_CONSEQUENCE` for what each helper's absence renders as.

    The last assertion executes the real spark_path.js, so the name charts.jsx
    binds is proven to EXIST rather than merely to be spelled consistently in
    two files.
    """
    destructured = _spark_path_destructure(charts_jsx_body)
    assert helper in destructured, (
        f'charts.jsx does not destructure `{helper}` off window.DF_SPARK_PATH. '
        f'It binds {sorted(destructured)}. Without it the re-export below is a '
        'reference to an undefined identifier.'
    )

    # The brace-hostile coupling, asserted in its own right so the constraint is
    # discoverable from the failure rather than only from the comment above.
    # It MUST come before `_df_charts_export_names`, which runs the very same
    # `search` and asserts on it internally: ordered the other way this line is
    # unreachable, because the helper always raises first with a message scoped
    # to this module's own read, which does not spell out that the pattern is
    # shared or where else the same nested brace surfaces.
    assert DF_CHARTS_EXPORT_RE.search(charts_jsx_body) is not None, (
        'the window.DF_CHARTS export literal no longer parses under the '
        r'`window\.DF_CHARTS\s*=\s*\{([^{}]*)\}` pattern of '
        '`_dashboard_helpers.py::DF_CHARTS_EXPORT_RE` — something in it grew a '
        'nested brace. That pattern is the single shared object EVERY DF_CHARTS '
        'consumer imports, so this breaks all of them at once: in '
        'test_tab_burndown.py the same miss is silent, and surfaces as an empty '
        'export set and a confusing "not a chart component" error far from the '
        'cause. Every DF_CHARTS export must stay a BARE IDENTIFIER.'
    )

    exported = _df_charts_export_names(charts_jsx_body)
    assert helper in exported, (
        f'charts.jsx does not re-export `{helper}` on window.DF_CHARTS. '
        f'It exports {sorted(exported)}. The tabs have no other route to the '
        f'helper — they never touch DF_SPARK_PATH — so {_MISSING_ROUTE_CONSEQUENCE[helper]}.'
    )

    surface = _spark_path_module_surface()
    assert surface.get(helper) == 'function', (
        f'requiring the real spark_path.js did not yield a `{helper}` function — '
        f'its surface is {surface}. charts.jsx would then destructure `undefined`, '
        f'so {_MISSING_ROUTE_CONSEQUENCE[helper]}.'
    )


@pytest.mark.parametrize(
    ('fixture', 'helper', 'consequence'),
    [
        pytest.param(
            'tabs_jsx_body',
            'formatCountTick',
            'Its MemoryTab reads/writes chart and MergeTab attempt-depth chart both '
            'pass `formatY={formatCountTick}`',
            id='tabs.jsx-formatCountTick',
        ),
        pytest.param(
            'tabs_jsx_body',
            'niceCountMax',
            'Its MemoryTab, MergeTab and both BurnTab count charts pass '
            '`snapMax={niceCountMax}`, so the Memory, Merge and Burndown tabs blank',
            id='tabs.jsx-niceCountMax',
        ),
        pytest.param(
            'tab_overview_jsx_body',
            'niceCountMax',
            'OverviewTab\'s memory-ops chart passes `snapMax={niceCountMax}`, so the '
            'Overview tab blanks — the landing page',
            id='tab_overview.jsx-niceCountMax',
        ),
    ],
)
def test_consumer_binds_count_axis_helper_off_df_charts(
    request: pytest.FixtureRequest, fixture: str, helper: str, consequence: str
) -> None:
    """A consumer that names a helper bare must DESTRUCTURE it, or its call sites throw.

    The last hop of the route, and the only one no other test in the repo can
    see. spark_path.js -> charts.jsx -> DF_CHARTS is pinned above; from there
    the tabs diverge. tab_escalation_analytics.jsx reads `C.<helper>` off a
    namespace binding, so it needs no per-name entry and cannot lose one.
    tabs.jsx and tab_overview.jsx name their helpers in a line-2 destructure,
    and if an entry were dropped every bare reference below becomes a free
    identifier: the shipped page throws a ReferenceError at render.

    The behavioural tests in this file do NOT cover this, and would stay GREEN
    through it: their programs bind the helpers off the real spark_path module
    directly, precisely so they measure what the caller's expression RENDERS
    rather than how the name reached it. Nor does
    test_charts_consumer_bindings.py, which only checks the opposite direction
    (destructured-but-never-used). This is the same silent two-token join the
    routing test above exists to prevent, one file further along.
    """
    bound = _df_charts_destructure(request.getfixturevalue(fixture))
    assert helper in bound, (
        f'{fixture} does not bind `{helper}` in its `const {{ ... }} = '
        f'window.DF_CHARTS` destructure. It binds {sorted(bound)}. {consequence}, '
        'which without this entry is an undefined identifier — a ReferenceError '
        'at render, not a fallback to the default.'
    )

# ---------------------------------------------------------------------------
# Cache-buster floor.
# ---------------------------------------------------------------------------


def test_index_html_cache_buster_not_reverted_below_charts_axis_floor(
    index_html_body: str,
) -> None:
    """Every /static/redux/* asset must stay at or past the floor this fix landed at.

    This is a user-visible RENDERING fix inside a browser-cached static asset:
    an already-open dashboard holds a cached copy of the BROKEN charts.jsx and
    would keep reading a mislabelled "0% / 0% / 100% / 100% / 100%" axis
    indefinitely without a new ``?v=`` — the exact rationale
    test_index_html.py:691-717 documents for its own floor.

    Deliberately asserts only this module's own floor, NOT uniformity: that
    same docstring makes ``test_redux_cache_buster_bumped`` the single home of
    the uniformity check and directs "every other module" to assert its own
    ``min(versions) >= N`` floor, which is sound without uniformity because the
    OLDEST asset is the one that would still serve stale code. Precedents:
    test_esc_flow_diagram.py:324 (>= 33), test_scheduler_page.py:1931 (>= 10).
    """
    versions = {int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)}

    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 44, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        'expected >= 44 (the floor the StackedAreaChart y-axis label fix '
        'landed at). Below that, an already-open dashboard keeps serving the '
        'cached charts.jsx whose Workflow axis reads 0%/0%/100%/100%/100%.'
    )


def test_index_html_cache_buster_not_reverted_below_count_tick_floor(
    index_html_body: str,
) -> None:
    """Every /static/redux/* asset must stay at or past task 4232's own floor.

    Deliberately ADDED alongside the >= 44 floor above rather than replacing it:
    that number records where task 4059's percent-axis fix landed, and folding
    the two together would erase which change each floor defends.

    TWO INDEPENDENT REASONS THIS BUMP IS MANDATORY — the number alone teaches
    nothing, so both are recorded here.

    1. charts.jsx:13-20's own stated rule. Its `const { ... } = window.DF_SPARK_PATH`
       destructure now reaches for `formatCountTick`, a name a CACHED copy of
       spark_path.js does not have. That destructure has no `|| {}` fallback,
       deliberately, so without a new ``?v=`` an already-open dashboard pairs a
       fresh charts.jsx with a stale spark_path.js, throws at charts.jsx LOAD
       time, and blanks every chart on every tab — not just the three this fix
       touches.
    2. This is a user-visible RENDERING change. A cached charts.jsx/tabs.jsx
       would keep drawing 1.75 / 3.5 / 5.25 on the memory and merge axes
       indefinitely — the same rationale as the >= 44 floor above.

    Asserts only THIS module's floor, per the convention
    test_index_html.py:691-717 sets out: uniformity lives solely in
    ``test_redux_cache_buster_bumped`` and the strictly-greater-than-merge-base
    gate in ``test_redux_cache_buster_is_newer_than_merge_base``.  A bare floor
    is sound without uniformity because the OLDEST asset is the one that would
    still serve stale code.  Precedents: test_tab_memory_evals.py (>= 43),
    test_esc_flow_diagram.py (>= 33).
    """
    versions = {int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)}

    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 52, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        'expected >= 52 (the floor the integer-count tick-label fix landed at). '
        'Below that, an already-open dashboard pairs a cached spark_path.js '
        'lacking formatCountTick with a charts.jsx that destructures it — which '
        'throws at load and blanks every chart, not merely the three this fix '
        'relabels.'
    )


# ---------------------------------------------------------------------------
# Negative control — mirrors test_charts_null_samples.py:304-336.
# ---------------------------------------------------------------------------


# The StackedAreaChart signature + y-tick <text> element + tick generator
# exactly as they stood before task 4059, verbatim from charts.jsx:150/177-178/186.
_PRE_FIX_SOURCE = """
function StackedAreaChart({ stacks, labels, height = 220, formatY = v => String(v), formatX = v => v }) {
  const padL = 38, padR = 12, padT = 8, padB = 22;
  const yToPx = v => padT + chartH - (v / maxV) * chartH;

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => (maxV * i) / ticks);

  return (
    <div ref={ref} style={{ width: '100%', height }}>
      <svg>
        {yTicks.map((t, i) => (
          <g key={i}>
            <text x={padL - 6} y={yToPx(t) + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(Math.round(t))}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
"""


def test_probes_and_extractors_actually_fire_on_pre_fix_source(tab_analytics_jsx_body: str) -> None:
    """Every probe and extractor above still detects the real pre-fix code.

    Without this, a reformat that inserts a space, a variable rename, or a
    change to the label markup would turn EVERY assertion in this file into a
    permanent false GREEN while the defect sat untouched in charts.jsx. Same
    guard as test_charts_null_samples.py:304-336 and test_index_html.py's own
    control.
    """
    body = _component_body(_PRE_FIX_SOURCE, 'StackedAreaChart')

    assert 'formatY(Math.round(' in body, (
        'the `formatY(Math.round(` probe did NOT fire on a verbatim copy of '
        'the pre-fix StackedAreaChart source. It has gone stale, which makes '
        'test_stacked_area_chart_does_not_pre_round_the_tick_before_format_y a '
        'permanent false GREEN.'
    )
    assert _tick_label_arg(body) == 'Math.round(t)', (
        'the tick-label-argument extractor no longer reads `Math.round(t)` out '
        f'of the pre-fix source — it returned {_tick_label_arg(body)!r}.'
    )
    assert _default_format_y(_PRE_FIX_SOURCE) == 'v => String(v)', (
        'the default-formatY extractor no longer reads `v => String(v)` out of '
        f'the pre-fix signature — it returned {_default_format_y(_PRE_FIX_SOURCE)!r}.'
    )

    # And the executable pipeline must still reproduce the user-visible defect
    # when fed pre-fix source, proving the node harness itself measures the bug.
    labels = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_workflow_panel_format_y(tab_analytics_jsx_body),
            max_vs=[1],
        )
    )['1']
    assert labels == ['0%', '0%', '100%', '100%', '100%'], (
        'the node harness did NOT reproduce the pre-fix Workflow axis '
        f'(got {labels}, expected the defective 0%/0%/100%/100%/100%). The '
        'harness no longer measures the defect, so '
        'test_workflow_panel_percent_axis_reads_0_25_50_75_100 could pass for '
        'reasons unrelated to the fix.'
    )

    # The pre-fix default composed with the pre-fix (pre-rounded) label
    # argument must reproduce the SAME integer-count axes the post-fix default
    # produces from the raw tick — the equivalence that lets the three
    # no-formatY callers keep byte-identical axes.
    axes = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_default_format_y(_PRE_FIX_SOURCE),
            max_vs=[7, 10],
        )
    )
    assert axes == {'7': ['0', '2', '4', '5', '7'], '10': ['0', '3', '5', '8', '10']}, (
        f'the pre-fix default-formatY composition renders {axes}, so the values '
        'pinned by test_default_format_y_callers_keep_integer_count_axes are no '
        'longer the pre-fix ones and that test no longer proves the '
        'no-formatY callers kept their axes.'
    )


# ---------------------------------------------------------------------------
# The caller audit, pinned executably (task 4232).
#
# Four LineChart call sites pass no formatY and inherit its default. Three plot
# integer COUNTS and must gain `formatCountTick`; the fourth plots a genuine
# RATIO and must keep the raw default. That split is the whole reason this fix
# is a per-caller wiring change rather than a new default — so it is pinned in
# both directions, with a frozen rounding-default control proving the ratio
# guard actually fires.
# ---------------------------------------------------------------------------

# maxV=7 is the fractional-label case from this task's title (the default draws
# 1.75 / 3.5 / 5.25); maxV=1 is the `plottableMax(values, 1)` seed floor that
# every idle count series sits at, and the case a ROUNDING helper would render
# as the duplicate 0/0/1/1/1.
#
# These maxima are fed straight to the tick generator, BYPASSING snapMax (task
# 5121). So the three caller tests below now pin the formatY wiring as defence
# in depth, not what the snapped chart draws; the snapped axes are pinned by
# test_count_axis_call_site_snaps_its_maximum.
_COUNT_AXIS_MAX_VS = [7, 1]
_COUNT_AXIS_EXPECTED = {'7': ['0', '', '', '', '7'], '1': ['0', '', '', '', '1']}


def test_memory_tab_reads_writes_axis_labels_whole_counts_only(
    charts_jsx_body: str, tabs_jsx_body: str
) -> None:
    """MemoryTab's reads-vs-writes axis is a COUNT axis and must read as one.

    `write_journal.get_memory_ops` is a SQL ``COUNT(*)`` bucketed per
    hour, so a "3.5" gridline label is not a rounding nicety — it is a count
    that cannot exist.
    """
    axes = _render_caller_axis(
        charts_jsx_body, _component_body(tabs_jsx_body, 'MemoryTab'), 'MEMORY_OPS.reads', _COUNT_AXIS_MAX_VS
    )
    assert axes == _COUNT_AXIS_EXPECTED, (
        f'MemoryTab reads/writes renders {axes}; expected {_COUNT_AXIS_EXPECTED}. '
        'It is still inheriting LineChart\'s raw default (1.75/3.5/5.25 at '
        'maxV=7) instead of passing formatY={formatCountTick}.'
    )


def test_merge_tab_attempt_axis_labels_whole_counts_only(
    charts_jsx_body: str, tabs_jsx_body: str
) -> None:
    """MergeTab's "Merge attempts · 15-min buckets" axis counts events per bucket."""
    axes = _render_caller_axis(
        charts_jsx_body,
        _component_body(tabs_jsx_body, 'MergeTab'),
        'd.depth.values',
        _COUNT_AXIS_MAX_VS,
    )
    assert axes == _COUNT_AXIS_EXPECTED, (
        f'MergeTab merge attempts renders {axes}; expected {_COUNT_AXIS_EXPECTED}.'
    )


def test_workflow_panel_churn_axis_labels_whole_counts_only(
    charts_jsx_body: str, tab_analytics_jsx_body: str
) -> None:
    """WorkflowPanel's churn axis counts same-task re-filings per day."""
    axes = _render_caller_axis(
        charts_jsx_body,
        _component_body(tab_analytics_jsx_body, 'WorkflowPanel'),
        'churnDaily',
        _COUNT_AXIS_MAX_VS,
    )
    assert axes == _COUNT_AXIS_EXPECTED, (
        f'WorkflowPanel churn renders {axes}; expected {_COUNT_AXIS_EXPECTED}.'
    )


def test_esc_per_done_ratio_axis_keeps_its_exact_fractions(
    charts_jsx_body: str, tab_analytics_jsx_body: str
) -> None:
    """THE CALLER THAT MAKES A ROUNDING DEFAULT UNSAFE — do not "fix" this axis.

    WorkflowPanel's escalations-per-task-done chart plots
    ``escalation_analytics.py::_esc_per_done``'s ``filings / done`` — a genuine
    float (``None`` when ``done == 0``). It is the fourth of the four call sites
    that inherit LineChart's default, and the reason task 4232 wires a helper
    per caller instead of rounding by default: a rounding default would collapse
    this axis to 0/0/0/1/1, re-filing task 4059's own defect one primitive over.

    So it must pass NO count formatter, and its labels must stay the exact
    fractions the raw default produces. The 0.6000000000000001 is not a typo:
    ``(0.8 * 3) / 4`` is genuinely that double, and it is pinned as what node
    actually prints rather than as what the arithmetic ought to look like.
    """
    body = _component_body(tab_analytics_jsx_body, 'WorkflowPanel')
    assert _line_chart_format_y(body, 'row.ratio') is None, (
        'the escalations-per-done chart has gained a formatY. If that is '
        '`formatCountTick`, its axis now blanks every non-integer gridline on a '
        'series whose values are almost never integers — the ratio would be '
        'unreadable. This chart is a FRACTION and must keep the raw default.'
    )

    axes = _render_caller_axis(charts_jsx_body, body, 'row.ratio', [0.8, 1])
    assert axes == {
        '0.8': ['0', '0.2', '0.4', '0.6000000000000001', '0.8'],
        '1': ['0', '0.25', '0.5', '0.75', '1'],
    }, (
        f'the esc-per-done ratio axis renders {axes} — its intermediate '
        'gridlines are no longer the exact tick values. Something rounded or '
        'blanked a ratio axis.'
    )


# ---------------------------------------------------------------------------
# Negative control for the caller audit — mirrors _PRE_FIX_SOURCE above.
# ---------------------------------------------------------------------------


# LineChart's scale/tick/label lines verbatim from charts.jsx, with ONE change:
# the formatY default rounds. This is option (a), the rounding default this task
# rejected — kept executable so the guard above is discriminating rather than
# vacuous.
_ROUNDING_DEFAULT_LINE_CHART = """
function LineChart({ series, labels, height = 220, yLabel, formatY = (v) => String(Math.round(v)), formatX = (v) => v }) {
  const maxV = plottableMax(all, 1);
  const minV = 0;
  const range = maxV - minV || 1;
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minV + (range * i) / ticks);
  return (
    <svg>
      {yTicks.map((t, i) => (
        <g key={i}>
          <text x={padL - 6} y={y + 3} fontSize="9" fill={PALETTE.fg3} textAnchor="end" fontFamily="JetBrains Mono">{formatY(t)}</text>
        </g>
      ))}
    </svg>
  );
}
"""

# The MergeTab call site exactly as it stood BEFORE this fix — no formatY at all.
_PRE_FIX_COUNT_CALL_SITE = """
<div className="panel-body"><LC labels={d.depth.labels.map(String)} series={[{ values: d.depth.values, color: CP.accent }]} height={180} formatX={window.DF_SHELL.fmtDateTime} /></div>
"""


def test_rounding_default_control_shows_the_ratio_guard_actually_fires() -> None:
    """The rejected option (a) really does wreck the ratio axis, measured.

    Without this, `test_esc_per_done_ratio_axis_keeps_its_exact_fractions` would
    be a guard against a danger no one had ever demonstrated — and the design
    decision it defends (default stays non-rounding) would read as taste. Here
    the rounding default is executed against the same real tick shape and shown
    to produce the collapse.

    Also checks `_line_chart_format_y` still returns None on a verbatim pre-fix
    count call site: that None is what routes a caller to the default, so an
    extractor that silently found nothing would make all three count assertions
    above measure the default forever and pass for the wrong reason.
    """
    frozen = _component_body(_ROUNDING_DEFAULT_LINE_CHART, 'LineChart')
    rounding_default = _default_format_y(_ROUNDING_DEFAULT_LINE_CHART, 'LineChart')
    assert rounding_default == '(v) => String(Math.round(v))', (
        'the default-formatY extractor no longer reads the rounding default out '
        f'of the frozen control — it returned {rounding_default!r}.'
    )

    axes = _run_node(_line_chart_axis_program(frozen, rounding_default, [0.8, 1]))
    assert axes['0.8'] == ['0', '0', '0', '1', '1'], (
        'a rounding default no longer collapses the esc-per-done ratio axis '
        f'(got {axes["0.8"]}). The control has gone stale, so the ratio guard '
        'above no longer defends against anything.'
    )
    assert axes['1'] == ['0', '0', '1', '1', '1'], (
        'a rounding default no longer duplicates labels at the maxV=1 seed '
        f'floor (got {axes["1"]}) — the task-4059 shape this fix declined to '
        'reproduce.'
    )

    assert _line_chart_format_y(_PRE_FIX_COUNT_CALL_SITE, 'd.depth.values') is None, (
        'the caller extractor no longer reports "no formatY" on a verbatim '
        'pre-fix count call site, so it can no longer tell a wired caller from '
        'an unwired one.'
    )


# tabs.jsx's line-2 DF_CHARTS destructure exactly as it stood BEFORE step 6
# wired the count formatter, and a DF_CHARTS export literal that grew a nested
# brace — the two shapes the routing guards above are supposed to catch.
_PRE_FIX_TABS_DESTRUCTURE = (
    'const { Sparkline: SP, LineChart: LC, StackedAreaChart: SA, BarChart: BC, '
    'HBarChart: HBC, Donut: DN, StatTile: ST, PALETTE: CP, deriveVelocitySeries, '
    'defaultSmoothingForWindow, smoothingLabelToSeconds, SMOOTHING_OPTIONS } = '
    'window.DF_CHARTS;'
)
_NESTED_BRACE_EXPORT = 'window.DF_CHARTS = { LineChart, formatCountTick, PALETTE: { fg3 } };'


def test_routing_guards_actually_fire_on_pre_fix_and_nested_brace_source() -> None:
    """The two source-text guards added for the routing hops are discriminating.

    Both assert on ABSENCE, which is the shape that rots silently: an extractor
    that stopped matching would report "no formatCountTick" as "nothing to
    check" and pass forever. So each is run against a frozen source in which it
    must report the bad answer.

    Also pins the ORDERING fix in the routing test: `DF_CHARTS_EXPORT_RE` is
    what fails on a nested-brace literal, and `_df_charts_export_names` raises on
    the identical `search`, so only the standalone assertion placed BEFORE that
    call can ever be the one that spells out the shared-pattern coupling in its
    message.
    """
    pre_fix = _df_charts_destructure(_PRE_FIX_TABS_DESTRUCTURE)
    assert 'LineChart' in pre_fix, (
        f'the DF_CHARTS destructure extractor misread the frozen pre-fix tabs.jsx '
        f'binding list — it returned {sorted(pre_fix)}, which does not even '
        'contain LineChart, so it is not reading binding names at all.'
    )
    assert 'formatCountTick' not in pre_fix, (
        'the DF_CHARTS destructure extractor reports `formatCountTick` bound in a '
        'source that predates the wiring, so it can no longer tell a wired '
        'consumer from an unwired one and its assertion is vacuous.'
    )
    assert 'niceCountMax' not in pre_fix, (
        'the DF_CHARTS destructure extractor reports `niceCountMax` bound in a '
        'source that predates the wiring — see the formatCountTick assertion above.'
    )

    assert DF_CHARTS_EXPORT_RE.search(_NESTED_BRACE_EXPORT) is None, (
        'the brace-hostile export pattern now matches a literal containing a '
        'nested brace, so NO DF_CHARTS consumer would notice one being '
        'introduced: they all read the single shared '
        '`_dashboard_helpers.py::DF_CHARTS_EXPORT_RE`, and test_tab_burndown.py '
        'would go on to fail opaquely on an empty export set.'
    )


# ---------------------------------------------------------------------------
# Count-axis snapping (task 5121) — the axis MAXIMUM, executed.
#
# `snapMax` maps the folded data maximum to the axis maximum. The default is
# the identity, so a chart that does not opt in keeps its exact geometry; with
# niceCountMax the maximum snaps up to a multiple of the tick count and every
# raw tick is whole. Every value below is an exact binary fraction, so `==` is
# a real comparison.
# ---------------------------------------------------------------------------

_SNAPPED_COUNT_TICKS = {'7': [0, 2, 4, 6, 8], '0': [0, 1, 2, 3, 4], '999': [0, 250, 500, 750, 1000]}
_UNSNAPPED_TICKS = {'7': [0, 1.75, 3.5, 5.25, 7], '2.5': [0, 0.625, 1.25, 1.875, 2.5]}


def _assert_axis_max_is_snap_max(charts_jsx_body: str, component: str, program) -> None:
    """``component``'s axis maximum is ``snapMax`` of the folded data maximum, default included."""
    body = _component_body(charts_jsx_body, component)

    snapped = _run_node(program(body, 'niceCountMax', [7, 0, 999]))
    assert snapped == _SNAPPED_COUNT_TICKS, (
        f'{component} with snapMax={{niceCountMax}} draws raw ticks {snapped}, '
        f'expected {_SNAPPED_COUNT_TICKS}. Its axis maximum does not go through '
        'snapMax, so a count axis still takes its maximum as given.'
    )

    default = _signature_default(charts_jsx_body, component, 'snapMax')
    unsnapped = _run_node(program(body, default, [7, 2.5]))
    assert unsnapped == _UNSNAPPED_TICKS, (
        f'{component} with its default snapMax ({default}) draws raw ticks '
        f'{unsnapped}, expected {_UNSNAPPED_TICKS}. An axis that does not opt in '
        'must keep its exact geometry.'
    )


def test_line_chart_axis_max_is_its_snap_max_of_the_folded_data_max(charts_jsx_body: str) -> None:
    """LineChart's gridlines and points share one maxV, so that line is the whole snap.

    Data max 0 exercises the ``plottableMax(all, 1)`` seed floor: 1 snaps to 4.
    """
    _assert_axis_max_is_snap_max(charts_jsx_body, 'LineChart', _line_chart_scaled_axis_program)


def test_stacked_area_chart_axis_max_is_its_snap_max_of_the_folded_stack_max(charts_jsx_body: str) -> None:
    """StackedAreaChart's snap runs inside the real ``stackedAreaPaths``, before any band is scaled."""
    _assert_axis_max_is_snap_max(charts_jsx_body, 'StackedAreaChart', _stacked_area_scaled_axis_program)


# LineChart's signature and axis lines verbatim from before task 5121: no
# snapMax, and `const ticks = 4;` declared after the scale.
_PRE_SNAP_LINE_CHART = """
function LineChart({ series, labels, height = 220, yLabel, formatY = (v) => String(v), formatX = (v) => v }) {
  const all = series.flatMap(s => s.values);
  const maxV = plottableMax(all, 1);
  const minV = 0;
  const range = maxV - minV || 1;
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minV + (range * i) / ticks);
}
"""

# The snap line reading `ticks` ABOVE its declaration: a TDZ ReferenceError at
# render that blanks the chart.
_TDZ_LINE_CHART = """
function LineChart({ series, labels, height = 220, yLabel, formatY = (v) => String(v), formatX = (v) => v, snapMax = (dataMax) => dataMax }) {
  const all = series.flatMap(s => s.values);
  const maxV = snapMax(plottableMax(all, 1), ticks);
  const minV = 0;
  const range = maxV - minV || 1;
  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minV + (range * i) / ticks);
}
"""


def test_scaled_axis_harness_is_discriminating() -> None:
    """The scaled-axis harness can fail: it neither invents a snap nor hides a TDZ.

    Mirrors `_PRE_FIX_SOURCE`'s role above. (a) A maxV line that ignores
    snapMax renders unsnapped ticks even when handed niceCountMax, so the
    snapped expectations above are not produced by the harness itself. (b)
    Emitting the statements in source order reproduces the ordering bug as the
    browser's own ReferenceError.
    """
    pre_snap = _run_node(
        _line_chart_scaled_axis_program(_component_body(_PRE_SNAP_LINE_CHART, 'LineChart'), 'niceCountMax', [7])
    )
    assert pre_snap == {'7': [0, 1.75, 3.5, 5.25, 7]}, (
        f'the harness drew {pre_snap} for a LineChart whose maxV line ignores '
        'snapMax. It is snapping on its own, so the snapped expectations prove '
        'nothing about charts.jsx.'
    )

    with pytest.raises(AssertionError, match='before initialization'):
        _run_node(
            _line_chart_scaled_axis_program(_component_body(_TDZ_LINE_CHART, 'LineChart'), 'niceCountMax', [7])
        )


# ---------------------------------------------------------------------------
# The count-axis snapping caller audit (task 5121), pinned in both directions.
#
# A COUNT axis must pass snapMax, so its maximum snaps to a multiple of the tick
# count. A FRACTION axis must not: a snapped 1 is 4, so a percent axis reads
# 0%..400%. The duration (fmtMs) and dollar charts are deliberately unpinned.
# ---------------------------------------------------------------------------

_COUNT_AXIS_SITES = [
    pytest.param('tabs_jsx_body', 'MemoryTab', 'LineChart', _LINE_CHART_TAG_RE, 'MEMORY_OPS.reads', id='MemoryTab-reads-writes'),
    pytest.param('tabs_jsx_body', 'MergeTab', 'LineChart', _LINE_CHART_TAG_RE, 'd.depth.values', id='MergeTab-attempt-depth'),
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', 'LineChart', _LINE_CHART_TAG_RE, 'churnDaily', id='WorkflowPanel-churn'),
    pytest.param('tab_overview_jsx_body', 'OverviewTab', 'LineChart', _LINE_CHART_TAG_RE, 'D.MEMORY_OPS.reads', id='OverviewTab-memory-ops'),
    pytest.param('tab_analytics_jsx_body', 'OriginPanel', 'StackedAreaChart', _STACKED_AREA_TAG_RE, 'stacks={stacks}', id='OriginPanel-filings-by-source'),
    pytest.param('tabs_jsx_body', 'BurnTab', 'StackedAreaChart', _STACKED_AREA_TAG_RE, 'burndownStacks(b,', id='BurnTab-aggregate'),
    pytest.param('tabs_jsx_body', 'BurnTab', 'StackedAreaChart', _STACKED_AREA_TAG_RE, 'burndownStacks(pb,', id='BurnTab-per-project'),
]

_FRACTION_AXIS_SITES = [
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', 'LineChart', _LINE_CHART_TAG_RE, 'row.ratio', id='WorkflowPanel-esc-per-done-ratio'),
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', 'StackedAreaChart', _STACKED_AREA_TAG_RE, 'labels={weeks}', id='WorkflowPanel-100pct-stack'),
    pytest.param('tab_analytics_jsx_body', 'LifespanPanel', 'LineChart', _LINE_CHART_TAG_RE, 'gridLabels', id='LifespanPanel-ECDF-percent'),
]


@pytest.mark.parametrize(('fixture', 'caller', 'component', 'tag_re', 'anchor'), _COUNT_AXIS_SITES)
def test_count_axis_call_site_snaps_its_maximum(
    request: pytest.FixtureRequest, charts_jsx_body: str, fixture: str, caller: str, component: str, tag_re, anchor: str
) -> None:
    """Every count axis snaps its maximum, so every one of its ticks is whole."""
    caller_body = _component_body(request.getfixturevalue(fixture), caller)
    assert _chart_prop(caller_body, tag_re, anchor, 'snapMax') is not None, (
        f'{caller}\'s {component} anchored at {anchor!r} passes no snapMax, so it '
        'still takes its count-axis maximum as given: at a data max of 7 it '
        'draws gridlines at 1.75 / 3.5 / 5.25 and labels only the floor and peak.'
    )

    ticks = _render_snapped_caller_axis(charts_jsx_body, component, caller_body, tag_re, anchor, [7, 0, 999])
    assert ticks == _SNAPPED_COUNT_TICKS, (
        f'{caller}\'s {component} anchored at {anchor!r} draws raw ticks {ticks}, '
        f'expected {_SNAPPED_COUNT_TICKS}. Its snapMax does not snap to a multiple '
        'of the tick count.'
    )


@pytest.mark.parametrize(('fixture', 'caller', 'component', 'tag_re', 'anchor'), _FRACTION_AXIS_SITES)
def test_fraction_axis_call_site_does_not_snap(
    request: pytest.FixtureRequest, charts_jsx_body: str, fixture: str, caller: str, component: str, tag_re, anchor: str
) -> None:
    """A fraction axis keeps its exact maximum: snapped, a 0..1 axis would read 0%..400%."""
    caller_body = _component_body(request.getfixturevalue(fixture), caller)
    assert _chart_prop(caller_body, tag_re, anchor, 'snapMax') is None, (
        f'{caller}\'s {component} anchored at {anchor!r} has gained a snapMax. It '
        'plots a FRACTION: niceCountMax snaps 1 up to 4, so its axis would read '
        '0%..400% (or 0..4 for a ratio) with the data squashed into the bottom quarter.'
    )

    ticks = _render_snapped_caller_axis(charts_jsx_body, component, caller_body, tag_re, anchor, [1, 2.5])
    expected = {'1': [0, 0.25, 0.5, 0.75, 1], '2.5': [0, 0.625, 1.25, 1.875, 2.5]}
    assert ticks == expected, (
        f'{caller}\'s {component} anchored at {anchor!r} draws raw ticks {ticks}, '
        f'expected {expected}. A fraction axis no longer keeps its exact maximum.'
    )
