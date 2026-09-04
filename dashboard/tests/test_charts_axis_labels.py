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
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from _dashboard_helpers import extract_function_body

# ---------------------------------------------------------------------------
# The served-asset fixtures (`charts_jsx_body`, `tab_analytics_jsx_body`,
# `index_html_body`) and the `_client` they read through now live in
# conftest.py (task 3549).  Reading through the app's own static route rather
# than the filesystem still proves the asset is actually SERVED, which is the
# property the cache-buster assertion below depends on.
# ---------------------------------------------------------------------------


def _extract_signature(src: str, fn_name: str) -> str:
    """Return the parameter-list text of ``function <fn_name>(...)``, parens excluded."""
    m = re.search(rf'\bfunction\s+{re.escape(fn_name)}\s*\(', src)
    assert m is not None, (
        f'no `function {fn_name}(` declaration found in the source — the '
        'component was renamed or converted to another declaration form, and '
        'every assertion in this file would go vacuously GREEN.'
    )
    paren_depth = 1
    i = m.end()
    while i < len(src) and paren_depth > 0:
        if src[i] == '(':
            paren_depth += 1
        elif src[i] == ')':
            paren_depth -= 1
        i += 1
    assert paren_depth == 0, f'unbalanced parameter list for `function {fn_name}(`'
    return src[m.end() : i - 1]


# ---------------------------------------------------------------------------
# Extractors.  Each asserts LOUDLY when its regex misses — a silent '' would
# make a rename turn this whole file into a permanent false GREEN.  That is
# also why `_component_body` needs no guard of its own: `extract_function_body`
# raises rather than returning '' (task 3549).
# ---------------------------------------------------------------------------


def _component_body(src: str, name: str) -> str:
    return extract_function_body(src, name)


def _default_format_y(charts_jsx_body: str, component: str = 'StackedAreaChart') -> str:
    """The ``formatY = ...`` default from ``component``'s signature.

    The default arrow contains neither a comma nor a brace (``v => String(v)``
    before the fix, ``v => String(Math.round(v))`` after), so a ``[^,}]+`` run
    over the signature slice captures exactly it. ``component`` defaults to
    StackedAreaChart, whose default task 4059 moved the rounding into; task
    4232's caller audit passes ``'LineChart'`` to read the default the four
    no-formatY callers actually inherit.
    """
    signature = _extract_signature(charts_jsx_body, component)
    m = re.search(r'formatY\s*=\s*([^,}]+)', signature)
    assert m is not None, (
        f'{component} no longer declares a `formatY = <default>` in its '
        f'signature. Signature was: {signature!r}'
    )
    return m.group(1).strip()


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
# Deliberately the SAME brace-hostile pattern as test_tab_burndown.py:53's
# `_DF_CHARTS_EXPORT_RE`, copied verbatim rather than imported (this repo's test
# modules do not import each other).  Asserted on explicitly below so that a
# nested `{}` in the export literal fails HERE, naming the coupling, instead of
# there as an opaque "could not parse the DF_CHARTS exports".
_DF_CHARTS_EXPORT_RE = re.compile(r'window\.DF_CHARTS\s*=\s*\{([^{}]*)\}')

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
    return {part.split(':', 1)[0].strip() for part in brace_body.split(',') if part.strip()}


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
    m = _DF_CHARTS_EXPORT_RE.search(charts_jsx_body)
    assert m is not None, (
        'could not read the `window.DF_CHARTS = { ... }` export literal in '
        'charts.jsx. The overwhelmingly likely cause is a NESTED BRACE inside '
        'that literal: the pattern is `[^{}]*` by design, and the identical '
        'pattern in test_tab_burndown.py:53 would silently yield an empty set '
        'and fail test_every_labels_prop_sits_on_a_chart_component with an '
        'unrelated-looking message. Keep every export a bare identifier.'
    )
    return _binding_names(m.group(1))


def _spark_path_module_surface() -> dict:
    """``{name: typeof}`` for the REAL shipped spark_path.js, executed under node.

    Proves the name charts.jsx binds actually EXISTS at the source, rather than
    merely being spelled the same in two files — the failure mode a pure
    source-text pairing cannot see.
    """
    return _run_node(
        'const api = require(%s);\n'
        'console.log(JSON.stringify(Object.fromEntries('
        'Object.keys(api).map(k => [k, typeof api[k]]))));'
        % json.dumps(str(_SPARK_PATH_JS))
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

# Both spellings the two tabs use: tabs.jsx aliases `LineChart: LC` in its
# line-2 DF_CHARTS destructure, tab_escalation_analytics.jsx goes through its
# `const C = window.DF_CHARTS` namespace.
_LINE_CHART_TAG_RE = re.compile(r'<(?:LC|C\.LineChart)\b')


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


def _line_chart_format_y(body: str, anchor: str):
    """The `formatY={...}` expression of the LineChart call site containing `anchor`.

    Returns the expression text, or ``None`` when the site passes no formatY at
    all and therefore inherits LineChart's default. `body` is an already-sliced
    COMPONENT body: several tabs render more than one LineChart, and the anchor
    (a series expression such as ``ts.reads``) is what names the axis.
    """
    matches = [el for el in _jsx_elements(body, _LINE_CHART_TAG_RE) if anchor in el]
    assert len(matches) == 1, (
        f'expected exactly one <LC>/<C.LineChart> element whose props mention '
        f'{anchor!r}, found {len(matches)}. The chart was renamed, removed, or '
        'its series expression changed — either way the caller-audit assertion '
        'that depends on it must FAIL rather than quietly measure a different '
        'chart (or nothing at all).'
    )
    element = matches[0]
    m = re.search(r'formatY=\{', element)
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
    assert depth == 0, f'unbalanced formatY={{...}} in the element anchored at {anchor!r}'
    return element[m.end() : i - 1].strip()


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
    test_charts_jsx_routes_format_count_tick_from_spark_path_to_df_charts pins
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


def test_charts_jsx_routes_format_count_tick_from_spark_path_to_df_charts(
    charts_jsx_body: str,
) -> None:
    """charts.jsx must import `formatCountTick` and re-export it on DF_CHARTS.

    This is the wiring the caller fix depends on, and it is invisible to every
    other test in the repo: tabs.jsx and tab_escalation_analytics.jsx reach the
    helper only through `window.DF_CHARTS`, so a missing hop here renders as
    `formatY={undefined}` — LineChart silently falls back to its own default and
    the fractional labels come straight back, with no error anywhere.

    The last assertion executes the real spark_path.js, so the name charts.jsx
    binds is proven to EXIST rather than merely to be spelled consistently in
    two files.
    """
    destructured = _spark_path_destructure(charts_jsx_body)
    assert 'formatCountTick' in destructured, (
        'charts.jsx does not destructure `formatCountTick` off window.DF_SPARK_PATH. '
        f'It binds {sorted(destructured)}. Without it the re-export below is a '
        'reference to an undefined identifier.'
    )

    exported = _df_charts_export_names(charts_jsx_body)
    assert 'formatCountTick' in exported, (
        'charts.jsx does not re-export `formatCountTick` on window.DF_CHARTS. '
        f'It exports {sorted(exported)}. tabs.jsx and tab_escalation_analytics.jsx '
        'have no other route to the helper — they never touch DF_SPARK_PATH.'
    )

    # The brace-hostile coupling, asserted in its own right so the constraint is
    # discoverable from the failure rather than only from the comment above.
    assert _DF_CHARTS_EXPORT_RE.search(charts_jsx_body) is not None, (
        'the window.DF_CHARTS export literal no longer parses under the '
        r'`window\.DF_CHARTS\s*=\s*\{([^{}]*)\}` pattern that '
        'test_tab_burndown.py:53 also uses — something in it grew a nested '
        'brace. There it fails as an empty export set and a confusing '
        '"not a chart component" error far from the cause. Every DF_CHARTS '
        'export must stay a BARE IDENTIFIER.'
    )

    surface = _spark_path_module_surface()
    assert surface.get('formatCountTick') == 'function', (
        'requiring the real spark_path.js did not yield a `formatCountTick` '
        f'function — its surface is {surface}. charts.jsx would then destructure '
        '`undefined` and hand it to LineChart as formatY, which throws on the '
        'first tick and blanks the entire chart.'
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
_COUNT_AXIS_MAX_VS = [7, 1]
_COUNT_AXIS_EXPECTED = {'7': ['0', '', '', '', '7'], '1': ['0', '', '', '', '1']}


def test_memory_tab_reads_writes_axis_labels_whole_counts_only(
    charts_jsx_body: str, tabs_jsx_body: str
) -> None:
    """MemoryTab's reads-vs-writes axis is a COUNT axis and must read as one.

    `write_journal.get_memory_timeseries` is a SQL ``COUNT(*)`` bucketed per
    hour, so a "3.5" gridline label is not a rounding nicety — it is a count
    that cannot exist.
    """
    axes = _render_caller_axis(
        charts_jsx_body, _component_body(tabs_jsx_body, 'MemoryTab'), 'ts.reads', _COUNT_AXIS_MAX_VS
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
