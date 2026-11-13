"""Behavioural contract: StackedAreaChart hands formatY the RAW y-axis tick.

WHY THIS FILE MIXES SOURCE-EXTRACTION WITH A NODE SUBPROCESS — charts.jsx is
JSX transformed by Babel, so node cannot import its component bodies, and when
this file was written they could not be rendered by React in any harness here.
They now can: ``dashboard/tests/js/dom_harness.mjs`` renders them in jsdom, and
the count-axis behaviour of both charts is tested that way in
``dashboard/tests/js/jsx_render.test.mjs`` (Decision 9).  The repo's usual
answer for executable chart math is extraction into a plain-JS classic script
(``spark_path.js`` +
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

ALSO HERE: the esc-per-done ratio guard that keeps LineChart's formatY default
non-rounding (task 4232), and the two-direction ``countAxis`` caller audit
(task 6370) — every count-axis chart call site declares ``countAxis`` and no
fraction-axis site does.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from _jsx_source_helpers import (
    extract_function_body,
    find_function_params,
    jsx_open_tags,
    jsx_prop_expr,
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
# its own any more: both are projections over shared `_jsx_source_helpers`
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
    (``v => String(Math.round(v))``, ``(v) => String(v)``), so a ``[^,}]+``
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


# ---------------------------------------------------------------------------
# Caller-audit extractors — the props (formatY) and flags (countAxis) that each
# LineChart and StackedAreaChart CALL SITE passes, read out of the tab sources.
# Each call site is read from its own opening tag, spanned by
# `_jsx_source_helpers.py::jsx_open_tag_end`, so a quoted `}` or `/>` cannot end
# it and a child element's attributes are never the chart's. A prop's
# expression is read out of that tag by `_jsx_source_helpers.py::jsx_prop_expr`.
#
# `_line_chart_format_y` returns None for a site that passes no formatter. That
# None is load-bearing for the esc-per-done ratio guard, which requires the site
# to keep the default, so a MISSING ELEMENT asserts instead of also returning None.
# ---------------------------------------------------------------------------

# Every spelling the tabs use: tabs.jsx aliases `LineChart: LC` (and
# `StackedAreaChart: SA`) in its line-2 DF_CHARTS destructure,
# tab_escalation_analytics.jsx goes through its `const C = window.DF_CHARTS`
# namespace, and tab_overview.jsx binds `LineChart` unaliased.
_LINE_CHART_TAG_RE = re.compile(r'<(?:LC|LineChart|C\.LineChart)\b')
_STACKED_AREA_TAG_RE = re.compile(r'<(?:SA|C\.StackedAreaChart)\b')


def _anchored_chart_tag(body: str, tag_re: re.Pattern, anchor: str) -> str:
    """The OPENING TAG, where every prop lives, of the one chart call site in
    `body` whose tag matches `tag_re` and whose props mention `anchor`.

    `body` is an already-sliced COMPONENT body: several tabs render more than
    one chart, and the anchor (a series expression such as ``MEMORY_OPS.reads``)
    is what names the axis.
    """
    matches = [tag for tag in jsx_open_tags(body, tag_re) if anchor in tag]
    assert len(matches) == 1, (
        f'expected exactly one chart tag matching {tag_re.pattern!r} whose '
        f'props mention {anchor!r}, found {len(matches)}. The chart was renamed, '
        'removed, or its series expression changed — either way the caller-audit '
        'assertion that depends on it must FAIL rather than quietly measure a '
        'different chart (or nothing at all).'
    )
    return matches[0]


def _chart_prop(body: str, tag_re: re.Pattern, anchor: str, prop: str):
    """The `<prop>={...}` expression of the chart call site containing `anchor`.

    Returns the expression text, or ``None`` when the site does not pass `prop`
    and therefore inherits the component's default.
    """
    return jsx_prop_expr(_anchored_chart_tag(body, tag_re, anchor), prop)


def _attribute_names_only(opening_tag: str) -> str:
    """`opening_tag` with every attribute VALUE blanked to spaces, offsets preserved.

    Blanks braced expressions (``series={[{ countAxis: 1 }]}``) and quoted
    strings at brace depth 0 (``title="countAxis"``), keeping their delimiters,
    so what is left to scan is the tag, the attribute names and the ``=`` signs.
    """
    depth = 0
    quote = None
    kept = []
    for ch in opening_tag:
        if quote is not None:
            kept.append(ch if ch == quote else ' ')
            if ch == quote:
                quote = None
        elif depth == 0 and ch in '"\'':
            quote = ch
            kept.append(ch)
        else:
            if ch == '{':
                depth += 1
            kept.append(ch if depth == 0 else ' ')
            if ch == '}':
                depth -= 1
    return ''.join(kept)


def _chart_flag(body: str, tag_re: re.Pattern, anchor: str, prop: str) -> bool:
    """Whether the chart call site containing `anchor` carries the bare attribute `prop`.

    Only the shorthand form counts (``<LC ... countAxis />``), so presence is the
    whole decision. Any ``prop=`` form RAISES rather than answering: read either
    way, ``countAxis={false}`` or ``countAxis={isCount}`` would be misread. Only
    attribute names are scanned (see `_attribute_names_only`), so the token
    inside a braced expression or a quoted string is not an attribute.
    """
    attributes = _attribute_names_only(_anchored_chart_tag(body, tag_re, anchor))
    assert re.search(rf'(?<![\w.]){prop}\s*=', attributes) is None, (
        f'the chart tag anchored at {anchor!r} assigns `{prop}=...`. The '
        f'caller audit requires the shorthand `{prop}` attribute, so that its '
        'presence is the whole decision; an assigned value could be false or '
        'computed, and would be misread as present or absent.'
    )
    return re.search(rf'\s{prop}(?=[\s/>])', attributes) is not None


def _line_chart_format_y(body: str, anchor: str):
    """The `formatY={...}` of the LineChart call site at `anchor`, or None (task 4232)."""
    return _chart_prop(body, _LINE_CHART_TAG_RE, anchor, 'formatY')


def _workflow_panel_format_y(tab_analytics_jsx_body: str) -> str:
    """WorkflowPanel's own percent formatter, from its ``<C.StackedAreaChart>`` call."""
    body = _component_body(tab_analytics_jsx_body, 'WorkflowPanel')
    format_y = _chart_prop(body, _STACKED_AREA_TAG_RE, 'labels={weeks}', 'formatY')
    assert format_y is not None, (
        'WorkflowPanel no longer passes a `formatY={...}` to <C.StackedAreaChart> '
        '— the 100%-normalized resolver-mix axis this task fixes has moved or '
        'changed shape.'
    )
    return format_y


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

    Sibling of `_axis_labels_program` above, differing in that it also carries
    LineChart's own `minV`/`range` lines (see `_line_chart_axis_scalars`), so
    each `maxV` is the axis maximum the tick generator actually spans.
    """
    return f"""
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
       destructure reached for `formatCountTick` at 4232, a name a CACHED copy of
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
# The ratio axis that keeps LineChart's formatY default non-rounding (task 4232).
#
# Count axes need no label rule of their own: `countAxis` snaps their maximum,
# so every tick is whole (task 6370). The LineChart caller whose labels depend
# on the default staying raw is the esc-per-done RATIO, pinned here with a
# frozen rounding-default control proving the guard actually fires.
# ---------------------------------------------------------------------------


def test_esc_per_done_ratio_axis_keeps_its_exact_fractions(
    charts_jsx_body: str, tab_analytics_jsx_body: str
) -> None:
    """THE CALLER THAT MAKES A ROUNDING DEFAULT UNSAFE — do not "fix" this axis.

    WorkflowPanel's escalations-per-task-done chart plots
    ``escalation_analytics.py::_esc_per_done``'s ``filings / done`` — a genuine
    float (``None`` when ``done == 0``). It is the reason LineChart's formatY
    default does not round (task 4232): a rounding default would collapse this
    axis to 0/0/0/1/1, re-filing task 4059's own defect one primitive over. It
    is also one of the fraction axes that must not declare ``countAxis`` (task
    6370; see the audit at the bottom).

    So it must pass NO count formatter, and its labels must stay the exact
    fractions the raw default produces. The 0.6000000000000001 is not a typo:
    ``(0.8 * 3) / 4`` is genuinely that double, and it is pinned as what node
    actually prints rather than as what the arithmetic ought to look like.
    """
    body = _component_body(tab_analytics_jsx_body, 'WorkflowPanel')
    assert _line_chart_format_y(body, 'row.ratio') is None, (
        'the escalations-per-done chart has gained a formatY. If that is a '
        'count formatter, its axis now blanks or rounds every non-integer '
        'gridline on a series whose values are almost never integers — the ratio '
        'would be unreadable. This chart is a FRACTION and must keep the raw default.'
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

# The MergeTab call site exactly as it stood BEFORE task 4232 — no formatY at all.
_PRE_FIX_COUNT_CALL_SITE = """
<div className="panel-body"><LC labels={d.depth.labels.map(String)} series={[{ values: d.depth.values, color: CP.accent }]} height={180} formatX={window.DF_SHELL.fmtDateTime} /></div>
"""

# The same call site as it stood BEFORE task 6370, which made `countAxis` the
# one count-axis prop: wired with a count formatter and a snapMax policy.
_PRE_6370_COUNT_CALL_SITE = (
    '<LC labels={d.depth.labels.map(String)} series={[{ values: d.depth.values, color: CP.accent }]} '
    'height={180} formatY={formatCountTick} snapMax={niceCountMax} formatX={window.DF_SHELL.fmtDateTime} />'
)


def test_rounding_default_control_shows_the_ratio_guard_actually_fires() -> None:
    """The rejected option (a) really does wreck the ratio axis, measured.

    Without this, `test_esc_per_done_ratio_axis_keeps_its_exact_fractions` would
    be a guard against a danger no one had ever demonstrated — and the design
    decision it defends (default stays non-rounding) would read as taste. Here
    the rounding default is executed against the same real tick shape and shown
    to produce the collapse.

    Also checks `_line_chart_format_y` both ways on frozen call sites: None for
    one that passes no formatY, and the expression for one that does. That None
    is what routes `_render_caller_axis` to the component default, so an
    extractor that silently found nothing would make the ratio guard above pass
    for the wrong reason.
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
    assert _line_chart_format_y(_PRE_6370_COUNT_CALL_SITE, 'd.depth.values') == 'formatCountTick', (
        'the caller extractor no longer reads the formatY of a frozen call site '
        'that passes one, so its None on the ratio chart proves nothing.'
    )


# ---------------------------------------------------------------------------
# The count-axis caller audit (task 6370), pinned in both directions.
#
# A COUNT axis must declare `countAxis`, so its maximum snaps to a multiple of
# the tick count. A FRACTION axis must not: a snapped 1 is 4, so a percent axis
# reads 0%..400%. What the flag DOES is rendered in
# dashboard/tests/js/jsx_render.test.mjs (Decision 9); this audit pins who sets
# it. The duration (fmtMs) and dollar charts are deliberately unpinned.
# ---------------------------------------------------------------------------

_COUNT_AXIS_SITES = [
    pytest.param('tabs_jsx_body', 'MemoryTab', _LINE_CHART_TAG_RE, 'MEMORY_OPS.reads', id='MemoryTab-reads-writes'),
    pytest.param('tabs_jsx_body', 'MergeTab', _LINE_CHART_TAG_RE, 'depthSeries.values', id='MergeTab-attempt-depth'),
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', _LINE_CHART_TAG_RE, 'churnDaily', id='WorkflowPanel-churn'),
    pytest.param('tab_overview_jsx_body', 'OverviewTab', _LINE_CHART_TAG_RE, 'D.MEMORY_OPS.reads', id='OverviewTab-memory-ops'),
    pytest.param('tab_analytics_jsx_body', 'OriginPanel', _STACKED_AREA_TAG_RE, 'stacks={stacks}', id='OriginPanel-filings-by-source'),
    pytest.param('tabs_jsx_body', 'BurnTab', _STACKED_AREA_TAG_RE, 'burndownStacks(b,', id='BurnTab-aggregate'),
    pytest.param('tabs_jsx_body', 'BurnTab', _STACKED_AREA_TAG_RE, 'burndownStacks(pb,', id='BurnTab-per-project'),
]

_FRACTION_AXIS_SITES = [
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', _LINE_CHART_TAG_RE, 'row.ratio', id='WorkflowPanel-esc-per-done-ratio'),
    pytest.param('tab_analytics_jsx_body', 'WorkflowPanel', _STACKED_AREA_TAG_RE, 'labels={weeks}', id='WorkflowPanel-100pct-stack'),
    pytest.param('tab_analytics_jsx_body', 'LifespanPanel', _LINE_CHART_TAG_RE, 'gridLabels', id='LifespanPanel-ECDF-percent'),
]


@pytest.mark.parametrize(('fixture', 'caller', 'tag_re', 'anchor'), _COUNT_AXIS_SITES)
def test_count_axis_call_site_declares_count_axis(
    request: pytest.FixtureRequest, fixture: str, caller: str, tag_re, anchor: str
) -> None:
    """Every count axis declares `countAxis`, so every one of its ticks is whole."""
    caller_body = _component_body(request.getfixturevalue(fixture), caller)
    assert _chart_flag(caller_body, tag_re, anchor, 'countAxis'), (
        f'{caller}\'s chart anchored at {anchor!r} does not declare countAxis, so '
        'it takes its count-axis maximum as given: at a data max of 7 it draws '
        'gridlines at 1.75 / 3.5 / 5.25.'
    )


@pytest.mark.parametrize(('fixture', 'caller', 'tag_re', 'anchor'), _FRACTION_AXIS_SITES)
def test_fraction_axis_call_site_does_not_declare_count_axis(
    request: pytest.FixtureRequest, fixture: str, caller: str, tag_re, anchor: str
) -> None:
    """A fraction axis keeps its exact maximum: snapped, a 0..1 axis would read 0%..400%."""
    caller_body = _component_body(request.getfixturevalue(fixture), caller)
    assert not _chart_flag(caller_body, tag_re, anchor, 'countAxis'), (
        f'{caller}\'s chart anchored at {anchor!r} declares countAxis. It plots a '
        'FRACTION: a snapped 0..1 axis reads 0%..400% (or 0..4 for a ratio), with '
        'the data squashed into the bottom quarter.'
    )


def test_chart_flag_extractor_is_discriminating() -> None:
    """`_chart_flag` answers both ways on frozen call sites, and refuses an assigned value.

    Both audit tests above would pass vacuously if the extractor always answered
    the same: run on the pre-6370 MergeTab call site it must report no flag.
    """
    pre_6370 = _PRE_6370_COUNT_CALL_SITE
    assert not _chart_flag(pre_6370, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        'the flag extractor reports countAxis on the pre-6370 MergeTab call site, '
        'which predates the prop.'
    )

    inline = pre_6370.replace(' formatX=', ' countAxis formatX=')
    assert _chart_flag(inline, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        'the flag extractor misses a shorthand countAxis between two attributes.'
    )

    multi_line = """
<LC
  labels={d.depth.labels.map(String)}
  series={[{ values: d.depth.values, color: CP.accent }]}
  countAxis
/>
"""
    assert _chart_flag(multi_line, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        'the flag extractor misses a shorthand countAxis on its own line before `/>`.'
    )

    for assigned in ('countAxis={true}', 'countAxis={false}', 'countAxis={isCount}'):
        with pytest.raises(AssertionError, match='shorthand'):
            _chart_flag(
                pre_6370.replace(' formatX=', f' {assigned} formatX='),
                _LINE_CHART_TAG_RE,
                'd.depth.values',
                'countAxis',
            )

    braced = '<LC labels={d.depth.labels} series={[{ values: d.depth.values, countAxis: 1 }]} />'
    assert not _chart_flag(braced, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        'the flag extractor reads a countAxis key inside a braced prop expression '
        'as the attribute.'
    )

    for quoted in ('title="per-hour countAxis count"', "aria-label='countAxis={true}'"):
        assert not _chart_flag(
            pre_6370.replace(' formatX=', f' {quoted} formatX='),
            _LINE_CHART_TAG_RE,
            'd.depth.values',
            'countAxis',
        ), f'the flag extractor reads a countAxis inside the quoted value {quoted} as the attribute.'


@pytest.mark.parametrize(
    'quoted',
    ['title="a } b"', 'aria-label="a/>b"'],
)
def test_chart_site_readers_confine_each_site_to_its_own_opening_tag(quoted: str) -> None:
    """A `}` or `/>` inside a quoted attribute neither ends nor derails the chart's tag."""
    src = f'<LC {quoted} series={{[{{ values: d.depth.values }}]}} countAxis />'

    assert _chart_flag(src, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        f'the caller audit loses the chart whose quoted attribute is {quoted}: '
        'a count axis that declares countAxis would be reported as renamed or '
        'removed, and the audit would fail on the prose, not the chart.'
    )


def test_a_child_elements_attributes_are_not_the_charts() -> None:
    """A chart with children is read from its opening tag only."""
    with_flagged_child = '<LC series={[{ values: d.depth.values }]}>\n  <Legend countAxis />\n</LC>'
    assert not _chart_flag(with_flagged_child, _LINE_CHART_TAG_RE, 'd.depth.values', 'countAxis'), (
        "the flag extractor reads a child <Legend>'s countAxis as the chart's: a "
        'fraction axis with such a child would pass the audit while snapping to 0%..400%.'
    )

    with_formatting_child = '<LC series={[{ values: d.depth.values }]}>\n  <Legend formatY={x} />\n</LC>'
    assert _line_chart_format_y(with_formatting_child, 'd.depth.values') is None, (
        "the prop extractor reads a child <Legend>'s formatY as the chart's: the "
        'ratio guard would accept a site that in fact keeps the rounding default.'
    )


def test_chart_site_readers_strip_comments_before_walking() -> None:
    """An apostrophe in a comment inside a multi-line prop must not derail the walk."""
    src = "<LC\n  series={[\n    // don't snap: a ratio\n    { values: row.ratio },\n  ]}\n  countAxis\n/>"

    assert _chart_flag(src, _LINE_CHART_TAG_RE, 'row.ratio', 'countAxis'), (
        "an apostrophe in a comment inside the chart's series reads as an open "
        'quote, so the audit would mis-span the chart and misreport its countAxis.'
    )
