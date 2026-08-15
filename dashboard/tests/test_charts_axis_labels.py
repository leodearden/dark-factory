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
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — served-asset text, same idiom as test_charts_null_samples.py:49-59
# and test_esc_flow_diagram.py:53.  Reading through the app's own static route
# rather than the filesystem also proves the asset is actually SERVED, which is
# the property the cache-buster assertion below depends on.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def charts_jsx(_client) -> str:
    return _client.get('/static/redux/charts.jsx').text


@pytest.fixture(scope='module')
def analytics_jsx(_client) -> str:
    return _client.get('/static/redux/tab_escalation_analytics.jsx').text


@pytest.fixture(scope='module')
def index_html_body(_client) -> str:
    return _client.get('/static/redux/index.html').text


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Copied from test_charts_null_samples.py:62-97 — copying rather than
    importing across test modules is this suite's established convention.  The
    paren-depth walk past the parameter list is load-bearing: all three
    components here take a destructured parameter whose own ``{``/``}`` sits
    INSIDE the parameter list, so a naive first-``{`` scan would return the
    destructuring pattern instead of the body.
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
# make a rename turn this whole file into a permanent false GREEN.
# ---------------------------------------------------------------------------


def _component_body(src: str, name: str) -> str:
    body = _extract_function_body(src, name)
    assert body, (
        f'could not slice the body of `function {name}(` out of the served '
        'source — it was renamed, moved, or converted to an arrow/const '
        'declaration. Every assertion keyed on this body would silently pass.'
    )
    return body


def _default_format_y(charts_jsx: str) -> str:
    """The ``formatY = ...`` default from the ``StackedAreaChart`` signature.

    The default arrow contains neither a comma nor a brace (``v => String(v)``
    before the fix, ``v => String(Math.round(v))`` after), so a ``[^,}]+`` run
    over the signature slice captures exactly it.
    """
    signature = _extract_signature(charts_jsx, 'StackedAreaChart')
    m = re.search(r'formatY\s*=\s*([^,}]+)', signature)
    assert m is not None, (
        'StackedAreaChart no longer declares a `formatY = <default>` in its '
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


def _workflow_panel_format_y(analytics_jsx: str) -> str:
    """WorkflowPanel's own percent formatter, from its ``<C.StackedAreaChart>`` call.

    Non-greedy up to ``} />``, which correctly steps over the ``${...}`` brace
    pair inside the template literal.
    """
    body = _component_body(analytics_jsx, 'WorkflowPanel')
    m = re.search(r'<C\.StackedAreaChart\b[^>]*?formatY=\{(.*?)\}\s*/>', body)
    assert m is not None, (
        'WorkflowPanel no longer passes a `formatY={...}` to <C.StackedAreaChart> '
        '— the 100%-normalized resolver-mix axis this task fixes has moved or '
        'changed shape.'
    )
    return m.group(1).strip()


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


def test_stacked_area_chart_does_not_pre_round_the_tick_before_format_y(charts_jsx: str) -> None:
    """StackedAreaChart must not snap the tick to an integer before formatY sees it.

    Pre-rounding destroys the fractional information every percent/decimal
    formatter needs — the caller's own formatter is the only thing that knows
    the axis UNITS. FAILED before the fix at charts.jsx:186.
    """
    body = _component_body(charts_jsx, 'StackedAreaChart')

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


def test_workflow_panel_percent_axis_reads_0_25_50_75_100(charts_jsx: str, analytics_jsx: str) -> None:
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
    body = _component_body(charts_jsx, 'StackedAreaChart')

    labels = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_workflow_panel_format_y(analytics_jsx),
            max_vs=[1],
        )
    )['1']

    assert labels == ['0%', '25%', '50%', '75%', '100%'], (
        "the Workflow panel's 100%-normalized y-axis renders as "
        f'{labels} instead of the expected 0%/25%/50%/75%/100%. The tick '
        'values reaching formatY have been rounded, scaled, or generated '
        'differently than the extracted source implies.'
    )


def test_default_format_y_callers_keep_integer_count_axes(charts_jsx: str) -> None:
    """The three no-formatY callers still render whole-number axes.

    tabs.jsx:1259, tabs.jsx:1329 and tab_escalation_analytics.jsx:244 pass no
    ``formatY`` at all and all plot integer counts. Their axes were
    integer-labelled only because of the old ``Math.round`` at the label site,
    so deleting that rounding outright would render ``2.5`` / ``7.5`` on all
    three — trading one mislabelled axis for three. The rounding therefore MOVED
    into the default, and these are the exact labels the pre-fix composition
    produced, pinned so the equivalence is checked rather than argued.
    """
    body = _component_body(charts_jsx, 'StackedAreaChart')

    axes = _run_node(
        _axis_labels_program(
            label_arg=_tick_label_arg(body),
            tick_map_param=_tick_map_param(body),
            tick_gen=_tick_generator(body),
            format_y=_default_format_y(charts_jsx),
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


def test_line_chart_hands_format_y_the_raw_tick(charts_jsx: str) -> None:
    """LineChart already handed formatY the raw tick — pin that, change nothing else.

    LineChart's non-rounding default is DELIBERATELY left alone by task 4059;
    the divergence and the caller audit it would need are documented at its
    signature in charts.jsx. This pins only the invariant the two primitives
    share — the value REACHING formatY is the unmodified tick — by executing
    LineChart's own label expression against a recording formatter rather than
    by matching its source text.
    """
    body = _component_body(charts_jsx, 'LineChart')

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


def test_probes_and_extractors_actually_fire_on_pre_fix_source(analytics_jsx: str) -> None:
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
            format_y=_workflow_panel_format_y(analytics_jsx),
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
