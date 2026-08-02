"""Wiring tests for the Memory-tab memory-evals section UI.

Tests parse JSX/JS/HTML source files as text and assert structural contracts
(endpoint registration, export names, load-order, badge derivation guards).
Follows the idiom established in test_tab_escalations.py, test_tab_curator.py
and test_index_html.py.

There is no JS test runner for ``.jsx`` in this project (``dashboard/tests/js/``
covers only plain ``.js`` modules), so JSX contracts are asserted as Python
source-as-text tests over files fetched from ``/static/redux/`` through a
module-scoped Starlette ``TestClient``.
"""

from __future__ import annotations

import html.parser
import re

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def data_js_body(_client):
    return _client.get('/static/redux/data.js').text


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def tab_escalations_jsx_body(_client):
    return _client.get('/static/redux/tab_escalations.jsx').text


@pytest.fixture(scope='module')
def tab_memory_evals_jsx_body(_client):
    return _client.get('/static/redux/tab_memory_evals.jsx').text


@pytest.fixture(scope='module')
def tab_memory_evals_jsx_code(tab_memory_evals_jsx_body):
    """`tab_memory_evals.jsx` with every comment stripped.

    The file carries ~150 lines of explanatory prose that names most of the
    payload fields the render code also names.  A bare whole-file substring
    grep is therefore satisfied by a MENTION: delete the render site, leave the
    comment, and the assertion stays green.  That false-pass mode is not
    hypothetical: `alarmed_open` and `clear` were once asserted present by a
    whole-file grep that only ever matched the explanatory prose above
    `verdictBadge` — the strings occurred nowhere in code.

    Field-presence assertions grep this code-only text instead, so a field only
    counts as "rendered" when it appears outside a comment.  Where the render
    POSITION also matters, callers additionally anchor to the accessing
    expression (`lim.alpha`, `storm.alarm_count`, ...) rather than the bare
    name.

    Safe to strip naively: the source contains no `//` inside a string literal
    (no URLs) and no regex literals, so no `/`-bearing code is eaten.
    """
    return re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', tab_memory_evals_jsx_body)


# ---------------------------------------------------------------------------
# Helper: extract a named seed block from window.DF_DATA (brace-aware)
# ---------------------------------------------------------------------------


def _extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included.

    Locates ``<key>:`` followed by ``{`` (allowing arbitrary whitespace), then
    walks forward counting ``{``/``}`` to find the matching close brace.
    This is brace-aware: a simple regex ``[^}]*`` would stop at the first
    nested ``}`` and miss later keys.
    Returns the empty string if no matching block is found.

    Note: the brace-depth walk does not skip ``{``/``}`` inside JS string
    literals.  This is acceptable because the data.js seed block uses simple
    numeric/array values and does not embed brace characters inside quoted
    strings.
    """
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening `{`
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware)
# ---------------------------------------------------------------------------


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Uses the same brace-depth walk as ``_extract_df_data_block``.  Only matches
    named ``function`` declarations — not arrow functions or class methods.
    Returns the empty string if the function is not found.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern (e.g. ``{ a, b }``) instead of the function body.
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
# Helper: extract a module-scope `const <name> = { ... }` / `[ ... ]` literal
# ---------------------------------------------------------------------------


def _extract_const_object(src: str, name: str, open_char: str = '{') -> str:
    """Return the literal assigned to ``const <name> =``, delimiters included.

    Same depth walk as ``_extract_df_data_block``, re-anchored: that helper
    only matches the ``key: {`` seed-object form used by data.js and so cannot
    locate a module-scope ``const`` declaration.  ``open_char`` selects the
    delimiter pair, so one walk serves both the ``PARITY_REFINEMENT`` object
    and the ``PARITY_PLAIN`` array.

    Returns the empty string if the declaration is not found — callers assert
    on that explicitly, because "the declaration was deleted" and "the
    declaration is empty" are different failures with different fixes.

    Same string-literal caveat as ``_extract_df_data_block``: the walk does not
    skip delimiters inside quoted strings.  Acceptable here for the same
    reason — these two declarations hold short identifier keys and plain
    prose values, neither of which embeds a brace or a bracket.
    """
    close_char = {'{': '}', '[': ']'}[open_char]
    m = re.search(rf'\bconst\s+{re.escape(name)}\s*=\s*{re.escape(open_char)}', src)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening delimiter
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


# ---------------------------------------------------------------------------
# Load-order helpers (copied from test_index_html.py / test_tab_escalations.py)
# ---------------------------------------------------------------------------


class _ScriptTagCollector(html.parser.HTMLParser):
    """Collects the attribute dicts for every <script> start-tag encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == 'script':
            self.script_attrs.append(dict(attrs))


def _find_script_position(
    body: str, src_prefix: str
) -> tuple[int, dict[str, str | None]] | None:
    """Return ``(index, attrs)`` for the first <script> tag whose ``src``
    starts with ``src_prefix``, or ``None`` if no such tag exists.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


def _assert_script_loads_before(
    body: str,
    before_src_prefix: str,
    after_src_prefix: str,
    before_label: str,
    after_label: str,
    consumer_note: str = '',
) -> None:
    """Assert that the script for ``before_src_prefix`` loads BEFORE the
    script for ``after_src_prefix`` in ``body``.  Combines a
    defer/async/type=module false-pass guard with the document-order
    position comparison.
    """
    before_result = _find_script_position(body, before_src_prefix)
    assert before_result is not None, (
        f'No <script src="{before_src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    before_pos, before_attrs = before_result
    before_src = before_attrs.get('src')

    after_result = _find_script_position(body, after_src_prefix)
    assert after_result is not None, (
        f'<script src="{after_src_prefix}..."> not found in index.html — '
        f'cannot verify load-order invariant for {before_label}.'
    )
    after_pos, after_attrs = after_result

    # Both tags must be classic synchronous scripts — otherwise document order
    # diverges from execution order and the position comparison below is moot.
    for _label, _attrs in [
        (before_label, before_attrs),
        (after_label, after_attrs),
    ]:
        assert 'defer' not in _attrs, (
            f'{_label} has a defer attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert 'async' not in _attrs, (
            f'{_label} has an async attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert (_attrs.get('type') or '').lower() != 'module', (
            f'{_label} has type="module"; ES modules are deferred by default, '
            f'so document order no longer implies execution order.'
        )

    assert before_pos < after_pos, (
        f'{before_label} (position {before_pos}, src={before_src!r}) must load '
        f'BEFORE {after_label} (position {after_pos}). '
        f'{consumer_note}'
    )


# ---------------------------------------------------------------------------
# step-1 test: data.js registers the memory-evals endpoint + seed
# ---------------------------------------------------------------------------


# The seven top-level keys of ``redux_api.shape_memory_evals``'s return body.
# This list IS the payload contract (that fn's own docstring says so, per PRD
# open question 4); the React section consumes exactly these spellings.
_MEMORY_EVALS_CONTRACT_KEYS = (
    'generated_at',
    'root_present',
    'storm_escape',
    'evals',
    'issues',
    'issue_count',
    'unmatched_escalations',
)


def test_data_js_registers_memory_evals_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/memory-evals -> ['MEMORY_EVALS']
    and seed DF_DATA.MEMORY_EVALS with the server's own default body.

    The seed is asserted key-by-key against the ``shape_memory_evals`` contract
    rather than eyeballed, and its defaults must be the *empty* shape
    (``root_present: false``, empty lists) — NOT illustrative rows.  A
    default-shaped body is structurally identical to a real one, so the
    pre-fetch render and a genuine empty response are indistinguishable and no
    component has to branch on which it got.  Any invented row would be
    synthetic data (feedback_redux_no_synthetic_data).
    """
    assert '/api/v2/dashboard/memory-evals' in data_js_body, (
        "data.js does not register '/api/v2/dashboard/memory-evals'. Add a "
        'static (non-windowed) row to endpointsFor() mapping it to '
        "['MEMORY_EVALS']."
    )

    # (b) key and value must be checked as a PAIR — two independent substring
    # hits would pass even if the endpoint mapped to some other DF_DATA key.
    assert re.search(
        r"""['"]/api/v2/dashboard/memory-evals['"]\s*:\s*\[\s*['"]MEMORY_EVALS['"]\s*,?\s*\]""",
        data_js_body,
    ), (
        "data.js's endpointsFor() must map '/api/v2/dashboard/memory-evals' to "
        "exactly ['MEMORY_EVALS']."
    )

    # (c) the DF_DATA seed block exists
    seed_block = _extract_df_data_block(data_js_body, 'MEMORY_EVALS')
    assert seed_block, (
        'data.js has no MEMORY_EVALS seed in the `window.DF_DATA = {...}` '
        'literal. Without it the first render before the fetch completes '
        'reads undefined and crashes the Memory tab.'
    )

    # (d) exactly the seven contract keys — no more, no fewer
    for key in _MEMORY_EVALS_CONTRACT_KEYS:
        assert re.search(rf'\b{key}\s*:', seed_block), (
            f"data.js MEMORY_EVALS seed is missing the '{key}' key required by "
            f'redux_api.shape_memory_evals. Seed block was: {seed_block}'
        )
    seeded = set(re.findall(r'\b(\w+)\s*:', seed_block))
    assert seeded == set(_MEMORY_EVALS_CONTRACT_KEYS), (
        'data.js MEMORY_EVALS seed keys do not match the shape_memory_evals '
        f'contract exactly. Extra: {sorted(seeded - set(_MEMORY_EVALS_CONTRACT_KEYS))}, '
        f'missing: {sorted(set(_MEMORY_EVALS_CONTRACT_KEYS) - seeded)}.'
    )

    # (e) the defaults are the server's own healthy no-artifacts shape
    assert re.search(r'\broot_present\s*:\s*false\b', seed_block), (
        "data.js MEMORY_EVALS seed must default root_present to `false` — the "
        "server's own default body. Seeding `true` would claim an eval root "
        'exists before anything has been fetched.'
    )
    assert re.search(r'\bgenerated_at\s*:\s*null\b', seed_block), (
        'data.js MEMORY_EVALS seed must default generated_at to `null`.'
    )
    assert re.search(r'\bstorm_escape\s*:\s*null\b', seed_block), (
        'data.js MEMORY_EVALS seed must default storm_escape to `null` — a '
        'non-null block renders the storm banner.'
    )
    assert re.search(r'\bissue_count\s*:\s*0\b', seed_block), (
        'data.js MEMORY_EVALS seed must default issue_count to `0`.'
    )
    for list_key in ('evals', 'issues', 'unmatched_escalations'):
        assert re.search(rf'\b{list_key}\s*:\s*\[\s*\]', seed_block), (
            f"data.js MEMORY_EVALS seed must default '{list_key}' to an empty "
            'array — never to illustrative rows '
            '(feedback_redux_no_synthetic_data). Seed block was: '
            f'{seed_block}'
        )


# ---------------------------------------------------------------------------
# step-3 test: index.html registers tab_memory_evals.jsx in the right position
# ---------------------------------------------------------------------------


_TAB_MEMEVALS_PREFIX = '/static/redux/tab_memory_evals.jsx'

_LOAD_ORDER_NOTE = (
    'tab_memory_evals.jsx exports window.DF_MEMORY_EVALS, which tabs.jsx '
    'destructures at MODULE TOP LEVEL — exactly as tab_scheduler.jsx:15 '
    'destructures window.DF_SCHED_HEATMAP. A later tag would leave the global '
    'undefined at tabs.jsx evaluation time, throwing and blanking EVERY tab '
    'defined in that file. This is the opposite direction from '
    'tab_escalations.jsx, which loads AFTER tabs.jsx because it additively '
    'mutates the window.DF_TABS object tabs.jsx creates.'
)


def test_index_html_registers_tab_memory_evals_load_order(
    index_html_body: str,
) -> None:
    """index.html must load tab_memory_evals.jsx after its dependencies and
    BEFORE tabs.jsx, as a classic synchronous text/babel script.

    All /static/redux/* tags are classic sync scripts, so document order IS
    execution order — which is why the no-defer / no-async / no-type=module
    guard runs before every position comparison.
    """
    # (a) the tag exists and is a classic synchronous text/babel script
    found = _find_script_position(index_html_body, _TAB_MEMEVALS_PREFIX)
    assert found is not None, (
        f'No <script src="{_TAB_MEMEVALS_PREFIX}..."> tag in index.html. '
        f'{_LOAD_ORDER_NOTE}'
    )
    _pos, attrs = found
    assert 'defer' not in attrs, (
        'tab_memory_evals.jsx must not carry defer — document order would no '
        'longer imply execution order and tabs.jsx could evaluate first.'
    )
    assert 'async' not in attrs, (
        'tab_memory_evals.jsx must not carry async — document order would no '
        'longer imply execution order and tabs.jsx could evaluate first.'
    )
    assert (attrs.get('type') or '').lower() == 'text/babel', (
        'tab_memory_evals.jsx contains JSX, so its <script> tag must be '
        f'type="text/babel"; got type={attrs.get("type")!r}. Note type="module" '
        'would additionally be deferred by default, breaking load order.'
    )

    # (b) after its dependencies: data.js (DF_DATA seed), charts.jsx
    #     (DF_CHARTS primitives + PALETTE), shell.jsx.
    for dep_prefix, dep_label, why in [
        (
            '/static/redux/data.js',
            'data.js',
            'tab_memory_evals.jsx reads window.DF_DATA, seeded by data.js.',
        ),
        (
            '/static/redux/charts.jsx',
            'charts.jsx',
            'tab_memory_evals.jsx destructures Sparkline/StepSpark/StatTile/'
            'PALETTE off window.DF_CHARTS at module top level.',
        ),
        (
            '/static/redux/shell.jsx',
            'shell.jsx',
            'tab_memory_evals.jsx uses window.DF_SHELL formatting helpers.',
        ),
    ]:
        _assert_script_loads_before(
            index_html_body,
            dep_prefix,
            _TAB_MEMEVALS_PREFIX,
            dep_label,
            'tab_memory_evals.jsx',
            why,
        )

    # (c) THE load-bearing assertion — before tabs.jsx.
    _assert_script_loads_before(
        index_html_body,
        _TAB_MEMEVALS_PREFIX,
        '/static/redux/tabs.jsx',
        'tab_memory_evals.jsx',
        'tabs.jsx',
        _LOAD_ORDER_NOTE,
    )

    # (d) transitively therefore before app.jsx, which destructures DF_TABS last.
    _assert_script_loads_before(
        index_html_body,
        _TAB_MEMEVALS_PREFIX,
        '/static/redux/app.jsx',
        'tab_memory_evals.jsx',
        'app.jsx',
        'app.jsx renders MemoryTab, which renders MemoryEvalsSection.',
    )

    # (e) one uniform cache-buster across every /static/redux/* asset,
    #     including the styles.css <link> — mirrors the guards in
    #     test_index_html.py:593 (floor 37) and test_tab_escalations.py:487.
    versions = set(
        re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)
    )
    assert len(versions) == 1, (
        f'index.html has mixed /static/redux/?v= cache-buster versions: '
        f'{sorted(versions)} — bump all of them uniformly, the stylesheet '
        '<link> included.'
    )
    v = int(next(iter(versions)))
    assert v >= 38, (
        f'index.html cache-buster version is {v}, expected >= 38 (proves the '
        'uniform bump for the memory-evals section actually reaches '
        'already-open browsers; the previous floor, 37, proved task 3332\'s '
        '`const API` collision fix).'
    )


# ---------------------------------------------------------------------------
# step-5 tests: tab_memory_evals.jsx is served and renders eval cards + trends
# ---------------------------------------------------------------------------


def test_tab_memory_evals_jsx_served_and_exports_section(_client) -> None:
    """The section file must be served and export MemoryEvalsSection on its
    own window global, following the scheduler_heatmap.jsx:191 producer idiom.
    """
    resp = _client.get('/static/redux/tab_memory_evals.jsx')
    assert resp.status_code == 200, (
        'GET /static/redux/tab_memory_evals.jsx returned '
        f'{resp.status_code} — index.html already references the file '
        '(step-4), so a missing file is a hard 404 in the browser.'
    )
    body = resp.text
    assert 'window.DF_MEMORY_EVALS = {' in body, (
        'tab_memory_evals.jsx must set `window.DF_MEMORY_EVALS = { ... }` at '
        'the file tail (the scheduler_heatmap.jsx:191 precedent) — that global '
        'is what tabs.jsx destructures at module top level.'
    )
    assert re.search(
        r'window\.DF_MEMORY_EVALS\s*=\s*\{[^}]*\bMemoryEvalsSection\b', body
    ), (
        'window.DF_MEMORY_EVALS must export MemoryEvalsSection — tabs.jsx '
        'renders it inside MemoryTab.'
    )


def test_tab_memory_evals_renders_eval_cards_and_trends(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """The section renders one card per eval and a trend per metric, from the
    payload's own parallel-array trend shape.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # (c) the component exists as a named function declaration
    assert re.search(r'\bfunction\s+MemoryEvalsSection\s*\(', body), (
        'tab_memory_evals.jsx must define `function MemoryEvalsSection(`.'
    )

    # (d) chart primitives + PALETTE destructured off DF_CHARTS at module top
    #     level, under file-unique aliases (the per-file alias convention:
    #     tabs.jsx uses CP/ST, tab_scheduler.jsx uses stUseState, ...).
    charts_destructure = re.search(
        r'const\s*\{([^}]*)\}\s*=\s*window\.DF_CHARTS\s*;', body
    )
    assert charts_destructure is not None, (
        'tab_memory_evals.jsx must destructure its chart primitives off '
        '`window.DF_CHARTS` at module top level.'
    )
    destructured = charts_destructure.group(1)
    assert 'PALETTE' in destructured, (
        'tab_memory_evals.jsx must destructure PALETTE off window.DF_CHARTS '
        '(no hard-coded colour literals).'
    )
    # Deliberately NOT asserted here: that the destructured primitives use
    # colon aliases.  That was a naming-convention check performed by
    # introspecting identifier spellings — it constrains how symbols are
    # SPELLED, not what the code does, and any junk alias satisfied it while a
    # safe unaliased import failed it.  The real invariant (no global
    # collision at in-browser-Babel load time) is enforced structurally by
    # test_index_html_registers_tab_memory_evals_load_order, which pins this
    # file's script tag ahead of tabs.jsx.
    charts_pos = charts_destructure.start()
    fn_pos = body.index('function MemoryEvalsSection')
    assert charts_pos < fn_pos, (
        'the window.DF_CHARTS destructure must sit at module top level, above '
        'MemoryEvalsSection — not inside it.'
    )

    # (e) reads DF_DATA.MEMORY_EVALS, maps evals keyed on eval_id, metrics on
    #     metric_id
    assert 'window.DF_DATA' in body, (
        'tab_memory_evals.jsx must read window.DF_DATA.'
    )
    assert 'MEMORY_EVALS' in body, (
        'tab_memory_evals.jsx must read the MEMORY_EVALS key of DF_DATA.'
    )
    assert re.search(r'\.evals\b', body), (
        'tab_memory_evals.jsx must render the payload\'s `evals` list.'
    )
    assert re.search(r'key=\{[^}]*\beval_id\b', body), (
        'each eval card must be keyed on `eval_id`.'
    )
    assert re.search(r'\.metrics\b', body), (
        "tab_memory_evals.jsx must render each eval's `metrics` list."
    )
    assert re.search(r'key=\{[^}]*\bmetric_id\b', body), (
        'each metric row must be keyed on `metric_id`.'
    )

    # (f) the trend renders from the PARALLEL-ARRAY shape, holes intact
    assert 'trend.labels' in code, (
        'the trend must be rendered from `trend.labels` (the payload ships two '
        'index-aligned parallel arrays, not point objects).'
    )
    assert 'trend.values' in code, (
        'the trend must be rendered from `trend.values`.'
    )
    # The invariant is NO SILENT DROPPING, not "no null handling".  DETECTING
    # holes — counting them, testing `=== null`, gating the chart on the count —
    # is explicitly permitted and in fact required (see
    # test_trend_holes_are_never_handed_to_a_chart_primitive below).  What is
    # forbidden is COMPACTION: a hole REMOVED from the series.
    for hostile, what in (
        # a chart fed a transformed series rather than the payload array itself
        (r'values\s*=\s*\{[^}\n]*\.\s*(?:filter|flatMap|reduce)\(', 'a chart is fed a transformed series'),
        # `.filter(Boolean)` drops nulls AND legitimate zeroes
        (r'trend\.(?:values|labels)\s*\.\s*(?:filter|flatMap)\(\s*Boolean\s*\)', '.filter(Boolean) drops holes'),
        # a keep-the-non-nulls predicate — the dropping shape.  Note `!==`, not
        # `===`: `.filter(v => v === null).length` COUNTS holes and is fine.
        (
            r'trend\.(?:values|labels)\s*\.\s*filter\([^)\n]*!==?\s*(?:null|undefined)',
            'a null-dropping filter predicate',
        ),
        (r'trend\.(?:values|labels)\s*\.\s*flatMap\(', 'flatMap can drop elements'),
    ):
        assert not re.search(hostile, body), (
            f'trend values/labels must never be COMPACTED ({what}). A `null` '
            'in `values` is a deliberate hole (that run produced no sample); '
            'dropping it would shift this metric\'s points against every '
            "other metric's, since all series share the run_stamps x-axis. "
            'Counting holes and suppressing the chart is the correct response; '
            'removing them is not.'
        )

    # (g) charts.jsx primitives only — no new chart library
    assert re.search(r'\b(MESpark|MEStep|METile|MELine|Sparkline|StepSpark|StatTile|LineChart)\b', body), (
        'the section must use at least one charts.jsx primitive '
        '(Sparkline / StepSpark / StatTile / LineChart).'
    )
    # Run over the comment-stripped `code`, not `body`: a comment that merely
    # NAMES a library ("deliberately not d3") is prose, not a dependency, and
    # must not fail the build.  Word-boundary matched for the same reason a
    # bare `'d3' in ...` is wrong — it fires on "PRD DD3".
    for lib in (r'd3', r'chart\.js', r'recharts', r'plotly'):
        assert not re.search(rf'\b{lib}\b', code, re.IGNORECASE), (
            f"tab_memory_evals.jsx references '{lib}' — the PRD and the task "
            'both forbid a new chart library; use charts.jsx primitives only.'
        )

    # (h) the kind vocabulary lives in the primitive-selection helper, with an
    #     explicit unknown/null fallback
    assert re.search(r'\bfunction\s+chartForKind\s*\(', body), (
        'tab_memory_evals.jsx must define `function chartForKind(kind)` — the '
        'single place the metric-kind vocabulary maps to a chart primitive.'
    )
    kind_body = _extract_function_body(body, 'chartForKind')
    assert kind_body, 'could not extract the chartForKind body.'
    for kind in ('tripwire', 'proportion', 'count', 'scalar'):
        assert f"'{kind}'" in kind_body, (
            f"chartForKind must name the '{kind}' metric kind — the payload's "
            'vocabulary is exactly {tripwire, proportion, count, scalar}.'
        )
    assert 'return null' in kind_body.replace('  ', ' '), (
        'chartForKind must fall back to `null` (value only, NO chart) for an '
        'unknown-or-null kind. A kind outside the known set is a rendering gap '
        "the payload already files an `unknown_kind` issue for; guessing a "
        'primitive would render an unvalidated shape as though it were '
        'understood.'
    )

    # (i) the truncation disclosure names both counts
    assert re.search(r'\btruncated\b', body), (
        "the eval-level `truncated` flag must gate a visible disclosure."
    )
    assert re.search(r'\bev\.run_count\b', code) and re.search(
        r'\bev\.runs_on_disk\b', code
    ), (
        'the truncation disclosure must READ `ev.run_count` shown of '
        '`ev.runs_on_disk` on disk — a bare "truncated" badge hides how much '
        'was dropped. Anchored to the accessing expression in comment-stripped '
        'source: a bare name grep is satisfied by the prose above the render.'
    )


# ---------------------------------------------------------------------------
# step-7 tests: verdict badges come from the payload; the browser derives none
# ---------------------------------------------------------------------------


# The four persisted verdict strings, passed through unmapped by the builder.
_VERDICTS = ('alarm', 'no_alarm', 'insufficient_data', 'grandfathered')

# The parity vocabulary is deliberately NOT restated here.  It is imported from
# the producer (`memory_evals.PARITY_STATES`) inside each test that needs it —
# a local copy is exactly the rot this suite now exists to prevent, and one
# lived here until task 3442: a hand-picked three-member tuple that could not
# notice the six states task 3363 added.  See
# `test_parity_vocabulary_fully_covered` for the completeness contract, and
# test_memory_evals_data.py::TestParityVocabularyIsClosedAndExported for the
# proof that the exported frozenset matches what the builder actually emits.


def test_parity_vocabulary_fully_covered(tab_memory_evals_jsx_code: str) -> None:
    """Every `PARITY_STATES` member is handled, and nothing else is.

    The JSX declares its whole view of the vocabulary in two module-scope
    names — `PARITY_REFINEMENT` (states whose badge is refined) and
    `PARITY_PLAIN` (states that deliberately decline refinement) — precisely so
    this can be checked against the PRODUCER rather than against a subset
    copied into this file.  Both directions are separate failures:

    * a member in NEITHER declaration is a state the server can emit today and
      the browser has never been told about;
    * a declared state OUTSIDE `PARITY_STATES` is a dead branch the producer no
      longer emits, which no render test would ever reach.

    An explicit opt-out list is what makes the first check possible at all.
    "Falls through to the plain badge" is otherwise a claim about which branch
    does NOT execute — unobservable to a source-assertion test, which is why
    the previous version of this suite simply omitted those states and went
    blind to six of them.
    """
    from dashboard.data.memory_evals import PARITY_STATES

    code = tab_memory_evals_jsx_code

    refinement = _extract_const_object(code, 'PARITY_REFINEMENT')
    assert refinement, (
        'tab_memory_evals.jsx must declare `const PARITY_REFINEMENT = {...}` at '
        'module scope in CODE (not in a comment). It is the one place the file '
        'says which parity states change the badge, and what this test compares '
        'against memory_evals.PARITY_STATES.'
    )
    plain = _extract_const_object(code, 'PARITY_PLAIN', open_char='[')
    assert plain, (
        'tab_memory_evals.jsx must declare `const PARITY_PLAIN = [...]` at '
        'module scope — the EXPLICIT opt-out list. Without it, a state that is '
        'merely unhandled is indistinguishable from one deliberately left to '
        'the plain verdict badge, and this test cannot tell a considered '
        'decision from an oversight.'
    )

    # Keys are quoted in the source (they are producer vocabulary strings, not
    # JS identifiers), but the quotes are tolerated rather than required here:
    # the presence grep in `test_verdict_badges_driven_by_persisted_verdict` is
    # what enforces the quoted form, and it says so clearly. Requiring them
    # here too would report a dropped quote as "the table holds no entries".
    entries = re.findall(r'[\'"]?(\w+)[\'"]?\s*:\s*\{([^{}]*)\}', refinement)
    handled = {key for key, _ in entries}
    declined = set(re.findall(r"'([^']*)'", plain))

    assert handled, 'PARITY_REFINEMENT is declared but holds no entries.'
    assert declined, 'PARITY_PLAIN is declared but holds no states.'

    overlap = handled & declined
    assert overlap == set(), (
        f'{sorted(overlap)} appear in BOTH PARITY_REFINEMENT and PARITY_PLAIN. '
        'A state is either refined or deliberately plain; declaring both makes '
        'the opt-out list stop meaning "considered and declined".'
    )

    unhandled = PARITY_STATES - (handled | declined)
    assert unhandled == set(), (
        f'memory_evals.PARITY_STATES member(s) {sorted(unhandled)} appear in '
        'neither PARITY_REFINEMENT nor PARITY_PLAIN. The server can emit these '
        'today and this file has never been told about them, so they render '
        'through whatever the fall-through happens to be. Add each to the '
        'table (with the fact its badge should carry) or to the plain list '
        '(if the verdict badge already says everything there is to say).'
    )

    dead = (handled | declined) - PARITY_STATES
    assert dead == set(), (
        f'{sorted(dead)} are declared here but are not in '
        'memory_evals.PARITY_STATES — the producer cannot emit them, so they '
        'are dead branches no render can reach. Delete them, or fix the '
        'spelling if the producer renamed the state.'
    )

    # The composition invariant, in its structural form: table values are
    # SUFFIXES, never whole labels, so there is no expression in this file
    # capable of returning a label that discards the verdict-derived base.
    for key, value in entries:
        keys = set(re.findall(r'(\w+)\s*:', value))
        assert keys == {'suffix', 'cls'}, (
            f"PARITY_REFINEMENT['{key}'] must be a {{ suffix, cls }} pair, got "
            f'keys {sorted(keys)}. Storing a whole label would let a parity '
            'branch report a state the payload never asserted — `_parity()` '
            'derives most states from (verdict class, linked?), so the verdict '
            'must survive into the label.'
        )

    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'
    labels = re.findall(r'label\s*:\s*([^,}\n]+)', badge_body)
    assert labels, 'verdictBadge returns no `label`.'
    for expr in labels:
        assert expr.strip().startswith('base'), (
            f'verdictBadge returns the label {expr.strip()!r}, which does not '
            'begin with the verdict-derived `base`. Every returned label must '
            'compose onto it: a fixed label reports a state the payload never '
            'asserted (for recovered_open that would include a metric which '
            'was never measured being shown as having recovered).'
        )
    assert any('suffix' in e for e in labels), (
        'no verdictBadge return composes a PARITY_REFINEMENT `suffix` onto '
        '`base` — the table is declared but never consumed, so every refined '
        'state renders as the plain verdict badge.'
    )


def test_verdict_badges_driven_by_persisted_verdict(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """Badges must name every persisted verdict and every server-derived
    parity state, and an absent verdict must render its own explicit state.

    Defaulting a null verdict to `no_alarm` would turn "we do not know" into
    "we checked and it is fine" — the builder itself refuses that at
    memory_evals.py:847-849, and the UI must not undo it.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    assert re.search(
        r'window\.DF_MEMORY_EVALS\s*=\s*\{[^}]*\bverdictBadge\b', body
    ), (
        'window.DF_MEMORY_EVALS must export a `verdictBadge` helper — the '
        'single place badge state is decided, so the no-derivation guard below '
        'has one function to check.'
    )

    # Grepped against COMMENT-STRIPPED source: every one of these strings also
    # appears in the prose above verdictBadge, so a whole-file grep would stay
    # green after the branch itself was deleted.
    for verdict in _VERDICTS:
        assert f"'{verdict}'" in code, (
            f"tab_memory_evals.jsx must name the persisted verdict '{verdict}' "
            'in CODE (not merely in a comment). The vocabulary is passed '
            'through unmapped by the builder; there is no client-side '
            'translation table.'
        )
    from dashboard.data.memory_evals import PARITY_STATES

    for parity in sorted(PARITY_STATES):
        assert f"'{parity}'" in code, (
            f"tab_memory_evals.jsx must name the server-derived parity state "
            f"'{parity}' in CODE, as a quoted string literal — the comment "
            'block above verdictBadge names all of them, so a whole-file grep '
            'proves nothing, and PARITY_REFINEMENT quotes its keys precisely '
            'so they are greppable in the form the payload carries. Iterated '
            'from the PRODUCER, so a state added there fails here rather than '
            'rendering through an unwritten branch.'
        )

    # Existing .badge vocabulary — no new CSS needed for four verdict states.
    for cls in ('badge bad', 'badge ok', 'badge warn', 'badge info', 'badge muted'):
        assert cls in code, (
            f"tab_memory_evals.jsx must use the existing '{cls}' class "
            '(styles.css:239-261) rather than inventing badge styling.'
        )

    # A null/absent verdict renders its own state, not a defaulted one.
    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'
    # (i) THE PARITY SHORT-CIRCUIT — that no parity branch may discard the
    #     verdict-derived label — now lives in
    #     `test_parity_vocabulary_fully_covered`, asserted structurally over
    #     every `label:` in this body rather than per-branch. Storing suffixes
    #     in PARITY_REFINEMENT is what makes the stronger form possible: there
    #     is no longer an expression in the file CAPABLE of returning a label
    #     that omits `base`, so the invariant holds for states this suite has
    #     not enumerated as well as for the ones it has.

    # (ii) the base label is derived from `verdict`, and an unrecognised
    #      verdict gets its own state rather than a defaulted one.
    base_m = re.search(r'\b(?:let|const|var)\s+(\w+)\s*=\s*(.+?);', badge_body)
    assert base_m, 'verdictBadge must compute a verdict-derived base label.'
    assert not re.search(
        r"\b(?:let|const|var)\s+\w+\s*=\s*'(no_alarm|alarm)'\s*;", badge_body
    ), (
        'the base label must not be INITIALISED to a real verdict — an '
        'unrecognised verdict would then inherit it. Absent is absent: seed '
        'the not-measured state and overwrite it only on a recognised verdict.'
    )
    assert not re.search(r"return\s*\{[^}]*\bcls\s*:\s*'badge ok'\s*,", badge_body) or \
        re.search(r"verdict\s*===\s*'no_alarm'", badge_body), (
        "the 'badge ok' styling may only be reached via an explicit "
        "`verdict === 'no_alarm'` test — never as a fall-through."
    )
    assert not re.search(
        r"verdict\s*(\|\||\?\?)\s*['\"]no_alarm['\"]", body
    ), (
        'a null verdict must NOT be defaulted to no_alarm — that would report '
        '"we did not measure" as "we measured and it is fine".'
    )


def test_no_client_side_alarm_derivation(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """THE load-bearing exclusion (PRD section 8 / G6 / INV-5).

    `verdict` and `parity` are the ONLY badge inputs.  memory_evals.py:660-661
    records why: `parity` exists precisely so "the UI [does not] re-deriv[e]
    badge state out of three separate fields, which is where the two sides
    would drift apart".  A browser-side value-vs-limit comparison would be a
    second, divergent alarm rule shipped next to the real one.

    Displaying `limits.alpha` is legal.  Comparing against it is the violation.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # (1) The badge helper reads verdict/parity and performs NO comparison.
    #     Extracted from COMMENT-STRIPPED source: the operator-facing `<`/`>`
    #     characters that appear in prose (an `->` arrow, a "<Chart" reference)
    #     are not comparisons, and failing on them would push the next author
    #     to reword a correct comment rather than fix real code.
    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'
    assert 'parity' in badge_body, (
        'verdictBadge must read `parity` — the server-derived display state.'
    )
    assert 'verdict' in badge_body, (
        'verdictBadge must read `verdict` — the persisted judgment.'
    )
    for op in ('<=', '>=', '<', '>'):
        assert op not in badge_body, (
            f'verdictBadge contains the comparison operator {op!r}. It may only '
            'test STRING EQUALITY against verdict/parity — any ordering '
            'comparison means badge state is being re-derived in the browser '
            '(PRD section 8, G6/INV-5). Body was:\n' + badge_body
        )

    # (2) Nowhere in the file may a threshold-ish field feed a comparison.
    #     Matched on MEMBER ACCESS (`lim.alpha <`) rather than the bare word, so
    #     a JSX text label like <span>min_samples</span> — where `<` belongs to
    #     the closing tag — is not a false positive.
    threshold_fields = (
        'limit',
        'limits',
        'limit_ref',
        'alpha',
        'threshold',
        'min_samples',
        'false_alarm_budget',
        'p_value',
        'denominator',
    )
    alternation = '|'.join(threshold_fields)
    for pattern, direction in (
        (rf'\.\s*({alternation})\b\s*(<=|>=|<|>)', 'field on the left'),
        (rf'(<=|>=|<|>)\s*\w*\.\s*({alternation})\b', 'field on the right'),
    ):
        hit = re.search(pattern, body)
        assert hit is None, (
            f'tab_memory_evals.jsx compares a threshold-ish field '
            f'({direction}): {hit.group(0)!r}. Alarm state comes from the '
            'persisted `verdict` plus the server-derived `parity` and nothing '
            'else; a value-vs-limit comparison here is a second, divergent '
            'alarm rule (PRD section 8, G6/INV-5).'
        )

    # (3) No statistics on the browser side at all.
    for ident in ('p_value', 'z_score', 'stddev', 'Math.sqrt', 'Math.log'):
        assert ident not in body, (
            f"tab_memory_evals.jsx references '{ident}'. The dashboard does no "
            'statistics — the eval runner computed the verdict and the '
            'dashboard displays it.'
        )


def test_trend_holes_are_never_handed_to_a_chart_primitive(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """A series containing a hole must NOT be drawn — charts.jsx cannot
    represent one, so drawing it fabricates a measurement.

    The defect: charts.jsx's `Sparkline` (:42-53) and `StepSpark` (:66-90) do
    plain arithmetic with no null handling — `Math.max(...values, 1)`,
    `Math.min(...values, 0)`, `y = height - ((v - min) / range) * height` — in
    which a `null` coerces to 0.  A hole handed to them is therefore drawn as a
    REAL point at the chart floor, joined by a line segment to its neighbours
    and indistinguishable from a measured regression to zero.  For a
    `proportion` metric sitting at 0.95 that reads as a plunge and recovery
    that never happened.

    That directly contradicts this file's own `dash()` invariant — "Missing
    scalars render an em-dash, never `|| 0`: a synthetic zero reads as a
    measured zero".  The trend column was the one place it was violated.

    The fix is suppression, not compaction: the array is still passed through
    verbatim (guarded above), but a series the primitive cannot represent is
    not drawn at all.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # (iv) hole DETECTION must still exist and still run — the fix is to act on
    #      the count, not to stop counting.
    assert re.search(r'\bfunction\s+trendGaps\s*\(', body), (
        'tab_memory_evals.jsx must still define `trendGaps(values)` — hole '
        'detection is required, not optional.'
    )
    assert len(re.findall(r'\btrendGaps\s*\(', body)) >= 2, (
        '`trendGaps` is defined but never called. Detecting holes and then '
        'ignoring the count is the defect this test exists to prevent.'
    )

    # (ii) the gap count must participate in the guard on the <Chart render
    #      site.  Two spellings are accepted — an inline `Chart && !gaps`, or a
    #      named local (`plottable` / `hasGaps` / ...) combining both — because
    #      the invariant is "the count gates the chart", not one exact phrasing.
    #
    #      KEPT DELIBERATELY, against a review suggestion to delete this block
    #      as identifier-spelling introspection covered by the `bare` negative
    #      check below.  Measured, not argued: regressing the JSX to
    #      `const plottable = Chart;` (gate removed, holed series reaches the
    #      primitive) fails HERE and `bare` passes it — `bare` only matches the
    #      literal `{Chart &&` / `{Chart ?` spelling, so any named local evades
    #      it entirely.  The two are not redundant: this is the only POSITIVE
    #      check that a gate exists at all, and deleting it silently reopens
    #      the defect 5ad120a0b3 fixed (charts.jsx coerces null to 0 and draws
    #      a fabricated plunge to the chart floor; see task 3436).
    inline = re.search(r'\{\s*Chart\s*&&[^\n]*\bgaps\b', body)
    via_local = None
    for decl in re.finditer(
        r'const\s+(\w+)\s*=\s*[^;\n]*\bChart\b[^;\n]*\bgaps\b[^;\n]*;', body
    ):
        if re.search(r'\{\s*' + re.escape(decl.group(1)) + r'\b', body):
            via_local = decl.group(1)
            break
    assert inline or via_local, (
        'the `<Chart ...>` render site is not gated on the gap count. A holed '
        'series must not reach a charts.jsx primitive: declare e.g. '
        '`const plottable = Chart && gaps === 0;` and render the chart only '
        'when it holds.'
    )

    # ...and the bare `Chart` guard must be gone, so no path reaches the
    # primitive without the gap check.
    bare = re.search(r'\{\s*Chart\s*(?:\?|&&)(?![^\n]*\bgaps\b)', body)
    assert bare is None, (
        f'a chart render site is still guarded by `Chart` alone: '
        f'{bare.group(0)!r} — a holed series would reach the primitive '
        'through it.'
    )

    # (iii) a suppressed series must still DISCLOSE its gap count — silently
    #       withholding the sparkline would read as a render bug.  Asserted
    #       structurally (the `gaps` local reaches a JSX render position in
    #       comment-stripped source) rather than as operator-facing copy:
    #       pinning the sentence would fail the suite on any rewording while
    #       proving nothing extra about the branch, which is the rule this
    #       file states at lines 1134-1138 and 1203-1211.
    assert re.search(r'\{\s*gaps\b', code), (
        'the `gaps` count is computed but never reaches render position. A '
        'holed series must disclose how many samples are missing — the '
        'operator still gets value, current_value, n, denominator, direction '
        'and the verdict badge, and only the sparkline is withheld.'
    )


# ---------------------------------------------------------------------------
# step-9 test: escalation links, storm aggregate banner, unmatched list
# ---------------------------------------------------------------------------


def test_escalation_links_and_storm_aggregate_banner(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """The escalation affordances must be real, built only from fields the
    projection actually carries, and read the banner from the top-level block.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # (a) a metric row carrying an escalation renders a link, guarded so a null
    #     projection renders no dead control.
    assert 'data-testid="memory-eval-escalation-link"' in code, (
        'a metric row carrying `m.escalation` must render a '
        'data-testid="memory-eval-escalation-link" control.'
    )
    assert re.search(r'\{\s*escalation\.id\b', code), (
        'the escalation control must RENDER `escalation.id` (a `{escalation.id}` '
        'JSX expression), not merely mention it in a comment.'
    )
    assert re.search(r'\{\s*escalation\.summary\b', code), (
        'the escalation control must RENDER `escalation.summary` — an opaque '
        'id alone makes the operator click to find out what it is. Anchored to '
        'the JSX expression position, not a bare name grep.'
    )
    assert re.search(r'm\.escalation\s*&&', body), (
        'the escalation control must be guarded by `m.escalation &&` so a null '
        'projection renders nothing rather than a dead control.'
    )

    # (b) built from `id` only — the projection has no `url` field.
    assert not re.search(r'escalation\s*\.\s*url\b', body), (
        'the escalation projection has exactly six keys — id, summary, '
        'severity, level, created_at, dedupe_fingerprint. There is no `url`; '
        'the UI constructs its own affordance from `id`.'
    )

    # (c) fingerprints are rendered whole, never parsed.
    #     memory_evals.py:576-579 — the fingerprint is the producer's private
    #     construction; the dashboard must not depend on its substructure.
    for field in ('dedupe_fingerprint', 'fingerprint'):
        for op in (r'\.split\(', r'\.slice\(', r'\.match\(', r'\.substring\('):
            assert not re.search(rf'{field}\s*{op}', body), (
                f'`{field}` must be rendered whole — never split/sliced/matched. '
                'The fingerprint is the producer\'s private construction '
                '(memory_evals.py:576-579); parsing its substructure here would '
                'couple the dashboard to a format it does not own.'
            )

    # (d) the storm banner reads the TOP-LEVEL block, not an eval row's copy.
    assert 'data-testid="memory-eval-storm-banner"' in code, (
        'a data-testid="memory-eval-storm-banner" element must render when the '
        'top-level storm_escape block is non-null.'
    )
    assert re.search(r'\bstorm\.alarm_count\b', code), (
        'the storm banner must READ `storm.alarm_count` — how many alarms were '
        'collapsed into the one aggregate. Anchored to the accessing '
        'expression in comment-stripped source: the bare name also appears in '
        'the prose describing the banner.'
    )
    assert re.search(
        r'(payload|MEDF\.MEMORY_EVALS|MEMORY_EVALS)\s*\.\s*storm_escape', body
    ), (
        'the storm banner must read the TOP-LEVEL MEMORY_EVALS.storm_escape '
        '(memory_evals.py:958-964). The identical object repeated on each eval '
        "row exists only to explain that row's missing link; electing an "
        'arbitrary eval row to read the banner from would break on a root with '
        'zero eval dirs.'
    )

    # (e) per-metric links are suppressed under storm.
    assert re.search(
        r"parity\s*===\s*'storm_collapsed'", body
    ), (
        "the metric row must branch on `parity === 'storm_collapsed'` and "
        'render the suppression reason instead of a per-metric link — under '
        'storm the individual escalations were deliberately collapsed into the '
        'aggregate.'
    )

    # (f) unmatched_escalations BRANCHES on reason, with distinct wording.
    assert re.search(r'\bpayload\.unmatched_escalations\b', code), (
        'the `unmatched_escalations` list must be rendered from '
        '`payload.unmatched_escalations` — anchored to the read, not the bare '
        'name, which the surrounding prose also uses.'
    )
    reasons = ('no_matching_verdict', 'storm_suppressed', 'no_fingerprint')
    for reason in reasons:
        assert f"'{reason}'" in code, (
            f"the unmatched-escalations block must name the '{reason}' reason. "
            'Collapsing the three into one undifferentiated "unexplained" list '
            'would fire on escalations that are in fact fully explained and '
            'train operators to ignore the one signal that catches a real '
            'parity orphan (memory_evals.py:530-534).'
        )
    # Distinct wording, not three branches sharing one string.
    #
    # Asserted UNCONDITIONALLY over the extracted function body. The previous
    # form — `re.findall(r"reason === '(\w+)' \? '([^']+)'", body)` behind an
    # `if wordings:` guard — matched only single-quoted TERNARY expressions,
    # while unmatchedReasonText is written as sequential `if (...) return ...;`
    # statements and one branch returns a DOUBLE-quoted string. It yielded zero
    # matches against the real source, so the guard skipped the assertion in
    # silence and the suite reported coverage it did not have: collapsing all
    # three reasons onto one shared string — the exact regression this block
    # exists to prevent — would have stayed green. A vacuous assertion is worse
    # than no assertion, because it also stops anyone adding a real one.
    fn_body = _extract_function_body(code, 'unmatchedReasonText')
    assert fn_body, (
        'could not extract `unmatchedReasonText` — the per-reason wording must '
        'live in a named function so this contract is checkable.'
    )
    texts = [
        a or b
        for a, b in re.findall(
            r'return\s+(?:\'([^\']*)\'|"([^"]*)")', fn_body
        )
    ]
    assert len(texts) >= 3, (
        'unmatchedReasonText must return a distinct literal string for each of '
        f'the three reasons; found {len(texts)}: {texts}. Asserted with a '
        'floor so a refactor that DROPS branches fails loudly here rather '
        'than passing vacuously on an empty match list.'
    )
    assert len(set(texts)) == len(texts), (
        'each unmatched-escalation reason must get DISTINCT wording; '
        f'found duplicates in {texts}. Collapsing them into one '
        'undifferentiated "unexplained" list would fire on escalations that '
        'are in fact fully explained and train operators to ignore the one '
        'signal that catches a real parity orphan (memory_evals.py:530-534).'
    )
    assert all(t.strip() for t in texts), (
        f'every reason wording must be non-empty; found a blank in {texts}.'
    )


# ---------------------------------------------------------------------------
# step-11 test: the limits provenance block
# ---------------------------------------------------------------------------


_LIMITS_KEYS = (
    'alpha',
    'false_alarm_budget',
    'runs_per_quarter',
    'min_samples',
    'baseline_window',
    'baseline_run_stamps',
    'grandfather_set_hash',
    'run_stamp',
    'generator',
    'stale_for_latest_run',
)


def test_limits_provenance_rendered(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """Every limits-provenance key must be rendered, the staleness of the
    provenance itself must be disclosed, and a null limits artifact must say so.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # Anchored to `lim.<key>` in comment-stripped source: the provenance keys
    # are all named in the prose above LimitsProvenance, so a bare-name grep
    # would stay green after the render block was deleted.
    for key in _LIMITS_KEYS:
        assert re.search(rf'\blim\.{key}\b', code), (
            f"the limits provenance block must READ `lim.{key}` — the whole "
            'point of shipping provenance is that the operator can see which '
            'alpha / baseline the verdict was judged against.'
        )
    assert re.search(r'\bm\.rule_kind\b', code), (
        'the per-metric `m.rule_kind` must be rendered alongside the '
        'eval-level limits provenance.'
    )

    # stale_for_latest_run gates a VISIBLE disclosure — provenance stamped at an
    # older run must never be presented as governing a newer displayed run
    # (memory_evals.py:237-241).
    assert re.search(r'stale_for_latest_run\s*&&', code), (
        '`limits.stale_for_latest_run` must GATE a visible disclosure, not just '
        'be printed as one more field. Otherwise alpha/baseline provenance '
        'reads as governing the newer run actually on screen.'
    )

    # A null limits artifact renders an explicit state, not a blank block.
    # Asserted as a BRANCH gated on the right field, not as exact UI copy:
    # LimitsProvenance must early-return its own element when `ev.limits` is
    # falsy.  Pinning the operator-facing sentence would fail the suite on any
    # rewording while proving nothing extra about the branch.
    prov_body = _extract_function_body(code, 'LimitsProvenance')
    assert prov_body, 'could not extract the LimitsProvenance body.'
    # The local's NAME is derived from the `ev.limits` read rather than pinned,
    # so renaming `lim` is not a test failure; what must hold is that whatever
    # it is called gates the early return.
    lim_local = re.search(r'\b(\w+)\s*=\s*ev\.limits\b', prov_body)
    assert lim_local is not None, (
        'LimitsProvenance must read `ev.limits` into a local.'
    )
    lim = re.escape(lim_local.group(1))
    assert re.search(r'if\s*\(\s*!\s*' + lim + r'\s*\)\s*\{?\s*return\s*\(?\s*<', prov_body), (
        'a null `ev.limits` must take an early-return branch rendering an '
        'explicit element rather than an empty block — a blank provenance '
        'section is indistinguishable from a rendering bug.'
    )

    # Compact / expandable so provenance does not dominate the card.
    assert re.search(r'<details|useOpenSet|usePersistedState|localStorage', body), (
        'the provenance block must be collapsed-by-default and expandable '
        '(a <details> element or a persisted open-state key), so it does not '
        'dominate the eval card.'
    )

    # Local re-assertion of the step-7 guard: alpha and min_samples appear only
    # in render position, never as an operand of a comparison.
    for field in ('alpha', 'min_samples'):
        assert not re.search(rf'\.\s*{field}\b\s*(<=|>=|<|>)', body), (
            f'`{field}` is compared somewhere in tab_memory_evals.jsx. '
            'Provenance is DISPLAYED, never used to re-derive a verdict '
            '(PRD section 8, G6/INV-5).'
        )
        assert not re.search(rf'(<=|>=|<|>)\s*\w*\.\s*{field}\b', body), (
            f'`{field}` is compared somewhere in tab_memory_evals.jsx. '
            'Provenance is DISPLAYED, never used to re-derive a verdict '
            '(PRD section 8, G6/INV-5).'
        )


# ---------------------------------------------------------------------------
# step-13 test: staleness wording, empty states, issues notice
# ---------------------------------------------------------------------------


def test_staleness_empty_states_and_issues_notice(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """Staleness is a HINT, the two empty states are distinct, missing scalars
    are em-dashes rather than zeros, and artifact issues are loudly visible.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    # (a) latest-run age renders, and the stale branch carries no alarm wording.
    #     Anchored to `ev.<field>` reads in comment-stripped source — both names
    #     also appear in the prose above ageText().
    assert re.search(r'\bev\.latest_run_age_seconds\b', code), (
        'the eval card must READ `ev.latest_run_age_seconds` beside '
        '`ev.latest_run_stamp` so the operator can see how old the run is.'
    )
    assert re.search(r'\bev\.latest_run_stamp\b', code), (
        'the eval card must READ `ev.latest_run_stamp`.'
    )
    stale_branch = re.search(
        r'ev\.stale\s*&&\s*\(([\s\S]{0,700}?)\n\s*\)\}', body
    )
    assert stale_branch is not None, (
        '`ev.stale` must gate a visible hint badge.'
    )
    # Deliberately NOT asserted here: that the stale branch avoids the words
    # 'alarm'/'escalation'/'error'.  That was a wording pin over raw source
    # (comments included) — rewording the badge or adding an explanatory
    # comment inside the branch failed the suite with nothing functionally
    # changed.  The invariant it reached for (the dashboard never re-derives a
    # staleness alarm, PRD DD6/INV-5) is already enforced STRUCTURALLY: the 36h
    # threshold is absent from the payload, so there is no field the UI could
    # compare against.  Staleness is display-only here because the eval runner
    # self-escalates; a source-substring test cannot express that split.

    # (b) the two empty states are DISTINCT — absent root vs healthy-but-empty.
    assert 'data-testid="memory-eval-empty"' in code, (
        '`root_present === false` must render a '
        'data-testid="memory-eval-empty" placeholder.'
    )
    # Asserted as two DISTINCT BRANCHES gated on the right payload fields,
    # rather than as exact operator-facing copy: pinning the sentences would
    # fail the suite on any rewording while proving nothing about the branching.
    assert re.search(r'!\s*payload\.root_present\s*&&', code), (
        'the root-absent empty state must be gated on `!payload.root_present`.'
    )
    assert re.search(
        r'payload\.root_present\s*&&\s*evals\.length\s*===\s*0\s*&&', code
    ), (
        'root_present === true with zero evals is an empty-but-HEALTHY state '
        '(memory_evals.py:972-974) and needs its OWN branch, gated on '
        '`payload.root_present && evals.length === 0` — folding it into the '
        'root-absent message would report a working system as a broken one.'
    )

    # (c) missing scalars are em-dashes, never synthetic zeros.
    #     Anchored to the dash() helper's own RETURN in comment-stripped source.
    #     A bare `'—' in body` is the weakest possible form of this check: the
    #     em-dash appears 30-odd times in this file's prose (it is the house
    #     punctuation), so the assertion would survive deleting every dash the
    #     UI actually renders.
    #     Extracted brace-aware rather than matched with a bounded-window
    #     regex: `ageText()` sits directly below `dash()` and returns the same
    #     escape, so a `function dash\(...{0,200}?return '—'` window
    #     happily matches ACROSS the function boundary and survives gutting
    #     dash() entirely.
    dash_body = _extract_function_body(code, 'dash')
    assert dash_body, 'could not extract the dash() body.'
    assert re.search(r"return\s*'(—|\\u2014)'", dash_body), (
        'missing scalars must render the em-dash placeholder the Memory tab '
        'already uses (tabs.jsx:589, :648-653), via a `dash()` helper that '
        "returns the em-dash (literal or '\\u2014') for an absent value."
    )
    assert re.search(r'===\s*null|==\s*null|\?\?', dash_body), (
        'dash() must actually TEST for the absent case — a helper that returns '
        'the em-dash unconditionally, or never, is not a null guard.'
    )
    assert re.search(r'\bdash\s*\(\s*ev\.', code), (
        'the `dash()` helper must actually be APPLIED to payload scalars — '
        'defining it and then interpolating raw fields leaves the synthetic-'
        'zero hole open (feedback_redux_no_synthetic_data).'
    )
    for field in ('current_value', 'value', 'n', 'denominator', 'alarm_count'):
        assert not re.search(rf'\.\s*{field}\s*\|\|\s*0\b', body), (
            f'`{field}` is defaulted with `|| 0`. A synthetic zero reads as a '
            'measured zero — use the dash() helper so an absent measurement '
            'looks absent.'
        )

    # (d) the issues notice is VISIBLE and lists the detail, not just a count.
    assert 'data-testid="memory-eval-issues"' in code, (
        'artifact issues must render a data-testid="memory-eval-issues" notice.'
    )
    assert re.search(r'issue_count\s*>\s*0', code), (
        'the issues notice must be gated on `issue_count > 0`.'
    )
    issues_block = re.search(
        r'data-testid="memory-eval-issues"([\s\S]{0,1600})', body
    )
    assert issues_block is not None
    issues_text = issues_block.group(1)
    for field in ('kind', 'eval_id', 'path', 'detail'):
        assert re.search(rf'\.\s*{field}\b', issues_text), (
            f"the issues notice must list each issue's `{field}` — a bare count "
            'tells the operator something is wrong but not what, which is the '
            'silent-degradation failure the notice exists to prevent '
            '(INV-2/INV-4).'
        )
    assert not re.search(r'<details[^>]*>\s*<summary[^>]*>\s*\{?[^<]{0,40}issue', body, re.IGNORECASE), (
        'the issues notice must be expanded by default — collapsing a '
        'degraded-state notice reproduces the silent degradation it exists to '
        'prevent (INV-2/INV-4, the 2658 parse_failures precedent).'
    )

    # (e) payload age is visible.
    assert re.search(r'\bpayload\.generated_at\b', code), (
        '`payload.generated_at` must be READ and rendered so the operator can '
        'see payload age — anchored to the access, not a bare name grep.'
    )


# ---------------------------------------------------------------------------
# step-15 test: tabs.jsx MemoryTab renders the section (and nothing more)
# ---------------------------------------------------------------------------


def test_tabs_jsx_memory_tab_renders_evals_section(
    tabs_jsx_body: str, shell_jsx_body: str, app_jsx_body: str
) -> None:
    """tabs.jsx consumes the section at module top level and renders it inside
    MemoryTab — a SECTION in the existing Memory tab, not a new top-level tab.
    """
    body = tabs_jsx_body

    # (a) module-top-level destructure, unguarded like the DF_* lines above it.
    m = re.search(
        r'const\s*\{[^}]*\bMemoryEvalsSection\b[^}]*\}\s*=\s*window\.DF_MEMORY_EVALS',
        body,
    )
    assert m is not None, (
        'tabs.jsx must destructure MemoryEvalsSection off window.DF_MEMORY_EVALS '
        'at module top level, beside its existing window.DF_CHARTS / DF_SHELL / '
        "DF_DATA lines. index.html's load order (asserted in step-3) is what "
        'makes the unguarded form safe — the tab_scheduler.jsx:15 precedent.'
    )

    # (b) genuinely module-scope: before the MemoryTab declaration.
    fn_pos = body.find('function MemoryTab(')
    assert fn_pos != -1, 'tabs.jsx no longer declares `function MemoryTab(`.'
    assert m.start() < fn_pos, (
        'the DF_MEMORY_EVALS destructure sits after `function MemoryTab(` — it '
        'must be at module scope, matching the other DF_* destructures.'
    )

    # (c) rendered inside MemoryTab and nowhere else; NOT a DF_TABS entry.
    memory_tab_body = _extract_function_body(body, 'MemoryTab')
    assert memory_tab_body, 'could not extract the MemoryTab body.'
    assert '<MemoryEvalsSection' in memory_tab_body, (
        'MemoryTab must render <MemoryEvalsSection ... /> (PRD DD3: the eval '
        'view lives with the memory panels).'
    )
    assert body.count('<MemoryEvalsSection') == 1, (
        'MemoryEvalsSection must be rendered exactly once, inside MemoryTab.'
    )
    df_tabs_line = re.search(r'window\.DF_TABS\s*=\s*\{[^}]*\}', body)
    assert df_tabs_line is not None, 'tabs.jsx no longer exports window.DF_TABS.'
    assert 'MemoryEvalsSection' not in df_tabs_line.group(0), (
        'MemoryEvalsSection must NOT be added to window.DF_TABS — it is a '
        'section inside the Memory tab, not a thirteenth top-level tab '
        '(PRD DD3). Asserting the negative keeps a later refactor from quietly '
        'promoting it.'
    )

    # (d) app.jsx still routes memory to MemoryTab; no new Rail entry.
    assert re.search(r"case\s*'memory'\s*:\s*return\s*<MemoryTab", app_jsx_body), (
        "app.jsx must still route `case 'memory':` to <MemoryTab."
    )
    for rail_id in ("'memevals'", "'memory-evals'", "'memory_evals'"):
        assert rail_id not in shell_jsx_body, (
            f'shell.jsx gained a Rail entry {rail_id} — PRD DD3 places this as '
            'a section in the existing Memory tab, with no new rail item.'
        )


# ---------------------------------------------------------------------------
# step-17 test: the escalation link actually navigates
# ---------------------------------------------------------------------------


def test_escalation_link_navigation_is_wired(
    app_jsx_body: str,
    tabs_jsx_body: str,
    tab_escalations_jsx_body: str,
    tab_memory_evals_jsx_body: str,
) -> None:
    """The link from step-10 must navigate for real.

    The SPA has no router — zero `location.hash` and zero `href=` uses across
    static/redux, and tab state is plain React state in app.jsx.  So without
    this plumbing the control would be a dead affordance, which is exactly the
    synthetic-UI failure feedback_redux_no_synthetic_data forbids.  The handoff
    therefore rides the existing state-lift idiom: app.jsx owns the focus state
    and threads a handler down.
    """
    # (a) the handler wired to <MemoryTab onNavigate={...}> must switch the tab
    #     AND record the focus id into the SAME state <EscalationsTab focusId=
    #     {...}> reads.
    #
    #     Every identifier below is DERIVED from the cross-file prop contract,
    #     never pinned by spelling: renaming the handler (navigate -> goToTab)
    #     or the state (escFocus -> focusedEsc) keeps this green.  Deriving is
    #     also STRICTLY STRONGER than the spelling pin it replaces — that
    #     version checked for a setter by name, so a handler writing to some
    #     OTHER state than the one feeding EscalationsTab passed it. Here that
    #     is a failure, which is the actual dead-affordance this test exists to
    #     catch.
    handler = re.search(r'<MemoryTab[^>]*onNavigate=\{(\w+)\}', app_jsx_body)
    assert handler is not None, (
        'app.jsx must pass a named handler to <MemoryTab onNavigate={...}>.'
    )
    focus_state = re.search(r'<EscalationsTab[^>]*focusId=\{(\w+)\}', app_jsx_body)
    assert focus_state is not None, (
        'app.jsx must pass a state variable to <EscalationsTab focusId={...}>.'
    )
    setter = re.search(
        r'const\s*\[\s*' + re.escape(focus_state.group(1)) + r'\s*,\s*(\w+)\s*\]\s*=\s*uS\(',
        app_jsx_body,
    )
    assert setter is not None, (
        f'the id passed to <EscalationsTab focusId must be React state, but no '
        f'`const [{focus_state.group(1)}, set...] = uS(` declaration exists.'
    )
    nav = re.search(
        r'const\s+' + re.escape(handler.group(1))
        + r'\s*=\s*\([^)]*\)\s*=>\s*\{([\s\S]{0,300}?)\}',
        app_jsx_body,
    )
    assert nav is not None, (
        f'app.jsx passes `{handler.group(1)}` to <MemoryTab but never defines '
        'it as an arrow-function handler.'
    )
    nav_body = nav.group(1)
    assert setter.group(1) + '(' in nav_body, (
        f'`{handler.group(1)}` must record the focus id via `{setter.group(1)}(`, '
        'the same state <EscalationsTab reads. Writing it anywhere else means '
        f'the link switches tab and lands on an unfocused list. Body: {nav_body!r}'
    )
    assert set(re.findall(r'\b(set\w+)\s*\(', nav_body)) - {setter.group(1)}, (
        f'`{handler.group(1)}` records the focus id but never switches the tab, '
        f'so the link highlights an escalation the operator cannot see. '
        f'Body: {nav_body!r}'
    )

    # (b)/(c) the handler reaches MemoryTab and the focus id reaches EscalationsTab.
    assert re.search(r'<MemoryTab[^>]*onNavigate=\{', app_jsx_body), (
        'app.jsx must pass onNavigate to <MemoryTab at the `case \'memory\':` '
        'branch — otherwise the section renders its link disabled.'
    )
    assert re.search(r'<EscalationsTab[^>]*focusId=\{', app_jsx_body), (
        'app.jsx must pass the focus id into <EscalationsTab.'
    )
    assert re.search(r'<EscalationsTab[^>]*onFocusConsumed=\{', app_jsx_body), (
        'app.jsx must pass onFocusConsumed into <EscalationsTab so the focus '
        'clears after it is used.'
    )

    # (d) tabs.jsx forwards it.
    assert re.search(r'function\s+MemoryTab\s*\(\s*\{[^}]*\bonNavigate\b', tabs_jsx_body), (
        'MemoryTab must accept `onNavigate` in its props destructure.'
    )
    assert re.search(r'<MemoryEvalsSection[^>]*onNavigate=\{', tabs_jsx_body), (
        'MemoryTab must forward onNavigate to <MemoryEvalsSection.'
    )

    # (e) tab_escalations.jsx consumes the focus and clears it.
    assert re.search(
        r'function\s+EscalationsTab\s*\(\s*\{[^}]*\bfocusId\b', tab_escalations_jsx_body
    ), 'EscalationsTab must accept a `focusId` prop.'
    assert re.search(
        r'function\s+EscalationsTab\s*\(\s*\{[^}]*\bonFocusConsumed\b',
        tab_escalations_jsx_body,
    ), 'EscalationsTab must accept an `onFocusConsumed` prop.'
    effect = re.search(
        r'uE\(\(\)\s*=>\s*\{([\s\S]{0,900}?)\n\s*\},\s*\[[^\]]*focusId[^\]]*\]\)',
        tab_escalations_jsx_body,
    )
    assert effect is not None, (
        'EscalationsTab must run a `uE` effect keyed on `focusId`.'
    )
    eff = effect.group(1)
    assert 'setSelected(' in eff, (
        'the focus effect must call the existing `setSelected(row)` to open the '
        'detail sidebar — reusing the tab\'s own selection mechanism rather '
        'than inventing a second one.'
    )
    assert 'onFocusConsumed' in eff, (
        'the focus effect must call onFocusConsumed() — including when no row '
        'matches (the escalation may have closed between poll and click). '
        'Leaving the focus set would reopen a stale drawer on every later '
        'visit to the tab.'
    )

    # (f) the producer end of the contract, re-asserted here.
    assert re.search(r"onNavigate\(\s*'esc'\s*,\s*escalation\.id\s*\)", tab_memory_evals_jsx_body), (
        "tab_memory_evals.jsx's link must call onNavigate('esc', escalation.id)."
    )
