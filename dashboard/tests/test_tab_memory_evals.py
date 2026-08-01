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
    hypothetical — it is exactly what `_PARITIES` (below) documents finding for
    `alarmed_open`/`clear`, where the string existed only in prose.

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
    assert re.search(r'\b\w+\s*:\s*\w+', destructured), (
        'chart primitives must be aliased to file-unique names (e.g. '
        '`Sparkline: MESpark`), following the codebase per-file alias '
        f'convention. Got: {destructured!r}'
    )
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
    # Word-boundary matched, not bare substring: a plain `'d3' in body` fires on
    # any prose mentioning "PRD DD3", which is a false positive, not a library.
    for lib in (r'd3', r'chart\.js', r'recharts', r'plotly'):
        assert not re.search(rf'\b{lib}\b', body, re.IGNORECASE), (
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

# The server-derived display states that verdictBadge actually BRANCHES on.
# memory_evals.py:660-661: `parity` "keeps the UI from re-deriving badge state
# out of three separate fields, which is where the two sides would drift apart".
#
# `alarmed_open` and `clear` are deliberately absent.  verdictBadge has no
# branch for them by design — they agree with the verdict, so they fall through
# to the plain verdict badge.  Asserting their string appears in the source
# only ever matched the explanatory COMMENTS (tab_memory_evals.jsx:84 and :100);
# the strings occur nowhere in code.  That pinned comment wording in place —
# rewording it would fail this suite with nothing functionally changed — while
# giving zero coverage of how those states render.  "Falls through to the plain
# badge" is a claim about which branch does NOT execute, which a source-
# substring test structurally cannot make; only a DOM/render test could.
_PARITIES = (
    'alarmed_unlinked',
    'recovered_open',
    'storm_collapsed',
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
    for parity in _PARITIES:
        assert f"'{parity}'" in code, (
            f"tab_memory_evals.jsx must name the server-derived parity state "
            f"'{parity}' in CODE — the comment block above verdictBadge names "
            'all of them, so a whole-file grep proves nothing.'
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
    # Asserted as a FALL-THROUGH BRANCH, not as exact badge copy: after every
    # recognised verdict is matched, the function must still return a badge of
    # its own (a muted/info class), so an unrecognised verdict renders a
    # distinct visible state.  Pinning the literal words "no verdict" would
    # fail on any rewording without testing the branch.
    tail = badge_body[badge_body.rfind('insufficient_data'):]
    assert re.search(r"return\s*\{[^}]*\bcls\s*:\s*'badge (muted|info)'", tail), (
        'verdictBadge must END with an unconditional return of a muted/info '
        'badge — the fall-through for a null/unrecognised verdict. Absent is '
        'absent: it needs its own visible state, never a silent default.'
    )
    assert not re.search(r"return\s*\{[^}]*\bcls\s*:\s*'badge ok'", tail), (
        'the fall-through badge for an unrecognised verdict must NOT reuse the '
        "no_alarm 'badge ok' styling — that renders \"we did not measure\" as "
        '"we measured and it is fine".'
    )
    assert not re.search(
        r"verdict\s*(\|\||\?\?)\s*['\"]no_alarm['\"]", body
    ), (
        'a null verdict must NOT be defaulted to no_alarm — that would report '
        '"we did not measure" as "we measured and it is fine".'
    )


def test_no_client_side_alarm_derivation(
    tab_memory_evals_jsx_body: str,
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

    # (1) The badge helper reads verdict/parity and performs NO comparison.
    badge_body = _extract_function_body(body, 'verdictBadge')
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

    # (iii) a suppressed series must still disclose its gap count AND say that
    #       no chart was drawn.  Both on one line: a count without the
    #       no-chart statement reads as "drawn, with N gaps".
    assert re.search(
        r'(?:\$\{gaps\}|\{gaps\})[^\n]*no chart|no chart[^\n]*(?:\$\{gaps\}|\{gaps\})',
        body,
    ), (
        'a holed series must disclose BOTH its gap count and the fact that no '
        'chart is drawn. The operator still gets value, current_value, n, '
        'denominator, direction and the verdict badge — only the sparkline is '
        'withheld, and silently withholding it would look like a render bug.'
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
    wordings = re.findall(r"reason\s*===\s*'(\w+)'\s*\?\s*'([^']+)'", body)
    if wordings:
        texts = [w[1] for w in wordings]
        assert len(set(texts)) == len(texts), (
            'each unmatched-escalation reason must get DISTINCT wording; '
            f'found duplicates in {texts}.'
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
    assert re.search(r'\blim\s*=\s*ev\.limits\b', prov_body), (
        'LimitsProvenance must read `ev.limits`.'
    )
    assert re.search(r'if\s*\(\s*!\s*lim\s*\)\s*\{?\s*return\s*\(?\s*<', prov_body), (
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
    stale_text = stale_branch.group(1).lower()
    for banned in ('alarm', 'escalation', 'error'):
        assert banned not in stale_text, (
            f"the stale branch uses the word {banned!r}. Staleness is DISPLAYED "
            "here and alarmed on by the eval runner's own self-escalation "
            '(PRD DD6/INV-5) — a second, divergent staleness alarm in the '
            'dashboard is exactly what that split avoids. The 36h threshold is '
            'deliberately absent from the payload so the UI cannot re-derive '
            f'it either. Branch was:\n{stale_branch.group(1)}'
        )

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
    # (a) app.jsx holds an escalation-focus state beside its tab state, and a
    #     navigate handler that sets both.
    assert re.search(r'const\s*\[\s*escFocus\s*,\s*setEscFocus\s*\]\s*=\s*uS\(', app_jsx_body), (
        'app.jsx must hold an escalation-focus state beside `const [tab, setTab] '
        "= uS('overview')`."
    )
    nav = re.search(r'const\s+navigate\s*=\s*\([^)]*\)\s*=>\s*\{([\s\S]{0,300}?)\}', app_jsx_body)
    assert nav is not None, (
        'app.jsx must define a `navigate` handler.'
    )
    assert 'setTab(' in nav.group(1) and 'setEscFocus(' in nav.group(1), (
        'the navigate handler must both switch the tab and record the focus id; '
        f'body was: {nav.group(1)!r}'
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
