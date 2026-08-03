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
# Helper: find the innermost `function` declaration enclosing an offset
# ---------------------------------------------------------------------------


def _enclosing_function(src: str, idx: int) -> tuple[str, list[str]] | None:
    """Return ``(name, param_names)`` of the innermost ``function`` enclosing ``idx``.

    Returns ``None`` when ``idx`` sits at module scope.

    WHY A BRACE WALK AND NOT A REGEX SPAN.  The obvious spelling —
    ``function\\s+(\\w+)\\s*\\(([^)]*)\\)\\s*\\{[\\s\\S]{0,300}?<call>`` — silently
    matches ACROSS function boundaries: with a short helper declared just above
    the one that actually makes the call, the bounded ``[\\s\\S]`` span reaches
    past the helper's closing brace and attributes the call to the WRONG
    function, along with the wrong parameter list.  Measured, not argued: over
    the shipped ``provOpenKey`` / ``readProvOpen`` pair that span attributes
    ``localStorage.getItem(key)`` to ``provOpenKey(evalId)`` and reports the
    key as not-a-parameter — a false FAILURE against correct code.  The walk
    below is exact.
    """
    best: tuple[str, list[str]] | None = None
    best_start = -1
    for m in re.finditer(r'\bfunction\s+(\w+)\s*\(', src):
        paren_depth = 1
        i = m.end()
        while i < len(src) and paren_depth > 0:
            if src[i] == '(':
                paren_depth += 1
            elif src[i] == ')':
                paren_depth -= 1
            i += 1
        if paren_depth != 0:
            continue
        params_text = src[m.end() : i - 1]
        start = src.find('{', i)
        if start == -1:
            continue
        depth = 0
        end = -1
        for j in range(start, len(src)):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            continue
        if start < idx < end and m.start() > best_start:
            best_start = m.start()
            best = (
                m.group(1),
                [p.strip() for p in params_text.split(',') if p.strip()],
            )
    return best


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
# Helper: the JSX's two parity declarations, parsed once
# ---------------------------------------------------------------------------


# One `'state': { ... }` entry of PARITY_REFINEMENT.  Declared once and shared:
# it is the non-trivial invariant three tests depend on, and three verbatim
# copies would have to be fixed three times.
_PARITY_ENTRY_RE = r'[\'"]?(\w+)[\'"]?\s*:\s*\{([^{}]*)\}'


def _parity_tables(code: str) -> tuple[dict[str, str], set[str]]:
    """Parse tab_memory_evals.jsx's parity vocabulary into ``(entries, declined)``.

    ``entries`` maps a refined parity state to the raw body of its
    ``{ suffix, cls }`` object; ``declined`` is the set of states listed in
    ``PARITY_PLAIN``.  Both declarations must exist and be non-empty, and every
    ``PARITY_REFINEMENT`` value must be a ``{ suffix, cls }`` pair — those are
    preconditions of every caller, so they are asserted HERE with one canonical
    message rather than re-stated per test.

    The shape check runs against the top-level keys of the literal, not against
    what the entry regex happened to match: an entry written with a non-object
    value would otherwise be silently absent from ``entries`` and get reported
    by the vocabulary test as an *unhandled state*, sending the reader to look
    for a missing declaration that is right there.
    """
    refinement = _extract_const_object(code, 'PARITY_REFINEMENT')
    assert refinement, (
        'tab_memory_evals.jsx must declare `const PARITY_REFINEMENT = {...}` at '
        'module scope in CODE (not in a comment). It is the one place the file '
        'says which parity states change the badge, and what these tests '
        'compare against memory_evals.PARITY_STATES.'
    )
    plain = _extract_const_object(code, 'PARITY_PLAIN', open_char='[')
    assert plain, (
        'tab_memory_evals.jsx must declare `const PARITY_PLAIN = [...]` at '
        'module scope — the EXPLICIT opt-out list. Without it, a state that is '
        'merely unhandled is indistinguishable from one deliberately left to '
        'the plain verdict badge, and these tests cannot tell a considered '
        'decision from an oversight.'
    )

    # Keys are quoted in the source (they are producer vocabulary strings, not
    # JS identifiers), but the quotes are tolerated rather than required here:
    # the presence grep in `test_verdict_badges_driven_by_persisted_verdict` is
    # what enforces the quoted form, and it says so clearly. Requiring them
    # here too would report a dropped quote as "the table holds no entries".
    entries = dict(re.findall(_PARITY_ENTRY_RE, refinement))
    declined = set(re.findall(r"'([^']*)'", plain))

    # Top-level keys, found by deleting the one level of nesting the literal
    # has (the `{ suffix, cls }` values) and reading what keys remain.
    declared = set(
        re.findall(r'[\'"]?(\w+)[\'"]?\s*:', re.sub(r'\{[^{}]*\}', '', refinement[1:-1]))
    )
    malformed = declared - set(entries)
    assert malformed == set(), (
        f'PARITY_REFINEMENT entr(ies) {sorted(malformed)} do not hold a '
        '`{ suffix, cls }` object literal. Storing a whole label (or anything '
        'else) would let a parity branch report a state the payload never '
        'asserted — `_parity()` derives most states from (verdict class, '
        'linked?), so the verdict must survive into the label.'
    )
    for key, value in entries.items():
        keys = set(re.findall(r'(\w+)\s*:', value))
        assert keys == {'suffix', 'cls'}, (
            f"PARITY_REFINEMENT['{key}'] must be a {{ suffix, cls }} pair, got "
            f'keys {sorted(keys)}. A `suffix` is composed onto the '
            'verdict-derived base and a `cls` selects an existing badge class; '
            'anything else is not consumed by verdictBadge at all.'
        )

    assert entries, 'PARITY_REFINEMENT is declared but holds no entries.'
    assert declined, 'PARITY_PLAIN is declared but holds no states.'
    return entries, declined


def _return_label_exprs(badge_body: str) -> list[str]:
    """Every `label:` expression in ``verdictBadge``'s ``return { ... }`` sites.

    Each is captured to the END of its return object rather than to the first
    comma: a label expression that legitimately contains one (a function call,
    a string with a comma in it) would otherwise be silently truncated and the
    assertions below would then hold — or fail — for the wrong reason.
    """
    exprs: list[str] = []
    for m in re.finditer(r'return\s*\{', badge_body):
        start = m.end() - 1
        depth = 0
        obj = ''
        for i in range(start, len(badge_body)):
            if badge_body[i] == '{':
                depth += 1
            elif badge_body[i] == '}':
                depth -= 1
                if depth == 0:
                    obj = badge_body[start + 1 : i]
                    break
        label = re.search(r'\blabel\s*:\s*', obj)
        if label:
            exprs.append(obj[label.end() :].strip().rstrip(',').strip())
    return exprs


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

    # NOTE: no cache-buster assertion here. The "all /static/redux/?v= share
    # ONE version" check is asserted once, canonically, in test_index_html.py —
    # it was replicated across five test modules, each with its own stale
    # monotonic floor, so every dashboard change re-failed several byte-identical
    # copies. The monotonic floor this task needs lives in
    # test_index_html_cache_buster_floor below.


def test_index_html_cache_buster_floor(index_html_body: str) -> None:
    """Every /static/redux/* asset must be at or past this task's version.

    Two of the three fixes in task 3470 are pure edits to .jsx bundles that an
    already-open dashboard has ALREADY cached — tab_memory_evals.jsx (per-eval
    provenance key, empty-trend state) and tab_escalations.jsx (focus-miss and
    focus-pending feedback). Neither adds a route, a template or any other
    server-side change that would force the browser to refetch. Without a bump
    an open dashboard keeps running the buggy bundle indefinitely: the operator
    sees every provenance <details> pop open on the next 3s poll, a blank 26px
    box where an empty trend should name itself, and a dead cross-tab link,
    while the fixed source sits on disk unread.

    Bumping is therefore part of the fix, not a cosmetic afterthought, so it is
    pinned by a test like every prior bump was.

    Scope, deliberately: the FLOOR only. Whether the versions are UNIFORM is one
    property of index.html, and asserting it here too made this the sixth
    byte-identical copy of that check (test_index_html.py, test_esc_flow_diagram
    .py, test_tab_escalation_analytics.py, test_scheduler_page.py, and twice in
    this module) — so any partial bump failed six tests with six different stale
    floors in the message, and a reader had to diff them to find the
    authoritative one. Uniformity is now asserted once, in test_index_html.py.
    A floor stated as `min(...)` needs no uniformity precondition to be sound:
    it is the strictly stronger claim under mixed versions, since the oldest
    asset is the one that would still serve stale module code.
    """
    versions = {
        int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)
    }
    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 42, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        "expected >= 42 so task 3470's tab_memory_evals.jsx and "
        'tab_escalations.jsx fixes actually reach already-open browsers. This '
        'floor supersedes the two it subsumes: >= 41 (test_index_html.py) and '
        '>= 38 (the memory-evals section landing in task 3216).'
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

    # `_parity_tables` also asserts both declarations exist and that every
    # table value is a `{ suffix, cls }` pair — the composition invariant in
    # its structural form: table values are SUFFIXES, never whole labels, so
    # there is no expression in this file capable of returning a label that
    # discards the verdict-derived base.
    entries, declined = _parity_tables(code)
    handled = set(entries)

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

    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'
    labels = _return_label_exprs(badge_body)
    assert labels, 'verdictBadge returns no `label`.'
    for expr in labels:
        # `base\b`, not `startswith('base')`: the latter also accepts an
        # unrelated identifier such as `baseline` or `baseLabel`, so it would
        # not actually pin composition onto the verdict-derived `base`.
        assert re.match(r'base\b', expr), (
            f'verdictBadge returns the label {expr!r}, which does not '
            'begin with the verdict-derived `base`. Every returned label must '
            'compose onto it: a fixed label reports a state the payload never '
            'asserted. `_parity()` is a three-case lookup over (verdict class, '
            'linked?) that gives every class its OWN pair of states — a fixed '
            'label collapses that back, e.g. reporting a metric nothing judged '
            '(`unjudged_open`, `insufficient_data_open`) as having recovered.'
        )
    assert any('suffix' in e for e in labels), (
        'no verdictBadge return composes a PARITY_REFINEMENT `suffix` onto '
        '`base` — the table is declared but never consumed, so every refined '
        'state renders as the plain verdict badge.'
    )


def test_every_open_parity_state_shows_the_escalation_affordance(
    tab_memory_evals_jsx_code: str,
) -> None:
    """A live escalation must be visible on the badge, for every `*_open` state.

    The requirement is DERIVED from the producer — `{s for s in PARITY_STATES
    if s.endswith('_open')}` — rather than listed here, which is the whole
    point: `_parity()` suffixes `_open` onto the linked variant of every
    non-alarm verdict class, so the next class the producer adds arrives with
    an `_open` partner and fails here instead of rendering unaffordanced.

    A `*_open` state falling through to the plain verdict badge tells the
    operator nothing about the live escalation behind the row — the badge reads
    exactly like the same metric with no escalation at all, which is the
    distinction the parity field was derived to make.
    """
    from dashboard.data.memory_evals import PARITY_STATES

    code = tab_memory_evals_jsx_code
    open_states = {s for s in PARITY_STATES if s.endswith('_open')}
    assert open_states, (
        'no member of memory_evals.PARITY_STATES ends in `_open` — the naming '
        'convention this assertion derives from is gone, so it would pass '
        'vacuously. Re-derive the affordance requirement from whatever '
        'replaced the suffix before deleting this guard.'
    )

    entries, declined = _parity_tables(code)

    for state in sorted(open_states):
        assert state not in declined, (
            f"'{state}' is in PARITY_PLAIN, so it renders as the bare verdict "
            'badge. But an escalation is OPEN on that row: the badge is then '
            'identical to the same verdict with nothing filed, and the '
            'operator has no way to tell a live escalation from none. States '
            'may decline refinement only when the verdict badge already says '
            'everything there is to say, which is never true while an '
            'escalation is open.'
        )
        assert state in entries, (
            f"'{state}' has no PARITY_REFINEMENT entry, so nothing on its "
            'badge discloses the open escalation the `_open` suffix asserts.'
        )
        suffix = re.search(r"suffix\s*:\s*'([^']*)'", entries[state])
        assert suffix, f"PARITY_REFINEMENT['{state}'] has no `suffix` string."
        assert 'escalation open' in suffix.group(1), (
            f"PARITY_REFINEMENT['{state}'] renders the suffix "
            f'{suffix.group(1)!r}, which never says the escalation is open. '
            "Today only `recovered_open` carries that affordance; every "
            '`_open` state means the same thing — an escalation is linked and '
            'still live — and must say so in the same words, so the operator '
            'reads one marker rather than learning six.'
        )
        # Severity follows the VERDICT, not the linkage — but never DOWN to
        # healthy. Without this, flipping the four warn-level `_open` entries
        # to 'badge ok' keeps the whole suite green while colouring a row that
        # has a live open escalation the same green as one with nothing filed.
        cls = re.search(r"cls\s*:\s*'([^']*)'", entries[state])
        assert cls, f"PARITY_REFINEMENT['{state}'] has no `cls` string."
        assert cls.group(1) != 'badge ok', (
            f"PARITY_REFINEMENT['{state}'] renders 'badge ok'. An escalation is "
            'OPEN on that row, so the badge may not be the healthy one: colour '
            'is the first thing scanned and a green row is not looked at '
            'twice, which would bury the very fact the suffix was added to '
            'disclose. Severity may follow the verdict down to warn, never to '
            'healthy.'
        )


def test_unknown_verdict_is_visibly_unrenderable(
    tab_memory_evals_jsx_code: str,
) -> None:
    """An unrecognised verdict must read as broken — not as absent, not as fine.

    Server-side this is a NAMED issue kind: the reader files an
    `unknown_verdict` issue naming the eval, metric and offending value,
    precisely so a value outside the closed vocabulary fails toward "something
    is wrong here".  This is the last render step and must not quietly undo it.

    Two substitutions are refused, because an operator reads the rendered badge
    and nothing else:

    * the CLASS may borrow neither the healthy badge nor the neutral
      not-measured one — the row is broken, not fine and not unmeasured;
    * the LABEL may not say the verdict is MISSING.  It is present, just
      unreadable, and the two are different facts.  Contrast `unjudged`, which
      correctly stays plain and muted reading 'no verdict': nothing judged that
      metric at all, so there that label is exactly true.

    Asserted over RENDERED OUTPUT (badge class and the label the operator
    reads), not over comment prose.
    """
    from dashboard.data.memory_evals import PARITY_STATES

    code = tab_memory_evals_jsx_code
    unknown = {s for s in PARITY_STATES if s.startswith('unknown_verdict')}
    assert unknown, (
        'no member of memory_evals.PARITY_STATES starts with '
        '`unknown_verdict` — this assertion would pass vacuously. Re-derive it '
        'from whatever the producer renamed the out-of-vocabulary state to '
        'before deleting this guard.'
    )

    entries, declined = _parity_tables(code)

    for state in sorted(unknown):
        assert state not in declined, (
            f"'{state}' is in PARITY_PLAIN, so it renders as the plain verdict "
            'badge — muted, with nothing marking the row as broken. That '
            'reports a present-but-unreadable verdict as an unremarkable one, '
            'which is the same substitution the absent-verdict guard exists to '
            'prevent, running the other way. The producer files a named '
            '`unknown_verdict` issue for this; the last render step must not '
            'quietly undo it.'
        )
        assert state in entries, (
            f"'{state}' has no PARITY_REFINEMENT entry, so it renders through "
            'the fall-through with nothing marking the verdict as unreadable.'
        )
        cls = re.search(r"cls\s*:\s*'([^']*)'", entries[state])
        assert cls, f"PARITY_REFINEMENT['{state}'] has no `cls` string."
        assert cls.group(1) not in ('badge ok', 'badge muted'), (
            f"PARITY_REFINEMENT['{state}'] renders {cls.group(1)!r}. It may "
            'borrow neither the healthy badge nor the neutral not-measured '
            'one: the value is present and outside the vocabulary, so the row '
            'is broken, not fine and not unmeasured.'
        )
    # The condition is named by the BASE, not by a parity suffix. `base` is the
    # verdict-derived half of every label, so it is the only half that can say
    # what happened to the VERDICT; a suffix appended to 'no verdict' cannot
    # un-say it, and the composed label would then assert both halves of the
    # distinction at once ("no verdict · unrecognised verdict" — absent in the
    # first clause, present-but-unreadable in the second).
    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'
    assigned = re.findall(r"\bbase\s*=\s*'([^']*)'", badge_body)
    assert assigned, 'verdictBadge assigns no literal `base` label.'
    absent = assigned[0]  # the seed, in force until a verdict is recognised
    unreadable = [v for v in assigned if re.search(r'unread|unrecognis|unknown', v, re.I)]
    assert unreadable, (
        'verdictBadge never assigns a `base` label naming an UNREADABLE '
        f'verdict — the literals it can render are {sorted(set(assigned))}. A '
        'verdict that is present but outside the closed vocabulary reaches '
        f'parity {sorted(unknown)}, and the operator must be able to see that '
        'from the label: memory_evals._verdict_class() buckets an absent value '
        'to `unjudged` and a present-but-out-of-vocabulary one to '
        '`unknown_verdict` precisely because they are different facts.'
    )
    assert absent not in unreadable, (
        f'verdictBadge seeds `base` to {absent!r} and uses that same label for '
        'an unreadable verdict, so the two states the producer distinguishes '
        'render identically. Seed the genuinely-ABSENT label and overwrite it '
        'on a present-but-unrecognised value.'
    )


def test_unrecognised_parity_is_marked_not_passed_through(
    tab_memory_evals_jsx_code: str,
) -> None:
    """A parity value in NEITHER declaration must render as visibly unknown.

    Two ways the badge can be wrong about a parity string it has never been
    told about, and the file refuses both:

    * an INHERITED member.  `PARITY_REFINEMENT` is a plain object literal, so a
      bare `PARITY_REFINEMENT[parity]` resolves through `Object.prototype`: a
      payload carrying parity 'constructor' / 'toString' / 'valueOf' /
      'hasOwnProperty' finds a truthy inherited function and the row renders
      `cls: undefined` with a label ending '· undefined'.  The lookup must be
      own-property guarded.
    * a genuinely NEW state — the producer added one and an already-open
      browser is still holding a cached copy of this file (the case the
      cache-buster bump exists for, and which it cannot make impossible).
      Passing it through to the plain verdict badge renders it identically to a
      state that DECLINED refinement: the unknown fails toward the healthy
      label, which is what the unknown-verdict pair exists to prevent one level
      up.  `PARITY_PLAIN` is what makes "considered and declined" and "never
      heard of it" distinguishable at all, so the fall-through must consult it.
    """
    code = tab_memory_evals_jsx_code
    badge_body = _extract_function_body(code, 'verdictBadge')
    assert badge_body, 'could not extract the verdictBadge body.'

    assert 'hasOwnProperty' in badge_body, (
        'verdictBadge reads PARITY_REFINEMENT with an unguarded bracket '
        'lookup. That resolves through Object.prototype, so a parity of '
        "'constructor' or 'toString' returns a truthy INHERITED member and the "
        "badge renders `cls: undefined`. Guard it with "
        '`Object.prototype.hasOwnProperty.call(PARITY_REFINEMENT, parity)`.'
    )
    assert 'PARITY_PLAIN' in badge_body, (
        'verdictBadge never consults PARITY_PLAIN, so it cannot tell a state '
        'that deliberately declined refinement from one it has never heard of '
        '— both take the same fall-through and render the same badge. The '
        'opt-out list has to be READ for it to mean anything at render time.'
    )

    marked = [
        obj
        for obj in re.findall(r'return\s*\{([^{}]*)\}', badge_body)
        if re.search(r'unrecognis|unknown', obj, re.IGNORECASE)
    ]
    assert marked, (
        'verdictBadge has no return that marks an unrecognised parity. A '
        'parity in neither PARITY_REFINEMENT nor PARITY_PLAIN is a state this '
        'copy of the file has never been told about; rendering it as the bare '
        'verdict badge tells the operator the row is ordinary, which is the '
        'one thing not known about it.'
    )
    for obj in marked:
        cls = re.search(r"cls\s*:\s*'([^']*)'", obj)
        assert cls, 'the unrecognised-parity return sets no literal `cls`.'
        assert cls.group(1) not in ('badge ok', 'badge muted'), (
            f'the unrecognised-parity return renders {cls.group(1)!r}. An '
            'unknown state may borrow neither the healthy badge nor the '
            'neutral not-measured one — both assert something about the row '
            'that this bundle has no basis for.'
        )


def test_verdict_badges_driven_by_persisted_verdict(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """Badges must name every persisted verdict and every server-derived
    parity state, and an absent verdict must render its own explicit state.

    Defaulting a null verdict to `no_alarm` would turn "we do not know" into
    "we checked and it is fine" — the builder itself refuses that in
    `memory_evals._verdict_class()`, which buckets an absent value to
    `unjudged` rather than to a verdict, and the UI must not undo it.
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

    `verdict` and `parity` are the ONLY badge inputs.  The docstring of
    `memory_evals._parity()` records why: `parity` exists precisely so "the UI
    [does not] re-deriv[e] badge state out of three separate fields, which is
    where the two sides would drift apart".  A browser-side value-vs-limit comparison would be a
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


def test_empty_trend_is_a_named_state_not_an_empty_chart_box(
    tab_memory_evals_jsx_body: str,
    tab_memory_evals_jsx_code: str,
) -> None:
    """A metric with NO runs is a third suppression state, not a chart.

    The defect: `trendGaps([])` counts zero holes in zero samples, so the
    `Chart && gaps === 0` gate is TRUE for a metric that has never been
    measured.  The empty series is then handed to a charts.jsx primitive —
    and both `Sparkline` (charts.jsx:58) and its `StepSpark` twin `return null`
    on a zero-length array.  The cell therefore renders an empty 26px <div>
    plus a "0 pts" footer: a blank box that is indistinguishable from a
    rendering bug, which is precisely the failure mode the gap-suppression
    path already exists to avoid.  The deliberate 'no runs' text the row
    ALREADY computes reaches only a `title=` on that invisible div, so the
    state is never stated in the open.

    Asserted on structure and on `data-testid` values, never on copy: a
    rewording keeps this green, deleting a state does not.
    """
    body = tab_memory_evals_jsx_body
    code = tab_memory_evals_jsx_code

    row_body = _extract_function_body(code, 'MemoryEvalMetricRow')
    assert row_body, 'could not extract the MemoryEvalMetricRow body.'

    # (a) something must MEASURE the series length. Nothing did.
    points_decl = re.search(
        r'const\s+(\w+)\s*=\s*[^;\n]*trend\.values[^;\n]*\.length', row_body
    )
    assert points_decl is not None, (
        'MemoryEvalMetricRow never measures the length of `trend.values`, so '
        'it cannot tell an empty series from a populated one. `trendGaps([])` '
        'is 0, so the gap check alone passes a no-runs metric straight to a '
        'chart primitive that renders nothing.'
    )
    points_local = points_decl.group(1)

    # (b) the chart gate must CONSUME that measurement. Re-derived exactly as
    #     test_trend_holes_are_never_handed_to_a_chart_primitive derives it, so
    #     the two tests cannot drift apart on what "the gate" means.
    gate_decl = None
    for decl in re.finditer(
        r'const\s+(\w+)\s*=\s*[^;\n]*\bChart\b[^;\n]*\bgaps\b[^;\n]*;', body
    ):
        if re.search(r'\{\s*' + re.escape(decl.group(1)) + r'\b', body):
            gate_decl = decl
            break
    assert gate_decl is not None, (
        'no single-line `const <name> = ...Chart...gaps...;` gate whose local '
        'reaches a `{<local>` JSX position — see '
        'test_trend_holes_are_never_handed_to_a_chart_primitive, which derives '
        'the gate the same way.'
    )
    assert re.search(r'\b' + re.escape(points_local) + r'\b', gate_decl.group(0)), (
        f'the chart gate {gate_decl.group(0).strip()!r} does not consume the '
        f'series-length local `{points_local}`. An empty series must never '
        'reach a charts.jsx primitive: both Sparkline and StepSpark return '
        'null for a zero-length array, leaving a blank 26px box.'
    )

    # (c) FIVE structurally distinct trend states. A separate no-runs arm is
    #     required rather than folding it into the gap message because
    #     "no chart — 0 of 0 runs produced no sample" is a nonsense sentence
    #     that reads as a bug — the very failure this suppression path exists
    #     to prevent.  A separate mismatch arm is required for the same reason,
    #     one payload shape further out: labels and values are PARALLEL arrays
    #     (memory_evals.py:955,993), so a disagreement makes every other
    #     sentence this cell could produce untrustworthy.
    testids = (
        'memory-eval-trend-chart',
        'memory-eval-trend-no-kind',
        'memory-eval-trend-mismatch',
        'memory-eval-trend-no-runs',
        'memory-eval-trend-gaps',
    )
    assert len(set(testids)) == len(testids), 'the five trend testids must be distinct.'
    for testid in testids:
        assert f'data-testid="{testid}"' in code, (
            f'the trend cell must render a `data-testid="{testid}"` arm. The '
            'five states — drawn chart, unrenderable kind, labels/values length '
            'disagreement, no runs yet, and holed series — must be structurally '
            'distinguishable, so a rewording cannot silently collapse two of '
            'them into one.'
        )

    # (d) the existing guards must SURVIVE, re-asserted here so a later
    #     refactor to a `trendState()` discriminator cannot quietly drop them.
    bare = re.search(r'\{\s*Chart\s*(?:\?|&&)(?![^\n]*\bgaps\b)', body)
    assert bare is None, (
        f'a chart render site is guarded by `Chart` alone: {bare.group(0)!r} — '
        'a holed OR empty series would reach the primitive through it.'
    )
    assert re.search(r'\{\s*gaps\b', code), (
        'the `gaps` count must still reach a render position — adding the '
        'no-runs state must not cost the holed-series disclosure.'
    )

    # (e) ONE series count feeds every disclosure in this cell.  Measuring the
    #     state from `trend.values` while the footer counts `trend.labels` lets
    #     a payload where the two disagree render the mutually contradictory
    #     pair "no runs yet — nothing to chart" next to "N pts": the same
    #     reads-as-a-bug outcome this state was added to prevent, moved one
    #     line down.
    assert re.search(r'\{\s*' + re.escape(points_local) + r'\s*\}', row_body), (
        f'the series-length local `{points_local}` never reaches a bare '
        '`{<local>}` render position, so the footer count is derived from some '
        'OTHER measurement than the one the trend states are gated on.'
    )
    stray = re.search(r'\{[^{}]*trend\.labels[^{}]*\.length[^{}]*\}', row_body)
    assert stray is None, (
        f'a render expression still counts `trend.labels` directly: '
        f'{stray.group(0)!r}. Every count in this cell must read the single '
        f'`{points_local}` local, or a labels/values disagreement prints two '
        'contradictory sentences side by side.'
    )

    # ...and the disagreement itself is NAMED rather than silently reconciled
    #    by picking one array's length over the other.
    mismatch_decl = re.search(
        r'const\s+(\w+)\s*=\s*[^;\n]*\.length\s*!==\s*[^;\n]*;', row_body
    )
    assert mismatch_decl is not None, (
        'MemoryEvalMetricRow never compares the labels and values lengths. They '
        'are parallel arrays built from one `runs` list server-side '
        '(memory_evals.py:955,993); a payload where they disagree is malformed, '
        'and nothing this cell says about the series would be trustworthy.'
    )
    mismatch_local = mismatch_decl.group(1)
    assert re.search(r'\b' + re.escape(mismatch_local) + r'\b', gate_decl.group(0)), (
        f'the chart gate {gate_decl.group(0).strip()!r} does not consume '
        f'`{mismatch_local}`, so a malformed series is still drawn — against a '
        'title derived from the other, differently-sized array.'
    )
    assert re.search(
        re.escape(mismatch_local)
        + r'[\s\S]{0,400}?data-testid="memory-eval-trend-mismatch"',
        row_body,
    ), (
        f'`{mismatch_local}` does not gate the '
        '`data-testid="memory-eval-trend-mismatch"` arm, so the disagreement is '
        'measured but never stated.'
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
    #     memory_evals._escalation_projection() — the fingerprint is the
    #     producer's private construction; the dashboard must not depend on
    #     its substructure.
    for field in ('dedupe_fingerprint', 'fingerprint'):
        for op in (r'\.split\(', r'\.slice\(', r'\.match\(', r'\.substring\('):
            assert not re.search(rf'{field}\s*{op}', body), (
                f'`{field}` must be rendered whole — never split/sliced/matched. '
                'The fingerprint is the producer\'s private construction '
                '(memory_evals._escalation_projection()); parsing its '
                'substructure here would couple the dashboard to a format it '
                'does not own.'
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
        '(memory_evals._build_payload() fills it; _empty_payload() declares it '
        'on every return path). The identical object repeated on each eval '
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
            'parity orphan (memory_evals._unmatched_projection()).'
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
        'signal that catches a real parity orphan (memory_evals._unmatched_projection()).'
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
    # (memory_evals._read_limits() stamps `stale_for_latest_run`).
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


def test_limits_provenance_open_state_is_per_eval(
    tab_memory_evals_jsx_code: str,
) -> None:
    """The provenance <details> open state must be PER EVAL and read once at mount.

    The defect this pins, in two halves:

    (1) A single module-global key.  `ME_PROV_OPEN_KEY = 'df.memevals.prov'`
        was one string shared by every eval card, so expanding one card's
        provenance wrote '1' and the next re-render expanded ALL of them.  The
        open state of a <details> is per-disclosure UI state; keying it on the
        section rather than on the eval makes one operator's click look like a
        rendering bug across every other card.

    (2) The attribute was a CALL: `<details open={readProvOpen()} ...>`.  A
        function call in a JSX attribute re-runs on every render — here a
        synchronous `localStorage.getItem` per eval card per 3s poll tick — and
        makes the DOM's own open state unrecoverable, since the next poll
        overwrites whatever the operator just toggled with the stored value.
        The fix is React state seeded from storage exactly once at mount.

    Every identifier below is DERIVED from the source, never pinned by
    spelling: renaming the state local, the helpers, or the hook alias is not a
    test failure.  What must hold is the structure.
    """
    code = tab_memory_evals_jsx_code

    prov_body = _extract_function_body(code, 'LimitsProvenance')
    assert prov_body, 'could not extract the LimitsProvenance body.'

    # (a) the `open=` attribute is a bare identifier, not a call.
    open_attr = re.search(r'<details\s[^>]*open=\{(\w+)\}', prov_body)
    assert open_attr is not None, (
        'the provenance <details> must take its open state from a bare '
        'identifier (`open={someLocal}`). Not found — see (b): it must be '
        'React state seeded from localStorage at mount.'
    )
    open_local = open_attr.group(1)

    # ...and the NEGATIVE that is the actual defect: no call in the attribute.
    call_attr = re.search(r'<details\s[^>]*open=\{\s*\w+\s*\(', code)
    assert call_attr is None, (
        f'the provenance <details> reads its open state from a function call '
        f'in the attribute: {call_attr.group(0)!r}. That is a synchronous '
        'localStorage read on EVERY render — once per eval card per 3s poll '
        'tick — and it clobbers the operator\'s toggle on the next poll. Seed '
        'React state from storage once at mount instead.'
    )

    # (b) that identifier is React useState, seeded by the read helper.
    #     Any useState alias is accepted (the file uses ME-prefixed aliases:
    #     MESpark / MEStep / MEC / MEDF), so the assertion is on the
    #     destructuring SHAPE plus the initializer, not on the hook's spelling.
    state_decl = re.search(
        r'const\s*\[\s*' + re.escape(open_local) + r'\s*,\s*(\w+)\s*\]\s*=\s*(\w+)\(([\s\S]{0,200}?)\);',
        prov_body,
    )
    assert state_decl is not None, (
        f'`{open_local}` reaches the <details> `open=` attribute but is not '
        f'declared as React state (`const [{open_local}, setX] = useState(...)`). '
        'Holding the open state in the component is what makes it per-card '
        'rather than per-section.'
    )
    initializer = state_decl.group(3)
    read_helpers = sorted({
        fn[0]
        for m in re.finditer(r'localStorage\.getItem\(', code)
        if (fn := _enclosing_function(code, m.start())) is not None
    })
    assert read_helpers, (
        'no function in tab_memory_evals.jsx reads localStorage — the '
        'provenance open state must still be PERSISTED across reloads.'
    )
    assert any(re.search(r'\b' + re.escape(h) + r'\s*\(', initializer) for h in read_helpers), (
        f'the state initializer {initializer.strip()!r} does not call the '
        f'localStorage read helper (one of {read_helpers}). The stored open '
        'state must seed the component exactly once at mount.'
    )

    # (c) HOOK ORDER: the state declaration must precede the `if (!lim)` early
    #     return LimitsProvenance already has for a null limits artifact. A
    #     hook after a conditional return is a Rules-of-Hooks violation that
    #     blanks the card at runtime the first time an eval has no limits.
    lim_local = re.search(r'\b(\w+)\s*=\s*ev\.limits\b', prov_body)
    assert lim_local is not None, (
        'LimitsProvenance must read `ev.limits` into a local.'
    )
    guard = re.search(
        r'if\s*\(\s*!\s*' + re.escape(lim_local.group(1)) + r'\s*\)', prov_body
    )
    assert guard is not None, (
        'LimitsProvenance must keep its `if (!lim)` early-return guard for a '
        'null limits artifact.'
    )
    assert state_decl.start() < guard.start(), (
        'the provenance open-state hook is declared AFTER the `if (!lim)` '
        'early return. That is a conditional hook (Rules of Hooks): the first '
        'eval without a limits artifact changes the hook count and React '
        'blanks the card. Move the hook above the guard.'
    )

    # (d) the persisted key is derived from the EVAL IDENTITY, so the key
    #     expression reaching localStorage varies per card.
    assert re.search(r'\bev\.eval_id\b', prov_body), (
        'LimitsProvenance never reads `ev.eval_id`, so its persisted open-state '
        'key cannot vary per eval — one card\'s toggle would expand every other '
        'card on the next poll-driven re-render.'
    )

    # (e) the read and write helpers take the key as a PARAMETER rather than
    #     closing over one module constant. This is what makes a single global
    #     key structurally unrepresentable, not merely absent today.
    for call in ('getItem', 'setItem'):
        sites = list(re.finditer(
            r'localStorage\.' + call + r'\(\s*([\w.]+)', code
        ))
        assert sites, f'no `localStorage.{call}(` call site found in tab_memory_evals.jsx.'
        for site in sites:
            first_arg = site.group(1)
            enclosing = _enclosing_function(code, site.start())
            assert enclosing is not None, (
                f'a `localStorage.{call}(` call sits at module scope — the '
                'storage key must be a parameter of an enclosing helper.'
            )
            fn_name, param_names = enclosing
            assert first_arg in param_names, (
                f'`{fn_name}` passes {first_arg!r} to localStorage.{call}() but '
                f'its parameters are {param_names}. The storage key must be a '
                'PARAMETER, so a single module-global key shared by every eval '
                'card is unrepresentable rather than merely not-currently-written.'
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
        '(memory_evals._build_payload()) and needs its OWN branch, gated on '
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
