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
