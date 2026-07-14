"""Smoke tests for /static/redux/index.html.

Guards against:
  * Silent removal of the marked / DOMPurify CDN script tags (the MarkdownText
    component falls back to plain text when these are missing — works "well
    enough" that the regression can ship unnoticed).
"""

from __future__ import annotations

import html.parser
import re

import pytest

_INDEX_URL = '/static/redux/index.html'

# Matches well-formed SRI hashes: sha256/384/512 followed by a base64 payload.
_SRI_HASH_RE = re.compile(r'^sha(256|384|512)-[A-Za-z0-9+/=]{20,}$')


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

    ``index`` is the tag's 0-based position in ``_ScriptTagCollector.script_attrs``
    (document order, since the list preserves insertion order).  Returning attrs
    alongside the position avoids a second parse when the caller also needs the
    src or other attributes.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


@pytest.fixture(scope='module')
def index_html_body():
    """Fetch /static/redux/index.html once for the whole test module."""
    from starlette.testclient import TestClient

    from dashboard.app import app

    with TestClient(app) as c:
        return c.get(_INDEX_URL).text


def test_static_index_html_serves_200(client):
    """GET /static/redux/index.html via the StaticFiles mount returns 200."""
    resp = client.get(_INDEX_URL)
    assert resp.status_code == 200, (
        f'expected 200 for {_INDEX_URL}, got {resp.status_code}'
    )


_CDN_SCRIPT_CASES = [
    (
        'https://unpkg.com/marked@',
        'marked',
        'MarkdownText (tab_tasks.jsx) depends on the global `marked` symbol — '
        'removing this tag breaks markdown rendering in Task Detail.',
    ),
    (
        'https://unpkg.com/dompurify@',
        'DOMPurify',
        'MarkdownText (tab_tasks.jsx) depends on the global `DOMPurify` symbol — '
        'removing this tag means rendered markdown bypasses sanitisation.',
    ),
]


@pytest.mark.parametrize(
    'src_prefix, lib_name, consumer_note',
    _CDN_SCRIPT_CASES,
    ids=['marked', 'dompurify'],
)
def test_cdn_script_has_sri_integrity(
    index_html_body: str, src_prefix: str, lib_name: str, consumer_note: str
) -> None:
    """Each CDN <script> tag is present with a well-formed SRI integrity hash.

    Parametrised over marked and DOMPurify — both are required by the
    MarkdownText component in tab_tasks.jsx.
    """
    result = _find_script_position(index_html_body, src_prefix)
    attrs = result[1] if result is not None else None
    assert attrs is not None, (
        f'No <script src="{src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    integrity = (attrs.get('integrity') or '').strip()
    assert integrity, (
        f'{lib_name} CDN tag has missing or empty integrity= attribute. '
        f'src={attrs.get("src")!r}'
    )
    assert _SRI_HASH_RE.match(integrity), (
        f'{lib_name} CDN tag integrity= is not a valid SRI hash '
        f'(expected sha256/384/512-<base64>): {integrity!r}'
    )


# ---------------------------------------------------------------------------
# Helper-level coverage for _find_script_position (synthetic HTML)
# ---------------------------------------------------------------------------

_MARKED_TAG = '<script src="https://unpkg.com/marked@x/y.js"></script>'
_TAB_TASKS_TAG = '<script src="/static/redux/tab_tasks.jsx?v=9"></script>'

_FIND_SCRIPT_POSITION_CASES = [
    # (a) marked-then-tab_tasks: correct load order
    (_MARKED_TAG + _TAB_TASKS_TAG, 'https://unpkg.com/marked@', 0),
    (_MARKED_TAG + _TAB_TASKS_TAG, '/static/redux/tab_tasks.jsx', 1),
    # (b) tab_tasks-then-marked: reversed order (regression scenario)
    (_TAB_TASKS_TAG + _MARKED_TAG, 'https://unpkg.com/marked@', 1),
    (_TAB_TASKS_TAG + _MARKED_TAG, '/static/redux/tab_tasks.jsx', 0),
    # (c) missing tag returns None
    (_TAB_TASKS_TAG, 'https://unpkg.com/marked@', None),
]


@pytest.mark.parametrize(
    'body, src_prefix, expected_position',
    _FIND_SCRIPT_POSITION_CASES,
    ids=[
        'marked-first-marked-pos',
        'marked-first-tab_tasks-pos',
        'reversed-marked-pos',
        'reversed-tab_tasks-pos',
        'missing-tag-returns-none',
    ],
)
def test_find_script_position_returns_document_order(
    body: str, src_prefix: str, expected_position: int | None
) -> None:
    """_find_script_position returns the 0-indexed document position of the
    first <script> tag whose src starts with src_prefix, or None if absent.

    Exercises synthetic HTML so that a future bad ordering of the real
    index.html would actually be caught (i.e. proves the helper distinguishes
    good-order from bad-order).
    """
    result = _find_script_position(body, src_prefix)
    actual_pos = result[0] if result is not None else None
    assert actual_pos == expected_position


# ---------------------------------------------------------------------------
# Regression guard: CDN globals must be defined before tab_tasks.jsx runs
# ---------------------------------------------------------------------------

_TAB_TASKS_PREFIX = '/static/redux/tab_tasks.jsx'


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
        f'If it loads after, {after_label} may execute before {before_label} '
        f'is defined — the silent-failure class the smoke test was added to catch.'
    )


@pytest.mark.parametrize(
    'src_prefix, lib_name, consumer_note',
    _CDN_SCRIPT_CASES,
    ids=['marked', 'dompurify'],
)
def test_cdn_script_loads_before_tab_tasks_jsx(
    index_html_body: str, src_prefix: str, lib_name: str, consumer_note: str
) -> None:
    """CDN scripts for marked/DOMPurify must appear BEFORE tab_tasks.jsx.

    Regression guard: moving either CDN tag *after* tab_tasks.jsx means
    MarkdownText's first render runs while ``marked`` / ``DOMPurify`` are still
    undefined, so it falls back to null — the dashboard still loads, so the
    regression can ship unnoticed (same silent-failure class the smoke test was
    added to catch).

    Naturally GREEN against the current correctly-ordered index.html; will fire
    loudly if a future edit moves either CDN tag below the tab_tasks.jsx tag.

    This test checks document order, which correctly predicts execution order
    only when both scripts are classic synchronous scripts (no defer, async, or
    type="module"). The test body asserts that assumption explicitly so that a
    future edit adding those attributes fails loudly rather than silently passing
    a check that no longer reflects execution order.
    """
    _assert_script_loads_before(
        index_html_body,
        src_prefix,
        _TAB_TASKS_PREFIX,
        before_label=lib_name,
        after_label='tab_tasks.jsx',
        consumer_note=consumer_note,
    )


# ---------------------------------------------------------------------------
# Guard-layer coverage: defer / async / type="module" must trigger an error
# ---------------------------------------------------------------------------

_DEFERRED_CDN_CASES = [
    # (extra_attrs_on_cdn_tag, regex_to_match_in_AssertionError_message)
    # Pinned to unique phrases in each guard's assertion message so that an
    # unrelated mention of the word (e.g. "deferred" in the position-comparison
    # failure message) cannot satisfy the match while the wrong guard fires.
    ('defer', r'defer attribute'),
    ('async', r'async attribute'),
    ('type="module"', r'type="module".*deferred by default'),
]

_DEFERRED_TAB_TASKS_CASES = [
    # (extra_attrs_on_tab_tasks_tag, regex_to_match_in_AssertionError_message)
    # Regex patterns are pinned to include 'tab_tasks.jsx' so that a CDN-branch
    # fire (which would produce a message referencing the CDN tag label) cannot
    # satisfy the match — same defensive unique-phrase approach used in
    # _DEFERRED_CDN_CASES.
    ('defer', r'tab_tasks\.jsx has a defer attribute'),
    ('async', r'tab_tasks\.jsx has an async attribute'),
    ('type="module"', r'tab_tasks\.jsx has type="module".*deferred by default'),
]


@pytest.mark.parametrize(
    'extra_attrs, match_pattern',
    _DEFERRED_CDN_CASES,
    ids=['defer', 'async', 'type-module'],
)
def test_load_order_assertion_fires_on_deferred_cdn(
    extra_attrs: str, match_pattern: str
) -> None:
    """When the CDN tag carries defer / async / type="module", the
    false-pass guard inside the load-order assertion must fire — document
    order no longer implies execution order in those cases.
    """
    cdn_tag = (
        f'<script src="https://unpkg.com/marked@x/y.js" '
        f'{extra_attrs}></script>'
    )
    body = cdn_tag + _TAB_TASKS_TAG
    with pytest.raises(AssertionError, match=match_pattern):
        _assert_script_loads_before(
            body,
            'https://unpkg.com/marked@',
            _TAB_TASKS_PREFIX,
            before_label='marked',
            after_label='tab_tasks.jsx',
        )


@pytest.mark.parametrize(
    'extra_attrs, match_pattern',
    _DEFERRED_TAB_TASKS_CASES,
    ids=['defer', 'async', 'type-module'],
)
def test_load_order_assertion_fires_on_deferred_tab_tasks(
    extra_attrs: str, match_pattern: str
) -> None:
    """When the tab_tasks.jsx tag carries defer / async / type="module", the
    false-pass guard inside the load-order assertion must fire.

    Covers the tab_tasks.jsx iteration of the for-loop in
    _assert_cdn_loads_before_tab_tasks, mirroring
    test_load_order_assertion_fires_on_deferred_cdn which covers the CDN branch.
    A clean CDN tag is placed first so the loop advances past the CDN iteration
    before raising on the tab_tasks.jsx tag.  The match regex is pinned to
    'tab_tasks.jsx' so a future regression that fires the CDN branch instead
    cannot satisfy the match and give a false pass.
    """
    cdn_tag = '<script src="https://unpkg.com/marked@x/y.js"></script>'
    bad_tab_tasks_tag = (
        f'<script src="/static/redux/tab_tasks.jsx?v=9" '
        f'{extra_attrs}></script>'
    )
    body = cdn_tag + bad_tab_tasks_tag
    with pytest.raises(AssertionError, match=match_pattern):
        _assert_script_loads_before(
            body,
            'https://unpkg.com/marked@',
            _TAB_TASKS_PREFIX,
            before_label='marked',
            after_label='tab_tasks.jsx',
        )


# ---------------------------------------------------------------------------
# Regression guard: tab_curator.jsx must load AFTER charts.jsx, shell.jsx,
# data.js, and BEFORE app.jsx (synchronous classic script, no defer/async).
# ---------------------------------------------------------------------------

_TAB_CURATOR_PREFIX = '/static/redux/tab_curator.jsx'
_CHARTS_PREFIX = '/static/redux/charts.jsx'
_SHELL_PREFIX = '/static/redux/shell.jsx'
_DATA_JS_PREFIX = '/static/redux/data.js'
_TABS_PREFIX = '/static/redux/tabs.jsx'
_APP_JSX_PREFIX = '/static/redux/app.jsx'

# Each pair: (before_src_prefix, before_label, after_src_prefix, after_label)
# tab_curator.jsx must load AFTER its four deps and BEFORE app.jsx.
# tabs.jsx is included because tab_curator.jsx destructures ChipList from
# window.DF_TABS at top-level execution time — if tabs.jsx moved after
# tab_curator.jsx the destructure would silently set ChipList to undefined.
_TAB_CURATOR_ORDER_CASES = [
    (_CHARTS_PREFIX, 'charts.jsx', _TAB_CURATOR_PREFIX, 'tab_curator.jsx'),
    (_SHELL_PREFIX, 'shell.jsx', _TAB_CURATOR_PREFIX, 'tab_curator.jsx'),
    (_DATA_JS_PREFIX, 'data.js', _TAB_CURATOR_PREFIX, 'tab_curator.jsx'),
    (_TABS_PREFIX, 'tabs.jsx', _TAB_CURATOR_PREFIX, 'tab_curator.jsx'),
    (_TAB_CURATOR_PREFIX, 'tab_curator.jsx', _APP_JSX_PREFIX, 'app.jsx'),
]


@pytest.mark.parametrize(
    'before_prefix, before_label, after_prefix, after_label',
    _TAB_CURATOR_ORDER_CASES,
    ids=[
        'charts-before-curator',
        'shell-before-curator',
        'data-before-curator',
        'tabs-before-curator',
        'curator-before-app',
    ],
)
def test_tab_curator_loads_before_app_jsx(
    index_html_body: str,
    before_prefix: str,
    before_label: str,
    after_prefix: str,
    after_label: str,
) -> None:
    """tab_curator.jsx must load AFTER its deps and BEFORE app.jsx.

    Parametrized over the four required ordering pairs using the generic
    _assert_script_loads_before helper rather than bespoke per-case logic.
    """
    _assert_script_loads_before(
        index_html_body,
        before_prefix,
        after_prefix,
        before_label=before_label,
        after_label=after_label,
    )


def test_load_order_assertion_passes_for_classic_scripts() -> None:
    """_assert_cdn_loads_before_tab_tasks raises no exception when both scripts
    are classic synchronous scripts placed in the correct document order.

    Happy-path contract: a clean CDN tag (no defer/async/type=module) placed
    before tab_tasks.jsx must pass both the false-pass guard and the position
    comparison without raising.  This test ensures an accidental edit that makes
    the guards always-fire (e.g. inverted assert conditions) is caught rather
    than silently masked by the negative-only parametrised cases above.
    """
    cdn_tag = '<script src="https://unpkg.com/marked@x/y.js"></script>'
    body = cdn_tag + _TAB_TASKS_TAG
    # Must complete without raising — classic script, correct document order.
    _assert_script_loads_before(
        body,
        'https://unpkg.com/marked@',
        _TAB_TASKS_PREFIX,
        before_label='marked',
        after_label='tab_tasks.jsx',
    )


# ---------------------------------------------------------------------------
# Regression guard: graph_layout.js must load BEFORE tab_tasks.jsx (task 2528)
# ---------------------------------------------------------------------------

_GRAPH_LAYOUT_PREFIX = '/static/redux/graph_layout.js'


def test_graph_layout_js_loads_before_tab_tasks(index_html_body: str) -> None:
    """graph_layout.js must load as a classic synchronous script BEFORE
    tab_tasks.jsx.

    tab_tasks.jsx destructures {computeTiers, partitionComponents, orderRows}
    from window.DF_GRAPH_LAYOUT at top-level execution time — if graph_layout.js
    loaded after (or not at all), that destructure would silently produce
    undefined, and TaskGraph would throw the moment it tries to call one of
    those functions.
    """
    _assert_script_loads_before(
        index_html_body,
        _GRAPH_LAYOUT_PREFIX,
        _TAB_TASKS_PREFIX,
        before_label='graph_layout.js',
        after_label='tab_tasks.jsx',
        consumer_note=(
            'tab_tasks.jsx destructures window.DF_GRAPH_LAYOUT at top level; '
            'graph_layout.js must define it first.'
        ),
    )


# ---------------------------------------------------------------------------
# Regression guard: all /static/redux/* cache-busters share one bumped version
# ---------------------------------------------------------------------------


def test_redux_cache_buster_bumped(index_html_body: str) -> None:
    """All /static/redux/*?v= cache-busters must share a single version >= 28,
    and graph_layout.js must be among the versioned assets.

    Mirrors the existing single-shared-version guards in test_tab_escalations.py,
    test_tab_scheduler.py, and test_scheduler_page.py, raising the floor to 28 to
    prove the uniform bump landed alongside task 2529's focus-mode changes to
    graph_layout.js, tab_tasks.jsx, and styles.css.
    """
    versions = set(re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body))
    assert len(versions) == 1, (
        f'index.html has mixed /static/redux/?v= cache-buster versions: {sorted(versions)} — '
        'bump all of them uniformly to the same value.'
    )
    v = int(next(iter(versions)))
    assert v >= 28, (
        f'index.html cache-buster version is {v}, expected >= 28 (proves the '
        'uniform bump from 27 alongside task 2529\'s focus-mode changes).'
    )
    assert re.search(r'/static/redux/graph_layout\.js\?v=\d+', index_html_body), (
        'graph_layout.js is not present among the versioned /static/redux/* '
        'assets in index.html.'
    )
