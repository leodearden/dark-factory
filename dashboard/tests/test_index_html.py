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


def _find_script_position(body: str, src_prefix: str) -> int | None:
    """Return the 0-indexed document position of the first <script> tag whose
    ``src`` starts with ``src_prefix``, or ``None`` if no such tag exists.

    Position-aware counterpart of ``_find_cdn_script_attrs``: position equals
    the tag's index in ``_ScriptTagCollector.script_attrs``, which preserves
    document order because it is a plain Python list.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i
    return None


def _find_cdn_script_attrs(
    body: str, src_prefix: str
) -> dict[str, str | None] | None:
    """Return the attribute dict for the first <script> whose ``src`` starts
    with ``src_prefix``, or ``None`` if no such tag exists.

    Uses ``html.parser.HTMLParser`` so attribute values containing ``>`` (e.g.
    inline JSON configs) are handled correctly — the regex ``[^>]*`` shortcut
    would truncate such tags and could miss attributes appearing after the ``>``.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for attrs in collector.script_attrs:
        src = attrs.get('src') or ''
        if src.startswith(src_prefix):
            return attrs
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
    attrs = _find_cdn_script_attrs(index_html_body, src_prefix)
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
# Step-1: Unit test for _find_script_position (helper-level, synthetic HTML)
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

    RED until _find_script_position is implemented (Step 2).
    """
    assert _find_script_position(body, src_prefix) == expected_position


# ---------------------------------------------------------------------------
# Step-3: Regression-guard — CDN tags must load BEFORE tab_tasks.jsx
# ---------------------------------------------------------------------------

_TAB_TASKS_PREFIX = '/static/redux/tab_tasks.jsx'


@pytest.mark.parametrize(
    'src_prefix, lib_name, consumer_note',
    _CDN_SCRIPT_CASES,
    ids=['marked', 'dompurify'],
)
def test_cdn_script_loads_before_tab_tasks_jsx(
    index_html_body: str, src_prefix: str, lib_name: str, consumer_note: str
) -> None:
    """CDN scripts for marked/DOMPurify must appear BEFORE tab_tasks.jsx.

    Regression guard for the silent-failure class described in reviewer note
    esc-1234-4 (suggestion 3): moving either CDN tag *after* tab_tasks.jsx
    means MarkdownText's first render runs while ``marked`` / ``DOMPurify``
    are still undefined, so it falls back to null — the dashboard still loads,
    so the regression can ship unnoticed (same failure class the smoke test was
    added to catch).

    Naturally GREEN against the current correctly-ordered index.html; will fire
    loudly if a future edit moves either CDN tag below the tab_tasks.jsx tag.
    """
    cdn_pos = _find_script_position(index_html_body, src_prefix)
    assert cdn_pos is not None, (
        f'No <script src="{src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )

    tab_tasks_pos = _find_script_position(index_html_body, _TAB_TASKS_PREFIX)
    assert tab_tasks_pos is not None, (
        f'<script src="{_TAB_TASKS_PREFIX}..."> not found in index.html — '
        f'cannot verify load-order invariant for {lib_name}.'
    )

    cdn_src = (_find_cdn_script_attrs(index_html_body, src_prefix) or {}).get('src')
    assert cdn_pos < tab_tasks_pos, (
        f'{lib_name} CDN tag (position {cdn_pos}, src={cdn_src!r}) must load '
        f'BEFORE tab_tasks.jsx (position {tab_tasks_pos}). '
        f'If it loads after, MarkdownText renders before {lib_name} is defined '
        f'and falls back to null on first render — the silent-failure class the '
        f'smoke test was added to catch.'
    )
