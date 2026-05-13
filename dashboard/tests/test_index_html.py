"""Smoke tests for /static/redux/index.html.

Guards against:
  * Silent removal of the marked / DOMPurify CDN script tags (the MarkdownText
    component falls back to plain text when these are missing — works "well
    enough" that the regression can ship unnoticed).
"""

from __future__ import annotations

import re

_INDEX_URL = '/static/redux/index.html'

_SCRIPT_TAG_RE = re.compile(r'<script\b[^>]*>', re.IGNORECASE)
_SRC_ATTR_RE = re.compile(r'\bsrc="([^"]+)"', re.IGNORECASE)
_INTEGRITY_ATTR_RE = re.compile(r'\bintegrity="([^"]*)"', re.IGNORECASE)


def _find_script_tag_with_src_prefix(body: str, src_prefix: str) -> str | None:
    """Return the full opening ``<script ... >`` tag whose ``src`` starts with
    ``src_prefix``, or ``None`` if no such tag exists.  Attribute order inside
    the tag does not matter."""
    for tag_match in _SCRIPT_TAG_RE.finditer(body):
        tag = tag_match.group(0)
        src_match = _SRC_ATTR_RE.search(tag)
        if src_match and src_match.group(1).startswith(src_prefix):
            return tag
    return None


def test_static_index_html_serves_200(client):
    """GET /static/redux/index.html via the StaticFiles mount returns 200."""
    resp = client.get(_INDEX_URL)
    assert resp.status_code == 200, (
        f'expected 200 for {_INDEX_URL}, got {resp.status_code}'
    )


def test_marked_cdn_script_has_sri_integrity(client):
    """The marked CDN <script> tag is present with a non-empty SRI integrity hash."""
    body = client.get(_INDEX_URL).text
    tag = _find_script_tag_with_src_prefix(body, 'https://unpkg.com/marked@')
    assert tag is not None, (
        'No <script src="https://unpkg.com/marked@..."> tag found in index.html. '
        'MarkdownText (tab_tasks.jsx) depends on the global `marked` symbol — '
        'removing this tag breaks markdown rendering in Task Detail.'
    )
    integrity_match = _INTEGRITY_ATTR_RE.search(tag)
    assert integrity_match is not None and integrity_match.group(1).strip(), (
        f'marked CDN tag is missing or has empty integrity= attribute: {tag!r}'
    )


def test_dompurify_cdn_script_has_sri_integrity(client):
    """The DOMPurify CDN <script> tag is present with a non-empty SRI integrity hash."""
    body = client.get(_INDEX_URL).text
    tag = _find_script_tag_with_src_prefix(body, 'https://unpkg.com/dompurify@')
    assert tag is not None, (
        'No <script src="https://unpkg.com/dompurify@..."> tag found in index.html. '
        'MarkdownText (tab_tasks.jsx) depends on the global `DOMPurify` symbol — '
        'removing this tag means rendered markdown bypasses sanitisation.'
    )
    integrity_match = _INTEGRITY_ATTR_RE.search(tag)
    assert integrity_match is not None and integrity_match.group(1).strip(), (
        f'dompurify CDN tag is missing or has empty integrity= attribute: {tag!r}'
    )

