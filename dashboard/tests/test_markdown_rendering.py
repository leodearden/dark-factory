"""Tests for markdown rendering infrastructure in the dashboard.

Asserts on the literal text of served static files (index.html, tab_tasks.jsx,
styles.css) to verify that the markdown rendering feature is properly wired up.
This follows the same 'asset content substring assertion' pattern established
in test_app.py::test_index_serves_redux_html and test_static_redux_assets_load.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# CDN script tags: marked + DOMPurify
# ---------------------------------------------------------------------------


def test_index_includes_marked_cdn_with_sri(client):
    """index.html must contain a <script> tag loading marked from unpkg with SRI."""
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.text
    # Allow flexible attribute order and whitespace
    pattern = re.compile(
        r'<script\s[^>]*src="https://unpkg\.com/marked@[^"]+"[^>]*'
        r'integrity="sha384-[A-Za-z0-9+/=]+"[^>]*'
        r'crossorigin="anonymous"[^>]*></script>',
        re.DOTALL,
    )
    assert pattern.search(body), (
        "index.html must include a <script src='https://unpkg.com/marked@...' "
        "integrity='sha384-...' crossorigin='anonymous'></script> tag"
    )


def test_index_includes_dompurify_cdn_with_sri(client):
    """index.html must contain a <script> tag loading DOMPurify from unpkg with SRI."""
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.text
    pattern = re.compile(
        r'<script\s[^>]*src="https://unpkg\.com/dompurify@[^"]+"[^>]*'
        r'integrity="sha384-[A-Za-z0-9+/=]+"[^>]*'
        r'crossorigin="anonymous"[^>]*></script>',
        re.DOTALL | re.IGNORECASE,
    )
    assert pattern.search(body), (
        "index.html must include a <script src='https://unpkg.com/dompurify@...' "
        "integrity='sha384-...' crossorigin='anonymous'></script> tag"
    )


# ---------------------------------------------------------------------------
# Cache-buster bump: ?v=7 → ?v=8
# ---------------------------------------------------------------------------


def test_index_local_assets_use_bumped_cache_buster(client):
    """All local /static/redux/*.{js,jsx,css} references must use ?v=8, none ?v=7."""
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.text
    versions = re.findall(r'/static/redux/[^?"]+\?v=(\d+)', body)
    assert versions, 'Expected at least one /static/redux/?v=N reference in index.html'
    unique_versions = set(versions)
    assert unique_versions == {'8'}, (
        f"All local asset references must use ?v=8; found versions: {unique_versions}"
    )


# ---------------------------------------------------------------------------
# MarkdownText component in tab_tasks.jsx
# ---------------------------------------------------------------------------


def test_tab_tasks_defines_markdowntext_component(client):
    """tab_tasks.jsx must define a MarkdownText component using marked + DOMPurify."""
    resp = client.get('/static/redux/tab_tasks.jsx')
    assert resp.status_code == 200
    body = resp.text
    assert 'function MarkdownText(' in body, (
        "tab_tasks.jsx must define 'function MarkdownText('"
    )
    assert 'marked.parse(' in body, (
        "MarkdownText must call marked.parse()"
    )
    assert 'DOMPurify.sanitize(' in body, (
        "MarkdownText must call DOMPurify.sanitize()"
    )
    assert 'dangerouslySetInnerHTML' in body, (
        "MarkdownText must use dangerouslySetInnerHTML"
    )
    assert 'return null' in body, (
        "MarkdownText must return null for empty input"
    )


# ---------------------------------------------------------------------------
# TaskDetail call sites: description + details via MarkdownText
# ---------------------------------------------------------------------------


def test_taskdetail_renders_description_and_details_via_markdowntext(client):
    """TaskDetail must route both description and details through MarkdownText."""
    resp = client.get('/static/redux/tab_tasks.jsx')
    assert resp.status_code == 200
    body = resp.text

    # Both fields must be routed through MarkdownText
    assert '<MarkdownText text={task.description}' in body, (
        "TaskDetail must render description via <MarkdownText text={task.description} ...>"
    )
    assert '<MarkdownText text={task.details}' in body, (
        "TaskDetail must render details via <MarkdownText text={task.details} ...>"
    )

    # Old raw renders must be gone
    assert '<div className="desc">{desc}</div>' not in body, (
        "Old raw description render '<div className=\"desc\">{desc}</div>' must be removed"
    )
    assert '<div className="desc">{task.details}</div>' not in body, (
        "Old raw details render '<div className=\"desc\">{task.details}</div>' must be removed"
    )

    # Ternary structure for description: task.description ? <MarkdownText ...> : <div>—</div>
    ternary_pattern = re.compile(r'task\.description\s*\?\s*<MarkdownText')
    assert ternary_pattern.search(body), (
        "TaskDetail must use a ternary 'task.description ? <MarkdownText ...' for description"
    )

    # Em-dash fallback as plain text (not through MarkdownText)
    assert '<div className="desc">—</div>' in body, (
        "Em-dash fallback must be a plain '<div className=\"desc\">—</div>' (not via MarkdownText)"
    )


# ---------------------------------------------------------------------------
# CSS markdown rules in styles.css
# ---------------------------------------------------------------------------


def test_styles_includes_markdown_rules_for_desc(client):
    """styles.css must include markdown-specific rules scoped under .task-detail .desc."""
    resp = client.get('/static/redux/styles.css')
    assert resp.status_code == 200
    body = resp.text

    # Each of the required element selectors must appear inside a .task-detail .desc rule
    required_selectors = ['h1', 'h2', 'h3', 'code', 'pre', 'ul', 'ol', 'p']
    for sel in required_selectors:
        assert f'.task-detail .desc {sel}' in body, (
            f"styles.css must include a '.task-detail .desc {sel}' rule"
        )

    # The --mono variable must be reused for code/pre styling
    # Verify var(--mono) appears after the first .task-detail .desc rule
    first_rule_pos = body.find('.task-detail .desc h1')
    assert first_rule_pos != -1, "'.task-detail .desc h1' rule not found"
    mono_pos = body.find('var(--mono)', first_rule_pos)
    assert mono_pos != -1, (
        "var(--mono) must appear in the .task-detail .desc rules (after the first h1 rule)"
    )
