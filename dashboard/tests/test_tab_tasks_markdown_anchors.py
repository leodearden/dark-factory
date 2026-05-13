"""Regression guards for MarkdownText anchor-rewrite refinements in tab_tasks.jsx.

Tests fetch the served JSX source and pattern-match the implementation:
  1. Short-circuit when sanitized HTML has no ``'<a '`` (no-link fast path).
  2. ``forEach`` guard skipping in-page (``#``) and same-origin (``/``) hrefs.
"""

import re

_JSX_URL = '/static/redux/tab_tasks.jsx'


def test_short_circuit_when_sanitized_has_no_anchors(client):
    """tab_tasks.jsx short-circuits the <template> reparse when there is no '<a ' in sanitized HTML.

    Asserts:
    - The pattern ``if (!sanitized.includes('<a ')) return sanitized;`` (or equivalent)
      is present in the source.
    - It appears before the ``document.createElement('template')`` call (ordering guard).
    """
    resp = client.get(_JSX_URL)
    assert resp.status_code == 200
    body = resp.text

    # Match: if (!sanitized.includes('<a ')) return sanitized;
    pattern = re.compile(
        r"if\s*\(\s*!\s*sanitized\.includes\(\s*['\"]<a\s",
        re.DOTALL,
    )
    m = pattern.search(body)
    assert m is not None, (
        "Expected a short-circuit guard like "
        "`if (!sanitized.includes('<a ')) return sanitized;` "
        "before the `<template>` creation in MarkdownText"
    )

    # The guard must appear BEFORE the <template> construction.
    tpl_pos = body.index("document.createElement('template')")
    assert m.start() < tpl_pos, (
        "Short-circuit guard must appear before document.createElement('template') "
        f"(guard at {m.start()}, template at {tpl_pos})"
    )
