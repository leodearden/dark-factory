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


def test_anchor_rewrite_skips_in_page_and_same_origin_links(client):
    """The forEach over querySelectorAll('a') guards fragment (#) and relative (/) hrefs.

    Asserts inside the forEach body (by string-index ordering in the file):
    - ``a.getAttribute('href')`` is read to obtain the href.
    - An early ``return`` is guarded by both ``href.startsWith('#')`` and
      ``href.startsWith('/')``.
    - ``setAttribute('target', '_blank')`` appears AFTER the startsWith guards
      (i.e. the existing rewrite only runs for external links).
    """
    resp = client.get(_JSX_URL)
    assert resp.status_code == 200
    body = resp.text

    # (a) href is read via getAttribute
    href_read = re.compile(r"getAttribute\(\s*['\"]href['\"]\s*\)")
    m_href = href_read.search(body)
    assert m_href is not None, (
        "Expected `a.getAttribute('href')` inside the forEach body"
    )

    # (b) early-return guard: both startsWith('#') and startsWith('/') must be present
    starts_hash = re.compile(r"href\.startsWith\(\s*['\"]#['\"]\s*\)")
    starts_slash = re.compile(r"href\.startsWith\(\s*['\"][/]['\"]")
    m_hash = starts_hash.search(body)
    m_slash = starts_slash.search(body)
    assert m_hash is not None, (
        "Expected `href.startsWith('#')` guard in the forEach body"
    )
    assert m_slash is not None, (
        "Expected `href.startsWith('/')` guard in the forEach body"
    )

    # (c) setAttribute('target', '_blank') must appear AFTER both guards
    set_blank = re.compile(r"setAttribute\(\s*['\"]target['\"]")
    m_blank = set_blank.search(body)
    assert m_blank is not None, (
        "Expected `setAttribute('target', ...)` in the forEach body"
    )
    assert m_hash.start() < m_blank.start(), (
        "startsWith('#') guard must precede setAttribute('target', '_blank') "
        f"(hash guard at {m_hash.start()}, setAttribute at {m_blank.start()})"
    )
    assert m_slash.start() < m_blank.start(), (
        "startsWith('/') guard must precede setAttribute('target', '_blank') "
        f"(slash guard at {m_slash.start()}, setAttribute at {m_blank.start()})"
    )
