"""Unit tests for the AST-based trailing-slash MCP URL detector.

The repo-wide sweep guard in ``test_mcp_post_transport.py`` used to look for
the substring ``}/mcp/'``. That anchor was chosen because a bare ``/mcp/``
also matches unrelated prose in ``merge_queue.py`` docstrings — but it bought
that immunity by hard-coding one single spelling of one single construct, so
it silently let through every other way of writing the same defect.

These cases are exactly the spellings the substring form got WRONG. Each is
real parsing behaviour fed to the detector as INPUT, not a constant pinned
against itself: the previous meta-test asserted only that an anchor string was
a substring of a literal written two lines above it, which tested nothing
about any system under test.

``find_trailing_slash_mcp_urls(source, *, filename)`` returns
``list[tuple[int, str]]`` — the 1-based line number and the stripped source
line — so the sweep guard can name offenders precisely.
"""

from __future__ import annotations

import textwrap

import pytest

from _mcp_url_scan import find_trailing_slash_mcp_urls


def _scan(source: str) -> list[tuple[int, str]]:
    """Dedent a snippet and scan it, so cases can be written as indented literals."""
    return find_trailing_slash_mcp_urls(textwrap.dedent(source), filename='<snippet>')


# ---------------------------------------------------------------------------
# POSITIVES — every one of these builds a URL the server 307-redirects away.
# ---------------------------------------------------------------------------


def test_flags_single_quoted_fstring():
    """The one form the old ``}/mcp/'`` substring anchor actually caught.

    This is the literal shape of all five defect sites, so a detector that
    missed it would be strictly worse than the anchor it replaces.
    """
    hits = _scan("""\
        resp = await client.post(f'{self.mcp.url}/mcp/', json=payload)
        """)

    assert hits == [(1, "resp = await client.post(f'{self.mcp.url}/mcp/', json=payload)")]


def test_flags_double_quoted_fstring():
    """The old anchor hard-coded ``'``, so the identical defect in ``"`` was invisible."""
    hits = _scan("""\
        resp = await client.post(f"{base_url}/mcp/", json=payload)
        """)

    assert hits == [(1, 'resp = await client.post(f"{base_url}/mcp/", json=payload)')]


def test_flags_plain_string_literal():
    """No ``}`` precedes the slash, so the old anchor could never match this."""
    hits = _scan("""\
        resp = await client.post('http://127.0.0.1:8002/mcp/', json=payload)
        """)

    assert hits == [(1, "resp = await client.post('http://127.0.0.1:8002/mcp/', json=payload)")]


def test_flags_percent_format_literal():
    """A live spelling: ``fused-memory/src/fused_memory/server/main.py:1149``.

    Percent-format defers interpolation to the logging call, so the URL is a
    plain ``Constant`` with no brace anywhere — invisible to the old anchor
    even though the trailing slash is right there in the literal.
    """
    hits = _scan("""\
        logger.info(
            '  Recon Report Endpoint: http://%s:%d/mcp/',
            host,
            port,
        )
        """)

    assert hits == [(2, "'  Recon Report Endpoint: http://%s:%d/mcp/',")]


def test_flags_string_concatenation():
    """``base + '/mcp/'`` — the tail lives in the right operand, not a brace."""
    hits = _scan("""\
        resp = await client.post(base + '/mcp/', json=payload)
        """)

    assert hits == [(1, "resp = await client.post(base + '/mcp/', json=payload)")]


# ---------------------------------------------------------------------------
# NEGATIVES — the fixed form, and the prose that broke the naive anchor.
# ---------------------------------------------------------------------------


def test_ignores_the_fixed_slashless_forms():
    """The shape every fixed call site now uses must be silent.

    A detector that flagged the fix would make the guard unsatisfiable, which
    is the failure mode a bare ``/mcp/`` substring anchor has.
    """
    hits = _scan("""\
        resp = await client.post(mcp_endpoint_url(self.mcp.url), json=payload)
        other = f'{url}/mcp'
        """)

    assert hits == []


def test_ignores_slash_mcp_slash_inside_prose():
    """The exact false positive that makes a bare ``/mcp/`` anchor unsatisfiable.

    ``merge_queue.py:1163`` and ``:2467`` both carry the phrase
    ``scheduler/mcp/usage_gate/cost_store`` in a docstring. Matching on the
    parsed literal's TAIL, rather than searching anywhere in the line, is what
    makes the detector immune: the prose contains ``/mcp/`` but the string does
    not end with it.
    """
    hits = _scan('''\
        def _advance(self):
            """Drive the scheduler/mcp/usage_gate/cost_store handoff."""
            return None
        ''')

    assert hits == []


# ---------------------------------------------------------------------------
# STRUCTURAL — the two ways a scanner is wrong without ever looking wrong.
# ---------------------------------------------------------------------------


def test_fstring_is_reported_once_not_twice():
    """A naive ``ast.walk`` double-reports every f-string site.

    ``f'{base}' '/mcp/'`` parses to ONE ``JoinedStr`` whose trailing element is
    a ``Constant``. Walking naively matches both, and because implicit
    concatenation puts them on DIFFERENT lines (measured: ``JoinedStr`` at 2,
    inner ``Constant`` at 3) the duplicate survives any de-duplication by
    line — it has to be prevented structurally, by skipping nodes nested
    inside a ``JoinedStr``.
    """
    hits = _scan("""\
        url = (
            f'{base}'
            '/mcp/'
        )
        """)

    assert hits == [(2, "f'{base}'")]


def test_unparseable_source_fails_loudly():
    """A scanner that swallows ``SyntaxError`` reports "clean" for a broken file.

    That is the vacuity failure mode that actually matters here: the sweep
    guard walks every file in five directories, so one unparseable file would
    silently shrink the guard's coverage while it kept passing green.
    """
    with pytest.raises(AssertionError) as excinfo:
        find_trailing_slash_mcp_urls(
            'def broken(:\n    pass\n', filename='orchestrator/src/orchestrator/nope.py'
        )

    assert 'orchestrator/src/orchestrator/nope.py' in str(excinfo.value)
