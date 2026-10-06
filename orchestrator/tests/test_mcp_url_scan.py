"""Unit tests for the AST-based trailing-slash MCP URL detector.

The repo-wide sweep guard in ``test_mcp_post_transport.py`` used to look for
the substring ``}/mcp/'``. That anchor was chosen because a bare ``/mcp/``
also matches unrelated prose in ``merge_queue.py`` docstrings — but it bought
that immunity by hard-coding one single spelling of one single construct, so
it silently let through every other way of writing the same defect.

These cases are exactly the spellings the substring form got WRONG — plus the
ones the FIRST AST cut still got wrong, which is the more useful half. Each is
real parsing behaviour fed to the detector as INPUT, not a constant pinned
against itself: the previous meta-test asserted only that an anchor string was
a substring of a literal written two lines above it, which tested nothing
about any system under test.

``'%s/mcp/' % base`` was a MEASURED false negative of that first cut (it
returned ``[]``), so "matching on the parsed tail catches every spelling" was
a claim, not yet a property. The percent-operator and unmodelled-operator
cases below are what turns it into one, and the "reported once not twice"
cases pin the cost of getting there — teaching the detector a new operator
must not start naming existing offenders twice.

``find_trailing_slash_mcp_urls(source, *, filename)`` returns
``list[tuple[int, str]]`` — the 1-based line number and the stripped source
line — so the sweep guard can name offenders precisely.

``sweep_source(source, *, filename)`` layers the inline
``# mcp-url-sweep: allow <reason>`` marker policy over that raw detector and
returns a ``SweepFindings``: the hits no well-formed marker exempts, the
markers that exempt nothing, and the markers that are not well-formed. The
ALLOW MARKERS cases below pin that policy; the raw-detector cases above stay
marker-unaware on purpose, because staleness is only checkable against the
unfiltered hits.
"""

from __future__ import annotations

import textwrap

import pytest
from _mcp_url_scan import SweepFindings, find_trailing_slash_mcp_urls, sweep_source


def _scan(source: str) -> list[tuple[int, str]]:
    """Dedent a snippet and scan it, so cases can be written as indented literals."""
    return find_trailing_slash_mcp_urls(textwrap.dedent(source), filename='<snippet>')


def _sweep(source: str) -> SweepFindings:
    """Dedent a snippet and sweep it under the allow-marker policy."""
    return sweep_source(textwrap.dedent(source), filename='<snippet>')


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
    """A live spelling: ``fused-memory/src/fused_memory/server/main.py::run_server``.

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


def test_flags_percent_format_operator_on_a_url():
    """MEASURED false negative of the first cut: ``'%s/mcp/' % base`` scored [].

    Distinct from the deferred-interpolation case above — here the ``%`` is a
    real ``BinOp``, so the URL is built eagerly and handed straight to
    ``client.post``.  It returned NO hits because the suppression pass treated
    every ``BinOp`` as owning its children's tails while ``_literal_tail``
    modelled only ``Add``: suppressed by one half, unmodelled by the other.
    That is the most dangerous shape a detector can have, because it reads as
    a clean sweep.
    """
    hits = _scan("""\
        resp = await client.post('%s/mcp/' % base, json=payload)
        """)

    assert hits == [(1, "resp = await client.post('%s/mcp/' % base, json=payload)")]


def test_flags_percent_format_operator_in_a_log_call():
    """The same operator spelling as ``server/main.py::run_server``, eagerly applied."""
    hits = _scan("""\
        logger.info('  Recon Report Endpoint: http://%s:%d/mcp/' % (host, port))
        """)

    assert hits == [
        (1, "logger.info('  Recon Report Endpoint: http://%s:%d/mcp/' % (host, port))")
    ]


def test_flags_a_slash_tail_under_an_unmodelled_operator():
    """An operator ``_literal_tail`` does not model must not SUPPRESS its operands.

    ``'/mcp/' * n`` is exotic, but it is the general form of the bug the
    percent case was an instance of: a node kind that suppressed its children
    while contributing no tail of its own erased the site entirely.  With
    suppression restricted to the operators actually modelled, the inner
    literal is reported instead — a double report at worst, never a miss.
    """
    hits = _scan("""\
        pad = '/mcp/' * n
        """)

    assert hits == [(1, "pad = '/mcp/' * n")]


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


def test_ignores_a_percent_template_that_does_not_end_in_the_slash():
    """``'/mcp/%s' % base`` cannot be redirected: the slash is not the tail.

    Pins that modelling ``Mod`` did not degrade into "any ``%`` template
    containing ``/mcp/``" — the same over-broad match that makes a bare
    ``/mcp/`` substring anchor unsatisfiable.
    """
    hits = _scan("""\
        resp = await client.post('/mcp/%s' % suffix, json=payload)
        """)

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


def test_percent_operator_site_is_reported_once_not_twice():
    """Teaching ``_literal_tail`` about ``Mod`` must not double-report the site.

    The outer ``BinOp`` and its left ``Constant`` both end in the slash, so a
    fix that only taught ``_literal_tail`` — without keeping the ``Mod``
    suppression — would name every percent-format offender twice and make the
    guard's failure message misleading about how many sites exist.
    """
    hits = _scan("""\
        url = ('%s' + '/mcp/') % base
        """)

    assert hits == [(1, "url = ('%s' + '/mcp/') % base")]


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


# ---------------------------------------------------------------------------
# ALLOW MARKERS — sweep_source
# ---------------------------------------------------------------------------


def test_reasoned_marker_on_the_hit_line_exempts_it():
    """A well-formed marker on the reported line is the one sanctioned exemption."""
    findings = _sweep("""\
        cfg = {'url': f'{base}/mcp/'}  # mcp-url-sweep: allow CLI config entry; its client follows redirects
        """)

    assert findings == SweepFindings()


def test_unmarked_hit_is_an_offender():
    findings = _sweep("""\
        resp = await client.post(f'{base}/mcp/', json=p)
        """)

    assert findings.offenders == ((1, "resp = await client.post(f'{base}/mcp/', json=p)"),)
    assert findings.stale_markers == ()
    assert findings.malformed_markers == ()


@pytest.mark.parametrize(
    ('comment', 'reported_as'),
    [
        ('# mcp-url-sweep: allow', '# mcp-url-sweep: allow'),
        ('# mcp-url-sweep: allow   ', '# mcp-url-sweep: allow'),
        ('# mcp-url-sweep:allow', '# mcp-url-sweep:allow'),
    ],
    ids=['bare', 'whitespace-only-reason', 'no-space-no-reason'],
)
def test_bare_marker_never_exempts(comment, reported_as):
    """A marker without a reason cannot be a silent mute: the hit stays an offender.

    It is also reported in its own bucket, so the failure names the marker
    that needs a reason rather than leaving the reader to wonder why a marked
    line still fails.
    """
    line = f"resp = client.post(f'{{b}}/mcp/')  {comment}"

    findings = _sweep(line + '\n')

    assert findings.offenders == ((1, line.strip()),)
    assert findings.malformed_markers == ((1, reported_as),)
    assert findings.stale_markers == ()


def test_misspelled_marker_verb_is_malformed_and_does_not_exempt():
    findings = _sweep("""\
        resp = client.post(f'{b}/mcp/')  # mcp-url-sweep: alow redirects ok
        """)

    assert findings.offenders == (
        (1, "resp = client.post(f'{b}/mcp/')  # mcp-url-sweep: alow redirects ok"),
    )
    assert findings.malformed_markers == ((1, '# mcp-url-sweep: alow redirects ok'),)
    assert findings.stale_markers == ()


def test_reasoned_marker_on_a_line_with_no_slashed_url_is_stale():
    """A marker that outlived its defect is a hole waiting for the next one."""
    findings = _sweep("""\
        x = f'{base}/mcp'  # mcp-url-sweep: allow was a log string
        """)

    assert findings.stale_markers == ((1, '# mcp-url-sweep: allow was a log string'),)
    assert findings.offenders == ()
    assert findings.malformed_markers == ()


@pytest.mark.parametrize(
    ('source', 'hit_line', 'marker_line'),
    [
        ("# mcp-url-sweep: allow covers it\nresp = client.post(f'{b}/mcp/')\n", 2, 1),
        ("resp = client.post(f'{b}/mcp/')\n# mcp-url-sweep: allow covers it\n", 1, 2),
    ],
    ids=['line-above', 'line-below'],
)
def test_marker_on_a_neighbouring_line_does_not_exempt(source, hit_line, marker_line):
    """A marker exempts only the expression it sits on, never a neighbouring line's."""
    findings = _sweep(source)

    assert findings.offenders == ((hit_line, "resp = client.post(f'{b}/mcp/')"),)
    assert findings.stale_markers == ((marker_line, '# mcp-url-sweep: allow covers it'),)
    assert findings.malformed_markers == ()


def test_marker_after_a_multiline_triple_quoted_literal_exempts_it():
    """A string spanning lines can only carry a comment after its closing quotes.

    The guard reports the line the string OPENS on, and that line ends inside
    the string, so a rule demanding the marker on exactly the reported line
    would leave this site impossible to exempt.
    """
    unmarked = _sweep('''\
        url = """http://host
        /mcp/"""
        ''')
    marked = _sweep('''\
        url = """http://host
        /mcp/"""  # mcp-url-sweep: allow fixture text, never fetched
        ''')

    assert unmarked.offenders == ((1, 'url = """http://host'),)
    assert marked == SweepFindings()


@pytest.mark.parametrize('marked_line', [1, 2], ids=['opening-line', 'literal-line'])
def test_marker_on_any_line_of_a_multiline_slashed_expression_exempts_it(marked_line):
    """``base +\\n '/mcp/'`` is reported where the ``+`` expression begins.

    That is the ``base`` line, not the literal's, so the policy accepts a
    marker on any line the flagged expression spans rather than making the
    author guess which of them the guard means.
    """
    lines = ['url = (base +', "       '/mcp/')"]
    unmarked = _sweep('\n'.join(lines) + '\n')
    lines[marked_line - 1] += '  # mcp-url-sweep: allow fixture text, never fetched'
    marked = _sweep('\n'.join(lines) + '\n')

    assert unmarked.offenders == ((1, 'url = (base +'),)
    assert marked == SweepFindings()


def test_marker_text_inside_a_string_or_docstring_is_not_a_marker():
    """Markers are comment TOKENS, not substrings of the line text.

    A substring search would let a string argument mute its own call's hit,
    and would report a docstring that merely documents the convention as a
    stale marker.
    """
    in_a_string = _sweep("""\
        resp = client.post(f'{b}/mcp/', note='# mcp-url-sweep: allow sneaky')
        """)
    in_a_docstring = _sweep('''\
        def documented():
            """Exempt a line with ``# mcp-url-sweep: allow <reason>``."""
            return None
        ''')

    assert in_a_string.offenders == (
        (1, "resp = client.post(f'{b}/mcp/', note='# mcp-url-sweep: allow sneaky')"),
    )
    assert in_a_string.stale_markers == ()
    assert in_a_string.malformed_markers == ()
    assert in_a_docstring == SweepFindings()


def test_prose_mentioning_the_convention_mid_comment_is_not_a_marker():
    """A marker must BEGIN the comment, like ``# type:`` and ``# pragma:``."""
    findings = _sweep("""\
        y = 1  # see the mcp-url-sweep: allow convention
        """)

    assert findings == SweepFindings()


def test_marker_on_a_multiline_call_belongs_on_the_literal_line():
    """The ``server/main.py::run_server`` shape: the literal opens on line 2, not 1."""
    findings = _sweep("""\
        logger.info(
            '  Recon Report Endpoint: http://%s:%d/mcp/',  # mcp-url-sweep: allow log string, never fetched
            host,
            port,
        )
        """)

    assert findings == SweepFindings()


def test_sweep_source_still_fails_loudly_on_unparseable_source():
    """The marker layer must not swallow the raw detector's parse failure."""
    with pytest.raises(AssertionError) as excinfo:
        sweep_source(
            'def broken(:\n    pass  # mcp-url-sweep: allow x\n',
            filename='orchestrator/src/orchestrator/nope.py',
        )

    assert 'orchestrator/src/orchestrator/nope.py' in str(excinfo.value)
