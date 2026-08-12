"""Contract tests for ``shared.phantom_verdict`` — the row-level phantom rule.

A PHANTOM verdict is a placeholder the reconciliation judge fabricates when it
could not parse its own model output into a verdict at all (see
``Judge._parse_verdict``'s except block).  There is no review behind it, so
counting one as a genuine ``severity=serious`` finding mis-reports a review
that never happened — which is exactly what the live DB does today: nine of
the ten ``serious`` rows are fabricated.

The rule lives in ``shared`` rather than in ``judge.py`` because its two new
consumers cannot import ``judge.py``: ``journal.py`` would form an import
cycle, and the ``dashboard`` package deliberately does not depend on
``fused-memory`` at all.  These cases pin the rule at the PRIMITIVE level
(``severity: str``, ``findings: list[dict]``) so both a pydantic model field
and a raw SQLite row can be classified by one implementation.

Every fixture shape below is copied from a row that actually exists in
``data/reconciliation/reconciliation.db`` rather than invented — the marked
phantom, the two pre-2947 unmarked legacy rows, and the sole genuine
``serious`` verdict (``bc9459b8``, five findings) that the single-finding
conjunct exists to protect.
"""

from __future__ import annotations

import enum

import pytest

from shared.phantom_verdict import (
    UNPARSEABLE_ISSUE_PREFIX,
    UNPARSEABLE_VERDICT_CODE,
    has_unparseable_marker,
    is_phantom_verdict_row,
)

# --- corpus: the real stored shapes ------------------------------------------

# The marked phantom, as written for every row after task 2947 landed
# (2026-07-23).  Seven of the nine live phantoms carry this shape.
MARKED_PHANTOM_FINDINGS = [
    {
        'code': 'unparseable_judge_response',
        'issue': 'Judge response could not be parsed: Expecting value: line 1 column 1',
        'severity': 'serious',
    }
]

# The pre-2947 legacy phantom: no 'code' key at all, identifiable ONLY by the
# fabricated shape.  Live rows e87d8e4a (2026-07-20) and 09d14a75 (2026-07-22).
LEGACY_PHANTOM_FINDINGS = [
    {'issue': 'Judge response could not be parsed: Expecting value: line 1 column 1'}
]

# The sole GENUINE serious verdict in the live DB (bc9459b8, 2026-04-19): five
# findings, no marker.  This is the row the single-finding conjunct protects.
GENUINE_SERIOUS_FINDINGS = [
    {'issue': "Task Knowledge Sync stage failed with 'failed_closed_re...'"},
    {'issue': 'Entity resolution dropped two candidate merges'},
    {'issue': 'Judge queue backlog exceeded the configured ceiling'},
    {'issue': 'Stage 2 emitted no cycle summary for the window'},
    {'issue': 'Verdict store lagged the run journal by three runs'},
]

ORDINARY_FINDINGS = [{'issue': 'minor drift in entity summary wording'}]


# --- (a) the marker constants are exported and verbatim -----------------------


def test_unparseable_verdict_code_constant():
    """The structured marker string, verbatim as ``_parse_verdict`` stamps it."""
    assert UNPARSEABLE_VERDICT_CODE == 'unparseable_judge_response'


def test_unparseable_issue_prefix_constant():
    """The fabricated finding's issue prefix — the ONLY signal on a pre-2947 row."""
    assert UNPARSEABLE_ISSUE_PREFIX == 'Judge response could not be parsed'


# --- (b)(c)(d)(e) the two phantom shapes, and what must NOT match -------------


def test_marked_phantom_is_phantom():
    """(b) The post-2947 structured-marker shape classifies as phantom."""
    assert is_phantom_verdict_row('serious', MARKED_PHANTOM_FINDINGS) is True


def test_legacy_unmarked_phantom_is_phantom():
    """(c) The pre-2947 unmarked shape is caught by the fallback branch."""
    assert is_phantom_verdict_row('serious', LEGACY_PHANTOM_FINDINGS) is True


def test_genuine_serious_verdict_is_not_phantom():
    """(d) A real severity=serious review with five findings is NOT phantom.

    The regression this whole rule exists to avoid: suppressing a genuine
    serious finding would be strictly worse than the over-count it fixes.
    """
    assert is_phantom_verdict_row('serious', GENUINE_SERIOUS_FINDINGS) is False


@pytest.mark.parametrize('severity', ['ok', 'minor', 'moderate'])
def test_ordinary_non_serious_verdicts_are_not_phantom(severity):
    """(e) An ordinary finding at any non-serious severity is not phantom."""
    assert is_phantom_verdict_row(severity, ORDINARY_FINDINGS) is False


def test_issue_mentioning_prefix_mid_string_is_not_phantom():
    """(f) ``startswith``, NOT a substring test.

    A genuine finding whose prose merely MENTIONS the fabrication text must
    not be swallowed by the legacy branch.
    """
    findings = [
        {
            'issue': (
                'The stage log shows the string "Judge response could not be '
                'parsed" appearing in the retry banner, which suggests a '
                'truncated upstream payload.'
            )
        }
    ]
    assert is_phantom_verdict_row('serious', findings) is False


def test_marker_branch_is_severity_agnostic():
    """(g) The structured marker wins regardless of severity.

    ``_parse_verdict`` only ever fabricates ``serious`` today, but the marker
    is the machine-readable ground truth: if a future producer stamps it at
    another severity, the row is still a phantom.
    """
    assert is_phantom_verdict_row('moderate', MARKED_PHANTOM_FINDINGS) is True
    assert is_phantom_verdict_row('ok', MARKED_PHANTOM_FINDINGS) is True


# --- (h) robustness on degenerate rows ---------------------------------------


@pytest.mark.parametrize(
    'findings',
    [
        pytest.param([], id='no-findings'),
        pytest.param([{}], id='finding-without-issue-key'),
        pytest.param([{'issue': None}], id='issue-is-none'),
        pytest.param([{'issue': 42}], id='issue-is-int'),
        pytest.param([{'code': None, 'issue': None}], id='both-keys-none'),
    ],
)
def test_degenerate_findings_are_not_phantom_and_do_not_raise(findings):
    """(h) A malformed row is classified as genuine, never an exception.

    Fail-closed in the safe direction: an unrecognised shape stays counted as
    a real verdict rather than silently vanishing from the totals.
    """
    assert is_phantom_verdict_row('serious', findings) is False


def test_multi_finding_marked_row_is_phantom():
    """The marker branch does not require a single finding, unlike the legacy one."""
    findings = [{'issue': 'unrelated'}] + MARKED_PHANTOM_FINDINGS
    assert is_phantom_verdict_row('serious', findings) is True


def test_two_findings_without_marker_is_not_phantom():
    """The legacy branch requires EXACTLY one finding.

    Two findings means the judge produced structured content, so the
    fabrication path cannot have written it.
    """
    findings = LEGACY_PHANTOM_FINDINGS + [{'issue': 'a second, real finding'}]
    assert is_phantom_verdict_row('serious', findings) is False


# --- (i) severity accepts a plain str and a StrEnum member alike --------------


class _SeverityLike(enum.StrEnum):
    """Local stand-in for ``fused_memory.models.reconciliation.VerdictSeverity``.

    Constructed HERE rather than imported: ``shared`` must not depend on
    ``fused_memory`` (that dependency direction is the whole reason this module
    exists), so the StrEnum-compatibility contract is pinned with a local enum
    of the same shape.  ``VerdictSeverity`` is itself a ``StrEnum``, so a
    member compares equal to its value and one primitive comparison serves both
    a model field and a raw SQLite string.
    """

    ok = 'ok'
    minor = 'minor'
    moderate = 'moderate'
    serious = 'serious'


@pytest.mark.parametrize(
    'severity',
    [
        pytest.param('serious', id='plain-str'),
        pytest.param(_SeverityLike.serious, id='strenum-member'),
    ],
)
def test_severity_accepts_str_and_strenum(severity):
    """(i) The same call works for a raw DB string and a StrEnum model field."""
    assert is_phantom_verdict_row(severity, LEGACY_PHANTOM_FINDINGS) is True
    assert is_phantom_verdict_row(severity, GENUINE_SERIOUS_FINDINGS) is False


# --- has_unparseable_marker, the marker-scan half on its own ------------------


def test_has_unparseable_marker_true_on_marked_row():
    assert has_unparseable_marker(MARKED_PHANTOM_FINDINGS) is True


@pytest.mark.parametrize(
    'findings',
    [
        pytest.param(LEGACY_PHANTOM_FINDINGS, id='legacy-unmarked'),
        pytest.param(GENUINE_SERIOUS_FINDINGS, id='genuine-serious'),
        pytest.param([], id='empty'),
        pytest.param([{}], id='no-code-key'),
    ],
)
def test_has_unparseable_marker_false_without_the_code(findings):
    """The marker scan is strictly the ``code`` check — it does not peek at issue."""
    assert has_unparseable_marker(findings) is False
