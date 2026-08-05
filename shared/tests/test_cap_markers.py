"""Contract tests for ``shared.cap_markers`` — the canonical cap/auth banner list.

This module owns the loose-substring banner contract shared by the
``pytest.skip`` guard in ``_capacity_skip`` (task 3483) and the census runtime's
usage-headroom probe (``scripts/legibility/census.py``).  Because two
independent sites had each grown their own list, and each blind spot had to be
found separately, the list is exercised here as a CONTRACT against a real-CLI
corpus rather than as a literal: a newly-observed cap phrasing is ONE edit to
``REAL_CLI_CAP_MESSAGES`` that turns both this suite and
``scripts/tests/test_legibility_census.py`` red until the markers cover it.

Deliberately carries NO ``@pytest.mark.integration`` marker, for the same
reason ``test_capacity_skip.py`` does not: ``shared/pyproject.toml`` sets
``addopts = "-m 'not integration'"``, so coverage parked in an integration
module never runs by default — which is how the marker list drifted unnoticed
the first time.  These cases must run in every ordinary ``uv run pytest
tests/``.
"""

from __future__ import annotations

import pytest

from shared.cap_markers import (
    AUTH_BANNER_MARKERS,
    BLOCKING_BANNER_MARKERS,
    CAPACITY_BANNER_MARKERS,
    REAL_CLI_CAP_MESSAGES,
    looks_like_blocking_banner,
    looks_like_capacity_banner,
)

# The two real weekly-limit spellings.  The middot form is the verbatim
# transcript already carried in the corpus (observed 2026-08-01, claude CLI
# 2.1.220); the hyphen form is the spelling quoted in task 3645's ticket.  Both
# are pinned explicitly because this exact message is the one that motivated
# the work: it contains NONE of the census's four original markers
# ('usage limit' / 'rate limit' / 'please run /login' / 'invalid api key'), so
# a weekly cap sailed straight through the preflight probe.
_WEEKLY_LIMIT_MIDDOT = "You've hit your weekly limit · resets Aug 5, 11am"
_WEEKLY_LIMIT_HYPHEN = "You've hit your weekly limit - resets Aug 5, 11am"

# Ordinary, non-blocking CLI failures.  A false positive here is not harmless:
# at the ``pytest.skip`` sites it silently skips a real regression, and in the
# census it defers a run that had every right to proceed.
_NEGATIVES = [
    'pong',                                       # the healthy headroom-probe reply
    'Error: unrecognized arguments: --nope',
    'Traceback (most recent call last): ...',
    'No such file or directory',
    'process spawn failed: ENOENT',
    'malformed JSON response: unexpected token',
    '',
]

# Auth banners: blocking (no useful model output is coming) but NOT capacity.
_AUTH_ONLY = [
    'Please run /login to authenticate.',
    'Invalid API key provided.',
]


class TestMarkerTupleShape:
    """Structural invariants the matchers depend on."""

    @pytest.mark.parametrize(
        'name,markers',
        [
            ('CAPACITY_BANNER_MARKERS', CAPACITY_BANNER_MARKERS),
            ('AUTH_BANNER_MARKERS', AUTH_BANNER_MARKERS),
            ('BLOCKING_BANNER_MARKERS', BLOCKING_BANNER_MARKERS),
        ],
    )
    def test_markers_non_empty(self, name, markers):
        """An empty tuple would make every matcher silently return None."""
        assert markers, f'{name} must not be empty'

    @pytest.mark.parametrize(
        'name,markers',
        [
            ('CAPACITY_BANNER_MARKERS', CAPACITY_BANNER_MARKERS),
            ('AUTH_BANNER_MARKERS', AUTH_BANNER_MARKERS),
            ('BLOCKING_BANNER_MARKERS', BLOCKING_BANNER_MARKERS),
        ],
    )
    def test_markers_are_lowercase(self, name, markers):
        """The matchers lowercase the HAYSTACK, not the needle.

        An uppercase character anywhere in a marker therefore makes that entry
        permanently dead — it can never match — which is a silent coverage
        hole, not a loud failure.  Pin it structurally.
        """
        for marker in markers:
            assert marker == marker.lower(), f'{name} entry {marker!r} is not lowercase'

    def test_capacity_and_auth_are_disjoint(self):
        """The two sets are deliberately separate policies, not one list.

        ``_capacity_skip`` matches capacity ONLY (an auth failure there must
        stay a loud red test, never a quiet skip); the census probe defers on
        either.  An entry drifting into both tuples would blur that split.
        """
        assert not (set(CAPACITY_BANNER_MARKERS) & set(AUTH_BANNER_MARKERS))

    def test_blocking_is_exactly_the_ordered_union(self):
        """``BLOCKING_BANNER_MARKERS`` is derived, never hand-maintained."""
        assert BLOCKING_BANNER_MARKERS == CAPACITY_BANNER_MARKERS + AUTH_BANNER_MARKERS


class TestLooksLikeCapacityBanner:
    """``looks_like_capacity_banner`` — returns the MATCHED marker, or None."""

    @pytest.mark.parametrize('message', REAL_CLI_CAP_MESSAGES)
    def test_every_real_cli_cap_message_matches(self, message):
        """The contract: every verbatim real-CLI cap message is covered.

        Asserting on the corpus rather than on hand-picked strings is the whole
        point — a test author who invents strings to match the markers already
        written proves nothing about text the CLI actually emits.
        """
        marker = looks_like_capacity_banner(message)
        assert marker is not None, f'no marker covers real CLI message {message!r}'
        assert isinstance(marker, str)

    @pytest.mark.parametrize('message', REAL_CLI_CAP_MESSAGES)
    def test_returned_marker_is_a_real_list_member(self, message):
        """Returning the matched marker (not a bool) is load-bearing.

        The census quotes it back in its deferral reason
        (``"...carries a banner marker: 'weekly limit'"``), so an operator can
        see WHICH signal fired.  It must therefore be an actual list member,
        not a reconstructed or prettified string.
        """
        assert looks_like_capacity_banner(message) in CAPACITY_BANNER_MARKERS

    @pytest.mark.parametrize(
        'message',
        [
            _WEEKLY_LIMIT_MIDDOT,
            _WEEKLY_LIMIT_HYPHEN,
            _WEEKLY_LIMIT_MIDDOT.upper(),
            _WEEKLY_LIMIT_MIDDOT.lower(),
        ],
    )
    def test_weekly_limit_both_spellings_and_cases(self, message):
        """Both observed separator spellings, case-insensitively.

        The separator is incidental punctuation the CLI has already varied; a
        marker list that depends on it is one release away from silence.
        """
        assert looks_like_capacity_banner(message) is not None

    @pytest.mark.parametrize('text', _NEGATIVES)
    def test_ordinary_failures_do_not_match(self, text):
        assert looks_like_capacity_banner(text) is None

    @pytest.mark.parametrize('text', _AUTH_ONLY)
    def test_auth_text_is_not_a_capacity_banner(self, text):
        """The deliberate split: auth is blocking, but it is not a cap.

        Folding auth into the capacity list would silently convert every auth
        failure in ``test_cli_invoke_integration.py`` from a red test into a
        skip — a behaviour regression in the module this consolidation is
        meant to unify with.
        """
        assert looks_like_capacity_banner(text) is None


class TestLooksLikeBlockingBanner:
    """``looks_like_blocking_banner`` — the union the census probe scans."""

    @pytest.mark.parametrize('message', REAL_CLI_CAP_MESSAGES)
    def test_capacity_messages_are_blocking(self, message):
        assert looks_like_blocking_banner(message) in BLOCKING_BANNER_MARKERS

    @pytest.mark.parametrize('text', _AUTH_ONLY)
    def test_auth_text_is_blocking(self, text):
        """Auth failure means no useful model output is coming either."""
        marker = looks_like_blocking_banner(text)
        assert marker is not None
        assert marker in AUTH_BANNER_MARKERS

    @pytest.mark.parametrize('text', _NEGATIVES)
    def test_ordinary_failures_do_not_match(self, text):
        assert looks_like_blocking_banner(text) is None
