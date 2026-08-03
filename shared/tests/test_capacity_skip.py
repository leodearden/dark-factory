"""Contract tests for the shared ``_capacity_skip`` module.

Locks the runtime behaviour of the capacity-failure skip guard that was
previously copied byte-for-byte into ``test_cli_invoke_integration.py`` and
``test_usage_gate.py``.  Both copies fed live ``pytest.skip`` call sites, so a
drift between them is a silently-skipped (or silently-red) integration test.

Deliberately carries NO ``@pytest.mark.integration`` marker.  ``shared/
pyproject.toml`` sets ``addopts = "-m 'not integration'"``, so coverage that
lives in an integration module never runs by default — which is exactly how
the marker list drifted unnoticed in the first place.  These cases must run in
every ordinary ``uv run pytest tests/`` invocation.
"""

from __future__ import annotations

import pytest
from _capacity_skip import (
    CAPACITY_FAILURE_MARKERS,
    REAL_CLI_CAP_HIT_MESSAGES,
    REAL_CLI_CAP_MESSAGES,
    REAL_CLI_NEAR_CAP_MESSAGES,
    looks_like_capacity_failure,
    result_looks_like_capacity_failure,
)

from shared.cli_invoke import AgentResult
from shared.invocation_outcome import CapHit, NearCap, classify_invocation

# The verbatim real-CLI strings are DERIVED from _capacity_skip, never restated
# here.  Restating them would put two hand-maintained copies of the same
# transcript in two files — the duplication class this module exists to remove,
# reintroduced at smaller scale — and these strings carry exact punctuation
# (note the '·' in the weekly-limit message) that a silent divergence would
# quietly stop testing.  Deriving also routes every corpus message through the
# AgentResult entry point that the live pytest.skip sites actually call.
#
# Migrated from test_cli_invoke_integration.py::TestLooksLikeCapacityFailure.
# What is spelled out below is only the helper-only phrases: realistic capacity
# wording that this guard matches but production deliberately does not classify
# as a cap, so it has no place in the shared real-CLI corpus.
_HELPER_ONLY_POSITIVES = [
    "Your account is capped until the next billing cycle.",
    "Rate limit exceeded. Please wait and retry.",
    "account unavailable at this time; try again later.",
]

_POSITIVE_OUTPUTS = [*REAL_CLI_CAP_MESSAGES, *_HELPER_ONLY_POSITIVES]

# A cap message can land on either stream, so stderr gets the same corpus.
_POSITIVE_STDERRS = [*REAL_CLI_CAP_MESSAGES, 'rate limit: too many requests']

_CASE_INSENSITIVE_OUTPUTS = [
    "YOU'VE HIT YOUR USAGE LIMIT FOR CLAUDE PRO.",
    "YOUR ACCOUNT IS CAPPED.",
    "RATE LIMIT EXCEEDED.",
]

_NEGATIVES = [
    # Generic non-capacity failures — must NOT trigger skip
    ('process spawn failed: ENOENT', 'Traceback (most recent call last): ...'),
    ('malformed JSON response: unexpected token', ''),
    ('OAuth token validation failed: 401 Unauthorized', ''),
    # Substring boundary collisions — the narrowed markers must not match
    ('account uncapped and ready to use', ''),         # 'uncapped' must not match ' capped'
    ('service unavailable: DNS resolution failed', ''),  # generic 'unavailable' != 'account unavailable'
    ("You've used the wrong format. Please retry.", ''),  # must NOT match loose "you've used"
    ("You've hit a snag — try again later.", ''),         # must NOT match loose "you've hit"
    ('', ''),  # empty result
]


class TestResultEntryPoint:
    """``result_looks_like_capacity_failure(AgentResult)`` — what the two skip
    sites in ``test_cli_invoke_integration.py`` pass."""

    @pytest.mark.parametrize('cli_output', _POSITIVE_OUTPUTS)
    def test_capacity_output_returns_true(self, cli_output):
        """Realistic Claude CLI cap messages in output are detected."""
        result = AgentResult(success=False, output=cli_output, stderr='')
        assert result_looks_like_capacity_failure(result)

    @pytest.mark.parametrize('cli_stderr', _POSITIVE_STDERRS)
    def test_capacity_stderr_returns_true(self, cli_stderr):
        """Realistic Claude CLI cap messages in stderr are also detected."""
        result = AgentResult(success=False, output='', stderr=cli_stderr)
        assert result_looks_like_capacity_failure(result)

    @pytest.mark.parametrize('output', _CASE_INSENSITIVE_OUTPUTS)
    def test_case_insensitive_returns_true(self, output):
        """Cap detection is case-insensitive."""
        result = AgentResult(success=False, output=output, stderr='')
        assert result_looks_like_capacity_failure(result)

    @pytest.mark.parametrize('output,stderr', _NEGATIVES)
    def test_non_capacity_failure_returns_false(self, output, stderr):
        """Generic failures and substring boundary cases do not trigger a skip."""
        result = AgentResult(success=False, output=output, stderr=stderr)
        assert not result_looks_like_capacity_failure(result)


class TestTextEntryPoint:
    """``looks_like_capacity_failure(text)`` — what ``test_usage_gate.py``'s
    probe-precedence guard passes (a pre-combined ``f'{stdout}\\n{stderr}'``)."""

    @pytest.mark.parametrize(
        'text', _POSITIVE_OUTPUTS + _POSITIVE_STDERRS + _CASE_INSENSITIVE_OUTPUTS,
    )
    def test_capacity_text_returns_true(self, text):
        assert looks_like_capacity_failure(text)

    @pytest.mark.parametrize('output,stderr', _NEGATIVES)
    def test_non_capacity_text_returns_false(self, output, stderr):
        assert not looks_like_capacity_failure(f'{output}\n{stderr}')


class TestResultFormStaysAPureDelegation:
    """``result_looks_like_capacity_failure`` is a calling convention, not a
    second policy — it must decide nothing ``looks_like_capacity_failure``
    does not already decide.

    Two cases, not a full sweep of the corpus: the delegation is one line, so
    sweeping every message only re-asserts the same expression on both sides.
    What is worth pinning is the shape — that the result form keeps deriving
    its answer from the text form over both streams — so this goes red if
    someone grows independent matching logic here (the two-implementations
    bug this module was extracted to kill, one level down).
    """

    @pytest.mark.parametrize(
        'output,stderr',
        [
            ("You've hit your weekly limit · resets Aug 5, 11am", ''),
            ('', 'rate limit: too many requests'),
        ],
    )
    def test_result_form_matches_text_form(self, output, stderr):
        result = AgentResult(success=False, output=output, stderr=stderr)
        assert (
            result_looks_like_capacity_failure(result)
            is looks_like_capacity_failure(f'{output}\n{stderr}')
        )


class TestNoDriftFromProductionDetector:
    """The drift guard.

    The skip helper and the production cap detector are two independent
    implementations of "is this a capacity failure?", and the helper's whole
    job is to not disagree with reality about a real cap message. A red here
    means one of the two has been narrowed until they disagree about a
    message the CLI genuinely emitted — which, at a skip call site, means an
    integration test fails loudly with a cap message dressed up as a
    regression (or, in the other direction, skips when it should not).

    Deliberately NOT a snapshot of the expected marker tuple: that pins the
    strings but not the property, and any future narrowing would be "fixed"
    by editing the expectation. Cross-checking against production re-fails
    automatically instead.

    Guarded in BOTH directions, because widening carries its own risk: this
    task widened ``"you've hit your usage"`` to ``"you've hit your"`` and added
    ``'weekly limit'`` / ``'session limit'``, all of which loosen the guard and
    so raise the false-skip risk. Accepting real cap messages is pinned over
    ``REAL_CLI_CAP_MESSAGES``; refusing ordinary failures is pinned over
    ``_NEGATIVES``, run past production too rather than against the helper
    alone.
    """

    @pytest.mark.parametrize('message', REAL_CLI_CAP_MESSAGES)
    def test_skip_guard_accepts_every_real_cap_message(self, message):
        assert looks_like_capacity_failure(message), (
            f'skip guard does NOT match a real CLI cap message: {message!r}. '
            f'The production detector classifies it as a cap; this helper does '
            f'not, so a capped account will fail its test loudly instead of '
            f'skipping. Widen CAPACITY_FAILURE_MARKERS in _capacity_skip.py.'
        )

    @pytest.mark.parametrize(
        'message,expected',
        [
            *((m, CapHit) for m in REAL_CLI_CAP_HIT_MESSAGES),
            *((m, NearCap) for m in REAL_CLI_NEAR_CAP_MESSAGES),
        ],
    )
    def test_production_detector_assigns_the_expected_cap_variant(
        self, message, expected,
    ):
        """The exact variant, not merely "some kind of cap".

        ``CapHit`` and ``NearCap`` are not interchangeable downstream:
        usage_gate routes CapHit to ``_handle_cap_detected`` (CAPPED phase,
        cap-retry / account failover) and NearCap to
        ``_handle_near_cap_warning``, which only annotates.  A loose
        ``isinstance(outcome, (CapHit, NearCap))`` would stay green through a
        regression that downgraded every genuine cap-hit to a near-cap
        warning — i.e. through a change that disables cap-retry entirely.
        """
        outcome = classify_invocation(
            AgentResult(success=False, output=message, stderr=''),
            strict_confirm=True,
        )
        assert isinstance(outcome, expected), (
            f'production classify_invocation assigns the wrong outcome to a '
            f'real CLI cap message: {message!r} -> {type(outcome).__name__}, '
            f'expected {expected.__name__}. A CapHit downgraded to NearCap '
            f'means a capped account is annotated instead of failed over; a '
            f'cap missed entirely means no cap-retry at all.'
        )

    @pytest.mark.parametrize('output,stderr', _NEGATIVES)
    def test_both_detectors_refuse_every_non_cap_failure(self, output, stderr):
        """The other direction: neither side may call an ordinary failure a cap.

        A red on the helper half means the markers have been widened until a
        plain failure trips the skip guard — a real regression silently
        skipped, which is the worse failure mode of the two.
        """
        assert not looks_like_capacity_failure(f'{output}\n{stderr}'), (
            f'skip guard matches a NON-capacity failure: '
            f'{output!r} / {stderr!r}. A marker has been widened too far; at a '
            f'skip call site this silently skips a genuine regression.'
        )
        outcome = classify_invocation(
            AgentResult(success=False, output=output, stderr=stderr),
            strict_confirm=True,
        )
        assert not isinstance(outcome, (CapHit, NearCap)), (
            f'production classify_invocation treats a NON-capacity failure as '
            f'a cap: {output!r} / {stderr!r} -> {outcome!r}. This would fail an '
            f'account over to the CAPPED phase on an ordinary error.'
        )


def test_markers_are_lowercase():
    """Matching lowercases the haystack, so an upper/mixed-case marker would
    silently never fire — a guard that always returns False reads as "no cap
    message ever seen" instead of failing."""
    assert CAPACITY_FAILURE_MARKERS
    assert all(m == m.lower() for m in CAPACITY_FAILURE_MARKERS), (
        f'non-lowercase marker(s): '
        f'{[m for m in CAPACITY_FAILURE_MARKERS if m != m.lower()]}'
    )
