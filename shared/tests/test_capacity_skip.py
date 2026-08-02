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
    looks_like_capacity_failure,
    result_looks_like_capacity_failure,
)

from shared.cli_invoke import AgentResult

# Migrated verbatim from test_cli_invoke_integration.py::TestLooksLikeCapacityFailure.
# The four leading entries are verbatim Claude CLI cap-hit messages — the actual
# strings that motivated the marker list.  (Their original provenance comment
# pointed at ``shared.usage_gate`` inline comments; task 2129 moved those string
# tables to ``shared.invocation_outcome`` — CAP_HIT_PREFIXES / CAP_CONFIRM_KEYWORDS
# / NEAR_CAP_PREFIXES.)
_POSITIVE_OUTPUTS = [
    "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
    "You've used all available credits. Upgrade your plan for more capacity.",
    "You're out of extra usage for this billing period. Your plan resets in 2h.",
    "You're close to reaching your usage limit. Your plan resets in 1h.",
    # Other realistic capacity phrases
    "Your account is capped until the next billing cycle.",
    "Rate limit exceeded. Please wait and retry.",
    "account unavailable at this time; try again later.",
]

_POSITIVE_STDERRS = [
    "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
    "You're out of extra usage for this billing period. Your plan resets in 2h.",
    "rate limit: too many requests",
]

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


class TestEntryPointsAgree:
    """The two entry points are one policy with two calling conventions.  If
    they can disagree, the two skip sites can disagree — the exact class of
    divergence this module exists to make impossible."""

    @pytest.mark.parametrize(
        'output,stderr',
        [(o, '') for o in _POSITIVE_OUTPUTS]
        + [('', s) for s in _POSITIVE_STDERRS]
        + [(o, '') for o in _CASE_INSENSITIVE_OUTPUTS]
        + _NEGATIVES,
    )
    def test_text_and_result_forms_agree(self, output, stderr):
        result = AgentResult(success=False, output=output, stderr=stderr)
        assert (
            result_looks_like_capacity_failure(result)
            is looks_like_capacity_failure(f'{output}\n{stderr}')
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
