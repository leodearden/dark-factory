"""Contract tests for ``_cross_account_evidence`` — the cross-account measurement kit.

An ORDINARY (unmarked) test module: it runs in CI and in the verify lane, makes
no real CLI calls, and costs nothing.  The live measurement it supports lives in
``test_cli_invoke_integration.py::TestCrossAccountResume`` behind
``-m integration``.

What these tests are for (task 3484).  The cross-account resume question has now
gone two rounds with **0 valid runs** (task 3454, then again on 2026-08-05)
because the integration module hard-coded its account pair to the first two
tokens in env, and those were capped.  ``select_token_pair`` makes the pair
steerable so a measurement can be aimed at whichever accounts are actually
healthy when a window opens; ``format_run_evidence`` / ``emit_run_evidence``
make each run leave a durable record.  Both are pure/injected, so both are
pinnable here rather than only observable during live spend.
"""

from __future__ import annotations

import pytest
from _cross_account_evidence import select_token_pair


def _env(*letters: str, **extra: str) -> dict[str, str]:
    """Build a fake environ holding ``CLAUDE_OAUTH_TOKEN_<L>`` for each letter."""
    environ = {f'CLAUDE_OAUTH_TOKEN_{ch}': f'tok-{ch.lower()}' for ch in letters}
    environ.update(extra)
    return environ


class TestSelectTokenPairDefault:
    """No override set — today's ``_AVAILABLE_TOKENS[0]/[1]`` behaviour, exactly."""

    def test_returns_first_two_in_scan_order(self):
        (name_a, tok_a), (name_b, tok_b) = select_token_pair(_env('B', 'C', 'D'))
        assert (name_a, tok_a) == ('CLAUDE_OAUTH_TOKEN_B', 'tok-b')
        assert (name_b, tok_b) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_scan_order_is_bcdefg_not_environ_order(self):
        """Order comes from the fixed B,C,D,E,F,G scan, not dict insertion order."""
        environ = {
            'CLAUDE_OAUTH_TOKEN_F': 'tok-f',
            'CLAUDE_OAUTH_TOKEN_C': 'tok-c',
            'CLAUDE_OAUTH_TOKEN_E': 'tok-e',
        }
        pair = select_token_pair(environ)
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_E',
        ]

    def test_g_is_reachable(self):
        """G is a real account in .env that the BCDEF scan could not reach at all."""
        pair = select_token_pair(_env('F', 'G'))
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_G',
        ]

    def test_empty_string_token_does_not_count_as_available(self):
        environ = _env('C', 'D')
        environ['CLAUDE_OAUTH_TOKEN_B'] = ''
        pair = select_token_pair(environ)
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_D',
        ]

    def test_fewer_than_two_available_raises(self):
        """Callers turn this into a skip — it must not silently return one account."""
        with pytest.raises(ValueError) as exc:
            select_token_pair(_env('C'))
        assert 'CLAUDE_OAUTH_TOKEN_C' in str(exc.value) or 'two' in str(exc.value).lower()

    def test_no_tokens_at_all_raises(self):
        with pytest.raises(ValueError):
            select_token_pair({})


class TestSelectTokenPairOverride:
    """``CROSS_ACCOUNT_RESUME_TOKENS`` aims the measurement at a healthy pair."""

    def test_override_selects_named_pair_in_order(self):
        """Order is load-bearing: A starts the session, B resumes it."""
        environ = _env('B', 'C', 'D', 'E', 'F', 'G')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,C'
        (name_a, tok_a), (name_b, tok_b) = select_token_pair(environ)
        assert (name_a, tok_a) == ('CLAUDE_OAUTH_TOKEN_F', 'tok-f')
        assert (name_b, tok_b) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_override_order_is_not_normalised_to_scan_order(self):
        """'F,C' must NOT come back as (C, F) — that would swap which account resumes."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,C'
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_accepts_full_var_names(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = (
            'CLAUDE_OAUTH_TOKEN_F,CLAUDE_OAUTH_TOKEN_C'
        )
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_tolerates_whitespace_and_case(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = '  f , C  '
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_naming_unset_var_raises_and_names_the_entry(self):
        """A silent fall-back to the default pair is how a run gets aimed at a
        capped account without anyone noticing.  Fail loudly instead."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,D'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        message = str(exc.value)
        assert 'D' in message
        assert 'CROSS_ACCOUNT_RESUME_TOKENS' in message

    def test_override_naming_unknown_letter_raises_and_names_the_entry(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,ZZZ'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        assert 'ZZZ' in str(exc.value)

    def test_override_naming_same_account_twice_raises(self):
        """A same-account 'cross-account' run is not a measurement."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'C,C'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        assert 'CLAUDE_OAUTH_TOKEN_C' in str(exc.value)

    def test_override_with_wrong_arity_raises(self):
        environ = _env('C', 'F', 'G')
        for value in ('C', 'C,F,G', ''):
            environ['CROSS_ACCOUNT_RESUME_TOKENS'] = value
            with pytest.raises(ValueError):
                select_token_pair(environ)

    def test_override_wins_even_when_default_pair_is_available(self):
        """The whole point: B,C are present and would be the default, but the
        operator has measured them capped and aimed at F,G instead."""
        environ = _env('B', 'C', 'F', 'G')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'G,F'
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_G',
            'CLAUDE_OAUTH_TOKEN_F',
        ]
