"""Contract tests for ``_oauth_accounts`` — the single home for the OAuth account scan.

An ORDINARY (unmarked) test module: it runs in CI and in the verify lane, makes
no real CLI calls, and costs nothing.  Same shape as ``test_capacity_skip.py``,
which pins the other shared test-kit module extracted for the same reason.

What these tests are for (task 3700).  The "which ``CLAUDE_OAUTH_TOKEN_*`` vars
exist?" scan was hand-rolled in four places over THREE different letter sets —
``BCDEF`` in ``test_usage_gate``, ``BCDEFG`` in ``_cross_account_evidence``, and
``ABCDEFG`` in both startup-probe sites.  ``CLAUDE_OAUTH_TOKEN_G`` is a real
fleet account in ``.env``, so the ``BCDEF`` copy skipped a live test that could
have run.  ``_oauth_accounts`` is now the single source; these tests pin its
contract, and the identity guards further down pin that each consumer actually
RESOLVES through it rather than holding a re-inlined copy.
"""

from __future__ import annotations

import pytest
from _oauth_accounts import (
    ALL_TOKEN_LETTERS,
    FLEET_TOKEN_LETTERS,
    available_tokens,
    first_available_token,
    token_var_names,
)


def _env(*letters: str, **extra: str) -> dict[str, str]:
    """Build a fake environ holding ``CLAUDE_OAUTH_TOKEN_<L>`` for each letter."""
    environ = {f'CLAUDE_OAUTH_TOKEN_{ch}': f'tok-{ch.lower()}' for ch in letters}
    environ.update(extra)
    return environ


class TestLetterSets:
    """The two named sets and, crucially, the DERIVATION between them."""

    def test_fleet_set_is_b_through_g(self):
        assert FLEET_TOKEN_LETTERS == ('B', 'C', 'D', 'E', 'F', 'G')

    def test_fleet_set_excludes_the_interactive_primary_account(self):
        """``A`` is the interactive/primary account, not a fleet worker.

        Pinned separately from the literal above so the reason survives: a
        future widening that swept ``A`` into the fleet set would aim
        capacity-spending measurements at the human's own account.
        """
        assert 'A' not in FLEET_TOKEN_LETTERS

    def test_all_set_is_derived_from_the_fleet_set(self):
        """Asserted as a DERIVATION, not a restated ``('A',...,'G')`` literal.

        This is the assertion that stops this task's own fix from re-creating
        the bug it fixes: adding an 8th fleet account must widen BOTH sets in
        one edit, or this goes red.  A restated literal would be a fourth copy
        wearing a single-home badge.
        """
        assert ALL_TOKEN_LETTERS == ('A', *FLEET_TOKEN_LETTERS)

    def test_all_set_reaches_the_interactive_primary_account(self):
        assert 'A' in ALL_TOKEN_LETTERS


class TestTokenVarNames:
    def test_fleet_letters_become_full_var_names_in_order(self):
        assert token_var_names(FLEET_TOKEN_LETTERS) == (
            'CLAUDE_OAUTH_TOKEN_B',
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_D',
            'CLAUDE_OAUTH_TOKEN_E',
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_G',
        )

    def test_all_letters_become_full_var_names_in_order(self):
        assert token_var_names(ALL_TOKEN_LETTERS) == (
            'CLAUDE_OAUTH_TOKEN_A',
            'CLAUDE_OAUTH_TOKEN_B',
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_D',
            'CLAUDE_OAUTH_TOKEN_E',
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_G',
        )

    def test_empty_letters_yield_no_names(self):
        assert token_var_names(()) == ()


class TestAvailableTokens:
    def test_returns_var_name_and_token_pairs(self):
        assert available_tokens(_env('C')) == [('CLAUDE_OAUTH_TOKEN_C', 'tok-c')]

    def test_scan_order_is_fixed_not_environ_insertion_order(self):
        """Insertion order is reversed; the result must still be B before F.

        Callers index this list positionally (``_AVAILABLE_TOKENS[0]``, and
        ``select_token_pair``'s default pair), so an order that tracked the
        environ would make WHICH account a live test spends depend on dict
        construction order.
        """
        environ = {}
        environ['CLAUDE_OAUTH_TOKEN_F'] = 'tok-f'
        environ['CLAUDE_OAUTH_TOKEN_B'] = 'tok-b'
        assert available_tokens(environ) == [
            ('CLAUDE_OAUTH_TOKEN_B', 'tok-b'),
            ('CLAUDE_OAUTH_TOKEN_F', 'tok-f'),
        ]

    def test_empty_string_token_is_not_available(self):
        """An unexpanded shell var / blank CI secret is absent, not present."""
        assert available_tokens({'CLAUDE_OAUTH_TOKEN_D': ''}) == []

    def test_no_tokens_yields_empty_list(self):
        assert available_tokens({}) == []

    def test_is_pure_reads_only_the_injected_mapping(self, monkeypatch):
        """Never ``os.environ`` — asserted with a mapping that DISAGREES with it.

        The real machine has A..G set in ``.env``; a mapping holding only C
        must yield only C, and a real env var absent from the mapping must not
        appear.
        """
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_B', 'real-b-from-os-environ')
        assert available_tokens(_env('C')) == [('CLAUDE_OAUTH_TOKEN_C', 'tok-c')]

    def test_fleet_scan_reaches_g(self):
        """The whole point of the task: ``G`` is a real account in ``.env``.

        The ``BCDEF`` copy in ``test_usage_gate`` could not see it, so a
        G-only-healthy machine skipped a live test that could have run.
        """
        assert available_tokens(_env('G')) == [('CLAUDE_OAUTH_TOKEN_G', 'tok-g')]

    def test_fleet_scan_does_not_see_the_interactive_primary_account(self):
        assert available_tokens(_env('A')) == []

    def test_all_scan_does_see_the_interactive_primary_account(self):
        assert available_tokens(_env('A'), ALL_TOKEN_LETTERS) == [
            ('CLAUDE_OAUTH_TOKEN_A', 'tok-a')
        ]

    def test_all_scan_puts_a_first(self):
        assert available_tokens(_env('A', 'C'), ALL_TOKEN_LETTERS) == [
            ('CLAUDE_OAUTH_TOKEN_A', 'tok-a'),
            ('CLAUDE_OAUTH_TOKEN_C', 'tok-c'),
        ]

    def test_default_letters_are_the_fleet_set(self):
        """The default is FLEET; an any-account caller must opt in explicitly."""
        environ = _env(*ALL_TOKEN_LETTERS)
        assert available_tokens(environ) == list(
            zip(
                token_var_names(FLEET_TOKEN_LETTERS),
                [f'tok-{ch.lower()}' for ch in FLEET_TOKEN_LETTERS],
                strict=True,
            )
        )

    def test_unrelated_env_entries_are_ignored(self):
        assert available_tokens(_env('C', PATH='/usr/bin', CLAUDE_OAUTH_TOKEN='x')) == [
            ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')
        ]


class TestFirstAvailableToken:
    def test_returns_the_first_hit_in_scan_order(self):
        assert first_available_token(_env('F', 'C')) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_returns_none_when_nothing_is_set(self):
        assert first_available_token({}) is None

    def test_returns_none_when_only_empty_tokens_are_set(self):
        assert first_available_token({'CLAUDE_OAUTH_TOKEN_B': ''}) is None

    def test_fleet_default_does_not_see_the_interactive_primary_account(self):
        assert first_available_token(_env('A')) is None

    def test_all_set_prefers_the_interactive_primary_account(self):
        """``A`` first in the ALL scan order, so an any-account probe finds it."""
        assert first_available_token(_env('A', 'G'), ALL_TOKEN_LETTERS) == (
            'CLAUDE_OAUTH_TOKEN_A',
            'tok-a',
        )

    def test_all_set_falls_through_to_g(self):
        assert first_available_token(_env('G'), ALL_TOKEN_LETTERS) == (
            'CLAUDE_OAUTH_TOKEN_G',
            'tok-g',
        )

    def test_agrees_with_available_tokens_first_entry(self):
        environ = _env('D', 'B', 'G')
        assert first_available_token(environ) == available_tokens(environ)[0]


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
