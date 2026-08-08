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

import importlib

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


@pytest.fixture
def reload_module_under_env(monkeypatch):
    """Reload a consumer module under a controlled fake environ.

    Yields ``reload(module_name, **tokens)`` -> the reloaded module.  Every
    ``CLAUDE_OAUTH_TOKEN_*`` letter and the pair-override var are cleared first,
    so the machine's real ``.env`` (which has A..G all set) cannot leak in and
    make these assertions environment-dependent.

    Generalised from ``test_cross_account_evidence.py``'s
    ``reload_integration_module``, which is pinned to the pair-selection concern;
    one fixture here serves ``test_usage_gate``, ``test_cli_invoke_integration``
    and ``test_startup_completion_fixtures``.

    Reloading mutates the module object other collected items may hold, so on
    teardown the real environment is restored (``monkeypatch.undo()`` FIRST,
    since this fixture finalises BEFORE monkeypatch's own teardown) and every
    module touched is reloaded once more under it — no reload side effect
    escapes to other collected items.
    """
    reloaded: list[object] = []

    def _reload(module_name: str, **tokens: str):
        module = importlib.import_module(module_name)
        if module not in reloaded:
            reloaded.append(module)
        for ch in ALL_TOKEN_LETTERS:
            monkeypatch.delenv(f'CLAUDE_OAUTH_TOKEN_{ch}', raising=False)
        monkeypatch.delenv('CROSS_ACCOUNT_RESUME_TOKENS', raising=False)
        for name, value in tokens.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(module)

    try:
        yield _reload
    finally:
        monkeypatch.undo()
        for module in reloaded:
            importlib.reload(module)


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
        derived_from_fleet = ('A', *FLEET_TOKEN_LETTERS)
        assert derived_from_fleet == ALL_TOKEN_LETTERS

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


class TestUsageGateResolvesThroughSingleHome:
    """``test_usage_gate``'s live gate must see every fleet account — G included.

    THE headline defect of task 3700.  That module shipped its own ``BCDEF``
    scan, so ``CLAUDE_OAUTH_TOKEN_G`` — a real fleet account in ``.env`` — was
    invisible to it and ``_need_one_account`` skipped a live test that had a
    perfectly healthy account to run on.  Under the capacity scarcity task 3484
    measured on 2026-08-05 (5 of 6 accounts capped) that is the difference
    between a live probe running and a question going another round with no
    observations.

    Runtime-behaviour tests of module-level resolution, not structure or
    docstring tests — the same class of assertion as
    ``test_cross_account_evidence.py``'s ``TestIntegrationModuleUsesSelectTokenPair``.
    """

    def test_g_only_machine_selects_the_live_test(self, reload_module_under_env):
        """With ONLY G set the gate must SELECT, not skip.

        Measured on the pre-fix tree: ``_AVAILABLE_TOKENS`` was ``[]`` here and
        the gate skipped.  ``.args[0]`` is the ``skipif`` CONDITION, so
        ``False`` means "do not skip".
        """
        module = reload_module_under_env(
            'test_usage_gate', CLAUDE_OAUTH_TOKEN_G='tok-g'
        )
        assert module._AVAILABLE_TOKENS == [('CLAUDE_OAUTH_TOKEN_G', 'tok-g')]
        assert module._need_one_account.args[0] is False

    def test_interactive_primary_account_alone_still_skips(
        self, reload_module_under_env
    ):
        """The non-regression direction: widening to G must NOT widen to A.

        This call site spends real capacity (it spawns ``claude --print`` with
        ``--max-budget-usd 0.01``), so it is a FLEET consumer.  A scan that
        silently reached the interactive/primary account would aim live spend at
        the human's own quota.
        """
        module = reload_module_under_env(
            'test_usage_gate', CLAUDE_OAUTH_TOKEN_A='tok-a'
        )
        assert module._AVAILABLE_TOKENS == []
        assert module._need_one_account.args[0] is True

    def test_scan_order_is_b_before_g(self, reload_module_under_env):
        """``_AVAILABLE_TOKENS[0]`` picks the account a live test spends."""
        module = reload_module_under_env(
            'test_usage_gate',
            CLAUDE_OAUTH_TOKEN_G='tok-g',
            CLAUDE_OAUTH_TOKEN_B='tok-b',
        )
        assert module._AVAILABLE_TOKENS == [
            ('CLAUDE_OAUTH_TOKEN_B', 'tok-b'),
            ('CLAUDE_OAUTH_TOKEN_G', 'tok-g'),
        ]

    def test_no_accounts_skips(self, reload_module_under_env):
        module = reload_module_under_env('test_usage_gate')
        assert module._AVAILABLE_TOKENS == []
        assert module._need_one_account.args[0] is True
