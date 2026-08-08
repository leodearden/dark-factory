"""Shared test kit: WHICH OAuth accounts exist in the environment (task 3700).

Canonical home for the ``CLAUDE_OAUTH_TOKEN_*`` scan that four places had each
hand-rolled, over THREE different letter sets:

- ``_cross_account_evidence.TOKEN_LETTERS`` — ``BCDEFG`` (the fleet set).
- ``test_usage_gate._TOKEN_ENV_VARS`` — ``BCDEF``.
- ``test_startup_completion_fixtures._OAUTH_TOKEN_PRESENT`` — ``ABCDEFG``.
- ``startup_completion_probe._oauth_token`` — ``ABCDEFG``.

They had DRIFTED three ways, and the drift is not cosmetic: every one of these
scans feeds a live ``pytest.skip`` gate, so a scan blind to an account skips a
live test that could have run.  Measured 2026-08-08 with only
``CLAUDE_OAUTH_TOKEN_G`` set: the probe found G, the integration module found G,
and ``test_usage_gate`` found nothing and skipped — on a machine that had a
perfectly healthy account to run on.  Under the capacity scarcity task 3484
measured on 2026-08-05 (5 of 6 accounts capped) that is the difference between
a measurement happening and a question going another round with 0 valid runs.

TWO named sets, deliberately, because this is two concepts and collapsing them
would be a silent behaviour change in two directions at once:

- ``FLEET_TOKEN_LETTERS`` — accounts a test may SPEND.  Excludes ``A``.
- ``ALL_TOKEN_LETTERS`` — "is there ANY account at all?".  Includes ``A``.

``ALL_TOKEN_LETTERS`` is DERIVED from ``FLEET_TOKEN_LETTERS`` rather than
restated as an ``('A', ..., 'G')`` literal, so an 8th fleet account widens both
sets in one edit and they cannot drift apart again.  A restated literal would be
a fourth copy wearing a single-home badge.

Consumers import by bare module name (``from _oauth_accounts import ...``) --
``conftest.py`` prepends ``shared/tests`` to ``sys.path``, same convention as
``_capacity_skip.py``; ``startup_completion_probe.py`` carries its own duplicate
bootstrap so it also resolves when run standalone.  The leading underscore keeps
pytest from collecting this module as a test file; its contract tests live in
``test_oauth_accounts.py``.

Deliberately import-light — no ``shared.cli_invoke``, no ``_capacity_skip`` — so
the standalone ``startup_completion_probe.py`` script can import it without
dragging in the package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: Accounts a test may SPEND: the fleet workers, scanned in this order.
#:
#: ``A`` stays out deliberately — it is the interactive/primary account, not a
#: fleet worker, and a capacity-spending measurement aimed at it burns the
#: human's own quota.
#:
#: ``G`` is IN, and that is the widening this module exists to make uniform:
#: ``CLAUDE_OAUTH_TOKEN_G`` is a real account in ``.env`` that the old ``BCDEF``
#: copy in ``test_usage_gate`` could not reach at all, needlessly shrinking the
#: pool its live gate draws from.
FLEET_TOKEN_LETTERS: tuple[str, ...] = ('B', 'C', 'D', 'E', 'F', 'G')

#: "Is there ANY account at all?" — the fleet set plus the interactive/primary
#: account ``A``, scanned first.
#:
#: WIDER than the fleet set on purpose.  Its callers (the startup-completion
#: probe and its fixture gate) do not spend fleet capacity to answer a
#: measurement question; they only need SOME account so a machine with none
#: records a legible skip instead of a spurious failure.  ``A`` is legitimate
#: for that, and excluding it would make a single-account machine report a
#: failure it cannot act on.
#:
#: DERIVED, never restated: see the module docstring.
ALL_TOKEN_LETTERS: tuple[str, ...] = ('A', *FLEET_TOKEN_LETTERS)


def token_var_names(letters: Iterable[str]) -> tuple[str, ...]:
    """``('B', 'C')`` -> ``('CLAUDE_OAUTH_TOKEN_B', 'CLAUDE_OAUTH_TOKEN_C')``.

    The one place the ``CLAUDE_OAUTH_TOKEN_`` prefix is spelled, so a diagnostic
    that wants to say WHICH vars were scanned sources its names from the same
    place the scan itself does.
    """
    return tuple(f'CLAUDE_OAUTH_TOKEN_{ch}' for ch in letters)


def available_tokens(
    environ: Mapping[str, str],
    letters: tuple[str, ...] = FLEET_TOKEN_LETTERS,
) -> list[tuple[str, str]]:
    """``[(var_name, token), ...]`` for every set token var, in scan order.

    Order is fixed by *letters* and is NOT the environ's own iteration order:
    callers index this list positionally (``_AVAILABLE_TOKENS[0]``,
    ``select_token_pair``'s default pair), so WHICH account a live test spends
    must not depend on how the environ mapping was built.

    A var present with an empty value counts as ABSENT — an unexpanded shell
    var or a blank CI secret is not an account.

    **The default *letters* is the FLEET set (``B``..``G``), not every account.**
    A caller that means "is there ANY account at all, including the
    interactive/primary one?" must pass ``ALL_TOKEN_LETTERS`` explicitly.  The
    default is the narrow, capacity-spending-safe one on purpose: a call site
    that forgets to choose gets the set that cannot silently aim a live run at
    the human's own account.

    Pure: reads only the injected *environ*, never ``os.environ``.
    """
    return [
        (var, environ[var])
        for var in token_var_names(letters)
        if environ.get(var)
    ]


def first_available_token(
    environ: Mapping[str, str],
    letters: tuple[str, ...] = FLEET_TOKEN_LETTERS,
) -> tuple[str, str] | None:
    """``(var_name, token)`` for the first set token var, or ``None`` if none is.

    Same policy as ``available_tokens`` — including the FLEET default and the
    empty-value-means-absent rule — in the calling convention a caller that
    needs exactly one account wants.  One policy, two calling conventions
    (the ``_capacity_skip.py`` precedent); this must remain a pure delegation so
    the two can never disagree about what "available" means.

    Pure: reads only the injected *environ*, never ``os.environ``.
    """
    for var, token in available_tokens(environ, letters):
        return (var, token)
    return None
