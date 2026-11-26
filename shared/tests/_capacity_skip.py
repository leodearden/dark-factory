"""Shared test kit: the capacity-failure skip guard.

A calling-convention shim over ``shared.cap_markers``, which is the CANONICAL
home for the marker list and the real-CLI corpus.  This module owns no policy
of its own — only the two entry points the ``pytest.skip`` call sites want.

Originally this module held the list itself, having been extracted from
byte-for-byte copies in ``test_cli_invoke_integration.py`` (as
``_CAPACITY_FAILURE_MARKERS`` / ``_looks_like_capacity_failure(AgentResult)``)
and ``test_usage_gate.py`` (same list, ``_looks_like_capacity_failure(str)``).
It has since moved one level further out: a THIRD copy turned up in the
legibility census's usage-headroom probe (``scripts/legibility/census.py``),
which could not import this module at all — ``shared/tests/`` is reachable
only through this suite's ``conftest.py`` sys.path insert, so the census
runtime had no way to share it and grew its own four-marker list instead.
That copy missed ``"You've hit your weekly limit · resets Aug 5, 11am``
entirely (task 3645).  The list therefore now lives in ``src/``, where both a
test suite and a runtime can reach it, and the identity of these re-exports is
pinned in ``test_cap_markers.py``.

Two entry points remain because the call sites genuinely differ in what they
hold: an ``AgentResult`` in the cli_invoke suite, a pre-combined
``f'{stdout}\\n{stderr}'`` string in the usage-gate probe guard.  One policy,
two calling conventions.

This module matches CAPACITY ONLY — deliberately not ``cap_markers``'
capacity-or-auth union.  These are live ``pytest.skip`` sites, and an auth
failure must stay a loud red test rather than becoming a quiet skip.

Consumers import by bare module name (``from _capacity_skip import ...``) --
``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention as
``_usage_gate_test_helpers.py``).  The leading underscore keeps pytest from
collecting this module as a test file; its contract tests live in
``test_capacity_skip.py``.
"""

from __future__ import annotations

from shared.cap_markers import CAPACITY_BANNER_MARKERS as CAPACITY_FAILURE_MARKERS
from shared.cap_markers import (
    REAL_CLI_CAP_HIT_MESSAGES,
    REAL_CLI_CAP_MESSAGES,
    REAL_CLI_NEAR_CAP_MESSAGES,
    looks_like_capacity_banner,
)
from shared.cli_invoke import AgentResult

__all__ = [
    'CAPACITY_FAILURE_MARKERS',
    'REAL_CLI_CAP_HIT_MESSAGES',
    'REAL_CLI_CAP_MESSAGES',
    'REAL_CLI_NEAR_CAP_MESSAGES',
    'looks_like_capacity_failure',
    'result_looks_like_capacity_failure',
]


def looks_like_capacity_failure(text: str) -> bool:
    """Return True when *text* looks like a Claude CLI capacity / quota failure.

    **Must remain a pure delegation.**  Every decision here is
    ``shared.cap_markers.looks_like_capacity_banner``'s decision; this function
    exists only to adapt its ``str | None`` return to the bool the
    ``pytest.skip`` sites want.  The rationale for the marker list's shape —
    why it is a purpose-built loose substring list rather than an import of
    ``invocation_outcome``'s strict prefix-AND-confirm policy, and why the
    conservative bias runs toward failing loudly — lives in that module's
    docstring, at the same level as the list it explains.
    """
    return looks_like_capacity_banner(text) is not None


def result_looks_like_capacity_failure(result: AgentResult) -> bool:
    """``looks_like_capacity_failure`` over an ``AgentResult``.

    Inspects both ``result.output`` and ``result.stderr`` — a cap message can
    arrive on either stream depending on how the CLI failed.

    **Must remain a pure delegation.** This is a calling convention, not a
    second policy: everything it decides is decided by
    ``looks_like_capacity_failure``, and through it by ``cap_markers``. Growing
    independent matching logic here re-creates the
    two-implementations-that-can-disagree bug this module was extracted to
    kill, one level down.
    """
    return looks_like_capacity_failure(f'{result.output}\n{result.stderr}')
