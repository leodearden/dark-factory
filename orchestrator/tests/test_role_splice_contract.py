"""Contract test for ``_role_splice_contract.py`` — the shared splice-contract shape.

Follows ``test_conftest_helpers.py``'s precedent of contract-testing test
infrastructure. ``_role_splice_contract.py`` is not production code; it is the
deduplicated body behind the per-role splice assertions in
``test_roles_wait_pattern.py`` (task 3607, ``BACKGROUND_WAIT_GUIDANCE``) and
``test_roles_tool_call_rejection.py`` (tasks 4273/4578,
``TOOL_CALL_REJECTION_GUIDANCE``).

WHY THE FIRING CASES MATTER, and why this file exists at all. Both consumer
modules name the same motivating risk in their docstrings: a prompt refactor
silently drops a mandated block and CI stays green. Once both files delegate
their assertions to one helper, that hazard concentrates. An assertion helper
that degrades into a no-op — an ``assert`` accidentally softened to a truthy
expression, an offender loop that never appends, a message that swallows the
call site's remedy — would leave all 24 consumer tests green while asserting
nothing at all. So every assertion here is pinned TWICE: once that it PASSES
on a conforming input, and once that it FIRES (raises ``AssertionError``, with
the offenders named) on a violating one. A helper that only ever passes is
indistinguishable from a helper that does nothing.

Everything under test here is driven by SYNTHETIC fixtures — string literals
and synthetic ``AgentRole`` mappings — never by a real prompt constant from
``roles.py``. This file tests the HELPER; whether the real prompts satisfy the
contract is the two consumer files' job, and duplicating that here would
recreate the very clone this task is closing.
"""

from __future__ import annotations

import pytest

from _role_splice_contract import assert_brace_free, assert_nonempty

# Synthetic remedy prose. The helper must render the caller's remedy into every
# failure message: the consumers' remediation sentences ARE the diagnostic value
# of these assertions, so a helper that generated its own generic message would
# trade duplicated lines for a real loss in legibility.
_REMEDY = 'RESTORE-THE-BLOCK-OR-THE-PROMPT-IS-WRONG'


def test_assert_nonempty_passes_for_a_non_empty_value() -> None:
    """A value with content returns None — the helper is not unconditionally loud."""
    assert assert_nonempty('SOME_CONSTANT', 'a prompt block', remedy=_REMEDY) is None


@pytest.mark.parametrize('value', ['', '   \n', '\t  \n\n '])
def test_assert_nonempty_fires_for_an_empty_value(value: str) -> None:
    """Empty and whitespace-only both fire, and the message carries name + remedy.

    Whitespace-only is included deliberately: a constant emptied to ``'\\n'``
    contributes nothing to a prompt but is truthy, so a bare ``assert value``
    would pass. The consumers assert ``value.strip()`` for exactly this reason.
    """
    with pytest.raises(AssertionError) as excinfo:
        assert_nonempty('SOME_CONSTANT', value, remedy=_REMEDY)

    message = str(excinfo.value)
    assert 'SOME_CONSTANT' in message
    assert _REMEDY in message


def test_assert_brace_free_passes_for_a_brace_free_value() -> None:
    """A value with no literal brace returns None."""
    assert assert_brace_free('SOME_CONSTANT', 'a prompt block', remedy=_REMEDY) is None


@pytest.mark.parametrize(
    'value',
    ['has a { brace', 'has a } brace', 'has {both} braces'],
)
def test_assert_brace_free_fires_for_a_literal_brace(value: str) -> None:
    """Open, close, and both fire — the message carries name + remedy.

    All three arms matter: a check written as ``'{' not in value`` alone would
    pass the close-brace case, and ``str.format`` raises on an unbalanced brace
    of either kind.
    """
    with pytest.raises(AssertionError) as excinfo:
        assert_brace_free('SOME_CONSTANT', value, remedy=_REMEDY)

    message = str(excinfo.value)
    assert 'SOME_CONSTANT' in message
    assert _REMEDY in message
