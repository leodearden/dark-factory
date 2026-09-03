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

import dataclasses
from collections.abc import Mapping

import pytest
from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.roles import ROLES, AgentRole

# Synthetic remedy prose. The helper must render the caller's remedy into every
# failure message: the consumers' remediation sentences ARE the diagnostic value
# of these assertions, so a helper that generated its own generic message would
# trade duplicated lines for a real loss in legibility.
_REMEDY = 'RESTORE-THE-BLOCK-OR-THE-PROMPT-IS-WRONG'

# A synthetic splice unit, shaped like the real ones: it OPENS with its own
# `\n## ` heading, so `find(_SPLICE) == find(MARKDOWN_HEADING)` is exactly the
# "spliced up front" condition the placement assertion checks.
_SPLICE = f'{MARKDOWN_HEADING}Synthetic Splice Unit\n\nA mandated rule.\n'


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


def _roles(**prompts: str) -> dict[str, AgentRole]:
    """A synthetic role mapping from ``name=prompt`` kwargs.

    ``AgentRole(name=..., system_prompt=...)`` constructs cleanly with the
    default empty ``allowed_tools`` — its ``__post_init__`` MCP-family
    capability assertion passes on an empty tool list — so a mapping built here
    exercises the helper without touching a single real prompt.
    """
    return {name: AgentRole(name=name, system_prompt=body) for name, body in prompts.items()}


# The two consumers' capability predicates, reproduced verbatim. Both are driven
# through the SAME contract below to pin that `capability` is genuinely a
# parameter and the helper hardcodes neither consumer's question.
def _has_bash(role: AgentRole) -> bool:
    return 'Bash' in role.allowed_tools


def _has_literal_prompt(role: AgentRole) -> bool:
    return role.prompt_spec is None


def _contract(
    all_roles: Mapping[str, AgentRole], roles: frozenset[str], **overrides: object
) -> SpliceContract:
    """A `SpliceContract` over synthetic roles, with the boilerplate defaulted."""
    fields: dict[str, object] = {
        'constant_name': 'SPLICE_UNIT',
        'constant': _SPLICE,
        'roles': roles,
        'role_set_name': '_SYNTHETIC_ROLE_SET',
        'capability': _has_bash,
        'capability_description': 'the synthetic capability',
        'all_roles': all_roles,
    }
    fields.update(overrides)
    return SpliceContract(**fields)  # type: ignore[arg-type]


def test_splice_contract_is_frozen_but_replaceable() -> None:
    """The contract is a frozen dataclass: `replace` works, assignment raises.

    Frozen so a consumer's module-level `_CONTRACT` cannot be mutated by one
    test and silently change what a later test in the same module asserts.
    """
    contract = _contract(_roles(alpha=_SPLICE), frozenset({'alpha'}))

    assert dataclasses.replace(contract, constant_name='OTHER').constant_name == 'OTHER'
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.constant_name = 'OTHER'  # type: ignore[misc]


def test_all_roles_defaults_to_the_real_roles_mapping() -> None:
    """Omitting `all_roles` binds the real `ROLES`, so consumers stay one-liners.

    Identity check ONLY. Nothing about the real roles' prompt CONTENTS is
    asserted here — that is the two consumer files' job, and duplicating it here
    would recreate the clone this task closes.
    """
    contract = SpliceContract(
        constant_name='SPLICE_UNIT',
        constant=_SPLICE,
        roles=frozenset(),
        role_set_name='_SYNTHETIC_ROLE_SET',
        capability=_has_bash,
        capability_description='the synthetic capability',
    )

    assert contract.all_roles is ROLES


def test_role_set_matching_capability_passes_when_the_sets_agree() -> None:
    """The hardcoded role set equals the set derived by applying the predicate."""
    all_roles = {
        'alpha': AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash']),
        'beta': AgentRole(name='beta', system_prompt='plain', allowed_tools=['Bash(git:*)']),
    }
    contract = _contract(all_roles, frozenset({'alpha'}))

    assert contract.assert_role_set_matches_capability(remedy=_REMEDY) is None


def test_role_set_matching_capability_fires_when_a_role_gains_the_capability() -> None:
    """A role OUTSIDE the set that satisfies the predicate is reported as gained."""
    all_roles = {
        'alpha': AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash']),
        'beta': AgentRole(name='beta', system_prompt='plain', allowed_tools=['Bash']),
    }
    contract = _contract(all_roles, frozenset({'alpha'}))

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_role_set_matches_capability(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "gained=['beta']" in message
    assert '_SYNTHETIC_ROLE_SET' in message
    assert _REMEDY in message


def test_role_set_matching_capability_fires_when_a_role_loses_the_capability() -> None:
    """A role INSIDE the set that stops satisfying the predicate is reported as lost."""
    all_roles = {
        'alpha': AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash']),
        'beta': AgentRole(name='beta', system_prompt='plain', allowed_tools=['Bash(git:*)']),
    }
    contract = _contract(all_roles, frozenset({'alpha', 'beta'}))

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_role_set_matches_capability(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "lost=['beta']" in message
    assert _REMEDY in message


def test_the_capability_predicate_is_genuinely_a_parameter() -> None:
    """A DIFFERENT predicate drives the same tripwire, over the same role mapping.

    The two consumers ask different capability questions — `'Bash' in
    role.allowed_tools` ("can this role launch a long-running command, so the
    wait guidance is not dead weight?") versus `role.prompt_spec is None` ("is
    this role's prompt literal text a splice can reliably reach?") — and both
    are correct for their own constant. Unifying them would either splice the
    wait block into a role where it is dead weight or drop the rejection
    guidance from a role that needs it, so the predicate is injected. This test
    pins that injection: with `_has_literal_prompt` the same mapping derives a
    DIFFERENT set than it does with `_has_bash` above, and the helper honours it.
    """
    all_roles = {
        'alpha': AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash']),
        'beta': AgentRole(name='beta', system_prompt='plain', allowed_tools=['Bash(git:*)']),
    }

    # Every synthetic role has `prompt_spec is None`, so this predicate derives
    # BOTH names where `_has_bash` derived only `alpha`.
    both = _contract(all_roles, frozenset({'alpha', 'beta'}), capability=_has_literal_prompt)
    assert both.assert_role_set_matches_capability(remedy=_REMEDY) is None

    only_alpha = _contract(all_roles, frozenset({'alpha'}), capability=_has_literal_prompt)
    with pytest.raises(AssertionError) as excinfo:
        only_alpha.assert_role_set_matches_capability(remedy=_REMEDY)
    assert "gained=['beta']" in str(excinfo.value)
