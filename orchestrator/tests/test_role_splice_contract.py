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


def test_every_role_carries_passes_when_the_whole_set_carries_the_constant() -> None:
    """Every name in `contract.roles` has the constant in its system_prompt."""
    contract = _contract(
        _roles(alpha=f'ident{_SPLICE}', beta=f'ident{_SPLICE}', gamma='no splice here'),
        frozenset({'alpha', 'beta'}),
    )

    assert contract.assert_every_role_carries(remedy=_REMEDY) is None


def test_every_role_carries_fires_and_names_the_role_that_lost_the_constant() -> None:
    """A covered role missing the constant is reported by name, as a sorted list."""
    contract = _contract(
        _roles(alpha=f'ident{_SPLICE}', beta='ident, but the splice was dropped'),
        frozenset({'alpha', 'beta'}),
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_every_role_carries(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "['beta']" in message
    assert 'SPLICE_UNIT' in message
    assert _REMEDY in message


def test_no_other_role_carries_passes_when_the_splice_stays_inside_the_set() -> None:
    """No role outside `contract.roles` carries the constant."""
    contract = _contract(
        _roles(alpha=f'ident{_SPLICE}', gamma='an excluded role, correctly bare'),
        frozenset({'alpha'}),
    )

    assert contract.assert_no_other_role_carries(remedy=_REMEDY) is None


def test_no_other_role_carries_fires_and_names_the_errant_splice() -> None:
    """A role OUTSIDE the set carrying the constant is reported by name."""
    contract = _contract(
        _roles(alpha=f'ident{_SPLICE}', gamma=f'excluded, but spliced anyway{_SPLICE}'),
        frozenset({'alpha'}),
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_no_other_role_carries(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "['gamma']" in message
    assert 'SPLICE_UNIT' in message
    assert _REMEDY in message


def test_containment_assertions_honour_the_half_scoped_constant_override() -> None:
    """`constant=`/`constant_name=` re-scope both containment checks to one HALF.

    This is the code path `test_artifact_pinned_role_does_not_carry_missing_'
    `required_parameter_shape` needs: a hand-splice of just ONE half into an
    excluded role is invisible to the whole-constant negative check (the composed
    unit is not present), so the half is checked separately through the same body
    rather than through a fourth copy of the offender-list idiom.
    """
    half = '\n### Half\n\nOne composed half.\n'
    composed = f'{_SPLICE}{half}'
    all_roles = _roles(
        alpha=f'ident{composed}',
        gamma=f'excluded, carrying ONLY the half{half}',
    )
    contract = _contract(all_roles, frozenset({'alpha'}), constant=composed)

    # The whole-constant negative check does NOT see the errant half-splice...
    assert contract.assert_no_other_role_carries(remedy=_REMEDY) is None

    # ...but the half-scoped one does.
    with pytest.raises(AssertionError) as excinfo:
        contract.assert_no_other_role_carries(
            constant=half, constant_name='HALF_CONSTANT', remedy=_REMEDY
        )
    message = str(excinfo.value)
    assert "['gamma']" in message
    assert 'HALF_CONSTANT' in message

    # The positive direction re-scopes the same way.
    assert (
        contract.assert_every_role_carries(
            constant=half, constant_name='HALF_CONSTANT', remedy=_REMEDY
        )
        is None
    )
    bare = _contract(
        _roles(alpha=f'ident{_SPLICE}'), frozenset({'alpha'}), constant=composed
    )
    with pytest.raises(AssertionError) as excinfo:
        bare.assert_every_role_carries(
            constant=half, constant_name='HALF_CONSTANT', remedy=_REMEDY
        )
    assert "['alpha']" in str(excinfo.value)


def test_an_emptied_constant_makes_every_containment_check_pass_vacuously() -> None:
    """The vacuous-pass hazard, pinned in executable form.

    The empty string is a substring of every string, so a contract whose
    `constant` has been emptied satisfies `assert_every_role_carries` for a role
    whose prompt carries nothing at all. That is not a defect in this helper —
    it is `str.__contains__`, and no containment check can detect it.

    It IS why the guards around containment are not redundant with it: each
    consumer keeps a separate `assert_nonempty` on its constant, and
    `assert_composes` checks each half's non-emptiness before checking that the
    half is contained. Removing either on the grounds that "the containment test
    already covers it" would reopen exactly this hole — which is the
    silent-removal-during-a-prompt-refactor regression both consumer modules name
    as their motivating risk.
    """
    contract = _contract(_roles(alpha='no splice anywhere in this prompt'), frozenset({'alpha'}))

    assert (
        contract.assert_every_role_carries(constant='', constant_name='EMPTIED', remedy=_REMEDY)
        is None
    )
    # ...and the guard that DOES catch it:
    with pytest.raises(AssertionError):
        assert_nonempty('EMPTIED', '', remedy=_REMEDY)


# BOTH count-zero behaviours are kept, deliberately, because the three consumer
# tests disagree and each argues its choice in its own docstring.
# `test_guidance_appears_exactly_once_per_role` SKIPS a count of 0 so a role that
# has not yet received the splice fails exactly ONE test for that one root cause
# (the containment test) instead of two.
# `test_combined_guidance_appears_exactly_once_per_role` and
# `test_missing_required_parameter_shape_appears_exactly_once_per_role` treat 0
# as an offender — for the latter, because the composition is separately pinned,
# so a 0 count for the half proves the whole composed splice is missing from that
# role. Unifying them would silently change one consumer's coverage under cover
# of a "pure refactor", which is the drift these anchor tests exist to prevent.
@pytest.mark.parametrize('absent_ok', [False, True])
def test_spliced_exactly_once_passes_for_a_count_of_one(absent_ok: bool) -> None:
    """Exactly one copy passes under either `absent_ok` setting."""
    contract = _contract(_roles(alpha=f'ident{_SPLICE}tail'), frozenset({'alpha'}))

    assert contract.assert_spliced_exactly_once(absent_ok=absent_ok, remedy=_REMEDY) is None


@pytest.mark.parametrize('absent_ok', [False, True])
def test_spliced_exactly_once_fires_for_a_duplicate_splice(absent_ok: bool) -> None:
    """A count of 2 fires under EITHER setting — `absent_ok` only governs 0.

    This is the stale-tail case: a leftover `+ CONSTANT` surviving beside a new
    up-front splice silently doubles the block in every session of that role.
    """
    contract = _contract(_roles(alpha=f'ident{_SPLICE}middle{_SPLICE}tail'), frozenset({'alpha'}))

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_spliced_exactly_once(absent_ok=absent_ok, remedy=_REMEDY)

    message = str(excinfo.value)
    assert "'alpha': 2" in message
    assert 'SPLICE_UNIT' in message
    assert _REMEDY in message


def test_spliced_exactly_once_skips_an_absent_splice_when_absent_ok() -> None:
    """`absent_ok=True`: a count of 0 is SKIPPED, not flagged."""
    contract = _contract(_roles(alpha='ident, no splice at all'), frozenset({'alpha'}))

    assert contract.assert_spliced_exactly_once(absent_ok=True, remedy=_REMEDY) is None


def test_spliced_exactly_once_flags_an_absent_splice_by_default() -> None:
    """`absent_ok=False` (the default): a count of 0 IS an offender, reported as 0."""
    contract = _contract(_roles(alpha='ident, no splice at all'), frozenset({'alpha'}))

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_spliced_exactly_once(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "'alpha': 0" in message
    assert _REMEDY in message


def test_spliced_exactly_once_honours_the_half_scoped_constant_override() -> None:
    """The count re-scopes to a named half, which is how a stale bare tail is caught.

    The wait file counts `BACKGROUND_TASK_WARNING` alongside its splice unit and
    the rejection file counts `MISSING_REQUIRED_PARAMETER_REJECTION`: a leftover
    bare `+ HALF` tail pushes the HALF's count to 2 while the composed unit's
    count stays at 1, so the whole-unit count alone cannot see it.
    """
    half = '\n### Half\n\nOne composed half.\n'
    composed = f'{_SPLICE}{half}'
    contract = _contract(
        _roles(alpha=f'ident{composed}tail, plus a stale bare half{half}'),
        frozenset({'alpha'}),
        constant=composed,
    )

    # The composed unit is spliced exactly once...
    assert contract.assert_spliced_exactly_once(remedy=_REMEDY) is None

    # ...but the half survives twice.
    with pytest.raises(AssertionError) as excinfo:
        contract.assert_spliced_exactly_once(
            constant=half, constant_name='HALF_CONSTANT', remedy=_REMEDY
        )
    message = str(excinfo.value)
    assert "'alpha': 2" in message
    assert 'HALF_CONSTANT' in message


def test_composes_passes_when_every_half_is_non_empty_and_contained() -> None:
    """The composition holds: each named half is non-empty AND inside the unit."""
    first, second = '\n### First\n\nrule one\n', '\n### Second\n\nrule two\n'
    contract = _contract(
        _roles(alpha='ident'), frozenset(), constant=f'{first}{second}'
    )

    assert (
        contract.assert_composes(
            [('FIRST_HALF', first), ('SECOND_HALF', second)], remedy=_REMEDY
        )
        is None
    )


def test_composes_fires_and_names_a_half_that_was_spliced_away() -> None:
    """A half no longer inside the composed unit is reported by NAME."""
    first, second = '\n### First\n\nrule one\n', '\n### Second\n\nrule two\n'
    contract = _contract(_roles(alpha='ident'), frozenset(), constant=first)

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_composes(
            [('FIRST_HALF', first), ('SECOND_HALF', second)], remedy=_REMEDY
        )

    message = str(excinfo.value)
    assert 'SECOND_HALF' in message
    assert _REMEDY in message


def test_composes_fires_for_an_emptied_half_even_though_it_is_contained() -> None:
    """The vacuous-containment guard, and the reason it is not redundant.

    An emptied half is a substring of every string, so the containment half of
    this assertion holds trivially for it. Without the per-half non-emptiness
    check, a refactor could empty one half and every containment, count and
    placement test in both consumer modules would stay green while the half's
    content vanished from every spliced role prompt. This is
    `test_an_emptied_constant_makes_every_containment_check_pass_vacuously`'s
    observation made actionable.
    """
    first = '\n### First\n\nrule one\n'
    contract = _contract(_roles(alpha='ident'), frozenset(), constant=first)

    # Containment alone would pass: '' is in everything.
    assert '' in contract.constant

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_composes([('FIRST_HALF', first), ('EMPTIED_HALF', '')], remedy=_REMEDY)

    assert 'EMPTIED_HALF' in str(excinfo.value)


@pytest.mark.parametrize('half_count', [1, 3])
def test_composes_is_not_hardcoded_to_two_halves(half_count: int) -> None:
    """Nothing in the composition check assumes exactly two halves.

    A splice unit that grows a third census shape (as
    `TOOL_CALL_REJECTION_GUIDANCE` did in task 4578) must not need a new helper.
    """
    halves = [(f'HALF_{i}', f'\n### Half {i}\n\nrule {i}\n') for i in range(half_count)]
    contract = _contract(
        _roles(alpha='ident'), frozenset(), constant=''.join(value for _, value in halves)
    )

    assert contract.assert_composes(halves, remedy=_REMEDY) is None
