"""Contract test for ``_role_splice_contract.py`` — the shared splice-contract shape.

Follows ``test_conftest_helpers.py``'s precedent of contract-testing test
infrastructure. ``_role_splice_contract.py`` is not production code; it is the
deduplicated body behind the per-role splice assertions in the ``test_roles_*``
anchor modules.

WHY THE FIRING CASES MATTER, and why this file exists at all. The consumer
modules name the same motivating risk: a prompt refactor silently drops a
mandated block and CI stays green. Once they all delegate their assertions to
one helper, that hazard concentrates. An assertion helper that degrades into a
no-op — an ``assert`` accidentally softened to a truthy expression, an offender
loop that never appends, a message that swallows the call site's remedy — would
leave every consumer test green while asserting nothing at all. So every
assertion here is pinned TWICE: once that it PASSES on a conforming input, and
once that it FIRES (raises ``AssertionError``, with the offenders named) on a
violating one. A helper that only ever passes is indistinguishable from a
helper that does nothing.

Everything under test here is driven by SYNTHETIC fixtures — string literals
and synthetic ``AgentRole`` mappings — never by a real prompt constant from
``roles.py``, except ``GREP_LOOKAROUND_GUIDANCE``, imported only as the opaque
landmark the preamble-tail order check anchors on and never asserted on. This
file tests the HELPER; whether the real prompts satisfy the contract is the
consumer modules' job, and duplicating that here would recreate the very clone
this task is closing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest
from _role_splice_contract import (
    BASH_CAPABLE_UNPINNED_ROLES,
    MARKDOWN_HEADING,
    PreambleTailBlock,
    PreambleTailContractTests,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
    bash_capable_unpinned,
    bash_capable_unpinned_contract,
)
from shared.prompt_artifact import PromptSpec

from orchestrator.agents.roles import GREP_LOOKAROUND_GUIDANCE, ROLES, AgentRole

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
    asserted here — that is the consumer modules' job, and duplicating it here
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


_SYNTHETIC_PROMPT_SPEC = PromptSpec(prompt_id='p', contract='c', baseline_heuristics='h')


@pytest.mark.parametrize(
    ('role', 'expected'),
    [
        (AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash']), True),
        (AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=['Bash(git:*)']), False),
        (AgentRole(name='alpha', system_prompt=_SPLICE, allowed_tools=[]), False),
        (
            AgentRole(
                name='alpha',
                system_prompt=_SPLICE,
                allowed_tools=['Bash'],
                prompt_spec=_SYNTHETIC_PROMPT_SPEC,
            ),
            False,
        ),
    ],
    ids=['literal-bash', 'git-only-bash', 'no-tools', 'prompt-spec-backed'],
)
def test_bash_capable_unpinned_predicate(role: AgentRole, expected: bool) -> None:
    """Literal prompt AND the exact `'Bash'` grant; `'Bash(git:*)'` does not qualify."""
    assert bash_capable_unpinned(role) is expected


def _bash_capable_mapping(**extra: AgentRole) -> dict[str, AgentRole]:
    """Every shared-set name as a literal Bash role, plus both excluded shapes.

    `judge` (git-only `Bash`) and a PromptSpec-backed `Bash` holder are both
    present so each exclusion arm of the predicate meets a role to exclude.
    """
    mapping = {
        name: AgentRole(name=name, system_prompt=_SPLICE, allowed_tools=['Bash'])
        for name in BASH_CAPABLE_UNPINNED_ROLES
    }
    mapping['judge'] = AgentRole(
        name='judge', system_prompt=_SPLICE, allowed_tools=['Bash(git:*)']
    )
    mapping['reviewer_synthetic'] = AgentRole(
        name='reviewer_synthetic',
        system_prompt=_SPLICE,
        allowed_tools=['Bash'],
        prompt_spec=_SYNTHETIC_PROMPT_SPEC,
    )
    mapping.update(extra)
    return mapping


def test_bash_capable_unpinned_contract_binds_the_shared_set() -> None:
    """The factory binds the shared set, the shared predicate and the caller's constant."""
    contract = bash_capable_unpinned_contract(
        'SPLICE_UNIT', _SPLICE, all_roles=_bash_capable_mapping()
    )

    assert contract.roles is BASH_CAPABLE_UNPINNED_ROLES
    assert contract.constant == _SPLICE
    assert contract.constant_name == 'SPLICE_UNIT'
    assert contract.assert_role_set_matches_capability(remedy=_REMEDY) is None
    # Identity check only, as in `test_all_roles_defaults_to_the_real_roles_mapping`.
    assert bash_capable_unpinned_contract('SPLICE_UNIT', _SPLICE).all_roles is ROLES


def test_bash_capable_unpinned_contract_fires_when_a_role_gains_bash() -> None:
    """A new literal `Bash` role is reported as gained, naming the shared set."""
    gamma = AgentRole(name='gamma', system_prompt=_SPLICE, allowed_tools=['Bash'])
    contract = bash_capable_unpinned_contract(
        'SPLICE_UNIT', _SPLICE, all_roles=_bash_capable_mapping(gamma=gamma)
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_role_set_matches_capability(remedy=_REMEDY)

    message = str(excinfo.value)
    assert "gained=['gamma']" in message
    assert 'BASH_CAPABLE_UNPINNED_ROLES' in message
    assert _REMEDY in message


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


# Synthetic prompts shaped like a real role prompt: an identity paragraph with
# no heading, then sections each introduced by `MARKDOWN_HEADING`. `_SPLICE`
# itself opens with that heading (as `BACKGROUND_WAIT_GUIDANCE` and
# `TOOL_CALL_REJECTION_GUIDANCE` both do), so `find(_SPLICE) ==
# find(MARKDOWN_HEADING)` is exactly the "spliced up front" condition.
_IDENTITY = 'You are a synthetic agent. You do synthetic work.'
_SECTION = f'{MARKDOWN_HEADING}A Later Section\n\nSome other rule.\n'
_PREDECESSOR = f'{MARKDOWN_HEADING}Predecessor Block\n\nThe block spliced first.\n'


def test_placement_up_front_passes_when_the_splice_owns_the_first_heading() -> None:
    """`follows=None`: the constant's own heading IS the prompt's first `##`."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    assert contract.assert_placement(remedy=_REMEDY) is None


def test_placement_up_front_fires_when_another_section_precedes_the_splice() -> None:
    """A `##` section ahead of the splice fires, reporting offset and first_heading."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_SECTION}{_SPLICE}'), frozenset({'alpha'})
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_placement(remedy=_REMEDY)

    message = str(excinfo.value)
    assert 'alpha' in message
    assert 'offset' in message
    assert 'first_heading' in message
    assert _REMEDY in message


def test_placement_up_front_fires_when_the_splice_lands_past_the_char_budget() -> None:
    """A pathologically long heading-free preamble fires even though it owns the first `##`.

    The budget is the secondary, deliberately loose bound: the real invariant is
    the structural one, and this only catches a preamble that is technically
    heading-free but far too long.
    """
    preamble = 'A very long identity paragraph. ' * 60  # ~1920 chars, no heading
    contract = _contract(
        _roles(alpha=f'{preamble}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    # Structurally correct — it still owns the first `##` heading.
    assert contract.assert_placement(remedy=_REMEDY) is None

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_placement(char_budget=1500, remedy=_REMEDY)
    assert 'alpha' in str(excinfo.value)


def test_placement_up_front_passes_within_the_char_budget() -> None:
    """A short preamble passes the budget check."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    assert contract.assert_placement(char_budget=1500, remedy=_REMEDY) is None


@pytest.mark.parametrize('follows', [None, _PREDECESSOR])
def test_placement_records_an_absent_splice_as_an_offender_in_both_modes(
    follows: str | None,
) -> None:
    """ABSENT is RECORDED, never skipped — in up-front AND follows mode.

    This is what stops the placement check passing vacuously on a role that
    dropped the splice entirely: with no index to compare, "no offender found"
    would otherwise read as "correctly placed".
    """
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SECTION}'), frozenset({'alpha'})
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_placement(follows=follows, follows_name='PREDECESSOR', remedy=_REMEDY)

    message = str(excinfo.value)
    assert 'alpha' in message
    assert 'ABSENT' in message


def test_placement_follows_passes_when_the_splice_abuts_its_predecessor() -> None:
    """`follows=`: the constant lands IMMEDIATELY after the predecessor, no gap."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    assert (
        contract.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
        is None
    )


def test_placement_follows_fires_when_text_is_inserted_before_the_splice() -> None:
    """A gap between predecessor and constant fires, reporting the expected index."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SECTION}{_SPLICE}'), frozenset({'alpha'})
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )

    message = str(excinfo.value)
    assert 'alpha' in message
    assert 'offset' in message
    assert 'PREDECESSOR' in message
    assert _REMEDY in message


def test_placement_follows_falls_back_to_the_up_front_rule_without_a_predecessor() -> None:
    """A role carrying no predecessor block falls back to the first-`##` rule.

    This is `judge` in production: it carries `TOOL_CALL_REJECTION_GUIDANCE` but
    no `BACKGROUND_WAIT_GUIDANCE`, so "immediately after the wait block" has no
    referent and the up-front landmark governs instead. Derived at runtime from
    whether the predecessor is present, never hardcoded to a role name.
    """
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )
    assert (
        contract.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
        is None
    )

    buried = _contract(
        _roles(alpha=f'{_IDENTITY}{_SECTION}{_SPLICE}'), frozenset({'alpha'})
    )
    with pytest.raises(AssertionError) as excinfo:
        buried.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
    assert 'first_heading' in str(excinfo.value)


@pytest.mark.parametrize('follows', [None, _PREDECESSOR])
def test_placement_char_budget_none_means_no_budget_check(follows: str | None) -> None:
    """`char_budget=None` (the default) disables the budget check in BOTH modes.

    The rejection file passes no budget at all: its splice sits after the wait
    block, which is itself budget-checked by the wait file, so a second budget
    would be a redundant hand-maintained number.
    """
    preamble = 'A very long identity paragraph. ' * 60
    contract = _contract(
        _roles(alpha=f'{preamble}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    assert (
        contract.assert_placement(follows=follows, follows_name='PREDECESSOR', remedy=_REMEDY)
        is None
    )


def test_placement_char_budget_applies_in_follows_mode_too() -> None:
    """`char_budget` is enforced ALONGSIDE `follows`, not only in the up-front arm.

    The regression this pins: the budget comparison once lived inside the
    up-front fallback branch, so a role that DID carry the predecessor took the
    `follows` branch and never met the budget at all — while the failure message
    still claimed "within N chars". A caller passing both got a silently
    vacuous bound, which is the exact failure class this module exists to catch.

    Both roles below are structurally PERFECT in `follows` mode — the splice
    abuts its predecessor with no gap — so the budget is the only thing that can
    distinguish them, and the fire half cannot pass by accident on the
    placement rule.
    """
    preamble = 'A very long identity paragraph. ' * 60  # ~1920 chars, no heading
    far = _contract(
        _roles(alpha=f'{preamble}{_PREDECESSOR}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    # Structurally correct on the placement rule alone: no budget, no offender.
    assert (
        far.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
        is None
    )

    with pytest.raises(AssertionError) as excinfo:
        far.assert_placement(
            follows=_PREDECESSOR,
            follows_name='PREDECESSOR',
            char_budget=1500,
            remedy=_REMEDY,
        )
    message = str(excinfo.value)
    assert 'alpha' in message
    assert 'over_budget' in message
    assert _REMEDY in message

    near = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )
    assert (
        near.assert_placement(
            follows=_PREDECESSOR,
            follows_name='PREDECESSOR',
            char_budget=1500,
            remedy=_REMEDY,
        )
        is None
    )


def test_lands_after_passes_when_the_splice_abuts_its_predecessor() -> None:
    """Order holds trivially when the constant starts exactly where `follows` ends."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SPLICE}{_SECTION}'), frozenset({'alpha'})
    )

    assert (
        contract.assert_lands_after(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
        is None
    )


def test_lands_after_passes_with_another_block_between_it_and_its_predecessor() -> None:
    """The discriminating case: adjacency fails, order holds.

    A block appended at the tail of a shared chain needs exactly this property,
    since sibling blocks may merge in between in any order.
    """
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_PREDECESSOR}{_SECTION}{_SPLICE}'), frozenset({'alpha'})
    )

    with pytest.raises(AssertionError):
        contract.assert_placement(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
    assert (
        contract.assert_lands_after(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )
        is None
    )


def test_lands_after_fires_when_the_splice_precedes_its_predecessor() -> None:
    """A constant ahead of `follows` fires, reporting the earliest allowed offset."""
    contract = _contract(
        _roles(alpha=f'{_IDENTITY}{_SPLICE}{_PREDECESSOR}{_SECTION}'), frozenset({'alpha'})
    )

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_lands_after(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )

    message = str(excinfo.value)
    assert 'alpha' in message
    assert 'earliest_allowed' in message
    assert 'PREDECESSOR' in message
    assert _REMEDY in message


@pytest.mark.parametrize(
    ('prompt', 'expected_key'),
    [
        (f'{_IDENTITY}{_PREDECESSOR}{_SECTION}', 'offset'),
        (f'{_IDENTITY}{_SPLICE}{_SECTION}', 'follows_offset'),
    ],
    ids=['splice-absent', 'predecessor-absent'],
)
def test_lands_after_records_a_missing_block_as_an_offender(
    prompt: str, expected_key: str
) -> None:
    """ABSENT is recorded, never skipped, so the check cannot pass vacuously.

    The second case also pins that there is NO up-front fallback:
    `assert_placement(follows=...)` would accept it, because there the splice
    owns the first heading.
    """
    contract = _contract(_roles(alpha=prompt), frozenset({'alpha'}))

    with pytest.raises(AssertionError) as excinfo:
        contract.assert_lands_after(
            follows=_PREDECESSOR, follows_name='PREDECESSOR', remedy=_REMEDY
        )

    # The key/value pair, not a bare 'ABSENT': the rule text names 'ABSENT' too.
    message = str(excinfo.value)
    assert 'alpha' in message
    assert f"'{expected_key}': 'ABSENT'" in message


# One distinct sentinel per block-owned prose field, so each fire case below
# proves the mixin renders THAT field, not merely some remedy.
_RESTORE_SENTINEL = 'RESTORE-SENTINEL'
_BRACE_SENTINEL = 'BRACE-SENTINEL'
_EXCLUDED_SENTINEL = 'EXCLUDED-ROLES-SENTINEL'
_CARRIER_SENTINEL = 'CARRIER-SENTINEL'
_ORDER_SENTINEL = 'ORDER-SENTINEL'

_TAIL_PROMPT = f'{_IDENTITY}{GREP_LOOKAROUND_GUIDANCE}{_SPLICE}'


def _tail_mapping(**overrides: str) -> dict[str, AgentRole]:
    """A conforming preamble-tail mapping, with ``name=prompt`` overrides.

    Every shared-set role carries the splice after the grep landmark, and
    `judge` holds git-only `Bash` and no splice. An override names an existing
    role to replace its prompt, or a NEW name to add a literal `Bash` role.
    """
    mapping = {
        name: AgentRole(name=name, system_prompt=_TAIL_PROMPT, allowed_tools=['Bash'])
        for name in BASH_CAPABLE_UNPINNED_ROLES
    }
    mapping['judge'] = AgentRole(
        name='judge', system_prompt=_IDENTITY, allowed_tools=['Bash(git:*)']
    )
    for name, prompt in overrides.items():
        base = mapping.get(name, AgentRole(name=name, system_prompt='', allowed_tools=['Bash']))
        mapping[name] = dataclasses.replace(base, system_prompt=prompt)
    return mapping


def _tail_block(**field_overrides: object) -> PreambleTailBlock:
    """A `PreambleTailBlock` over `_tail_mapping()`, with every prose field a sentinel."""
    fields: dict[str, object] = {
        'constant_name': 'SPLICE_UNIT',
        'constant': _SPLICE,
        'restore_remedy': _RESTORE_SENTINEL,
        'brace_remedy': _BRACE_SENTINEL,
        'excluded_roles_remedy': _EXCLUDED_SENTINEL,
        'carrier_note': _CARRIER_SENTINEL,
        'order_note': _ORDER_SENTINEL,
        'all_roles': _tail_mapping(),
    }
    fields.update(field_overrides)
    return PreambleTailBlock(**fields)  # type: ignore[arg-type]


def _probe(block: PreambleTailBlock) -> PreambleTailContractTests:
    """A mixin instance bound to ``block``; its `_` name keeps pytest from collecting it."""
    return type('_Probe', (PreambleTailContractTests,), {'block': block})()


class TestPreambleTailContractOnAConformingSplice(PreambleTailContractTests):
    """The PASS case, run exactly as a consumer runs it: every inherited method."""

    block = _tail_block()


@pytest.mark.parametrize(
    ('method', 'block', 'expected'),
    [
        ('test_guidance_is_nonempty', _tail_block(constant='\n'), [_RESTORE_SENTINEL]),
        ('test_guidance_is_brace_free', _tail_block(constant=f'{_SPLICE}{{'), [_BRACE_SENTINEL]),
        (
            'test_guidance_opens_its_own_section',
            _tail_block(constant='no heading\n'),
            ['SPLICE_UNIT'],
        ),
        (
            'test_role_set_matches_bash_capability',
            _tail_block(all_roles=_tail_mapping(gamma=_TAIL_PROMPT)),
            ["gained=['gamma']", _CARRIER_SENTINEL],
        ),
        (
            'test_every_bash_capable_role_carries_guidance',
            _tail_block(all_roles=_tail_mapping(merger=f'{_IDENTITY}{GREP_LOOKAROUND_GUIDANCE}')),
            ["['merger']"],
        ),
        (
            'test_no_other_role_carries_guidance',
            _tail_block(all_roles=_tail_mapping(judge=f'{_IDENTITY}{_SPLICE}')),
            ["['judge']", _EXCLUDED_SENTINEL],
        ),
        (
            'test_guidance_appears_exactly_once_per_role',
            _tail_block(all_roles=_tail_mapping(merger=f'{_TAIL_PROMPT}{_SPLICE}')),
            ["'merger': 2"],
        ),
        (
            'test_guidance_lands_after_the_grep_block',
            _tail_block(
                all_roles=_tail_mapping(merger=f'{_IDENTITY}{_SPLICE}{GREP_LOOKAROUND_GUIDANCE}')
            ),
            ['earliest_allowed', _ORDER_SENTINEL],
        ),
    ],
    ids=[
        'emptied',
        'brace',
        'unheaded',
        'role-gained-bash',
        'carrier-lost-splice',
        'excluded-role-spliced',
        'duplicate-splice',
        'ahead-of-grep-block',
    ],
)
def test_preamble_tail_contract_fires_with_the_block_remedy(
    method: str, block: PreambleTailBlock, expected: list[str]
) -> None:
    """Each inherited invariant FIRES on its violation and renders the block's own prose."""
    with pytest.raises(AssertionError) as excinfo:
        getattr(_probe(block), method)()

    message = str(excinfo.value)
    for fragment in expected:
        assert fragment in message


def test_tail_exactly_once_skips_an_absent_splice() -> None:
    """``absent_ok=True``: a missing splice fails the containment test only.

    One root cause, one failing test.
    """
    probe = _probe(
        _tail_block(all_roles=_tail_mapping(merger=f'{_IDENTITY}{GREP_LOOKAROUND_GUIDANCE}'))
    )

    assert probe.test_guidance_appears_exactly_once_per_role() is None
    with pytest.raises(AssertionError):
        probe.test_every_bash_capable_role_carries_guidance()
