"""Guards for the defect-test rule and its splice into the fixing roles.

What this module guards (PRD ``plans/defect-tests-prd.md`` decision 17, leaf γ):
the shape of ``DEFECT_TEST_RULE``, the one helper that splices it, and the
system prompts of the roles that fix code.

Every assertion is a containment, count or equality check against a NAMED live
constant or a sentinel: no regex over prose (INV-10 guards-exercise-behaviour).
The standing rule and its rationale live in
``orchestrator/tests/_role_splice_contract.py``.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import (
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
    assert_opens_own_section,
)

from orchestrator.agents.roles import (
    DEFECT_TEST_RULE,
    ESCALATION_LADDER_CORE,
    FIXING_ROLE_PROMPT_BUILDERS,
    ROLES,
    splice_defect_test_rule,
)

_SENTINEL = '\n## DEFECT-TEST-RULE-SENTINEL-7047\n'

#: The roles that fix code (PRD decision 17). Editorial, not derived: ARCHITECT
#: also runs the replan and IMPLEMENTER the amender.
_FIXING_ROLES = frozenset({'architect', 'implementer', 'simple_task', 'debugger', 'steward'})

_RULE_CONTRACT = SpliceContract(
    constant_name='DEFECT_TEST_RULE',
    constant=DEFECT_TEST_RULE,
    roles=_FIXING_ROLES,
    role_set_name='_FIXING_ROLES',
    capability=lambda role: role.name in _FIXING_ROLES,
    capability_description='fixing code (PRD decision 17)',
)


class TestDefectTestRuleShape:
    def test_rule_is_nonempty(self):
        assert_nonempty(
            'DEFECT_TEST_RULE',
            DEFECT_TEST_RULE,
            remedy='Restore the decision-17 rule text in roles.py.',
        )

    def test_rule_opens_its_own_section(self):
        assert_opens_own_section(
            'DEFECT_TEST_RULE',
            DEFECT_TEST_RULE,
            remedy=(
                'Spliced just ahead of ESCALATION_LADDER_CORE, it would read as the '
                'tail of whichever block precedes it. Start it with a `## ` heading.'
            ),
        )

    def test_rule_is_brace_free(self):
        assert_brace_free(
            'DEFECT_TEST_RULE',
            DEFECT_TEST_RULE,
            remedy='Spell the marker and any placeholder without `{` or `}`.',
        )


class TestSpliceHelper:
    """The synthetic half: the splice mechanism, driven with a sentinel."""

    _PROMPT = 'intro\n' + ESCALATION_LADDER_CORE + 'tail\n'

    def test_the_rule_lands_once_immediately_before_the_escalation_section(self):
        result = splice_defect_test_rule(self._PROMPT, _SENTINEL)

        assert result.count(_SENTINEL) == 1
        assert _SENTINEL + ESCALATION_LADDER_CORE in result

    def test_nothing_else_in_the_prompt_changes(self):
        result = splice_defect_test_rule(self._PROMPT, _SENTINEL)

        assert result.replace(_SENTINEL, '', 1) == self._PROMPT

    @pytest.mark.parametrize(
        ('prompt', 'count'),
        [
            ('intro\nno escalation section here\n', 0),
            ('intro\n' + ESCALATION_LADDER_CORE + ESCALATION_LADDER_CORE, 2),
        ],
        ids=['absent', 'duplicated'],
    )
    def test_a_prompt_without_exactly_one_anchor_is_refused(self, prompt, count):
        with pytest.raises(ValueError) as excinfo:
            splice_defect_test_rule(prompt, _SENTINEL)

        message = str(excinfo.value)
        assert 'ESCALATION_LADDER_CORE' in message
        assert str(count) in message


class TestFixingRolePrompts:
    """The wired half: the five fixing roles' prompts, over the live constants."""

    def test_the_builder_mapping_covers_exactly_the_fixing_roles(self):
        assert frozenset(FIXING_ROLE_PROMPT_BUILDERS) == _FIXING_ROLES
        assert _FIXING_ROLES.issubset(ROLES)

    @pytest.mark.parametrize('name', sorted(_FIXING_ROLES))
    def test_the_builder_carries_only_the_rule_it_is_given(self, name):
        # Identity, not wording: with the constant substituted, a hand-pasted
        # copy of the rule inside a builder would survive and go red here.
        built = FIXING_ROLE_PROMPT_BUILDERS[name](_SENTINEL)

        assert built.count(_SENTINEL) == 1
        assert DEFECT_TEST_RULE not in built

    @pytest.mark.parametrize('name', sorted(_FIXING_ROLES))
    def test_the_dispatched_prompt_is_the_builders_render(self, name):
        # startswith, not ==: ARCHITECT and IMPLEMENTER get post-construction appends.
        assert ROLES[name].system_prompt.startswith(
            FIXING_ROLE_PROMPT_BUILDERS[name](DEFECT_TEST_RULE)
        ), (
            f'ROLES[{name!r}].system_prompt does not start with its '
            'FIXING_ROLE_PROMPT_BUILDERS render. Construct the role with '
            'system_prompt=<its builder>(DEFECT_TEST_RULE).'
        )

    @pytest.mark.parametrize('name', sorted(_FIXING_ROLES))
    def test_the_rule_lands_immediately_before_the_escalation_section(self, name):
        assert DEFECT_TEST_RULE + ESCALATION_LADDER_CORE in ROLES[name].system_prompt

    # assert_role_set_matches_capability is not called: _FIXING_ROLES is an
    # editorial set, so the check would compare it with itself.

    def test_every_fixing_role_carries_the_rule(self):
        _RULE_CONTRACT.assert_every_role_carries(
            remedy=(
                'Route the role\'s system prompt through splice_defect_test_rule '
                'in its _<role>_system_prompt builder.'
            ),
        )

    def test_the_rule_is_spliced_exactly_once(self):
        # absent_ok=True leaves a missing splice to fail only the containment
        # test above: one root cause, one failing test.
        _RULE_CONTRACT.assert_spliced_exactly_once(
            absent_ok=True,
            remedy='Keep splice_defect_test_rule as the only site that adds DEFECT_TEST_RULE.',
        )

    def test_no_other_role_carries_the_rule(self):
        _RULE_CONTRACT.assert_no_other_role_carries(
            remedy=(
                'Remove the splice. Only the roles that fix code carry the rule '
                '(PRD decision 17); reviewers, the judge and the merger do not.'
            ),
        )
