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
from _role_splice_contract import assert_brace_free, assert_nonempty, assert_opens_own_section

from orchestrator.agents.roles import (
    DEFECT_TEST_RULE,
    ESCALATION_LADDER_CORE,
    splice_defect_test_rule,
)

_SENTINEL = '\n## DEFECT-TEST-RULE-SENTINEL-7047\n'


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
