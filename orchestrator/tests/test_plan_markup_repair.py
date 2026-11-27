"""orchestrator.mcp.plan_markup_repair, driven by a SYNTHETIC field table.

The table names no plan-tools tool, so these pass only if the mechanism takes
everything it knows about the plan's writers from the table it is handed.
Specimens are built from ``_markup_helpers`` so this file spells no raw
envelope literal.
"""
from __future__ import annotations

import copy
from types import MappingProxyType

import _plan_markup_fixtures
from _markup_helpers import INVOKE_CLOSER, closer, param_opener
from _plan_markup_fixtures import DECISION_PROSE, corrupt_plan, on_disk
from shared.toolcall_markup import detect

from orchestrator.mcp.plan_markup_repair import (
    PlanField,
    carries_markup,
    read_plan_repaired,
    repair_plan_fields,
)

#: pytest finds a fixture by its name in the requesting module's namespace.
isolate_the_refusal_memo = _plan_markup_fixtures.isolate_the_refusal_memo

RATIONALE_PROSE = 'A rationale the harness folded into the decision before it.'

TABLE = (
    PlanField(
        collection='design_decisions',
        field='decision',
        tool='record_choice',
        schema_params=('decision', 'rationale'),
        target_keys=MappingProxyType({'decision': 'decision', 'rationale': 'rationale'}),
        also_written_by=(),
    ),
)


def _absorbed_decision() -> dict:
    """A decision that absorbed its rationale sibling, over an empty rationale."""
    absorbed = (
        DECISION_PROSE + closer('decision') + '\n' + param_opener('rationale') + RATIONALE_PROSE
    )
    return corrupt_plan(design_decisions=[{'decision': absorbed, 'rationale': ''}])


def _poisoned_analysis() -> dict:
    """A trailing-residue ``analysis``, a field the table does not list."""
    return corrupt_plan(
        analysis='Clean analysis prose.' + closer('analysis') + '\n' + INVOKE_CLOSER + '\n',
    )


class TestRepairPlanFields:
    def test_an_absorbed_sibling_is_recovered_under_the_injected_tool(self):
        plan = _absorbed_decision()
        before = copy.deepcopy(plan)

        repaired, facts = repair_plan_fields(plan, TABLE)

        item = repaired['design_decisions'][0]
        assert item['decision'] == DECISION_PROSE
        assert detect(item['decision']) is None
        assert item['rationale'] == RATIONALE_PROSE
        assert [(f['outcome'], f['tool'], f['field']) for f in facts] == [
            ('repaired', 'record_choice', 'decision'),
        ]
        assert facts[0]['recovered_params'] == ['rationale']
        assert plan == before

    def test_a_refused_value_is_reported_under_the_injected_tool(self):
        quoted = 'The harness ends the call with ' + INVOKE_CLOSER + ' and then keeps going.'
        plan = corrupt_plan(design_decisions=[{'decision': quoted, 'rationale': ''}])

        repaired, facts = repair_plan_fields(plan, TABLE)

        assert repaired['design_decisions'][0]['decision'] == quoted
        assert [(f['outcome'], f['tool'], f['field']) for f in facts] == [
            ('unrepairable', 'record_choice', 'decision'),
        ]

    def test_a_field_the_table_does_not_list_is_left_alone(self):
        plan = _poisoned_analysis()
        analysis = plan['analysis']

        repaired, facts = repair_plan_fields(plan, TABLE)

        assert facts == []
        assert repaired['analysis'] == analysis


class TestCarriesMarkup:
    def test_only_a_listed_field_counts(self):
        assert carries_markup(_absorbed_decision(), TABLE) is True
        assert carries_markup(_poisoned_analysis(), TABLE) is False


class TestReadPlanRepaired:
    def test_the_repair_is_persisted_and_reported(self, plan_artifacts):
        plan_artifacts.write_plan(_absorbed_decision())

        _, facts = read_plan_repaired(plan_artifacts, TABLE)

        assert [(f['outcome'], f['tool']) for f in facts] == [('repaired', 'record_choice')]
        item = on_disk(plan_artifacts)['design_decisions'][0]
        assert item['decision'] == DECISION_PROSE
        assert item['rationale'] == RATIONALE_PROSE
