"""The curator's complexity hint, and the simple declaration applied on routing experiment 2's treated arm.

Contract: plans/routing-experiment-substrate-prd.md §"Experiment 2: the curator hint (ε)".

Driven through public seams only: jsonschema over the exported output
schemas, and ``TaskCurator.curate_prepared`` with its invoker injected
through the constructor. No private name is patched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest
from _curator_helpers import (
    agent_result,
    escalator_double,
    invoke_by_role,
    make_config,
    pool_with_ids,
    prepared,
    tripwire,
)

from fused_memory.middleware.curator_types import CandidateTask, CuratorDecision
from fused_memory.middleware.task_curator import (
    CURATOR_BATCH_OUTPUT_SCHEMA,
    CURATOR_OUTPUT_SCHEMA,
    TaskCurator,
)

PROJECT_ID = 'p'
PROJECT_ROOT = '/x'

_ABSENT = object()

_BATCH_ITEM_SCHEMA: dict[str, Any] = CURATOR_BATCH_OUTPUT_SCHEMA['properties']['decisions']['items']

_SCHEMAS = {
    'single': (CURATOR_OUTPUT_SCHEMA, {'action': 'create', 'justification': 'new work'}),
    'batch-item': (
        _BATCH_ITEM_SCHEMA,
        {'candidate_index': 0, 'action': 'create', 'justification': 'new work'},
    ),
}


def _with_hint(decision: dict[str, Any], hint: object) -> dict[str, Any]:
    return dict(decision) if hint is _ABSENT else {**decision, 'complexity_hint': hint}


def _curator(tmp_path: Path, *, single: Any, batch: Any) -> TaskCurator:
    return TaskCurator(
        config=make_config(), taskmaster=None, cwd=tmp_path, escalator=escalator_double(),
        invoke=invoke_by_role(single=single, batch=batch),
    )


def _bundle(title: str, pool: tuple[tuple[str, str], ...] = ()):
    return prepared(
        title,
        pool=pool_with_ids(*pool),
        candidate=CandidateTask(title=title, description=f'{title} body'),
    )


class TestTheSchemasCarryTheHint:
    @pytest.mark.parametrize('spec', sorted(_SCHEMAS))
    @pytest.mark.parametrize('hint', ['simple', None, _ABSENT], ids=['simple', 'null', 'absent'])
    def test_a_create_decision_with_a_valid_or_missing_hint_validates(self, spec, hint):
        schema, decision = _SCHEMAS[spec]

        jsonschema.validate(_with_hint(decision, hint), schema)

    @pytest.mark.parametrize('spec', sorted(_SCHEMAS))
    def test_a_hint_other_than_simple_or_null_is_refused(self, spec):
        schema, decision = _SCHEMAS[spec]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_with_hint(decision, 'complex'), schema)


@pytest.mark.asyncio
class TestTheCuratorParsesTheHint:
    @pytest.mark.parametrize(('raw', 'parsed'), [
        ('simple', 'simple'),
        (' Simple ', 'simple'),
        (None, None),
        ('complex', None),
        (_ABSENT, None),
    ], ids=['simple', 'padded-mixed-case', 'null', 'not-simple', 'absent'])
    async def test_the_single_spec_carries_the_hint(self, tmp_path, raw, parsed):
        answer = _with_hint({'action': 'create', 'justification': 'new work'}, raw)
        curator = _curator(
            tmp_path,
            single=agent_result(answer),
            batch=tripwire('one bundle is a batch of one'),
        )

        [decision] = await curator.curate_prepared(
            [_bundle('Fix the wording of the guide')], PROJECT_ID, PROJECT_ROOT,
        )

        assert (decision.action, decision.complexity_hint) == ('create', parsed)

    async def test_the_batch_spec_carries_each_items_hint(self, tmp_path):
        curator = _curator(
            tmp_path,
            single=tripwire('two bundles use the batch spec'),
            batch=agent_result({'decisions': [
                {'candidate_index': 0, 'action': 'create', 'justification': 'a',
                 'complexity_hint': 'simple'},
                {'candidate_index': 1, 'action': 'create', 'justification': 'b',
                 'complexity_hint': None},
            ]}),
        )

        decisions = await curator.curate_prepared(
            [_bundle('Fix the wording of the guide'), _bundle('Redesign the scheduler')],
            PROJECT_ID, PROJECT_ROOT,
        )

        assert [(d.action, d.complexity_hint) for d in decisions] == [
            ('create', 'simple'), ('create', None),
        ]

    async def test_a_combine_keeps_its_hint(self, tmp_path):
        curator = _curator(
            tmp_path,
            single=agent_result({
                'action': 'combine',
                'target_id': '7',
                'target_fingerprint': 't',
                'justification': 'same concern',
                'rewritten_task': {'title': 'Fix the guide wording', 'details': '- one bullet'},
                'complexity_hint': 'simple',
            }),
            batch=tripwire('one bundle is a batch of one'),
        )

        [decision] = await curator.curate_prepared(
            [_bundle('Fix the wording of the guide', pool=(('7', 'pending'),))],
            PROJECT_ID, PROJECT_ROOT,
        )

        assert (decision.action, decision.target_id, decision.complexity_hint) == (
            'combine', '7', 'simple',
        )


@pytest.mark.parametrize('hint', ['complex', 'Simple', ''])
def test_a_decision_refuses_a_hint_other_than_simple_or_none(hint):
    with pytest.raises(ValueError, match=rf'complexity_hint.*{hint!r}'):
        CuratorDecision(action='create', complexity_hint=hint)
