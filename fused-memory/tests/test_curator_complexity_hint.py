"""The curator's complexity hint, and the simple declaration applied on routing experiment 2's treated arm.

Contract: plans/routing-experiment-substrate-prd.md §"Experiment 2: the curator hint (ε)".

Driven through public seams only: jsonschema over the exported output
schemas, ``TaskCurator.curate_prepared`` with its invoker injected through
the constructor, the pure ``with_complexity_hint``, and the real submit stack
of ``_submit_stack.py`` with its scripted curator. No private name is
patched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
from _submit_stack import CuratorScript, Stack
from _submit_stack import curator_script as curator_script
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.routing_split import split_bucket

from fused_memory.middleware.candidate_key import compute_candidate_key
from fused_memory.middleware.curation_deferral import DEDUPE_SKIPPED_KEY
from fused_memory.middleware.curator_complexity_hint import with_complexity_hint
from fused_memory.middleware.curator_types import CandidateTask, CuratorDecision
from fused_memory.middleware.simple_task_files_guard import simple_task_files_error
from fused_memory.middleware.task_curator import (
    CURATOR_BATCH_OUTPUT_SCHEMA,
    CURATOR_OUTPUT_SCHEMA,
    TaskCurator,
)

PROJECT_ID = 'p'
CURATION_ROOT = '/x'

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
            [_bundle('Fix the wording of the guide')], PROJECT_ID, CURATION_ROOT,
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
            PROJECT_ID, CURATION_ROOT,
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
            PROJECT_ID, CURATION_ROOT,
        )

        assert (decision.action, decision.target_id, decision.complexity_hint) == (
            'combine', '7', 'simple',
        )

    @pytest.mark.parametrize(('answer', 'pool', 'vetoed'), [
        ({'action': 'drop', 'target_id': '99'}, (('7', 'pending'),), 'drop'),
        ({'action': 'drop', 'target_id': '7'}, (('7', 'unknown'),), 'drop'),
        (
            {
                'action': 'combine',
                'target_id': '7',
                'target_fingerprint': 't',
                'rewritten_task': {'title': 'Fix the guide wording', 'details': '- one bullet'},
                'complexity_hint': 'simple',
            },
            (('7', 'done'),),
            'combine',
        ),
    ], ids=['invalid-target', 'unknown-status-target', 'invalid-combine-target'])
    async def test_a_pool_vetoed_drop_or_combine_is_a_hintless_create_naming_the_veto(
        self, tmp_path, answer, pool, vetoed,
    ):
        curator = _curator(
            tmp_path,
            single=agent_result({**answer, 'justification': 'same concern'}),
            batch=tripwire('one bundle is a batch of one'),
        )

        [decision] = await curator.curate_prepared(
            [_bundle('Fix the wording of the guide', pool=pool)], PROJECT_ID, CURATION_ROOT,
        )

        assert (
            decision.action, decision.degraded, decision.vetoed_action, decision.complexity_hint,
        ) == ('create', False, vetoed, None)


@pytest.mark.parametrize('hint', ['complex', 'Simple', ''])
def test_a_decision_refuses_a_hint_other_than_simple_or_none(hint):
    with pytest.raises(ValueError, match=rf'complexity_hint.*{hint!r}'):
        CuratorDecision(action='create', complexity_hint=hint)


@pytest.mark.parametrize('fields', [
    {'action': 'drop', 'target_id': '7', 'vetoed_action': 'drop'},
    {'action': 'create', 'degraded': True, 'vetoed_action': 'combine'},
    {'action': 'create', 'vetoed_action': 'create'},
], ids=['not-a-create', 'degraded-create', 'neither-drop-nor-combine'])
def test_a_decision_refuses_a_vetoed_action_outside_a_pool_veto(fields):
    with pytest.raises(ValueError, match='vetoed_action'):
        CuratorDecision(**fields)


@pytest.mark.parametrize(('decision', 'usable'), [
    (CuratorDecision(action='create'), True),
    (CuratorDecision(action='create', vetoed_action='drop'), True),
    (CuratorDecision(action='route_deterministic'), True),
    (CuratorDecision(action='create', degraded=True), False),
    (CuratorDecision(action='drop', target_id='7'), False),
    (CuratorDecision(action='combine', target_id='7'), False),
], ids=['create', 'pool-vetoed-create', 'route-deterministic', 'degraded-create', 'drop', 'combine'])
def test_a_usable_create_verdict_is_a_create_no_fail_open_or_fall_through_reached(
    decision, usable,
):
    assert decision.is_usable_create_verdict() is usable


FILES = ['src/app/guide.py']
SIMPLE_CREATE = CuratorDecision(action='create', justification='new work', complexity_hint='simple')
HINT = 'x_curator_complexity_hint'
ARM_KEY = 'x_routing_exp2_arm_key'


def _title_on_arm(bucket: int, files: list[str]) -> str:
    """The first numbered guide-wording title whose routing-exp-2 arm is *bucket* over *files*.

    The salt, bucket count and treated bucket are spelled literally: they are
    the cross-leaf contract the experiment readout recomputes.
    """
    for n in range(1000):
        title = f'Fix the wording of the guide ({n})'
        key = compute_candidate_key(title, files)
        assert key is not None
        if split_bucket('routing-exp-2', key, 2) == bucket:
            return title
    raise AssertionError(f'no title lands on bucket {bucket}')


def _hinted(metadata: Any, title: str, decision: CuratorDecision | None = SIMPLE_CREATE) -> Any:
    """``with_complexity_hint``'s result, checked to be one the submit gate still accepts."""
    result = with_complexity_hint(metadata, title=title, decision=decision)
    if isinstance(result, Mapping):
        assert simple_task_files_error(result) is None
    return result


class TestTheHintIsAppliedOnTheTreatedArm:
    def test_the_treated_arm_declares_simple(self):
        title = _title_on_arm(1, FILES)

        assert _hinted({'files': FILES}, title) == {
            'files': FILES,
            HINT: 'simple',
            ARM_KEY: compute_candidate_key(title, FILES),
            'complexity': 'simple',
        }

    def test_the_untreated_arm_records_the_hint_and_the_arm_key_only(self):
        title = _title_on_arm(0, FILES)

        assert _hinted({'files': FILES}, title) == {
            'files': FILES,
            HINT: 'simple',
            ARM_KEY: compute_candidate_key(title, FILES),
        }

    @pytest.mark.parametrize('filed', ['standard', 'simple'])
    def test_the_filers_complexity_wins(self, filed):
        metadata = {'files': FILES, 'complexity': filed}

        assert _hinted(metadata, _title_on_arm(1, FILES)) == {**metadata, HINT: 'simple'}

    @pytest.mark.parametrize('title', ['', '   '], ids=['empty', 'blank'])
    def test_an_uncomputable_key_records_the_hint_only(self, title):
        assert _hinted({'files': FILES}, title) == {'files': FILES, HINT: 'simple'}

    def test_a_null_hint_is_recorded_and_declares_nothing(self):
        unsure = CuratorDecision(action='create', justification='new work')

        assert _hinted({'files': FILES}, _title_on_arm(1, FILES), unsure) == {
            'files': FILES, HINT: None,
        }

    @pytest.mark.parametrize(('metadata', 'key_files'), [
        ({}, []),
        ({'files': []}, []),
        ({'files': [5]}, []),
        ({'files_to_modify': FILES}, FILES),
    ], ids=['no-files', 'empty-files', 'non-str-files', 'legacy-files-to-modify'])
    def test_no_declarable_files_records_the_hint_only(self, metadata, key_files):
        assert _hinted(metadata, _title_on_arm(1, key_files)) == {**metadata, HINT: 'simple'}


class TestOnlyAHealthyCreateIsStamped:
    @pytest.mark.parametrize('decision', [
        None,
        CuratorDecision(
            action='create', justification='curator-failed', degraded=True,
            complexity_hint='simple',
        ),
        CuratorDecision(action='drop', target_id='7', justification='dup', complexity_hint='simple'),
        CuratorDecision(
            action='combine', target_id='7', justification='combine-refused',
            complexity_hint='simple',
        ),
        CuratorDecision(action='route_deterministic', justification='operational'),
        CuratorDecision(action='create', justification='invalid-target', vetoed_action='drop'),
        CuratorDecision(
            action='create', justification='invalid-combine-target', vetoed_action='combine',
        ),
    ], ids=[
        'no-decision', 'degraded-create', 'drop', 'combine', 'route-deterministic',
        'pool-vetoed-drop', 'pool-vetoed-combine',
    ])
    def test_any_other_decision_leaves_the_metadata_unchanged(self, decision):
        metadata = {'files': FILES}

        assert _hinted(metadata, _title_on_arm(1, FILES), decision) == {'files': FILES}


class TestWireForms:
    def test_a_json_string_is_decoded_and_stamped(self):
        title = _title_on_arm(1, FILES)

        assert _hinted(json.dumps({'files': FILES}), title) == {
            'files': FILES,
            HINT: 'simple',
            ARM_KEY: compute_candidate_key(title, FILES),
            'complexity': 'simple',
        }

    @pytest.mark.parametrize('wire', ['not json', json.dumps(FILES)], ids=['not-json', 'json-list'])
    def test_an_unreadable_value_is_returned_as_is(self, wire):
        assert _hinted(wire, _title_on_arm(1, FILES)) is wire

    def test_the_input_dict_is_not_mutated(self):
        metadata = {'files': FILES}

        _hinted(metadata, _title_on_arm(1, FILES))

        assert metadata == {'files': FILES}


@pytest.mark.parametrize('shape', [
    {},
    {'files': FILES},
    {'complexity': 'simple', 'files': FILES},
], ids=['empty', 'files', 'simple-with-files'])
def test_the_hint_key_does_not_change_what_the_submit_gate_accepts(shape):
    assert simple_task_files_error({**shape, HINT: 'simple'}) is None


_DESCRIPTION = 'One-line wording fix.'


@pytest.mark.asyncio
class TestTheSubmitBoundary:
    async def test_the_treated_arm_lands_simple_keyed_on_the_stored_candidate_key(
        self, stack: Stack, curator_script: CuratorScript,
    ):
        title = _title_on_arm(1, FILES)
        curator_script.decisions.append(SIMPLE_CREATE)

        task_id = await stack.file_normal(
            title=title, description=_DESCRIPTION, metadata={'files': FILES},
        )

        row = await stack.row(task_id)
        metadata = row['metadata']
        assert (metadata['complexity'], metadata[HINT], metadata[ARM_KEY]) == (
            'simple', 'simple', compute_candidate_key(title, FILES),
        )
        assert metadata[ARM_KEY] == row['candidate_key']

    async def test_the_untreated_arm_lands_with_the_hint_and_no_declaration(
        self, stack: Stack, curator_script: CuratorScript,
    ):
        title = _title_on_arm(0, FILES)
        curator_script.decisions.append(SIMPLE_CREATE)

        task_id = await stack.file_normal(
            title=title, description=_DESCRIPTION, metadata={'files': FILES},
        )

        metadata = await stack.metadata(task_id)
        assert (metadata[HINT], metadata[ARM_KEY]) == ('simple', compute_candidate_key(title, FILES))
        assert 'complexity' not in metadata

    async def test_the_filers_complexity_wins(self, stack: Stack, curator_script: CuratorScript):
        curator_script.decisions.append(SIMPLE_CREATE)

        task_id = await stack.file_normal(
            title=_title_on_arm(1, FILES), description=_DESCRIPTION,
            metadata={'files': FILES, 'complexity': 'standard'},
        )

        metadata = await stack.metadata(task_id)
        assert (metadata['complexity'], metadata[HINT]) == ('standard', 'simple')

    async def test_the_default_create_records_a_null_hint_and_declares_nothing(
        self, stack: Stack, curator_script: CuratorScript,
    ):
        task_id = await stack.file_normal(
            title=_title_on_arm(1, FILES), description=_DESCRIPTION, metadata={'files': FILES},
        )

        metadata = await stack.metadata(task_id)
        assert metadata[HINT] is None
        assert 'complexity' not in metadata
        assert ARM_KEY not in metadata

    async def test_a_pool_vetoed_create_is_judged_for_dedupe_but_carries_no_hint(
        self, stack: Stack, curator_script: CuratorScript,
    ):
        curator_script.decisions.append(
            CuratorDecision(action='create', justification='invalid-target', vetoed_action='drop'),
        )

        task_id = await stack.file_normal(
            title=_title_on_arm(1, FILES), description=_DESCRIPTION, metadata={'files': FILES},
        )

        metadata = await stack.metadata(task_id)
        assert (DEDUPE_SKIPPED_KEY in metadata, HINT in metadata) == (False, False)

    async def test_planning_mode_is_untouched(self, stack: Stack, curator_script: CuratorScript):
        curator_script.decisions.append(SIMPLE_CREATE)

        task_id = await stack.file_planned(
            title=_title_on_arm(1, FILES), description=_DESCRIPTION, metadata={'files': FILES},
        )

        metadata = await stack.metadata(task_id)
        assert HINT not in metadata
        assert 'complexity' not in metadata
        assert curator_script.handed == []
