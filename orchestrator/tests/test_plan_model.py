"""The typed Plan model in ``orchestrator/src/orchestrator/artifacts.py`` (S9 C1/D1/D2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import (
    PLAN_BASE_COMMIT_KEY,
    PLAN_TASK_TEXT_FP_KEY,
    Plan,
    PlanConfirmation,
    PlanLifecycle,
    PlanState,
    plan_base_commit,
    plan_task_text_fp,
)

_FIXTURES = Path(__file__).parent / 'fixtures'

_STEP = {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending'}

LIVE_SHAPED_PLAN: dict[str, Any] = {
    'task_id': '9170',
    'title': 'A live-shaped plan',
    'analysis': 'Premises checked.',
    'files': ['orchestrator/src/orchestrator/artifacts.py'],
    'prerequisites': [
        {'id': 'pre-1', 'description': 'Seed.', 'status': 'done', 'commit': 'a' * 40},
    ],
    'steps': [
        {
            'id': 'step-1',
            'type': 'test',
            'description': '[COMMITTED bbbbbbbbbbbb] RED: pin it.',
            'status': 'done',
            'commit': 'b' * 40,
            'deviations': [{'kind': 'other', 'observed': 'x', 'expected': 'y'}],
        },
        {'id': 'step-2', 'type': 'impl', 'description': 'GREEN.', 'status': 'pending', 'commit': None},
    ],
    'design_decisions': [{'decision': 'd', 'rationale': 'r'}],
    'reuse': [{'what': 'w', 'where': 'x', 'how': 'h'}],
    'dropped_files': [],
    '_markup_rejections': {'count': 1, 'by_tool': {'add_plan_step': 1}, 'events': [], 'note': 'n'},
    '_rescoped_at': '2026-11-10T00:00:00+00:00',
    '_rescoped_by': 'architect',
    '_rescoped_note': 'dropped a done step',
    '_descoped_steps': [{'id': 'step-0', 'commit': 'c' * 40, 'disposition': 'orphan'}],
    '_schema_version': 1,
    '_finalized_at': '2026-11-11T11:28:13.011975+00:00',
    '_revalidated_at': '2026-11-11T11:28:13.011975+00:00',
    '_session_id': '9170-f4c58365',
    '_created_at': '2026-11-11T11:31:45.208202+00:00',
    PLAN_TASK_TEXT_FP_KEY: 'tfp:7335c99f3942fee1d098cef6367dd78c',
    PLAN_BASE_COMMIT_KEY: '1f240e6da21230dba301b4a2c2b9ebefdca60699',
}

VALID_PLAN_WITH_STEPS: dict[str, Any] = {
    'task_id': 'task-1',
    'title': 'Test Task',
    'steps': [dict(_STEP)],
}

CREATE_PLAN_SHELL: dict[str, Any] = {
    'task_id': 'task-1',
    'title': 'T',
    'analysis': 'A',
    'files': ['a.py'],
    'prerequisites': [],
    'steps': [],
    'design_decisions': [],
    'reuse': [],
    'dropped_files': [],
    '_schema_version': 1,
}


def _fixture_plans() -> list[Any]:
    return [
        pytest.param(json.loads(path.read_text()), id=f'fixture:{path.relative_to(_FIXTURES)}')
        for path in sorted(_FIXTURES.glob('**/plan.json'))
    ]


ROUND_TRIP_INPUTS = [
    *_fixture_plans(),
    pytest.param(LIVE_SHAPED_PLAN, id='live-shaped'),
    pytest.param(VALID_PLAN_WITH_STEPS, id='valid-plan-with-steps'),
    pytest.param(CREATE_PLAN_SHELL, id='create-plan-shell'),
    pytest.param({'task_id': 'task-1', 'title': 'no steps key'}, id='no-steps-key'),
    pytest.param({'task_id': 'task-1', 'steps': 'not-a-list'}, id='steps-not-a-list'),
    pytest.param({**VALID_PLAN_WITH_STEPS, '_finalized_at': ''}, id='empty-finalized-at'),
    pytest.param({**VALID_PLAN_WITH_STEPS, '_session_id': 7}, id='int-session-id'),
    pytest.param({**VALID_PLAN_WITH_STEPS, '_schema_version': True}, id='bool-schema-version'),
    pytest.param(
        {**VALID_PLAN_WITH_STEPS, '_finalized_at': '2026-11-11T00:00:00+00:00', '_confirmed_by': 'nobody'},
        id='unknown-confirmer',
    ),
    pytest.param({**VALID_PLAN_WITH_STEPS, '_confirmed_by': 'architect'}, id='confirmer-without-marker'),
]


class TestRoundTrip:
    @pytest.mark.parametrize('document', ROUND_TRIP_INPUTS)
    def test_to_json_reproduces_the_document_byte_for_byte(self, document: dict[str, Any]):
        out = Plan.from_json(document).to_json()
        assert out == document
        assert list(out.items()) == list(document.items())
        assert json.dumps(out) == json.dumps(document)

    @pytest.mark.parametrize('document', ROUND_TRIP_INPUTS)
    def test_the_plan_does_not_alias_its_input(self, document: dict[str, Any]):
        source = copy.deepcopy(document)
        expected = copy.deepcopy(document)
        plan = Plan.from_json(source)
        source['injected'] = 'later'
        for value in source.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item['mutated'] = True
                value.append('appended')
        assert plan.to_json() == expected

    @pytest.mark.parametrize('document', ROUND_TRIP_INPUTS)
    def test_each_to_json_is_a_fresh_document(self, document: dict[str, Any]):
        plan = Plan.from_json(document)
        first, second = plan.to_json(), plan.to_json()
        assert first is not second
        for key, value in first.items():
            if isinstance(value, (list, dict)):
                assert value is not second[key]
        first['injected'] = 'later'
        assert plan.to_json() == document

    def test_a_non_mapping_is_refused_naming_its_type(self):
        with pytest.raises(TypeError, match='list'):
            Plan.from_json([])  # type: ignore[arg-type]


_CONFIRMED_AT = '2026-11-11T00:00:00+00:00'


class TestState:
    @pytest.mark.parametrize(('document', 'state'), [
        pytest.param({'task_id': 't'}, PlanState.SHELL, id='no-steps-key'),
        pytest.param({'task_id': 't', 'steps': []}, PlanState.SHELL, id='empty-steps'),
        pytest.param({'task_id': 't', 'steps': 'not-a-list'}, PlanState.SHELL, id='steps-not-a-list'),
        pytest.param(
            {'task_id': 't', 'steps': [], '_finalized_at': _CONFIRMED_AT, '_session_id': 's'},
            PlanState.SHELL, id='confirmed-owned-but-stepless',
        ),
        pytest.param(VALID_PLAN_WITH_STEPS, PlanState.PARTIAL, id='steps-unconfirmed'),
        pytest.param({**VALID_PLAN_WITH_STEPS, '_session_id': 's'}, PlanState.PARTIAL, id='owned-unconfirmed'),
        pytest.param({**VALID_PLAN_WITH_STEPS, '_finalized_at': ''}, PlanState.PARTIAL, id='empty-marker'),
        pytest.param({**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT}, PlanState.CONFIRMED, id='confirmed'),
        pytest.param(
            {**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT, '_session_id': 's'},
            PlanState.STAMPED, id='confirmed-owned',
        ),
        pytest.param(LIVE_SHAPED_PLAN, PlanState.STAMPED, id='live-shaped'),
    ])
    def test_state_derives_from_steps_and_lifecycle(self, document: dict[str, Any], state: PlanState):
        assert Plan.from_json(document).state is state


class TestConfirmation:
    def test_an_unconfirmed_plan_has_no_confirmer(self):
        assert Plan.from_json(VALID_PLAN_WITH_STEPS).lifecycle.confirmation is None

    def test_a_legacy_confirmed_plan_reads_as_the_architect_without_gaining_the_key(self):
        document = {**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT}
        plan = Plan.from_json(document)
        assert plan.lifecycle.confirmed_by is None
        assert plan.lifecycle.confirmation is PlanConfirmation.ARCHITECT
        assert '_confirmed_by' not in plan.to_json()

    def test_an_injected_confirmation_reads_as_injected(self):
        document = {**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT, '_confirmed_by': 'injected'}
        assert Plan.from_json(document).lifecycle.confirmation is PlanConfirmation.INJECTED

    def test_an_unknown_confirmer_stays_in_the_body_verbatim(self):
        document = {**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT, '_confirmed_by': 'nobody'}
        plan = Plan.from_json(document)
        assert plan.lifecycle.confirmed_by is None
        assert plan.body['_confirmed_by'] == 'nobody'


class TestLifecycleConstruction:
    @pytest.mark.parametrize(('kwargs', 'field', 'value'), [
        pytest.param({'session_id': ''}, 'session_id', '', id='empty-str'),
        pytest.param({'base_commit': 7}, 'base_commit', 7, id='non-str'),
        pytest.param(
            {'confirmed_at': _CONFIRMED_AT, 'confirmed_by': 'architect'}, 'confirmed_by', 'architect',
            id='plain-str-confirmer',
        ),
        pytest.param(
            {'confirmed_by': PlanConfirmation.ARCHITECT}, 'confirmed_by', PlanConfirmation.ARCHITECT,
            id='confirmer-without-marker',
        ),
        pytest.param({'schema_version': True}, 'schema_version', True, id='bool-schema-version'),
        pytest.param({'schema_version': '1'}, 'schema_version', '1', id='str-schema-version'),
    ])
    def test_a_malformed_field_is_refused_naming_it_and_its_value(
        self, kwargs: dict[str, Any], field: str, value: object,
    ):
        with pytest.raises(ValueError) as excinfo:
            PlanLifecycle(**kwargs)
        assert field in str(excinfo.value)
        assert repr(value) in str(excinfo.value)

    def test_markers_present_lists_the_set_on_disk_keys_sorted(self):
        lifecycle = Plan.from_json(LIVE_SHAPED_PLAN).lifecycle
        assert lifecycle.markers_present() == tuple(sorted(
            key for key in LIVE_SHAPED_PLAN if key in {
                '_schema_version', '_finalized_at', '_revalidated_at', '_session_id',
                '_created_at', PLAN_TASK_TEXT_FP_KEY, PLAN_BASE_COMMIT_KEY,
            }
        ))

    @pytest.mark.parametrize(('lifecycle', 'session', 'owned'), [
        pytest.param(PlanLifecycle(session_id='a'), 'a', True, id='planner'),
        pytest.param(PlanLifecycle(session_id='a', revalidated_by_session='b'), 'b', True, id='revalidator'),
        pytest.param(PlanLifecycle(session_id='a', revalidated_by_session='b'), 'c', False, id='third-party'),
        pytest.param(PlanLifecycle(), 'a', False, id='unowned'),
    ])
    def test_owned_by_accepts_the_planner_or_the_revalidator(
        self, lifecycle: PlanLifecycle, session: str, owned: bool,
    ):
        assert lifecycle.owned_by(session) is owned


_MALFORMED_STAMPS = [
    pytest.param({}, id='absent'),
    pytest.param('', id='empty'),
    pytest.param(None, id='none'),
    pytest.param(7, id='int'),
]


class TestStampReadersAgreeWithTheModel:
    @pytest.mark.parametrize('value', [*_MALFORMED_STAMPS, pytest.param('sha-or-fp', id='set')])
    def test_plan_base_commit(self, value: object):
        document = {} if value == {} else {PLAN_BASE_COMMIT_KEY: value}
        assert plan_base_commit(document) == Plan.from_json(document).lifecycle.base_commit
        assert plan_base_commit(document) == (value if value == 'sha-or-fp' else None)

    @pytest.mark.parametrize('value', [*_MALFORMED_STAMPS, pytest.param('sha-or-fp', id='set')])
    def test_plan_task_text_fp(self, value: object):
        document = {} if value == {} else {PLAN_TASK_TEXT_FP_KEY: value}
        assert plan_task_text_fp(document) == Plan.from_json(document).lifecycle.task_text_fp
        assert plan_task_text_fp(document) == (value if value == 'sha-or-fp' else None)
