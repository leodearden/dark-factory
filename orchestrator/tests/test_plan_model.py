"""The typed Plan model in ``orchestrator/src/orchestrator/artifacts.py`` (S9 C1/D1/D2)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _plan_model_helpers import read_model
from shared.plan_deviation import DeviationRecord

from orchestrator import plan_descope
from orchestrator.artifacts import (
    PLAN_BASE_COMMIT_KEY,
    PLAN_SCHEMA_VERSION,
    PLAN_TASK_TEXT_FP_KEY,
    Plan,
    PlanConfirmation,
    PlanLifecycle,
    PlanMutexNotHeld,
    PlanNotConfirmed,
    PlanState,
    TaskArtifacts,
    plan_base_commit,
    plan_task_text_fp,
)
from orchestrator.mcp import plan_markup_stamp
from orchestrator.mcp.plan_tools import (
    _add_plan_step,
    _confirm_plan,
    _create_plan,
    _drop_plan_file,
    _record_descoped_step,
    _update_plan_metadata,
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
            Plan.from_json([])


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


_LATER = '2026-11-12T00:00:00+00:00'

CONFIRMED_PLAN: dict[str, Any] = {**VALID_PLAN_WITH_STEPS, '_finalized_at': _CONFIRMED_AT}
OWNED_PARTIAL_PLAN: dict[str, Any] = {**VALID_PLAN_WITH_STEPS, '_session_id': 'planner'}

_CARRIED: dict[str, Any] = {
    '_markup_rejections': LIVE_SHAPED_PLAN['_markup_rejections'],
    '_rescoped_at': LIVE_SHAPED_PLAN['_rescoped_at'],
    '_descoped_steps': LIVE_SHAPED_PLAN['_descoped_steps'],
}


def _recreate(plan: Plan, carried: dict[str, Any]) -> Plan:
    return plan.recreated(task_id='9170', title='T', analysis='A', files=['a.py'], carried=carried)


class TestRecreated:
    def test_create_plan_over_a_stamped_plan_gives_a_shell_carrying_only_the_carried_keys(self):
        plan = _recreate(Plan.from_json(LIVE_SHAPED_PLAN), _CARRIED)
        assert plan.state is PlanState.SHELL
        assert plan.lifecycle == PlanLifecycle()
        expected = {
            'task_id': '9170', 'title': 'T', 'analysis': 'A', 'files': ['a.py'],
            'prerequisites': [], 'steps': [], 'design_decisions': [], 'reuse': [], 'dropped_files': [],
            **_CARRIED,
        }
        out = plan.to_json()
        assert list(out.items()) == list(expected.items())

    @pytest.mark.parametrize('key', ['_finalized_at', '_session_id', '_schema_version', 'steps'])
    def test_a_carried_lifecycle_key_or_steps_is_refused_naming_it(self, key: str):
        with pytest.raises(ValueError, match=key):
            _recreate(Plan.from_json(LIVE_SHAPED_PLAN), {**_CARRIED, key: 'x'})


_KEPT_ON_RESTRUCTURE = ('session_id', 'created_at', 'task_text_fp', 'base_commit', 'revalidated_at')


class TestRestructured:
    @pytest.mark.parametrize('document', [
        pytest.param(LIVE_SHAPED_PLAN, id='stamped'),
        pytest.param({**CONFIRMED_PLAN, '_confirmed_by': 'architect'}, id='confirmed'),
    ])
    def test_a_step_set_edit_clears_confirmation_and_keeps_the_rest(self, document: dict[str, Any]):
        before = Plan.from_json(document)
        after = before.restructured()
        assert after.state is PlanState.PARTIAL
        assert after.lifecycle.confirmed_at is None
        assert after.lifecycle.confirmed_by is None
        for name in _KEPT_ON_RESTRUCTURE:
            assert getattr(after.lifecycle, name) == getattr(before.lifecycle, name)
        assert after.steps == before.steps
        assert '_finalized_at' not in after.to_json()
        assert '_confirmed_by' not in after.to_json()

    def test_the_changes_are_applied_and_new_keys_appended(self):
        new_steps = [dict(_STEP), {**_STEP, 'id': 'step-2'}]
        prerequisites = [{'id': 'pre-1', 'description': 'p', 'status': 'pending'}]
        after = Plan.from_json(CONFIRMED_PLAN).restructured(steps=new_steps, prerequisites=prerequisites)
        assert after.state is PlanState.PARTIAL
        out = after.to_json()
        assert list(out) == ['task_id', 'title', 'steps', 'prerequisites']
        assert out['steps'] == new_steps
        assert out['prerequisites'] == prerequisites

    def test_a_malformed_marker_in_the_body_is_dropped(self):
        document = {**VALID_PLAN_WITH_STEPS, '_finalized_at': '', '_confirmed_by': 'nobody'}
        out = Plan.from_json(document).restructured().to_json()
        assert '_finalized_at' not in out
        assert '_confirmed_by' not in out

    @pytest.mark.parametrize('key', ['_finalized_at', '_confirmed_by', '_session_id', PLAN_BASE_COMMIT_KEY])
    def test_a_lifecycle_named_change_is_refused_naming_it(self, key: str):
        with pytest.raises(ValueError, match=key):
            Plan.from_json(LIVE_SHAPED_PLAN).restructured(**{key: 'x'})


class TestConfirmed:
    def test_confirm_sets_the_marker_the_confirmer_and_the_revalidation_age(self):
        plan = Plan.from_json(VALID_PLAN_WITH_STEPS).confirmed(at=_LATER, by=PlanConfirmation.ARCHITECT)
        assert plan.lifecycle.confirmed_at == _LATER
        assert plan.lifecycle.confirmed_by is PlanConfirmation.ARCHITECT
        assert plan.lifecycle.revalidated_at == _LATER
        assert plan.state is PlanState.CONFIRMED

    def test_confirming_an_owned_plan_gives_stamped(self):
        plan = Plan.from_json(OWNED_PARTIAL_PLAN).confirmed(at=_LATER, by=PlanConfirmation.INJECTED)
        assert plan.state is PlanState.STAMPED
        assert plan.lifecycle.confirmation is PlanConfirmation.INJECTED

    def test_confirming_a_shell_is_refused_naming_the_task_the_state_and_the_markers(self):
        with pytest.raises(PlanNotConfirmed) as excinfo:
            Plan.from_json(CREATE_PLAN_SHELL).confirmed(at=_LATER, by=PlanConfirmation.ARCHITECT)
        message = str(excinfo.value)
        assert 'task-1' in message
        assert 'shell' in message
        assert '_schema_version' in message
        assert excinfo.value.state is PlanState.SHELL
        assert excinfo.value.action == 'confirm'


def _stamp(plan: Plan, *, fp: str | None = None, base: str | None = None) -> Plan:
    return plan.stamped(session_id='stamper', at=_LATER, task_text_fp=fp, base_commit=base)


class TestStamped:
    def test_stamping_a_confirmed_plan_gives_stamped_and_keeps_the_confirmation(self):
        before = Plan.from_json({**CONFIRMED_PLAN, '_confirmed_by': 'injected'})
        after = _stamp(before)
        assert after.state is PlanState.STAMPED
        assert after.lifecycle.session_id == 'stamper'
        assert after.lifecycle.created_at == _LATER
        assert after.lifecycle.confirmed_at == before.lifecycle.confirmed_at
        assert after.lifecycle.confirmed_by is PlanConfirmation.INJECTED

    @pytest.mark.parametrize('given', [None, ''])
    def test_an_absent_fingerprint_or_base_keeps_the_existing_one(self, given: str | None):
        after = _stamp(Plan.from_json(LIVE_SHAPED_PLAN), fp=given, base=given)
        assert after.lifecycle.task_text_fp == LIVE_SHAPED_PLAN[PLAN_TASK_TEXT_FP_KEY]
        assert after.lifecycle.base_commit == LIVE_SHAPED_PLAN[PLAN_BASE_COMMIT_KEY]

    def test_a_given_fingerprint_and_base_overwrite(self):
        after = _stamp(Plan.from_json(LIVE_SHAPED_PLAN), fp='tfp:new', base='newbase')
        assert after.lifecycle.task_text_fp == 'tfp:new'
        assert after.lifecycle.base_commit == 'newbase'

    @pytest.mark.parametrize(('document', 'state'), [
        pytest.param(OWNED_PARTIAL_PLAN, PlanState.PARTIAL, id='partial'),
        pytest.param(CREATE_PLAN_SHELL, PlanState.SHELL, id='shell'),
    ])
    def test_stamping_an_unconfirmed_plan_is_refused(self, document: dict[str, Any], state: PlanState):
        plan = Plan.from_json(document)
        with pytest.raises(PlanNotConfirmed) as excinfo:
            _stamp(plan)
        message = str(excinfo.value)
        assert 'task-1' in message
        assert state.value in message
        for marker in plan.lifecycle.markers_present():
            assert marker in message
        assert excinfo.value.action == 'stamp'


class TestWithFiles:
    def test_a_files_edit_keeps_confirmation(self):
        after = Plan.from_json(LIVE_SHAPED_PLAN).with_files(['b.py'], owner=LIVE_SHAPED_PLAN['_session_id'], at=_LATER)
        assert after.state is PlanState.STAMPED
        assert after.to_json()['files'] == ['b.py']
        assert after.lifecycle == Plan.from_json(LIVE_SHAPED_PLAN).lifecycle

    def test_a_revalidating_owner_leaves_the_planner_s_provenance_alone(self):
        before = Plan.from_json({**LIVE_SHAPED_PLAN, '_revalidated_by_session': 'reval'})
        after = before.with_files(['b.py'], owner='reval', at=_LATER)
        assert after.lifecycle == before.lifecycle

    def test_a_new_owner_takes_ownership_and_backfills_created_at(self):
        after = Plan.from_json(CONFIRMED_PLAN).with_files(['b.py'], owner='scope', at=_LATER)
        assert after.state is PlanState.STAMPED
        assert after.lifecycle.session_id == 'scope'
        assert after.lifecycle.created_at == _LATER

    def test_a_new_owner_does_not_clobber_an_existing_created_at(self):
        after = Plan.from_json(LIVE_SHAPED_PLAN).with_files(['b.py'], owner='scope', at=_LATER)
        assert after.lifecycle.session_id == 'scope'
        assert after.lifecycle.created_at == LIVE_SHAPED_PLAN['_created_at']


class TestRevalidated:
    def test_revalidation_records_the_session_and_keeps_confirmation_and_ownership(self):
        before = Plan.from_json(LIVE_SHAPED_PLAN)
        after = before.revalidated(at=_LATER, by_session='reval', base_commit='newbase')
        assert after.lifecycle.revalidated_at == _LATER
        assert after.lifecycle.revalidated_by_session == 'reval'
        assert after.lifecycle.base_commit == 'newbase'
        assert after.lifecycle.confirmed_at == before.lifecycle.confirmed_at
        assert after.lifecycle.session_id == before.lifecycle.session_id
        assert after.state is PlanState.STAMPED

    @pytest.mark.parametrize('given', [None, ''])
    def test_an_absent_base_keeps_the_existing_one(self, given: str | None):
        after = Plan.from_json(LIVE_SHAPED_PLAN).revalidated(at=_LATER, by_session='reval', base_commit=given)
        assert after.lifecycle.base_commit == LIVE_SHAPED_PLAN[PLAN_BASE_COMMIT_KEY]


class TestConfirmedSince:
    def test_a_fresh_confirm_counts(self):
        pre = Plan.from_json(LIVE_SHAPED_PLAN)
        assert pre.restructured().confirmed(at=_LATER, by=PlanConfirmation.ARCHITECT).confirmed_since(pre)

    def test_no_change_does_not_count(self):
        pre = Plan.from_json(LIVE_SHAPED_PLAN)
        assert not Plan.from_json(LIVE_SHAPED_PLAN).confirmed_since(pre)

    def test_a_cleared_marker_does_not_count(self):
        pre = Plan.from_json(LIVE_SHAPED_PLAN)
        assert not pre.restructured().confirmed_since(pre)


class TestWithLifecycle:
    def test_this_plan_s_steps_and_body_go_under_the_other_lifecycle(self):
        mine = Plan.from_json({**VALID_PLAN_WITH_STEPS, 'analysis': 'mine', '_finalized_at': ''})
        other = Plan.from_json(LIVE_SHAPED_PLAN)
        joined = mine.with_lifecycle(other.lifecycle)
        assert joined.lifecycle == other.lifecycle
        assert joined.steps == mine.steps
        assert '_finalized_at' not in joined.body
        assert joined.to_json()['analysis'] == 'mine'
        assert joined.to_json()['_finalized_at'] == LIVE_SHAPED_PLAN['_finalized_at']


_VALUE_METHODS = [
    pytest.param(lambda p: _recreate(p, _CARRIED), id='recreated'),
    pytest.param(lambda p: p.restructured(steps=[dict(_STEP)], analysis='new'), id='restructured'),
    pytest.param(lambda p: p.confirmed(at=_LATER, by=PlanConfirmation.ARCHITECT), id='confirmed'),
    pytest.param(lambda p: _stamp(p, fp='tfp:new', base='newbase'), id='stamped'),
    pytest.param(lambda p: p.with_files(['b.py'], owner='scope', at=_LATER), id='with_files'),
    pytest.param(lambda p: p.revalidated(at=_LATER, by_session='reval', base_commit='newbase'), id='revalidated'),
    pytest.param(lambda p: p.with_lifecycle(PlanLifecycle()), id='with_lifecycle'),
]


def test_nested_values_are_frozen_so_a_derived_plan_cannot_reach_its_receiver():
    plan = Plan.from_json(LIVE_SHAPED_PLAN)
    derived = plan.restructured()
    with pytest.raises(AttributeError):
        derived.steps[0]['deviations'].append('leak')
    with pytest.raises(TypeError):
        derived.steps[0]['deviations'][0]['kind'] = 'leak'
    with pytest.raises(AttributeError):
        derived.body['files'].append('leak')
    assert plan.to_json() == LIVE_SHAPED_PLAN
    assert derived.to_json()['steps'][0]['deviations'] == LIVE_SHAPED_PLAN['steps'][0]['deviations']


@pytest.mark.parametrize('method', _VALUE_METHODS)
def test_every_value_method_leaves_its_receiver_unchanged(method: Any):
    plan = Plan.from_json(LIVE_SHAPED_PLAN)
    method(plan)
    assert plan.to_json() == LIVE_SHAPED_PLAN
    assert list(plan.to_json()) == list(LIVE_SHAPED_PLAN)


@pytest.fixture
def artifacts(tmp_path: Path) -> TaskArtifacts:
    worktree = tmp_path / 'worktree'
    worktree.mkdir()
    task_artifacts = TaskArtifacts(worktree)
    task_artifacts.init('task-1', 'Test Task', 'A test task description')
    return task_artifacts


def _plan_path(artifacts: TaskArtifacts) -> Path:
    return artifacts.artifact_path('plan.json')


def _on_disk(artifacts: TaskArtifacts) -> dict[str, Any]:
    return json.loads(_plan_path(artifacts).read_text())


def _listing(artifacts: TaskArtifacts) -> list[str]:
    return sorted(path.name for path in _plan_path(artifacts).parent.iterdir())


MALFORMED_SHAPED_PLAN: dict[str, Any] = {
    'task_id': 'task-1',
    'title': 'Test Task',
    'tdd_steps': ['write the test', 'make it pass'],
}


class TestReadAndPeek:
    def test_no_plan_file_reads_as_none(self, artifacts: TaskArtifacts):
        assert artifacts.read_plan_model() is None
        assert artifacts.peek_plan_model() is None

    def test_an_empty_document_reads_as_a_shell(self, artifacts: TaskArtifacts):
        _plan_path(artifacts).write_text('{}')
        for plan in (artifacts.read_plan_model(), artifacts.peek_plan_model()):
            assert plan is not None
            assert plan.state is PlanState.SHELL

    def test_read_repairs_a_malformed_shape_once_and_returns_what_it_wrote(self, artifacts: TaskArtifacts):
        _plan_path(artifacts).write_text(json.dumps(MALFORMED_SHAPED_PLAN))
        raw_bytes = _plan_path(artifacts).read_bytes()
        os.utime(_plan_path(artifacts), ns=(0, 0))
        plan = artifacts.read_plan_model()
        assert plan is not None
        assert plan.state is PlanState.PARTIAL
        assert [step['description'] for step in plan.steps] == MALFORMED_SHAPED_PLAN['tdd_steps']
        assert _plan_path(artifacts).read_bytes() != raw_bytes
        assert _plan_path(artifacts).stat().st_mtime_ns != 0
        assert _on_disk(artifacts)['_schema_version'] == PLAN_SCHEMA_VERSION
        assert plan.to_json() == _on_disk(artifacts)
        assert Plan.from_json(_on_disk(artifacts)) == plan

        repaired_bytes = _plan_path(artifacts).read_bytes()
        os.utime(_plan_path(artifacts), ns=(0, 0))
        assert artifacts.read_plan_model() == plan
        assert _plan_path(artifacts).read_bytes() == repaired_bytes
        assert _plan_path(artifacts).stat().st_mtime_ns == 0

    def test_peek_normalizes_the_same_shape_without_writing(self, artifacts: TaskArtifacts):
        _plan_path(artifacts).write_text(json.dumps(MALFORMED_SHAPED_PLAN))
        raw_bytes = _plan_path(artifacts).read_bytes()
        listing = _listing(artifacts)
        peeked = artifacts.peek_plan_model()
        assert _plan_path(artifacts).read_bytes() == raw_bytes
        assert _listing(artifacts) == listing
        assert peeked is not None
        assert peeked.state is PlanState.PARTIAL
        assert peeked == artifacts.read_plan_model()


class TestWritePlanModel:
    def test_the_write_stamps_the_schema_version(self, artifacts: TaskArtifacts):
        plan = Plan.from_json(VALID_PLAN_WITH_STEPS)
        artifacts.write_plan_model(plan)
        expected = {**plan.to_json(), '_schema_version': PLAN_SCHEMA_VERSION}
        assert list(_on_disk(artifacts).items()) == list(expected.items())

    def test_the_write_returns_the_plan_it_wrote(self, artifacts: TaskArtifacts):
        written = artifacts.write_plan_model(Plan.from_json(VALID_PLAN_WITH_STEPS))
        assert written.lifecycle.schema_version == PLAN_SCHEMA_VERSION
        assert written.to_json() == _on_disk(artifacts)
        assert written == read_model(artifacts)

    def test_a_claimed_mutex_that_is_not_held_writes_nothing(self, artifacts: TaskArtifacts):
        artifacts.write_plan_model(Plan.from_json(VALID_PLAN_WITH_STEPS))
        before = _plan_path(artifacts).read_bytes()
        with pytest.raises(PlanMutexNotHeld):
            artifacts.write_plan_model(Plan.from_json(LIVE_SHAPED_PLAN), mutex_held=True)
        assert _plan_path(artifacts).read_bytes() == before

    @pytest.mark.parametrize('document', ROUND_TRIP_INPUTS)
    def test_the_dict_adapter_writes_the_same_bytes(self, artifacts: TaskArtifacts, document: dict[str, Any]):
        artifacts.write_plan(copy.deepcopy(document))
        through_adapter = _plan_path(artifacts).read_bytes()
        artifacts.write_plan_model(Plan.from_json(document))
        assert _plan_path(artifacts).read_bytes() == through_adapter

    def test_the_dict_adapter_still_stamps_the_caller_s_dict(self, artifacts: TaskArtifacts):
        document = copy.deepcopy(VALID_PLAN_WITH_STEPS)
        artifacts.write_plan(document)
        assert document['_schema_version'] == PLAN_SCHEMA_VERSION


_STEP_PROGRESS_WRITERS = [
    pytest.param(lambda a: a.update_step_status('step-2', 'done', 'd' * 40), id='update_step_status'),
    pytest.param(lambda a: a.mark_step_committed('step-2', 'd' * 40), id='mark_step_committed'),
    pytest.param(
        lambda a: a.append_step_deviation('step-1', DeviationRecord(
            kind='other', observed='o', expected='e', action_taken='a', prose_only=False, note='',
            recorded_at=_LATER, recorded_by=None,
        )),
        id='append_step_deviation',
    ),
]


@pytest.mark.parametrize('write', _STEP_PROGRESS_WRITERS)
def test_step_progress_keeps_a_stamped_plan_stamped(artifacts: TaskArtifacts, write: Any):
    artifacts.write_plan(copy.deepcopy(LIVE_SHAPED_PLAN))
    before = artifacts.read_plan_model()
    assert before is not None
    write(artifacts)
    after = artifacts.read_plan_model()
    assert after is not None
    assert after.state is PlanState.STAMPED
    assert after.lifecycle == before.lifecycle
    assert after.steps != before.steps


_SEED = RepoSeed(files=(('.gitignore', '.task/\n'), ('m.py', 'x = 1\n')), message='Initial')


@pytest.fixture(autouse=True)
def _clear_pending_refusals() -> Iterator[None]:
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


@dataclass(frozen=True)
class _Lane:
    artifacts: TaskArtifacts
    head: str


@pytest.fixture
def lane(tmp_path: Path) -> _Lane:
    repo = seed_repo(tmp_path / 'lane', _SEED)
    lane_artifacts = TaskArtifacts(repo)
    lane_artifacts.init('test-1', 'Test task', 'A test')
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return _Lane(lane_artifacts, head)


def _build_two_step_plan(artifacts: TaskArtifacts) -> None:
    assert _create_plan(artifacts, 'test-1', 'T', 'A', ['a.py', 'b.py'])['status'] == 'ok'
    assert _add_plan_step(artifacts, 'step-1', 'test', 'first')['status'] == 'ok'
    assert _add_plan_step(artifacts, 'step-2', 'impl', 'second')['status'] == 'ok'


def _stamped_lane_plan(artifacts: TaskArtifacts) -> Plan:
    _build_two_step_plan(artifacts)
    assert _confirm_plan(artifacts)['status'] == 'ok'
    stamped = artifacts.stamp_plan_provenance('stamper', task_text_fp='tfp:authored', base_commit='base0')
    assert stamped.state is PlanState.STAMPED
    return stamped


def test_row_1_an_authored_plan_confirms_as_the_architect_with_today_s_bytes(lane: _Lane):
    _build_two_step_plan(lane.artifacts)
    assert _confirm_plan(lane.artifacts)['status'] == 'ok'

    plan = read_model(lane.artifacts)
    assert plan.state is PlanState.CONFIRMED
    assert plan.lifecycle.confirmation is PlanConfirmation.ARCHITECT
    on_disk = _on_disk(lane.artifacts)
    assert on_disk['_finalized_at'] == on_disk['_revalidated_at']
    expected = {
        'task_id': 'test-1',
        'title': 'T',
        'analysis': 'A',
        'files': ['a.py', 'b.py'],
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'first', 'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'description': 'second', 'status': 'pending', 'commit': None},
        ],
        'design_decisions': [],
        'reuse': [],
        'dropped_files': [],
        '_schema_version': PLAN_SCHEMA_VERSION,
        '_finalized_at': on_disk['_finalized_at'],
        '_confirmed_by': 'architect',
        '_revalidated_at': on_disk['_revalidated_at'],
    }
    assert _plan_path(lane.artifacts).read_text() == json.dumps(expected, indent=2) + '\n'


def test_row_2_create_plan_over_a_stamped_plan_gives_a_shell_keeping_the_carried_blocks(lane: _Lane):
    _build_two_step_plan(lane.artifacts)
    assert _record_descoped_step(lane.artifacts, 'step-1', lane.head, 'orphan', 'esc-probe', 'note')['status'] == 'ok'
    with lane.artifacts.plan_mutex():
        document = lane.artifacts.read_plan()
        document[plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY] = plan_markup_stamp.block_of(
            plan_markup_stamp.build_event({'tool': 'add_plan_step', 'param': 'description', 'outcome': 'rejected'}),
        )
        lane.artifacts.write_plan(document, mutex_held=True)
    assert _confirm_plan(lane.artifacts)['status'] == 'ok'
    lane.artifacts.stamp_plan_provenance('stamper', task_text_fp='tfp:authored')
    before = lane.artifacts.read_plan()

    assert _create_plan(lane.artifacts, 'test-1', 'Re-plan', 'A2', ['a.py'])['status'] == 'ok'

    plan = read_model(lane.artifacts)
    assert plan.state is PlanState.SHELL
    assert plan.lifecycle.markers_present() == ('_schema_version',)
    after = lane.artifacts.read_plan()
    for key in (plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY, *plan_descope.DESCOPE_PLAN_KEYS):
        assert after[key] == before[key], key


def test_confirm_refuses_a_step_list_holding_a_non_step_with_its_envelope(lane: _Lane):
    _build_two_step_plan(lane.artifacts)
    with lane.artifacts.plan_mutex():
        document = lane.artifacts.read_plan()
        document['steps'].append(None)
        lane.artifacts.write_plan(document, mutex_held=True)
    before = _plan_path(lane.artifacts).read_bytes()

    envelope = _confirm_plan(lane.artifacts)

    assert envelope['status'] == 'error'
    assert 'no steps' in envelope['message']
    assert _plan_path(lane.artifacts).read_bytes() == before


def test_row_3_stamping_a_partial_plan_is_refused_and_writes_nothing(lane: _Lane):
    _build_two_step_plan(lane.artifacts)
    before = _plan_path(lane.artifacts).read_bytes()
    with pytest.raises(PlanNotConfirmed) as excinfo:
        lane.artifacts.write_plan_model(
            read_model(lane.artifacts).stamped(session_id='stamper', at=_LATER, task_text_fp=None, base_commit=None),
        )
    message = str(excinfo.value)
    assert 'test-1' in message
    assert 'partial' in message
    assert '_schema_version' in message
    assert _plan_path(lane.artifacts).read_bytes() == before


def test_row_3b_a_step_set_edit_on_a_stamped_plan_clears_confirmation(lane: _Lane):
    stamped = _stamped_lane_plan(lane.artifacts)
    assert _add_plan_step(lane.artifacts, 'step-3', 'impl', 'third')['status'] == 'ok'
    plan = read_model(lane.artifacts)
    assert plan.state is PlanState.PARTIAL
    assert plan.lifecycle.session_id == stamped.lifecycle.session_id
    assert plan.lifecycle.task_text_fp == stamped.lifecycle.task_text_fp


@pytest.mark.parametrize('tighten', [
    pytest.param(lambda a: _update_plan_metadata(a, files=['a.py']), id='update_plan_metadata'),
    pytest.param(lambda a: _drop_plan_file(a, 'b.py', 'narrowed at merge'), id='drop_plan_file'),
])
def test_row_3c_a_files_only_tightening_pass_keeps_a_stamped_plan_stamped(lane: _Lane, tighten: Any):
    stamped = _stamped_lane_plan(lane.artifacts)
    assert tighten(lane.artifacts)['status'] == 'ok'
    plan = read_model(lane.artifacts)
    assert plan.state is PlanState.STAMPED
    assert plan.lifecycle.confirmed_at == stamped.lifecycle.confirmed_at
    assert plan.to_json()['files'] == ['a.py']
