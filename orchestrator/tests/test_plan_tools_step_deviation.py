"""The plan-tools ``record_step_deviation`` tool (PRD contract §6.2).

``plans/plan-deviation-recording-prd.md`` D4: an agent records that a plan
step's text proved wrong at execution time. The record lands on
``step['deviations']`` and never changes the step's ``status``, ``commit`` or
``description``. A sibling of ``test_plan_tools_server.py``.

Markup specimens are BUILT from ``_markup_helpers``, never spelled verbatim:
see that module's docstring for the sentinel-literal hazard.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from _markup_helpers import INVOKE_CLOSER, closer
from _plan_markup_fixtures import (
    corrupt_plan,
    isolate_the_refusal_memo,  # noqa: F401 — autouse; pytest resolves it by name
    on_disk,
    plan_artifacts,  # noqa: F401 — a fixture, requested by test parameter name
)
from shared.plan_deviation import DEVIATION_KINDS

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.mcp.plan_tools import (
    _add_plan_step,
    _add_prerequisite,
    _create_plan,
    _mark_step_done,
    _record_step_deviation,
)

_RECORD_KEYS = {
    'kind',
    'observed',
    'expected',
    'action_taken',
    'prose_only',
    'note',
    'recorded_at',
    'recorded_by',
}


@pytest.fixture()
def artifacts(tmp_path):
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def _seed_plan(artifacts: TaskArtifacts) -> None:
    _create_plan(artifacts, 'test-1', 'Test task', 'Analysis', ['mod_a/foo.py'])
    _add_prerequisite(artifacts, 'pre-1', 'Setup config')
    _add_plan_step(artifacts, 'step-1', 'test', 'Write test for foo')


def _record(artifacts: TaskArtifacts, step_id: str = 'step-1', kind: str = 'other') -> dict:
    return _record_step_deviation(
        artifacts,
        step_id,
        kind,
        'The helper already existed.',
        'The step said to write the helper.',
        'Reused the existing helper.',
    )


def _item(plan: dict, item_id: str) -> dict:
    return next(
        item
        for collection in ('prerequisites', 'steps')
        for item in plan[collection]
        if item['id'] == item_id
    )


class TestRecordStepDeviation:
    def test_appends_a_complete_record_and_reports_the_count(self, artifacts):
        _seed_plan(artifacts)
        before = _item(artifacts.read_plan(), 'step-1')

        result = _record_step_deviation(
            artifacts,
            'step-1',
            'wrong_inference',
            'Two callers, not one.',
            'One caller.',
            'Updated both callers.',
            prose_only=True,
            note='Docstring claim adjusted.',
        )

        assert result == {'status': 'ok', 'step_id': 'step-1', 'deviations': 1}
        step = _item(artifacts.read_plan(), 'step-1')
        [record] = step['deviations']
        assert set(record) == _RECORD_KEYS
        assert record['kind'] == 'wrong_inference'
        assert record['observed'] == 'Two callers, not one.'
        assert record['expected'] == 'One caller.'
        assert record['action_taken'] == 'Updated both callers.'
        assert record['prose_only'] is True
        assert record['note'] == 'Docstring claim adjusted.'
        assert datetime.fromisoformat(record['recorded_at']).utcoffset() == timedelta(0)
        for key in ('status', 'commit', 'description'):
            assert step[key] == before[key]

        assert _record(artifacts)['deviations'] == 2

    def test_accepts_a_step_already_marked_done(self, artifacts):
        _seed_plan(artifacts)
        assert _mark_step_done(artifacts, 'step-1', 'a' * 40)['status'] == 'ok'

        assert _record(artifacts)['status'] == 'ok'

        step = _item(artifacts.read_plan(), 'step-1')
        assert step['status'] == 'done'
        assert len(step['deviations']) == 1

    def test_accepts_a_prerequisite(self, artifacts):
        _seed_plan(artifacts)

        assert _record(artifacts, 'pre-1') == {
            'status': 'ok', 'step_id': 'pre-1', 'deviations': 1,
        }

    def test_the_record_survives_a_later_mark_step_done(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts)

        assert _mark_step_done(artifacts, 'step-1', 'b' * 40)['status'] == 'ok'

        step = _item(artifacts.read_plan(), 'step-1')
        assert step['commit'] == 'b' * 40
        assert len(step['deviations']) == 1

    def test_an_unknown_kind_is_refused_naming_the_vocabulary(self, artifacts):
        _seed_plan(artifacts)
        before = artifacts.read_plan()

        result = _record(artifacts, kind='typo')

        assert result['status'] == 'error'
        for kind in DEVIATION_KINDS:
            assert kind in result['message']
        assert result['kind'] == 'typo'
        assert result['allowed_kinds'] == sorted(DEVIATION_KINDS)
        assert artifacts.read_plan() == before

    def test_an_unknown_step_is_refused_naming_it(self, artifacts):
        _seed_plan(artifacts)
        before = artifacts.read_plan()

        result = _record(artifacts, 'nope')

        assert result['status'] == 'error'
        assert 'nope' in result['message']
        assert artifacts.read_plan() == before

    def test_no_plan_is_refused(self, artifacts):
        assert _record(artifacts)['status'] == 'error'


class TestRecordedBy:
    def test_the_agent_session_sidecar_wins(self, artifacts):
        _seed_plan(artifacts)
        artifacts.stamp_plan_provenance('orch-sess-1')
        artifacts.write_agent_session(
            'claude-sess-1', 'implementer', '2026-10-15T07:00:00+00:00',
        )

        _record(artifacts)

        [record] = _item(artifacts.read_plan(), 'step-1')['deviations']
        assert record['recorded_by'] == 'claude-sess-1'

    def test_falls_back_to_the_plan_session_id(self, artifacts):
        _seed_plan(artifacts)
        artifacts.stamp_plan_provenance('orch-sess-1')

        _record(artifacts)

        [record] = _item(artifacts.read_plan(), 'step-1')['deviations']
        assert record['recorded_by'] == 'orch-sess-1'

    def test_is_none_when_no_identity_is_in_hand(self, artifacts):
        _seed_plan(artifacts)

        _record(artifacts)

        [record] = _item(artifacts.read_plan(), 'step-1')['deviations']
        assert record['recorded_by'] is None


@pytest.mark.asyncio
async def test_the_tool_is_registered_and_defaults_its_optional_fields(artifacts):
    _seed_plan(artifacts)
    server = plan_tools.create_server(artifacts)
    tool = await server.get_tool('record_step_deviation')
    assert tool is not None

    result = await tool.run({
        'step_id': 'step-1',
        'kind': 'other',
        'observed': 'o',
        'expected': 'e',
        'action_taken': 'a',
    })

    assert result.structured_content is not None
    assert result.structured_content['status'] == 'ok'
    [record] = _item(artifacts.read_plan(), 'step-1')['deviations']
    assert record['prose_only'] is False
    assert record['note'] == ''


def test_read_repair_keeps_the_deviations_list(plan_artifacts):
    deviations = [{
        'kind': 'other',
        'observed': 'Observed.',
        'expected': 'Expected.',
        'action_taken': 'Acted.',
        'prose_only': False,
        'note': '',
        'recorded_at': '2026-10-15T07:00:00+00:00',
        'recorded_by': None,
    }]
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = (
        'A clean rationale.' + closer('rationale') + '\n' + INVOKE_CLOSER + '\n'
    )
    plan['steps'][0]['deviations'] = deviations
    plan_artifacts.write_plan(plan)

    repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

    assert any(fact['outcome'] == 'repaired' for fact in facts)
    assert repaired['steps'][0]['deviations'] == deviations
    assert on_disk(plan_artifacts)['steps'][0]['deviations'] == deviations
