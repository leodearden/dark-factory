"""The plan-tools ``record_descoped_step`` tool (task 4032, β).

``plans/task-amendment-delivery-prd.md`` C4: a revalidation that genuinely
drops a done step records the drop and its disposition, so the done-step
integrity check can tell a sanctioned descope from a silent wipe. A sibling
of ``test_plan_tools_step_deviation.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.mcp.plan_tools import (
    _add_plan_step,
    _create_plan,
    _mark_step_done,
    _record_descoped_step,
)
from orchestrator.plan_descope import (
    DESCOPE_DISPOSITIONS,
    DESCOPE_PLAN_KEYS,
    DESCOPED_STEPS_KEY,
    RESCOPED_AT_KEY,
    RESCOPED_BY_KEY,
    RESCOPED_BY_UNKNOWN,
    RESCOPED_NOTE_KEY,
)

_ENTRY_KEYS = {'step_id', 'commit', 'disposition', 'authority', 'note'}
_NOTE = 'superseded by the ratified code-only scope'


@pytest.fixture()
def artifacts(tmp_path):
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def _seed_plan(artifacts: TaskArtifacts) -> None:
    _create_plan(artifacts, 'test-1', 'Test task', 'Analysis', ['mod_a/foo.py'])
    _add_plan_step(artifacts, 'step-1', 'test', 'Write test for foo')
    _add_plan_step(artifacts, 'step-2', 'impl', 'Implement foo')
    assert _mark_step_done(artifacts, 'step-1', '309c22ab6f')['status'] == 'ok'


def _record(artifacts: TaskArtifacts, **overrides) -> dict:
    args = {
        'step_id': 'step-1',
        'commit': '309c22ab6f',
        'disposition': 'orphan',
        'authority': 'esc-2971-6',
        'note': _NOTE,
    }
    args.update(overrides)
    return _record_descoped_step(artifacts, **args)


def _plan_bytes(artifacts: TaskArtifacts) -> bytes:
    return (artifacts.artifact_path('plan.json')).read_bytes()


class TestRecordDescopedStep:
    def test_appends_a_complete_entry_and_stamps_the_rescope_header(self, artifacts):
        _seed_plan(artifacts)

        result = _record(artifacts)

        assert result == {
            'status': 'ok', 'step_id': 'step-1', 'commit': '309c22ab6f', 'descoped_steps': 1,
        }
        plan = artifacts.read_plan()
        [entry] = plan[DESCOPED_STEPS_KEY]
        assert set(entry) == _ENTRY_KEYS
        assert entry == {
            'step_id': 'step-1',
            'commit': '309c22ab6f',
            'disposition': 'orphan',
            'authority': 'esc-2971-6',
            'note': _NOTE,
        }
        assert datetime.fromisoformat(plan[RESCOPED_AT_KEY]).utcoffset() == timedelta(0)
        assert plan[RESCOPED_NOTE_KEY] == _NOTE
        assert RESCOPED_BY_KEY in plan

    def test_rescoped_by_is_the_agent_session_sidecar(self, artifacts):
        _seed_plan(artifacts)
        artifacts.stamp_plan_provenance('orch-sess-1')
        artifacts.write_agent_session('claude-sess-1', 'architect', '2026-10-30T07:00:00+00:00')

        _record(artifacts)

        assert artifacts.read_plan()[RESCOPED_BY_KEY] == 'claude-sess-1'

    def test_rescoped_by_falls_back_to_the_plan_session_id(self, artifacts):
        _seed_plan(artifacts)
        artifacts.stamp_plan_provenance('orch-sess-1')

        _record(artifacts)

        assert artifacts.read_plan()[RESCOPED_BY_KEY] == 'orch-sess-1'

    def test_rescoped_by_is_a_str_marker_with_no_session_in_hand(self, artifacts):
        _seed_plan(artifacts)
        assert '_session_id' not in artifacts.read_plan()
        assert artifacts.read_agent_session() is None

        _record(artifacts)

        assert artifacts.read_plan()[RESCOPED_BY_KEY] == RESCOPED_BY_UNKNOWN

    def test_a_prefix_of_a_recorded_commit_upserts_the_same_entry(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts, commit='309c22ab6f' + '0' * 30)

        assert _record(artifacts, commit='309c22ab6f')['descoped_steps'] == 1

    def test_out_of_vocabulary_disposition_is_refused_without_a_write(self, artifacts):
        _seed_plan(artifacts)
        before = _plan_bytes(artifacts)

        result = _record(artifacts, disposition='delete')

        assert result['status'] == 'error'
        assert result['field'] == 'disposition'
        assert result['allowed_dispositions'] == sorted(DESCOPE_DISPOSITIONS)
        for disposition in DESCOPE_DISPOSITIONS:
            assert disposition in result['message']
        assert _plan_bytes(artifacts) == before

    @pytest.mark.parametrize('field', ['authority', 'note'])
    def test_blank_field_is_refused_without_a_write(self, artifacts, field):
        _seed_plan(artifacts)
        before = _plan_bytes(artifacts)

        result = _record(artifacts, **{field: '  '})

        assert result['status'] == 'error'
        assert result['field'] == field
        assert _plan_bytes(artifacts) == before

    def test_recording_the_same_pair_twice_upserts_one_entry(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts)

        result = _record(artifacts, disposition='ride-along', note='second thoughts')

        assert result['descoped_steps'] == 1
        plan = artifacts.read_plan()
        [entry] = plan[DESCOPED_STEPS_KEY]
        assert entry['disposition'] == 'ride-along'
        assert entry['note'] == 'second thoughts'
        assert plan[RESCOPED_NOTE_KEY] == 'second thoughts'

    def test_a_different_commit_for_the_same_step_is_a_second_entry(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts)

        assert _record(artifacts, commit='c0609049a7')['descoped_steps'] == 2

    def test_no_plan_is_refused(self, artifacts):
        assert _record(artifacts)['status'] == 'error'
        assert artifacts.read_plan() == {}


class TestCreatePlanCarriesTheRecordForward:
    def test_a_wholesale_overwrite_keeps_the_four_keys_and_resets_steps(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts)
        carried = {key: artifacts.read_plan()[key] for key in DESCOPE_PLAN_KEYS}

        assert _create_plan(artifacts, 'test-1', 'Re-plan', 'A2', ['b.py'])['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['steps'] == []
        assert {key: plan[key] for key in DESCOPE_PLAN_KEYS} == carried

    def test_a_plan_without_a_record_gains_none(self, artifacts):
        _seed_plan(artifacts)

        _create_plan(artifacts, 'test-1', 'Re-plan', 'A2', ['b.py'])

        plan = artifacts.read_plan()
        assert not set(DESCOPE_PLAN_KEYS) & set(plan)


class TestStampingLeavesTheRecordAlone:
    def test_provenance_and_revalidation_stamps_keep_the_four_keys(self, artifacts):
        _seed_plan(artifacts)
        _record(artifacts)
        before = json.dumps({key: artifacts.read_plan()[key] for key in DESCOPE_PLAN_KEYS})

        artifacts.stamp_plan_provenance('orch-sess-2', task_text_fp='tfp:abc')
        artifacts.bump_revalidation_stamp('orch-sess-3')

        plan = artifacts.read_plan()
        assert json.dumps({key: plan[key] for key in DESCOPE_PLAN_KEYS}) == before


@pytest.mark.asyncio
async def test_the_tool_is_registered(artifacts):
    _seed_plan(artifacts)
    server = plan_tools.create_server(artifacts)
    tool = await server.get_tool('record_descoped_step')
    assert tool is not None

    result = await tool.run({
        'step_id': 'step-1',
        'commit': '309c22ab6f',
        'disposition': 'revert',
        'authority': 'human:leo',
        'note': 'n',
    })

    assert result.structured_content is not None
    assert result.structured_content['status'] == 'ok'
    [entry] = artifacts.read_plan()[DESCOPED_STEPS_KEY]
    assert entry['disposition'] == 'revert'
