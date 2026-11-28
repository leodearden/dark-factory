"""The plan-tools ``supersede_design_decision`` tool (task 3865).

A recorded design decision whose decision or rationale proves wrong is
rewritten in place, its prior pair kept under ``superseded_versions``. A
sibling of ``test_plan_tools_descope.py``; the transition's own invariants are
driven in ``test_plan_decisions.py``.
"""
from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.mcp.plan_tools import (
    _add_design_decision,
    _add_plan_step,
    _confirm_plan,
    _create_plan,
    _supersede_design_decision,
)
from orchestrator.plan_decisions import SUPERSEDED_VERSIONS_KEY

_FIRST = ('Key the sweep on text', 'Titles are stable')
_SECOND = ('Page the store', 'It is large')
_NEW_DECISION = 'Stream the store'
_NEW_RATIONALE = 'Paging was measured to be slow'
_REASON = 'the paging cost was never measured'


@pytest.fixture()
def artifacts(tmp_path):
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def _seed_plan(artifacts: TaskArtifacts) -> None:
    _create_plan(artifacts, 'test-1', 'Test task', 'Analysis', ['mod_a/foo.py'])
    _add_plan_step(artifacts, 'step-1', 'test', 'Write test for foo')
    _add_design_decision(artifacts, *_FIRST)
    _add_design_decision(artifacts, *_SECOND)


def _supersede(artifacts: TaskArtifacts, decision_index: int = 1, **overrides) -> dict:
    arguments = {
        'decision': _NEW_DECISION,
        'rationale': _NEW_RATIONALE,
        'reason': _REASON,
        **overrides,
    }
    return _supersede_design_decision(artifacts, decision_index, **arguments)


def _plan_bytes(artifacts: TaskArtifacts) -> bytes:
    return artifacts.artifact_path('plan.json').read_bytes()


class TestSupersedeDesignDecision:
    def test_rewrites_the_entry_and_keeps_its_prior_pair(self, artifacts):
        _seed_plan(artifacts)

        result = _supersede(artifacts, 1)

        assert result == {'status': 'ok', 'decision_index': 1, 'superseded_versions': 1}
        first, second = artifacts.read_plan()['design_decisions']
        assert first == {'decision': _FIRST[0], 'rationale': _FIRST[1]}
        assert second == {
            'decision': _NEW_DECISION,
            'rationale': _NEW_RATIONALE,
            SUPERSEDED_VERSIONS_KEY: [
                {'decision': _SECOND[0], 'rationale': _SECOND[1], 'reason': _REASON},
            ],
        }

    def test_a_second_supersession_reports_two_versions(self, artifacts):
        _seed_plan(artifacts)
        _supersede(artifacts, 1)

        result = _supersede(
            artifacts, 1, decision='Cache the store', reason='streaming was measured to be slow',
        )

        assert result['superseded_versions'] == 2
        [_, entry] = artifacts.read_plan()['design_decisions']
        assert [v['decision'] for v in entry[SUPERSEDED_VERSIONS_KEY]] == [_SECOND[0], _NEW_DECISION]

    @pytest.mark.parametrize(('decision_index', 'overrides', 'field'), [
        pytest.param(1, {'reason': '   '}, 'reason', id='blank-reason'),
        pytest.param(2, {}, 'decision_index', id='out-of-range-index'),
        pytest.param(
            1, {'decision': _SECOND[0], 'rationale': _SECOND[1]}, 'decision', id='identical-restatement',
        ),
    ])
    def test_a_refusal_writes_nothing(self, artifacts, decision_index, overrides, field):
        _seed_plan(artifacts)
        before = _plan_bytes(artifacts)

        result = _supersede(artifacts, decision_index, **overrides)

        assert result['status'] == 'error'
        assert result['field'] == field
        assert field in result['message']
        assert _plan_bytes(artifacts) == before

    def test_no_plan_is_an_error(self, artifacts):
        result = _supersede(artifacts, 0)

        assert result['status'] == 'error'
        assert 'No plan exists' in result['message']

    def test_a_confirmed_plan_stays_confirmed(self, artifacts):
        _seed_plan(artifacts)
        assert _confirm_plan(artifacts)['status'] == 'ok'

        assert _supersede(artifacts, 0)['status'] == 'ok'

        assert '_finalized_at' in artifacts.read_plan()


def test_add_design_decision_reports_the_new_entry_s_index(artifacts):
    _create_plan(artifacts, 'test-1', 'Test task', 'Analysis', ['mod_a/foo.py'])

    first = _add_design_decision(artifacts, *_FIRST)
    second = _add_design_decision(artifacts, *_SECOND)

    assert (first['decision_index'], first['total_decisions']) == (0, 1)
    assert (second['decision_index'], second['total_decisions']) == (1, 2)


@pytest.mark.asyncio
async def test_the_tool_is_registered(artifacts):
    _seed_plan(artifacts)
    server = plan_tools.create_server(artifacts)
    tool = await server.get_tool('supersede_design_decision')
    assert tool is not None

    result = await tool.run({
        'decision_index': 0,
        'decision': _FIRST[0],
        'rationale': 'Titles were measured to be stable for a year',
        'reason': 'the stability claim was unmeasured',
    })

    assert result.structured_content is not None
    assert result.structured_content['status'] == 'ok'
    entry = artifacts.read_plan()['design_decisions'][0]
    assert entry['rationale'] == 'Titles were measured to be stable for a year'
    assert entry[SUPERSEDED_VERSIONS_KEY][0]['rationale'] == _FIRST[1]
