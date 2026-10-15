"""Every plan-tools mutator holds the plan mutex for its whole read-modify-write.

The parametrization of ``test_a_mutator_waits_for_the_mutex_and_both_writes_survive``
is the census of plan.json writers on this server: a new mutator belongs in it.
Each case races the mutator, on a second thread, against a holder already inside
the mutex (``_plan_mutex_race``); both writes surviving is the property.
"""
from __future__ import annotations

import inspect
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _plan_mutex_race import HOLDER_MARKER_KEY, mark_holder, race_writer_against_held_mutex

from orchestrator.artifacts import PlanMutexNotHeld, TaskArtifacts
from orchestrator.mcp import plan_markup_stamp
from orchestrator.mcp.plan_tools import (
    _add_design_decision,
    _add_plan_step,
    _add_prerequisite,
    _add_reuse_item,
    _confirm_plan,
    _create_plan,
    _drop_plan_file,
    _mark_step_committed,
    _mark_step_done,
    _remove_plan_step,
    _replace_plan_step,
    _update_plan_metadata,
)

_SEED = RepoSeed(files=(('.gitignore', '.task/\n'), ('m.py', 'x = 1\n')), message='Initial')
_REJECTIONS = plan_markup_stamp.PLAN_MARKUP_REJECTIONS_KEY
_UNBLOCKED_WRITER_BOUND_SECS = 10.0


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
    repo = seed_repo(tmp_path / 'worktree', _SEED)
    artifacts = TaskArtifacts(repo)
    artifacts.init('test-1', 'Test task', 'A test')
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    _create_plan(artifacts, 'test-1', 'T', 'A', ['a.py', 'b.py'])
    _add_plan_step(artifacts, 'step-1', 'test', 'first')
    _add_plan_step(artifacts, 'step-2', 'impl', 'second')
    return _Lane(artifacts, head)


def _ids(plan: dict, collection: str = 'steps') -> list[str]:
    return [item['id'] for item in plan.get(collection, [])]


def _step(plan: dict, step_id: str) -> dict:
    return next(step for step in plan['steps'] if step['id'] == step_id)


def _holder_marked(plan: dict) -> bool:
    return plan.get(HOLDER_MARKER_KEY) is True


def _stamp_one_rejection(plan: dict) -> None:
    plan[_REJECTIONS] = plan_markup_stamp.block_of(plan_markup_stamp.build_event(
        {'tool': 'add_plan_step', 'param': 'description', 'outcome': 'rejected'},
    ))


def _one_rejection_carried(plan: dict) -> bool:
    return plan.get(_REJECTIONS, {}).get('count') == 1


@dataclass(frozen=True)
class _Case:
    mutate: Callable[[TaskArtifacts, str], dict[str, Any]]
    landed: Callable[[dict, str], bool]
    holder_edit: Callable[[dict], None] = mark_holder
    holder_landed: Callable[[dict], bool] = _holder_marked


_CENSUS = [
    pytest.param(_Case(
        mutate=lambda ta, head: _create_plan(ta, 'test-1', 'Re-plan', 'A2', ['a.py']),
        landed=lambda plan, head: plan['title'] == 'Re-plan',
        holder_edit=_stamp_one_rejection,
        holder_landed=_one_rejection_carried,
    ), id='create_plan'),
    pytest.param(_Case(
        mutate=lambda ta, head: _add_plan_step(ta, 'step-3', 'impl', 'third'),
        landed=lambda plan, head: _ids(plan) == ['step-1', 'step-2', 'step-3'],
    ), id='add_plan_step'),
    pytest.param(_Case(
        mutate=lambda ta, head: _add_prerequisite(ta, 'pre-1', 'a prerequisite'),
        landed=lambda plan, head: _ids(plan, 'prerequisites') == ['pre-1'],
    ), id='add_prerequisite'),
    pytest.param(_Case(
        mutate=lambda ta, head: _add_design_decision(ta, 'D', 'R'),
        landed=lambda plan, head: plan['design_decisions'] == [
            {'decision': 'D', 'rationale': 'R'},
        ],
    ), id='add_design_decision'),
    pytest.param(_Case(
        mutate=lambda ta, head: _add_reuse_item(ta, 'W', 'H', 'how'),
        landed=lambda plan, head: plan['reuse'] == [{'what': 'W', 'where': 'H', 'how': 'how'}],
    ), id='add_reuse_item'),
    pytest.param(_Case(
        mutate=lambda ta, head: _drop_plan_file(ta, 'b.py', 'reason'),
        landed=lambda plan, head: (
            plan['files'] == ['a.py']
            and [drop['path'] for drop in plan['dropped_files']] == ['b.py']
        ),
    ), id='drop_plan_file'),
    pytest.param(_Case(
        mutate=lambda ta, head: _mark_step_done(ta, 'step-1', head),
        landed=lambda plan, head: (
            _step(plan, 'step-1')['status'] == 'done' and _step(plan, 'step-1')['commit'] == head
        ),
    ), id='mark_step_done'),
    pytest.param(_Case(
        mutate=lambda ta, head: _update_plan_metadata(ta, analysis='new'),
        landed=lambda plan, head: plan['analysis'] == 'new',
    ), id='update_plan_metadata'),
    pytest.param(_Case(
        mutate=lambda ta, head: _remove_plan_step(ta, 'step-2'),
        landed=lambda plan, head: _ids(plan) == ['step-1'],
    ), id='remove_plan_step'),
    pytest.param(_Case(
        mutate=lambda ta, head: _replace_plan_step(ta, 'step-2', 'impl', 'new'),
        landed=lambda plan, head: _step(plan, 'step-2')['description'] == 'new',
    ), id='replace_plan_step'),
    pytest.param(_Case(
        mutate=lambda ta, head: _confirm_plan(ta),
        landed=lambda plan, head: '_finalized_at' in plan,
    ), id='confirm_plan'),
    pytest.param(_Case(
        mutate=lambda ta, head: _mark_step_committed(ta, 'step-1', head),
        landed=lambda plan, head: (
            _step(plan, 'step-1')['status'] == 'done'
            and _step(plan, 'step-1')['description'] == f'[COMMITTED {head[:12]}] first'
        ),
    ), id='mark_step_committed'),
]


@pytest.mark.parametrize('case', _CENSUS)
def test_a_mutator_waits_for_the_mutex_and_both_writes_survive(lane: _Lane, case: _Case) -> None:
    envelopes: list[dict[str, Any]] = []

    outcome = race_writer_against_held_mutex(
        lane.artifacts,
        lambda: envelopes.append(case.mutate(lane.artifacts, lane.head)),
        holder_edit=case.holder_edit,
    )

    assert outcome.blocked, 'the mutator did not wait for the held plan mutex'
    assert outcome.writer_error is None
    assert envelopes[0].get('status') != 'error', envelopes[0]
    assert case.holder_landed(outcome.plan), f'the holder write was lost: {outcome.plan}'
    assert case.landed(outcome.plan, lane.head), f'the mutator write was lost: {outcome.plan}'


def test_mark_step_done_validates_against_the_plan_it_writes(lane: _Lane) -> None:
    """A step removed while the call waits is reported missing, never acknowledged as done."""
    envelopes: list[dict[str, Any]] = []

    def remove_step_1(plan: dict) -> None:
        plan['steps'] = [step for step in plan['steps'] if step['id'] != 'step-1']

    outcome = race_writer_against_held_mutex(
        lane.artifacts,
        lambda: envelopes.append(_mark_step_done(lane.artifacts, 'step-1', lane.head)),
        holder_edit=remove_step_1,
    )

    assert outcome.writer_error is None
    assert envelopes[0]['status'] == 'error', envelopes[0]
    assert _ids(outcome.plan) == ['step-2']


def test_a_mutator_without_its_mutex_refuses_to_write(lane: _Lane) -> None:
    """A forgotten ``_plan_mutation`` fails loudly at the write, not as a lost update."""
    plan_path = lane.artifacts.root / 'plan.json'
    before = plan_path.read_bytes()
    undecorated = inspect.unwrap(_add_plan_step)

    with pytest.raises(PlanMutexNotHeld):
        undecorated(lane.artifacts, 'step-3', 'impl', 'third')

    assert plan_path.read_bytes() == before


def test_mark_step_committed_refuses_an_off_branch_sha_without_waiting_for_the_mutex(
    lane: _Lane,
) -> None:
    """The reachability probe reads no plan, so no plan writer waits on its git subprocess."""
    envelopes: list[dict[str, Any]] = []

    outcome = race_writer_against_held_mutex(
        lane.artifacts,
        lambda: envelopes.append(_mark_step_committed(lane.artifacts, 'step-1', 'f' * 40)),
        probe_secs=_UNBLOCKED_WRITER_BOUND_SECS,
    )

    assert not outcome.blocked, 'the off-branch refusal waited for the held plan mutex'
    assert outcome.writer_error is None
    assert envelopes[0]['ok'] is False, envelopes[0]
    assert _holder_marked(outcome.plan)
    assert _step(outcome.plan, 'step-1')['status'] != 'done'
