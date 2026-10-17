"""merge_status's boundary suite, rows B1-B9, across a simulated restart.

The executable form of ``plans/merge-status-durable-non-landed-prd.md``'s
Contract (B+H) and Boundary-test sketch, rows B1-B9.  B10 lives in
``orchestrator/tests/test_merge_status_boundary_b10.py`` because its subject is
EventStore's cross-run read.

Real git, real producers, real ``escalation.git_authority.probe_landing``: the
prior run's state is written by the orchestrator's own finalize and journal
writers, and the simulated restart is a fresh run_id plus a fresh journal store
over the prior run's files.  The unit-scope tier tests are
``escalation/tests/test_merge_status_tiers.py``.
"""
from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _merge_tool_calls import call_merge_status
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    enqueue_merge_request,
)
from orchestrator.merge_lane.queue_store import MergeQueueStore
from shared.merge_state import EPISTEMIC_STATES, OUTCOME_STATES, MergeReason, MergeState

from escalation.git_authority import GitAuthorityOutcome
from escalation.merge_status_answer import hint_for
from escalation.queue import EscalationQueue
from escalation.server import create_server

pytestmark = pytest.mark.asyncio

_RUN_PRIOR = 'run-prior'
_RUN_CURRENT = 'run-current'

#: I1: the only tiers whose evidence may set an outcome state.
_PRESENT_TRUTH_SOURCES = frozenset({'live', 'event_store', 'git'})

_BRANCH_KEY_DEFECT = (
    'esc-4834-3, follow-up tkt_0RVWJQ8EMTQQ4JX0F81NN4237P: every producer writes '
    "branch as QueuedBranch.bare_id while merge_status matches the branch key "
    "literally, so the 'task/<id>' form the merge-queue, unblock and "
    'unblock-low-risk skills document answers no_record/never_submitted for a '
    "prior run's row."
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class _World:
    """One project on disk: a git repo plus the files a run leaves behind."""

    repo: Path
    config: OrchestratorConfig
    git_ops: GitOps
    db: Path
    journal_path: Path
    escalations: Path


@pytest.fixture
def world(tmp_path: Path) -> _World:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@test.com')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / 'README.md').write_text('boundary suite\n')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', 'initial commit')
    config = OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
    )
    return _World(
        repo=repo,
        config=config,
        git_ops=GitOps(config.git, config.project_root),
        db=tmp_path / 'runs.db',
        journal_path=tmp_path / 'merge_queue.json',
        escalations=tmp_path / 'escalations',
    )


def _unmerged_branch(world: _World, task_id: str) -> None:
    """task/<task_id> carries one commit main does not."""
    _git(world.repo, 'checkout', '-q', '-b', f'task/{task_id}', 'main')
    (world.repo / f'{task_id}.txt').write_text(f'work for {task_id}\n')
    _git(world.repo, 'add', '-A')
    _git(world.repo, 'commit', '-q', '-m', f'work for {task_id}')
    _git(world.repo, 'checkout', '-q', 'main')


def _request(world: _World, task_id: str) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, world.config.git.branch_prefix),
        worktree=world.repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=world.config,
        result=asyncio.get_running_loop().create_future(),
    )


async def _finalize_in_prior_run(
    world: _World, task_id: str, outcome: MergeOutcome,
) -> MergeRequest:
    """The prior run submits *task_id* and settles it; the real finalize callback writes the row."""
    req = _request(world, task_id)
    await enqueue_merge_request(asyncio.Queue(), req, EventStore(world.db, _RUN_PRIOR))
    req.result.set_result(outcome)
    await asyncio.sleep(0)
    return req


def _journal_in_prior_run(world: _World, task_id: str) -> MergeRequest:
    """The prior run journals an owned request and dies before it finalizes."""
    req = _request(world, task_id)
    MergeQueueStore(world.journal_path).record(req)
    return req


def _live_entry(req: MergeRequest, state: str) -> dict[str, Any]:
    """The keys merge_status reads from orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot's entries."""
    return {
        'task_id': req.task_id,
        'branch': req.branch.bare_id,
        'state': state,
        'enqueued_at': req.enqueued_at,
        'position': 0,
        'waiter_alive': not req.result.cancelled(),
        'request_id': req.request_id,
    }


def _restart(
    world: _World,
    *,
    live_entries: Sequence[Mapping[str, Any]] = (),
    git_ops: GitOps | None = None,
) -> Any:
    """The restarted process: a fresh run_id and a fresh journal store over the prior run's files."""
    worker = (
        SimpleNamespace(snapshot=lambda: {'entries': list(live_entries)})
        if live_entries else None
    )
    harness = SimpleNamespace(
        _merge_worker=worker,
        _merge_store=MergeQueueStore(world.journal_path),
        git_ops=world.git_ops if git_ops is None else git_ops,
    )
    return create_server(
        EscalationQueue(world.escalations),
        orch_config=world.config,
        event_store=EventStore(world.db, _RUN_CURRENT),
        harness=harness,
    )


def _assert_wire_contract(response: Mapping[str, Any]) -> None:
    state = response['state']
    reason = response.get('reason')
    last = response['last_outcome']
    assert state in frozenset(MergeState), f'I4: state is not a MergeState value: {response}'
    if state in OUTCOME_STATES:
        assert response['source'] in _PRESENT_TRUTH_SOURCES, (
            f'I1: outcome state from a history source: {response}'
        )
        assert 'hint' not in response, f'an outcome state carries no hint: {response}'
    if last is not None and last['is_current_run'] is False and response['source'] != 'git':
        # A git-proven state beside equal history is corroboration:
        # escalation/src/escalation/merge_status_answer.py (module docstring).
        assert last['state'] != state, f'I2: a prior-run row set the state: {response}'
    assert (state == MergeState.unknown) == (reason == MergeReason.probe_failed), (
        f'I3: unknown iff probe_failed: {response}'
    )
    if state in EPISTEMIC_STATES:
        assert reason in frozenset(MergeReason), f'epistemic state without a MergeReason: {response}'
        assert response['hint'] == hint_for(MergeReason(reason)), f'hint not derived from reason: {response}'
    if last is not None:
        assert isinstance(last['run_id'], str), f'last_outcome.run_id: {response}'
        assert isinstance(last['finished_at'], str), f'last_outcome.finished_at: {response}'
        assert isinstance(last['is_current_run'], bool), f'last_outcome.is_current_run: {response}'


async def _poll(server: Any, **key: str) -> dict[str, Any]:
    response = await call_merge_status(server, **key)
    _assert_wire_contract(response)
    return response


class TestB1PriorRunBlockedAfterRestart:
    """B1 is THE REGRESSION PIN — must not be waived: a prior run's blocked is never this run's state."""

    @pytest.mark.parametrize('key', ['task_id', 'request_id'])
    async def test_b1_a_prior_run_blocked_answers_stale_record_never_blocked(
        self, world: _World, key: str,
    ) -> None:
        _unmerged_branch(world, '101')
        prior = await _finalize_in_prior_run(
            world, '101', MergeOutcome('blocked', reason='post-merge verify failed'),
        )
        poll_key = {'task_id': '101', 'request_id': prior.request_id}[key]

        response = await _poll(_restart(world), **{key: poll_key})

        assert response['state'] != 'blocked'
        assert response['state'] not in OUTCOME_STATES
        assert response['state'] == MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['source'] == 'event_store_cross_run'
        assert response['hint'] == hint_for(MergeReason.prior_run_terminal)
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert last['run_id'] == _RUN_PRIOR
        assert last['request_id'] == prior.request_id


class TestB2CurrentRunOnlySetsState:
    async def test_b2_a_live_resubmission_sets_state_and_the_prior_conflict_stays_history(
        self, world: _World,
    ) -> None:
        await _finalize_in_prior_run(
            world, '102', MergeOutcome('conflict', reason='rebase conflict'),
        )
        resubmitted = _request(world, '102')
        MergeQueueStore(world.journal_path).record(resubmitted)
        server = _restart(world, live_entries=[_live_entry(resubmitted, 'queued')])

        response = await _poll(server, task_id='102')

        assert response['state'] != 'conflict'
        assert response['state'] == MergeState.queued
        assert response['source'] == 'live'
        assert response['request_id'] == resubmitted.request_id
        last = response['last_outcome']
        assert last['state'] == 'conflict'
        assert last['run_id'] == _RUN_PRIOR
        assert last['is_current_run'] is False


class TestB3JournaledNotYetRecovered:
    @pytest.mark.parametrize('key', ['task_id', 'request_id'])
    async def test_b3_a_journaled_request_with_no_live_entry_answers_journaled(
        self, world: _World, key: str,
    ) -> None:
        _unmerged_branch(world, '103')
        journaled = _journal_in_prior_run(world, '103')
        poll_key = {'task_id': '103', 'request_id': journaled.request_id}[key]

        response = await _poll(_restart(world), **{key: poll_key})

        assert response['state'] == MergeState.journaled
        assert response['source'] == 'journal'
        assert response['reason'] == MergeReason.journaled_not_live
        assert response['request_id'] == journaled.request_id
        assert response['last_outcome'] is None


class TestB4NeverSubmitted:
    async def test_b4_nothing_anywhere_answers_no_record_never_submitted(
        self, world: _World,
    ) -> None:
        response = await _poll(_restart(world), task_id='104')

        assert response['state'] == MergeState.no_record
        assert response['reason'] == MergeReason.never_submitted
        assert response['source'] == 'none'
        assert response['last_outcome'] is None
        assert response['hint'] == hint_for(MergeReason.never_submitted)


class TestB5RebaseLandingIsConsumedFromGit:
    async def test_b5_a_rebase_landing_answers_done_from_git_with_no_landing_call_of_its_own(
        self, world: _World, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_tip = _rebase_landing(world, '105')
        await _finalize_in_prior_run(
            world, '105', MergeOutcome('blocked', reason='post-merge verify failed'),
        )
        assert not _git_succeeds(world.repo, 'merge-base', '--is-ancestor', original_tip, 'main')
        assert _git(world.repo, 'show', 'main:105.txt') == 'work for 105'
        assert not _git_succeeds(world.repo, 'rev-parse', '--verify', '--quiet', 'refs/heads/task/105')
        spy = _LandingPrimitiveSpy.install(monkeypatch, world.git_ops)

        response = await _poll(_restart(world), task_id='105')

        assert response['state'] == MergeState.done
        assert response['kind'] == 'found_on_main'
        assert response['source'] == 'git'
        (verdict,) = spy.verdicts
        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert response['merge_sha'] == verdict.merge_sha
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert spy.calls_outside_probe == []
        assert spy.calls_inside_probe


@pytest.mark.xfail(strict=True, reason=_BRANCH_KEY_DEFECT)
async def test_branch_key_in_the_skills_full_ref_form_resolves_a_prior_run_row(
    world: _World,
) -> None:
    _unmerged_branch(world, '110')
    prior = await _finalize_in_prior_run(
        world, '110', MergeOutcome('blocked', reason='post-merge verify failed'),
    )

    response = await _poll(_restart(world), branch='task/110')

    assert response['state'] == MergeState.stale_record
    assert response['last_outcome'] is not None
    assert response['last_outcome']['request_id'] == prior.request_id
