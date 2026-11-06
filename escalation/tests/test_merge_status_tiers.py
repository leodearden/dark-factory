"""merge_status's tier walk, end to end through ``create_server``.

PRD ``plans/merge-status-durable-non-landed-prd.md`` D5 order — live ->
event store (current run) -> git -> journal -> event store (cross run) ->
no_record — and boundary rows B1-B8.  The response model itself is pinned in
``test_merge_status_answer.py``; this module pins which tier answers.

Real collaborators throughout: ONE ``runs.db`` shared by two ``EventStore``
instances (``run-prior`` writes, the server reads as ``run-current``: a
simulated restart) and a real ``MergeQueueStore`` over a tmp journal file
mounted as ``harness._merge_store``.  The git tier is driven through the seam
I5 says merge_status consumes: the public ``git_authority.probe_landing``
verdict.
"""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from _merge_tool_calls import call_merge_cancel, call_merge_status
from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    GroupMergeRequest,
    InFlightMergeRegistry,
    MergeCancelRefusal,
    MergeCancelVerdict,
    MergeLane,
    MergeRequest,
    OperatorMergeCancel,
    QueuedBranch,
    WaiterRecord,
    enqueue_merge_request,
)
from orchestrator.merge_lane.queue_store import (
    MergeQueueStore,
)
from shared.merge_state import CANCEL_STATES, OUTCOME_STATES, MergeReason, MergeState

from escalation import git_authority
from escalation.git_authority import GitAuthorityArm, GitAuthorityOutcome, GitAuthorityVerdict
from escalation.merge_status_answer import hint_for
from escalation.queue import EscalationQueue
from escalation.server import create_server

pytestmark = pytest.mark.asyncio

_PRIOR = 'run-prior'
_CURRENT = 'run-current'
_TASK = '42'
_BRANCH = 'task/42'
_MERGE_SHA = 'c' * 40
_PRESENT_TRUTH = {'live', 'event_store', 'git'}


def _assert_d2(response: dict[str, Any]) -> None:
    if response['state'] in OUTCOME_STATES:
        assert response['source'] in _PRESENT_TRUTH, (
            f'D2/I1: outcome state {response["state"]!r} from source '
            f'{response["source"]!r}: {response}'
        )


async def _poll(server: Any, **key: Any) -> dict[str, Any]:
    response = await call_merge_status(server, **key)
    _assert_d2(response)
    return response


def _event_stores(tmp_path: Path) -> tuple[Any, Any]:
    db = tmp_path / 'runs.db'
    return EventStore(db, _PRIOR), EventStore(db, _CURRENT)


def _finalize(
    store: Any,
    *,
    request_id: str,
    state: str,
    branch: str = _TASK,
    task_id: str = _TASK,
    **extra: Any,
) -> None:
    store.emit(
        EventType.merge_finalized,
        task_id=task_id,
        data={'request_id': request_id, 'branch': branch, 'state': state, **extra},
    )


def _journal_entry(
    request_id: str,
    *,
    branch: str = _TASK,
    task_id: str = _TASK,
    enqueued_at: float = 1_700_000_000.0,
) -> dict[str, Any]:
    return {
        'request_id': request_id,
        'task_id': task_id,
        'branch': branch,
        'worktree': '/tmp/wt',
        'pre_rebased': False,
        'task_files': None,
        'snapshot_tip': None,
        'generation': 1,
        'lane': 'default',
        'enqueued_at': enqueued_at,
    }


def _journal(
    tmp_path: Path, *entries: dict[str, Any], store_type: type = MergeQueueStore,
) -> Any:
    path = tmp_path / 'merge_queue.json'
    path.write_text(json.dumps({e['request_id']: e for e in entries}))
    return store_type(path)


class _UnreadableJournal(MergeQueueStore):
    def load(self) -> list[Any]:
        raise OSError('journal unreadable')


def _live_worker(*entries: dict[str, Any]) -> types.SimpleNamespace:
    return types.SimpleNamespace(snapshot=lambda: {'entries': list(entries)})


def _live_entry(
    request_id: str, *, state: str = 'queued', waiter_alive: bool = True,
) -> dict[str, Any]:
    return {
        'request_id': request_id,
        'branch': _TASK,
        'task_id': _TASK,
        'state': state,
        'position': 0,
        'enqueued_at': 1_700_000_100.0,
        'waiter_alive': waiter_alive,
    }


def _git_ops_spies() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        is_ancestor=AsyncMock(return_value=False),
        find_merge_marker=AsyncMock(return_value=None),
        resolve_branch_sha=AsyncMock(return_value=None),
    )


def _config(tmp_path: Path) -> Any:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _server(
    tmp_path: Path,
    current: Any,
    *,
    worker: Any = None,
    journal: Any = None,
    git_ops: Any = None,
) -> Any:
    harness = types.SimpleNamespace(
        _merge_worker=worker,
        _merge_store=journal if journal is not None else _journal(tmp_path),
        git_ops=git_ops if git_ops is not None else _git_ops_spies(),
    )
    return create_server(
        EscalationQueue(tmp_path / 'esc'),
        orch_config=_config(tmp_path),
        event_store=current,
        harness=harness,
    )


def _verdict(outcome: GitAuthorityOutcome) -> GitAuthorityVerdict:
    if outcome is GitAuthorityOutcome.found_on_main:
        return GitAuthorityVerdict(
            outcome=outcome, merge_sha=_MERGE_SHA, arm=GitAuthorityArm.ancestor,
        )
    return GitAuthorityVerdict(outcome=outcome)


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=_verdict(GitAuthorityOutcome.no_signal))
    monkeypatch.setattr(git_authority, 'probe_landing', mock)
    return mock


class TestHistoryNeverSetsAnOutcomeState:
    async def test_b1_prior_run_blocked_is_a_stale_record(self, tmp_path: Path) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='blocked')

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] != 'blocked'
        assert response['state'] is MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['source'] == 'event_store_cross_run'
        assert response['hint'] == hint_for(MergeReason.prior_run_terminal)
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert last['run_id'] == _PRIOR

    async def test_b2_live_entry_answers_and_history_is_labelled(
        self, tmp_path: Path
    ) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='conflict')
        server = _server(tmp_path, current, worker=_live_worker(_live_entry('mr-new')))

        response = await _poll(server, branch=_BRANCH)

        assert response['state'] == 'queued'
        assert response['source'] == 'live'
        assert response['request_id'] == 'mr-new'
        last = response['last_outcome']
        assert last['state'] == 'conflict'
        assert last['run_id'] == _PRIOR
        assert last['is_current_run'] is False

    async def test_current_run_blocked_keeps_its_free_text_reason(
        self, tmp_path: Path
    ) -> None:
        _, current = _event_stores(tmp_path)
        _finalize(current, request_id='mr-1', state='blocked', reason='wip on main checkout')

        response = await _poll(_server(tmp_path, current), request_id='mr-1')

        assert response['state'] is MergeState.blocked
        assert response['source'] == 'event_store'
        assert response['reason'] == 'wip on main checkout'
        assert response['outcome'] == 'blocked'
        assert isinstance(response['finished_at'], str)
        assert 'hint' not in response
        assert response['last_outcome']['is_current_run'] is True
        assert response['last_outcome']['run_id'] == _CURRENT


class TestJournalTier:
    @pytest.mark.parametrize(
        'key',
        [{'request_id': 'mr-j'}, {'branch': _BRANCH}, {'task_id': _TASK}],
        ids=['request_id', 'branch', 'task_id'],
    )
    async def test_b3_journal_only_is_journaled(
        self, tmp_path: Path, key: dict[str, str]
    ) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(tmp_path, _journal_entry('mr-j', enqueued_at=1_700_000_500.0))

        response = await _poll(_server(tmp_path, current, journal=journal), **key)

        assert response['state'] is MergeState.journaled
        assert response['reason'] == MergeReason.journaled_not_live
        assert response['source'] == 'journal'
        assert response['request_id'] == 'mr-j'
        assert response['enqueued_at'] == 1_700_000_500.0
        assert response['hint'] == hint_for(MergeReason.journaled_not_live)

    async def test_newest_journal_entry_wins(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(
            tmp_path,
            _journal_entry('mr-older', enqueued_at=1_700_000_000.0),
            _journal_entry('mr-newer', enqueued_at=1_700_000_900.0),
        )

        response = await _poll(_server(tmp_path, current, journal=journal), task_id=_TASK)

        assert response['request_id'] == 'mr-newer'

    async def test_an_unreadable_journal_is_a_journal_probe_failure(
        self, tmp_path: Path
    ) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(tmp_path, store_type=_UnreadableJournal)

        response = await _poll(_server(tmp_path, current, journal=journal), task_id=_TASK)

        assert response['state'] is MergeState.unknown
        assert response['reason'] == MergeReason.probe_failed
        assert response['source'] == 'journal'


class TestCleanMiss:
    async def test_b4_nothing_anywhere_is_no_record(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.no_record
        assert response['reason'] == MergeReason.never_submitted
        assert response['source'] == 'none'
        assert response['last_outcome'] is None
        assert response['hint'] == hint_for(MergeReason.never_submitted)

    async def test_landed_unconfirmed_alone_is_an_unattributable_no_record(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.landed_unconfirmed)
        _, current = _event_stores(tmp_path)

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.no_record
        assert response['reason'] == MergeReason.landing_unattributable
        assert response['source'] == 'none'
        assert response['hint'] == hint_for(MergeReason.landing_unattributable)

    async def test_landed_unconfirmed_beside_history_is_an_unattributable_stale_record(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.landed_unconfirmed)
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='blocked')

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.stale_record
        assert response['reason'] == MergeReason.landing_unattributable
        assert response['source'] == 'event_store_cross_run'


class TestGitOutranksCheapTiers:
    async def test_found_on_main_beats_a_prior_run_blocked_row(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.found_on_main)
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='blocked')

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.done
        assert response['source'] == 'git'
        assert response['kind'] == 'found_on_main'
        assert response['merge_sha'] == _MERGE_SHA
        assert response['last_outcome']['state'] == 'blocked'
        assert response['last_outcome']['is_current_run'] is False

    async def test_found_on_main_beats_a_journal_entry(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.found_on_main)
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='blocked')
        journal = _journal(tmp_path, _journal_entry('mr-j'))

        response = await _poll(_server(tmp_path, current, journal=journal), branch=_BRANCH)

        assert response['state'] is MergeState.done
        assert response['source'] == 'git'

    async def test_found_on_main_beside_a_prior_run_done_labels_the_history(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.found_on_main)
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='done')

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.done
        assert response['source'] == 'git'
        assert response['last_outcome']['state'] == 'done'
        assert response['last_outcome']['is_current_run'] is False

    async def test_i5_merge_status_makes_no_landing_call_of_its_own(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.found_on_main)
        _, current = _event_stores(tmp_path)
        git_ops = _git_ops_spies()

        response = await _poll(_server(tmp_path, current, git_ops=git_ops), task_id=_TASK)

        assert response['state'] is MergeState.done
        git_ops.is_ancestor.assert_not_awaited()
        git_ops.find_merge_marker.assert_not_awaited()
        git_ops.resolve_branch_sha.assert_not_awaited()


class TestProbeFailureEndsTheWalk:
    async def test_b8_git_probe_failure_is_unknown_not_stale(
        self, tmp_path: Path, probe: AsyncMock
    ) -> None:
        probe.return_value = _verdict(GitAuthorityOutcome.probe_failed)
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='blocked')

        response = await _poll(_server(tmp_path, current), task_id=_TASK)

        assert response['state'] is MergeState.unknown
        assert response['reason'] == MergeReason.probe_failed
        assert response['source'] == 'git'
        assert response['hint'] == hint_for(MergeReason.probe_failed)
        assert response['last_outcome']['run_id'] == _PRIOR

    async def test_snapshot_failure_is_a_live_probe_failure(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        worker = types.SimpleNamespace(snapshot=Mock(side_effect=RuntimeError('boom')))

        response = await _poll(_server(tmp_path, current, worker=worker), request_id='mr-1')

        assert response['state'] is MergeState.unknown
        assert response['reason'] == MergeReason.probe_failed
        assert response['source'] == 'live'


class TestAliasResolution:
    async def test_b6_a_train_id_resolves_to_its_newest_member_record(
        self, tmp_path: Path
    ) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(
            prior, request_id='mr-a', state='superseded', branch='1', task_id='1',
            superseded_by='coalesce-x-1',
        )
        _finalize(
            prior, request_id='mr-b', state='superseded', branch='2', task_id='2',
            superseded_by='coalesce-x-1',
        )

        response = await _poll(_server(tmp_path, current), request_id='coalesce-x-1')

        assert response['state'] is MergeState.stale_record
        assert response['reason'] == MergeReason.coalesce_train_id
        assert response['request_id'] == 'coalesce-x-1'
        last = response['last_outcome']
        assert last['state'] == 'superseded'
        assert last['superseded_by'] == 'coalesce-x-1'
        assert last['request_id'] == 'mr-b'
        assert last['branch'] == '2'

    async def test_b7_a_prior_run_absorbed_loser_resolves_to_its_primary(
        self, tmp_path: Path
    ) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(
            prior, request_id='mr-primary', state='done',
            absorbed_request_ids=['mr-loser'],
        )

        response = await _poll(_server(tmp_path, current), request_id='mr-loser')

        assert response['state'] is MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['last_outcome']['request_id'] == 'mr-primary'
        assert response['last_outcome']['is_current_run'] is False

    async def test_b7_a_current_run_absorbed_loser_takes_its_primarys_outcome(
        self, tmp_path: Path
    ) -> None:
        _, current = _event_stores(tmp_path)
        _finalize(
            current, request_id='mr-primary', state='conflict',
            absorbed_request_ids=['mr-loser'],
        )

        response = await _poll(_server(tmp_path, current), request_id='mr-loser')

        assert response['state'] is MergeState.conflict
        assert response['source'] == 'event_store'
        assert response['request_id'] == 'mr-primary'
        assert response['last_outcome']['request_id'] == 'mr-primary'


@pytest.mark.parametrize('branch', [_TASK, _BRANCH], ids=['bare', 'full_ref'])
class TestBranchKeySpelling:
    """Each non-git tier answers a branch poll whether or not it carries the prefix."""

    async def test_the_live_tier(self, tmp_path: Path, branch: str) -> None:
        _, current = _event_stores(tmp_path)
        server = _server(tmp_path, current, worker=_live_worker(_live_entry('mr-live')))

        response = await _poll(server, branch=branch)

        assert response['state'] == 'queued'
        assert response['source'] == 'live'
        assert response['request_id'] == 'mr-live'

    async def test_the_current_run_tier(self, tmp_path: Path, branch: str) -> None:
        _, current = _event_stores(tmp_path)
        _finalize(current, request_id='mr-1', state='blocked')

        response = await _poll(_server(tmp_path, current), branch=branch)

        assert response['state'] is MergeState.blocked
        assert response['source'] == 'event_store'
        assert response['request_id'] == 'mr-1'

    async def test_the_journal_tier(self, tmp_path: Path, branch: str) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(tmp_path, _journal_entry('mr-j'))

        response = await _poll(_server(tmp_path, current, journal=journal), branch=branch)

        assert response['state'] is MergeState.journaled
        assert response['source'] == 'journal'
        assert response['request_id'] == 'mr-j'

    async def test_the_prior_run_tier(self, tmp_path: Path, branch: str) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='conflict')

        response = await _poll(_server(tmp_path, current), branch=branch)

        assert response['state'] is MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['source'] == 'event_store_cross_run'
        assert response['last_outcome']['request_id'] == 'mr-old'


class TestMergeCancelDurableResolution:
    """merge_cancel with no live waiter reuses merge_status's durable resolution."""

    async def _cancel(self, server: Any, request_id: str) -> dict[str, Any]:
        response = await call_merge_cancel(server, request_id=request_id)
        assert response['cancelled'] is False
        assert response['state'] in CANCEL_STATES
        assert isinstance(response['reason'], str) and response['reason']
        return response

    async def test_never_submitted_is_no_record(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)

        response = await self._cancel(_server(tmp_path, current), 'mr-never')

        assert response['state'] is MergeState.no_record

    async def test_finalized_in_a_prior_run_is_a_stale_record(self, tmp_path: Path) -> None:
        prior, current = _event_stores(tmp_path)
        _finalize(prior, request_id='mr-old', state='conflict')

        response = await self._cancel(_server(tmp_path, current), 'mr-old')

        assert response['state'] is MergeState.stale_record

    async def test_journal_only_is_journaled(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(tmp_path, _journal_entry('mr-j'))

        response = await self._cancel(_server(tmp_path, current, journal=journal), 'mr-j')

        assert response['state'] is MergeState.journaled

    async def test_finalized_this_run_keeps_its_outcome(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        _finalize(current, request_id='mr-1', state='conflict')

        response = await self._cancel(_server(tmp_path, current), 'mr-1')

        assert response['state'] is MergeState.conflict
        assert response['reason'] == 'Request already finalized; cannot cancel.'

    async def test_an_unreadable_journal_is_unknown(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        journal = _journal(tmp_path, store_type=_UnreadableJournal)

        response = await self._cancel(_server(tmp_path, current, journal=journal), 'mr-x')

        assert response['state'] is MergeState.unknown

    async def test_a_live_request_the_lane_refuses_reports_its_live_state(
        self, tmp_path: Path
    ) -> None:
        _, current = _event_stores(tmp_path)
        worker = _live_worker(_live_entry('mr-live', state='finalizing'))
        worker.cancel_request = lambda request_id: MergeCancelVerdict(
            request_id, MergeCancelRefusal.FINALIZE_WINDOW,
        )
        server = _server(
            tmp_path, current, worker=worker,
            journal=_journal(tmp_path, _journal_entry('mr-live')),
        )

        response = await self._cancel(server, 'mr-live')

        assert response['state'] is not MergeState.journaled
        assert response['state'] is MergeState.finalizing
        assert response['code'] == 'finalize_window'


class TestLiveTier:
    async def test_a_cancelled_snapshot_entry_is_not_live(self, tmp_path: Path) -> None:
        """A worker entry whose waiter was cancelled has no live outcome left.

        The merge_cancel half passes before the fix, because merge_cancel does
        not consult live yet. It guards the change that makes it do so: without
        the cancelled-entry filter, a double-cancel would read state 'queued'.
        """
        _, current = _event_stores(tmp_path)
        _finalize(current, request_id='mr-c', state='abandoned')
        server = _server(
            tmp_path, current, worker=_live_worker(_live_entry('mr-c', waiter_alive=False)),
        )

        polled = await _poll(server, request_id='mr-c')
        cancelled = await call_merge_cancel(server, request_id='mr-c')

        assert polled['state'] is MergeState.abandoned
        assert polled['source'] == 'event_store'
        assert cancelled['cancelled'] is False
        assert cancelled['state'] == 'abandoned'
        assert cancelled['reason'] == 'Request already finalized; cannot cancel.'


class TestMergeCancelThroughTheLane:
    """merge_cancel of a live request no caller is blocked on (task 5248).

    The request reaches a real lane the way a workflow's does, through
    ``enqueue_merge_request``, so this server holds no waiter for it.  Every
    operator cancel ends with someone owning it: the TaskWorkflow that awaits
    the request, or else an L1 this server files.
    """

    async def _lane_holding(
        self, tmp_path: Path, current: Any, request: MergeRequest,
        registry: InFlightMergeRegistry | None = None,
    ) -> Any:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = MergeLane(GitOps(_config(tmp_path).git, tmp_path), queue)
        await enqueue_merge_request(queue, request, current)
        git_ops = _git_ops_spies()
        git_ops.find_inflight_merge_worktree = AsyncMock(return_value=None)
        return create_server(
            EscalationQueue(tmp_path / 'esc'),
            orch_config=_config(tmp_path),
            event_store=current,
            harness=types.SimpleNamespace(
                _merge_worker=lane, _merge_store=_journal(tmp_path), git_ops=git_ops,
            ),
            merge_inflight_registry=registry,
        )

    def _request(
        self, tmp_path: Path, *, train: dict[str, Any] | None = None, **flags: Any,
    ) -> MergeRequest:
        fields: dict[str, Any] = dict(
            task_id=_TASK,
            branch=QueuedBranch.parse(_BRANCH, 'task/'),
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=_config(tmp_path),
            result=asyncio.get_running_loop().create_future(),
            **flags,
        )
        if train is None:
            return MergeRequest(**fields)
        return GroupMergeRequest(**fields, **train)

    def _escalations(self, tmp_path: Path) -> list[Any]:
        return EscalationQueue(tmp_path / 'esc').get_by_task(_TASK)

    async def test_a_queued_request_is_cancelled_and_then_polls_abandoned(
        self, tmp_path: Path,
    ) -> None:
        _, current = _event_stores(tmp_path)
        request = self._request(tmp_path, awaited_by_workflow=True)
        server = await self._lane_holding(tmp_path, current, request)
        queued = await _poll(server, request_id=request.request_id)

        response = await call_merge_cancel(server, request_id=request.request_id)

        assert queued['state'] is MergeState.queued
        assert response == {'cancelled': True, 'state': MergeState.abandoned, 'reason': None}
        assert (await _poll(server, request_id=request.request_id))['state'] is (
            MergeState.abandoned
        )
        assert self._escalations(tmp_path) == [], 'the awaiting workflow owns the cancel'

    async def test_a_workflowless_request_is_cancelled_and_escalated_here(
        self, tmp_path: Path,
    ) -> None:
        _, current = _event_stores(tmp_path)
        request = self._request(tmp_path)
        server = await self._lane_holding(tmp_path, current, request)

        response = await call_merge_cancel(server, request_id=request.request_id)

        assert response['cancelled'] is True
        [escalation] = self._escalations(tmp_path)
        assert escalation.category == 'merge_cancelled_by_operator'
        assert escalation.level == 1
        assert request.request_id in escalation.summary
        assert _TASK in escalation.summary

    async def test_an_attached_workflow_peer_receives_the_operator_cancel(
        self, tmp_path: Path,
    ) -> None:
        _, current = _event_stores(tmp_path)
        request = self._request(tmp_path)
        registry = InFlightMergeRegistry()
        registry.acquire(
            request.branch.bare_id, _TASK, request.result, request_id=request.request_id,
        )
        peer = asyncio.get_running_loop().create_future()
        registry.attach(request.branch.bare_id, WaiterRecord(
            request_id='mr-peer', future=peer, source='workflow',
        ))
        server = await self._lane_holding(tmp_path, current, request, registry)

        response = await call_merge_cancel(server, request_id=request.request_id)

        assert response['cancelled'] is True
        assert peer.cancelled()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            peer.result()
        assert OperatorMergeCancel.carried_by(cancelled.value) == (
            OperatorMergeCancel(request.request_id)
        )
        assert self._escalations(tmp_path) == [], 'the attached workflow owns the cancel'

    async def test_a_train_is_refused_with_its_code(self, tmp_path: Path) -> None:
        _, current = _event_stores(tmp_path)
        train = self._request(tmp_path, train=dict(
            train_id='train-1',
            member_task_ids=['41', _TASK],
            tip_branch=QueuedBranch.parse(_BRANCH, 'task/'),
            tip_task_id=_TASK,
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        ))
        server = await self._lane_holding(tmp_path, current, train)

        response = await call_merge_cancel(server, request_id=train.request_id)

        assert response['cancelled'] is False
        assert response['state'] is MergeState.queued
        assert response['code'] == 'train_cancel_unsupported'
        assert 'Block the member' in response['reason']
        assert not train.result.done()
        assert self._escalations(tmp_path) == []
