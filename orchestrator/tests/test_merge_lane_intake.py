"""The merge lane's request intake (PRD ``plans/merge-lane-quality-prd.md`` task ι).

``coalesce_or_enqueue_merge_request`` is the ``merge_request`` MCP tool's
door: it dispatches a request, coalesces it onto a live merge of the same
branch, or rejects it. Every case here drives that public function with a
real ``InFlightMergeRegistry``, a real ``TerminalOutcomeRetention`` and the
function's own injected seams (``git_ops``, ``classifier_git_ops``,
``live_snapshot``), and observes only what a caller can see: the returned
``MergeDispatchResult``, the queue, the registry, the futures, and the
events a recording store was handed.

The C3 rows already pinned with the same observations live in
``test_merge_queue_c3_submit_identity.py``: a newer tip while the old one is
in verify rejects (``TestC3SubmitGateInVerify``), and the queued SUBSET,
rebased-twin and force-push arms (``TestC3SubmitGateQueued``).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _git_fixtures import seed_repo

import orchestrator.merge_lane as facade
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import intake
from orchestrator.merge_lane.intake import (
    coalesce_or_enqueue_merge_request,
    register_and_enqueue_merge_request,
)
from orchestrator.merge_lane.types import (
    InFlightMergeRegistry,
    MergeDispatchResult,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    TerminalOutcomeRetention,
)


class RecordingEvents(EventStore):
    """An event store that keeps every emission in memory instead of on disk."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(db_path=tmp_path / 'intake-events.db', run_id='intake')
        self.emitted: list[tuple[EventType, str | None, dict]] = []

    def emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.emitted.append((event_type, task_id, dict(data or {})))

    def data_of(self, event_type: EventType) -> list[dict]:
        return [data for kind, _, data in self.emitted if kind is event_type]


class MergeWorktrees:
    """The disk-scan seam: one planted ``_merge-*`` worktree, or none.

    ``removes`` decides whether a reap actually takes the tree off disk, so
    a test can tell the reported ``removed`` flag from the reap's intent.
    """

    def __init__(self, worktree: Path | None, *, removes: bool = True) -> None:
        self.worktree = worktree
        self.removes = removes
        self.cleaned: list[Path] = []
        self.guarded_removals: list[tuple[Path, str]] = []

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:
        return self.worktree

    async def cleanup_merge_worktree(self, merge_wt: Path) -> None:
        self.cleaned.append(merge_wt)
        if self.removes:
            merge_wt.rmdir()

    async def remove_merge_worktree_guarded(self, path: Path, *, reason: str) -> str:
        self.guarded_removals.append((path, reason))
        path.rmdir()
        return 'removed'


class RivalWinsTheScan(MergeWorktrees):
    """A disk scan during which a rival submission claims the branch's slot."""

    def __init__(self, rival_submits: Callable[[], Awaitable[bool]]) -> None:
        super().__init__(None)
        self.rival_submits = rival_submits

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:
        await self.rival_submits()
        return None


class ClosedQueue(asyncio.Queue):
    """A queue that refuses every request, as a closed queue would."""

    async def put(self, item: Any) -> None:
        raise RuntimeError('queue closed')


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


@pytest.fixture
def events(tmp_path: Path) -> RecordingEvents:
    return RecordingEvents(tmp_path)


@pytest.fixture
def registry() -> InFlightMergeRegistry:
    return InFlightMergeRegistry()


@pytest.fixture
def retention() -> TerminalOutcomeRetention:
    return TerminalOutcomeRetention()


def merge_request(
    config: OrchestratorConfig,
    branch: str = '900',
    *,
    request_id: str | None = None,
    snapshot_tip: str | None = None,
) -> MergeRequest:
    identity: dict[str, Any] = {}
    if request_id is not None:
        identity['request_id'] = request_id
    return MergeRequest(
        task_id=f'task-{branch}',
        branch=QueuedBranch.parse(branch, 'task/'),
        worktree=config.project_root,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        snapshot_tip=snapshot_tip,
        **identity,
    )


def no_live_entries() -> dict:
    return {'entries': []}


def coalesced_data(branch: str, source: str, eta: int | None) -> dict:
    data: dict = {'branch': branch, 'source': source}
    if eta is not None:
        data['eta_seconds'] = eta
    return data


async def settle() -> None:
    """Let the futures' done-callbacks run."""
    for _ in range(3):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# (A) The façade seam
# ---------------------------------------------------------------------------


def test_the_facade_resolves_the_request_entry_to_intake() -> None:
    assert facade.coalesce_or_enqueue_merge_request is intake.coalesce_or_enqueue_merge_request
    assert facade.enqueue_merge_request is intake.enqueue_merge_request


# ---------------------------------------------------------------------------
# (B) Boundary row 4a: request-level coalescing through the façade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_request_for_a_branch_coalesces_onto_the_first(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    retention: TerminalOutcomeRetention,
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    first = merge_request(config)
    second = merge_request(config)

    dispatched = await facade.coalesce_or_enqueue_merge_request(
        queue, first, events, registry, retention=retention,
    )
    assert dispatched == MergeDispatchResult(dispatched=True, in_flight=False, branch='900')
    assert queue.get_nowait() is first

    coalesced = await facade.coalesce_or_enqueue_merge_request(
        queue, second, events, registry, retention=retention,
    )
    assert coalesced.in_flight is True
    assert coalesced.dispatched is False
    assert coalesced.source == 'registry'
    assert coalesced.inflight_request_id == first.request_id
    assert coalesced.inflight_task_id == first.task_id
    assert queue.empty()
    assert events.data_of(EventType.merge_coalesced) == [
        coalesced_data('900', 'registry', coalesced.eta_seconds),
    ]

    first.result.set_result(MergeOutcome('done', merge_sha='abc123'))
    await settle()

    for request_id in (first.request_id, second.request_id):
        record = retention.get(request_id)
        assert record is not None
        assert (record.request_id, record.state) == (first.request_id, 'done')
    assert not registry.is_inflight('900')


# ---------------------------------------------------------------------------
# (C) The decomposed gate, branch by branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slot_whose_request_left_the_worker_is_reaped_and_dispatched_fresh(
    config: OrchestratorConfig, events: RecordingEvents, registry: InFlightMergeRegistry,
) -> None:
    dead = merge_request(config, request_id='mr-dead')
    registry.acquire('900', dead.task_id, dead.result, request_id=dead.request_id)
    queue: asyncio.Queue = asyncio.Queue()
    fresh = merge_request(config, request_id='mr-fresh')

    result = await coalesce_or_enqueue_merge_request(
        queue, fresh, events, registry, live_snapshot=no_live_entries,
    )

    assert result == MergeDispatchResult(dispatched=True, in_flight=False, branch='900')
    assert dead.result.cancelled()
    entry = registry.entry('900')
    assert entry is not None and entry.request_id == 'mr-fresh'
    assert queue.get_nowait() is fresh
    assert events.data_of(EventType.merge_coalesced) == []


@pytest.mark.asyncio
async def test_a_live_merge_worktree_on_disk_coalesces_without_enqueueing(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / '_merge-live'
    worktree.mkdir()
    disk = MergeWorktrees(worktree)
    queue: asyncio.Queue = asyncio.Queue()

    result = await coalesce_or_enqueue_merge_request(
        queue, merge_request(config), events, registry, disk,
    )

    assert result == MergeDispatchResult(
        dispatched=False, in_flight=True, branch='900', source='worktree',
    )
    assert queue.empty()
    assert not registry.is_inflight('900')
    assert disk.cleaned == []
    assert events.data_of(EventType.merge_coalesced) == [coalesced_data('900', 'worktree', None)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('owned', 'liveness_secs', 'removes', 'reason'),
    [
        (True, 10800, True, 'finalized_owner'),
        (False, -1, True, 'stale_inflight'),
        (True, 10800, False, 'finalized_owner'),
    ],
    ids=['owner-finalized', 'cold-mtime', 'reap-skipped'],
)
async def test_a_corpse_merge_worktree_is_reaped_then_dispatched(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    tmp_path: Path,
    owned: bool,
    liveness_secs: int,
    removes: bool,
    reason: str,
) -> None:
    worktree = tmp_path / '_merge-corpse'
    worktree.mkdir()
    disk = MergeWorktrees(worktree, removes=removes)
    ledger = [str(worktree)] if owned else []
    queue: asyncio.Queue = asyncio.Queue()
    req = merge_request(config)

    result = await coalesce_or_enqueue_merge_request(
        queue, req, events, registry, disk,
        liveness_secs=liveness_secs,
        live_snapshot=lambda: {'entries': [], 'owned_merge_worktrees': ledger},
    )

    assert disk.cleaned == [worktree]
    assert events.data_of(EventType.worktree_reaped) == [{
        'branch': '900', 'path': str(worktree), 'reason': reason, 'removed': removes,
    }]
    assert result == MergeDispatchResult(dispatched=True, in_flight=False, branch='900')
    assert queue.get_nowait() is req


@pytest.mark.asyncio
async def test_a_rival_that_claims_the_slot_during_the_disk_scan_absorbs_the_request(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    retention: TerminalOutcomeRetention,
) -> None:
    rival = merge_request(config, request_id='mr-rival')
    late = merge_request(config, request_id='mr-late')
    queue: asyncio.Queue = asyncio.Queue()
    disk = RivalWinsTheScan(
        lambda: register_and_enqueue_merge_request(
            queue, rival, events, registry, retention=retention,
        ),
    )

    result = await coalesce_or_enqueue_merge_request(
        queue, late, events, registry, disk, retention=retention,
    )

    assert result.in_flight is True
    assert result.source == 'registry'
    assert result.inflight_request_id == 'mr-rival'
    assert queue.get_nowait() is rival
    assert queue.empty()
    assert events.data_of(EventType.merge_coalesced) == [
        coalesced_data('900', 'registry', result.eta_seconds),
    ]
    rival.result.set_result(MergeOutcome('done'))
    await settle()
    record = retention.get('mr-late')
    assert record is not None and record.request_id == 'mr-rival'


@pytest.mark.asyncio
async def test_a_dispatch_whose_enqueue_fails_frees_the_branch(
    config: OrchestratorConfig, events: RecordingEvents, registry: InFlightMergeRegistry,
) -> None:
    with pytest.raises(RuntimeError, match='queue closed'):
        await coalesce_or_enqueue_merge_request(
            ClosedQueue(), merge_request(config), events, registry,
        )
    assert not registry.is_inflight('900')


@pytest.mark.asyncio
async def test_a_workflow_enqueue_that_fails_frees_the_branch(
    config: OrchestratorConfig, events: RecordingEvents, registry: InFlightMergeRegistry,
) -> None:
    with pytest.raises(RuntimeError, match='queue closed'):
        await register_and_enqueue_merge_request(
            ClosedQueue(), merge_request(config), events, registry,
        )
    assert not registry.is_inflight('900')


# ---------------------------------------------------------------------------
# (C, continued) The C3 submit gate, classified against a real repository
# ---------------------------------------------------------------------------


async def _commit(repo: Path, name: str) -> str:
    (repo / f'{name}.txt').write_text(f'{name}\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', name], cwd=repo)
    _, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return out.strip()


@pytest_asyncio.fixture
async def history(tmp_path: Path) -> tuple[GitOps, str, str]:
    """A real repository with two commits, the second descending from the first."""
    repo = seed_repo(tmp_path / 'repo')
    older = await _commit(repo, 'older')
    newer = await _commit(repo, 'newer')
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, repo), older, newer


@pytest.mark.asyncio
async def test_a_newer_tip_replaces_a_queued_merge_and_reaps_its_scratch(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    history: tuple[GitOps, str, str],
    tmp_path: Path,
) -> None:
    classifier, older, newer = history
    queued = merge_request(config, request_id='mr-queued', snapshot_tip=older)
    registry.acquire(
        '900', queued.task_id, queued.result,
        request_id=queued.request_id, snapshot_tip=older,
    )
    scratch = tmp_path / '_merge-scratch'
    scratch.mkdir()
    disk = MergeWorktrees(scratch)
    queue: asyncio.Queue = asyncio.Queue()
    newer_req = merge_request(config, request_id='mr-newer', snapshot_tip=newer)

    result = await coalesce_or_enqueue_merge_request(
        queue, newer_req, events, registry, disk, classifier_git_ops=classifier,
    )

    assert result == MergeDispatchResult(dispatched=True, in_flight=False, branch='900')
    assert queued.result.cancelled()
    entry = registry.entry('900')
    assert entry is not None and entry.request_id == 'mr-newer'
    assert disk.guarded_removals == [(scratch, 'c3_replace_drop_queued')]
    assert events.data_of(EventType.worktree_reaped) == [{
        'branch': '900', 'path': str(scratch),
        'reason': 'c3_replace_drop_queued', 'outcome': 'removed',
    }]
    assert queue.get_nowait() is newer_req


@pytest.mark.asyncio
async def test_a_replacement_whose_enqueue_fails_frees_the_branch(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    history: tuple[GitOps, str, str],
) -> None:
    classifier, older, newer = history
    queued = merge_request(config, snapshot_tip=older)
    registry.acquire(
        '900', queued.task_id, queued.result,
        request_id=queued.request_id, snapshot_tip=older,
    )

    with pytest.raises(RuntimeError, match='queue closed'):
        await coalesce_or_enqueue_merge_request(
            ClosedQueue(), merge_request(config, snapshot_tip=newer), events, registry,
            classifier_git_ops=classifier,
        )
    assert not registry.is_inflight('900')


@pytest.mark.asyncio
async def test_the_registry_verifying_flag_alone_rejects_a_newer_tip(
    config: OrchestratorConfig,
    events: RecordingEvents,
    registry: InFlightMergeRegistry,
    history: tuple[GitOps, str, str],
) -> None:
    classifier, older, newer = history
    verifying = merge_request(config, request_id='mr-verifying', snapshot_tip=older)
    registry.acquire(
        '900', verifying.task_id, verifying.result,
        request_id=verifying.request_id, snapshot_tip=older,
    )
    registry.set_verifying('900')
    queue: asyncio.Queue = asyncio.Queue()

    result = await coalesce_or_enqueue_merge_request(
        queue, merge_request(config, snapshot_tip=newer), events, registry,
        classifier_git_ops=classifier,
    )

    assert result == MergeDispatchResult(
        dispatched=False,
        in_flight=False,
        branch='900',
        rejected=True,
        reject_code='duplicate_in_verify',
        inflight_task_id=verifying.task_id,
        inflight_request_id='mr-verifying',
        existing_sha=older,
        verify_age_secs=None,
    )
    assert not verifying.result.done()
    assert queue.empty()


# ---------------------------------------------------------------------------
# (D) Queue-to-land: each merge_finalized row times its own request
# ---------------------------------------------------------------------------

# queue_to_land_secs is rounded to the millisecond, so it may sit up to half
# a millisecond above the span the test measured around it.
_MS_ROUNDING = 0.0005


def queue_to_land_of(events: RecordingEvents, request_id: str) -> float:
    [row] = [
        data for data in events.data_of(EventType.merge_finalized)
        if data['request_id'] == request_id
    ]
    value = row['queue_to_land_secs']
    assert isinstance(value, float)
    return value


@pytest.mark.asyncio
async def test_a_landed_request_reports_its_queue_to_land_time(
    config: OrchestratorConfig, events: RecordingEvents,
) -> None:
    req = merge_request(config, request_id='mr-landed')
    before = time.monotonic()

    await intake.enqueue_merge_request(asyncio.Queue(), req, events)
    await asyncio.sleep(0.02)
    req.result.set_result(MergeOutcome(status='done', merge_sha='abc123'))
    await settle()
    span = time.monotonic() - before

    assert 0 <= queue_to_land_of(events, 'mr-landed') <= span + _MS_ROUNDING


@pytest.mark.asyncio
async def test_an_abandoned_request_still_reports_its_queue_time(
    config: OrchestratorConfig, events: RecordingEvents,
) -> None:
    req = merge_request(config, request_id='mr-abandoned')

    await intake.enqueue_merge_request(asyncio.Queue(), req, events)
    req.result.cancel()
    await settle()

    assert [data['state'] for data in events.data_of(EventType.merge_finalized)] == ['abandoned']
    assert queue_to_land_of(events, 'mr-abandoned') >= 0


@pytest.mark.asyncio
async def test_a_re_enqueued_task_times_each_request_from_its_own_enqueue(
    config: OrchestratorConfig, events: RecordingEvents,
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    first = merge_request(config, request_id='mr-first')
    await intake.enqueue_merge_request(queue, first, events)
    await asyncio.sleep(0.05)
    first.result.set_result(MergeOutcome(status='blocked', reason='verify failed'))
    await settle()

    second = merge_request(config, request_id='mr-second')
    await intake.enqueue_merge_request(queue, second, events)
    second.result.set_result(MergeOutcome(status='done', merge_sha='def456'))
    await settle()

    assert first.task_id == second.task_id
    assert queue_to_land_of(events, 'mr-second') < queue_to_land_of(events, 'mr-first')
