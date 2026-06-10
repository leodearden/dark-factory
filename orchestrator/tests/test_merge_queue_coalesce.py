"""Tests for SpeculativeMergeWorker retroactive coalescing pass (task γ/1719).

Each TestClass maps to one plan step:
  step-1:  TestConfigAndEventType   — new config knob + new EventType
  step-3:  TestNoOpGuards           — early-return guards of _maybe_coalesce_waiting_singles
  step-5:  TestCoreFormation        — happy-path coalesce: 3 disjoint singles → 1 train
  step-7:  TestExclusionIdempotency — in-flight, detached, and GroupMergeRequest exclusions
  step-9:  TestPartialStackability  — overlap, stack-conflict eject, survivors<2
  step-11: TestDebounce             — signature-based deduplication
  step-13: TestEndToEndWiring       — merger-loop wiring: coalesced train dispatched end-to-end
"""

from __future__ import annotations

import asyncio
import collections
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

# ─── Step 1 ─────────────────────────────────────────────────────────────────

class TestConfigAndEventType:
    """step-1 (RED): new config knob and EventType member exist with correct defaults."""

    def test_merge_train_coalesce_enabled_default_false(self):
        """OrchestratorConfig.merge_train_coalesce_enabled defaults to False (OFF by default,
        human-flips after soak — fold-the-decision norm).
        """
        from orchestrator.config import GitConfig, OrchestratorConfig
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OrchestratorConfig(
                project_root=Path(tmp),
                git=GitConfig(
                    main_branch='main',
                    branch_prefix='task/',
                    remote='origin',
                    worktree_dir='.worktrees',
                ),
            )
            assert cfg.merge_train_coalesce_enabled is False

    def test_event_type_train_coalesced(self):
        """EventType.train_coalesced has value 'train_coalesced'."""
        from orchestrator.event_store import EventType
        assert EventType.train_coalesced == 'train_coalesced'


# ─── helpers shared across steps ────────────────────────────────────────────

def _make_config(tmp_path: Path, *, coalesce_enabled: bool = False) -> 'OrchestratorConfig':
    from orchestrator.config import GitConfig, OrchestratorConfig
    return OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
        merge_train_coalesce_enabled=coalesce_enabled,
    )


def _make_single_req(
    task_id: str,
    *,
    config: 'OrchestratorConfig',
    worktree: Path | None = None,
    lane: Literal['normal', 'high'] = 'normal',
) -> 'MergeRequest':
    from orchestrator.merge_queue import MergeRequest
    loop = asyncio.get_event_loop()
    try:
        future = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=f'task/{task_id}',
        worktree=worktree or Path(f'/tmp/wt-{task_id}'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
    )


def _stub_factory() -> 'TrainCallbackFactory':
    """Return a simple TrainCallbackFactory stub."""
    from orchestrator.merge_queue import TrainCallbacks
    def _factory(train_id: str) -> 'TrainCallbacks':
        return TrainCallbacks(
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )
    return _factory


def _make_worker(
    queue: 'asyncio.Queue',
    config: 'OrchestratorConfig',
    git_ops: 'GitOps | None' = None,
    factory=None,
) -> 'SpeculativeMergeWorker':
    from orchestrator.merge_queue import SpeculativeMergeWorker
    from orchestrator.git_ops import GitOps, GitConfig
    if git_ops is None:
        # stub git_ops — not used in no-op guard tests
        git_ops = MagicMock()
        git_ops.config = config
    worker = SpeculativeMergeWorker(
        git_ops,
        queue,
        train_callback_factory=factory,
    )
    # Wire config onto the worker for the pass to read it.
    worker._config = config
    return worker


# ─── Step 3 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNoOpGuards:
    """step-3 (RED): early-return guards of _maybe_coalesce_waiting_singles."""

    async def test_knob_disabled_returns_false_no_mutation(self, tmp_path: Path):
        """(a) merge_train_coalesce_enabled=False → returns False, buffer unchanged."""
        from orchestrator.merge_queue import GroupMergeRequest
        cfg = _make_config(tmp_path, coalesce_enabled=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=_stub_factory())

        req1 = _make_single_req('t1', config=cfg)
        req2 = _make_single_req('t2', config=cfg)
        worker._lane_buffers['normal'].append(req1)
        worker._lane_buffers['normal'].append(req2)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False, 'should return False when knob is disabled'
        buf = list(worker._lane_buffers['normal'])
        assert buf == [req1, req2], 'buffer must be unchanged'
        # No futures resolved
        assert not req1.result.done()
        assert not req2.result.done()

    async def test_single_candidate_returns_false(self, tmp_path: Path):
        """(b) knob enabled, only 1 candidate → returns False (need ≥2)."""
        cfg = _make_config(tmp_path, coalesce_enabled=True)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=_stub_factory())

        req1 = _make_single_req('t1', config=cfg)
        worker._lane_buffers['normal'].append(req1)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False
        assert list(worker._lane_buffers['normal']) == [req1]
        assert not req1.result.done()

    async def test_no_factory_returns_false(self, tmp_path: Path):
        """(c) knob enabled but _train_callback_factory is None → returns False."""
        cfg = _make_config(tmp_path, coalesce_enabled=True)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=None)  # no factory

        req1 = _make_single_req('t1', config=cfg)
        req2 = _make_single_req('t2', config=cfg)
        worker._lane_buffers['normal'].append(req1)
        worker._lane_buffers['normal'].append(req2)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False
        assert not req1.result.done()
        assert not req2.result.done()


# ─── git repo fixtures ───────────────────────────────────────────────────────

async def _setup_repo(repo: Path) -> None:
    from orchestrator.git_ops import _run
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config():
    from orchestrator.config import GitConfig
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config, git_repo: Path):
    from orchestrator.git_ops import GitOps
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config) -> 'OrchestratorConfig':
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(project_root=git_repo, git=git_config)


@pytest.fixture
def coalesce_config(git_repo: Path, git_config) -> 'OrchestratorConfig':
    """Config with merge_train_coalesce_enabled=True."""
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        merge_train_coalesce_enabled=True,
    )


async def _make_branch_with_file(
    git_ops: 'GitOps',
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    from orchestrator.git_ops import _run
    await _run(['git', 'add', '-A'], cwd=worktree)
    from orchestrator.git_ops import GitOps
    # Use the git_ops.commit method if available, else manual
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: 'OrchestratorConfig',
) -> 'MergeRequest':
    from orchestrator.merge_queue import MergeRequest
    future = asyncio.get_running_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane='normal',
    )


def _events_of_type(db_path: Path, event_type: str) -> list[dict]:
    """Return all events of the given type from the event store."""
    import json, sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT event_type, task_id, data FROM events WHERE event_type = ?",
        (event_type,),
    ).fetchall()
    conn.close()
    return [
        {'event_type': r[0], 'task_id': r[1], 'data': json.loads(r[2])}
        for r in rows
    ]


# ─── Step 5 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCoreFormation:
    """step-5 (RED): 3 disjoint-file singles → one GroupMergeRequest."""

    async def test_three_disjoint_singles_coalesce(
        self,
        git_ops: 'GitOps',
        coalesce_config: 'OrchestratorConfig',
        tmp_path: Path,
    ):
        """3 waiting singles with disjoint files coalesce into one GroupMergeRequest.

        Assertions:
          (a) returns True
          (b) normal buffer has exactly 1 item: a GroupMergeRequest; 3 singles gone
          (c) GroupMergeRequest fields correct: tip_task_id, member_task_ids (anchor first),
              status_check / mark_member_done from factory
          (d) all 3 futures resolve MergeOutcome(status='superseded', superseded_by=train_id)
          (e) EventType.train_coalesced event emitted with train_id, absorbed_request_ids,
              member_task_ids
        """
        from orchestrator.config import OrchestratorConfig
        from orchestrator.event_store import EventStore, EventType
        from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome, SpeculativeMergeWorker

        # Set up 3 branches each touching a unique file (disjoint → line-stackable).
        # Branch names are BARE (no prefix): create_worktree('t1') creates branch
        # 'task/t1' at worktree .worktrees/t1.  req.branch = 't1' (bare).
        wt1 = await _make_branch_with_file(git_ops, 't1', 'file_a.py', 'a = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 't2', 'file_b.py', 'b = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 't3', 'file_c.py', 'c = 3\n')

        # Event store to capture train_coalesced event.
        db_path = tmp_path / 'es_coalesce.db'
        es = EventStore(db_path=db_path, run_id='run-coalesce-test')

        # Stub factory: records calls with per-train marks.
        mark_done = AsyncMock()
        status_check = AsyncMock(return_value={'t1': 'merge-deferred', 't2': 'merge-deferred', 't3': 'merge-deferred'})
        from orchestrator.merge_queue import TrainCallbacks
        def factory(train_id: str) -> TrainCallbacks:
            return TrainCallbacks(
                status_check=status_check,
                mark_member_done=mark_done,
            )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=factory,
        )

        # Build 3 requests and populate the lane buffer.
        req1 = _make_req('t1', 'task/t1', wt1, coalesce_config)
        req2 = _make_req('t2', 'task/t2', wt2, coalesce_config)
        req3 = _make_req('t3', 'task/t3', wt3, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        # ── invoke ──────────────────────────────────────────────────────────
        result = await worker._maybe_coalesce_waiting_singles()

        # (a) returns True
        assert result is True, 'should return True when a train is formed'

        # (b) buffer has exactly 1 GroupMergeRequest; all 3 singles removed.
        buf = list(worker._lane_buffers['normal'])
        assert len(buf) == 1, f'Expected 1 item in buffer, got {len(buf)}: {buf}'
        group_req = buf[0]
        assert isinstance(group_req, GroupMergeRequest), (
            f'Expected GroupMergeRequest, got {type(group_req)}'
        )

        # (c) GroupMergeRequest fields.
        assert len(group_req.member_task_ids) == 3, (
            f'Expected 3 members, got: {group_req.member_task_ids}'
        )
        assert group_req.member_task_ids[0] == 't1', (
            f'Anchor (FIFO head) must be first; got {group_req.member_task_ids}'
        )
        assert group_req.tip_task_id == group_req.task_id, (
            'tip_task_id must equal task_id (tip req)'
        )
        assert group_req.train_id.startswith('coalesce-'), (
            f'train_id must start with "coalesce-"; got {group_req.train_id!r}'
        )

        # (d) all 3 original futures resolved 'superseded'.
        for req in (req1, req2, req3):
            assert req.result.done(), f'Future for {req.task_id} must be resolved'
            outcome: MergeOutcome = req.result.result()
            assert outcome.status == 'superseded', (
                f'Expected superseded for {req.task_id}, got {outcome.status!r}'
            )
            assert outcome.superseded_by == group_req.train_id, (
                f'superseded_by mismatch for {req.task_id}: '
                f'{outcome.superseded_by!r} != {group_req.train_id!r}'
            )

        # (e) train_coalesced event emitted.
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1, f'Expected 1 train_coalesced event, got {len(events)}'
        data = events[0]['data']
        assert data['train_id'] == group_req.train_id, 'event train_id mismatch'
        assert set(data['member_task_ids']) == {'t1', 't2', 't3'}, (
            f'event member_task_ids mismatch: {data["member_task_ids"]}'
        )
        assert set(data.get('absorbed_request_ids', [])) == {
            req1.request_id, req2.request_id, req3.request_id,
        }, f'absorbed_request_ids mismatch: {data.get("absorbed_request_ids")}'


# ─── Step 7 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestExclusionIdempotency:
    """step-7 (RED): in-flight, detached, and GroupMergeRequest exclusion rules."""

    async def test_inflight_request_never_absorbed(
        self,
        git_ops: 'GitOps',
        coalesce_config: 'OrchestratorConfig',
        tmp_path: Path,
    ):
        """(a) IN-FLIGHT NEVER ABSORBED.

        A request that lives in _inflight_req (not in _lane_buffers) must never
        be touched by the pass — its future remains unresolved and its task_id
        does not appear in the train's member_task_ids.
        """
        from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome, SpeculativeMergeWorker

        # Three disjoint-file branches so all are line-stackable.
        wt0 = await _make_branch_with_file(git_ops, 'in0', 'inflight.py', 'x = 0\n')
        wt1 = await _make_branch_with_file(git_ops, 'in1', 'file_x.py', 'x = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'in2', 'file_y.py', 'y = 2\n')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            train_callback_factory=_stub_factory(),
        )

        req0 = _make_req('in0', 'task/in0', wt0, coalesce_config)
        req1 = _make_req('in1', 'task/in1', wt1, coalesce_config)
        req2 = _make_req('in2', 'task/in2', wt2, coalesce_config)

        # Simulate req0 being in-flight (currently merging/verifying).
        # It is NOT in the lane buffer — that is the structural invariant.
        worker._inflight_req = req0

        # Only req1 and req2 are in the buffer.
        worker._lane_buffers['normal'].extend([req1, req2])

        result = await worker._maybe_coalesce_waiting_singles()

        # Pass should succeed (req1+req2 form a train).
        assert result is True, 'pass must coalesce req1+req2 into a train'

        # req0's future must still be unresolved — the pass must not touch it.
        assert not req0.result.done(), (
            'in-flight request future must remain unresolved after the pass'
        )

        # req0's task_id must not appear in the formed train's member_task_ids.
        buf = list(worker._lane_buffers['normal'])
        assert len(buf) == 1 and isinstance(buf[0], GroupMergeRequest)
        group_req = buf[0]
        assert 'in0' not in group_req.member_task_ids, (
            f'in-flight task must not be absorbed: member_task_ids={group_req.member_task_ids}'
        )

    async def test_cancelled_waiter_skipped_no_set_result(
        self,
        git_ops: 'GitOps',
        coalesce_config: 'OrchestratorConfig',
        tmp_path: Path,
    ):
        """(b) DETACHED WAITER SKIPPED WITHOUT set_result.

        A request whose future is already cancelled must not be absorbed and
        set_result must never be called on it — it stays cancelled (CancelledError),
        not overwritten with 'superseded'.  It also remains in the buffer.
        """
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'c1', 'alpha.py', 'a = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'c2', 'beta.py', 'b = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'c3', 'gamma.py', 'g = 3\n')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('c1', 'task/c1', wt1, coalesce_config)
        req2 = _make_req('c2', 'task/c2', wt2, coalesce_config)
        req3 = _make_req('c3', 'task/c3', wt3, coalesce_config)

        # Cancel req3 to simulate a detached waiter.
        req3.result.cancel()
        assert req3.result.cancelled(), 'precondition: req3 future is cancelled'

        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must coalesce the 2 live singles'

        # req3's future must NOT have been overwritten — it stays cancelled.
        assert req3.result.cancelled(), (
            'cancelled future must remain cancelled — set_result must not be called'
        )

        # req3 must still be in the buffer (it was not absorbed).
        buf = list(worker._lane_buffers['normal'])
        task_ids_in_buf = [
            r.task_id for r in buf if not isinstance(r, GroupMergeRequest)
        ]
        assert 'c3' in task_ids_in_buf, (
            f'cancelled request must remain in the buffer; buf task_ids={task_ids_in_buf}'
        )

        # Sanity: req3 is not in the formed train's member_task_ids.
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        assert len(trains) == 1
        assert 'c3' not in trains[0].member_task_ids

    async def test_group_merge_request_in_buffer_not_reabsorbed(
        self,
        git_ops: 'GitOps',
        coalesce_config: 'OrchestratorConfig',
        tmp_path: Path,
    ):
        """(c) IDEMPOTENCY.

        A GroupMergeRequest already sitting in the normal buffer must never be
        selected as a candidate — it is not re-absorbed.  Its future must remain
        unresolved after the pass.
        """
        from orchestrator.merge_queue import (
            GroupMergeRequest, SpeculativeMergeWorker, TrainCallbacks,
        )

        wt1 = await _make_branch_with_file(git_ops, 'g1', 'one.py', 'n = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'g2', 'two.py', 'n = 2\n')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('g1', 'task/g1', wt1, coalesce_config)
        req2 = _make_req('g2', 'task/g2', wt2, coalesce_config)

        # A pre-existing GroupMergeRequest sitting in the buffer (e.g. formed by
        # the β-former or a prior coalescing pass iteration).
        existing_group_future = asyncio.get_running_loop().create_future()
        existing_group = GroupMergeRequest(
            task_id='gx',
            branch='task/gx',
            worktree=tmp_path / 'wt-gx',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=coalesce_config,
            result=existing_group_future,
            train_id='train-existing',
            member_task_ids=['ga', 'gx'],
            tip_branch='task/gx',
            tip_task_id='gx',
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )

        # Put the existing group first, then the 2 singles.
        worker._lane_buffers['normal'].extend([existing_group, req1, req2])

        result = await worker._maybe_coalesce_waiting_singles()

        # (c1) The existing GroupMergeRequest's future must be unresolved.
        assert not existing_group_future.done(), (
            'pre-existing GroupMergeRequest future must not be resolved by the pass'
        )

        # (c2) The existing GroupMergeRequest must still be in the buffer.
        buf = list(worker._lane_buffers['normal'])
        group_reqs = [r for r in buf if isinstance(r, GroupMergeRequest)]
        group_train_ids = {gr.train_id for gr in group_reqs}
        assert 'train-existing' in group_train_ids, (
            f'pre-existing GroupMergeRequest must remain in buffer; '
            f'found train_ids: {group_train_ids}'
        )

        # (c3) The existing group was not in any newly formed train's member_task_ids.
        new_groups = [gr for gr in group_reqs if gr.train_id != 'train-existing']
        for ng in new_groups:
            assert 'gx' not in ng.member_task_ids, (
                f'pre-existing GroupMergeRequest task_id must not appear in new train: '
                f'{ng.member_task_ids}'
            )
