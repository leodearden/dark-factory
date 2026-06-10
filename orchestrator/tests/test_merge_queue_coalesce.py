"""Tests for SpeculativeMergeWorker retroactive coalescing pass (task γ/1719)
and merge-ready confidence gate (task δ/1720).

Each TestClass maps to one plan step:
  step-1:  TestConfigAndEventType        — new config knob + new EventType
  step-3:  TestNoOpGuards                — early-return guards of _maybe_coalesce_waiting_singles
  step-5:  TestCoreFormation             — happy-path coalesce: 3 disjoint singles → 1 train
  step-7:  TestExclusionIdempotency      — in-flight, detached, and GroupMergeRequest exclusions
  step-9:  TestPartialStackability       — overlap, stack-conflict eject, survivors<2
  step-11: TestDebounce                  — signature-based deduplication
  step-13: TestEndToEndWiring            — merger-loop wiring: coalesced train dispatched end-to-end

Task δ/1720 additions (merge-ready confidence gate):
  step-1:  TestMergeReadyPredicateSeam   — injectable predicate seam honored
  step-3:  TestBlockedHistoryGate        — event-store blocked-history excludes candidate
  step-5:  TestOneStrikeExclusion        — one-strike registry excludes candidate
  step-7:  TestOneStrikeRecording        — derailed coalesce train marks its members
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps
    from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker, TrainCallbackFactory

# ─── Step 1 ─────────────────────────────────────────────────────────────────

class TestConfigAndEventType:
    """step-1 (RED): new config knob and EventType member exist with correct defaults."""

    def test_merge_train_coalesce_enabled_default_false(self):
        """OrchestratorConfig.merge_train_coalesce_enabled defaults to False (OFF by default,
        human-flips after soak — fold-the-decision norm).
        """
        import tempfile

        from orchestrator.config import GitConfig, OrchestratorConfig
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

def _make_config(tmp_path: Path, *, coalesce_enabled: bool = False) -> OrchestratorConfig:
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
    config: OrchestratorConfig,
    worktree: Path | None = None,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    from orchestrator.merge_queue import MergeRequest
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


def _stub_factory() -> TrainCallbackFactory:
    """Return a simple TrainCallbackFactory stub."""
    from orchestrator.merge_queue import TrainCallbacks
    def _factory(train_id: str) -> TrainCallbacks:
        return TrainCallbacks(
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )
    return _factory


def _make_worker(
    queue: asyncio.Queue,
    config: OrchestratorConfig,
    git_ops: GitOps | None = None,
    factory=None,
) -> SpeculativeMergeWorker:
    from orchestrator.merge_queue import SpeculativeMergeWorker
    if git_ops is None:
        # stub git_ops — not used in no-op guard tests
        git_ops = MagicMock()
        git_ops.config = config
    worker = SpeculativeMergeWorker(
        git_ops,
        queue,
        train_callback_factory=factory,
    )
    return worker


# ─── Step 3 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNoOpGuards:
    """step-3 (RED): early-return guards of _maybe_coalesce_waiting_singles."""

    async def test_knob_disabled_returns_false_no_mutation(self, tmp_path: Path):
        """(a) merge_train_coalesce_enabled=False → returns False, buffer unchanged."""
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
def config(git_repo: Path, git_config) -> OrchestratorConfig:
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(project_root=git_repo, git=git_config)


@pytest.fixture
def coalesce_config(git_repo: Path, git_config) -> OrchestratorConfig:
    """Config with merge_train_coalesce_enabled=True."""
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        merge_train_coalesce_enabled=True,
    )


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    from orchestrator.git_ops import _run
    await _run(['git', 'add', '-A'], cwd=worktree)
    # Use the git_ops.commit method if available, else manual
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
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
    import json
    import sqlite3
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
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
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
        from orchestrator.event_store import EventStore
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
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """(a) IN-FLIGHT NEVER ABSORBED.

        A request that lives in _inflight_req (not in _lane_buffers) must never
        be touched by the pass — its future remains unresolved and its task_id
        does not appear in the train's member_task_ids.
        """
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

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
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
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
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """(c) IDEMPOTENCY.

        A GroupMergeRequest already sitting in the normal buffer must never be
        selected as a candidate — it is not re-absorbed.  Its future must remain
        unresolved after the pass.
        """
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            SpeculativeMergeWorker,
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

        await worker._maybe_coalesce_waiting_singles()

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


# ─── Step 9 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPartialStackability:
    """step-9 (RED): overlap eject, stack-conflict eject, survivors<2."""

    async def test_overlap_pair_stays_solo(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """(a) OVERLAP: 2 tasks editing overlapping lines + 1 disjoint task.

        The overlapping pair is not mutually stackable; only the anchor + the
        disjoint task form a viable train.  The non-stackable task stays solo
        in the buffer with its future unresolved.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.git_ops import TrainStackResult
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        db_path = tmp_path / 'es_overlap.db'
        es = EventStore(db_path=db_path, run_id='run-overlap')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('ov1', 'task/ov1', tmp_path / 'wt-ov1', coalesce_config)
        req2 = _make_req('ov2', 'task/ov2', tmp_path / 'wt-ov2', coalesce_config)
        req3 = _make_req('ov3', 'task/ov3', tmp_path / 'wt-ov3', coalesce_config)

        worker._lane_buffers['normal'].extend([req1, req2, req3])

        # ov1 and ov2 both touch file_a.py lines 1–5 / 3–7 → overlapping → NOT stackable.
        # ov3 touches file_b.py only → disjoint from both → stackable with ov1.
        async def _mock_gcr(task_id: str):
            return {
                'ov1': {'file_a.py': [(1, 5)]},
                'ov2': {'file_a.py': [(3, 7)]},
                'ov3': {'file_b.py': [(1, 3)]},
            }.get(task_id, {})

        # stack_train_branches([ov1, ov3]) succeeds (ov1=anchor, ov3 rebased onto ov1).
        async def _mock_stb(member_ids: list) -> TrainStackResult:
            return TrainStackResult(survivors=list(member_ids), ejected=[])

        with patch.object(git_ops, 'get_changed_line_ranges', side_effect=_mock_gcr), \
             patch.object(git_ops, 'stack_train_branches', side_effect=_mock_stb):
            result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must form a train from the 2 stackable tasks'

        buf = list(worker._lane_buffers['normal'])
        # ov2 must remain as a solo in the buffer (not absorbed).
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]
        assert len(solos) == 1, f'Expected 1 solo; got {[r.task_id for r in solos]}'
        assert solos[0].task_id == 'ov2', (
            f'ov2 must stay solo; solo task_id={solos[0].task_id}'
        )
        assert not req2.result.done(), 'ov2 future must remain unresolved (not absorbed)'

        # The train must contain only ov1 + ov3.
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        assert len(trains) == 1
        assert set(trains[0].member_task_ids) == {'ov1', 'ov3'}, (
            f'train members must be ov1+ov3; got {trains[0].member_task_ids}'
        )

    async def test_stack_conflict_eject(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """(b) STACK CONFLICT EJECT.

        stack_train_branches returns survivors=[m1, m2], ejected=[m3].  The train
        absorbs only the 2 survivors; m3 stays solo (future unresolved, in buffer);
        the train_coalesced event data includes the ejected list.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.git_ops import TrainStackResult
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        db_path = tmp_path / 'es_eject.db'
        es = EventStore(db_path=db_path, run_id='run-eject')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('ej1', 'task/ej1', tmp_path / 'wt-ej1', coalesce_config)
        req2 = _make_req('ej2', 'task/ej2', tmp_path / 'wt-ej2', coalesce_config)
        req3 = _make_req('ej3', 'task/ej3', tmp_path / 'wt-ej3', coalesce_config)

        worker._lane_buffers['normal'].extend([req1, req2, req3])

        # All disjoint → _select_train_members selects all 3.
        async def _mock_gcr(task_id: str):
            return {
                'ej1': {'fa.py': [(1, 2)]},
                'ej2': {'fb.py': [(1, 2)]},
                'ej3': {'fc.py': [(1, 2)]},
            }.get(task_id, {})

        # Stack ejects ej3 due to a simulated rebase conflict.
        async def _mock_stb(member_ids: list) -> TrainStackResult:
            survivors = [m for m in member_ids if m != 'ej3']
            ejected = ['ej3'] if 'ej3' in member_ids else []
            return TrainStackResult(survivors=survivors, ejected=ejected)

        with patch.object(git_ops, 'get_changed_line_ranges', side_effect=_mock_gcr), \
             patch.object(git_ops, 'stack_train_branches', side_effect=_mock_stb):
            result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must form a train from ej1+ej2 survivors'

        buf = list(worker._lane_buffers['normal'])
        # ej3 must remain in the buffer as a solo.
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]
        assert len(solos) == 1 and solos[0].task_id == 'ej3', (
            f'ej3 must stay solo; got {[r.task_id for r in solos]}'
        )
        assert not req3.result.done(), 'ejected ej3 future must remain unresolved'

        # Train must cover only ej1 + ej2.
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        assert len(trains) == 1
        assert set(trains[0].member_task_ids) == {'ej1', 'ej2'}, (
            f'train must absorb ej1+ej2; got {trains[0].member_task_ids}'
        )

        # train_coalesced event must list ej3 in ejected.
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1, f'expected 1 event; got {len(events)}'
        assert events[0]['data'].get('ejected') == ['ej3'], (
            f'event ejected mismatch: {events[0]["data"].get("ejected")}'
        )

    async def test_survivors_lt2_no_train(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """(c) SURVIVORS<2: stack ejects all but the anchor → no train formed.

        The pass returns False, no future is resolved, no event is emitted, and
        all candidates remain in the buffer.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.git_ops import TrainStackResult
        from orchestrator.merge_queue import SpeculativeMergeWorker

        db_path = tmp_path / 'es_nosurv.db'
        es = EventStore(db_path=db_path, run_id='run-nosurv')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('ns1', 'task/ns1', tmp_path / 'wt-ns1', coalesce_config)
        req2 = _make_req('ns2', 'task/ns2', tmp_path / 'wt-ns2', coalesce_config)
        req3 = _make_req('ns3', 'task/ns3', tmp_path / 'wt-ns3', coalesce_config)

        worker._lane_buffers['normal'].extend([req1, req2, req3])

        async def _mock_gcr(task_id: str):
            return {}  # all disjoint → all selected

        # Only the anchor survives — not enough for a train.
        async def _mock_stb(member_ids: list) -> TrainStackResult:
            return TrainStackResult(survivors=[member_ids[0]], ejected=list(member_ids[1:]))

        with patch.object(git_ops, 'get_changed_line_ranges', side_effect=_mock_gcr), \
             patch.object(git_ops, 'stack_train_branches', side_effect=_mock_stb):
            result = await worker._maybe_coalesce_waiting_singles()

        assert result is False, 'must return False when survivors < 2'

        # All 3 candidates must remain in the buffer unchanged.
        buf = list(worker._lane_buffers['normal'])
        buf_ids = [r.task_id for r in buf]
        assert set(buf_ids) == {'ns1', 'ns2', 'ns3'}, (
            f'all candidates must remain in buffer; got {buf_ids}'
        )

        # No futures resolved.
        for req in (req1, req2, req3):
            assert not req.result.done(), f'{req.task_id} future must remain unresolved'

        # No train_coalesced event emitted.
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 0, f'no event must be emitted; got {len(events)}'


# ─── Step 11 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDebounce:
    """step-11 (RED): signature-based deduplication prevents redundant stacking."""

    async def test_unchanged_set_skipped_on_second_call(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """Unchanged waiting set: 2nd call skips get_changed_line_ranges entirely.

        Two non-stackable singles are placed in the buffer (no train forms on the
        first call).  The signature is recorded.  The second call with the same
        buffer composition must short-circuit BEFORE calling get_changed_line_ranges.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('db1', 'task/db1', tmp_path / 'wt-db1', coalesce_config)
        req2 = _make_req('db2', 'task/db2', tmp_path / 'wt-db2', coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2])

        # Spy on get_changed_line_ranges (non-stackable: both touch the same file).
        call_count = 0

        async def _mock_gcr(task_id: str):
            nonlocal call_count
            call_count += 1
            return {'shared_file.py': [(1, 5)]}  # both overlap → no train

        async def _mock_stb(member_ids: list):
            from orchestrator.git_ops import TrainStackResult
            return TrainStackResult(survivors=list(member_ids), ejected=[])

        with patch.object(git_ops, 'get_changed_line_ranges', side_effect=_mock_gcr), \
             patch.object(git_ops, 'stack_train_branches', side_effect=_mock_stb):

            # First call: runs the selection (2 get_changed_line_ranges invocations).
            result1 = await worker._maybe_coalesce_waiting_singles()
            calls_after_first = call_count

            # Second call: same buffer composition → must short-circuit, 0 new calls.
            result2 = await worker._maybe_coalesce_waiting_singles()
            calls_after_second = call_count

        assert result1 is False, 'first call: no train (non-stackable pair)'
        assert result2 is False, 'second call: still no train'
        assert calls_after_first == 2, (
            f'first call must fetch ranges for both tasks; got {calls_after_first} calls'
        )
        assert calls_after_second == calls_after_first, (
            f'second call must short-circuit — 0 new get_changed_line_ranges calls; '
            f'got {calls_after_second - calls_after_first} extra'
        )

    async def test_new_request_rearms_debounce(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """Adding a new request re-arms the debounce (different signature → re-runs)."""
        from orchestrator.merge_queue import SpeculativeMergeWorker

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('ra1', 'task/ra1', tmp_path / 'wt-ra1', coalesce_config)
        req2 = _make_req('ra2', 'task/ra2', tmp_path / 'wt-ra2', coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2])

        call_count = 0

        async def _mock_gcr(task_id: str):
            nonlocal call_count
            call_count += 1
            return {'same_file.py': [(1, 5)]}  # non-stackable pair

        async def _mock_stb(member_ids: list):
            from orchestrator.git_ops import TrainStackResult
            return TrainStackResult(survivors=list(member_ids), ejected=[])

        with patch.object(git_ops, 'get_changed_line_ranges', side_effect=_mock_gcr), \
             patch.object(git_ops, 'stack_train_branches', side_effect=_mock_stb):

            # First call with [req1, req2]: records signature.
            await worker._maybe_coalesce_waiting_singles()
            calls_after_first = call_count

            # Second call SAME set: short-circuits.
            await worker._maybe_coalesce_waiting_singles()
            assert call_count == calls_after_first, 'same set must short-circuit'

            # Add a new request — different signature → must re-run.
            req3 = _make_req('ra3', 'task/ra3', tmp_path / 'wt-ra3', coalesce_config)
            worker._lane_buffers['normal'].append(req3)

            await worker._maybe_coalesce_waiting_singles()
            assert call_count > calls_after_first, (
                'new request must re-arm the debounce; get_changed_line_ranges must '
                'be called again'
            )

    async def test_successful_coalesce_updates_signature(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """After a successful coalesce the signature reflects the NEW composition.

        A subsequent call with an unchanged remaining set (just the GroupMergeRequest)
        is a no-op (only 0 regular-candidate singles → guard < 2 fires before
        the debounce check even matters).  The important property is that
        _last_coalesce_signature was set to the pre-coalesce candidate set, so a
        caller that somehow re-invokes without any new singles sees no extra stacking.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'sc1', 'aa.py', 'x = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'sc2', 'bb.py', 'y = 2\n')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('sc1', 'task/sc1', wt1, coalesce_config)
        req2 = _make_req('sc2', 'task/sc2', wt2, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2])

        # First call: successful coalesce (real git: disjoint files → stackable).
        result = await worker._maybe_coalesce_waiting_singles()
        assert result is True, 'precondition: first call must succeed'

        # After the coalesce the buffer holds a GroupMergeRequest; the absorbed
        # singles' futures are resolved (done=True) → 0 regular candidates.
        # A second call must return False immediately (guard < 2) and must NOT
        # have stale state that causes incorrect behaviour.
        sig_after_coalesce = worker._last_coalesce_signature

        result2 = await worker._maybe_coalesce_waiting_singles()
        assert result2 is False, 'second call with no candidates must return False'

        # Signature must have been set during the first call (non-None after a
        # successful coalesce).
        assert sig_after_coalesce is not None, '_last_coalesce_signature must be set'
        assert sig_after_coalesce == {req1.request_id, req2.request_id}, (
            'signature must capture the pre-coalesce request_ids of both singles'
        )


# ─── Step 13 ────────────────────────────────────────────────────────────────

def _gated_verify(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event | None = None,
):
    """Block the FIRST run_scoped_verification call until gate_release is set.

    gate_entered (optional): set when the first call starts, giving the test a
    synchronisation point.  Subsequent calls pass immediately (MagicMock passed=True).
    Local copy — test_merge_queue_coalesce must not import from test_merge_queue.
    """
    _first_blocked = [False]

    from unittest.mock import MagicMock as _MagicMock  # local alias for clarity

    async def _side_effect(*args, **kwargs):
        if not _first_blocked[0]:
            _first_blocked[0] = True
            if gate_entered is not None:
                gate_entered.set()
            await gate_release.wait()
        return _MagicMock(passed=True, summary='')

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.asyncio
class TestEndToEndWiring:
    """step-13 (RED): merger-loop wiring — coalesced train dispatched end-to-end.

    Three disjoint singles are pre-enqueued before the worker starts so the
    coalescing pass fires on the FIRST merger iteration (spec_base=None,
    prefetched=None).  A gated-verify blocks the TRAIN's first
    run_scoped_verification call, providing a deterministic sync point:

      • When gate_entered fires, coalescing has already happened (coalescing
        pass ran before _acquire_next_request; the 3 singles' futures are
        already resolved 'superseded') and the train's merge commit is on disk.
      • Releasing the gate completes verify → advance_main → mark_member_done.

    Asserts:
      (a) all 3 singles' futures are resolved 'superseded' before verify starts
      (b) 1 GroupMergeRequest was dispatched through _do_train_merge (train path)
      (c) all 3 member files appear on main after the train lands
      (d) mark_member_done called once per member, all with the same merge SHA
      (e) EventType.train_coalesced event emitted with correct member_task_ids
    """

    async def test_coalesced_train_dispatched_and_lands(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        import contextlib

        from orchestrator.event_store import EventStore
        from orchestrator.git_ops import _run
        from orchestrator.merge_queue import SpeculativeMergeWorker, TrainCallbacks

        # Build 3 disjoint-file branches off main.
        # Each touches a unique file → mutually line-stackable.
        wt1 = await _make_branch_with_file(git_ops, 'e2e1', 'file_e2e1.py', 'e1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'e2e2', 'file_e2e2.py', 'e2 = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'e2e3', 'file_e2e3.py', 'e3 = 3\n')

        db_path = tmp_path / 'e2e_coalesce.db'
        es = EventStore(db_path=db_path, run_id='e2e-coalesce')

        # Stub factory: status_check always returns all merge-deferred; mark_member_done
        # records (task_id, sha) and sets all_done when all 3 members are notified.
        mark_done_calls: list[tuple[str, str]] = []
        all_done = asyncio.Event()

        async def _mark_done(task_id: str, sha: str) -> None:
            mark_done_calls.append((task_id, sha))
            if len(mark_done_calls) >= 3:
                all_done.set()

        status_check = AsyncMock(return_value={
            'e2e1': 'merge-deferred',
            'e2e2': 'merge-deferred',
            'e2e3': 'merge-deferred',
        })

        def factory(train_id: str) -> TrainCallbacks:
            return TrainCallbacks(
                status_check=status_check,
                mark_member_done=AsyncMock(side_effect=_mark_done),
            )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es, train_callback_factory=factory,
        )

        # Build requests and pre-enqueue all 3 BEFORE starting the worker.
        # The coalescing pass fires on the first merger iteration (spec_base=None,
        # prefetched=None) and drains all 3 from the asyncio.Queue in one shot.
        # IMPORTANT: branch must be the BARE name (no 'task/' prefix) — merge_to_main
        # prepends branch_prefix itself; passing 'task/e2e1' would produce 'task/task/e2e1'.
        req1 = _make_req('e2e1', 'e2e1', wt1, coalesce_config)
        req2 = _make_req('e2e2', 'e2e2', wt2, coalesce_config)
        req3 = _make_req('e2e3', 'e2e3', wt3, coalesce_config)
        await queue.put(req1)
        await queue.put(req2)
        await queue.put(req3)

        # Gate the train's first run_scoped_verification call.
        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _gated_verify(gate_release, gate_entered),
        ):
            worker_task = asyncio.create_task(worker.run())

            # (a) Wait for the train's verify to start.
            # When gate_entered fires: _maybe_coalesce_waiting_singles has already
            # run (resolving the 3 singles as 'superseded') AND _do_train_merge has
            # run the rebase + merge steps before calling run_scoped_verification.
            await asyncio.wait_for(gate_entered.wait(), timeout=60)

            for req, label in ((req1, 'e2e1'), (req2, 'e2e2'), (req3, 'e2e3')):
                assert req.result.done(), (
                    f'{label}: future must be resolved as superseded before verify starts'
                )
                outcome = req.result.result()
                assert outcome.status == 'superseded', (
                    f'{label}: expected superseded, got {outcome.status!r}'
                )

            # Release the gate — train verify completes, advance_main runs.
            gate_release.set()

            # (d) Wait for all 3 mark_member_done calls (train has fully landed).
            await asyncio.wait_for(all_done.wait(), timeout=60)

        # (c) All 3 member files appear on main.
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'file_e2e1.py' in main_files, 'file_e2e1.py must be on main after train lands'
        assert 'file_e2e2.py' in main_files, 'file_e2e2.py must be on main after train lands'
        assert 'file_e2e3.py' in main_files, 'file_e2e3.py must be on main after train lands'

        # (d) mark_member_done called once per member with the same merge SHA.
        assert len(mark_done_calls) == 3, (
            f'expected 3 mark_member_done calls, got {len(mark_done_calls)}'
        )
        task_ids_notified = {tid for tid, _ in mark_done_calls}
        assert task_ids_notified == {'e2e1', 'e2e2', 'e2e3'}, (
            f'unexpected member task IDs: {task_ids_notified}'
        )
        merge_shas = {sha for _, sha in mark_done_calls}
        assert len(merge_shas) == 1, (
            f'all mark_member_done calls must share one SHA; got: {merge_shas}'
        )

        # (e) train_coalesced event emitted with correct member_task_ids.
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1, (
            f'expected 1 train_coalesced event, got {len(events)}'
        )
        event_data = events[0]['data']
        assert set(event_data.get('member_task_ids', [])) == {'e2e1', 'e2e2', 'e2e3'}, (
            f'train_coalesced event member_task_ids mismatch: {event_data}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ─── Task δ/1720 — merge-ready confidence gate ──────────────────────────────

# ─── δ Step 1 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestMergeReadyPredicateSeam:
    """δ step-1 (RED): injectable merge_ready_predicate seam is honored.

    Build 3 disjoint-file stackable singles, inject a predicate that vetoes t2
    with reason 'custom_reason'.  Asserts:
      (a) returns True (train forms with the 2 eligible members)
      (b) t2 NOT in train.member_task_ids; train == {p1, p3}
      (c) req2.result NOT done() — excluded request stays unresolved
      (d) t2 remains as a solo in the buffer
      (e) train_coalesced event data['exclusions'] == [{request_id: req2.request_id,
          reason: 'custom_reason'}]
    """

    async def test_predicate_excludes_one_member(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'p1', 'file_p1.py', 'p1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'p2', 'file_p2.py', 'p2 = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'p3', 'file_p3.py', 'p3 = 3\n')

        db_path = tmp_path / 'es_pred.db'
        es = EventStore(db_path=db_path, run_id='run-pred-seam')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
            merge_ready_predicate=lambda req: 'custom_reason' if req.task_id == 'p2' else None,
        )

        req1 = _make_req('p1', 'task/p1', wt1, coalesce_config)
        req2 = _make_req('p2', 'task/p2', wt2, coalesce_config)
        req3 = _make_req('p3', 'task/p3', wt3, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        # (a) returns True — a train was formed
        assert result is True, 'pass must form a train with the 2 eligible candidates'

        buf = list(worker._lane_buffers['normal'])
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]

        # (b) train contains exactly p1 + p3
        assert len(trains) == 1, f'Expected 1 train; got {len(trains)}'
        train = trains[0]
        assert 'p2' not in train.member_task_ids, (
            f'p2 must be excluded from train; members={train.member_task_ids}'
        )
        assert set(train.member_task_ids) == {'p1', 'p3'}, (
            f'train must contain p1+p3; got {train.member_task_ids}'
        )

        # (c) excluded req2 future remains unresolved
        assert not req2.result.done(), 'excluded req2 future must remain unresolved'

        # (d) p2 stays as a solo in the buffer
        solo_ids = [r.task_id for r in solos]
        assert 'p2' in solo_ids, f'excluded p2 must remain as solo; buf ids={solo_ids}'

        # (e) train_coalesced event carries the exclusion
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1
        exc = events[0]['data'].get('exclusions', [])
        assert len(exc) == 1, f'Expected 1 exclusion entry; got {exc}'
        assert exc[0] == {'request_id': req2.request_id, 'reason': 'custom_reason'}, (
            f'exclusion entry mismatch: {exc[0]}'
        )


# ─── δ Step 3 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBlockedHistoryGate:
    """δ step-3 (RED): event-store blocked-history excludes a candidate.

    Uses the DEFAULT predicate (no merge_ready_predicate injected).
    A synthetic prior terminal merge_finalized event with state='blocked'
    for t1's branch causes the default predicate to exclude t1.

    Sub-tests:
      (a) With prior blocked finalized event for t1's branch → t1 excluded,
          train forms with t2+t3, exclusions entry carried in event.
      (b) No merge_finalized emitted → all 3 coalesce, exclusions == [].
    """

    async def test_prior_blocked_excludes_candidate(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        from orchestrator.event_store import EventStore, EventType
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'bh1', 'file_bh1.py', 'bh1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'bh2', 'file_bh2.py', 'bh2 = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'bh3', 'file_bh3.py', 'bh3 = 3\n')

        db_path = tmp_path / 'es_bh.db'
        es = EventStore(db_path=db_path, run_id='run-bh-test')

        # Emit a prior blocked outcome for bh1's branch.
        es.emit(
            EventType.merge_finalized,
            task_id='bh1',
            phase='merge',
            role='merger',
            data={
                'request_id': 'mr-old-bh1',
                'branch': 'task/bh1',
                'state': 'blocked',
                'merge_sha': None,
            },
        )

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('bh1', 'task/bh1', wt1, coalesce_config)
        req2 = _make_req('bh2', 'task/bh2', wt2, coalesce_config)
        req3 = _make_req('bh3', 'task/bh3', wt3, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must form a train from the 2 clean candidates'

        buf = list(worker._lane_buffers['normal'])
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]

        # bh1 excluded — NOT in train
        assert len(trains) == 1
        train = trains[0]
        assert 'bh1' not in train.member_task_ids, (
            f'bh1 must be excluded (prior blocked); members={train.member_task_ids}'
        )
        assert set(train.member_task_ids) == {'bh2', 'bh3'}, (
            f'train must contain bh2+bh3; got {train.member_task_ids}'
        )

        # req1 future unresolved (bh1 excluded, not absorbed)
        assert not req1.result.done(), 'excluded bh1 future must remain unresolved'

        # bh1 stays as a solo in the buffer
        solo_ids = [r.task_id for r in solos]
        assert 'bh1' in solo_ids, f'bh1 must remain as solo; got {solo_ids}'

        # train_coalesced event carries the exclusion with state-based reason
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1
        exc = events[0]['data'].get('exclusions', [])
        assert len(exc) == 1, f'Expected 1 exclusion; got {exc}'
        assert exc[0]['request_id'] == req1.request_id
        # Reason must reference the blocked state
        assert 'blocked' in exc[0]['reason'], (
            f"reason must contain 'blocked'; got {exc[0]['reason']!r}"
        )

    async def test_no_history_all_coalesce(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """No merge_finalized emitted → default predicate returns None for all →
        all 3 coalesce normally; exclusions == [].
        """
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'nh1', 'file_nh1.py', 'nh1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'nh2', 'file_nh2.py', 'nh2 = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'nh3', 'file_nh3.py', 'nh3 = 3\n')

        db_path = tmp_path / 'es_nh.db'
        es = EventStore(db_path=db_path, run_id='run-nh-test')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        req1 = _make_req('nh1', 'task/nh1', wt1, coalesce_config)
        req2 = _make_req('nh2', 'task/nh2', wt2, coalesce_config)
        req3 = _make_req('nh3', 'task/nh3', wt3, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must form a train when no risk signals'

        buf = list(worker._lane_buffers['normal'])
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        assert len(trains) == 1
        assert set(trains[0].member_task_ids) == {'nh1', 'nh2', 'nh3'}

        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1
        assert events[0]['data'].get('exclusions') == [], (
            f'exclusions must be empty when no risk signals; got {events[0]["data"].get("exclusions")}'
        )


# ─── δ Step 5 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestOneStrikeExclusion:
    """δ step-5 (RED): one-strike registry excludes a candidate.

    Pre-populate worker._coalesce_derailed_task_ids = {'os2'} (simulating
    os2 having derailed a prior coalesce train). Run the pass with 3 disjoint
    stackable singles and DEFAULT predicate.

    Asserts:
      (a) returns True — train forms with the 2 clean members
      (b) os2 NOT in train.member_task_ids; train == {os1, os3}
      (c) req2.result NOT done() — excluded request stays unresolved
      (d) os2 remains as solo in the buffer
      (e) exclusions contains {'request_id': req2.request_id,
          'reason': 'coalesce_derailed_one_strike'}
    """

    async def test_one_strike_excludes_candidate(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'os1', 'file_os1.py', 'os1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'os2', 'file_os2.py', 'os2 = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'os3', 'file_os3.py', 'os3 = 3\n')

        db_path = tmp_path / 'es_os.db'
        es = EventStore(db_path=db_path, run_id='run-os-test')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        # Pre-populate the one-strike set (simulating a prior coalesce derail).
        worker._coalesce_derailed_task_ids = {'os2'}

        req1 = _make_req('os1', 'task/os1', wt1, coalesce_config)
        req2 = _make_req('os2', 'task/os2', wt2, coalesce_config)
        req3 = _make_req('os3', 'task/os3', wt3, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2, req3])

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is True, 'pass must form a train from the 2 clean candidates'

        buf = list(worker._lane_buffers['normal'])
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]

        # (b) os2 excluded — NOT in train
        assert len(trains) == 1
        train = trains[0]
        assert 'os2' not in train.member_task_ids, (
            f'os2 must be excluded (one-strike); members={train.member_task_ids}'
        )
        assert set(train.member_task_ids) == {'os1', 'os3'}, (
            f'train must contain os1+os3; got {train.member_task_ids}'
        )

        # (c) req2 future unresolved
        assert not req2.result.done(), 'excluded os2 future must remain unresolved'

        # (d) os2 stays as a solo
        solo_ids = [r.task_id for r in solos]
        assert 'os2' in solo_ids, f'os2 must remain as solo; got {solo_ids}'

        # (e) exclusions with one-strike reason
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 1
        exc = events[0]['data'].get('exclusions', [])
        assert len(exc) == 1, f'Expected 1 exclusion; got {exc}'
        assert exc[0] == {
            'request_id': req2.request_id,
            'reason': 'coalesce_derailed_one_strike',
        }, f'exclusion entry mismatch: {exc[0]}'


# ─── δ Step 7 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestOneStrikeRecording:
    """δ step-7 (RED): a derailed coalesce train marks its members one-strike.

    Drives the merger loop pattern from TestEndToEndWiring.  Patches
    _do_train_merge to return MergeOutcome('blocked').  A coalesce-prefixed
    train (train_id startswith 'coalesce-') must populate
    worker._coalesce_derailed_task_ids with all its member_task_ids.

    Negative control: a non-coalesce train (train_id does NOT start with
    'coalesce-') that derails must NOT populate the set.
    """

    async def test_coalesce_train_derail_marks_members(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """Coalesce-prefixed train derail → members marked one-strike."""
        import contextlib

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            MergeOutcome,
            SpeculativeMergeWorker,
        )

        db_path = tmp_path / 'es_osr.db'
        es = EventStore(db_path=db_path, run_id='run-osr-test')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        # Build a GroupMergeRequest with a coalesce-prefixed train_id.
        group_future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        group = GroupMergeRequest(
            task_id='m2',
            branch='task/m2',
            worktree=tmp_path / 'wt-m2',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=coalesce_config,
            result=group_future,
            train_id='coalesce-tipm-deadbeef',
            member_task_ids=['m1', 'm2'],
            tip_branch='task/m2',
            tip_task_id='m2',
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )

        # Patch _do_train_merge to return blocked (simulating verify failure).
        with patch(
            'orchestrator.merge_queue._do_train_merge',
            AsyncMock(return_value=MergeOutcome('blocked', reason='verify red')),
        ):
            await queue.put(group)
            worker_task = asyncio.create_task(worker.run())

            # Wait for the group future to be resolved by the verifier path.
            await asyncio.wait_for(group_future, timeout=30)

        assert worker._coalesce_derailed_task_ids == {'m1', 'm2'}, (
            f'coalesce train members must be marked one-strike after derail; '
            f'got {worker._coalesce_derailed_task_ids}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_non_coalesce_train_derail_does_not_mark(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """Non-coalesce-prefixed train derail → set remains empty."""
        import contextlib

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            MergeOutcome,
            SpeculativeMergeWorker,
        )

        db_path = tmp_path / 'es_ncosr.db'
        es = EventStore(db_path=db_path, run_id='run-ncosr-test')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        group_future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        group = GroupMergeRequest(
            task_id='nc2',
            branch='task/nc2',
            worktree=tmp_path / 'wt-nc2',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=coalesce_config,
            result=group_future,
            train_id='train-beta-xyz',   # NOT coalesce-prefixed
            member_task_ids=['nc1', 'nc2'],
            tip_branch='task/nc2',
            tip_task_id='nc2',
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )

        with patch(
            'orchestrator.merge_queue._do_train_merge',
            AsyncMock(return_value=MergeOutcome('blocked', reason='verify red')),
        ):
            await queue.put(group)
            worker_task = asyncio.create_task(worker.run())
            await asyncio.wait_for(group_future, timeout=30)

        assert worker._coalesce_derailed_task_ids == set(), (
            f'non-coalesce train derail must NOT populate one-strike set; '
            f'got {worker._coalesce_derailed_task_ids}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ─── δ Amendment: observability + robustness ─────────────────────────────────

@pytest.mark.asyncio
class TestSuppressionObservability:
    """amend(1720): exclusion gate suppresses near-train (2 candidates, 1 excluded).

    When there are exactly 2 structural candidates and the predicate excludes
    one of them, eligible drops below 2 and the pass returns False without
    forming a train.  Both candidate futures must remain unresolved; no
    train_coalesced event is emitted.

    This tests the <2-eligible return path that was previously untested.
    """

    async def test_two_candidates_one_excluded_no_train(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """2 stackable singles, 1 excluded by predicate → no train, both futures unresolved."""
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import GroupMergeRequest, SpeculativeMergeWorker

        wt1 = await _make_branch_with_file(git_ops, 'sup1', 'file_sup1.py', 'sup1 = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'sup2', 'file_sup2.py', 'sup2 = 2\n')

        db_path = tmp_path / 'es_supp.db'
        es = EventStore(db_path=db_path, run_id='run-supp-test')

        queue: asyncio.Queue = asyncio.Queue()
        # Predicate excludes sup1 — only sup2 eligible, which is < 2 → no train.
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
            merge_ready_predicate=lambda req: 'test_risky' if req.task_id == 'sup1' else None,
        )

        req1 = _make_req('sup1', 'task/sup1', wt1, coalesce_config)
        req2 = _make_req('sup2', 'task/sup2', wt2, coalesce_config)
        worker._lane_buffers['normal'].extend([req1, req2])

        result = await worker._maybe_coalesce_waiting_singles()

        # pass returns False — insufficient eligible candidates
        assert result is False, (
            'pass must return False when exclusion gate drops eligible below 2'
        )

        # No train formed in the buffer
        buf = list(worker._lane_buffers['normal'])
        trains = [r for r in buf if isinstance(r, GroupMergeRequest)]
        assert len(trains) == 0, f'no train must form; found {len(trains)}'

        # Both futures remain unresolved
        assert not req1.result.done(), 'excluded req1 future must remain unresolved'
        assert not req2.result.done(), 'eligible-but-insufficient req2 future must remain unresolved'

        # No train_coalesced event emitted
        events = _events_of_type(db_path, 'train_coalesced')
        assert len(events) == 0, f'no train_coalesced event must be emitted; got {events}'


# ─── δ Amendment: design coherence — error outcome also marks one-strike ──────

@pytest.mark.asyncio
class TestOneStrikeOnErrorOutcome:
    """amend(1720): 'error' outcome on coalesce train also marks members one-strike.

    The recording trigger was changed from ``outcome.status == 'blocked'`` to
    ``outcome.status in _COALESCE_RISKY_TERMINAL_STATES`` so both 'blocked'
    (verify failure) and 'error' (infra/git error) drive the one-strike marker,
    aligning with the exclusion predicate's _COALESCE_RISKY_TERMINAL_STATES check.
    """

    async def test_error_outcome_marks_members_one_strike(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        """Coalesce-prefixed train derail with 'error' → members marked one-strike."""
        import contextlib

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            MergeOutcome,
            SpeculativeMergeWorker,
        )

        db_path = tmp_path / 'es_osr_err.db'
        es = EventStore(db_path=db_path, run_id='run-osr-err-test')

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops,
            queue,
            event_store=es,
            train_callback_factory=_stub_factory(),
        )

        group_future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        group = GroupMergeRequest(
            task_id='e2',
            branch='task/e2',
            worktree=tmp_path / 'wt-e2',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=coalesce_config,
            result=group_future,
            train_id='coalesce-tipm-errtest',
            member_task_ids=['e1', 'e2'],
            tip_branch='task/e2',
            tip_task_id='e2',
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )

        # Patch _do_train_merge to return 'error' (infra/git failure).
        with patch(
            'orchestrator.merge_queue._do_train_merge',
            AsyncMock(return_value=MergeOutcome('error', reason='git push failed')),
        ):
            await queue.put(group)
            worker_task = asyncio.create_task(worker.run())
            await asyncio.wait_for(group_future, timeout=30)

        assert worker._coalesce_derailed_task_ids == {'e1', 'e2'}, (
            f'coalesce train members must be marked one-strike on error derail; '
            f'got {worker._coalesce_derailed_task_ids}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
