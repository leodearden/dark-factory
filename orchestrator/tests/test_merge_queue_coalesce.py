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
        wt1 = await _make_branch_with_file(git_ops, 'task/t1', 'file_a.py', 'a = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'task/t2', 'file_b.py', 'b = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'task/t3', 'file_c.py', 'c = 3\n')

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
