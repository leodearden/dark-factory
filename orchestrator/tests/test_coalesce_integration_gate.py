"""Integration gate (ζ/1722): end-to-end retro-coalescing boundary scenarios.

Exercises the full α–ε stack (tasks 1717-1721) together through the REAL
SpeculativeMergeWorker + asyncio.Queue driven by the REAL
build_train_callback_factory(FakeScheduler).  Fakes are restricted to:
  • git edge  — real local git-repo fixture
  • scheduler edge — FakeScheduler (no HTTP calls)

Four boundary scenarios + one speculative-cap bookkeeping assertion:
  Scenario 1: 3 disjoint stackable singles coalesce, MCP member merge_status-observable
  Scenario 2: partial stackability — overlap keeps 3rd as solo
  Scenario 3: confidence-gate exclusion visible in train_coalesced event
  Scenario 4: in-flight + detached-waiter exclusion invariants
  Bookkeeping: depth-1/K cap accounting stays consistent across a coalesce
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from _workflow_helpers import FakeScheduler
from test_merge_queue_coalesce import (
    _events_of_type,
    _gated_verify,
    _make_branch_with_file,
    _make_req,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps


# ─── Fixtures ────────────────────────────────────────────────────────────────


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
def coalesce_config(git_repo: Path, git_config) -> OrchestratorConfig:
    """Config with merge_train_coalesce_enabled=True."""
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        merge_train_coalesce_enabled=True,
    )


# ─── Scenario 1 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestScenario1:
    """step-1 (RED): 3 disjoint stackable singles coalesce into one train end-to-end.

    Three disjoint-file singles are pre-enqueued before the worker starts.
    The coalescing pass fires on the first merger iteration.  A gated-verify
    blocks the train's first run_scoped_verification call.

    Assertions (step-1):
      (a) exactly one GroupMergeRequest dispatched, train_id starts with 'coalesce-'
      (b) all 3 waiter futures resolve MergeOutcome(status='superseded',
          superseded_by=train_id) BEFORE verify starts (α-consumer input contract)
      (c) a single train_coalesced event names member_task_ids + absorbed_request_ids
      (d) the train entry is visible in worker.snapshot() while gated (verifying state)
      (e) after gate_release all 3 member files appear on main

    RED: FakeScheduler.statuses is empty → build_train_callback_factory's
    mark_member_done no-op guard fires (member absent from scheduler) →
    scheduler.mark_done is never called → all_done never fires → timeout.
    step-2 fixes this by pre-seeding scheduler.statuses for the 3 members.
    """

    async def test_three_singles_coalesce_end_to_end(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        import contextlib
        from unittest.mock import patch

        from orchestrator.event_store import EventStore
        from orchestrator.harness import build_train_callback_factory
        from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome, SpeculativeMergeWorker

        # 3 disjoint-file branches (each touches a unique file → line-stackable).
        # Branch names are BARE (no 'task/' prefix): create_worktree('ig1') creates
        # 'task/ig1'.  Passing 'ig1' (bare) to _make_req so merge_to_main doesn't
        # double-prefix to 'task/task/ig1'.
        wt1 = await _make_branch_with_file(git_ops, 'ig1', 'file_ig1.py', 'a = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'ig2', 'file_ig2.py', 'b = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'ig3', 'file_ig3.py', 'c = 3\n')

        db_path = tmp_path / 'es_ig1.db'
        es = EventStore(db_path=db_path, run_id='ig1-run')

        # Use the REAL build_train_callback_factory backed by FakeScheduler.
        scheduler = FakeScheduler()
        # Wrap scheduler.mark_done to count calls and fire all_done when complete.
        # RED: scheduler.statuses is empty → mark_member_done existence check no-ops →
        # mark_done wrapper is never reached → all_done never fires → timeout below.
        mark_done_calls: list[str] = []
        all_done = asyncio.Event()
        _real_mark_done = scheduler.mark_done

        async def _counting_mark_done(task_id: str, *, kind: str, sha: str, note: str | None = None) -> None:
            await _real_mark_done(task_id, kind=kind, sha=sha, note=note)
            mark_done_calls.append(task_id)
            if len(mark_done_calls) >= 3:
                all_done.set()

        scheduler.mark_done = _counting_mark_done

        factory = build_train_callback_factory(scheduler)

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es, train_callback_factory=factory,
        )

        # Pre-enqueue all 3 BEFORE starting the worker (coalescing pass fires on
        # first iteration when spec_base=None / prefetched=None).
        req1 = _make_req('ig1', 'ig1', wt1, coalesce_config)
        req2 = _make_req('ig2', 'ig2', wt2, coalesce_config)
        req3 = _make_req('ig3', 'ig3', wt3, coalesce_config)
        await queue.put(req1)
        await queue.put(req2)
        await queue.put(req3)

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _gated_verify(gate_release, gate_entered),
        ):
            worker_task = asyncio.create_task(worker.run())

            # Wait for gated verify.  When gate_entered fires:
            #   • _maybe_coalesce_waiting_singles has run — all 3 singles resolved
            #     MergeOutcome('superseded') and the GroupMergeRequest is in _lane_buffers.
            #   • _do_train_merge has picked up the GroupMergeRequest (_inflight_req set),
            #     completed rebase + merge, and is now suspended inside
            #     run_scoped_verification (which is gated).
            await asyncio.wait_for(gate_entered.wait(), timeout=60)

            # (b) All 3 singles' futures resolved 'superseded' before verify starts.
            # (These are resolved by _maybe_coalesce_waiting_singles, BEFORE the train
            # is even dequeued for merging — so they're guaranteed done at gate_entered.)
            for req_i, label_i in ((req1, 'ig1'), (req2, 'ig2'), (req3, 'ig3')):
                assert req_i.result.done(), (
                    f'{label_i}: future must be resolved as superseded at gate_entered'
                )
                outcome_i: MergeOutcome = req_i.result.result()
                assert outcome_i.status == 'superseded', (
                    f'{label_i}: expected superseded, got {outcome_i.status!r}'
                )

            # (a) Exactly one GroupMergeRequest dispatched, train_id starts 'coalesce-'.
            # The GroupMergeRequest is currently _inflight_req (merger loop suspended
            # inside _do_train_merge at run_scoped_verification).  confirm via snapshot
            # (depth=1, state='merging') and via the shared train_id from the futures.
            train_id = req1.result.result().superseded_by
            assert train_id is not None and train_id.startswith('coalesce-'), (
                f'train_id must start with "coalesce-"; got {train_id!r}'
            )
            # All 3 futures must share the same train_id.
            for req_i, label_i in ((req2, 'ig2'), (req3, 'ig3')):
                assert req_i.result.result().superseded_by == train_id, (
                    f'{label_i}: superseded_by={req_i.result.result().superseded_by!r} '
                    f'!= train_id={train_id!r}'
                )

            # (d) GroupMergeRequest visible in snapshot while gated.
            # When gate_entered fires the train is _inflight_req (state='merging').
            snapshot = worker.snapshot()
            assert snapshot['depth'] >= 1, (
                f'Expected >= 1 entry in snapshot; depth={snapshot["depth"]}'
            )
            merging = [e for e in snapshot['entries'] if e['state'] == 'merging']
            assert len(merging) == 1, (
                f'Expected 1 merging entry; got states='
                f'{[e["state"] for e in snapshot["entries"]]}'
            )

            # (c) Single train_coalesced event with member_task_ids + absorbed_request_ids.
            events = _events_of_type(db_path, 'train_coalesced')
            assert len(events) == 1, f'Expected 1 train_coalesced event; got {len(events)}'
            event_data = events[0]['data']
            assert set(event_data.get('member_task_ids', [])) == {'ig1', 'ig2', 'ig3'}, (
                f'train_coalesced member_task_ids mismatch: {event_data}'
            )
            absorbed = event_data.get('absorbed_request_ids', [])
            assert len(absorbed) == 3, (
                f'Expected 3 absorbed_request_ids; got {absorbed}'
            )

            # Release gate — verify completes, advance_main runs.
            gate_release.set()

            # Wait for all 3 mark_member_done calls (train fully landed).
            # RED: FakeScheduler.statuses is empty → build_train_callback_factory's
            # mark_member_done existence check no-ops (mid absent from scheduler) →
            # scheduler.mark_done is never called → _counting_mark_done never fires →
            # all_done never set → asyncio.TimeoutError.
            await asyncio.wait_for(all_done.wait(), timeout=30)

        # (e) All 3 member files on main after landing.
        from orchestrator.git_ops import _run
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'file_ig1.py' in main_files, 'file_ig1.py must be on main after train lands'
        assert 'file_ig2.py' in main_files, 'file_ig2.py must be on main after train lands'
        assert 'file_ig3.py' in main_files, 'file_ig3.py must be on main after train lands'

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
