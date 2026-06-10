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
    """step-2 (GREEN): 3 disjoint stackable singles coalesce into one train end-to-end.

    Three disjoint-file singles are pre-enqueued before the worker starts.
    The coalescing pass fires on the first merger iteration.  A gated-verify
    blocks the train's first run_scoped_verification call.

    Assertions (step-1, extended by step-2):
      (a) exactly one GroupMergeRequest dispatched, train_id starts with 'coalesce-'
      (b) all 3 waiter futures resolve MergeOutcome(status='superseded',
          superseded_by=train_id) BEFORE verify starts (α-consumer input contract)
      (c) a single train_coalesced event names member_task_ids + absorbed_request_ids
      (d) the train entry is visible in worker.snapshot() while gated (verifying state)
      (e) after gate_release all 3 member files appear on main
      (f) FakeScheduler.provenance has kind='merged' + shared SHA for all 3 members (β)
      (g) _handle_superseded parks status='merge-deferred', never 'done' (α consumer)
      (h) MCP member is merge_status-observable via event-store tier: the member
          entered via enqueue_merge_request has a merge_finalized record with
          state='superseded' + superseded_by=train_id; the train has a train_merged event.

    GREEN (step-2): scheduler.statuses pre-seeded for ig1/ig2/ig3 so that
    mark_member_done's existence check passes and mark_done is called.
    req1 enters via enqueue_merge_request (not bare queue.put) so that _on_finalized
    is registered and emits the durable merge_finalized record on superseded resolution.
    """

    async def test_three_singles_coalesce_end_to_end(
        self,
        git_ops: GitOps,
        coalesce_config: OrchestratorConfig,
        tmp_path: Path,
    ):
        import contextlib
        from unittest.mock import AsyncMock, MagicMock, patch

        from orchestrator.event_store import EventStore
        from orchestrator.harness import build_train_callback_factory
        from orchestrator.merge_queue import (
            GroupMergeRequest,
            MergeOutcome,
            SpeculativeMergeWorker,
            enqueue_merge_request,
        )
        from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

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
        # GREEN (step-2): pre-seed scheduler.statuses for ig1/ig2/ig3 so that
        # mark_member_done's get_statuses existence check sees the members and
        # does NOT no-op.  Without this, get_statuses returns {} (no known ids)
        # and mark_done is never called → all_done never fires → timeout.
        for mid in ('ig1', 'ig2', 'ig3'):
            scheduler.statuses[mid] = ['merge-deferred']

        # Wrap scheduler.mark_done to count calls and fire all_done when complete.
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
        # (h) ε surface: req1 enters via enqueue_merge_request so _on_finalized is
        # registered.  When req1.result is resolved 'superseded' by the coalescing
        # pass, the callback fires and writes a merge_finalized record to the event
        # store (state='superseded', superseded_by=train_id).
        await enqueue_merge_request(queue, req1, es)
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

        # (f) β merged provenance: FakeScheduler.provenance has kind='merged' for all 3
        # members, and all 3 share the same merge SHA (the train's advance commit).
        for mid in ('ig1', 'ig2', 'ig3'):
            prov = scheduler.provenance.get(mid, {})
            assert prov.get('kind') == 'merged', (
                f'{mid}: expected kind="merged" in scheduler.provenance; got {prov!r}'
            )
        merge_shas = {scheduler.provenance[mid]['commit'] for mid in ('ig1', 'ig2', 'ig3')}
        assert len(merge_shas) == 1, (
            f'All 3 members must share one merge SHA; got {merge_shas!r}'
        )
        merged_sha = next(iter(merge_shas))

        # (g) α workflow consumer: _handle_superseded parks as merge-deferred, never done.
        # Mirror the _Fixture pattern from test_workflow_merge_superseded: build a minimal
        # TaskWorkflow backed by a mock scheduler and drive _handle_superseded directly.
        _asgn = MagicMock()
        _asgn.task_id = 'probe-alpha'
        _asgn.task = {
            'id': 'probe-alpha', 'title': 'Probe', 'description': 'α probe',
            'metadata': {},
        }
        _asgn.modules = []  # empty → _resolve_module_configs returns [] safely

        _sched_probe = MagicMock()
        _sched_probe.set_task_status = AsyncMock()

        _wf_probe = TaskWorkflow(
            assignment=_asgn,
            config=coalesce_config,
            git_ops=git_ops,
            scheduler=_sched_probe,
            briefing=MagicMock(),
            mcp=MagicMock(),
        )
        _wf_probe.event_store = None  # None-safe: skip event emission

        alpha_result = await _wf_probe._handle_superseded(
            MergeOutcome('superseded', superseded_by=train_id)
        )
        assert alpha_result == WorkflowOutcome.MERGE_DEFERRED, (
            f'α consumer: expected MERGE_DEFERRED, got {alpha_result!r}'
        )
        _sched_probe.set_task_status.assert_any_await('probe-alpha', 'merge-deferred')
        assert not any(
            call.args == ('probe-alpha', 'done')
            for call in _sched_probe.set_task_status.await_args_list
        ), 'α consumer: set_task_status must never be called with "done" on superseded outcome'

        # (h) ε surface: req1 (entered via enqueue_merge_request) is merge_status-observable.
        # _on_finalized fires when req1.result resolves; it writes a merge_finalized row
        # with state='superseded' and superseded_by=train_id to the event store.
        # Give the done_callback a moment to fire — it runs synchronously on the event
        # loop after set_result, but asyncio may not have drained it yet.
        await asyncio.sleep(0)
        finalized_member = es.latest_merge_finalized(request_id=req1.request_id)
        assert finalized_member is not None, (
            'ε surface: req1 entered via enqueue_merge_request must have a '
            'merge_finalized record after its future is resolved superseded'
        )
        assert finalized_member['state'] == 'superseded', (
            f'ε surface: member merge_finalized state must be "superseded"; '
            f'got {finalized_member["state"]!r}'
        )
        assert finalized_member['superseded_by'] == train_id, (
            f'ε surface: superseded_by must be {train_id!r}; '
            f'got {finalized_member["superseded_by"]!r}'
        )
        # 'Follow the train': the train's train_merged event confirms it landed done.
        train_merged_events = _events_of_type(db_path, 'train_merged')
        assert len(train_merged_events) == 1, (
            f'ε surface: expected 1 train_merged event; got {len(train_merged_events)}'
        )
        assert train_merged_events[0]['data'].get('merge_commit_sha') == merged_sha, (
            f'ε surface: train_merged event must carry the same SHA as merged provenance; '
            f'got {train_merged_events[0]["data"].get("merge_commit_sha")!r}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ─── Scenario 2 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestScenario2:
    """step-3 (RED): partial stackability — overlap keeps 3rd as solo.

    Three branches: s21 and s22 are disjoint (unique files → stackable);
    s23 edits the SAME file as s21 → get_changed_line_ranges returns
    overlapping ranges → _select_train_members rejects co-selection.

    Assertions:
      (a) the formed train has exactly 2 members (s21 + s22), NOT s23
      (b) s23's future is UNRESOLVED at gate_entered — it is NOT absorbed
          into the train (status NOT 'superseded')
      (c) train_coalesced event member_task_ids == {s21, s22} (excludes s23)
      (d) after gate_release the 2-member train lands; s21/s22 files on main
      (e) s23 stays in the worker queue as a solo MergeRequest (not a GroupMergeRequest)

    RED: s23 currently uses a UNIQUE file (file_s23.py) — disjoint from s21 →
    all 3 get coalesced into a 3-member train → assertion (b) fails because
    s23.result.done() is True (absorbed).  step-4 fixes this by giving s23
    the SAME filename as s21 (file_overlap.py) so ranges overlap.
    """

    async def test_partial_stackability_overlap_keeps_solo(
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

        # Three branches:
        #   s21: adds file_overlap.py → old-side range (0,0) on file_overlap.py
        #   s22: adds file_s22.py (unique) → no shared file with s21 → stackable
        #   s23: RED — adds file_s23.py (unique) instead of file_overlap.py →
        #        disjoint from everyone → all 3 get coalesced → assertion (b) fails.
        #        step-4 fixes s23 to add file_overlap.py → overlaps with s21 → stays solo.
        wt1 = await _make_branch_with_file(git_ops, 'ig_s21', 'file_overlap.py', 'x = 1\n')
        wt2 = await _make_branch_with_file(git_ops, 'ig_s22', 'file_s22.py',      'y = 2\n')
        wt3 = await _make_branch_with_file(git_ops, 'ig_s23', 'file_s23.py',      'z = 3\n')

        db_path = tmp_path / 'es_s2.db'
        es = EventStore(db_path=db_path, run_id='s2-run')

        scheduler = FakeScheduler()
        for mid in ('ig_s21', 'ig_s22', 'ig_s23'):
            scheduler.statuses[mid] = ['merge-deferred']

        # Collect mark_done calls for the 2-member train (s21 + s22).
        mark_done_calls: list[str] = []
        train_done = asyncio.Event()
        _real_mark_done = scheduler.mark_done

        async def _counting_mark_done(task_id: str, *, kind: str, sha: str, note: str | None = None) -> None:
            await _real_mark_done(task_id, kind=kind, sha=sha, note=note)
            mark_done_calls.append(task_id)
            # Train has 2 members; fire when both are flipped.
            if len(mark_done_calls) >= 2:
                train_done.set()

        scheduler.mark_done = _counting_mark_done

        factory = build_train_callback_factory(scheduler)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, queue, event_store=es, train_callback_factory=factory,
        )

        req1 = _make_req('ig_s21', 'ig_s21', wt1, coalesce_config)
        req2 = _make_req('ig_s22', 'ig_s22', wt2, coalesce_config)
        req3 = _make_req('ig_s23', 'ig_s23', wt3, coalesce_config)
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

            await asyncio.wait_for(gate_entered.wait(), timeout=60)

            # (b) s23's future must be UNRESOLVED — it is not absorbed.
            # RED: s23 uses file_s23.py (disjoint) → all 3 coalesced → s23.result.done()
            # is True → this assertion FAILS.
            assert not req3.result.done(), (
                's23 must remain solo (future unresolved); '
                'expected s23 NOT absorbed by the train'
            )

            # (a) Train has exactly 2 members: s21 + s22.
            train_id = req1.result.result().superseded_by
            assert train_id is not None and train_id.startswith('coalesce-'), (
                f'train_id must start with "coalesce-"; got {train_id!r}'
            )
            assert req2.result.result().superseded_by == train_id, (
                f's22 must be absorbed into the same train as s21'
            )

            # (c) train_coalesced event member_task_ids == {{s21, s22}}.
            events = _events_of_type(db_path, 'train_coalesced')
            assert len(events) == 1, f'Expected 1 train_coalesced event; got {len(events)}'
            event_data = events[0]['data']
            assert set(event_data.get('member_task_ids', [])) == {'ig_s21', 'ig_s22'}, (
                f'train_coalesced member_task_ids must be {{s21, s22}}; got {event_data}'
            )

            gate_release.set()
            await asyncio.wait_for(train_done.wait(), timeout=30)

        # (d) s21 and s22 files on main after the train lands.
        from orchestrator.git_ops import _run
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'file_overlap.py' in main_files, 'file_overlap.py (s21) must be on main'
        assert 'file_s22.py' in main_files,     'file_s22.py (s22) must be on main'

        # (e) s23 remains in worker queue as solo (not GroupMergeRequest).
        # The worker stopped processing after the train landed; s23 is still pending.
        buf = list(worker._lane_buffers['normal'])
        solos = [r for r in buf if not isinstance(r, GroupMergeRequest)]
        assert any(r.task_id == 'ig_s23' for r in solos), (
            f's23 must remain as a solo in lane_buffers; buf={[r.task_id for r in buf]}'
        )

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
