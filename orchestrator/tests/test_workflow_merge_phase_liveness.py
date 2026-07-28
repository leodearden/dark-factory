"""Durable merge-phase liveness stamp/clear + entry wiring (task 2991).

Task 2991 (successor to task 2931): a legitimately-live task in the
pre-enqueue MERGE phase runs a rebase/scoped-verify/queue-submit loop with NO
LLM calls, so it never refreshes ``metadata.routing.latest`` — the orphan-L0
divergence reaper's task-2931 ``_has_fresh_dispatch`` gate cannot see it and
false-promotes its scope-invariant L0 to a human-facing L1 (cluster
esc-2789-22). The fix is a DURABLE ``metadata.merge_phase_liveness =
{'entered_at': <iso>}`` stamp — the restart-survivable analog of
``routing.latest`` — written at merge entry and read by
``_has_fresh_merge_phase`` (see test_orphan_l0_reaper.py).

This module unit-tests the workflow producer side:

* ``_stamp_merge_phase_entered`` / ``_clear_merge_phase_entered`` —
  best-effort metadata stamp/clear helpers (mirroring
  ``_stamp_merge_retry_pending`` / ``_clear_merge_retry_pending``).
* the ``_run_merge_phase`` entry ordering — the stamp is written after
  ``_enter_phase(MERGE)`` and BEFORE ``_check_scope_invariant`` (so it always
  precedes the scope-invariant escalation filing and the gating bail).
* the symmetric clears at the durable-enqueue boundary
  (``_submit_to_merge_queue``) and on merge success (``_merge_and_finalise``).

Follows the mock-fixture style of ``test_workflow_merge_retry_pending.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    scheduler: MagicMock
    git_ops: MagicMock
    queue: MagicMock


def _make(
    *,
    task_id: str = '77',
    metadata: dict | None = None,
    worktree: Path = Path('/tmp/wt-2991'),
    main_sha: str = 'BASE-SHA',
    backend_metadata: dict | None = None,
    update_task_raises: bool = False,
    merge_queue=None,
    merge_inflight_registry=None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    # _merge_fresh_metadata reads backend metadata before every stamp/clear.
    scheduler.get_task = AsyncMock(
        return_value={
            'metadata': backend_metadata if backend_metadata is not None else {},
        },
    )

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
        merge_queue=merge_queue,
        merge_inflight_registry=merge_inflight_registry,
    )
    wf.artifacts = MagicMock()
    wf.worktree = worktree

    return _Fixture(
        wf=wf, update_task=update_task, scheduler=scheduler,
        git_ops=git_ops, queue=queue,
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


def _fake_run(*, head: str = 'HEAD-SHA', rc: int = 0):
    """Return an async stand-in for orchestrator.workflow._run (git rev-parse)."""

    async def _run(cmd, cwd=None, **kwargs):  # noqa: ARG001
        if rc == 0:
            return (0, head + '\n', '')
        return (rc, '', 'fatal: bad revision')

    return _run


# ---------------------------------------------------------------------------
# step-3: _stamp_merge_phase_entered helper
# ---------------------------------------------------------------------------


class TestStampHelper:
    @pytest.mark.asyncio
    async def test_stamp_persists_entered_at_iso(self):
        f = _make(metadata={'retry_ledger': {'x': 1}})

        await f.wf._stamp_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        stamp = meta['merge_phase_liveness']
        # (a) persisted stamp is exactly {'entered_at': <non-empty iso str>}.
        assert set(stamp) == {'entered_at'}
        assert isinstance(stamp['entered_at'], str) and stamp['entered_at']
        # Parseable tz-aware ISO-8601 (what _has_fresh_merge_phase reads).
        parsed = datetime.fromisoformat(stamp['entered_at'])
        assert parsed.tzinfo is not None
        # (b) sibling metadata keys preserved (read-modify-write via
        # _merge_fresh_metadata, not a clobber).
        assert meta['retry_ledger'] == {'x': 1}
        # (c) in-memory task metadata mirrors the persisted stamp.
        assert f.wf.task['metadata']['merge_phase_liveness'] == stamp

    @pytest.mark.asyncio
    async def test_stamp_overlays_backend_sibling_keys(self):
        # A backend-only sibling key (present in get_task but NOT in the
        # in-memory copy) surviving into the persisted write proves the stamp
        # goes through _merge_fresh_metadata's backend overlay, not a raw
        # in-memory clobber — e.g. memory_hints re-attached by Stage-2
        # reconciliation after self.task was loaded.
        f = _make(
            metadata={'retry_ledger': {'x': 1}},
            backend_metadata={'memory_hints': ['h1']},
        )

        await f.wf._stamp_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        assert meta['merge_phase_liveness']['entered_at']
        assert meta['retry_ledger'] == {'x': 1}   # in-memory sibling kept
        assert meta['memory_hints'] == ['h1']     # backend sibling overlaid

    @pytest.mark.asyncio
    async def test_stamp_is_best_effort_and_does_not_raise_on_persist_failure(self):
        f = _make(update_task_raises=True)

        # scheduler.update_task raises — the helper must swallow it (durability
        # is best-effort; a lost stamp is self-healing on the next entry).
        await f.wf._stamp_merge_phase_entered()

        # (d) In-memory metadata is still updated (persist is the last step).
        assert f.wf.task['metadata']['merge_phase_liveness']['entered_at']


# ---------------------------------------------------------------------------
# step-5: _clear_merge_phase_entered helper
# ---------------------------------------------------------------------------


_STAMP = {'entered_at': '2026-07-24T05:00:00+00:00'}


class TestClearHelper:
    @pytest.mark.asyncio
    async def test_clear_removes_key_from_persisted_and_in_memory(self):
        f = _make(
            metadata={
                'merge_phase_liveness': dict(_STAMP),
                'retry_ledger': {'x': 1},
            },
        )

        await f.wf._clear_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        # (a) key removed from the persisted full-dict write.
        assert 'merge_phase_liveness' not in meta
        # (b) clearing one key leaves siblings intact.
        assert meta['retry_ledger'] == {'x': 1}
        # (a) key also removed from in-memory task metadata.
        assert 'merge_phase_liveness' not in f.wf.task['metadata']
        # The payload assertion above is NOT sufficient on its own: an omitted
        # metadata_mode resolves to 'merge' (scheduler.py:3795-3799), which the
        # backend implements as shallow json.dumps({**old, **new})
        # (sqlite_task_backend.py:3362-3364) — "omitted keys are PRESERVED", so
        # popping a key out of the payload is a backend NO-OP. Only the
        # sanctioned delete-by-omission mode actually removes it.
        assert f.update_task.await_args.kwargs['metadata_mode'] == 'replace'

    @pytest.mark.asyncio
    async def test_clear_replace_payload_carries_backend_only_keys(self):
        # A 'replace' write is a whole-blob overwrite, so the payload MUST be
        # built from the freshly-read backend blob — otherwise every
        # backend-only key (memory_hints re-attached by Stage-2 reconciliation,
        # _causation_id) is deleted along with the stamp (the #4271
        # sibling-clobber bug). Nothing but the popped key may disappear.
        f = _make(
            metadata={'merge_phase_liveness': dict(_STAMP)},
            backend_metadata={
                'memory_hints': {'q': 1},
                'merge_phase_liveness': dict(_STAMP),
            },
        )

        await f.wf._clear_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        assert meta['memory_hints'] == {'q': 1}
        assert 'merge_phase_liveness' not in meta
        assert f.update_task.await_args.kwargs['metadata_mode'] == 'replace'

    @pytest.mark.asyncio
    async def test_clear_skips_durable_write_when_fresh_read_fails(self):
        # _merge_fresh_metadata falls back to in-memory-only metadata when
        # get_task raises. Under the old 'merge' mode that fallback was
        # harmless (omitted backend-only keys survived); under 'replace' it
        # would DELETE every backend-only key. So a failed read must SKIP the
        # durable write entirely — bounded cost: the reaper may keep deferring
        # for up to orphan_l0_merge_phase_freshness_secs until the stale stamp
        # ages out (deferral, never suppression).
        f = _make(
            metadata={
                'merge_phase_liveness': dict(_STAMP),
                'retry_ledger': {'x': 1},
            },
        )
        f.scheduler.get_task = AsyncMock(side_effect=RuntimeError('mcp down'))

        await f.wf._clear_merge_phase_entered()   # must not raise

        f.update_task.assert_not_awaited()
        # In-memory copy is still cleared (siblings intact).
        assert 'merge_phase_liveness' not in f.wf.task['metadata']
        assert f.wf.task['metadata']['retry_ledger'] == {'x': 1}

    @pytest.mark.asyncio
    async def test_clear_skips_durable_write_when_get_task_returns_non_dict(self):
        # Same guard for the silent non-dict branch (get_task returning None on
        # a vanished/unreadable task): a 'replace' built from in-memory-only
        # metadata would clobber backend-only siblings.
        f = _make(
            metadata={
                'merge_phase_liveness': dict(_STAMP),
                'retry_ledger': {'x': 1},
            },
        )
        f.scheduler.get_task = AsyncMock(return_value=None)

        await f.wf._clear_merge_phase_entered()   # must not raise

        f.update_task.assert_not_awaited()
        assert 'merge_phase_liveness' not in f.wf.task['metadata']
        assert f.wf.task['metadata']['retry_ledger'] == {'x': 1}

    @pytest.mark.asyncio
    async def test_clear_is_best_effort_and_does_not_raise_on_persist_failure(self):
        f = _make(
            metadata={'merge_phase_liveness': dict(_STAMP)},
            update_task_raises=True,
        )
        # (c) must not propagate the update_task failure, and still clears
        # in-memory.
        await f.wf._clear_merge_phase_entered()
        assert 'merge_phase_liveness' not in f.wf.task['metadata']

    @pytest.mark.asyncio
    async def test_clear_is_noop_when_no_stamp_present(self):
        # (d) cheap idempotent no-op — no fresh-metadata read, no backend write
        # — when nothing was ever stamped, so it is safe to call
        # unconditionally on every enqueue / merge success (the common case
        # never stamped).
        f = _make(metadata={'retry_ledger': {'x': 1}})

        await f.wf._clear_merge_phase_entered()

        f.update_task.assert_not_awaited()
        f.scheduler.get_task.assert_not_awaited()
        # Metadata is left untouched.
        assert f.wf.task['metadata'] == {'retry_ledger': {'x': 1}}


# ---------------------------------------------------------------------------
# step-7: _run_merge_phase entry ordering — stamp BEFORE _check_scope_invariant
# ---------------------------------------------------------------------------


_PASSING_VERIFY_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all green',
)


class TestEntryOrdering:
    @pytest.mark.asyncio
    async def test_stamp_written_before_scope_invariant_at_merge_entry(
        self, monkeypatch,
    ):
        """Task 2991 fix-direction (c): the durable stamp is written at merge
        entry — after ``_enter_phase(MERGE)`` and BEFORE
        ``_check_scope_invariant`` — so it always precedes the scope-invariant
        escalation filing (and the gating bail), refreshing on every
        (re-)dispatch into merge phase, including passes that immediately bail
        to ESCALATED.

        Modeled on test_workflow.py::test_run_merge_phase_stamps_note_on_entry.
        ``_enter_phase`` / ``_stamp_merge_phase_entered`` /
        ``_check_scope_invariant`` are attached to one parent mock so
        ``parent.mock_calls`` records their global call order; asserts
        enter < stamp < scope and that stamp + scope were each awaited once.

        RED on main: ``_run_merge_phase`` never calls
        ``_stamp_merge_phase_entered`` (the stamp name is absent from
        mock_calls -> ValueError in .index()).
        """
        f = _make()
        wf = f.wf
        wf.plan = {'files': ['a.py']}
        wf.event_store = None
        wf.config.max_merge_retries = 1
        wf.config.max_pre_merge_retries = 1

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'branch_head_sha\n', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=_PASSING_VERIFY_RESULT),
        )

        # One parent mock records the global order of the three entry calls.
        parent = MagicMock()
        enter = MagicMock()   # _enter_phase is sync
        stamp = AsyncMock()   # _stamp_merge_phase_entered
        scope = AsyncMock()   # _check_scope_invariant
        parent.attach_mock(enter, 'enter')
        parent.attach_mock(stamp, 'stamp')
        parent.attach_mock(scope, 'scope')
        wf._enter_phase = enter                # type: ignore[method-assign]
        wf._stamp_merge_phase_entered = stamp  # type: ignore[method-assign]
        wf._check_scope_invariant = scope      # type: ignore[method-assign]

        # Stub the rest of the merge internals so the phase reaches Phase 2 and
        # the stubbed _submit_to_merge_queue returns DONE (loop breaks → None).
        wf._maybe_defer_as_train_member = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
        wf._recover_before_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf.git_ops.get_main_sha = AsyncMock(return_value='main_sha')
        wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)
        wf._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )

        result = await wf._run_merge_phase('task/77')

        assert result is None  # fell through to SUCCESS
        stamp.assert_awaited_once()
        scope.assert_awaited_once()
        # Global call order across the shared parent: enter → stamp → scope.
        names = [c[0] for c in parent.mock_calls]
        assert names.index('enter') < names.index('stamp') < names.index('scope'), (
            f'expected enter < stamp < scope; got {names}'
        )


# ---------------------------------------------------------------------------
# step-9: symmetric clear wiring — durable-enqueue boundary + merge success
# ---------------------------------------------------------------------------


class TestEnqueueClearWiring:
    """The durable stamp is discharged at the enqueue boundary.

    Once the request is on the durable merge journal the pre-enqueue window is
    over: the task is no longer "live in merge phase" in the sense the reaper
    gate protects, so a lingering fresh stamp would briefly defer an unrelated
    stranded divergence orphan. Mirrors the in-memory ``clear_merge_phase``
    call it is co-located with (test_workflow.py::
    test_submit_clears_merge_phase_after_enqueue).
    """

    @pytest.mark.asyncio
    async def test_submit_clears_durable_stamp_after_enqueue(
        self, tmp_path: Path, monkeypatch,
    ):
        """RED on main: ``_submit_to_merge_queue`` never clears the durable stamp."""
        import asyncio

        from orchestrator import merge_queue as merge_queue_mod
        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        f = _make(
            task_id='B',
            metadata={'merge_phase_liveness': dict(_STAMP)},
            worktree=wt,
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf = f.wf
        wf.config.git.branch_prefix = 'task/'  # real str for QueuedBranch.parse
        wf.config.project_root = tmp_path
        wf.event_store = None
        wf.plan = {}          # empty → _task_files=None → pre-merge gate skipped
        wf._base_commit = None
        wf._module_configs = []
        wf.git_ops.rebind_branch_to_head = AsyncMock(return_value=True)

        # Order the durable enqueue against the clear on one parent mock: the
        # clear must fire AFTER the request is on the crash-safe journal, not
        # before (a pre-enqueue clear would re-open the false-positive window
        # the stamp exists to close).
        parent = MagicMock()
        enqueue = AsyncMock(
            side_effect=merge_queue_mod.register_and_enqueue_merge_request,
        )
        clear = AsyncMock()
        parent.attach_mock(enqueue, 'enqueue')
        parent.attach_mock(clear, 'clear')
        monkeypatch.setattr(
            merge_queue_mod, 'register_and_enqueue_merge_request', enqueue,
        )
        wf._clear_merge_phase_entered = clear  # type: ignore[method-assign]

        async def _worker():
            req = await real_queue.get()
            req.result.set_result(MergeOutcome(status='done', merge_sha='sha'))

        worker_task = asyncio.create_task(_worker())
        outcome = await wf._submit_to_merge_queue('B')
        await worker_task

        assert outcome == WorkflowOutcome.DONE
        clear.assert_awaited_once()
        # Co-located with the in-memory grace clear (task 2753).
        f.scheduler.clear_merge_phase.assert_called_once_with('B')
        names = [c[0] for c in parent.mock_calls]
        assert names.index('enqueue') < names.index('clear'), (
            f'expected enqueue < clear; got {names}'
        )


class TestMergeSuccessClearWiring:
    """Merge SUCCESS discharges the durable stamp.

    Covers the ghost-loop / eval-success paths that reach DONE without passing
    through the enqueue clear, so a DONE task never carries a stale
    ``merge_phase_liveness``. Mirrors test_workflow_merge_retry_pending.py::
    test_merge_success_discharges_present_merge_retry_pending_stamp (the real
    ``_clear_merge_phase_entered`` runs — it is NOT spied here).
    """

    @staticmethod
    def _wire_finalise_spies(f: _Fixture, *, merge_result) -> None:
        wf = f.wf
        wf._run_merge_phase = AsyncMock(return_value=merge_result)  # type: ignore[method-assign]
        wf._write_completion_to_memory = AsyncMock()  # type: ignore[method-assign]
        wf._ensure_steward_started = AsyncMock()  # type: ignore[method-assign]
        wf._await_steward_completion = AsyncMock()  # type: ignore[method-assign]
        wf._finalise_merged_done = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._enter_phase = MagicMock()  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_merge_success_discharges_present_stamp(self):
        """RED on main: ``_merge_and_finalise`` never clears the durable stamp."""
        f = _make(
            metadata={
                'merge_phase_liveness': dict(_STAMP),
                'retry_ledger': {'x': 1},
            },
        )
        self._wire_finalise_spies(f, merge_result=None)  # None → merge SUCCEEDED

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        # Discharged from in-memory AND persisted metadata; siblings preserved.
        assert 'merge_phase_liveness' not in f.wf.task['metadata']
        assert f.wf.task['metadata']['retry_ledger'] == {'x': 1}
        persisted = _persisted_metadata(f.update_task)
        assert 'merge_phase_liveness' not in persisted
        assert persisted['retry_ledger'] == {'x': 1}

    @pytest.mark.asyncio
    async def test_merge_success_without_stamp_makes_no_clear_write(self):
        # Mirror: the unstamped common case triggers no backend write from the
        # clear-on-success call (the no-op guard holds at this call site).
        f = _make()
        self._wire_finalise_spies(f, merge_result=None)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        f.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_terminal_merge_outcome_leaves_stamp_intact(self):
        # A merge that did NOT succeed returns early — the stamp must survive so
        # the reaper keeps deferring while the task cycles through merge phase.
        f = _make(metadata={'merge_phase_liveness': dict(_STAMP)})
        self._wire_finalise_spies(f, merge_result=WorkflowOutcome.BLOCKED)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.BLOCKED
        assert f.wf.task['metadata']['merge_phase_liveness'] == _STAMP
        f.update_task.assert_not_awaited()
