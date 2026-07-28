"""Tests for durable merge-retry intent on merge-phase resume (task 2795).

When a merge-phase escalation is resolved via ``resume``, ``_requeue`` returns
``WorkflowOutcome.REQUEUED`` while leaving the task ``in-progress`` for an
in-RAM in-place merge retry — a restart mid-retry silently loses that
obligation (Reify 5166). This module unit-tests the durable fix:

* ``_stamp_merge_retry_pending`` / ``_clear_merge_retry_pending`` — best-effort
  metadata stamp/clear helpers (mirroring ``_stamp_first_merge_enqueue``).
* the ``_requeue`` wiring — the merge_phase=True StewardResolved path stamps
  before returning REQUEUED; the merge_phase=False path does not.
* ``_merge_and_finalise`` — the extracted MERGE+SUCCESS tail shared by the
  normal ``_drive`` path and the resume guard.
* ``_resume_merge_retry_if_pending`` — the ``_drive`` early guard that jumps
  straight back to the merge phase when the post-rebase worktree HEAD still
  matches the stamped ``branch_head``.

Follows the mock-fixture style of ``test_workflow_merge_thrash.py``: a local
``_make`` builds a minimal :class:`TaskWorkflow` with mocks, and the
module-level ``_run`` (git rev-parse) is controlled by monkeypatching
``orchestrator.workflow._run``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import ConflictProbe
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState
from orchestrator.workflow_types import StewardResolved


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
    worktree: Path = Path('/tmp/wt-2795'),
    main_sha: str = 'BASE-SHA',
    backend_metadata: dict | None = None,
    update_task_raises: bool = False,
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
        return_value={'metadata': backend_metadata if backend_metadata is not None else {}}
    )

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    # The resume guard probes whether the (unchanged) branch STILL merges cleanly
    # onto current main before fast-pathing to merge.  Default to clean so the
    # HEAD-match resume path behaves as before; conflict tests override this.
    git_ops.merge_tree_conflicts = AsyncMock(
        return_value=ConflictProbe(clean=True, conflicted_paths=[])
    )

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


def _persisted_mode(update_task: AsyncMock) -> str | None:
    """Return the ``metadata_mode`` the persist call actually passed.

    Asserting only that the payload OMITS a key is not sufficient to prove the
    key was deleted: ``scheduler.update_task`` with no ``metadata_mode``
    resolves to ``'merge'`` (scheduler.py), and the backend's merge mode is a
    shallow ``{**existing, **incoming}`` that PRESERVES omitted keys
    (sqlite_task_backend ``_merge_metadata``).  Only ``'replace'``
    (whole-blob overwrite, delete-by-omission) removes anything, so the mode is
    half of the persisted-clear contract and must be pinned alongside the payload.
    """
    assert update_task.await_args is not None
    return update_task.await_args.kwargs.get('metadata_mode')


def _fake_run(*, head: str = 'HEAD-SHA', rc: int = 0):
    """Return an async stand-in for orchestrator.workflow._run (git rev-parse)."""

    async def _run(cmd, cwd=None, **kwargs):  # noqa: ARG001
        if rc == 0:
            return (0, head + '\n', '')
        return (rc, '', 'fatal: bad revision')

    return _run


_STAMP = {'branch_head': 'HEAD-SHA', 'base_sha': 'BASE-SHA', 'resolved_at': '2026-07-20T05:00:00+00:00'}


# ---------------------------------------------------------------------------
# step-3: stamp / clear helpers
# ---------------------------------------------------------------------------


class TestStampClearHelpers:
    @pytest.mark.asyncio
    async def test_stamp_persists_branch_head_base_sha_resolved_at(self, monkeypatch):
        f = _make(metadata={'retry_ledger': {'x': 1}}, main_sha='BASE-SHA')
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        await f.wf._stamp_merge_retry_pending()

        meta = _persisted_metadata(f.update_task)
        mrp = meta['merge_retry_pending']
        assert mrp['branch_head'] == 'HEAD-SHA'
        assert mrp['base_sha'] == 'BASE-SHA'
        assert isinstance(mrp['resolved_at'], str) and mrp['resolved_at']
        # Sibling metadata keys are preserved (read-modify-write, not clobber).
        assert meta['retry_ledger'] == {'x': 1}
        # In-memory task metadata mirrors the persisted stamp.
        assert f.wf.task['metadata']['merge_retry_pending'] == mrp

    @pytest.mark.asyncio
    async def test_stamp_is_best_effort_and_does_not_raise_on_persist_failure(
        self, monkeypatch,
    ):
        f = _make(update_task_raises=True)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        # scheduler.update_task raises — the helper must swallow it.
        await f.wf._stamp_merge_retry_pending()

        # In-memory metadata is still updated (persist is the last, best-effort step).
        assert f.wf.task['metadata']['merge_retry_pending']['branch_head'] == 'HEAD-SHA'

    @pytest.mark.asyncio
    async def test_clear_removes_key_from_persisted_and_in_memory(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})

        await f.wf._clear_merge_retry_pending()

        meta = _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in meta
        # Clearing one key leaves siblings intact.
        assert meta['retry_ledger'] == {'x': 1}
        assert 'merge_retry_pending' not in f.wf.task['metadata']
        # task 3024: the payload omitting the key is NOT enough — the default
        # 'merge' mode preserves omitted keys, so a clear that ships no
        # metadata_mode logs a removal it never performs.  Pin the mode.
        #
        # Task 2991 filed an identical assertion plus a skip-the-write-on-failed-
        # read test for this same clear; task 3024 landed first with strictly
        # broader coverage (see the "'replace' makes a STALE payload
        # destructive" block below, which covers BOTH the raising and the
        # returns-None refresh paths), so the 2991 duplicates were dropped at
        # rebase rather than kept as a second, weaker copy.
        assert _persisted_mode(f.update_task) == 'replace'

    @pytest.mark.asyncio
    async def test_clear_is_best_effort_and_does_not_raise_on_persist_failure(self):
        f = _make(
            metadata={'merge_retry_pending': dict(_STAMP)},
            update_task_raises=True,
        )
        # Must not propagate the update_task failure.
        await f.wf._clear_merge_retry_pending()
        # Amendment: the in-memory delete is withheld until the backend confirms
        # the write, so a failed persist leaves memory and backend AGREEING that
        # the stamp is still there.  Clearing memory anyway would make the
        # merge-success clear early-return on the missing key and land a DONE
        # task still carrying merge_retry_pending server-side.
        assert f.wf.task['metadata']['merge_retry_pending'] == dict(_STAMP)

    @pytest.mark.asyncio
    async def test_clear_leaves_stamp_when_update_task_reports_failure(self, caplog):
        """update_task reports MCP failures by RETURNING FALSE, not by raising.

        ``Scheduler.update_task`` logs and returns ``False`` on an isError
        response or a structured rejection (scheduler.py), so a clear that
        ignored the boolean would treat a rejected write as a landed one: the
        in-memory stamp would be dropped, the merge-success retry would be
        skipped, and nothing in the log would say the obligation survived.
        """
        f = _make(metadata={'merge_retry_pending': dict(_STAMP)})
        f.update_task.return_value = False

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            await f.wf._clear_merge_retry_pending()

        f.update_task.assert_awaited_once()
        assert f.wf.task['metadata']['merge_retry_pending'] == dict(_STAMP)
        assert any(
            'could not clear merge_retry_pending' in r.getMessage()
            for r in caplog.records
        ), f'a rejected write must be logged as a failure; records={caplog.records}'

    # -- task 3024 step-11: 'replace' makes a STALE payload destructive --

    @pytest.mark.asyncio
    async def test_clear_skips_replace_write_when_refresh_fails(self):
        """A failed backend refresh must skip the write, not replace with a guess.

        ``metadata_mode='replace'`` changes the blast radius of a stale payload.
        ``_merge_fresh_metadata`` deliberately swallows a ``get_task`` failure and
        returns the in-memory metadata ALONE (its own docstring warns
        'memory_hints may be clobbered').  Under the old 'merge' mode that only
        risked the keys actually supplied; under 'replace' the same fallback
        DELETES every backend-only key — so the fix for a stamp that will not
        clear could destroy ``memory_hints`` re-attached by Stage-2 reconciliation.

        Skipping the write is strictly the safer failure: a surviving stamp is
        self-healing (a later dispatch retries the clear, and the conflict probe
        and the harness restart clear are two independent remedies for the same
        wedge), whereas clobbered metadata is unrecoverable.
        """
        f = _make(metadata={'merge_retry_pending': dict(_STAMP)})
        f.scheduler.get_task = AsyncMock(side_effect=RuntimeError('mcp down'))

        # Best-effort contract holds: the refresh failure is swallowed.
        await f.wf._clear_merge_retry_pending()

        # No destructive whole-blob write on an unverified payload.
        f.update_task.assert_not_awaited()
        # The obligation survives intact, so a later dispatch can retry the clear.
        assert f.wf.task['metadata']['merge_retry_pending'] == dict(_STAMP)

    @pytest.mark.asyncio
    async def test_clear_skips_replace_write_when_refresh_returns_none(self):
        """Same contract as the raise case, via the path that actually occurs.

        ``Scheduler.get_task`` catches every exception and returns ``None``
        (scheduler.py), so in production a failed refresh reaches
        ``_merge_fresh_metadata`` as a non-dict result — the ``require_fresh``
        isinstance guard — not as a propagated error.  The destructive
        whole-blob replace must be skipped on that path too.
        """
        f = _make(metadata={'merge_retry_pending': dict(_STAMP)})
        f.scheduler.get_task = AsyncMock(return_value=None)

        await f.wf._clear_merge_retry_pending()

        f.update_task.assert_not_awaited()
        assert f.wf.task['metadata']['merge_retry_pending'] == dict(_STAMP)

    @pytest.mark.asyncio
    async def test_clear_preserves_backend_only_keys_on_successful_refresh(self):
        """The replace only ever ships a backend-VERIFIED blob.

        Mirror of the skip case: when the refresh succeeds, backend-only keys the
        live workflow object never saw must ride along in the whole-blob write,
        because under 'replace' every omitted key is deleted.
        """
        f = _make(
            metadata={'merge_retry_pending': dict(_STAMP)},
            backend_metadata={'memory_hints': ['h1']},
        )

        await f.wf._clear_merge_retry_pending()

        assert _persisted_mode(f.update_task) == 'replace'
        persisted = _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in persisted
        assert persisted['memory_hints'] == ['h1']

    @pytest.mark.asyncio
    async def test_clear_is_noop_when_no_stamp_present(self):
        # Amendment (reviewer robustness): the clear helper must be a cheap
        # idempotent no-op — no fresh-metadata read, no backend write — when
        # nothing was ever stamped, so it is safe to call unconditionally on
        # every merge success (the common case never stamped).
        f = _make(metadata={'retry_ledger': {'x': 1}})

        await f.wf._clear_merge_retry_pending()

        f.update_task.assert_not_awaited()
        f.scheduler.get_task.assert_not_awaited()
        # Metadata is left untouched.
        assert f.wf.task['metadata'] == {'retry_ledger': {'x': 1}}


# ---------------------------------------------------------------------------
# step-5: _requeue wiring — merge_phase=True stamps, merge_phase=False does not
# ---------------------------------------------------------------------------


async def _drive_requeue_via_mark_blocked(f: _Fixture, *, merge_phase: bool) -> WorkflowOutcome:
    """Drive _mark_blocked to the StewardResolved _requeue path with a spy stamp.

    Wires a truthy steward whose completion resolves to StewardResolved so the
    nested _requeue() closure runs, and spies _stamp_merge_retry_pending so the
    test can assert whether the merge_phase branch stamped. The escalation-queue
    filing and (for merge_phase=False) the BLOCKED status transition + dry-run
    spawn are stubbed to no-ops so only the requeue branch is under test.
    """
    wf = f.wf
    f.queue.has_open_l1 = MagicMock(return_value=False)
    f.queue.make_id = MagicMock(return_value='esc-1')
    f.queue.submit = MagicMock()

    wf._ensure_steward_started = AsyncMock()  # type: ignore[method-assign]
    wf._steward = MagicMock()  # truthy → the `if self._steward:` branch runs
    wf._await_steward_completion = AsyncMock(  # type: ignore[method-assign]
        return_value=StewardResolved(resolution_text='resolved'),
    )
    # merge_phase=False path transitions state + spawns a dry-run; stub both so
    # the test stays focused on the requeue branch (and avoids state-machine
    # transition constraints in this minimal fixture).
    wf._enter_phase = MagicMock()  # type: ignore[method-assign]
    wf._spawn_dry_run_unblock = MagicMock()  # type: ignore[method-assign]

    return await wf._mark_blocked(
        reason='merge verification failed',
        merge_phase=merge_phase,
        category='merge_error',
    )


class TestRequeueWiring:
    @pytest.mark.asyncio
    async def test_merge_phase_requeue_stamps_exactly_once_before_requeued(self):
        f = _make()
        stamp_spy = AsyncMock()
        f.wf._stamp_merge_retry_pending = stamp_spy  # type: ignore[method-assign]

        outcome = await _drive_requeue_via_mark_blocked(f, merge_phase=True)

        assert outcome == WorkflowOutcome.REQUEUED
        stamp_spy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_merge_phase_requeue_does_not_stamp(self):
        f = _make()
        stamp_spy = AsyncMock()
        f.wf._stamp_merge_retry_pending = stamp_spy  # type: ignore[method-assign]

        outcome = await _drive_requeue_via_mark_blocked(f, merge_phase=False)

        assert outcome == WorkflowOutcome.REQUEUED
        stamp_spy.assert_not_awaited()
        # The non-merge_phase requeue durably re-pends through the scheduler.
        f.scheduler.set_task_status.assert_any_await(f.wf.task_id, 'pending')


# ---------------------------------------------------------------------------
# step-7: _merge_and_finalise(branch_name) extraction contract
# ---------------------------------------------------------------------------


@dataclass
class _FinaliseSpies:
    run_merge_phase: AsyncMock
    write_completion_to_memory: AsyncMock
    await_steward_completion: AsyncMock
    finalise_merged_done: AsyncMock
    enter_phase: MagicMock


def _wire_finalise_spies(f: _Fixture, *, merge_result) -> _FinaliseSpies:
    """Spy _run_merge_phase and the finalise tail on the fixture's workflow.

    Returns the typed mock handles so assertions read them directly, rather
    than through the workflow's real bound methods (which pyright resolves to
    MethodType, not the assigned mock).
    """
    wf = f.wf
    spies = _FinaliseSpies(
        run_merge_phase=AsyncMock(return_value=merge_result),
        write_completion_to_memory=AsyncMock(),
        await_steward_completion=AsyncMock(),
        finalise_merged_done=AsyncMock(return_value=WorkflowOutcome.DONE),
        enter_phase=MagicMock(),
    )
    wf._run_merge_phase = spies.run_merge_phase  # type: ignore[method-assign]
    wf._write_completion_to_memory = spies.write_completion_to_memory  # type: ignore[method-assign]
    wf._ensure_steward_started = AsyncMock()  # type: ignore[method-assign]
    wf._await_steward_completion = spies.await_steward_completion  # type: ignore[method-assign]
    wf._finalise_merged_done = spies.finalise_merged_done  # type: ignore[method-assign]
    wf._enter_phase = spies.enter_phase  # type: ignore[method-assign]
    return spies


class TestMergeAndFinalise:
    @pytest.mark.asyncio
    async def test_terminal_merge_outcome_returned_as_is_without_finalise_tail(self):
        # (a) _run_merge_phase returns a terminal WorkflowOutcome (merge did NOT
        # succeed) → return it verbatim, never run the finalise/DONE tail.
        f = _make()
        spies = _wire_finalise_spies(f, merge_result=WorkflowOutcome.BLOCKED)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.BLOCKED
        spies.run_merge_phase.assert_awaited_once()
        spies.write_completion_to_memory.assert_not_awaited()
        spies.finalise_merged_done.assert_not_awaited()
        spies.enter_phase.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_merge_result_runs_finalise_tail_and_returns_done(self):
        # (b) _run_merge_phase returns None (merge SUCCEEDED) → run the finalise
        # tail and return _finalise_merged_done()'s value, entering DONE.
        f = _make()
        spies = _wire_finalise_spies(f, merge_result=None)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        spies.run_merge_phase.assert_awaited_once()
        spies.write_completion_to_memory.assert_awaited_once()
        spies.await_steward_completion.assert_awaited_once()
        spies.enter_phase.assert_any_call(WorkflowState.DONE)
        spies.finalise_merged_done.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_eval_mode_skips_merge_phase_but_runs_finalise_tail(self):
        # (c) eval mode (_worktree_external=True) never merges into main, but the
        # SUCCESS/finalise tail still runs.
        f = _make()
        spies = _wire_finalise_spies(f, merge_result=WorkflowOutcome.BLOCKED)
        f.wf._worktree_external = True

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        spies.run_merge_phase.assert_not_awaited()
        spies.write_completion_to_memory.assert_awaited_once()
        spies.enter_phase.assert_any_call(WorkflowState.DONE)
        spies.finalise_merged_done.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_merge_success_discharges_present_merge_retry_pending_stamp(self):
        # Amendment (reviewer robustness): on the normal resolve→in-place-retry→
        # success path _requeue's stamp is otherwise never cleared (the resume
        # guard's clear fires only on a RESTART re-dispatch). _merge_and_finalise
        # must discharge it on merge SUCCESS so a DONE task carries no stale
        # merge_retry_pending. (real _clear_merge_retry_pending runs — not spied.)
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})
        spies = _wire_finalise_spies(f, merge_result=None)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        spies.finalise_merged_done.assert_awaited_once()
        # Stamp discharged from both in-memory and persisted metadata; siblings kept.
        assert 'merge_retry_pending' not in f.wf.task['metadata']
        assert f.wf.task['metadata']['retry_ledger'] == {'x': 1}
        persisted = _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in persisted

    @pytest.mark.asyncio
    async def test_merge_success_without_stamp_makes_no_clear_write(self):
        # Mirror: the unstamped common case triggers no backend write from the
        # clear-on-success call (the no-op guard holds inside _merge_and_finalise).
        f = _make()
        spies = _wire_finalise_spies(f, merge_result=None)

        outcome = await f.wf._merge_and_finalise('task/77')

        assert outcome == WorkflowOutcome.DONE
        spies.finalise_merged_done.assert_awaited_once()
        f.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_merge_success_clear_retries_a_persist_the_resume_clear_lost(
        self, monkeypatch,
    ):
        """A clear that never landed is retried within the SAME dispatch.

        End-to-end reason the in-memory delete waits for a confirmed persist: the
        resume guard clears the stamp before delegating to _merge_and_finalise,
        which clears again on merge SUCCESS.  Had the first (failed) clear dropped
        the in-memory key anyway, the second clear's no-stamp fast-return would
        fire and a DONE task would land still carrying merge_retry_pending
        server-side (which the done-metadata reconcile then writes back, since
        its `{**in_memory, **backend}` union lets the backend key win).
        """
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})
        _wire_finalise_spies(f, merge_result=None)
        # First persist (resume-guard clear) fails; second (merge-success) lands.
        f.update_task.side_effect = [RuntimeError('mcp down'), True]

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome == WorkflowOutcome.DONE
        assert f.update_task.await_count == 2, (
            'the merge-success clear must retry the write the resume-guard clear lost'
        )
        assert _persisted_mode(f.update_task) == 'replace'
        assert 'merge_retry_pending' not in _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in f.wf.task['metadata']
        assert f.wf.task['metadata']['retry_ledger'] == {'x': 1}


# ---------------------------------------------------------------------------
# step-9: _resume_merge_retry_if_pending guard + _drive placement
# ---------------------------------------------------------------------------


@dataclass
class _ResumeGuardSpies:
    merge_and_finalise: AsyncMock
    clear_merge_retry_pending: AsyncMock


def _wire_resume_guard_spies(f: _Fixture) -> _ResumeGuardSpies:
    """Spy _merge_and_finalise and _clear_merge_retry_pending for guard tests.

    Returns the typed mock handles so assertions read them directly.
    """
    wf = f.wf
    spies = _ResumeGuardSpies(
        merge_and_finalise=AsyncMock(return_value=WorkflowOutcome.DONE),
        clear_merge_retry_pending=AsyncMock(),
    )
    wf._merge_and_finalise = spies.merge_and_finalise  # type: ignore[method-assign]
    wf._clear_merge_retry_pending = spies.clear_merge_retry_pending  # type: ignore[method-assign]
    return spies


class TestResumeGuard:
    @pytest.mark.asyncio
    async def test_no_stamp_returns_none_no_delegation_no_clear(self, monkeypatch):
        f = _make(metadata={})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        assert await f.wf._resume_merge_retry_if_pending('task/77') is None
        spies.merge_and_finalise.assert_not_awaited()
        spies.clear_merge_retry_pending.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_dict_stamp_returns_none(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': 'not-a-dict'})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        assert await f.wf._resume_merge_retry_if_pending('task/77') is None
        spies.merge_and_finalise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_head_match_clears_then_delegates_and_returns_outcome(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'HEAD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome == WorkflowOutcome.DONE
        spies.clear_merge_retry_pending.assert_awaited_once()
        spies.merge_and_finalise.assert_awaited_once_with('task/77')
        # task 3024: the clean-merge probe must run on this path too — a clean
        # ConflictProbe is what licenses the fast-path, so it is never skipped.
        f.git_ops.merge_tree_conflicts.assert_awaited_once_with('BASE-SHA', 'HEAD-SHA')

    @pytest.mark.asyncio
    async def test_head_mismatch_clears_stale_stamp_and_returns_none(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'OLD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        # A mismatch clears the stale stamp (the branch moved; it can never match).
        spies.clear_merge_retry_pending.assert_awaited_once()
        spies.merge_and_finalise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rev_parse_failure_returns_none_no_delegation(self, monkeypatch):
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'HEAD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(rc=1))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        spies.merge_and_finalise.assert_not_awaited()

    # -- task 3024 step-1: HEAD match but the branch no longer merges cleanly --

    @pytest.mark.asyncio
    async def test_head_match_but_since_developed_conflict_clears_and_falls_back(
        self, monkeypatch,
    ):
        """HEAD still matches the stamp, but main advanced and introduced a conflict.

        The stamped branch is unchanged (HEAD match), so the pre-3024 guard
        fast-pathed straight to _merge_and_finalise with an empty ``plan.files``
        — tripping the merge-entry scope invariant and looping forever.  The
        clean-merge probe must void the now-unsatisfiable stamp and fall back to
        the full plan/execute/verify/review pipeline instead.
        """
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'HEAD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f.git_ops.merge_tree_conflicts = AsyncMock(
            return_value=ConflictProbe(
                clean=False,
                conflicted_paths=['orchestrator/src/orchestrator/merge_queue.py'],
            )
        )

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        # A confirmed conflict voids the resume obligation — clear it.
        spies.clear_merge_retry_pending.assert_awaited_once()
        # ...and never hand an empty-plan workflow to the merge phase.
        spies.merge_and_finalise.assert_not_awaited()
        f.git_ops.merge_tree_conflicts.assert_awaited_once_with('BASE-SHA', 'HEAD-SHA')

    # -- task 3024 step-3: the probe itself failing is transient, not a verdict --

    @pytest.mark.asyncio
    async def test_probe_error_returns_none_without_clearing(self, monkeypatch):
        """A merge_tree_conflicts failure is uncertain — preserve the obligation.

        Unlike a confirmed conflict (which voids the stamp), a probe error says
        nothing about whether the branch still merges.  Fall back to the full
        pipeline for THIS dispatch, but leave the durable stamp in place so a
        later dispatch can still honour it (mirrors the rev-parse fail-safe).
        """
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'HEAD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f.git_ops.merge_tree_conflicts = AsyncMock(side_effect=RuntimeError('bad ref'))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        spies.merge_and_finalise.assert_not_awaited()
        spies.clear_merge_retry_pending.assert_not_awaited()

    # -- task 3024 step-9: end-to-end — the REAL clear, at the scheduler boundary --

    @pytest.mark.asyncio
    async def test_conflict_arm_persists_a_replace_write_that_deletes_the_stamp(
        self, monkeypatch,
    ):
        """The conflict arm's clear must actually DELETE the stamp server-side.

        Deliberately does NOT spy ``_clear_merge_retry_pending``: the sibling
        guard tests assert the arm CALLS the helper, which says nothing about
        whether the helper achieves anything.  A ``metadata_mode``-less write
        resolves to 'merge' on the backend, whose ``{**existing, **incoming}``
        preserves omitted keys — so the stamp would survive, the guard would
        re-fire on the next dispatch, and the wedge this task fixes would only
        move one layer down.  Assert the real persisted call instead.
        """
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})
        merge_and_finalise = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._merge_and_finalise = merge_and_finalise  # type: ignore[method-assign]
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f.git_ops.merge_tree_conflicts = AsyncMock(
            return_value=ConflictProbe(clean=False, conflicted_paths=['x.py']),
        )

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        merge_and_finalise.assert_not_awaited()
        # The write that lands must be able to remove a key, and must carry
        # every sibling key through (whole-blob replace ⇒ omission = deletion).
        assert _persisted_mode(f.update_task) == 'replace'
        persisted = _persisted_metadata(f.update_task)
        assert 'merge_retry_pending' not in persisted
        assert persisted['retry_ledger'] == {'x': 1}

    @pytest.mark.asyncio
    async def test_probe_error_arm_persists_no_write_at_all(self, monkeypatch):
        """Mirror: a transient probe failure must leave the backend untouched.

        Also unspied, so this pins the composed behaviour rather than a mock:
        the durable obligation survives a non-verdict, and no whole-blob
        ``replace`` is issued on the strength of an error.
        """
        f = _make(metadata={'merge_retry_pending': dict(_STAMP), 'retry_ledger': {'x': 1}})
        merge_and_finalise = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._merge_and_finalise = merge_and_finalise  # type: ignore[method-assign]
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f.git_ops.merge_tree_conflicts = AsyncMock(side_effect=RuntimeError('bad ref'))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        merge_and_finalise.assert_not_awaited()
        f.update_task.assert_not_awaited()
        assert f.wf.task['metadata']['merge_retry_pending'] == dict(_STAMP)

    @pytest.mark.asyncio
    async def test_main_sha_error_returns_none_without_clearing(self, monkeypatch):
        """Same fail-safe when the probe's BASE (current main tip) can't be read."""
        f = _make(metadata={'merge_retry_pending': {**_STAMP, 'branch_head': 'HEAD-SHA'}})
        spies = _wire_resume_guard_spies(f)
        monkeypatch.setattr('orchestrator.workflow._run', _fake_run(head='HEAD-SHA'))
        f.git_ops.get_main_sha = AsyncMock(side_effect=RuntimeError('main read failed'))

        outcome = await f.wf._resume_merge_retry_if_pending('task/77')

        assert outcome is None
        spies.merge_and_finalise.assert_not_awaited()
        spies.clear_merge_retry_pending.assert_not_awaited()


class TestDrivePlacement:
    @pytest.mark.asyncio
    async def test_drive_returns_guard_outcome_without_invoking_plan(self):
        f = _make(metadata={})
        wf = f.wf
        # Stub setup so _drive reaches the guard with worktree/artifacts intact.
        wf._setup_worktree_and_artifacts = AsyncMock()  # type: ignore[method-assign]
        wf._recover_if_already_merged = AsyncMock(return_value=None)  # type: ignore[method-assign]
        guard = AsyncMock(return_value=WorkflowOutcome.DONE)
        wf._resume_merge_retry_if_pending = guard  # type: ignore[method-assign]
        plan_spy = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._plan = plan_spy  # type: ignore[method-assign]
        wf._enter_phase = MagicMock()  # type: ignore[method-assign]

        outcome = await wf._drive()

        # The resume guard short-circuits the pipeline: _drive returns its
        # outcome and never runs PLAN (zero architect/implementer/reviewer).
        assert outcome == WorkflowOutcome.DONE
        guard.assert_awaited_once()
        plan_spy.assert_not_awaited()
