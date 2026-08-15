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
    async def test_stamp_survives_corrupt_non_dict_backend_metadata(self):
        # The stamp's read-modify-write goes through _merge_fresh_metadata ->
        # _read_fresh_backend_metadata, whose documented contract is
        # {} = read OK / dict = the blob / None = could not read. A CORRUPT
        # persisted blob (non-dict) must map to "could not read", not be
        # returned raw — `{**in_memory, **'corrupt'}` raises
        # `TypeError: 'str' object is not a mapping` from a code path with no
        # try/except around it, on the merge critical path. Not hypothetical:
        # the whole point of the stamp is to be readable after a crash, i.e.
        # exactly when a partial write may have left a bad blob.
        f = _make(metadata={'retry_ledger': {'x': 1}})
        f.scheduler.get_task = AsyncMock(return_value={'metadata': 'corrupt'})

        await f.wf._stamp_merge_phase_entered()   # must not raise

        meta = _persisted_metadata(f.update_task)
        assert meta['merge_phase_liveness']['entered_at']
        # Fell back to in-memory metadata alone (the corrupt blob is ignored,
        # not overlaid). The stamp write itself stays 'merge' mode, so the
        # backend repairs the key it can and leaves the rest alone.
        assert meta['retry_ledger'] == {'x': 1}

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
    async def test_clear_replace_loses_concurrent_backend_write_in_read_window(self):
        """Characterization test — documents an ACCEPTED cost of 'replace',
        not desired behaviour.

        Switching the clear to ``metadata_mode='replace'`` is what makes
        delete-by-omission work, but it opens a lost-update window that
        ``'merge'`` mode did not have: read-modify-write across an ``await``
        boundary, then a WHOLE-BLOB overwrite. Any key another process writes
        between the two round-trips is silently dropped — including
        ``routing.latest``, the input to the reaper's sibling
        ``_has_fresh_dispatch`` gate. Pinned here so the next reader finds it
        as a known property rather than a surprise; it cannot be eliminated
        without a targeted key-delete mode (the backend accepts only
        {'merge', 'additive', 'replace'} — sqlite_task_backend._METADATA_MODES).

        Models a real backend blob so the loss is observable rather than
        asserted about a mock payload.
        """
        backend_blob = {
            'merge_phase_liveness': dict(_STAMP),
            'memory_hints': ['h1'],
        }
        f = _make(metadata={'merge_phase_liveness': dict(_STAMP)})

        async def _get_task(_task_id):
            snapshot = {'metadata': dict(backend_blob)}
            # ...and now ANOTHER writer lands a key before our write does.
            backend_blob['routing'] = {
                'latest': {'decided_at': '2026-07-24T05:00:05+00:00'},
            }
            return snapshot

        async def _update_task(_task_id, metadata=None, metadata_mode=None, **_kw):
            assert metadata_mode == 'replace'
            backend_blob.clear()
            backend_blob.update(metadata or {})   # whole-blob overwrite
            return True

        f.scheduler.get_task = AsyncMock(side_effect=_get_task)
        f.scheduler.update_task = AsyncMock(side_effect=_update_task)

        await f.wf._clear_merge_phase_entered()

        # The clear does its job: the stamp is really gone from the backend
        # copy, and the sibling key read in the same snapshot survives.
        assert 'merge_phase_liveness' not in backend_blob
        assert backend_blob['memory_hints'] == ['h1']
        # ACCEPTED LOSS: the interleaved write is gone. Under the old 'merge'
        # mode it would have survived (omitted keys were preserved) — but so
        # would the stamp, which is the bug this clear exists to fix.
        assert 'routing' not in backend_blob

    @pytest.mark.asyncio
    async def test_clear_skips_durable_write_when_backend_metadata_is_non_dict(self):
        # Third unreadable-blob shape: get_task returns a task whose persisted
        # `metadata` is CORRUPT (a non-dict) — a case scheduler.update_task's
        # own docstring acknowledges ("the sanctioned repair path if a task's
        # persisted metadata is corrupt (non-dict)"). Without the isinstance
        # guard in _read_fresh_backend_metadata the corrupt value is returned
        # and `{**metadata, **backend}` raises
        # `TypeError: 'str' object is not a mapping` OUTSIDE the try/except
        # that wraps only update_task — propagating out of
        # _submit_to_merge_queue on the merge critical path. Treat it as
        # unreadable: skip the whole-blob 'replace' write (never overwrite a
        # blob nobody understands) and clear in-memory only.
        f = _make(
            metadata={
                'merge_phase_liveness': dict(_STAMP),
                'retry_ledger': {'x': 1},
            },
        )
        f.scheduler.get_task = AsyncMock(return_value={'metadata': 'corrupt'})

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
        enter < FIRST stamp < scope (``.index`` finds the first occurrence, and
        the merge-entry stamp is the only one above the retry loop).

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
        # Two writes with max_merge_retries=1: the merge-entry stamp asserted
        # here, plus the review-fix R2-B re-stamp at the top of the single
        # retry attempt (see TestRetryLoopReStamp). The duplicate on attempt 1
        # is the accepted cost of keeping both write points — the entry one
        # must precede _check_scope_invariant, which sits above the loop.
        assert stamp.await_count == 2, (
            f'expected entry stamp + 1 loop-top re-stamp; got '
            f'{stamp.await_count}'
        )
        scope.assert_awaited_once()
        # Global call order across the shared parent: enter → stamp → scope.
        names = [c[0] for c in parent.mock_calls]
        assert names.index('enter') < names.index('stamp') < names.index('scope'), (
            f'expected enter < stamp < scope; got {names}'
        )


class TestEntryReadCoalescing:
    """Review amendment (efficiency): the merge-entry stamp and the
    scope-invariant check read the SAME backend blob, so they must share one
    ``get_task`` round-trip instead of issuing two back-to-back (15s timeout
    each) on the merge hot path. Sharing also makes the two evaluate one
    snapshot rather than two taken at different instants."""

    @pytest.mark.asyncio
    async def test_scope_check_uses_prefetched_metadata_without_a_second_read(self):
        f = _make()
        f.wf.plan = {'files': ['pkg/a.py']}
        spy = MagicMock()
        f.wf._escalate_scope_invariant_violation = spy  # type: ignore[method-assign]

        await f.wf._check_scope_invariant(
            backend_metadata={'files': ['other/b.py']},
        )

        f.scheduler.get_task.assert_not_awaited()
        # The prefetched blob really drove the comparison (divergent modules
        # at lock_depth=2 -> escalation filed with those metadata files).
        # 'pkg/a.py' is not covered by metadata's 'other/b.py' (task 3429:
        # third positional arg is the uncovered module list).
        spy.assert_called_once_with(['pkg/a.py'], ['other/b.py'], ['pkg/a.py'])

    @pytest.mark.asyncio
    async def test_scope_check_falls_back_to_own_read_when_prefetch_unavailable(self):
        # None means "the stamp could not read the backend" — the check must
        # fall back to its own get_task, so its documented fail-safe (an
        # unreadable task is skipped, never treated as divergent) is unchanged.
        f = _make()
        f.wf.plan = {'files': ['pkg/a.py']}
        f.scheduler.get_task = AsyncMock(
            return_value={'metadata': {'files': ['other/b.py']}},
        )
        spy = MagicMock()
        f.wf._escalate_scope_invariant_violation = spy  # type: ignore[method-assign]

        await f.wf._check_scope_invariant(backend_metadata=None)

        f.scheduler.get_task.assert_awaited_once()
        # task 3429: third positional arg is the uncovered module list.
        spy.assert_called_once_with(['pkg/a.py'], ['other/b.py'], ['pkg/a.py'])

    @pytest.mark.asyncio
    async def test_scope_check_skips_when_own_read_returns_none(self):
        f = _make()
        f.wf.plan = {'files': ['pkg/a.py']}
        f.scheduler.get_task = AsyncMock(return_value=None)
        spy = MagicMock()
        f.wf._escalate_scope_invariant_violation = spy  # type: ignore[method-assign]

        await f.wf._check_scope_invariant()

        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_entry_issues_one_get_task_for_stamp_and_scope_check(
        self, monkeypatch,
    ):
        """Drive the real merge entry with the REAL stamp and REAL scope check
        and count backend reads at the moment the check completes.

        Before the amendment: stamp read (1) + scope-check read (2). After:
        the scope check reuses the stamp's blob, so the count is still 1.
        (Reads by the retry-loop re-stamp land after the check and are
        deliberately outside this measurement.)
        """
        f = _make(backend_metadata={'files': ['other/b.py']})
        wf = f.wf
        wf.plan = {'files': ['pkg/a.py']}
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

        reads_at_scope_check: list[int] = []
        real_scope = wf._check_scope_invariant

        async def _spy_scope(**kwargs):
            await real_scope(**kwargs)
            reads_at_scope_check.append(f.scheduler.get_task.await_count)

        wf._check_scope_invariant = _spy_scope  # type: ignore[method-assign]
        escalate = MagicMock()
        wf._escalate_scope_invariant_violation = escalate  # type: ignore[method-assign]

        wf._enter_phase = MagicMock()  # type: ignore[method-assign]
        wf._maybe_defer_as_train_member = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
        wf._recover_before_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf.git_ops.get_main_sha = AsyncMock(return_value='main_sha')
        wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)
        wf._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )

        assert await wf._run_merge_phase('task/77') is None

        assert reads_at_scope_check == [1], (
            f'stamp + scope check must share ONE get_task; got '
            f'{reads_at_scope_check}'
        )
        # The shared blob still drove a real comparison, not a skipped one.
        # task 3429: third positional arg is the uncovered module list.
        escalate.assert_called_once_with(
            ['pkg/a.py'], ['other/b.py'], ['pkg/a.py'],
        )


# ---------------------------------------------------------------------------
# step-13: the stamp must survive into merge attempt 2+ (review-fix R2-B)
# ---------------------------------------------------------------------------


class TestRetryLoopReStamp:
    @pytest.mark.asyncio
    async def test_stamp_refreshed_at_top_of_each_merge_attempt(self, monkeypatch):
        """Every pre-enqueue window starts with a FRESH durable stamp.

        The merge-entry stamp is written once, ABOVE the ``for _merge_attempt
        in range(max_merge_retries)`` loop, while ``_submit_to_merge_queue``
        discharges it INSIDE that loop at the enqueue boundary. So once the
        clear actually deletes (step-12), attempt 2+ runs its entire
        pre-enqueue window — Phase-1 rebase plus a full scoped re-verify,
        minutes long, zero LLM calls and therefore no fresh ``routing.latest``
        either — with NO liveness signal at all, recurring exactly the
        divergence false positive this task exists to suppress. (A REQUEUED
        retry passes through a steward resolution first, so the merge-entry L0
        has already aged past ``orphan_l0_timeout_secs`` by then.)

        The in-memory ``note_merge_phase_entered`` is already re-stamped at the
        loop top for precisely this reason; the durable one must be symmetric.

        RED on main: snapshot 2 carries no ``merge_phase_liveness``.
        """
        import copy

        f = _make(metadata={'retry_ledger': {'x': 1}})
        wf = f.wf
        wf.plan = {'files': ['a.py']}
        wf.event_store = None
        wf.config.max_merge_retries = 2
        wf.config.max_pre_merge_retries = 1

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'branch_head_sha\n', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=_PASSING_VERIFY_RESULT),
        )

        wf._maybe_defer_as_train_member = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf._enter_phase = MagicMock()  # type: ignore[method-assign]
        wf._check_scope_invariant = AsyncMock()  # type: ignore[method-assign]
        wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
        wf._recover_before_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf._check_merge_outcome_thrash = AsyncMock(return_value=None)  # type: ignore[method-assign]
        wf.git_ops.get_main_sha = AsyncMock(return_value='main_sha')
        wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)
        # False → the "branch landed on main after steward resolution" break
        # does not fire, so the loop actually reaches attempt 2.
        wf.git_ops.is_ancestor = AsyncMock(return_value=False)

        # The REAL stamp/clear helpers run; the stamp is only wrapped so calls
        # are counted. get_task mirrors the in-memory blob so the
        # read-modify-write round-trips coherently.
        f.scheduler.get_task = AsyncMock(
            side_effect=lambda *_a, **_kw: {
                'metadata': copy.deepcopy(wf.task.get('metadata') or {}),
            },
        )
        real_stamp = wf._stamp_merge_phase_entered
        stamp_spy = AsyncMock(side_effect=real_stamp)
        wf._stamp_merge_phase_entered = stamp_spy  # type: ignore[method-assign]

        snapshots: list[dict] = []
        outcomes = [WorkflowOutcome.REQUEUED, WorkflowOutcome.DONE]

        async def _submit(*_a, **_kw):
            # Snapshot the metadata the reaper would see during this attempt's
            # pre-enqueue window, then faithfully model the production
            # enqueue-boundary clear (_submit_to_merge_queue).
            snapshots.append(copy.deepcopy(wf.task.get('metadata') or {}))
            await wf._clear_merge_phase_entered()
            return outcomes[len(snapshots) - 1]

        wf._submit_to_merge_queue = AsyncMock(side_effect=_submit)  # type: ignore[method-assign]

        result = await wf._run_merge_phase('task/77')

        assert result is None            # attempt 2 returned DONE → break
        assert len(snapshots) == 2       # the loop really ran twice

        stamps = []
        for i, snap in enumerate(snapshots, start=1):
            assert 'merge_phase_liveness' in snap, (
                f'merge attempt {i} ran its pre-enqueue window with no durable '
                f'liveness stamp: {snap}'
            )
            entered_at = snap['merge_phase_liveness']['entered_at']
            assert isinstance(entered_at, str) and entered_at
            stamps.append(entered_at)

        # A genuine re-stamp, not a survivor of the attempt-1 write.
        assert stamps[0] != stamps[1], (
            f'expected a fresh entered_at per attempt; got {stamps}'
        )
        # merge entry + once per attempt.
        assert stamp_spy.await_count == 3, (
            f'expected 3 stamp writes (entry + 2 attempts); '
            f'got {stamp_spy.await_count}'
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
