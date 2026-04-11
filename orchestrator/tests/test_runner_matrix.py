"""Tests for run_eval_matrix exception-handling loop.

Covers:
  - asyncio.gather stdlib contract (foundation test)
  - CancelledError propagation (TDD: fails before fix, passes after)
  - Non-cancel exception log-and-continue regression guard
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from unittest.mock import MagicMock

import orchestrator.evals.runner as runner_mod
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.runner import EvalResult, run_eval_matrix

_CFG = EvalConfig(name='test-cfg', backend='claude', model='sonnet', effort='high')


def _stem_loader(path: Path) -> dict:
    """Minimal task dict used by all matrix tests.

    Safe because every caller passes ``force=True``, which skips
    ``_result_exists`` — the only place in the non-mocked ``_run_one``
    closure that reads ``task['id']``.  ``run_eval`` is separately
    monkeypatched in each test, so the dict is never deeply inspected.
    """
    return {'id': path.stem}


@pytest.fixture()
def patch_load_task(monkeypatch: pytest.MonkeyPatch):
    """Patch ``runner_mod.load_task`` with the shared _stem_loader stub."""
    monkeypatch.setattr(runner_mod, 'load_task', _stem_loader)


@pytest.fixture()
def single_task_matrix_case(
    patch_load_task,  # noqa: ARG001 — applied for side-effect (load_task stub)
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Shared setup for single-task run_eval_matrix tests.

    Creates a single task file at ``tmp_path / 'task_a.json'`` and returns
    ``(task_path, run_matrix_helper)``.

    ``run_matrix_helper(fake_run_eval)`` monkeypatches ``runner_mod.run_eval``
    with *fake_run_eval* and awaits
    ``run_eval_matrix([task_path], [_CFG], force=True)``.

    Depends on ``patch_load_task`` so ``runner_mod.load_task`` is already
    stubbed before the helper runs — callers do not need to patch it separately.
    """
    task_path = tmp_path / 'task_a.json'
    task_path.touch()

    async def run_matrix_helper(fake_run_eval):
        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)
        return await run_eval_matrix([task_path], [_CFG], force=True)

    return task_path, run_matrix_helper


@pytest.mark.asyncio
class TestGatherContract:
    """Documents asyncio.gather(..., return_exceptions=True) stdlib contract.

    This test pins behavior that run_eval_matrix relies on.  It should always
    pass; a failure here indicates a Python regression worth investigating.
    """

    async def test_gather_with_cancellederror_captures_in_results(self):
        """gather(return_exceptions=True) captures CancelledError in the result list."""

        async def _raise_cancel():
            raise asyncio.CancelledError('simulated')

        async def _return_value():
            return 42

        results = await asyncio.gather(
            _raise_cancel(),
            _return_value(),
            return_exceptions=True,
        )

        # CancelledError is placed in the result list — it does NOT propagate.
        # The two-coroutine form also verifies insertion-order preservation:
        # results[0] corresponds to _raise_cancel() (first arg) and
        # results[1] corresponds to _return_value() (second arg).
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1] == 42


@pytest.mark.asyncio
class TestRunEvalMatrixCancellation:
    """run_eval_matrix must re-raise asyncio.CancelledError instead of swallowing it."""

    async def test_run_eval_matrix_reraises_cancellederror(
        self, single_task_matrix_case, caplog: pytest.LogCaptureFixture
    ):
        """CancelledError from an inner eval propagates out of run_eval_matrix.

        This test FAILS against the un-patched runner.py because the current
        ``isinstance(r, BaseException)`` branch logs 'Eval failed' and then
        discards the exception; pytest.raises(CancelledError) never sees it.
        """
        _task_path, run_matrix = single_task_matrix_case

        async def fake_run_eval(*args, **kwargs):
            raise asyncio.CancelledError('simulated cancel')

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'), pytest.raises(asyncio.CancelledError):
            await run_matrix(fake_run_eval)

        assert any(
            'cancelled' in record.message.lower()
            for record in caplog.records
        ), (
            f'Expected a log record containing "cancelled". '
            f'Got: {[r.message for r in caplog.records]}'
        )

        # Negative guard: ensure the elif→if regression doesn't fire the 'failed' branch too.
        # If CancelledError were caught by the BaseException branch first, both 'cancelled'
        # and 'failed' would appear in the logs.
        assert not any(
            'failed' in record.message.lower()
            for record in caplog.records
        ), (
            f'Unexpected "failed" log record — elif→if routing regression detected. '
            f'Got: {[r.message for r in caplog.records]}'
        )

    async def test_cancel_wins_over_partial_results(
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """CancelledError discards any partial results — cancellation wins.

        With two tasks (one cancels, one succeeds), the gather loop raises
        CancelledError when it reaches the cancelled result, discarding the
        successful result that follows.  This locks in the 'cancel discards
        everything' contract.
        """
        cancel_path = tmp_path / 'task_cancel.json'
        ok_path = tmp_path / 'task_ok.json'
        cancel_path.touch()
        ok_path.touch()

        async def fake_run_eval(task_path: Path, config: EvalConfig, *args, **kwargs) -> EvalResult:
            if 'cancel' in task_path.stem:
                raise asyncio.CancelledError('simulated cancel')
            return EvalResult(
                task_id='task_ok',
                config_name=config.name,
                outcome='success',
                metrics={},
                worktree_path='/tmp/stub',
            )

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        with pytest.raises(asyncio.CancelledError):
            await run_eval_matrix(
                [cancel_path, ok_path],
                [_CFG],
                force=True,
            )


    async def test_siblings_cancelled_promptly_on_cancel(
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Slow siblings must be cancelled promptly when one eval raises CancelledError.

        With asyncio.gather(return_exceptions=True) the slow sibling runs to
        full completion (2 s) before CancelledError propagates, making total
        elapsed ≈ 2 s.

        After the asyncio.wait(FIRST_COMPLETED) refactor the monitor loop
        cancels the slow sibling immediately, so elapsed should be well under
        1 s (typically < 0.1 s).

        This test FAILS against the current gather implementation.
        """
        cancel_path = tmp_path / 'task_cancel.json'
        slow_path = tmp_path / 'task_slow.json'
        cancel_path.touch()
        slow_path.touch()

        slow_completed = False

        async def fake_run_eval(task_path: Path, config: EvalConfig, *args, **kwargs) -> EvalResult:
            nonlocal slow_completed
            if 'cancel' in task_path.stem:
                raise asyncio.CancelledError('immediate cancel')
            # Slow task: sleep 2 s then mark completion
            await asyncio.sleep(2.0)
            slow_completed = True
            return EvalResult(
                task_id='task_slow',
                config_name=config.name,
                outcome='success',
                metrics={},
                worktree_path='/tmp/stub',
            )

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        import time
        start = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await run_eval_matrix(
                [cancel_path, slow_path],
                [_CFG],
                force=True,
            )
        elapsed = time.monotonic() - start

        # (a) Slow sibling must NOT have run to completion
        assert not slow_completed, 'Slow sibling ran to completion — siblings were not cancelled promptly'

        # (b) Total elapsed must be well under 2 s (gather would take ~2 s)
        assert elapsed < 1.0, f'Elapsed {elapsed:.2f}s ≥ 1.0s — siblings were not cancelled promptly'

    async def test_external_cancel_cleans_up_all_tasks(
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """External cancellation (e.g. SIGINT / asyncio.wait_for timeout) cleans up all tasks.

        Wraps run_eval_matrix in asyncio.wait_for with a 0.3s timeout while
        all evals sleep for 10s.  Asserts that:
        (a) asyncio.TimeoutError (which wraps CancelledError in Python 3.11+)
            is raised — the external cancel propagates out.
        (b) A cancellation-tracking flag confirms that all tasks received
            cancellation before run_eval_matrix returned.
        """
        path1 = tmp_path / 'task_a.json'
        path2 = tmp_path / 'task_b.json'
        path3 = tmp_path / 'task_c.json'
        path1.touch()
        path2.touch()
        path3.touch()

        cancelled_count = 0
        started_count = 0

        async def fake_run_eval(task_path: Path, config: EvalConfig, *args, **kwargs) -> EvalResult:
            nonlocal cancelled_count, started_count
            started_count += 1
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled_count += 1
                raise
            return EvalResult(
                task_id=task_path.stem,
                config_name=config.name,
                outcome='success',
                metrics={},
                worktree_path='/tmp/stub',
            )

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        # asyncio.wait_for raises TimeoutError (Python 3.11+) or CancelledError
        with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
            await asyncio.wait_for(
                run_eval_matrix(
                    [path1, path2, path3],
                    [_CFG],
                    force=True,
                ),
                timeout=0.3,
            )

        # All started tasks must have received cancellation
        assert started_count > 0, 'No tasks were started'
        assert cancelled_count == started_count, (
            f'Only {cancelled_count}/{started_count} tasks were cancelled — '
            'some tasks were left as orphans after external cancellation'
        )


    async def test_cancelled_error_log_carries_exc_info(
        self, single_task_matrix_case, caplog: pytest.LogCaptureFixture
    ):
        """CancelledError log record must carry exc_info (traceback attached).

        The message should be stable ('Eval cancelled') rather than including
        the f-string repr of the exception, so that it is useful even when
        str(CancelledError) is empty.

        This test would FAIL without the exc_info fix because the old
        ``logger.error(f'Eval cancelled: {r}')`` call did not set exc_info.
        """
        _task_path, run_matrix = single_task_matrix_case

        async def fake_run_eval(*args, **kwargs):
            raise asyncio.CancelledError('simulated cancel')

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'), pytest.raises(asyncio.CancelledError):
            await run_matrix(fake_run_eval)

        cancel_records = [r for r in caplog.records if 'cancelled' in r.message.lower()]
        assert cancel_records, (
            f'Expected at least one log record containing "cancelled". '
            f'Got: {[r.message for r in caplog.records]}'
        )
        record = cancel_records[0]

        # (a) Message must be exactly 'Eval cancelled' — no f-string interpolation
        assert record.message == 'Eval cancelled', (
            f"Expected message 'Eval cancelled', got {record.message!r}"
        )

        # (b) exc_info must be a 3-tuple (type, value, traceback) — not None
        assert isinstance(record.exc_info, tuple) and len(record.exc_info) == 3, (
            f'Expected exc_info to be a 3-tuple, got {record.exc_info!r}'
        )
        exc_type, exc_val, exc_tb = record.exc_info
        assert exc_type is asyncio.CancelledError, (
            f'Expected exc_type to be CancelledError, got {exc_type!r}'
        )
        assert isinstance(exc_val, asyncio.CancelledError), (
            f'Expected exc_val to be CancelledError instance, got {exc_val!r}'
        )


@pytest.mark.asyncio
class TestRunEvalMatrixNonCancelPath:
    """Non-cancel exceptions must be logged and the matrix must continue."""

    async def test_run_eval_matrix_logs_exception_and_returns_other_results(
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """RuntimeError from one eval is logged; results from other evals are returned.

        Regression guard: the 'Eval failed' log-and-continue behavior for normal
        eval failures must not be broken by the CancelledError fix.
        """
        fail_path = tmp_path / 'task_fail.json'
        ok_path = tmp_path / 'task_ok.json'
        fail_path.touch()
        ok_path.touch()

        async def fake_run_eval(task_path: Path, config: EvalConfig, *args, **kwargs) -> EvalResult:
            if 'fail' in task_path.stem:
                raise RuntimeError('simulated failure')
            return EvalResult(
                task_id='task_ok',
                config_name=config.name,
                outcome='success',
                metrics={},
                worktree_path='/tmp/stub',
            )

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'):
            results = await run_eval_matrix(
                [fail_path, ok_path],
                [_CFG],
                force=True,
            )

        # (a) One result returned — from the non-failing task
        assert len(results) == 1, f'Expected 1 result, got {len(results)}: {results}'

        # (b) The result is from the ok task
        assert results[0].task_id == 'task_ok'

        # (c) A 'failed' log record was emitted for the failing task
        assert any(
            'failed' in record.message.lower()
            for record in caplog.records
        ), (
            f'Expected a log record containing "failed". '
            f'Got: {[r.message for r in caplog.records]}'
        )

        # (d) No 'cancelled' log record (this was not a cancellation)
        assert not any(
            'cancelled' in record.message.lower()
            for record in caplog.records
        ), (
            f'Unexpected "cancelled" log record. '
            f'Got: {[r.message for r in caplog.records]}'
        )

    async def test_normal_exception_does_not_cancel_pending_siblings(
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """RuntimeError from one eval must NOT cancel slow sibling tasks.

        Three tasks: one raises RuntimeError immediately, two sleep 0.5s then
        return results.  The monitor loop must log-and-continue for the failed
        task and wait for both slow siblings to complete successfully.

        Guards against the monitor loop accidentally cancelling siblings on
        non-cancel exceptions.
        """
        fail_path = tmp_path / 'task_fail.json'
        slow1_path = tmp_path / 'task_slow1.json'
        slow2_path = tmp_path / 'task_slow2.json'
        fail_path.touch()
        slow1_path.touch()
        slow2_path.touch()

        async def fake_run_eval(task_path: Path, config: EvalConfig, *args, **kwargs) -> EvalResult:
            if 'fail' in task_path.stem:
                raise RuntimeError('immediate failure')
            # Slow tasks: sleep briefly then return a result
            await asyncio.sleep(0.1)
            return EvalResult(
                task_id=task_path.stem,
                config_name=config.name,
                outcome='success',
                metrics={},
                worktree_path='/tmp/stub',
            )

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'):
            results = await run_eval_matrix(
                [fail_path, slow1_path, slow2_path],
                [_CFG],
                force=True,
            )

        # (a) Both slow tasks completed and returned results
        result_ids = {r.task_id for r in results}
        assert 'task_slow1' in result_ids, f'task_slow1 missing from results: {result_ids}'
        assert 'task_slow2' in result_ids, f'task_slow2 missing from results: {result_ids}'

        # (b) Exactly 2 results (failed task does not produce a result)
        assert len(results) == 2, f'Expected 2 results, got {len(results)}: {results}'

        # (c) RuntimeError was logged for the failing task
        assert any(
            'failed' in record.message.lower()
            for record in caplog.records
        ), f'Expected "failed" log record. Got: {[r.message for r in caplog.records]}'

        # (d) No 'cancelled' log record
        assert not any(
            'cancelled' in record.message.lower()
            for record in caplog.records
        ), f'Unexpected "cancelled" log record. Got: {[r.message for r in caplog.records]}'

    async def test_failed_error_log_carries_exc_info(
        self, single_task_matrix_case, caplog: pytest.LogCaptureFixture
    ):
        """Non-cancel RuntimeError log record must carry exc_info (traceback attached).

        The message should be stable ('Eval failed') rather than including the
        f-string repr of the exception, so that it is useful even when
        str(exc) is empty.

        This test would FAIL against the old code because
        ``logger.error(f'Eval failed: {exc}')`` does not set exc_info and
        embeds the exception message in the log string instead.
        """
        _task_path, run_matrix = single_task_matrix_case

        async def fake_run_eval(*args, **kwargs):
            raise RuntimeError('simulated failure')

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'):
            await run_matrix(fake_run_eval)

        failed_records = [r for r in caplog.records if r.message == 'Eval failed']
        assert failed_records, (
            f'Expected at least one log record with message == "Eval failed". '
            f'Got: {[r.message for r in caplog.records]}'
        )
        record = failed_records[0]

        # (a) Message must be exactly 'Eval failed' — no f-string interpolation
        assert record.message == 'Eval failed', (
            f"Expected message 'Eval failed', got {record.message!r}"
        )

        # (b) exc_info must be a 3-tuple (type, value, traceback) — not None
        assert isinstance(record.exc_info, tuple) and len(record.exc_info) == 3, (
            f'Expected exc_info to be a 3-tuple, got {record.exc_info!r}'
        )
        exc_type, exc_val, exc_tb = record.exc_info

        # (c) exc_type must be RuntimeError
        assert exc_type is RuntimeError, (
            f'Expected exc_type to be RuntimeError, got {exc_type!r}'
        )

        # (d) exc_val must be a RuntimeError instance
        assert isinstance(exc_val, RuntimeError), (
            f'Expected exc_val to be RuntimeError instance, got {exc_val!r}'
        )

        # (e) exc_tb must be present — the whole point of exc_info is to preserve
        #     the traceback for post-mortem debugging
        assert exc_tb is not None, 'Expected traceback to be attached to the RuntimeError'


@pytest.mark.asyncio
class TestCollectCancelErrors:
    """Unit tests for the _collect_cancel_errors helper.

    These tests exercise the classification logic in isolation by constructing
    known done sets and asserting the helper's output — no scheduling or
    asyncio.wait semantics involved.
    """

    async def test_collects_all_cancels_from_real_tasks(self):
        """_collect_cancel_errors returns one CancelledError per cancelled task.

        TEST INTENT: Verify that when two real asyncio.Task objects have been
        .cancel()'d and awaited, _collect_cancel_errors returns a list with
        exactly two CancelledError instances.

        PASS/FAIL CONDITION: Fails with AttributeError if _collect_cancel_errors
        does not exist on runner_mod. Fails with assertion error if the returned
        list length != 2 or elements are not CancelledError instances.
        """

        async def long_sleep():
            await asyncio.sleep(3600)

        task_a = asyncio.create_task(long_sleep())
        task_b = asyncio.create_task(long_sleep())
        task_a.cancel()
        task_b.cancel()
        for t in (task_a, task_b):
            try:
                await t
            except asyncio.CancelledError:
                pass

        result = runner_mod._collect_cancel_errors({task_a, task_b})

        assert len(result) == 2, f'Expected 2 CancelledErrors, got {len(result)}: {result}'
        for i, err in enumerate(result):
            assert isinstance(err, asyncio.CancelledError), (
                f'Element {i}: expected CancelledError, got {err!r}'
            )

    async def test_collects_cancel_errors_from_defensive_branch_via_mock(self):
        """_collect_cancel_errors collects CancelledError via the defensive branch.

        TEST INTENT: Explicitly exercise the branch where task.cancelled()
        returns False but task.exception() returns a CancelledError instance —
        the belt-and-suspenders path that is unreachable via real CPython 3.11+
        coroutines but kept for hypothetical future runtimes.

        PASS/FAIL CONDITION: Passes if the returned list contains exactly the
        CancelledError instance returned by mock.exception().
        """
        ce = asyncio.CancelledError('defensive')
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = ce

        result = runner_mod._collect_cancel_errors({mock_task})

        assert len(result) == 1, f'Expected 1 CancelledError, got {len(result)}: {result}'
        assert result[0] is ce, (
            f'Expected the exact CancelledError from mock.exception(), got {result[0]!r}'
        )

    async def test_collects_mixed_cancel_branches(self):
        """_collect_cancel_errors collects from both the real and defensive branches.

        TEST INTENT: Build a done set with one real cancelled task (hits the
        task.cancelled() branch) and one MagicMock task (hits the defensive
        task.exception() branch), then verify that both CancelledErrors are
        collected. This is the deterministic, scheduling-free replacement for
        the gate-based test_multiple_simultaneous_cancellederrors_all_logged.

        PASS/FAIL CONDITION: Returns a list of length 2 containing two
        CancelledError instances.
        """

        async def long_sleep():
            await asyncio.sleep(3600)

        real_task = asyncio.create_task(long_sleep())
        real_task.cancel()
        try:
            await real_task
        except asyncio.CancelledError:
            pass

        mock_ce = asyncio.CancelledError('mock-branch')
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = mock_ce

        result = runner_mod._collect_cancel_errors({real_task, mock_task})

        assert len(result) == 2, f'Expected 2 CancelledErrors, got {len(result)}: {result}'
        for i, err in enumerate(result):
            assert isinstance(err, asyncio.CancelledError), (
                f'Element {i}: expected CancelledError, got {err!r}'
            )
