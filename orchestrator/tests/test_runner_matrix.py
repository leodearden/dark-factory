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
        self, patch_load_task, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """CancelledError from an inner eval propagates out of run_eval_matrix.

        This test FAILS against the un-patched runner.py because the current
        ``isinstance(r, BaseException)`` branch logs 'Eval failed' and then
        discards the exception; pytest.raises(CancelledError) never sees it.
        """
        task_path = tmp_path / 'task_a.json'
        task_path.touch()

        async def fake_run_eval(*args, **kwargs):
            raise asyncio.CancelledError('simulated cancel')

        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'), pytest.raises(asyncio.CancelledError):
            await run_eval_matrix(
                [task_path],
                [_CFG],
                force=True,
            )

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
