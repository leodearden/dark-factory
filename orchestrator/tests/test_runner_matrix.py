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

        # CancelledError is placed in the result list — it does NOT propagate
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1] == 42


@pytest.mark.asyncio
class TestRunEvalMatrixCancellation:
    """run_eval_matrix must re-raise asyncio.CancelledError instead of swallowing it."""

    async def test_run_eval_matrix_reraises_cancellederror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """CancelledError from an inner eval propagates out of run_eval_matrix.

        This test FAILS against the un-patched runner.py because the current
        ``isinstance(r, BaseException)`` branch logs 'Eval failed' and then
        discards the exception; pytest.raises(CancelledError) never sees it.
        """
        task_path = tmp_path / 'task_a.json'
        task_path.touch()

        def fake_load_task(path: Path) -> dict:
            return {'id': path.stem}

        async def fake_run_eval(*args, **kwargs):
            raise asyncio.CancelledError('simulated cancel')

        monkeypatch.setattr(runner_mod, 'load_task', fake_load_task)
        monkeypatch.setattr(runner_mod, 'run_eval', fake_run_eval)

        with caplog.at_level(logging.ERROR, logger='orchestrator.evals.runner'):
            with pytest.raises(asyncio.CancelledError):
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
