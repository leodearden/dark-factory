"""Tests for Harness's leftover ``df-verify-*.scope`` reaper wiring (task 2829).

Companion to reify flipping ``verify_use_cgroup_scope: true``. Mirrors
test_harness_interactive_reaper.py's structure:

  step-07/08 — ``_run_leftover_verify_scope_reaper_pass()`` delegates to
               ``verify.reap_leftover_verify_scopes(self.config.project_root)``,
               logs one INFO line per reaped unit + a summary INFO line, and
               never raises.
  step-09/10 — an unconditional sweep runs once at ``run()`` startup, before
               first dispatch.
  step-11    — the task's mandated real-systemd live-sleeper acceptance signal
               (skip-guarded): the tagged scope is stopped while a
               differently-tagged sibling scope SURVIVES (cross-project safety).
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factory (mirrors test_harness_interactive_reaper._make_harness)
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore."""
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-verify-scope-reaper-0001'
    harness.event_store = EventStore(
        tmp_path / 'events.db', 'run-verify-scope-reaper-0001',
    )
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# Step-07: _run_leftover_verify_scope_reaper_pass delegates to
# verify.reap_leftover_verify_scopes, logs per-unit + a summary, never raises.
# ---------------------------------------------------------------------------


class TestRunLeftoverVerifyScopeReaperPass:
    """RED until step-08 adds _run_leftover_verify_scope_reaper_pass to Harness."""

    @pytest.mark.asyncio
    async def test_pass_delegates_and_logs_per_unit_and_summary(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Awaits reap_leftover_verify_scopes(project_root) once; logs one INFO
        line naming each reaped unit + a summary INFO line with the count."""
        harness, _rs = _make_harness(tmp_path)
        units = ['df-verify-proj-1a2b3c4d-abcabcabcabc.scope']
        mock_reap = AsyncMock(return_value=units)

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.INFO, logger='orchestrator.harness'),
        ):
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once_with(harness.config.project_root)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(units[0] in r.getMessage() for r in info_records), (
            f'expected an INFO line naming the reaped unit {units[0]!r}; '
            f'got: {[r.getMessage() for r in info_records]}'
        )
        assert any(
            'reaper' in r.getMessage().lower() and str(len(units)) in r.getMessage()
            for r in info_records
        ), (
            'expected a summary INFO line naming the reaped count; '
            f'got: {[r.getMessage() for r in info_records]}'
        )

    @pytest.mark.asyncio
    async def test_pass_logs_debug_when_nothing_reaped(
        self, tmp_path: Path, caplog,
    ) -> None:
        """No leftovers -> no INFO summary noise (DEBUG instead), never raises."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(return_value=[])

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.INFO, logger='orchestrator.harness'),
        ):
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once_with(harness.config.project_root)
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert not info_records, (
            f'expected NO INFO lines when nothing was reaped; '
            f'got: {[r.getMessage() for r in info_records]}'
        )

    @pytest.mark.asyncio
    async def test_pass_swallows_exception_and_logs_error(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A raising reap_leftover_verify_scopes() does not propagate; a bounded
        error line is logged instead."""
        harness, _rs = _make_harness(tmp_path)
        mock_reap = AsyncMock(side_effect=RuntimeError('boom'))

        with (
            patch('orchestrator.verify.reap_leftover_verify_scopes', mock_reap),
            caplog.at_level(logging.ERROR, logger='orchestrator.harness'),
        ):
            # Must not raise.
            await harness._run_leftover_verify_scope_reaper_pass()

        mock_reap.assert_awaited_once()
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, (
            'expected an ERROR log when reap_leftover_verify_scopes raises'
        )
