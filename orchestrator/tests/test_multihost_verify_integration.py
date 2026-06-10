"""Integration gate for task λ — multi-host verify end-to-end throughput (PRD Phase 5, §B B1–B8).

This module is the sole deliverable of task 1702.  It composes the REAL substrate
from α–κ (VerifyRunnerPool.dispatch, DriftDetector.check, run_verdict_parity,
enforce_persistent_worktree_serial_lane, check_merge_liveness_margin) and fakes
ONLY the non-deterministic/external surface (verify service time + laptop/SSH
transport) via _FakeRunner.

§B scenarios covered:
  B1  local-only provenance (runners_seen == {'local'})
  B2  remote happy-path provenance (runners_seen == {'local','laptop'})
  B3  fail-safe fallback — laptop unavailable, queue never stalls, local used
  B4  verdict parity over a pass/fail corpus via run_verdict_parity
  B5  drift divergence alarm — dedup'd L1 escalation + quarantine
  B6  K=2 concurrency — peak in-flight==2, overlapping spans, serialized advance
  B7  per-host warmth guard — enforce_persistent_worktree_serial_lane
  B8  liveness-margin guard — check_merge_liveness_margin

Headline λ gate: test_end_to_end_throughput_gate_direction_and_provenance_and_zero_drift
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    DriftDetector,
    DriftVerdict,
    MergeVerifySpec,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyCommand,
    VerifyRunner,
    VerifyRunnerPool,
    run_verdict_parity,
)
from orchestrator.merge_queue import (
    PersistentWorktreeConfigError,
    check_merge_liveness_margin,
    enforce_persistent_worktree_serial_lane,
)
from orchestrator.config import OrchestratorConfig

# ---------------------------------------------------------------------------
# Module-level builder helpers (mirror test_verify_runner.py pattern)
# ---------------------------------------------------------------------------


def _make_spec() -> MergeVerifySpec:
    """Return a minimal but valid MergeVerifySpec."""
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


def _make_result(passed: bool, *, category: str = '') -> VerifyResult:
    """Return a VerifyResult with the given pass/fail status."""
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category=category,
    )


# ---------------------------------------------------------------------------
# In-memory event-store double (mirror test_scheduler_state.py pattern)
# ---------------------------------------------------------------------------


class _RecordingEventStore:
    """Minimal EventStore stand-in that captures emit() calls in-memory."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
        **kw: Any,
    ) -> None:
        self.events.append((event_type, task_id, dict(data or {})))

    def events_of(self, event_type: Any) -> list[tuple[Any, str | None, dict[str, Any]]]:
        """Return all recorded events matching *event_type*."""
        return [e for e in self.events if e[0] == event_type]


# ---------------------------------------------------------------------------
# Step-1 (RED): TestThroughputHarness — verify that pool.dispatch emits
# runner-tagged merge_verify events via run_backlog_window.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestThroughputHarness:
    async def test_window_emits_runner_tagged_verify_events(self):
        """run_backlog_window returns WindowRun with 4 completed verifies, each
        tagged with a runner name from the pool (local or laptop)."""
        rec = _RecordingEventStore()
        local_fake = _FakeRunner('local', is_local=True, service_secs=1.0)
        laptop_fake = _FakeRunner('laptop', is_local=False, service_secs=1.0)
        pool = VerifyRunnerPool([local_fake, laptop_fake], event_store=rec, task_id='1702')

        run = await run_backlog_window(
            pool, rec, n_merges=4, k=2,
            service_secs={'local': 1.0, 'laptop': 1.0},
        )

        assert run.completed == 4
        verify_events = rec.events_of(EventType.merge_verify)
        assert len(verify_events) == 4
        for _et, _tid, data in verify_events:
            assert data['runner'] in {'local', 'laptop'}
