"""Tests for Harness._maybe_pipeline_landing_tripwire wiring (task 2382, merge-skew δ).

M3 of plans/merge-skew-attribution-prd.md — the proactive pipeline-landing
tripwire (PRD task δ, invariant I6, boundary rows 5-6).  This module tests
ONLY the harness-layer adapter that wires the pure
``orchestrator.merge_skew_tripwire.emit_pipeline_landing_tripwire`` into the
post-advance ``_note_merge_all`` landing hook: reading
``config.git.load_bearing_oracle_cmd``, enumerating in-flight tasks
(excluding the just-landed task), and injecting real
``git_ops.get_branch_changed_files`` / ``scheduler.update_task`` callables.

Mirrors test_harness_offline_lane_trigger.py's lightweight-harness
construction and test_merge_skew_tripwire.py's fake escalation queue
(duplicated per-file per that module's documented convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness


class _FakeEscalationQueue:
    """Minimal fake escalation queue — see test_merge_skew_tripwire.py's
    ``_FakeEscalationQueue`` for the shared rationale (per-file duplication
    convention: test_merge_queue_lifecycle_registry.py:86 /
    test_multihost_verify_integration.py:639)."""

    def __init__(self) -> None:
        self._seq = 0
        self.submitted: list[Any] = []

    def get_by_task(self, task_id: str, status: str | None = None) -> list:
        return []

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{task_id}-{self._seq}'

    def submit(self, esc: Any) -> None:
        self.submitted.append(esc)


def _write_oracle_script(project_root: Path, *, exit_code: int) -> Path:
    """Write a real executable bash oracle script exiting *exit_code*."""
    script = project_root / 'oracle.sh'
    script.write_text(f"""\
#!/usr/bin/env bash
exit {exit_code}
""")
    script.chmod(0o755)
    return script


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps, modeled on
    test_harness_offline_lane_trigger.py / test_harness_service_restart.py."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.event_store = MagicMock()
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    h._escalation_queue = _FakeEscalationQueue()  # type: ignore[assignment]
    return h


@pytest.mark.asyncio
class TestMaybePipelineLandingTripwire:
    """_maybe_pipeline_landing_tripwire: the harness adapter over
    emit_pipeline_landing_tripwire."""

    async def test_oracle_positive_escalates_only_overlapping_and_excludes_landing_task(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """Boundary row 5: an oracle-positive landing with one overlapping and
        one non-overlapping in-flight task escalates exactly once, naming
        only the overlapping task; the just-landed task is excluded from the
        in-flight scan entirely (its branch is never even diffed)."""
        script = _write_oracle_script(tmp_path, exit_code=0)  # load-bearing
        harness.config.git.load_bearing_oracle_cmd = ['bash', str(script)]

        harness.scheduler.get_tasks = AsyncMock(return_value=[
            {'id': '999', 'status': 'in-progress'},  # the just-landed task
            {'id': '101', 'status': 'in-progress'},  # overlapping
            {'id': '202', 'status': 'in-progress'},  # non-overlapping
        ])

        async def get_branch_changed_files(ref: str):
            if ref == 'task/999':
                raise AssertionError(
                    'the just-landed task must be excluded from the '
                    'in-flight scan, not diffed'
                )
            if ref == 'task/101':
                return (['src/a.py', 'unrelated_101.py'], None)
            if ref == 'task/202':
                return (['unrelated_202.py'], None)
            raise AssertionError(f'unexpected ref {ref!r}')

        harness.git_ops.get_branch_changed_files = AsyncMock(
            side_effect=get_branch_changed_files,
        )

        landing_sha = 'f' * 40
        await harness._maybe_pipeline_landing_tripwire(
            '999', 'base-sha', landing_sha, ['src/a.py'],
        )

        fake_eq: _FakeEscalationQueue = harness._escalation_queue  # type: ignore[assignment]
        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation; got {fake_eq.submitted!r}'
        )
        esc = fake_eq.submitted[0]
        assert esc.severity == 'info'
        assert '101' in esc.summary or '101' in esc.detail
        assert '202' not in esc.summary and '202' not in esc.detail

        harness.scheduler.update_task.assert_awaited_once()  # type: ignore[attr-defined]
        call = harness.scheduler.update_task.await_args  # type: ignore[attr-defined]
        assert call.args[0] == '101', (
            f'update_task must be called for the overlapping task only; got {call.args[0]!r}'
        )

    async def test_uses_configured_branch_prefix_not_hardcoded_task_slash(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """A project that customizes config.git.branch_prefix must still
        resolve real in-flight branch refs — the adapter must NOT hardcode
        the ``task/`` literal, or every in-flight task silently fails to
        diff (get_branch_changed_files errors -> None -> skipped -> no
        escalation, with no diagnostic)."""
        script = _write_oracle_script(tmp_path, exit_code=0)  # load-bearing
        harness.config.git.load_bearing_oracle_cmd = ['bash', str(script)]
        harness.config.git.branch_prefix = 'feature/'

        harness.scheduler.get_tasks = AsyncMock(return_value=[
            {'id': '999', 'status': 'in-progress'},  # the just-landed task
            {'id': '101', 'status': 'in-progress'},  # overlapping
        ])

        async def get_branch_changed_files(ref: str):
            if ref == 'feature/101':
                return (['src/a.py'], None)
            raise AssertionError(
                f'expected the configured feature/ prefix, got ref {ref!r}'
            )

        harness.git_ops.get_branch_changed_files = AsyncMock(
            side_effect=get_branch_changed_files,
        )

        await harness._maybe_pipeline_landing_tripwire(
            '999', 'base-sha', 'e' * 40, ['src/a.py'],
        )

        fake_eq: _FakeEscalationQueue = harness._escalation_queue  # type: ignore[assignment]
        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation using the configured branch '
            f'prefix; got {fake_eq.submitted!r}'
        )
        harness.scheduler.update_task.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_oracle_cmd_none_short_circuits_before_get_tasks(
        self, harness: Harness,
    ) -> None:
        """config.git.load_bearing_oracle_cmd is None (default) -> no
        escalation, no oracle call, and no scheduler.get_tasks call at all —
        the tripwire must be a clean no-op ahead of any work."""
        assert harness.config.git.load_bearing_oracle_cmd is None

        await harness._maybe_pipeline_landing_tripwire(
            '999', 'base-sha', 'f' * 40, ['src/a.py'],
        )

        fake_eq: _FakeEscalationQueue = harness._escalation_queue  # type: ignore[assignment]
        assert fake_eq.submitted == []
        harness.scheduler.get_tasks.assert_not_awaited()  # type: ignore[attr-defined]
        harness.scheduler.update_task.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_note_merge_all_invokes_tripwire_and_raise_does_not_block_coordinators(
        self, harness: Harness,
    ) -> None:
        """_note_merge_all must invoke the tripwire, and a raising tripwire
        must not prevent the service-restart coordinator fan-out (fail-open,
        I6: the tripwire can never delay/break the landing hot path)."""
        harness._maybe_pipeline_landing_tripwire = AsyncMock(
            side_effect=RuntimeError('tripwire boom'),
        )
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        # Must NOT raise.
        await harness._note_merge_all('task-9', 'base9', 'head9')

        harness._maybe_pipeline_landing_tripwire.assert_awaited_once_with(
            'task-9', 'base9', 'head9', [],
        )
        coord_a.note_merge.assert_awaited_once_with(
            'task-9', 'base9', 'head9', prefetched_diff=[],
        )
        coord_b.note_merge.assert_awaited_once_with(
            'task-9', 'base9', 'head9', prefetched_diff=[],
        )
