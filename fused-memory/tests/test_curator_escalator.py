"""Tests for the curator → orchestrator escalation router (R2).

``CuratorEscalator`` decides what happens when the curator's LLM call
fails. The policy is:

* If the project's orchestrator holds its exclusive lock → queue a
  level-1 escalation; the watcher will run ``/unblock``.
* If no orchestrator is running → raise :class:`CuratorFailureError` so
  the interactive MCP caller sees the outage loudly.

These tests exercise both branches plus the 1-hour per-project cooldown
that suppresses spam from a stuck curator.
"""

from __future__ import annotations

import fcntl
from typing import IO, Any, Literal, overload

import pytest

from fused_memory.middleware.curator_escalator import CuratorEscalator
from fused_memory.middleware.task_curator import CuratorFailureError


@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[True]) -> IO[bytes]: ...
@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[False]) -> None: ...
def _make_orchestrator_layout(root, *, hold_lock: bool) -> IO[bytes] | None:
    """Create the orchestrator.lock file; optionally hold LOCK_EX on it.

    Returns the open file handle when ``hold_lock=True`` so the caller
    can keep the lock alive for the duration of the test. When
    ``hold_lock=False`` the file exists but nothing holds an exclusive
    lock — matching the "orchestrator not running" case.
    """
    lock_dir = root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / 'orchestrator.lock'
    lock_path.write_text('')
    if not hold_lock:
        return None
    handle = lock_path.open('r+b')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


class TestOrchestratorLivenessProbe:
    def test_missing_lock_file_reports_not_running(self, tmp_path):
        escalator = CuratorEscalator()
        # No lock file created — treat as "no orchestrator".
        assert escalator._orchestrator_running(str(tmp_path)) is False

    def test_unlocked_file_reports_not_running(self, tmp_path):
        _make_orchestrator_layout(tmp_path, hold_lock=False)
        escalator = CuratorEscalator()
        assert escalator._orchestrator_running(str(tmp_path)) is False

    def test_held_exclusive_lock_reports_running(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator()
            assert escalator._orchestrator_running(str(tmp_path)) is True
        finally:
            handle.close()


class TestReportFailure:
    @pytest.mark.asyncio
    async def test_no_orchestrator_raises(self, tmp_path):
        escalator = CuratorEscalator()
        with pytest.raises(CuratorFailureError):
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='boom',
                candidate_title='some candidate',
            )

    @pytest.mark.asyncio
    async def test_orchestrator_running_queues_escalation(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator()
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='max_turns exhausted',
                candidate_title='Add Type::Error arm',
            )

            queue_dir = tmp_path / 'data' / 'escalations'
            files = sorted(queue_dir.glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            assert 'curator_failure' in body
            assert '"level": 1' in body
            assert 'max_turns exhausted' in body
            assert 'Add Type::Error arm' in body
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_first_three_escalate_then_suppress(self, tmp_path):
        """Policy: first three failures in window escalate; 4th+ are suppressed."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            for _ in range(5):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            # First three calls produce queue files; 4th and 5th are suppressed.
            assert len(files) == 3
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_third_escalation_carries_suppression_note(self, tmp_path):
        """The third escalation embeds an explicit 'further suppressed' note
        with an absolute resume-time timestamp so operators can see the
        window boundary without reading logs.
        """
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            for _ in range(3):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3
            third = files[-1].read_text()
            assert 'Further curator failures will be suppressed' in third
            # ISO-8601 UTC timestamp with timezone offset should be present.
            assert '+00:00' in third
            # Burst count N of 3 should be visible on each escalation.
            first = files[0].read_text()
            second = files[1].read_text()
            assert 'failures_in_window=1 of 3' in first
            assert 'failures_in_window=2 of 3' in second
            assert 'failures_in_window=3 of 3' in third
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_fourth_failure_within_window_is_suppressed(self, tmp_path):
        """A failure after the 3rd within cooldown does not enqueue."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            for _ in range(4):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3  # 4th was suppressed

            # Another call also remains suppressed.
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='still broken',
                candidate_title='T',
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_window_reset_after_cooldown_re_escalates(self, tmp_path):
        """A failure past the cooldown cutoff resets the counter."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            # Short cooldown so we don't need to mock monotonic clocks.
            escalator = CuratorEscalator(cooldown_secs=0.05)
            for _ in range(3):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='burst-1',
                    candidate_title='T',
                )
            # 4th call inside the short window is suppressed…
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-x',
                justification='burst-1', candidate_title='T',
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3

            # Sleep past the cooldown and confirm the next failure escalates
            # again (window reset).
            import asyncio
            await asyncio.sleep(0.1)
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-x',
                justification='burst-2', candidate_title='T',
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 4
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_escalation_detail_includes_timed_out_and_duration(
        self, tmp_path,
    ):
        """When the curator attaches timed_out/duration_ms, surface them in
        the escalation JSON so operators see '240s timeout' not just
        'produced no output'.
        """
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='timed out at 240s',
                candidate_title='T',
                timed_out=True,
                duration_ms=240003,
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            assert 'timed_out=True' in body
            assert 'duration_ms=240003' in body
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_zero_cooldown_queues_each_call(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=0.0)
            for _ in range(2):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 2
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_separate_projects_dont_share_cooldown(self, tmp_path):
        """Cooldown is per-project. A noisy project A must not silence project B."""
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        handle_a = _make_orchestrator_layout(root_a, hold_lock=True)
        handle_b = _make_orchestrator_layout(root_b, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(root_a), project_id='proj-a',
                justification='boom', candidate_title='T',
            )
            await escalator.report_failure(
                project_root=str(root_b), project_id='proj-b',
                justification='boom', candidate_title='T',
            )
            assert len(list((root_a / 'data' / 'escalations').glob('esc-*.json'))) == 1
            assert len(list((root_b / 'data' / 'escalations').glob('esc-*.json'))) == 1
        finally:
            handle_a.close()
            handle_b.close()


class TestSchemaToolDeniedEscalation:
    """CLI-semantics-break guard (CLI 2.1.168): a ``schema_tool_denied`` failure
    is a systemic config break (the cli_invoke deny-list no longer permits the
    ``StructuredOutput`` schema tool), NOT a flaky candidate.  It must ALWAYS
    surface — bypassing the first-3/hr burst suppression — with a distinct,
    self-describing summary that names the concrete fix location.
    """

    @pytest.mark.asyncio
    async def test_bypasses_burst_suppression(self, tmp_path):
        """Even as the 4th+ failure in a window (where ordinary failures are
        suppressed), a schema-tool-denied failure still enqueues an escalation."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            # Exhaust the window with three ordinary failures (all escalate).
            for _ in range(3):
                await escalator.report_failure(
                    project_root=str(tmp_path), project_id='proj-x',
                    justification='ordinary', candidate_title='T',
                )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3
            # A normal 4th would be suppressed; schema-tool-denied must NOT be.
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-x',
                justification='StructuredOutput denied x4', candidate_title='T',
                schema_tool_denied=True,
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 4
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_distinct_summary_and_fix_location(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-x',
                justification='StructuredOutput denied x4', candidate_title='T',
                schema_tool_denied=True,
            )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            # Distinct, self-describing summary naming the schema tool — must be
            # unmistakable vs the generic 'curator LLM failing' escalation.
            assert 'StructuredOutput' in body
            # ...and the concrete fix location for whoever picks it up.
            assert 'shared/cli_invoke.py' in body
        finally:
            handle.close()


class TestZeroOutputTimeoutEscalation:
    """Step-3 RED: a zero-output/full-timeout hang is a distinct INFRA event,
    not a flaky candidate.  It must ALWAYS surface (bypassing the 1-hour burst
    window) with a self-describing summary + forensic evidence.
    """

    @pytest.mark.asyncio
    async def test_bypasses_burst_suppression(self, tmp_path):
        """Even as the 4th failure in the window, a ZOT failure still enqueues."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            # Exhaust the window with three ordinary failures.
            for _ in range(3):
                await escalator.report_failure(
                    project_root=str(tmp_path), project_id='proj-z',
                    justification='ordinary', candidate_title='T',
                )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 3
            # A normal 4th would be suppressed; ZOT must NOT be.
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-z',
                justification='ZOT x4', candidate_title='T',
                zero_output_timeout=True,
                account_name='max-g',
                proc_tree='TREE-XYZ',
                timed_out=True,
                duration_ms=181_600,
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 4
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_distinct_summary_and_evidence(self, tmp_path):
        """ZOT escalation body: distinct summary + category + forensic evidence."""
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(tmp_path), project_id='proj-z',
                justification='ZOT', candidate_title='T',
                zero_output_timeout=True,
                account_name='max-g',
                proc_tree='TREE-XYZ',
                timed_out=True,
                duration_ms=181_600,
            )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            # Distinct summary — must be unmistakable vs generic 'curator LLM failing'.
            assert 'zero-output' in body or 'full-timeout' in body or 'infra hang' in body.lower()
            # Distinct category.
            assert 'curator_zero_output_hang' in body
            # Forensic evidence.
            assert 'max-g' in body
            assert 'TREE-XYZ' in body
            assert '181600' in body
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_zot_dedup_within_short_window(self, tmp_path):
        """Multiple rapid ZOT reports for the same project enqueue only the first.

        A batch of N candidates all hitting ZOT bisects to N concurrent size-1
        curate() calls, each of which calls report_failure independently.  The
        dedup window (60 s) ensures only one escalation is submitted per outage
        event so the queue is not flooded.
        """
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            common: dict[str, Any] = dict(
                project_root=str(tmp_path), project_id='proj-dedup',
                zero_output_timeout=True, timed_out=True, duration_ms=181_000,
                account_name='max-g', proc_tree='TREE-DEDUP',
            )
            # First report: should be submitted.
            await escalator.report_failure(
                justification='ZOT 1', candidate_title='Alpha', **common,
            )
            # Rapid follow-ups (simulating concurrent batch bisect): should be deduped.
            await escalator.report_failure(
                justification='ZOT 2', candidate_title='Beta', **common,
            )
            await escalator.report_failure(
                justification='ZOT 3', candidate_title='Gamma', **common,
            )
            files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 1, (
                f'Expected 1 escalation (dedup), got {len(files)}'
            )
        finally:
            handle.close()


# ----------------------------------------------------------------------
# Burst log keyed by (project_id, subtype) — independent counters
# ----------------------------------------------------------------------


class TestBurstLogCompositeKey:
    """Lever 2b — different subtypes for the same project are counted
    independently, so an error_max_budget_usd trip doesn't absorb burst
    quota meant for error_max_turns or vice-versa.
    """

    @pytest.mark.asyncio
    async def test_different_subtypes_have_independent_burst_counters(self, tmp_path):
        """Two report_failure calls for the same project with DIFFERENT subtypes
        should each produce 'failures_in_window=1 of 3' — independent counters.
        Under the old project_id-only keying the second call would read '2 of 3'.
        """
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            # First failure: subtype='error_max_budget_usd'.
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='budget exceeded',
                candidate_title='T',
                subtype='error_max_budget_usd',
            )
            # Second failure: different subtype for the same project.
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='turns exceeded',
                candidate_title='T',
                subtype='error_max_turns',
            )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            # Both subtypes should produce escalations.
            assert len(files) == 2
            bodies = [f.read_text() for f in files]
            # Each should be '1 of 3' — independent per-subtype counters.
            for body in bodies:
                assert 'failures_in_window=1 of 3' in body, (
                    f'Expected independent counter (1 of 3); got:\n{body}'
                )
        finally:
            handle.close()


# ----------------------------------------------------------------------
# Generic detail renders cost_usd= and pool_sizes= (Fix R-detail)
# ----------------------------------------------------------------------


class TestOverBudgetDetail:
    """Lever 2a — generic curator_failure detail includes cost_usd and pool_sizes.

    When report_failure is called with subtype='error_max_budget_usd',
    cost_usd, and pool_sizes, the queued escalation body (generic
    curator_failure path, NOT ZOT/schema-denied) must contain
    `cost_usd=` and `pool_sizes=` for operator triage.
    """

    @pytest.mark.asyncio
    async def test_generic_detail_includes_cost_usd_and_pool_sizes(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='budget',
                candidate_title='T',
                subtype='error_max_budget_usd',
                cost_usd=0.30574,
                pool_sizes={
                    'anchor': 1,
                    'module': 15,
                    'embedding': 10,
                    'dependency': 3,
                },
            )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            # Must be on the generic curator_failure path.
            assert 'curator_failure' in body
            # ZOT/schema-denied branches must NOT be triggered.
            assert 'curator_zero_output_hang' not in body
            assert 'curator_schema_tool_denied' not in body
            # cost_usd and pool_sizes must appear in the detail.
            assert 'cost_usd=' in body
            assert 'pool_sizes=' in body
        finally:
            handle.close()
