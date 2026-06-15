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
        import time as _time_module
        from unittest.mock import patch
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            # Control time.time() so the burst window is deterministic
            # regardless of I/O latency.  Writing 3 escalation files can
            # easily exceed a 50 ms real cooldown on loaded CI workers.
            fake_t = [0.0]
            with patch.object(_time_module, 'time', lambda: fake_t[0]):
                escalator = CuratorEscalator(cooldown_secs=3600.0)
                for _ in range(3):
                    await escalator.report_failure(
                        project_root=str(tmp_path),
                        project_id='proj-x',
                        justification='burst-1',
                        candidate_title='T',
                    )
                # 4th call at the same frozen time is inside the window → suppressed.
                await escalator.report_failure(
                    project_root=str(tmp_path), project_id='proj-x',
                    justification='burst-1', candidate_title='T',
                )
                files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
                assert len(files) == 3

                # Advance past the cooldown — next failure opens a fresh window.
                fake_t[0] = 3601.0
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


# ----------------------------------------------------------------------
# Restart-durable burst counter (step-11 RED / step-12 GREEN)
# ----------------------------------------------------------------------


class TestRestartDurableBurstCounter:
    """Lever 2b — the burst counter survives a process restart via state_path.

    When CuratorEscalator is constructed with a state_path, it persists
    _failure_log to that file after each generic-path mutation.  A fresh
    escalator constructed with the same path reloads the log so the "N of
    3" count is correct even after a watchdog restart.

    Wall-clock teeth: the persisted timestamps must be wall-clock
    time.time() values (not monotonic uptime), so reloaded values remain
    valid across restarts.
    """

    @pytest.mark.asyncio
    async def test_burst_counter_survives_restart(self, tmp_path):
        """Three report_failure calls split across a simulated restart must
        read '3 of 3', not '1 of 3'.
        """
        import json
        import time

        state_path = tmp_path / 'curator_escalator_state.json'
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            # ── First "process": two failures ───────────────────────────────
            escalator1 = CuratorEscalator(cooldown_secs=3600.0, state_path=state_path)
            await escalator1.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='budget exceeded (1)',
                candidate_title='T',
                subtype='error_max_budget_usd',
            )
            await escalator1.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='budget exceeded (2)',
                candidate_title='T',
                subtype='error_max_budget_usd',
            )

            # State file must exist and contain timestamps ≈ wall-clock now.
            assert state_path.exists(), 'state_path must be written after report_failure'
            records = json.loads(state_path.read_text())
            assert len(records) >= 1
            # Find the record for our (project_id, subtype).
            our_record = next(
                (r for r in records
                 if r['project_id'] == 'proj-x' and r['subtype'] == 'error_max_budget_usd'),
                None,
            )
            assert our_record is not None, 'record not found in persisted state'
            assert len(our_record['timestamps']) == 2
            for ts in our_record['timestamps']:
                assert abs(ts - time.time()) < 5, (
                    f'persisted timestamp {ts} is not within 5s of time.time() — '
                    f'check that wall-clock time.time() is used, not monotonic'
                )

            # ── Simulated restart: fresh escalator reloads the same file ────
            escalator2 = CuratorEscalator(cooldown_secs=3600.0, state_path=state_path)
            await escalator2.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='budget exceeded (3)',
                candidate_title='T',
                subtype='error_max_budget_usd',
            )

            # The third call should count as 3 of 3, not 1 of 3.
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            # Find the newest file (third escalation).
            assert len(files) == 3, (
                f'Expected 3 escalation files (one per failure in-window), got {len(files)}'
            )
            newest_body = files[-1].read_text()
            assert 'failures_in_window=3 of 3' in newest_body, (
                f'Expected "3 of 3" after restart-reload; got:\n{newest_body}'
            )
        finally:
            handle.close()


# ----------------------------------------------------------------------
# Concurrent persist-state serialization (step-15 RED / step-16 GREEN)
# ----------------------------------------------------------------------


class TestPersistStateConcurrency:
    """_persist_state must serialize concurrent writes (addresses race flagged in review).

    A batch ZOT bisect spawns N concurrent size-1 curate() calls, each of
    which eventually calls report_failure → _persist_state.  Without a
    mutual-exclusion lock, all N calls await asyncio.to_thread(_write)
    concurrently, every one targeting the SAME `<state>.tmp` path.  Two
    in-flight writes can clobber each other's bytes, and the second
    os.replace can hit an already-moved tmp (FileNotFoundError), both
    swallowed as WARNING — silently degrading the durable burst counter.

    Step-16 adds an asyncio.Lock around (prune + snapshot + to_thread) so
    only one persist is ever in flight, and gives each _write call a unique
    temp filename so no two writers ever share a temp path.
    """

    @pytest.mark.asyncio
    async def test_persist_state_serializes_writes(self, tmp_path, monkeypatch):
        """Concurrent report_failure calls must serialize their writes (max 1 in-flight)
        and must not corrupt or lose any burst-log entry."""
        import asyncio
        import json

        state_path = tmp_path / 'curator_escalator_state.json'
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0, state_path=state_path)

            # ── Instrumentation ─────────────────────────────────────────────────
            in_flight = 0
            max_in_flight = 0
            real_to_thread = asyncio.to_thread

            async def _patched_to_thread(func, *args, **kwargs):
                nonlocal in_flight, max_in_flight
                in_flight += 1
                if in_flight > max_in_flight:
                    max_in_flight = in_flight
                # Widen the interleave window so concurrent coroutines can enter
                # before the first one returns — exposes the race without the lock.
                await asyncio.sleep(0.02)
                try:
                    return await real_to_thread(func, *args, **kwargs)
                finally:
                    in_flight -= 1

            monkeypatch.setattr(asyncio, 'to_thread', _patched_to_thread)

            # ── Fire 5 concurrent report_failure calls ───────────────────────
            # Same project_id, DISTINCT subtypes → independent _failure_log keys
            # so each call goes through the generic burst path (one _persist_state
            # per call) rather than being coalesced before the lock.
            subtypes = [
                'error_max_budget_usd',
                'error_max_turns',
                'error_max_tokens',
                'error_a',
                'error_b',
            ]
            await asyncio.gather(*[
                escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification=f'failure {subtype}',
                    candidate_title='T',
                    subtype=subtype,
                    zero_output_timeout=False,
                    schema_tool_denied=False,
                )
                for subtype in subtypes
            ])

            # ── Deterministic assertion: writes must be serialized ────────────
            assert max_in_flight == 1, (
                f'Expected max concurrent in-flight writes == 1 (serialized via '
                f'asyncio.Lock), got {max_in_flight}. _persist_state has no '
                f'mutual exclusion — add asyncio.Lock around (prune+snapshot+'
                f'to_thread) in _persist_state.'
            )

            # ── Outcome assertion: all 5 keys must survive in the state file ──
            assert state_path.exists(), 'state_path must be written'
            records = json.loads(state_path.read_text())
            persisted_keys = {(r['project_id'], r['subtype']) for r in records}
            for subtype in subtypes:
                assert ('proj-x', subtype) in persisted_keys, (
                    f'Missing key (proj-x, {subtype!r}) in persisted state — '
                    f'a concurrent write clobbered or lost this entry. '
                    f'Persisted keys: {persisted_keys}'
                )
        finally:
            handle.close()
