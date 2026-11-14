"""Phase-1 integration gate for ``plans/server-side-api-error-handling-prd.md``.

One simulated incident hour, driven once through the real harness, workflow,
scheduler and escalation queue by ``_api_outage_harness.run_incident_burst``,
must satisfy boundary rows 1-5 together. Each assertion helper owns one row and
prefixes its messages ``row N:`` so a failure names the row it broke.
"""

from __future__ import annotations

import pytest
from _api_outage_harness import (
    BURST_SIZE,
    KILLS_PER_TASK,
    PROVIDER_529_ERROR,
    BurstLedger,
    incident_529_result,
    run_incident_burst,
)
from shared.cli_invoke import AgentFailureKind, classify_agent_failure, is_zero_output_timeout

from orchestrator.scheduler import is_transient_api_requeue
from orchestrator.workflow import WorkflowOutcome

_API_ERROR_MARKER = 'agent API error: HTTP 529'


@pytest.mark.asyncio
async def test_incident_burst_rows_1_to_5_hold_in_one_harness_run(tmp_path, monkeypatch):
    ledger = await run_incident_burst(tmp_path, monkeypatch)

    _assert_burst_premises(ledger)
    _assert_row_1(ledger)
    _assert_row_5(ledger)


def _assert_burst_premises(ledger: BurstLedger) -> None:
    config = ledger.config
    assert config.max_consecutive_zero_output_timeouts == 2, 'premise: zero-output threshold default'
    assert config.transient_requeue_cap == 10, 'premise: transient requeue cap default'
    assert config.transient_requeue_backoff_base_secs == 30.0, 'premise: backoff base default'
    assert config.transient_requeue_backoff_cap_secs == 900.0, 'premise: backoff cap default'
    assert config.requeue_cooldown_secs == 30.0, 'premise: flat requeue cooldown default'
    assert config.park_stop_enabled is True, 'premise: park-stop enabled by default'
    assert config.park_stop_parked_window_hours == 1.0, 'premise: park-stop window default'
    assert config.park_stop_parked_threshold <= BURST_SIZE, (
        f'premise: a burst of {BURST_SIZE} wrongful blocks must be able to trip park-stop '
        f'(threshold {config.park_stop_parked_threshold})'
    )

    shape = incident_529_result()
    assert is_zero_output_timeout(shape), 'premise: the injected 529 shape is a zero-output timeout'
    classified = classify_agent_failure(shape)
    assert classified.kind == AgentFailureKind.API_ERROR, (
        f'premise: the injected 529 shape classifies API_ERROR, got {classified.kind!r}'
    )
    assert _API_ERROR_MARKER in classified.summary, f'premise: marker missing from {classified.summary!r}'

    kill_times = [r.at for tid in ledger.cast.burst for r in ledger.kills_of(tid)]
    assert len(kill_times) == BURST_SIZE * KILLS_PER_TASK, (
        f'premise: expected {BURST_SIZE * KILLS_PER_TASK} burst kills, got {len(kill_times)}'
    )
    window_secs = config.park_stop_parked_window_hours * 3600
    assert max(kill_times) - min(kill_times) < window_secs, (
        f'premise: the burst must sit inside one park-stop window ({window_secs}s); '
        f'it spans {max(kill_times) - min(kill_times):.0f}s'
    )


def _assert_row_1(ledger: BurstLedger) -> None:
    for tid in ledger.cast.burst:
        kills = ledger.kills_of(tid)
        assert len(kills) == KILLS_PER_TASK, f'row 1: task {tid} had {len(kills)} kills'
        for n, record in enumerate(kills, start=1):
            report = record.report
            where = f'row 1: task {tid} kill {n}'
            assert report.outcome == WorkflowOutcome.REQUEUED, f'{where}: outcome {report.outcome!r}'
            assert report.api_error_status == 529, f'{where}: api_error_status {report.api_error_status!r}'
            assert _API_ERROR_MARKER in report.block_reason, f'{where}: reason {report.block_reason!r}'
            assert is_transient_api_requeue(
                report.block_reason, api_error_status=report.api_error_status,
            ), f'{where}: the scheduler does not read this report as a transient requeue'
            assert record.cli_calls == 1, (
                f'{where}: {record.cli_calls} CLI calls — a 5xx must leave on the first, '
                'not feed the zero-output counter'
            )
            assert record.transient_requeue_count == n, (
                f'{where}: transient_requeue_count {record.transient_requeue_count}'
            )
            assert record.genuine_requeue_count == 0, (
                f'{where}: genuine requeue count {record.genuine_requeue_count}'
            )
            assert len(record.evidence) == 1, f'{where}: evidence files {record.evidence!r}'
            evidence = record.evidence[0]
            assert evidence.api_error_status == 529, f'{where}: evidence {evidence!r}'
            assert 'overloaded_error' in evidence.output_tail, f'{where}: evidence {evidence!r}'
            assert evidence.output_tail.endswith(PROVIDER_529_ERROR), f'{where}: evidence {evidence!r}'
        writes = ledger.status_writes[tid]
        assert 'blocked' not in writes, f'row 1: task {tid} was written blocked: {writes}'
        assert ledger.escalations_by_task[tid] == (), (
            f'row 1: task {tid} filed escalations: {ledger.escalations_by_task[tid]}'
        )


def _assert_row_5(ledger: BurstLedger) -> None:
    assert ledger.park_stop_trip_wired, (
        'row 5: the scheduler has no on_park_stop_trip callback, so an empty deque proves nothing'
    )
    phase_a = [r for tid in ledger.cast.burst for r in ledger.kills_of(tid)]
    assert phase_a, 'row 5: no phase-A dispatches recorded'
    for record in phase_a:
        where = f'row 5: task {record.task_id} at t={record.at:.0f}s'
        assert record.parked_live_count == 0, f'{where}: parked_live_count {record.parked_live_count}'
        assert record.parked_window_churn_count == 0, (
            f'{where}: parked_window_churn_count {record.parked_window_churn_count}'
        )
        assert record.paused is False, f'{where}: scheduler paused'
    end = ledger.after_phase_a
    assert end.pending_escalations == (), f'row 5: escalations pending after phase A: {end.pending_escalations}'
    assert end.paused is False, 'row 5: scheduler paused after phase A'
    assert end.parked_live_count == 0, f'row 5: parked_live_count {end.parked_live_count} after phase A'
    assert end.parked_window_churn_count == 0, (
        f'row 5: parked_window_churn_count {end.parked_window_churn_count} after phase A'
    )
