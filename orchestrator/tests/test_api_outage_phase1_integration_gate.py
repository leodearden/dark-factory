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
    DispatchRecord,
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
    _assert_row_4(ledger)
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


_TRANSIENT_ENVELOPES = (30.0, 60.0, 120.0, 240.0, 480.0)


def _assert_row_4(ledger: BurstLedger) -> None:
    assert len(_TRANSIENT_ENVELOPES) == KILLS_PER_TASK
    for tid in ledger.cast.burst:
        for n, record in enumerate(ledger.kills_of(tid), start=1):
            where = f'row 4: task {tid} kill {n}'
            cooldown = record.cooldown
            assert cooldown is not None, f'{where}: no cooldown armed'
            envelope = _TRANSIENT_ENVELOPES[n - 1]
            assert cooldown.envelope_secs == envelope, f'{where}: envelope {cooldown.envelope_secs}'
            assert envelope / 2 <= cooldown.armed_secs <= envelope, (
                f'{where}: armed {cooldown.armed_secs}s outside [{envelope / 2}, {envelope}]'
            )
            assert cooldown.transient is True, f'{where}: cooldown not marked transient'
            assert cooldown.transient_count == n, f'{where}: transient_count {cooldown.transient_count}'
    for n in range(1, KILLS_PER_TASK + 1):
        armed = {_armed_secs(ledger.kills_of(tid)[n - 1]) for tid in ledger.cast.burst}
        assert len(armed) > 1, f'row 4: no jitter at kill {n} — every burst task armed {armed}'

    genuine = ledger.cast.genuine
    first = ledger.dispatches_of(genuine)[0]
    where = f'row 4: genuine control {genuine}'
    assert first.report.outcome == WorkflowOutcome.REQUEUED, f'{where}: outcome {first.report.outcome!r}'
    assert first.report.api_error_status is None, f'{where}: api_error_status {first.report.api_error_status}'
    assert not is_transient_api_requeue(first.report.block_reason, api_error_status=None), (
        f'{where}: reason reads as a transient requeue: {first.report.block_reason!r}'
    )
    assert first.cooldown is not None, f'{where}: no cooldown armed'
    assert first.cooldown.armed_secs == ledger.config.requeue_cooldown_secs, (
        f'{where}: armed {first.cooldown.armed_secs}s, expected the flat {ledger.config.requeue_cooldown_secs}s'
    )
    assert first.cooldown.transient is False, f'{where}: cooldown marked transient'
    assert first.transient_requeue_count == 0, f'{where}: transient count {first.transient_requeue_count}'
    assert first.genuine_requeue_count == 1, f'{where}: genuine count {first.genuine_requeue_count}'


def _armed_secs(record: DispatchRecord) -> float:
    assert record.cooldown is not None, f'row 4: task {record.task_id} has no cooldown armed'
    return record.cooldown.armed_secs


def _assert_row_5(ledger: BurstLedger) -> None:
    assert ledger.park_stop_trip_wired, (
        'row 5: the scheduler has no on_park_stop_trip callback, so an empty deque proves nothing'
    )
    phase_a = ledger.phase_a()
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
