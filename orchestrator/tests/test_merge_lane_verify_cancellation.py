"""The ``merge_verify_cancelled`` row: how far a cancelled verify got (task 8523).

``merge_verify_cancelled_data`` builds the row's data from a ``VerifyCancel``
(why the lane cancelled, when the verify started, on which host) and a
``VerifyDurationWindow`` (the host's rolling median completed verify), so a
reader can tell a verify cancelled seconds in from one cancelled at 90%.
"""
from __future__ import annotations

import pytest

from orchestrator.merge_lane.verify_cancellation import (
    RecordedDispatches,
    VerifyCancel,
    VerifyCancelCause,
    VerifyDurationWindow,
    merge_verify_cancelled_data,
)


def _window(host: str, *durations_ms: float) -> VerifyDurationWindow:
    window = VerifyDurationWindow(maxlen=3)
    for duration in durations_ms:
        window.record(host, duration)
    return window


class TestVerifyDurationWindow:
    def test_no_fraction_before_the_host_completed_a_verify(self) -> None:
        assert VerifyDurationWindow(maxlen=3).fraction('laptop', 500) is None

    def test_fraction_is_elapsed_over_the_rolling_median(self) -> None:
        window = _window('laptop', 1000, 3000, 2000)
        assert window.fraction('laptop', 1000) == 0.5

    def test_a_record_past_maxlen_evicts_the_oldest(self) -> None:
        window = _window('laptop', 1000, 3000, 2000)
        window.record('laptop', 4000)
        assert window.fraction('laptop', 1500) == 0.5, 'median of 3000/2000/4000'

    def test_hosts_are_independent(self) -> None:
        window = _window('laptop', 1000)
        assert window.fraction('local', 1000) is None

    def test_no_elapsed_gives_no_fraction(self) -> None:
        assert _window('laptop', 1000).fraction('laptop', None) is None

    @pytest.mark.parametrize('duration_ms', [0, -5.0])
    def test_non_positive_durations_are_ignored(self, duration_ms: float) -> None:
        window = VerifyDurationWindow(maxlen=3)
        window.record('laptop', duration_ms)
        assert window.fraction('laptop', 1000) is None


class TestMergeVerifyCancelledData:
    def test_a_full_row(self) -> None:
        data = merge_verify_cancelled_data(
            VerifyCancel(VerifyCancelCause.CASCADE, started_at=100.0, host='laptop'),
            now=101.5,
            window=_window('laptop', 3000),
        )
        assert data == {
            'elapsed_ms': 1500, 'fraction': 0.5, 'cause': 'cascade', 'host': 'laptop',
        }
        assert type(data['cause']) is str

    def test_no_start_gives_no_elapsed_and_no_fraction(self) -> None:
        data = merge_verify_cancelled_data(
            VerifyCancel(VerifyCancelCause.SHUTDOWN, started_at=None, host='laptop'),
            now=101.5,
            window=_window('laptop', 3000),
        )
        assert data['elapsed_ms'] is None
        assert data['fraction'] is None

    def test_no_host_gives_no_fraction(self) -> None:
        data = merge_verify_cancelled_data(
            VerifyCancel(VerifyCancelCause.ABANDONED, started_at=100.0, host=None),
            now=101.5,
            window=_window('laptop', 3000),
        )
        assert data['elapsed_ms'] == 1500
        assert data['fraction'] is None
        assert data['host'] is None

    def test_a_backwards_clock_clamps_elapsed_to_zero(self) -> None:
        data = merge_verify_cancelled_data(
            VerifyCancel(VerifyCancelCause.NO_PROGRESS, started_at=100.0, host='laptop'),
            now=99.0,
            window=VerifyDurationWindow(maxlen=3),
        )
        assert data['elapsed_ms'] == 0


class TestRecordedDispatches:
    """One row per verify dispatch, whichever canceller reaches it first."""

    def test_a_dispatch_is_claimed_once(self) -> None:
        recorded = RecordedDispatches(cap=4)
        halt = VerifyCancel(VerifyCancelCause.OPERATOR_HALT, started_at=100.0, host='local')
        drain = VerifyCancel(VerifyCancelCause.SHUTDOWN, started_at=100.0, host='local')
        assert recorded.claim('t1', halt)
        assert not recorded.claim('t1', drain)

    def test_another_dispatch_or_task_is_claimed_afresh(self) -> None:
        recorded = RecordedDispatches(cap=4)
        first = VerifyCancel(VerifyCancelCause.CASCADE, started_at=100.0, host='laptop')
        redispatch = VerifyCancel(VerifyCancelCause.CASCADE, started_at=105.0, host='laptop')
        assert recorded.claim('t1', first)
        assert recorded.claim('t1', redispatch)
        assert recorded.claim('t2', first)

    def test_a_cancel_with_no_start_is_always_claimed(self) -> None:
        recorded = RecordedDispatches(cap=4)
        unknown = VerifyCancel(VerifyCancelCause.SHUTDOWN, started_at=None, host=None)
        assert recorded.claim('t1', unknown)
        assert recorded.claim('t1', unknown)

    def test_past_the_cap_the_oldest_claim_is_forgotten(self) -> None:
        recorded = RecordedDispatches(cap=1)
        a = VerifyCancel(VerifyCancelCause.CASCADE, started_at=1.0, host='local')
        b = VerifyCancel(VerifyCancelCause.CASCADE, started_at=2.0, host='local')
        assert recorded.claim('t1', a)
        assert recorded.claim('t1', b)
        assert recorded.claim('t1', a)

    def test_a_cap_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match='cap'):
            RecordedDispatches(cap=0)


def test_cause_vocabulary_is_closed() -> None:
    assert {cause.value for cause in VerifyCancelCause} == {
        'cascade', 'abandoned', 'operator_halt', 'no_progress', 'tip_adopted',
        'shutdown', 'finalize_error', 'gate_reanchor',
    }
