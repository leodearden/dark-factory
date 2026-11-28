"""The contract of ``merge_lane/gate_reaper_watch.py::GateReaperWatch`` (task 8520).

It is driven only through ``observe``, over a real ``EscalationQueue`` on a
scratch directory, so dedup and resolve are real.  Launches are an hour
apart, as the reprobe sweep's gate reap is.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.merge_lane.gate_reaper_watch import GateReaperWatch
from orchestrator.remote_gate import ReapStamp

_HOST = 'laptop'
_OTHER = 'desktop'
_HOUR = 3600.0
_NEVER_REAPED = ReapStamp(path='/gates/.last-reap', gate_root=True, age_secs=None)
_JUST_REAPED = ReapStamp(path='/gates/.last-reap', gate_root=True, age_secs=0)


def _stamp(age: int | None) -> ReapStamp:
    return ReapStamp(path='/gates/.last-reap', gate_root=True, age_secs=age)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


def _watch(queue: EscalationQueue | None) -> GateReaperWatch:
    return GateReaperWatch(escalation_queue=lambda: queue)


def _feed(
    watch: GateReaperWatch, readings: Iterable[ReapStamp | None], *,
    host: str = _HOST, start_hour: int = 0,
) -> None:
    """Observe *readings* for *host*, one launch an hour apart from *start_hour*."""
    for hour, reading in enumerate(readings, start=start_hour):
        watch.observe(host, reading, launched_at=hour * _HOUR, ssh_target='leo-laptop')


def _pending(queue: EscalationQueue, host: str = _HOST) -> list[Escalation]:
    return queue.get_by_task(f'__gate_reaper_stale__{host}', status='pending')


def test_a_hosts_first_reading_never_files(queue: EscalationQueue):
    _feed(_watch(queue), [_NEVER_REAPED])

    assert _pending(queue) == []


def test_a_run_completed_since_the_previous_launch_files_nothing(queue: EscalationQueue):
    _feed(_watch(queue), [_stamp(0), _stamp(3500)])

    assert _pending(queue) == []


@pytest.mark.parametrize('age', [None, 90000])
def test_two_consecutive_misses_file_one_l1(queue: EscalationQueue, age: int | None):
    watch = _watch(queue)

    _feed(watch, [_stamp(age)] * 3)

    [record] = _pending(queue)
    assert record.category == 'gate_reaper_stale'
    assert record.task_id == '__gate_reaper_stale__laptop'
    assert record.level == 1
    assert record.severity == 'blocking'
    assert 'leo-laptop' in record.detail
    assert '/gates/.last-reap' in record.detail

    _feed(watch, [_stamp(age)], start_hour=3)

    assert len(_pending(queue)) == 1


def test_a_completed_run_resets_the_streak(queue: EscalationQueue):
    _feed(_watch(queue), [_NEVER_REAPED, _NEVER_REAPED, _JUST_REAPED, _stamp(90000)])

    assert _pending(queue) == []


def test_a_completed_run_resolves_the_alarm(queue: EscalationQueue):
    watch = _watch(queue)
    _feed(watch, [_NEVER_REAPED] * 3)
    assert len(_pending(queue)) == 1

    _feed(watch, [_JUST_REAPED], start_hour=3)

    assert _pending(queue) == []


def test_an_alarm_left_by_an_earlier_process_resolves_at_the_first_completed_judgement(
    queue: EscalationQueue,
):
    _feed(_watch(queue), [_NEVER_REAPED] * 3)
    fresh = _watch(queue)

    _feed(fresh, [_JUST_REAPED], start_hour=4)
    assert len(_pending(queue)) == 1

    _feed(fresh, [_JUST_REAPED], start_hour=5)
    assert _pending(queue) == []


def test_a_failed_launch_is_judged_against_the_last_real_launch(queue: EscalationQueue):
    _feed(_watch(queue), [_stamp(0), None, _stamp(7000)])

    assert _pending(queue) == []


def test_a_failed_launch_neither_counts_as_a_miss_nor_resets_the_streak(queue: EscalationQueue):
    uncounted = _watch(queue)
    _feed(uncounted, [_NEVER_REAPED, _NEVER_REAPED, None, _stamp(7000)])
    assert _pending(queue) == []

    unreset = _watch(queue)
    _feed(unreset, [_NEVER_REAPED, _NEVER_REAPED, None, _NEVER_REAPED], host=_OTHER)
    assert len(_pending(queue, _OTHER)) == 1


def test_an_unmarked_root_never_files(queue: EscalationQueue):
    unmarked = ReapStamp(path='/gates/.last-reap', gate_root=False, age_secs=None)

    _feed(_watch(queue), [unmarked] * 4)

    assert _pending(queue) == []


def test_misses_count_per_host(queue: EscalationQueue):
    watch = _watch(queue)
    for hour in range(2):
        for host in (_HOST, _OTHER):
            watch.observe(host, _NEVER_REAPED, launched_at=hour * _HOUR, ssh_target=None)
    assert _pending(queue, _HOST) == [] and _pending(queue, _OTHER) == []

    watch.observe(_HOST, _NEVER_REAPED, launched_at=2 * _HOUR, ssh_target=None)

    assert len(_pending(queue, _HOST)) == 1
    assert _pending(queue, _OTHER) == []


def test_no_escalation_queue_never_raises():
    _feed(_watch(None), [_NEVER_REAPED] * 4 + [_JUST_REAPED])
