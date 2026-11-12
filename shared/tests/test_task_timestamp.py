"""The task store's one timestamp form: ``shared.task_timestamp.format_task_timestamp``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from shared.task_timestamp import format_task_timestamp


def test_utc_instant_formats_with_microseconds_truncated_to_milliseconds():
    instant = datetime(2026, 4, 9, 16, 53, 21, 119808, tzinfo=UTC)

    assert format_task_timestamp(instant) == '2026-04-09T16:53:21.119Z'


def test_whole_second_instant_carries_three_zero_milliseconds():
    instant = datetime(2026, 4, 9, 16, 53, 21, tzinfo=UTC)

    assert format_task_timestamp(instant) == '2026-04-09T16:53:21.000Z'


def test_non_utc_instant_is_converted_to_utc_first():
    plus_two = timezone(timedelta(hours=2))
    instant = datetime(2026, 4, 9, 18, 0, 0, 999999, tzinfo=plus_two)

    assert format_task_timestamp(instant) == '2026-04-09T16:00:00.999Z'


def test_naive_datetime_is_refused_naming_the_invariant():
    naive = datetime(2026, 4, 9, 16, 53, 21)

    with pytest.raises(ValueError) as refusal:
        format_task_timestamp(naive)

    message = str(refusal.value)
    assert 'task-store timestamps are UTC instants; a naive datetime has no instant' in message
    assert repr(naive) in message
