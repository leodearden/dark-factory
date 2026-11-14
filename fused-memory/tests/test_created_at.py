"""``fused_memory.utils.created_at``: the one reading of a record's ``created_at``."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from fused_memory.utils.created_at import parse_created_at


def test_an_aware_timestamp_keeps_its_offset() -> None:
    moment = parse_created_at('2026-11-10T02:00:00+02:00')

    assert moment is not None
    assert moment == datetime(2026, 11, 10, tzinfo=UTC)
    assert moment.tzinfo == timezone(timedelta(hours=2))


def test_a_naive_timestamp_is_read_as_utc() -> None:
    assert parse_created_at('2026-11-10T00:00:00') == datetime(2026, 11, 10, tzinfo=UTC)


@pytest.mark.parametrize('value', [None, '', 'yesterday', 17, 1.5, ['2026-11-10']])
def test_an_absent_or_unparseable_value_is_none(value: object) -> None:
    assert parse_created_at(value) is None
