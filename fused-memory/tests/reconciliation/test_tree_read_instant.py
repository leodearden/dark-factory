"""resolve_tree_read_instant: the one rule for a caller-supplied task tree's read instant."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from fused_memory.reconciliation.tree_read_instant import resolve_tree_read_instant

_EVENT = 'reconciliation.test_tree_read_instant_missing'
_EXTRA = {'project_id': 'p', 'run_id': 'r'}


def _fallback_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == _EVENT]


def test_aware_instant_is_returned_unchanged_and_silently(caplog):
    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)

    with caplog.at_level(logging.WARNING):
        assert resolve_tree_read_instant(fetched_at, event=_EVENT, extra=_EXTRA) is fetched_at

    assert _fallback_records(caplog) == []


@pytest.mark.parametrize(
    ('fetched_at', 'reason'),
    [(None, 'not_supplied'), (datetime(2026, 1, 1), 'naive_datetime')],
    ids=['missing', 'naive'],
)
def test_unusable_instant_falls_back_to_wall_clock_with_one_warning(caplog, fetched_at, reason):
    before = datetime.now(UTC)

    with caplog.at_level(logging.WARNING):
        resolved = resolve_tree_read_instant(fetched_at, event=_EVENT, extra=_EXTRA)

    assert before <= resolved <= datetime.now(UTC)
    [record] = _fallback_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.__dict__['reason'] == reason
    assert record.__dict__['project_id'] == 'p'
    assert record.__dict__['run_id'] == 'r'
