"""Tests for the analyze_modules per-module conflict helper."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.analyze_modules import (
    ModuleStats,
    _first_component,
    _iter_events,
    _parse_since,
    aggregate,
    render_json,
    render_table,
    suggest_max_per_module,
)
from orchestrator.event_store import EventStore, EventType


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='test')


def test_first_component_strips_and_splits():
    assert _first_component('autopilot/analyze/asr') == 'autopilot'
    assert _first_component('/crates/foo/src') == 'crates'
    assert _first_component('bare') == 'bare'
    assert _first_component('') == ''


def test_parse_since_duration_shorthand():
    # '7d' shouldn't raise and must yield a past timestamp.
    from datetime import UTC, datetime
    cut = _parse_since('7d')
    assert cut < datetime.now(UTC)


def test_parse_since_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_since('nonsense')


def test_aggregate_counts_dispatches_and_skips(tmp_path: Path, event_store: EventStore):
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.lock_released,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    from datetime import UTC, datetime, timedelta
    stats = aggregate(event_store.db_path, datetime.now(UTC) - timedelta(days=1))
    assert 'crates' in stats
    assert stats['crates'].dispatches == 1
    assert stats['crates'].skipped_waiting == 2
    assert stats['crates'].conflict_rate() == pytest.approx(2.0)


def test_aggregate_ignores_events_before_since(tmp_path: Path, event_store: EventStore):
    from datetime import UTC, datetime, timedelta
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    # Pretend these happened in the past: cut is "now + 1 hour" so nothing qualifies.
    stats = aggregate(event_store.db_path, datetime.now(UTC) + timedelta(hours=1))
    assert stats == {}


def test_suggest_max_per_module_tiers():
    assert suggest_max_per_module(ModuleStats(dispatches=1, skipped_waiting=2)) == 1
    assert suggest_max_per_module(ModuleStats(dispatches=2, skipped_waiting=1)) == 2
    assert suggest_max_per_module(ModuleStats(dispatches=10, skipped_waiting=1)) == 3
    assert suggest_max_per_module(ModuleStats(dispatches=100, skipped_waiting=0)) == 4


def test_render_table_orders_by_conflict_desc():
    stats = {
        'low': ModuleStats(dispatches=10, skipped_waiting=0),
        'hot': ModuleStats(dispatches=5, skipped_waiting=20),
    }
    table = render_table(stats)
    lines = table.splitlines()
    # Header + two rows, hot first.
    assert lines[1].startswith('hot')
    assert lines[2].startswith('low')


def test_render_json_is_machine_readable():
    # Asymmetric values so a dispatches<->skipped swap would be caught:
    # swapped input would be (7, 4) → ratio 4/7 ≈ 0.57 → still suggest 2,
    # so the conflict_rate assertion does the distinguishing.
    stats = {
        'crates': ModuleStats(dispatches=4, skipped_waiting=7),
    }
    payload = json.loads(render_json(stats))
    assert payload['crates']['dispatches'] == 4
    assert payload['crates']['skipped_waiting'] == 7
    # conflict = 7/4 = 1.75 (>= 0.5 but < 2.0) → suggest 2.
    assert payload['crates']['conflict_rate'] == 1.75
    assert payload['crates']['suggested_max_per_module'] == 2


def test_iter_events_opens_runs_db_read_only(event_store: EventStore) -> None:
    """_iter_events must open runs.db via a read-only file:/// URI (uri=True, ?mode=ro).

    Modelled on dashboard/tests/test_db.py::test_get_builds_sqlite_uri_via_path_as_uri
    and fused-memory/tests/test_scheduler_state_tools.py.

    The spy delegates to the real sqlite3.connect so the SELECT still executes
    and the generator's finally-block closes the connection normally.
    """
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )

    real_connect = sqlite3.connect
    captured_uri: str | None = None
    captured_uri_kwarg: bool | None = None

    def spy(*args, **kwargs):
        nonlocal captured_uri, captured_uri_kwarg
        captured_uri = args[0] if args else None
        captured_uri_kwarg = kwargs.get('uri')
        return real_connect(*args, **kwargs)

    since = datetime.now(UTC) - timedelta(days=1)
    with patch('orchestrator.analyze_modules.sqlite3.connect', spy):
        rows = list(_iter_events(event_store.db_path, since))

    # URI contract — single exact-match subsumes all substring checks.
    # Enforces: file:/// prefix, .resolve() present, ?mode=ro suffix, uri=True kwarg.
    expected_uri = event_store.db_path.resolve().as_uri() + '?mode=ro'
    assert captured_uri == expected_uri, (
        f'Expected read-only URI {expected_uri!r}, got {captured_uri!r}'
    )
    assert captured_uri_kwarg is True, (
        f'Expected uri=True kwarg, got uri={captured_uri_kwarg!r}'
    )

    # Reads still work — seeded event is returned.
    assert len(rows) == 1
    _ts, event_type, _task_id, _data = rows[0]
    assert event_type == 'lock_acquired'
