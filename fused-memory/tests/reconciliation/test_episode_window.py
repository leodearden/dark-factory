"""Where a completed full cycle starts the NEXT cycle's episode window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.reconciliation.episode_window import (
    UNFINISHED_WRITE_START_STAT,
    EpisodeWindowAnchor,
    next_episode_window_anchor,
    recorded_unfinished_write_start,
    unfinished_write_start_stat,
)

RUN_START = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
STAGE1 = StageId.memory_consolidator.value


def _run(stage1_stats: dict | None) -> ReconciliationRun:
    """A run whose Stage 1 report carries *stage1_stats*; None = no Stage 1 report."""
    reports: dict[str, StageReport | dict] = {}
    if stage1_stats is not None:
        reports[STAGE1] = StageReport(
            stage=StageId.memory_consolidator,
            started_at=RUN_START,
            completed_at=RUN_START + timedelta(minutes=1),
            stats=stage1_stats,
        )
    return ReconciliationRun(
        id='run-1',
        project_id='p',
        run_type=RunType.full,
        trigger_reason='t',
        started_at=RUN_START,
        status=RunStatus.running,
        stage_reports=reports,
    )


def _run_recording(start: datetime | None) -> ReconciliationRun:
    return _run({UNFINISHED_WRITE_START_STAT: unfinished_write_start_stat(start)})


def test_no_stage1_report_anchors_at_run_start():
    run = _run(None)

    assert next_episode_window_anchor(run).at == RUN_START
    assert recorded_unfinished_write_start(run) is None


def test_stage1_report_without_the_stat_anchors_at_run_start():
    assert next_episode_window_anchor(_run({})).at == RUN_START


def test_nothing_executing_anchors_at_run_start():
    run = _run_recording(None)

    assert next_episode_window_anchor(run).at == RUN_START
    assert recorded_unfinished_write_start(run) is None


def test_write_unfinished_since_before_run_start_lowers_the_anchor():
    start = RUN_START - timedelta(minutes=3)
    run = _run_recording(start)

    assert recorded_unfinished_write_start(run) == start
    assert next_episode_window_anchor(run) == EpisodeWindowAnchor(
        at=start, unfinished_write_start=start,
    )


def test_write_begun_after_run_start_leaves_the_anchor_at_run_start():
    begun = RUN_START + timedelta(seconds=30)
    run = _run_recording(begun)

    assert next_episode_window_anchor(run) == EpisodeWindowAnchor(
        at=RUN_START, unfinished_write_start=begun,
    )


def test_anchor_survives_the_journal_stage_report_round_trip():
    start = RUN_START - timedelta(minutes=3)
    run = _run_recording(start)
    report = run.stage_reports[STAGE1]
    assert isinstance(report, StageReport)
    reloaded = run.model_copy(
        update={'stage_reports': {STAGE1: StageReport(**report.model_dump(mode='json'))}},
    )

    assert next_episode_window_anchor(reloaded).at == start


def test_start_with_a_non_utc_offset_compares_by_instant():
    start = RUN_START - timedelta(minutes=3)
    run = _run_recording(start.astimezone(timezone(timedelta(hours=2))))

    assert recorded_unfinished_write_start(run) == start
    assert next_episode_window_anchor(run).at == start


@pytest.mark.parametrize(
    'encoded',
    [RUN_START - timedelta(minutes=3), 1760788620, ['2026-10-18T11:57:00+00:00']],
    ids=['raw-datetime', 'epoch-int', 'list'],
)
def test_wrongly_typed_stat_raises_naming_the_stat_its_type_and_value(encoded):
    run = _run({UNFINISHED_WRITE_START_STAT: encoded})

    with pytest.raises(ValueError, match=UNFINISHED_WRITE_START_STAT) as raised:
        recorded_unfinished_write_start(run)
    assert type(encoded).__name__ in str(raised.value)
    assert repr(encoded) in str(raised.value)
    with pytest.raises(ValueError, match=UNFINISHED_WRITE_START_STAT):
        next_episode_window_anchor(run)


@pytest.mark.parametrize(
    'encoded', ['not an instant', '2026-10-18T11:57:00'], ids=['garbage', 'no-offset'],
)
def test_string_stat_that_is_not_an_instant_with_offset_raises_naming_the_stat(encoded):
    run = _run({UNFINISHED_WRITE_START_STAT: encoded})

    with pytest.raises(ValueError, match=UNFINISHED_WRITE_START_STAT) as raised:
        recorded_unfinished_write_start(run)
    assert repr(encoded) in str(raised.value)
