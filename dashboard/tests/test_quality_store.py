"""quality.db: where it lives, its tables, and the LandingRow / sample codec over them."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import GitCrossCheck, Landing, LandingSource, LinesBase
from shared.quality_series.rows import (
    CharterClass,
    CharterClassSource,
    Cost,
    LandingRow,
    Outcomes,
    ReviewRoundsSource,
    SourceEpochs,
)

from dashboard.config import DashboardConfig
from dashboard.data.quality_store import (
    QualitySample,
    SampleState,
    insert_sample,
    read_landing_rows,
    read_samples,
    stored_cursor,
    upsert_landing_rows,
)
from dashboard.loops import _QualityStore

T0 = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
SAMPLED_AT = T0 + timedelta(days=1)

NO_OUTCOMES = Outcomes(
    rejected=None, rejection_events=None, reviewed=None, review_blocking=None,
    flake_confirmed=None, flake_real_red=None, flake_rows=None,
    reverify_count=None, verify_failed_rows=None,
)
NO_COST = Cost(
    all_role_usd=None, by_role=(), implementer_invocations=None, execute_iterations=None,
    turns=None, turns_capped=False, code_lines=None, src_lines=None, test_lines=None,
    review_rounds=None, review_rounds_source=None, charter_class=None,
    charter_class_source=None,
)
NO_EPOCHS = SourceEpochs(
    rejections=None, merge_verify=None, review_completed=None, flake_occurrence=None,
    invocation_turns=None, invocations=None,
)


def _landing(
    task_id: str, merge_sha: str, ordinal: int, landed_at: datetime, *,
    project_id: str = 'p', late_claim: bool = False, git_first_parent: bool = True,
) -> Landing:
    return Landing(
        project_id=project_id,
        merge_sha=merge_sha,
        task_id=task_id,
        landed_at=landed_at,
        ordinal=ordinal,
        source=LandingSource.MERGE_FINALIZED,
        unit_id=f'{merge_sha}:{task_id}' if late_claim else merge_sha,
        unit_size=1,
        late_claim=late_claim,
        git_first_parent=git_first_parent,
        prev_landed_at=None,
        lines_base=LinesBase.UNMEASURABLE,
        base_sha=None,
    )


def _bare_row(landing: Landing) -> LandingRow:
    return LandingRow(
        landing=landing, outcomes=NO_OUTCOMES, cost=NO_COST, modules=(), rule_vintage=NO_EPOCHS,
    )


def _train_member(task_id: str, *, execute_iterations: int | None = 3) -> LandingRow:
    plus_one = timezone(timedelta(hours=1))
    landing = Landing(
        project_id='p',
        merge_sha='train-sha',
        task_id=task_id,
        landed_at=datetime(2026, 11, 2, 11, 30, tzinfo=plus_one),
        ordinal=7 if task_id == '102' else 8,
        source=LandingSource.TRAIN_MERGED,
        unit_id='train-sha',
        unit_size=2,
        late_claim=False,
        git_first_parent=True,
        prev_landed_at=datetime(2026, 10, 30, 8, 15, 30, 250000, tzinfo=UTC),
        lines_base=LinesBase.FIRST_PARENT,
        base_sha='base-sha',
    )
    outcomes = Outcomes(
        rejected=True, rejection_events=2, reviewed=True, review_blocking=False,
        flake_confirmed=True, flake_real_red=False, flake_rows=3,
        reverify_count=1, verify_failed_rows=1,
    )
    cost = Cost(
        all_role_usd=1.75, by_role=(('implementer', 1.25), ('reviewer', 0.5)),
        implementer_invocations=2, execute_iterations=execute_iterations, turns=140,
        turns_capped=True, code_lines=52, src_lines=40, test_lines=12, review_rounds=2,
        review_rounds_source=ReviewRoundsSource.REVIEW_COMPLETED,
        charter_class=CharterClass.STALE_CHARTER,
        charter_class_source=CharterClassSource.REGEX_BACKFILL,
    )
    epochs = SourceEpochs(
        rejections=datetime(2026, 1, 5, 10, 0, tzinfo=UTC), merge_verify=None,
        review_completed=datetime(2026, 3, 1, 0, 0, 1, tzinfo=plus_one), flake_occurrence=None,
        invocation_turns=datetime(2026, 6, 7, 8, 9, 10, tzinfo=UTC), invocations=None,
    )
    return LandingRow(
        landing=landing, outcomes=outcomes, cost=cost,
        modules=(ModuleLines('dashboard/data', 40),), rule_vintage=epochs,
    )


@pytest.fixture
async def conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    store = _QualityStore(tmp_path / 'quality.db')
    await store.open()
    try:
        yield store.connection
    finally:
        await store.close()


async def _store_rows(conn: aiosqlite.Connection, rows: list[LandingRow]) -> None:
    await upsert_landing_rows(conn, rows, sampled_at=SAMPLED_AT)
    await conn.commit()


def test_quality_db_sits_beside_the_other_sampler_stores(tmp_path: Path) -> None:
    config = DashboardConfig(project_root=tmp_path)

    assert config.quality_db == config.project_root / 'data' / 'burndown' / 'quality.db'
    assert config.quality_db.parent == config.burndown_db.parent == config.metrics_db.parent


async def test_a_fresh_store_has_the_four_quality_tables_and_no_rows(
    conn: aiosqlite.Connection,
) -> None:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ) as cursor:
        tables = {name for (name,) in await cursor.fetchall()}

    assert tables == {'landings', 'late_events', 'markers_derived', 'sample_state'}
    assert await read_landing_rows(conn, 'p') == []
    assert await read_samples(conn, 'p') == []


async def test_a_row_with_every_optional_field_none_round_trips(
    conn: aiosqlite.Connection,
) -> None:
    row = _bare_row(_landing('101', 'plain-sha', 0, T0))

    await _store_rows(conn, [row])

    assert await read_landing_rows(conn, 'p') == [row]


async def test_a_populated_train_member_round_trips_by_instant(
    conn: aiosqlite.Connection,
) -> None:
    row = _train_member('102')

    await _store_rows(conn, [row])

    assert await read_landing_rows(conn, 'p') == [row]


async def test_train_members_sharing_a_merge_sha_are_distinct_rows(
    conn: aiosqlite.Connection,
) -> None:
    members = [_train_member('102'), _train_member('103')]

    await _store_rows(conn, members)

    stored = await read_landing_rows(conn, 'p')
    assert stored == members
    assert {row.landing.unit_id for row in stored} == {'train-sha'}


async def test_rewriting_a_key_updates_its_one_row_in_place(
    conn: aiosqlite.Connection,
) -> None:
    first = _train_member('102', execute_iterations=None)
    second = _train_member('102', execute_iterations=4)

    await _store_rows(conn, [first])
    await _store_rows(conn, [second])

    assert await read_landing_rows(conn, 'p') == [second]


async def test_stored_cursor_names_the_newest_first_claim_on_main_by_the_cutoff(
    conn: aiosqlite.Connection,
) -> None:
    hour = timedelta(hours=1)
    await _store_rows(conn, [
        _bare_row(_landing('101', 'first-sha', 0, T0)),
        _bare_row(_landing('102', 'first-sha', 1, T0 + hour, late_claim=True)),
        _bare_row(_landing('103', 'side-sha', 2, T0 + 2 * hour, git_first_parent=False)),
        _bare_row(_landing('104', 'newest-sha', 3, T0 + 3 * hour)),
        _bare_row(_landing('901', 'other-sha', 9, T0 + 9 * hour, project_id='q')),
    ])

    assert await stored_cursor(conn, 'p') == 'newest-sha'
    assert await stored_cursor(conn, 'p', landed_by=T0 + 2.5 * hour) == 'first-sha'
    assert await stored_cursor(conn, 'p', landed_by=T0 + 3 * hour) == 'newest-sha'
    assert await stored_cursor(conn, 'p', landed_by=T0 - hour) is None
    assert await stored_cursor(conn, 'empty') is None


async def test_value_and_gap_samples_round_trip_in_time_order(
    conn: aiosqlite.Connection,
) -> None:
    gap = QualitySample.gap('p', T0 + timedelta(minutes=10), 'OperationalError: database is locked')
    value = QualitySample.value(
        'p', T0, cursor_sha='train-sha', landings_sampled=3, landings_deferred=0,
        git_check=GitCrossCheck(off_first_parent=0, merges_without_landing=1, late_claims=0),
    )

    await insert_sample(conn, gap)
    await insert_sample(conn, value)
    await insert_sample(conn, QualitySample.gap('q', T0, 'not this project'))
    await conn.commit()

    assert await read_samples(conn, 'p') == [value, gap]
    stored_gap = (await read_samples(conn, 'p'))[1]
    assert stored_gap.state is SampleState.GAP
    assert (
        stored_gap.cursor_sha, stored_gap.landings_sampled, stored_gap.landings_deferred,
        stored_gap.git_check,
    ) == (None, None, None, None)


def test_a_gap_needs_a_reason() -> None:
    with pytest.raises(ValueError, match=r"gap.*reason=''"):
        QualitySample.gap('p', T0, '')


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('cursor_sha', 'sha'),
        ('landings_sampled', 0),
        ('landings_deferred', 0),
        ('git_check', GitCrossCheck(0, 0, 0)),
    ],
)
def test_a_gap_carrying_a_measurement_is_refused(field: str, value: object) -> None:
    gap = QualitySample.gap('p', T0, 'boom')

    with pytest.raises(ValueError, match='gap.*' + re.escape(f'{field}={value!r}')):
        replace(gap, **{field: value})


def test_a_value_carrying_a_reason_or_lacking_a_git_check_is_refused() -> None:
    value = QualitySample.value(
        'p', T0, cursor_sha=None, landings_sampled=0, landings_deferred=0,
        git_check=GitCrossCheck(0, 0, 0),
    )

    with pytest.raises(ValueError, match=r"value.*reason='boom'"):
        replace(value, reason='boom')
    with pytest.raises(ValueError, match=r'value.*git_check=None'):
        replace(value, git_check=None)
    with pytest.raises(ValueError, match=r'value.*landings_deferred=-1'):
        replace(value, landings_deferred=-1)


def test_a_sample_whose_state_disagrees_with_its_fields_is_refused() -> None:
    value = QualitySample.value(
        'p', T0, cursor_sha='sha', landings_sampled=1, landings_deferred=0,
        git_check=GitCrossCheck(0, 0, 0),
    )

    with pytest.raises(ValueError, match=r'gap.*landings_sampled=1'):
        replace(value, state=SampleState.GAP, reason='boom')


def test_a_sample_time_without_an_offset_is_refused() -> None:
    with pytest.raises(ValueError, match=r'ts=datetime\.datetime\(2026, 11, 2, 9, 0\)'):
        QualitySample.gap('p', T0.replace(tzinfo=None), 'boom')
