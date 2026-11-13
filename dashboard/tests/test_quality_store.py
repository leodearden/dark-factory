"""quality.db: where it lives, its tables, and the LandingRow / sample codec over them."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
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
from dashboard.data.quality_markers import (
    AxisLanding,
    DerivedState,
    Marker,
    MarkerKind,
    MarkerSource,
    PromptCoverage,
    RoutingFold,
    RoutingStep,
    RuleSpan,
    SourceLayer,
    SourceType,
    StoredMarkerState,
)
from dashboard.data.quality_store import (
    NewestSamples,
    QualitySample,
    SampleState,
    _require_every_field_coded,
    insert_sample,
    landing_axis,
    newest_samples,
    read_landing_rows,
    read_marker_state,
    read_routing_fold,
    read_samples,
    record_marker_failure,
    stored_cursor,
    upsert_landing_rows,
    write_marker_state,
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


QUALITY_TABLES = {
    'landings', 'late_events', 'sample_state',
    'markers_derived', 'routing_steps', 'routing_rules', 'marker_state',
}


async def _tables(conn: aiosqlite.Connection) -> set[str]:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ) as cursor:
        return {name for (name,) in await cursor.fetchall()}


async def test_a_fresh_store_has_every_quality_table_and_no_rows(
    conn: aiosqlite.Connection,
) -> None:
    assert await _tables(conn) == QUALITY_TABLES
    assert await read_landing_rows(conn, 'p') == []
    assert await read_samples(conn, 'p') == []
    assert await read_marker_state(conn, 'p') is None
    assert await landing_axis(conn, 'p') == []


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


async def _edit_stored_rule_vintage(
    conn: aiosqlite.Connection, edit: Callable[[dict[str, object]], object],
) -> None:
    """Stand in for a row an older SourceEpochs wrote, which nothing will ever re-measure."""
    async with conn.execute('SELECT rule_vintage FROM landings') as cursor:
        [(text,)] = await cursor.fetchall()
    stored = json.loads(text)
    edit(stored)
    await conn.execute('UPDATE landings SET rule_vintage = ?', (json.dumps(stored),))
    await conn.commit()


async def test_a_source_added_since_a_row_was_stored_reads_as_not_started(
    conn: aiosqlite.Connection,
) -> None:
    row = _train_member('102')
    await _store_rows(conn, [row])

    await _edit_stored_rule_vintage(conn, lambda stored: stored.pop('review_completed'))

    assert await read_landing_rows(conn, 'p') == [
        replace(row, rule_vintage=replace(row.rule_vintage, review_completed=None)),
    ]


async def test_a_stored_source_that_source_epochs_lacks_is_refused_by_name(
    conn: aiosqlite.Connection,
) -> None:
    await _store_rows(conn, [_train_member('102')])

    await _edit_stored_rule_vintage(conn, lambda stored: stored.update(retired_source=None))

    with pytest.raises(ValueError, match=re.escape("['retired_source']")):
        await read_landing_rows(conn, 'p')


def test_a_codec_out_of_step_with_its_dataclass_names_both_sides() -> None:
    @dataclass(frozen=True)
    class Grown:
        kept: int
        added: int

    with pytest.raises(TypeError, match=re.escape("uncoded ['added'], not fields ['dropped']")):
        _require_every_field_coded(Grown, ('kept', 'dropped'))


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


def _value_sample(project_id: str, ts: datetime, cursor_sha: str = 'sha') -> QualitySample:
    return QualitySample.value(
        project_id, ts, cursor_sha=cursor_sha, landings_sampled=1, landings_deferred=0,
        git_check=GitCrossCheck(off_first_parent=0, merges_without_landing=0, late_claims=0),
    )


async def _insert_samples(conn: aiosqlite.Connection, samples: list[QualitySample]) -> None:
    for sample in samples:
        await insert_sample(conn, sample)
    await conn.commit()


async def test_newest_samples_of_an_empty_store_are_none(conn: aiosqlite.Connection) -> None:
    assert await newest_samples(conn, 'p') == NewestSamples(latest=None, latest_value=None)


async def test_newest_samples_name_the_newest_sample_and_the_newest_value(
    conn: aiosqlite.Connection,
) -> None:
    older_value = _value_sample('p', T0, 'older')
    value = _value_sample('p', T0 + timedelta(minutes=10), 'newer')
    gap = QualitySample.gap('p', T0 + timedelta(minutes=20), 'OperationalError: database is locked')
    await _insert_samples(conn, [gap, older_value, value])

    assert await newest_samples(conn, 'p') == NewestSamples(latest=gap, latest_value=value)


async def test_newest_samples_with_only_gaps_have_no_value(conn: aiosqlite.Connection) -> None:
    first = QualitySample.gap('p', T0, 'GitUnavailable: not a git repository')
    second = QualitySample.gap('p', T0 + timedelta(minutes=10), 'GitTimeout: rev-list')
    await _insert_samples(conn, [first, second])

    assert await newest_samples(conn, 'p') == NewestSamples(latest=second, latest_value=None)


async def test_newest_samples_never_read_another_projects_samples(
    conn: aiosqlite.Connection,
) -> None:
    own = _value_sample('p', T0)
    await _insert_samples(conn, [
        own,
        _value_sample('q', T0 + timedelta(hours=1)),
        QualitySample.gap('q', T0 + timedelta(hours=2), 'not this project'),
    ])

    assert await newest_samples(conn, 'p') == NewestSamples(latest=own, latest_value=own)
    assert await newest_samples(conn, 'r') == NewestSamples(latest=None, latest_value=None)


async def test_newest_samples_break_a_tie_on_ts_by_insertion_like_read_samples(
    conn: aiosqlite.Connection,
) -> None:
    first = _value_sample('p', T0, 'first')
    second = _value_sample('p', T0, 'second')
    await _insert_samples(conn, [first, second])

    newest = await newest_samples(conn, 'p')

    assert newest == NewestSamples(latest=second, latest_value=second)
    assert (await read_samples(conn, 'p'))[-1] == newest.latest


DF = 'dark_factory'
DECIDED = datetime(2026, 10, 27, 13, 0, tzinfo=UTC)
REFRESHED = DECIDED + timedelta(days=5)
DIGEST = 'a' * 64


def _minute(minutes: int) -> datetime:
    return DECIDED + timedelta(minutes=minutes)


def _route_step(
    role: str, first_event_id: int, decisions: int, *, layer: SourceLayer = SourceLayer.CONFIG,
    model: str = 'opus', digest: str | None = None,
) -> RoutingStep:
    return RoutingStep(
        role=role, first_event_id=first_event_id, first_at=_minute(first_event_id),
        last_event_id=first_event_id + decisions - 1,
        last_at=_minute(first_event_id + decisions - 1), decisions=decisions,
        source_layer=layer, model=model, effort='high', budget_usd=12.5, max_turns=60,
        prompt_digest=digest,
    )


STEPS = (
    _route_step('merger', 10, 3),
    _route_step('reviewer_comprehensive', 11, 4, model='fable'),
    _route_step('merger', 20, 1, layer=SourceLayer.ROLE_DEFAULT),
    _route_step('merger', 30, 6, digest=DIGEST),
)
RULES = (
    RuleSpan('large-plan-steps', first_event_id=12, first_at=_minute(12), last_at=_minute(40),
             decisions=7),
    RuleSpan('retry-tier-up', first_event_id=31, first_at=_minute(31), last_at=_minute(31),
             decisions=1),
)

MARKER_SOURCES: dict[SourceType, tuple[MarkerKind, MarkerSource, str | None]] = {
    SourceType.ROUTING: (
        MarkerKind.EFFORT, MarkerSource(SourceType.ROUTING, role='merger', path='effort.merger'),
        '30',
    ),
    SourceType.CONFIG_RELOAD: (
        MarkerKind.REVIEWER_POLICY,
        MarkerSource(SourceType.CONFIG_RELOAD, path='review.reviewer_briefing'), '7',
    ),
    SourceType.POLICY_RULE: (
        MarkerKind.ROUTING_RULE, MarkerSource(SourceType.POLICY_RULE, rule='large-plan-steps'),
        None,
    ),
    SourceType.DEPLOY: (MarkerKind.DEPLOY, MarkerSource(SourceType.DEPLOY), 'c' * 40),
    SourceType.MODEL_VERSION: (
        MarkerKind.MODEL, MarkerSource(SourceType.MODEL_VERSION, alias='opus'), 'claude-opus-5-5',
    ),
    SourceType.INSTRUMENT: (MarkerKind.INSTRUMENT, MarkerSource(SourceType.INSTRUMENT), 'd' * 40),
    SourceType.CURATED: (
        MarkerKind.REGIME, MarkerSource(SourceType.CURATED, file='regime-markers.yaml'), None,
    ),
}


def _derived_marker(project_id: str, source_type: SourceType, minutes: int) -> Marker:
    kind, source, ref = MARKER_SOURCES[source_type]
    return Marker(
        marker_id=f'{source_type.value}:{minutes}', at=_minute(minutes), kind=kind,
        scope=project_id, series=None, until=None, label=f'{source_type.value} changed',
        source=source, ref=ref,
    )


def _every_source_marker(project_id: str) -> tuple[Marker, ...]:
    return tuple(
        _derived_marker(project_id, source_type, 10 + index)
        for index, source_type in enumerate(SourceType)
    )


COUNTED = PromptCoverage(digest_since=None, roles_py_landings=4, counted_since=DECIDED,
                         reason=None)
UNCOUNTED = PromptCoverage(digest_since=None, roles_py_landings=None, counted_since=None,
                           reason='GitUnavailable: not a git repository')
WRITTEN = PromptCoverage(digest_since=_minute(30), roles_py_landings=None, counted_since=None,
                         reason=None)


def _fold(project_id: str = DF, *, digest_since: datetime | None = None) -> RoutingFold:
    return RoutingFold(
        project_id=project_id, cursor=41, steps=STEPS, rules=RULES, first_at=_minute(10),
        newest_at=_minute(40), digest_since=digest_since, skipped=2, complete=False,
    )


def _derived_state(
    project_id: str = DF, *, coverage: PromptCoverage = COUNTED,
    refreshed_at: datetime = REFRESHED,
) -> DerivedState:
    return DerivedState(
        fold=_fold(project_id, digest_since=coverage.digest_since),
        markers=_every_source_marker(project_id), skipped=3, prompt_coverage=coverage,
        refreshed_at=refreshed_at,
    )


async def _write(conn: aiosqlite.Connection, state: DerivedState) -> None:
    await write_marker_state(conn, state)
    await conn.commit()


def _stored(state: DerivedState) -> StoredMarkerState:
    return StoredMarkerState(derived=state, attempted_at=state.refreshed_at, failure=None)


def test_the_marker_fixture_covers_every_source_type() -> None:
    assert {marker.source.type for marker in _every_source_marker(DF)} == set(SourceType)


@pytest.mark.parametrize('coverage', [COUNTED, UNCOUNTED, WRITTEN],
                         ids=['counted', 'uncounted', 'written'])
async def test_a_derived_state_round_trips_by_value(
    conn: aiosqlite.Connection, coverage: PromptCoverage,
) -> None:
    state = _derived_state(coverage=coverage)

    await _write(conn, state)

    assert await read_marker_state(conn, DF) == _stored(state)


async def test_a_complete_empty_fold_round_trips(conn: aiosqlite.Connection) -> None:
    state = DerivedState(
        fold=replace(RoutingFold.empty('reify'), cursor=9, skipped=9, complete=True), markers=(),
        skipped=0, prompt_coverage=UNCOUNTED, refreshed_at=REFRESHED,
    )

    await _write(conn, state)

    assert await read_marker_state(conn, 'reify') == _stored(state)


async def test_a_second_write_replaces_the_projects_state_and_no_other(
    conn: aiosqlite.Connection,
) -> None:
    other = _derived_state('reify')
    first = _derived_state()
    later = REFRESHED + timedelta(minutes=10)
    second = replace(
        first,
        fold=replace(first.fold, cursor=60, steps=STEPS[:1], rules=RULES[1:], complete=True),
        markers=first.markers[:2], skipped=0, refreshed_at=later,
    )
    await _write(conn, other)
    await _write(conn, first)

    await _write(conn, second)

    assert await read_marker_state(conn, DF) == _stored(second)
    assert await read_marker_state(conn, 'reify') == _stored(other)


async def test_a_failure_keeps_the_previous_derivation_and_names_the_attempt(
    conn: aiosqlite.Connection,
) -> None:
    state = _derived_state()
    attempted = REFRESHED + timedelta(minutes=10)
    await _write(conn, state)

    await record_marker_failure(conn, DF, attempted_at=attempted,
                                failure='OperationalError: database is locked')
    await conn.commit()

    assert await read_marker_state(conn, DF) == StoredMarkerState(
        derived=state, attempted_at=attempted, failure='OperationalError: database is locked',
    )


async def test_a_failure_before_any_derivation_has_no_derived_state(
    conn: aiosqlite.Connection,
) -> None:
    await record_marker_failure(conn, DF, attempted_at=REFRESHED, failure='runs.db is absent')
    await conn.commit()

    assert await read_marker_state(conn, DF) == StoredMarkerState(
        derived=None, attempted_at=REFRESHED, failure='runs.db is absent',
    )
    assert await read_marker_state(conn, 'reify') is None


async def test_a_derivation_after_a_failure_clears_it(conn: aiosqlite.Connection) -> None:
    await record_marker_failure(conn, DF, attempted_at=REFRESHED, failure='runs.db is absent')
    state = _derived_state(refreshed_at=REFRESHED + timedelta(minutes=10))

    await _write(conn, state)

    assert await read_marker_state(conn, DF) == _stored(state)


async def test_the_routing_fold_reads_alone_as_its_derivation_stored_it(
    conn: aiosqlite.Connection,
) -> None:
    state = _derived_state()
    await _write(conn, _derived_state('reify'))
    await _write(conn, state)
    await record_marker_failure(conn, DF, attempted_at=REFRESHED + timedelta(minutes=10),
                                failure='OperationalError: database is locked')
    await conn.commit()

    assert await read_routing_fold(conn, DF) == state.fold


async def test_no_derivation_is_no_routing_fold(conn: aiosqlite.Connection) -> None:
    await record_marker_failure(conn, DF, attempted_at=REFRESHED, failure='runs.db is absent')
    await conn.commit()

    assert await read_routing_fold(conn, DF) is None
    assert await read_routing_fold(conn, 'reify') is None


async def test_the_routing_fold_reads_without_decoding_the_markers(
    conn: aiosqlite.Connection,
) -> None:
    state = _derived_state()
    await _write(conn, state)
    await conn.execute("UPDATE markers_derived SET kind = 'retired_kind' WHERE project_id = ?",
                       (DF,))
    await conn.commit()

    with pytest.raises(ValueError, match='retired_kind'):
        await read_marker_state(conn, DF)
    assert await read_routing_fold(conn, DF) == state.fold


def _with_marker(**overrides: object) -> DerivedState:
    state = _derived_state()
    return replace(state, markers=(*state.markers, replace(state.markers[0], **overrides)))


@pytest.mark.parametrize('state', [
    _with_marker(marker_id='bad', scope='reify'),
    _with_marker(marker_id='bad', scope=None),
    _with_marker(marker_id='bad', series=('flakes_confirmed',)),
    _with_marker(marker_id='bad', tasks=(8460,)),
    _with_marker(marker_id='bad', ordinal=3),
    _with_marker(marker_id='bad', kind=MarkerKind.ARTEFACT, until=_minute(90)),
], ids=['other-project', 'fleet', 'series', 'tasks', 'placed', 'artefact-range'])
async def test_a_marker_the_table_cannot_carry_is_refused_by_id(
    conn: aiosqlite.Connection, state: DerivedState,
) -> None:
    with pytest.raises(ValueError, match="marker_id='bad'"):
        await write_marker_state(conn, state)


async def test_a_coverage_disagreeing_with_its_folds_digest_is_refused(
    conn: aiosqlite.Connection,
) -> None:
    state = replace(_derived_state(coverage=WRITTEN), fold=_fold(digest_since=_minute(31)))

    with pytest.raises(ValueError, match='digest_since'):
        await write_marker_state(conn, state)


async def test_the_landing_axis_is_the_projects_ordinals_in_order(
    conn: aiosqlite.Connection,
) -> None:
    hour = timedelta(hours=1)
    await _store_rows(conn, [
        _bare_row(_landing('103', 'third-sha', 2, T0 + 2 * hour)),
        _bare_row(_landing('101', 'first-sha', 0, T0)),
        _bare_row(_landing('102', 'second-sha', 1, T0 + hour, late_claim=True)),
        _bare_row(_landing('901', 'other-sha', 0, T0 - hour, project_id='q')),
    ])

    assert await landing_axis(conn, 'p') == [
        AxisLanding(0, T0), AxisLanding(1, T0 + hour), AxisLanding(2, T0 + 2 * hour),
    ]
    assert await landing_axis(conn, 'r') == []


BETA_MARKERS_DERIVED = """\
CREATE TABLE markers_derived (
    project_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    at          TEXT    NOT NULL,
    ordinal     INTEGER,
    PRIMARY KEY (project_id, role, kind, at)
);
INSERT INTO markers_derived VALUES ('p', 'merger', 'model', '2026-10-27T13:46:00+00:00', 4);
"""
"""The markers table as quality.db carried it before the marker producer existed."""


def _make_beta_era(path: Path) -> None:
    raw = sqlite3.connect(path)
    try:
        raw.executescript(
            'DROP TABLE IF EXISTS markers_derived; DROP TABLE IF EXISTS routing_steps; '
            'DROP TABLE IF EXISTS routing_rules; DROP TABLE IF EXISTS marker_state; '
            + BETA_MARKERS_DERIVED
        )
    finally:
        raw.close()


async def _columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    async with conn.execute(f'PRAGMA table_info({table})') as cursor:
        return {row[1] for row in await cursor.fetchall()}


async def test_opening_a_beta_era_store_reshapes_its_markers_and_keeps_its_landings(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'quality.db'
    row = _train_member('102')
    sample = _value_sample('p', T0)
    async with _QualityStore(path) as store:
        await _store_rows(store.connection, [row])
        await _insert_samples(store.connection, [sample])
    _make_beta_era(path)

    async with _QualityStore(path) as store:
        conn = store.connection
        assert await _tables(conn) == QUALITY_TABLES
        assert {'marker_id', 'label', 'source'} <= await _columns(conn, 'markers_derived')
        assert not {'role', 'ordinal'} & await _columns(conn, 'markers_derived')
        async with conn.execute('SELECT COUNT(*) FROM markers_derived') as cursor:
            assert await cursor.fetchone() == (0,)
        assert await read_landing_rows(conn, 'p') == [row]
        assert await read_samples(conn, 'p') == [sample]
        await _write(conn, _derived_state())


async def test_opening_a_current_store_again_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / 'quality.db'
    state = _derived_state()
    row = _train_member('102')
    async with _QualityStore(path) as store:
        await _store_rows(store.connection, [row])
        await _write(store.connection, state)

    async with _QualityStore(path) as store:
        assert await read_marker_state(store.connection, DF) == _stored(state)
        assert await read_landing_rows(store.connection, 'p') == [row]
