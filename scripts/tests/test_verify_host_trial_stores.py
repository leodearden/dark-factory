"""verify_host_trial_stores: each project's runs.db and the samples store, read-only, into typed rows."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.project_layout import runs_db
from verify_host_trial_fixtures import (
    admission_plan,
    cap_hit,
    dispatch_deferred,
    insert_rows,
    invocation,
    lifecycle,
    merge_finalized,
    merge_verify,
    project_root,
    samples_db,
    scheduler_tick,
    task_completed,
    task_result,
    task_started,
    write_samples,
)
from verify_host_trial_stores import (
    CPU_ARM_SPECS,
    LOCK_CONFLICT_REASONS,
    AdmissionPlan,
    CapHit,
    CapKind,
    CpuSample,
    Deferral,
    EventType,
    Invocation,
    LegHold,
    LifecycleRow,
    ProjectStatus,
    SchedulerTick,
    Source,
    SourceState,
    TaskCompletion,
    TaskStart,
    columns,
    connect_read_only,
    read_cpu_pressure,
    read_project,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SINCE = datetime(2026, 11, 11, tzinfo=UTC)
UNTIL = datetime(2026, 11, 25, tzinfo=UTC)
HOST = 'laptop'


def day(n: float) -> datetime:
    return SINCE + timedelta(days=n)


@pytest.fixture
def store(tmp_path, make_runs_db):
    """Build a project root whose runs.db holds *events* and each table's *rows*; return the root."""

    def build(events=(), *, name='dark-factory', **tables):
        root = project_root(tmp_path, name)
        path = make_runs_db(events, directory=runs_db(root).parent)
        for table, rows in tables.items():
            insert_rows(path, table, rows)
        return root

    return build


def read(root):
    return read_project(root, HOST, SINCE, UNTIL)


class TestMergeVerify:
    def test_start_is_the_end_timestamp_less_the_payload_duration(self, store):
        project = read(store([merge_verify(day(2), HOST, 600, '7')]))

        (row,) = project.merge_verifies
        assert (row.start, row.end, row.runner, row.task_id) == (
            day(2) - timedelta(seconds=600),
            day(2),
            HOST,
            '7',
        )

    def test_only_rows_touching_the_window_are_returned(self, store):
        root = store([
            merge_verify(day(-20), HOST, 300, 'before'),
            merge_verify(SINCE + timedelta(minutes=5), HOST, 1800, 'straddles-since'),
            merge_verify(day(3), 'local', 900, 'inside'),
            merge_verify(UNTIL + timedelta(hours=2), HOST, 3 * 3600, 'started-inside'),
            merge_verify(UNTIL + timedelta(hours=2), HOST, 600, 'started-after'),
        ])

        assert {row.task_id for row in read(root).merge_verifies} == {
            'straddles-since', 'inside', 'started-inside'
        }

    def test_the_first_host_verify_is_the_earliest_host_start_over_the_whole_store(self, store):
        root = store([
            merge_verify(day(-40), 'local', 60, 'other-runner'),
            merge_verify(day(-20), HOST, 300, 'early-end'),
            merge_verify(day(-19), HOST, 3 * 86400, 'earlier-start'),
            merge_verify(day(2), HOST, 600, 'in-window'),
        ])

        assert read(root).first_host_verify_at == day(-22)

    def test_a_store_where_the_host_never_verified_has_no_first_verify(self, store):
        assert read(store([merge_verify(day(2), 'local', 600, '7')])).first_host_verify_at is None

    def test_a_row_that_does_not_parse_is_counted_not_dropped(self, store):
        bad_duration = merge_verify(day(2), HOST, 600, 'bad')
        del bad_duration['data']['duration_ms']
        not_json = {**merge_verify(day(3), HOST, 600, 'garbage'), 'data': 'garbage'}

        project = read(store([bad_duration, not_json, merge_verify(day(4), HOST, 60, 'ok')]))

        assert [row.task_id for row in project.merge_verifies] == ['ok']
        assert project.skipped[Source.MERGE_VERIFY] == 2


class TestLifecycle:
    def test_the_hosts_rows_over_the_whole_store_and_no_other_hosts(self, store):
        root = store([
            lifecycle(day(-3), 'verify_host_admitted', HOST),
            lifecycle(day(5), 'verify_host_retired', HOST),
            lifecycle(day(6), 'verify_host_admitted', 'boat'),
        ])

        project = read(root)

        assert project.lifecycle == (
            LifecycleRow(day(-3), EventType.VERIFY_HOST_ADMITTED),
            LifecycleRow(day(5), EventType.VERIFY_HOST_RETIRED),
        )
        assert project.lifecycle_ever_written

    def test_another_hosts_rows_prove_the_producer_wrote(self, store):
        project = read(store([lifecycle(day(6), 'verify_host_admitted', 'boat')]))

        assert project.lifecycle == ()
        assert project.lifecycle_ever_written

    def test_a_store_with_no_lifecycle_row_says_the_producer_never_wrote(self, store):
        project = read(store([merge_verify(day(2), HOST, 600, '7')]))

        assert not project.lifecycle_ever_written
        assert project.sources[Source.LIFECYCLE].state is SourceState.NEVER_WRITTEN


class TestLandings:
    def test_each_task_lands_once_at_its_first_done_row_over_the_whole_store(self, store):
        root = store(
            [
                merge_finalized(day(-2), '1', 'done'),
                merge_finalized(day(1), '2', 'blocked'),
                merge_finalized(day(4), '2', 'done'),
                merge_finalized(day(5), '4', 'conflict'),
                merge_finalized(day(6), '6', 'done'),
                merge_finalized(day(7), '6', 'done'),
            ],
            task_results=[
                task_result('1', 'done', day(3)),
                task_result('2', 'done', day(2), run_id='run-2'),
                task_result('3', 'requeued', day(1)),
            ],
        )

        landings = {landing.task_id: landing.at for landing in read(root).landings}

        assert landings == {'1': day(-2), '2': day(2), '6': day(6)}

    def test_a_done_result_without_a_completion_time_is_counted_as_skipped(self, store):
        project = read(store(task_results=[task_result('5', 'done', None)]))

        assert project.landings == ()
        assert project.skipped[Source.TASK_RESULTS] == 1


class TestTheOtherSources:
    def test_task_lifecycle_rows_carry_their_run(self, store):
        root = store([
            task_started(day(1), '10', 'run-a'),
            task_completed(day(1.5), '10', 'requeued', 'plan_blast_radius_lock_conflict', 'run-a'),
        ])

        project = read(root)

        assert project.task_starts == (TaskStart(day(1), 'run-a', '10'),)
        assert project.task_completions == (
            TaskCompletion(day(1.5), 'run-a', '10', 'requeued', 'plan_blast_radius_lock_conflict'),
        )

    def test_ticks_and_deferrals_carry_their_run_and_arm(self, store):
        root = store([
            dispatch_deferred(day(1) - timedelta(minutes=1), 'cpu_some_avg10', 'run-a'),
            scheduler_tick(day(1), 'run-a'),
        ])

        project = read(root)

        assert project.deferrals == (Deferral(day(1) - timedelta(minutes=1), 'run-a', 'cpu_some_avg10'),)
        assert project.ticks == (SchedulerTick(day(1), 'run-a'),)

    def test_task_admission_plans_carry_their_leg_holds_and_background_plans_are_dropped(self, store):
        arrival, completed = day(1), day(1) + timedelta(seconds=100)
        legs = [(arrival + timedelta(seconds=10), 20.0), (arrival + timedelta(seconds=50), 10.0)]
        root = store([
            admission_plan('10', 'task', arrival, completed, legs),
            admission_plan(None, 'background', arrival, completed, legs),
        ])

        assert read(root).admission_plans == (
            AdmissionPlan(
                '10',
                arrival,
                completed,
                (
                    LegHold(arrival + timedelta(seconds=10), arrival + timedelta(seconds=30)),
                    LegHold(arrival + timedelta(seconds=50), arrival + timedelta(seconds=60)),
                ),
            ),
        )

    def test_invocations_are_read_from_a_full_cycle_before_since(self, store):
        root = store(invocations=[
            invocation(day(-8), 'max-a', 9.0),
            invocation(day(-3), 'max-a', 1.5),
            invocation(day(1), 'max-b', 2.5),
        ])

        assert read(root).invocations == (
            Invocation(day(-3), 'max-a', 1.5),
            Invocation(day(1), 'max-b', 2.5),
        )

    def test_cap_hits_carry_their_reset_or_none(self, store):
        root = store(account_events=[cap_hit(day(2), 'max-a', day(4)), cap_hit(day(3), 'max-b', None)])

        assert read(root).cap_hits == (
            CapHit('max-a', day(2), day(4), CapKind.WEEKLY),
            CapHit('max-b', day(3), None, CapKind.WEEKLY),
        )

    def test_each_lock_conflict_reason_is_one_the_workflow_writes(self):
        """The producer passes each reason as a bare literal; scripts/tests imports no orchestrator, so read its text."""
        producer = (REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'workflow.py').read_text(encoding='utf-8')

        assert LOCK_CONFLICT_REASONS
        for reason in LOCK_CONFLICT_REASONS:
            assert f"'{reason}'" in producer or f'"{reason}"' in producer, reason


FABLE_SCOPE = 'claude-fable-5-1'


class TestCapKind:
    """Each cap_hit is classified at the store boundary; a reason it does not recognise is read, never dropped."""

    @pytest.mark.parametrize(
        ('reason', 'scope', 'kind'),
        [
            pytest.param(
                "You've hit your weekly limit · resets Nov 19, 1pm (Europe/London)", None, CapKind.WEEKLY, id='weekly'
            ),
            pytest.param("You've hit your session limit · resets 3pm (Europe/London)", None, CapKind.SESSION, id='session'),
            pytest.param("You've hit your 5-hour limit · resets 7pm (Europe/London)", None, CapKind.SESSION, id='5-hour'),
            pytest.param(
                "You've hit your Sonnet limit · resets 2pm (Europe/London)", None, CapKind.PER_MODEL, id='model-word'
            ),
            pytest.param(
                "You've hit your Fable limit · resets Nov 25, 10am (Europe/London). Switch to another model to continue.",
                FABLE_SCOPE,
                CapKind.PER_MODEL,
                id='model-word-and-scope',
            ),
            pytest.param(
                "You've hit your weekly limit · resets 1pm (Europe/London)",
                FABLE_SCOPE,
                CapKind.PER_MODEL,
                id='scope-wins-over-weekly',
            ),
            pytest.param(
                "You're out of extra usage · resets 9:40am (Europe/London)", None, CapKind.UNRECOGNISED, id='extra-usage'
            ),
            pytest.param("You've hit your org's monthly usage limit", None, CapKind.UNRECOGNISED, id='org-monthly'),
            pytest.param(
                "You've hit your limit · resets 11:10pm (Europe/London)", None, CapKind.UNRECOGNISED, id='no-limit-word'
            ),
            pytest.param(
                'Heuristic cap: zero-cost instant exit — Agent produced no output',
                None,
                CapKind.UNRECOGNISED,
                id='heuristic',
            ),
            pytest.param(None, None, CapKind.UNRECOGNISED, id='no-reason'),
        ],
    )
    def test_a_live_reason_is_read_with_its_kind(self, store, reason, scope, kind):
        root = store(account_events=[cap_hit(day(2), 'max-a', day(4), reason=reason, scope=scope)])

        reading = read(root)

        assert reading.cap_hits == (CapHit('max-a', day(2), day(4), kind),)
        assert reading.skipped[Source.CAP_HIT] == 0

    def test_a_reason_that_is_not_text_is_counted_as_skipped(self, store):
        malformed = {**cap_hit(day(2), 'max-a', day(4)), 'details': {'reason': 42, 'resets_at': day(4).isoformat()}}
        root = store(account_events=[malformed, cap_hit(day(3), 'max-b', day(5))])

        reading = read(root)

        assert reading.cap_hits == (CapHit('max-b', day(3), day(5), CapKind.WEEKLY),)
        assert reading.skipped[Source.CAP_HIT] == 1


class TestReadOnly:
    def test_the_connection_refuses_a_write(self, store):
        path = runs_db(store())
        conn = connect_read_only(path)
        try:
            with pytest.raises(sqlite3.OperationalError, match='readonly'):
                conn.execute("INSERT INTO events (timestamp, run_id, event_type) VALUES ('t', 'r', 'x')")
        finally:
            conn.close()

    def test_a_read_leaves_the_store_byte_identical(self, store):
        root = store([merge_verify(day(2), HOST, 600, '7'), scheduler_tick(day(1))])
        path = runs_db(root)
        before = path.read_bytes()

        read(root)

        assert path.read_bytes() == before


class TestColumnsAreAsked:
    def test_columns_lists_a_tables_columns(self, store):
        conn = connect_read_only(runs_db(store()))
        try:
            assert {'task_id', 'outcome', 'completed_at'} <= columns(conn, 'task_results')
            assert columns(conn, 'no_such_table') == frozenset()
        finally:
            conn.close()

    def test_a_missing_column_makes_that_source_unavailable_and_the_rest_still_read(self, store):
        root = store([merge_finalized(day(2), '1', 'done'), merge_verify(day(2), HOST, 600, '7')])
        conn = sqlite3.connect(runs_db(root))
        conn.executescript(
            'DROP TABLE task_results;'
            'CREATE TABLE task_results (run_id TEXT, task_id TEXT, outcome TEXT);'
            "INSERT INTO task_results VALUES ('run-1', '2', 'done');"
        )
        conn.close()

        project = read(root)

        source = project.sources[Source.TASK_RESULTS]
        assert source.state is SourceState.UNAVAILABLE
        assert source.reason is not None
        assert 'task_results' in source.reason
        assert 'completed_at' in source.reason
        assert project.sources[Source.MERGE_FINALIZED].state is SourceState.READ
        assert [landing.task_id for landing in project.landings] == ['1']
        assert len(project.merge_verifies) == 1

    def test_a_missing_table_makes_that_source_unavailable(self, store):
        root = store([merge_verify(day(2), HOST, 600, '7')])
        conn = sqlite3.connect(runs_db(root))
        conn.execute('DROP TABLE task_results')
        conn.close()

        source = read(root).sources[Source.TASK_RESULTS]

        assert source.state is SourceState.UNAVAILABLE
        assert source.reason is not None
        assert 'task_results' in source.reason


class TestProducerProof:
    def test_a_producer_that_never_wrote_differs_from_rows_outside_the_window(self, store):
        never = read(store([merge_verify(day(2), HOST, 600, '7')], name='never'))
        outside = read(store([scheduler_tick(day(-10))], name='outside'))

        assert never.sources[Source.SCHEDULER_TICK].state is SourceState.NEVER_WRITTEN
        assert outside.sources[Source.SCHEDULER_TICK].state is SourceState.READ
        assert outside.ticks == ()

    def test_an_empty_table_is_a_producer_that_never_wrote(self, store):
        project = read(store())

        assert project.sources[Source.INVOCATIONS].state is SourceState.NEVER_WRITTEN
        assert project.sources[Source.TASK_RESULTS].state is SourceState.NEVER_WRITTEN


class TestTheStoreItself:
    def test_a_missing_runs_db_is_named(self, tmp_path):
        root = project_root(tmp_path, 'pump-web-ui')

        project = read(root)

        assert project.status is ProjectStatus.MISSING
        assert project.runs_db == runs_db(root)

    def test_a_corrupt_runs_db_is_unreadable_with_the_sqlite_error(self, tmp_path):
        root = project_root(tmp_path, 'corrupt')
        runs_db(root).write_bytes(b'this is not a sqlite database at all' * 100)

        project = read(root)

        assert project.status is ProjectStatus.UNREADABLE
        assert project.detail is not None
        assert 'not a database' in project.detail


class TestCpuPressure:
    def test_the_arm_specs_are_the_two_cpu_gate_arms(self):
        assert {spec.arm for spec in CPU_ARM_SPECS} == {'cpu_some_avg10', 'runqueue_ratio'}

    def test_samples_in_the_window_are_read_and_an_unreadable_one_dropped(self, tmp_path):
        db = samples_db(tmp_path / 'load-samples.db')
        t0 = int(SINCE.timestamp())
        gated = next(spec for spec in CPU_ARM_SPECS if spec.readability is not None)
        assert gated.readability is not None
        write_samples(db, [
            *((t0 - 60, spec, 9.0) for spec in CPU_ARM_SPECS),
            *((t0 + 60, spec, 1.0) for spec in CPU_ARM_SPECS),
            *((t0 + 120, spec, 5.0) for spec in CPU_ARM_SPECS),
            *((int(UNTIL.timestamp()), spec, 9.0) for spec in CPU_ARM_SPECS),
            (t0 + 60, gated.readability, 1.0),
            (t0 + 120, gated.readability, 0.0),
        ])
        before = db.read_bytes()

        cpu = read_cpu_pressure(db, SINCE, UNTIL)

        assert cpu.reason is None
        at_60, at_120 = SINCE + timedelta(seconds=60), SINCE + timedelta(seconds=120)
        for spec in CPU_ARM_SPECS:
            expected = (CpuSample(at_60, 1.0),) if spec is gated else (CpuSample(at_60, 1.0), CpuSample(at_120, 5.0))
            assert cpu.samples[spec] == expected
        assert cpu.unreadable_dropped == 1
        assert db.read_bytes() == before

    def test_a_missing_samples_store_is_unavailable_not_an_exception(self, tmp_path):
        absent = tmp_path / 'absent.db'

        cpu = read_cpu_pressure(absent, SINCE, UNTIL)

        assert cpu.reason is not None
        assert str(absent) in cpu.reason
        assert all(samples == () for samples in cpu.samples.values())
