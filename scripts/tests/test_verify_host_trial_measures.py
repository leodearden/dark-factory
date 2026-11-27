"""verify_host_trial_measures: each per-block figure, from typed store rows."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from verify_host_trial_blocks import Block, BlockOrigin, HostState
from verify_host_trial_measures import (
    FIGURE_RULES,
    CpuStat,
    DeferralHours,
    FigureId,
    Measured,
    NeverWritten,
    NoHostVerify,
    RunnerMedian,
    TurnWait,
    Unavailable,
    cpu_pressure,
    deferral_hours_by_metric,
    host_verifies,
    landings,
    landings_per_day,
    lock_conflict_requeues,
    merge_verify_median_by_runner,
    project_figures,
    spend_per_day,
    turn_wait_share,
)
from verify_host_trial_stores import (
    CPU_ARM_SPECS,
    AdmissionPlan,
    CpuRead,
    CpuSample,
    Deferral,
    Invocation,
    Landing,
    LegHold,
    MergeVerify,
    ProjectRead,
    ProjectStatus,
    SchedulerTick,
    Source,
    SourceRead,
    SourceState,
    TaskCompletion,
    TaskStart,
)

T0 = datetime(2026, 11, 11, tzinfo=UTC)
HOST = 'laptop'


def day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def sec(n: float) -> datetime:
    return T0 + timedelta(seconds=n)


def block(start: datetime, end: datetime, state: HostState = HostState.ON) -> Block:
    return Block(state, start, end, BlockOrigin.ADMITTED, start)


A = block(day(0), day(2))
B = block(day(2), day(4), HostState.OFF)


def project(never: tuple[Source, ...] = (), unavailable: tuple[Source, ...] = (), **rows) -> ProjectRead:
    """A read project whose every source read, except those named never written or unavailable."""

    def state(source: Source) -> SourceRead:
        if source in never:
            return SourceRead(SourceState.NEVER_WRITTEN, None, 0)
        if source in unavailable:
            return SourceRead(SourceState.UNAVAILABLE, f'{source} lacks column x', 0)
        return SourceRead(SourceState.READ, None, 0)

    sources = MappingProxyType({source: state(source) for source in Source})
    return ProjectRead(Path('p'), Path('p/runs.db'), ProjectStatus.READ, None, sources=sources, **rows)


def verify(start: datetime, seconds: float, runner: str = HOST) -> MergeVerify:
    return MergeVerify(start, start + timedelta(seconds=seconds), runner, '7')


class TestLandingsPerDay:
    def test_each_first_landing_counts_once_in_the_block_holding_it(self):
        landings = (Landing('1', day(1)), Landing('2', day(1.5)), Landing('3', day(3)))
        read = project(landings=landings)

        assert landings_per_day(read, A) == Measured(1.0)
        assert landings_per_day(read, B) == Measured(0.5)

    def test_the_count_behind_the_rate_is_a_figure_too(self):
        read = project(landings=(Landing('1', day(1)), Landing('2', day(1.5)), Landing('3', day(3))))

        assert landings(read, A) == Measured(2)
        assert landings(read, B) == Measured(1)

    def test_a_store_where_neither_producer_wrote(self):
        read = project(never=(Source.MERGE_FINALIZED, Source.TASK_RESULTS))

        assert landings_per_day(read, A) == NeverWritten('merge_finalized or task_results done')
        assert landings(read, A) == NeverWritten('merge_finalized or task_results done')

    def test_an_unavailable_landing_source_makes_the_figure_unavailable(self):
        figure = landings_per_day(project(unavailable=(Source.TASK_RESULTS,)), A)

        assert isinstance(figure, Unavailable)
        assert 'task_results' in figure.reason


class TestMergeVerify:
    def test_the_median_wall_per_runner_assigned_by_start(self):
        rows = (
            verify(day(0.5), 600),
            verify(day(1), 3000),
            verify(day(2) - timedelta(seconds=60), 1200),
            verify(day(1), 100, 'local'),
            verify(day(1.5), 300, 'local'),
        )

        read = project(merge_verifies=rows)

        assert merge_verify_median_by_runner(read, A) == Measured(
            {HOST: RunnerMedian(1200.0, 3), 'local': RunnerMedian(200.0, 2)}
        )
        assert merge_verify_median_by_runner(read, B) == Measured({})

    def test_host_verifies_counts_host_starts_and_zero_is_no_host_verify(self):
        rows = (verify(day(0.5), 600), verify(day(2) - timedelta(seconds=60), 1200), verify(day(3), 60, 'local'))
        read = project(merge_verifies=rows)

        assert host_verifies(read, HOST, A) == Measured(2)
        assert host_verifies(read, HOST, B) == NoHostVerify()

    def test_host_verifies_in_a_store_that_never_verified(self):
        assert host_verifies(project(never=(Source.MERGE_VERIFY,)), HOST, A) == NeverWritten('merge_verify')


class TestTurnWaitShare:
    WINDOW = block(sec(0), sec(1000))

    def test_the_wait_is_plan_time_no_leg_held_a_slot_over_run_slot_time(self):
        plan = AdmissionPlan('10', sec(0), sec(100), (LegHold(sec(10), sec(30)), LegHold(sec(50), sec(60))))
        read = project(
            admission_plans=(plan,),
            task_starts=(TaskStart(sec(0), 'run-a', '10'),),
            task_completions=(TaskCompletion(sec(500), 'run-a', '10', 'done', ''),),
        )

        assert turn_wait_share(read, self.WINDOW) == Measured(TurnWait(70 / 500, 70.0, 500.0, 0))

    def test_overlapping_legs_cover_their_union_once(self):
        plan = AdmissionPlan('10', sec(0), sec(100), (LegHold(sec(10), sec(40)), LegHold(sec(20), sec(50))))
        read = project(
            admission_plans=(plan,),
            task_starts=(TaskStart(sec(0), 'run-a', '10'),),
            task_completions=(TaskCompletion(sec(200), 'run-a', '10', 'done', ''),),
        )

        figure = turn_wait_share(read, self.WINDOW)

        assert isinstance(figure, Measured)
        assert figure.value.wait_secs == 60.0

    def test_plans_and_run_slots_are_clipped_to_the_block(self):
        plan = AdmissionPlan('10', sec(-50), sec(20), (LegHold(sec(-40), sec(-10)),))
        read = project(
            admission_plans=(plan,),
            task_starts=(TaskStart(sec(-100), 'run-a', '10'),),
            task_completions=(TaskCompletion(sec(400), 'run-a', '10', 'done', ''),),
        )

        assert turn_wait_share(read, self.WINDOW) == Measured(TurnWait(20 / 400, 20.0, 400.0, 0))

    def test_an_unpaired_start_is_excluded_and_counted(self):
        read = project(
            admission_plans=(),
            task_starts=(
                TaskStart(sec(0), 'run-a', '10'),
                TaskStart(sec(600), 'run-a', '11'),
                TaskStart(sec(0), 'run-b', '10'),
            ),
            task_completions=(TaskCompletion(sec(500), 'run-a', '10', 'done', ''),),
        )

        assert turn_wait_share(read, self.WINDOW) == Measured(TurnWait(0.0, 0.0, 500.0, 2))

    def test_a_store_whose_admission_plans_never_wrote(self):
        read = project(never=(Source.ADMISSION_PLAN,))

        assert turn_wait_share(read, self.WINDOW) == NeverWritten('verify_admission_plan')

    def test_no_run_slot_time_in_the_block_is_unavailable(self):
        assert isinstance(turn_wait_share(project(), self.WINDOW), Unavailable)


class TestDeferralHours:
    def test_each_deferral_spans_its_tick_to_the_next_tick_of_its_run(self):
        ticks = (
            SchedulerTick(sec(60), 'run-a'),
            SchedulerTick(sec(100), 'run-b'),
            SchedulerTick(sec(135), 'run-a'),
            SchedulerTick(sec(210), 'run-a'),
        )
        deferrals = (
            Deferral(sec(50), 'run-a', 'cpu_some_avg10'),
            Deferral(sec(130), 'run-a', 'io_some_avg10'),
            Deferral(sec(200), 'run-a', 'cpu_some_avg10'),
            Deferral(sec(90), 'run-c', 'cpu_some_avg10'),
        )

        figure = deferral_hours_by_metric(project(ticks=ticks, deferrals=deferrals), block(sec(0), sec(1000)))

        assert figure == Measured(
            DeferralHours({'cpu_some_avg10': 75 / 3600, 'io_some_avg10': 75 / 3600}, unspanned=2)
        )

    def test_a_deferral_belongs_to_the_block_of_its_own_timestamp(self):
        ticks = (SchedulerTick(sec(60), 'run-a'), SchedulerTick(sec(135), 'run-a'))
        read = project(ticks=ticks, deferrals=(Deferral(sec(50), 'run-a', 'cpu_some_avg10'),))

        assert deferral_hours_by_metric(read, block(sec(100), sec(1000))) == Measured(DeferralHours({}, 0))

    def test_a_store_with_no_deferral_ever(self):
        read = project(never=(Source.DISPATCH_DEFERRED,))

        assert deferral_hours_by_metric(read, A) == NeverWritten('dispatch_deferred')


def test_lock_conflict_requeues_count_by_reason():
    completions = (
        TaskCompletion(day(1), 'r', '1', 'requeued', 'plan_blast_radius_lock_conflict'),
        TaskCompletion(day(1), 'r', '2', 'requeued', 'plan_blast_radius_lock_conflict'),
        TaskCompletion(day(1), 'r', '3', 'requeued', 'scope_grant_lock_conflict'),
        TaskCompletion(day(1), 'r', '4', 'requeued', 'usage_cap'),
        TaskCompletion(day(1), 'r', '5', 'blocked', 'plan_blast_radius_lock_conflict'),
        TaskCompletion(day(3), 'r', '6', 'requeued', 'scope_grant_lock_conflict'),
    )

    assert lock_conflict_requeues(project(task_completions=completions), A) == Measured(
        {'plan_blast_radius_lock_conflict': 2, 'scope_grant_lock_conflict': 1}
    )


def test_spend_per_day_sums_the_blocks_invocations_over_its_days():
    invocations = (Invocation(day(0.5), 'max-a', 2.5), Invocation(day(1), 'max-b', 1.5), Invocation(day(3), 'max-a', 9.0))

    assert spend_per_day(project(invocations=invocations), A) == Measured(2.0)


class TestCpuPressure:
    def cpu(self, values: dict, reason: str | None = None) -> CpuRead:
        samples = {spec: tuple(CpuSample(sec(i), v) for i, v in enumerate(values.get(spec, ()))) for spec in CPU_ARM_SPECS}
        return CpuRead(Path('load-samples.db'), reason, MappingProxyType(samples), 0)

    def test_mean_and_nearest_rank_p90_per_arm_spec(self):
        first, second = CPU_ARM_SPECS
        read = self.cpu({first: [float(v) for v in range(1, 11)], second: [2.0, 4.0]})

        assert cpu_pressure(read, block(sec(0), sec(100))) == Measured(
            {first: CpuStat(5.5, 9.0, 10), second: CpuStat(3.0, 4.0, 2)}
        )

    def test_an_arm_with_no_sample_in_the_block_has_no_stat(self):
        first, second = CPU_ARM_SPECS
        figure = cpu_pressure(self.cpu({first: [1.0]}), block(sec(0), sec(100)))

        assert figure == Measured({first: CpuStat(1.0, 1.0, 1), second: None})

    def test_a_block_with_no_sample_is_unavailable_not_zero(self):
        first, _ = CPU_ARM_SPECS

        assert isinstance(cpu_pressure(self.cpu({first: [1.0]}), block(sec(50), sec(100))), Unavailable)

    def test_an_unreadable_samples_store_is_unavailable_with_its_reason(self):
        assert cpu_pressure(self.cpu({}, reason='no samples store at x'), A) == Unavailable('no samples store at x')


class TestRules:
    def test_every_figure_has_exactly_one_rule(self):
        assert set(FIGURE_RULES) == set(FigureId)
        assert all(rule.strip() for rule in FIGURE_RULES.values())

    def test_project_figures_cover_every_project_level_figure(self):
        figures = project_figures(project(), HOST, A)

        assert set(figures) | {FigureId.CPU_PRESSURE} == set(FigureId)
