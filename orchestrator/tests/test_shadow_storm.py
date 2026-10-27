"""The shadow coordinator's storm escape (plans/live-shadow-eval-prd.md decision 10, INV-4).

Part 1 covers the pure storm policy. Part 2 drives the coordinator's escape
(row 11) on real stores: a tripped guard pauses durably, files exactly one
escalation, survives a restart, and a resume restarts every window.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from _shadow_coordinator_doubles import (
    INCUMBENTS,
    PROJECT_ID,
    RUN_ID,
    START,
    CoordinatorRig,
    build_coordinator,
    completed,
    make_task,
    shadow_config,
)
from shared.cost_store import CostStore

from orchestrator.event_store import EventType, ShadowSkipReason
from orchestrator.run_store import ShadowCell, ShadowCellQuery
from orchestrator.shadow_eval import (
    PRODUCTION_COST_BASIS,
    TRAILING_WINDOW,
    CostPair,
    CostRatioVerdict,
    HostVerdict,
    OpenInputs,
    RatioSkip,
    RunEndedCell,
    StormFacts,
    StormState,
    StormTrip,
    budget_exhausted,
    cost_ratio_verdict,
    decide_open,
    evaluate_storm,
    failure_streak_tripped,
    storm_window_start,
)
from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
    ShadowStormCause,
)
from orchestrator.workflow_types import WorkflowState

_S = ShadowCellState
_M = CostBasis.METERED
_SUB = CostBasis.SUBSCRIPTION


class TestFailureStreak:
    def test_the_newest_threshold_cells_all_failed_or_expired_trip(self):
        states = [_S.FAILED, _S.EXPIRED, _S.FAILED, _S.FAILED, _S.EXPIRED, _S.SETTLED]
        assert failure_streak_tripped(states, 5) is True

    @pytest.mark.parametrize('success', [_S.AWAITING_REFERENCE, _S.SETTLED])
    def test_a_successful_run_inside_the_window_breaks_the_streak(self, success):
        states = [_S.FAILED, _S.FAILED, success, _S.FAILED, _S.FAILED, _S.FAILED]
        assert failure_streak_tripped(states, 5) is False

    def test_fewer_than_threshold_cells_never_trip(self):
        assert failure_streak_tripped([_S.FAILED] * 4, 5) is False


def _pair(cell: float, cell_basis: CostBasis, production: float) -> CostPair:
    return CostPair(
        cell_cost_usd=cell,
        cell_basis=cell_basis,
        production_cost_usd=production,
        production_basis=PRODUCTION_COST_BASIS,
    )


class TestCostRatioVerdict:
    def test_production_is_token_metered(self):
        assert PRODUCTION_COST_BASIS is CostBasis.METERED

    def test_metered_pairs_give_a_measured_ratio(self):
        verdict = cost_ratio_verdict([_pair(3.0, _M, 1.0), _pair(1.0, _M, 1.0)], ceiling=3.0)

        assert verdict.ratio == 2.0
        assert verdict.skipped is None
        assert verdict.tripped is False

    def test_a_ratio_over_the_ceiling_trips(self):
        verdict = cost_ratio_verdict([_pair(7.0, _M, 2.0)], ceiling=3.0)

        assert verdict.ratio == 3.5
        assert verdict.tripped is True

    def test_a_ratio_at_the_ceiling_does_not_trip(self):
        assert cost_ratio_verdict([_pair(6.0, _M, 2.0)], ceiling=3.0).tripped is False

    def test_a_subscription_arm_is_a_named_skip_never_a_zero_ratio(self):
        verdict = cost_ratio_verdict([_pair(0.0001, _SUB, 1.0)], ceiling=3.0)

        assert verdict.skipped is RatioSkip.COST_BASIS_MISMATCH
        assert verdict.ratio is None
        assert verdict.tripped is False

    def test_mixed_bases_compare_only_the_matching_pairs(self):
        verdict = cost_ratio_verdict(
            [_pair(8.0, _M, 2.0), _pair(0.0001, _SUB, 100.0)], ceiling=3.0,
        )

        assert verdict.ratio == 4.0
        assert verdict.tripped is True

    def test_no_pairs_is_a_named_skip(self):
        verdict = cost_ratio_verdict([], ceiling=3.0)

        assert (verdict.ratio, verdict.skipped) == (None, RatioSkip.NO_PAIRS)

    def test_a_zero_production_sum_is_a_named_skip(self):
        verdict = cost_ratio_verdict([_pair(1.0, _M, 0.0)], ceiling=3.0)

        assert (verdict.ratio, verdict.skipped) == (None, RatioSkip.NO_PRODUCTION_COST)

    @pytest.mark.parametrize(
        'ratio,skipped',
        [(None, None), (1.0, RatioSkip.NO_PAIRS)],
        ids=['neither', 'both'],
    )
    def test_exactly_one_of_ratio_and_skip_is_set(self, ratio, skipped):
        with pytest.raises(ValueError, match='ratio'):
            CostRatioVerdict(ratio=ratio, skipped=skipped, ceiling=3.0)


def _ended(cell_id: str, state: ShadowCellState) -> RunEndedCell:
    return RunEndedCell(cell_id=cell_id, state=state)


_CALM = StormFacts(
    spent_usd=0.5,
    daily_budget_usd=1.0,
    run_ends_newest_first=(_ended('c1', _S.SETTLED),),
    failure_streak_pause=3,
    ratios={'glm-5.3-endpoint': cost_ratio_verdict([_pair(1.0, _SUB, 1.0)], ceiling=3.0)},
)
_OVER_BUDGET = {'spent_usd': 1.2}
_STREAK = {
    'run_ends_newest_first': (
        _ended('c9', _S.FAILED),
        _ended('c8', _S.EXPIRED),
        _ended('c7', _S.FAILED),
    ),
}
_RATIO = {'ratios': {'minimax-m3-endpoint': cost_ratio_verdict([_pair(9.0, _M, 2.0)], ceiling=3.0)}}


class TestEvaluateStorm:
    def test_calm_facts_trip_nothing(self):
        assert evaluate_storm(_CALM) is None

    def test_the_budget_trip_names_spend_and_cap(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_OVER_BUDGET))

        assert isinstance(trip, StormTrip)
        assert trip.cause is ShadowStormCause.DAILY_BUDGET
        assert '1.20' in trip.detail
        assert '1.00' in trip.detail

    def test_the_streak_trip_names_its_cells(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_STREAK))

        assert trip is not None
        assert trip.cause is ShadowStormCause.FAILURE_STREAK
        assert all(cell_id in trip.detail for cell_id in ('c9', 'c8', 'c7'))

    def test_the_ratio_trip_names_candidate_ratio_and_ceiling(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_RATIO))

        assert trip is not None
        assert trip.cause is ShadowStormCause.COST_RATIO
        assert 'minimax-m3-endpoint' in trip.detail
        assert '4.50' in trip.detail
        assert '3.00' in trip.detail

    def test_budget_outranks_streak_outranks_ratio(self):
        everything = dataclasses.replace(_CALM, **_OVER_BUDGET, **_STREAK, **_RATIO)
        streak_and_ratio = dataclasses.replace(_CALM, **_STREAK, **_RATIO)

        causes = [evaluate_storm(facts) for facts in (everything, streak_and_ratio)]

        assert [trip and trip.cause for trip in causes] == [
            ShadowStormCause.DAILY_BUDGET,
            ShadowStormCause.FAILURE_STREAK,
        ]


def _inputs_with_budget(spent: float, cap: float) -> OpenInputs:
    return OpenInputs(
        task_id='5387',
        shape=ShadowShape.IMPLEMENTER,
        candidate='c',
        stratum=('df', 'feature', 'full'),
        base_sha='a' * 40,
        seed=0,
        sample_rate=1.0,
        enabled=True,
        enabled_shapes=frozenset({ShadowShape.IMPLEMENTER}),
        end_to_end_candidates=frozenset(),
        paused=False,
        already_open=False,
        has_credentials=True,
        isolation_available=True,
        credits_exhausted_recently=False,
        spent_usd=spent,
        daily_budget_usd=cap,
        held_cells=0,
        max_concurrent=1,
        host=HostVerdict.CLEAR,
    )


@pytest.mark.parametrize('spent,cap', [(0.0, 0.0), (0.99, 1.0), (1.0, 1.0), (1.01, 1.0), (5.0, 50.0)])
def test_the_open_gate_and_the_storm_share_one_budget_rule(spent, cap):
    expected = budget_exhausted(spent, cap)
    storm = evaluate_storm(dataclasses.replace(_CALM, spent_usd=spent, daily_budget_usd=cap))
    decision = decide_open(_inputs_with_budget(spent, cap))

    assert (storm is not None and storm.cause is ShadowStormCause.DAILY_BUDGET) is expected
    assert (decision.reason is ShadowSkipReason.BUDGET_EXHAUSTED) is expected


_NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


class TestStormWindow:
    def test_the_window_is_the_trailing_day(self):
        assert timedelta(hours=24) == TRAILING_WINDOW
        assert storm_window_start(_NOW, None) == _NOW - TRAILING_WINDOW

    def test_a_recent_resume_restarts_the_window(self):
        resumed = _NOW - timedelta(hours=2)
        assert storm_window_start(_NOW, resumed) == resumed

    def test_an_old_resume_leaves_the_trailing_day(self):
        assert storm_window_start(_NOW, _NOW - timedelta(days=3)) == _NOW - TRAILING_WINDOW


# --------------------------------------------------------------------------- #
# Part 2: the coordinator's storm escape (row 11)                              #
# --------------------------------------------------------------------------- #

_METERED_CANDIDATE = 'claude-sonnet-max'
_SUBSCRIPTION_CANDIDATE = 'glm-5.3-endpoint'
_STREAK_LENGTH = 5
_STREAK_TASKS = tuple(f'streak-{i}' for i in range(_STREAK_LENGTH))
_SEEDED_REASONS = {_S.FAILED: 'runner_raised:RuntimeError', _S.EXPIRED: 'settle_deadline'}


@pytest_asyncio.fixture
async def costs(tmp_path):
    async with CostStore(tmp_path / 'runs.db') as store:
        yield store


def _storm_rig(tmp_path, costs, **config) -> CoordinatorRig:
    return build_coordinator(tmp_path, costs, config=shadow_config(**config))


def _restarted(tmp_path, costs, rig: CoordinatorRig) -> CoordinatorRig:
    """A fresh coordinator on the same runs.db, as after the fleet redeploy."""
    return build_coordinator(
        tmp_path, costs, config=rig.config, clock=rig.clock, run_id='run-after-restart',
    )


def _cells(rig: CoordinatorRig) -> list[ShadowCell]:
    return rig.store.find_shadow_cells(ShadowCellQuery(project_id=PROJECT_ID))


def _skip_reasons(rig: CoordinatorRig) -> list[str]:
    return [row['data']['reason'] for row in rig.events_of(EventType.shadow_cell_skipped)]


def _causes(rig: CoordinatorRig) -> list[ShadowStormCause]:
    return [trip.cause for trip in rig.escalator.trips]


def _fail_every_run(rig: CoordinatorRig) -> None:
    rig.runner.script(
        ShadowShape.IMPLEMENTER, _METERED_CANDIDATE, run_raises=RuntimeError('runner broke'),
    )


async def _open_and_finish(rig: CoordinatorRig, *task_ids: str) -> list[str]:
    opened: list[str] = []
    for task_id in task_ids:
        opened += await rig.coordinator.on_phase(
            make_task(task_id), WorkflowState.PLAN, entering=False,
        )
    await rig.coordinator.join()
    return opened


async def _tripped_streak(tmp_path, costs) -> CoordinatorRig:
    rig = _storm_rig(tmp_path, costs, failure_streak_pause=_STREAK_LENGTH)
    _fail_every_run(rig)
    opened = await _open_and_finish(rig, *_STREAK_TASKS)
    assert len(opened) == _STREAK_LENGTH
    return rig


def _seed(rig: CoordinatorRig, state: ShadowCellState, *, task_id: str) -> ShadowCell:
    """Store a cell already in *state*, opened now; the clock then moves on a minute."""
    now = rig.clock.now()
    cell = ShadowCell(
        cell_id=f'seed-{task_id}',
        project_id=PROJECT_ID,
        task_id=task_id,
        shape=ShadowShape.IMPLEMENTER,
        candidate=_METERED_CANDIDATE,
        incumbent=INCUMBENTS[ShadowShape.IMPLEMENTER],
        stratum=('df', 'feature', 'full'),
        base_sha='a' * 40,
        state=state,
        cost_basis=CostBasis.METERED,
        plan_source=PlanSource.PRODUCTION,
        reason=_SEEDED_REASONS.get(state),
        reference_kind=ReferenceKind.NONE if state is _S.SETTLED else None,
        run_id=RUN_ID,
        opened_at=now,
        settle_deadline=now + timedelta(hours=168),
        settled_at=now if state.stamps_settled_at else None,
    )
    rig.store.insert_shadow_cell(cell)
    rig.clock.advance(timedelta(minutes=1))
    return cell


async def _seed_production_cost(costs: CostStore, task_id: str, cost_usd: float) -> None:
    await costs.save_invocation(
        run_id='run-production', task_id=task_id, project_id=PROJECT_ID,
        account_name='max-a', model='opus', role='implementer',
        cost_usd=cost_usd, input_tokens=None, output_tokens=None,
        cache_read_tokens=None, cache_create_tokens=None, duration_ms=100, capped=False,
        started_at=START.isoformat(), completed_at=START.isoformat(),
    )


class TestFailureStreakEscape:
    @pytest.mark.asyncio
    async def test_a_streak_pauses_once_and_holds_every_later_open(self, tmp_path, costs):
        rig = await _tripped_streak(tmp_path, costs)

        assert {cell.state for cell in _cells(rig)} == {_S.FAILED}
        state = rig.coordinator.storm_state()
        assert (state.paused, state.cause) == (True, ShadowStormCause.FAILURE_STREAK)
        assert _causes(rig) == [ShadowStormCause.FAILURE_STREAK]
        detail = rig.escalator.trips[0].detail
        assert state.detail == detail
        assert all(cell.cell_id in detail for cell in _cells(rig))

        later = await rig.coordinator.on_phase(
            make_task('after-the-storm'), WorkflowState.PLAN, entering=False,
        )

        assert later == []
        assert _skip_reasons(rig) == ['paused']
        assert len(_cells(rig)) == _STREAK_LENGTH

    @pytest.mark.asyncio
    async def test_a_further_failure_while_paused_files_no_second_escalation(self, tmp_path, costs):
        rig = await _tripped_streak(tmp_path, costs)
        _seed(rig, _S.FAILED, task_id='sixth-failure')

        await rig.coordinator.expire(rig.clock.now())

        assert rig.coordinator.storm_state().paused is True
        assert len(rig.escalator.trips) == 1

    @pytest.mark.asyncio
    async def test_the_pause_survives_a_restart_without_a_second_escalation(self, tmp_path, costs):
        rig = await _tripped_streak(tmp_path, costs)
        restarted = _restarted(tmp_path, costs, rig)

        await restarted.coordinator.reconcile_on_start()

        state = restarted.coordinator.storm_state()
        assert (state.paused, state.cause) == (True, ShadowStormCause.FAILURE_STREAK)
        assert restarted.escalator.trips == []
        assert len(rig.escalator.trips) == 1
        assert await restarted.coordinator.on_phase(
            make_task('after-restart'), WorkflowState.PLAN, entering=False,
        ) == []

    @pytest.mark.asyncio
    async def test_resume_lifts_the_pause_and_restarts_the_streak(self, tmp_path, costs):
        rig = await _tripped_streak(tmp_path, costs)
        rig.clock.advance(timedelta(minutes=5))

        assert await rig.coordinator.resume() is True

        assert rig.coordinator.storm_state() == StormState(paused=False, cause=None, detail=None)
        assert rig.store.get_shadow_storm(PROJECT_ID).paused is False

        opened = await _open_and_finish(rig, 'after-resume')
        await rig.coordinator.expire(rig.clock.now())

        assert [cell.state for cell in _cells(rig) if cell.cell_id in opened] == [_S.FAILED]
        assert rig.coordinator.storm_state().paused is False
        assert rig.store.get_shadow_storm(PROJECT_ID).paused is False
        assert len(rig.escalator.trips) == 1

    @pytest.mark.asyncio
    async def test_resume_without_a_pause_changes_nothing(self, tmp_path, costs):
        rig = _storm_rig(tmp_path, costs)

        assert await rig.coordinator.resume() is False
        assert rig.store.get_shadow_storm(PROJECT_ID).resumed_at is None

    @pytest.mark.parametrize(
        'breaker,expected',
        [
            (_S.AWAITING_REFERENCE, []),
            (_S.SETTLED, []),
            (_S.FAILED, [ShadowStormCause.FAILURE_STREAK]),
        ],
        ids=['awaiting', 'settled', 'control-failed'],
    )
    @pytest.mark.asyncio
    async def test_a_completed_run_breaks_the_streak(self, tmp_path, costs, breaker, expected):
        rig = _storm_rig(tmp_path, costs, failure_streak_pause=_STREAK_LENGTH)
        _seed(rig, breaker, task_id='breaker')
        for i in range(_STREAK_LENGTH - 1):
            _seed(rig, _S.FAILED, task_id=f'failed-{i}')

        await rig.coordinator.expire(rig.clock.now())

        assert _causes(rig) == expected
        assert rig.coordinator.storm_state().paused is bool(expected)


class TestCostGuardsEscape:
    @pytest.mark.parametrize(
        'candidate,basis,expected',
        [
            (_METERED_CANDIDATE, CostBasis.METERED, [ShadowStormCause.DAILY_BUDGET]),
            (_SUBSCRIPTION_CANDIDATE, CostBasis.SUBSCRIPTION, []),
        ],
        ids=['metered', 'subscription'],
    )
    @pytest.mark.asyncio
    async def test_the_daily_budget_counts_only_metered_dollars(
        self, tmp_path, costs, candidate, basis, expected,
    ):
        rig = _storm_rig(tmp_path, costs, candidates=[candidate], daily_budget_usd=1.0)
        rig.runner.default = completed(cost_usd=0.6)

        opened = await _open_and_finish(rig, 'budget-a', 'budget-b')

        assert len(opened) == 2
        assert [(cell.cost_usd, cell.cost_basis) for cell in _cells(rig)] == [(0.6, basis)] * 2
        assert _causes(rig) == expected
        assert rig.coordinator.storm_state().paused is bool(expected)

    @pytest.mark.parametrize(
        'candidate,expected',
        [
            (_METERED_CANDIDATE, [ShadowStormCause.COST_RATIO]),
            (_SUBSCRIPTION_CANDIDATE, []),
        ],
        ids=['metered', 'subscription'],
    )
    @pytest.mark.asyncio
    async def test_the_cost_ratio_trips_only_on_a_production_basis_arm(
        self, tmp_path, costs, candidate, expected,
    ):
        rig = _storm_rig(tmp_path, costs, candidates=[candidate], cost_ratio_ceiling=3.0)
        rig.runner.default = completed(cost_usd=4.0)
        tasks = ('ratio-a', 'ratio-b')
        for task_id in tasks:
            await _seed_production_cost(costs, task_id, 1.0)
        await _open_and_finish(rig, *tasks)

        for task_id in tasks:
            await rig.coordinator.on_merge_finalized(
                {'task_id': task_id, 'state': 'done', 'merge_sha': 'f' * 40},
            )

        assert {cell.state for cell in _cells(rig)} == {_S.SETTLED}
        assert _causes(rig) == expected
        assert rig.coordinator.storm_state().paused is bool(expected)


@pytest.mark.asyncio
async def test_a_failing_escalation_leaves_the_pause_standing(tmp_path, costs, caplog):
    rig = _storm_rig(tmp_path, costs, failure_streak_pause=_STREAK_LENGTH)
    rig.escalator.raises = RuntimeError('escalation server down')
    _fail_every_run(rig)

    with caplog.at_level(logging.ERROR, logger='orchestrator.shadow_eval'):
        await _open_and_finish(rig, *_STREAK_TASKS)

    assert len(rig.escalator.trips) == 1
    assert rig.coordinator.storm_state().paused is True
    assert rig.store.get_shadow_storm(PROJECT_ID).cause is ShadowStormCause.FAILURE_STREAK
    assert any(
        record.levelno == logging.ERROR
        and record.exc_info is not None
        and record.exc_info[1] is rig.escalator.raises
        for record in caplog.records
    )

    restarted = _restarted(tmp_path, costs, rig)
    await restarted.coordinator.reconcile_on_start()

    assert restarted.escalator.trips == []
    assert restarted.coordinator.storm_state().paused is True
