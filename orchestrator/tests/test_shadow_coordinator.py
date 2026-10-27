"""The shadow coordinator's open path and run ends (plans/live-shadow-eval-prd.md C2).

Real RunStore/EventStore/CostStore on one tmp runs.db; only the runner, clock,
escalation sink, host sample and isolation probe are faked.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from _orch_helpers import wait_responsive
from _shadow_coordinator_doubles import (
    INCUMBENTS,
    PROJECT_ID,
    RUN_ID,
    CoordinatorRig,
    FakeClock,
    FakeRunner,
    RecordingEscalator,
    blank_token_endpoint_config,
    build_coordinator,
    calm_sample,
    completed,
    make_task,
    metered_native_config,
    shadow_config,
)
from shared.cost_store import CostStore

from orchestrator.config import PsiAdmissionConfig
from orchestrator.evals.live_fixture import live_stratum
from orchestrator.event_store import EventStore, EventType
from orchestrator.run_store import RunStore, ShadowCell, ShadowCellQuery
from orchestrator.shadow_eval import (
    CREDIT_WINDOW,
    CellRunOutcome,
    HostVerdict,
    RunEnd,
    ShadowCoordinator,
    ShadowCoordinatorDeps,
    cost_basis_of,
)
from orchestrator.shadow_scoring import ShadowReference, score_architect_cell
from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
)
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

_S = ShadowCellState


@pytest_asyncio.fixture
async def costs(tmp_path):
    async with CostStore(tmp_path / 'runs.db') as store:
        yield store


def _cells(rig: CoordinatorRig) -> list[ShadowCell]:
    return rig.store.find_shadow_cells(ShadowCellQuery(project_id=PROJECT_ID))


def _only_cell(rig: CoordinatorRig) -> ShadowCell:
    cells = _cells(rig)
    assert len(cells) == 1, cells
    return cells[0]


def _skip_reasons(rig: CoordinatorRig) -> list[str]:
    return [row['data']['reason'] for row in rig.events_of(EventType.shadow_cell_skipped)]


async def _plan_exit(rig: CoordinatorRig, task: dict) -> list[str]:
    return await rig.coordinator.on_phase(task, WorkflowState.PLAN, entering=False)


@pytest.mark.asyncio
async def test_a_sampled_out_trigger_opens_nothing_and_says_why(tmp_path, costs):
    config = shadow_config(
        sample_rate=0.0, candidates=['claude-sonnet-max', 'glm-5.3-endpoint'],
    )
    rig = build_coordinator(tmp_path, costs, config=config)

    opened = await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    assert opened == []
    assert _cells(rig) == []
    skipped = rig.events_of(EventType.shadow_cell_skipped)
    assert sorted((row['data']['shape'], row['data']['candidate']) for row in skipped) == [
        ('implementer', 'claude-sonnet-max'),
        ('implementer', 'glm-5.3-endpoint'),
    ]
    assert {row['data']['reason'] for row in skipped} == {'sampled_out'}
    assert {row['task_id'] for row in skipped} == {'5387'}
    assert rig.runner.prepared == []
    assert rig.runner.ran == []


@pytest.mark.asyncio
async def test_a_disabled_coordinator_opens_nothing_and_emits_nothing(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(enabled=False))

    assert await _plan_exit(rig, make_task('5387')) == []
    assert _cells(rig) == []
    assert rig.events_of(EventType.shadow_cell_skipped) == []


@pytest.mark.asyncio
async def test_the_happy_path_drives_a_cell_to_awaiting_reference(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    rig.runner.default = completed(
        result_path=tmp_path / 'results' / 'cell.json', cost_usd=0.4, cost_credits=0.0,
    )
    task = make_task('5387')

    opened = await _plan_exit(rig, task)
    await rig.coordinator.join()

    cell = _only_cell(rig)
    assert opened == [cell.cell_id]
    assert cell.state is _S.AWAITING_REFERENCE
    assert cell.worktree_path == rig.runner.worktree_for(rig.runner.prepared[0])
    assert cell.running_at is not None
    assert cell.settled_at is None
    assert (cell.result_path, cell.cost_usd, cell.cost_credits) == (
        tmp_path / 'results' / 'cell.json',
        0.4,
        0.0,
    )
    assert cell.run_id == RUN_ID
    assert cell.owner == 'shadow_coordinator'
    assert cell.settle_deadline == cell.opened_at + timedelta(hours=rig.config.settle_deadline_hours)
    assert cell.stratum == live_stratum(task, project=PROJECT_ID)
    assert cell.incumbent == INCUMBENTS[ShadowShape.IMPLEMENTER]
    assert cell.plan_source is PlanSource.PRODUCTION
    assert cell.cost_basis is cost_basis_of(metered_native_config())
    assert cell.base_sha == 'a' * 40

    opened_events = rig.events_of(EventType.shadow_cell_opened)
    assert len(opened_events) == 1
    assert opened_events[0]['task_id'] == '5387'
    assert opened_events[0]['data']['cell_id'] == cell.cell_id
    assert opened_events[0]['data']['host'] == HostVerdict.CLEAR.value

    [request] = rig.runner.ran
    assert request.task == task
    assert request.candidate.name == 'claude-sonnet-max'
    assert request.cell.cell_id == cell.cell_id


@pytest.mark.asyncio
async def test_an_unreadable_host_opens_and_the_event_records_it(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    rig.host[0] = dataclasses.replace(rig.host[0], read_ok=False, mem_some10=99.0, io_some10=99.0)

    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    [event] = rig.events_of(EventType.shadow_cell_opened)
    assert event['data']['host'] == HostVerdict.UNREADABLE.value


@pytest.mark.asyncio
async def test_a_second_identical_trigger_skips_already_open(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    task = make_task('5387')

    await _plan_exit(rig, task)
    await rig.coordinator.join()
    second = await _plan_exit(rig, task)
    await rig.coordinator.join()

    assert second == []
    assert len(_cells(rig)) == 1
    assert _skip_reasons(rig) == ['already_open']
    assert len(rig.runner.ran) == 1


@pytest.mark.asyncio
async def test_a_subscription_candidate_records_its_basis_and_credits(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(candidates=['glm-5.3-endpoint']))
    rig.runner.default = completed(cost_usd=0.0001, cost_credits=812.5)

    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    cell = _only_cell(rig)
    assert cell.cost_basis is CostBasis.SUBSCRIPTION
    assert cell.cost_credits == 812.5
    assert cell.state is _S.AWAITING_REFERENCE


@pytest.mark.asyncio
async def test_an_unresolvable_candidate_is_a_named_skip(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(candidates=['no-such-candidate']))

    assert await _plan_exit(rig, make_task('5387')) == []

    assert _cells(rig) == []
    assert _skip_reasons(rig) == ['unknown_candidate']


@pytest.mark.asyncio
async def test_a_task_without_a_base_sha_is_a_named_skip(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)

    assert await _plan_exit(rig, make_task('5387', base_sha=None)) == []

    assert _cells(rig) == []
    assert _skip_reasons(rig) == ['no_base_sha']


@pytest.mark.asyncio
async def test_a_blank_credential_is_a_named_skip(tmp_path, costs):
    blank = blank_token_endpoint_config()
    rig = build_coordinator(
        tmp_path, costs,
        config=shadow_config(candidates=[blank.name]),
        candidates={blank.name: blank},
    )

    assert await _plan_exit(rig, make_task('5387')) == []
    assert _skip_reasons(rig) == ['no_credentials']


def _ended_events(rig: CoordinatorRig) -> list[dict]:
    return rig.events_of(EventType.shadow_cell_ended)


@pytest.mark.asyncio
async def test_a_raising_run_fails_the_cell_with_the_exception_class(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    rig.runner.script(ShadowShape.IMPLEMENTER, 'claude-sonnet-max', run_raises=RuntimeError('boom'))

    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    cell = _only_cell(rig)
    assert cell.state is _S.FAILED
    assert cell.reason == 'runner_raised:RuntimeError'
    assert cell.settled_at is None
    [ended] = _ended_events(rig)
    assert ended['data']['cell_id'] == cell.cell_id
    assert (ended['data']['state'], ended['data']['reason']) == ('failed', 'runner_raised:RuntimeError')
    assert len(rig.runner.ran) == 1


@pytest.mark.asyncio
async def test_a_raising_prepare_fails_the_cell_from_sampled(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    rig.runner.script(ShadowShape.IMPLEMENTER, 'claude-sonnet-max', prepare_raises=OSError('disk'))

    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    cell = _only_cell(rig)
    assert cell.state is _S.FAILED
    assert cell.reason == 'runner_raised:OSError'
    assert cell.worktree_path is None
    assert rig.runner.ran == []
    assert len(_ended_events(rig)) == 1


@pytest.mark.parametrize(
    'end,reason',
    [(RunEnd.USAGE_CAP, 'usage_cap'), (RunEnd.CREDITS_EXHAUSTED, 'credits_exhausted')],
)
@pytest.mark.asyncio
async def test_a_capped_run_is_cap_excluded_without_retry(tmp_path, costs, end, reason):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(candidates=['glm-5.3-endpoint']))
    rig.runner.script(
        ShadowShape.IMPLEMENTER, 'glm-5.3-endpoint',
        outcome=CellRunOutcome(end=end, cost_usd=0.01, cost_credits=40.0),
    )

    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()

    cell = _only_cell(rig)
    assert cell.state is _S.CAP_EXCLUDED
    assert cell.reason == reason
    assert cell.cost_credits == 40.0
    assert len(rig.runner.ran) == 1
    [ended] = _ended_events(rig)
    assert (ended['data']['state'], ended['data']['reason']) == ('cap_excluded', reason)


@pytest.mark.asyncio
async def test_credits_exhaustion_holds_the_candidate_for_the_credit_window(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(candidates=['glm-5.3-endpoint']))
    rig.runner.script(
        ShadowShape.IMPLEMENTER, 'glm-5.3-endpoint',
        outcome=CellRunOutcome(end=RunEnd.CREDITS_EXHAUSTED),
    )
    await _plan_exit(rig, make_task('5387'))
    await rig.coordinator.join()
    rig.runner.scripts.clear()

    rig.clock.advance(CREDIT_WINDOW - timedelta(minutes=1))
    assert await _plan_exit(rig, make_task('5388')) == []
    assert _skip_reasons(rig) == ['credits_exhausted']

    rig.clock.advance(timedelta(minutes=2))
    opened = await _plan_exit(rig, make_task('5389'))
    await rig.coordinator.join()
    assert len(opened) == 1


@pytest.mark.asyncio
async def test_a_raising_resolver_is_an_unknown_candidate_skip_naming_the_error(
    tmp_path, costs, caplog,
):
    def exploding_resolver(name: str):
        raise RuntimeError(f'cannot resolve {name}')

    rig = build_coordinator(tmp_path, costs, resolve_candidate=exploding_resolver)

    with caplog.at_level(logging.ERROR, logger='orchestrator.shadow_eval'):
        assert await _plan_exit(rig, make_task('5387')) == []

    assert _cells(rig) == []
    [skip] = rig.events_of(EventType.shadow_cell_skipped)
    assert skip['data']['reason'] == 'unknown_candidate'
    assert skip['data']['resolver_error'] == 'RuntimeError'
    assert any(
        record.levelno == logging.ERROR and 'claude-sonnet-max' in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_clean_resolver_miss_names_no_resolver_error(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs, config=shadow_config(candidates=['no-such-candidate']))

    await _plan_exit(rig, make_task('5387'))

    [skip] = rig.events_of(EventType.shadow_cell_skipped)
    assert 'resolver_error' not in skip['data']


@pytest.mark.asyncio
async def test_on_phase_never_raises_on_a_malformed_task(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)

    assert await _plan_exit(rig, {'title': 'no id at all'}) == []
    assert _cells(rig) == []


@pytest.mark.asyncio
async def test_shutdown_fails_a_running_cell_before_releasing_it(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    rig.runner.script(ShadowShape.IMPLEMENTER, 'claude-sonnet-max', block=asyncio.Event())

    await _plan_exit(rig, make_task('5387'))
    await wait_responsive(rig.runner.run_started.wait(), label='the shadow implementer run starts')
    assert _only_cell(rig).state is _S.RUNNING

    await asyncio.wait_for(rig.coordinator.aclose(), timeout=5)

    cell = _only_cell(rig)
    assert cell.state is _S.FAILED
    assert cell.reason == 'cancelled_at_shutdown'
    assert len(_ended_events(rig)) == 1


class _OverlappingSnapshots:
    """An isolation probe whose first call waits up to a second for a concurrent second call.

    The probe runs inside the open snapshot, so two openers that are not
    serialised both take their snapshot before either inserts its cell.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = 0
        self._second_arrived = threading.Event()

    def __call__(self) -> bool:
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first:
            self._second_arrived.wait(timeout=1.0)
        else:
            self._second_arrived.set()
        return True


@pytest.mark.asyncio
async def test_concurrent_triggers_never_open_past_max_concurrent(tmp_path, costs):
    rig = build_coordinator(
        tmp_path, costs,
        config=shadow_config(max_concurrent=1),
        isolation_probe=_OverlappingSnapshots(),
    )
    release_runs = asyncio.Event()
    rig.runner.script(ShadowShape.IMPLEMENTER, 'claude-sonnet-max', block=release_runs)

    opened = await asyncio.gather(
        _plan_exit(rig, make_task('5387')), _plan_exit(rig, make_task('5388')),
    )
    release_runs.set()
    await rig.coordinator.join()

    assert sorted(len(ids) for ids in opened) == [0, 1]
    assert len(_cells(rig)) == 1
    assert _skip_reasons(rig) == ['concurrency_full']


@pytest.mark.asyncio
async def test_should_open_is_repeatable_and_writes_nothing(tmp_path, costs):
    rig = build_coordinator(tmp_path, costs)
    task = make_task('5387')
    candidate = metered_native_config()

    first = await rig.coordinator.should_open(task, ShadowShape.IMPLEMENTER, candidate)
    second = await rig.coordinator.should_open(task, ShadowShape.IMPLEMENTER, candidate)

    assert first == second
    assert first.open is True
    assert _cells(rig) == []
    assert rig.events_of(EventType.shadow_cell_skipped) == []
    assert rig.events_of(EventType.shadow_cell_opened) == []


def test_the_constructor_refuses_a_missing_incumbent(tmp_path):
    incumbents = {s: n for s, n in INCUMBENTS.items() if s is not ShadowShape.END_TO_END}
    with pytest.raises(ValueError, match='end-to-end'):
        ShadowCoordinator(
            shadow_config(), _deps_only(tmp_path), project_id=PROJECT_ID, run_id=RUN_ID,
            incumbents=incumbents,
        )


def test_the_constructor_refuses_an_unknown_project(tmp_path):
    with pytest.raises(ValueError, match='no-such-project'):
        ShadowCoordinator(
            shadow_config(), _deps_only(tmp_path), project_id='no-such-project',
            run_id=RUN_ID, incumbents=INCUMBENTS,
        )


def _deps_only(tmp_path: Path) -> ShadowCoordinatorDeps:
    db_path = tmp_path / 'runs.db'
    return ShadowCoordinatorDeps(
        store=RunStore(db_path),
        events=EventStore(db_path, RUN_ID),
        costs=CostStore(db_path),
        runner=FakeRunner(tmp_path),
        escalate=RecordingEscalator(),
        clock=FakeClock().now,
        isolation=lambda: True,
        read_host_sample=calm_sample,
        psi_config=PsiAdmissionConfig(),
    )


_CANDIDATE_PLAN = {
    'steps': [{'id': 'step-1', 'type': 'test'}, {'id': 'step-2', 'type': 'impl'}],
    'files': ['orchestrator/src/orchestrator/shadow_eval.py'],
}


def _consequence_rig(tmp_path, costs, *, shapes=(ShadowShape.ARCHITECT_CONSEQUENCE,)) -> CoordinatorRig:
    return build_coordinator(tmp_path, costs, config=shadow_config(shapes=list(shapes)))


def _script_leg_one(rig: CoordinatorRig, **metrics) -> None:
    rig.runner.script(
        ShadowShape.ARCHITECT, 'claude-sonnet-max',
        outcome=completed(plan=metrics.pop('plan', None), **metrics),
    )


async def _plan_entry(rig: CoordinatorRig, task: dict) -> list[str]:
    opened = await rig.coordinator.on_phase(task, WorkflowState.PLAN, entering=True)
    await rig.coordinator.join()
    return opened


def _of_shape(rig: CoordinatorRig, shape: ShadowShape) -> list[ShadowCell]:
    return [cell for cell in _cells(rig) if cell.shape is shape]


@pytest.mark.asyncio
async def test_plan_entry_opens_the_architect_leg_one_cell(tmp_path, costs):
    rig = _consequence_rig(tmp_path, costs)

    opened = await _plan_entry(rig, make_task('5387'))

    [leg_one] = _cells(rig)
    assert opened == [leg_one.cell_id]
    assert leg_one.shape is ShadowShape.ARCHITECT
    assert leg_one.plan_source is None
    assert leg_one.incumbent == INCUMBENTS[ShadowShape.ARCHITECT]


@pytest.mark.asyncio
async def test_a_declined_leg_one_opens_no_leg_two_and_records_the_decline(tmp_path, costs):
    rig = _consequence_rig(tmp_path, costs)
    _script_leg_one(rig, terminal_kind='already_done', plan_steps=0)

    [leg_one_id] = await _plan_entry(rig, make_task('5387'))

    assert _of_shape(rig, ShadowShape.ARCHITECT_CONSEQUENCE) == []
    assert [cell for cell in _cells(rig) if cell.parent_cell_id == leg_one_id] == []
    leg_one = rig.store.get_shadow_cell(leg_one_id)
    assert leg_one is not None
    assert leg_one.state is _S.AWAITING_REFERENCE

    await rig.coordinator.on_production_terminal('5387', WorkflowOutcome.CANCELLED)

    leg_one = rig.store.get_shadow_cell(leg_one_id)
    assert leg_one is not None
    assert (leg_one.state, leg_one.reference_kind) == (_S.SETTLED, ReferenceKind.NONE)
    [request] = rig.runner.ran
    outcome = rig.runner.scripts[(ShadowShape.ARCHITECT, 'claude-sonnet-max')].outcome
    assert outcome is not None and outcome.metrics is not None
    score = score_architect_cell(outcome.metrics, ShadowReference.none(), None)
    assert score.terminal_kind == 'already_done'
    assert request.cell.cell_id == leg_one_id


@pytest.mark.asyncio
async def test_a_plan_then_decline_is_a_decline(tmp_path, costs):
    rig = _consequence_rig(tmp_path, costs)
    _script_leg_one(rig, terminal_kind='already_done', plan_steps=6, plan=_CANDIDATE_PLAN)

    await _plan_entry(rig, make_task('5387'))

    assert _of_shape(rig, ShadowShape.ARCHITECT_CONSEQUENCE) == []


@pytest.mark.asyncio
async def test_a_planned_leg_one_opens_leg_two_on_the_candidate_plan(tmp_path, costs):
    rig = _consequence_rig(tmp_path, costs)
    _script_leg_one(rig, terminal_kind='planned', plan_steps=3, plan=_CANDIDATE_PLAN)

    [leg_one_id] = await _plan_entry(rig, make_task('5387'))

    [leg_two] = _of_shape(rig, ShadowShape.ARCHITECT_CONSEQUENCE)
    leg_one = rig.store.get_shadow_cell(leg_one_id)
    assert leg_one is not None
    assert leg_two.parent_cell_id == leg_one_id
    assert leg_two.plan_source is PlanSource.CANDIDATE
    assert leg_two.base_sha == leg_one.base_sha
    assert leg_two.incumbent == INCUMBENTS[ShadowShape.ARCHITECT_CONSEQUENCE]
    assert leg_two.state is _S.AWAITING_REFERENCE
    [leg_two_request] = [
        request for request in rig.runner.ran
        if request.cell.shape is ShadowShape.ARCHITECT_CONSEQUENCE
    ]
    assert leg_two_request.plan == _CANDIDATE_PLAN


@pytest.mark.asyncio
async def test_without_the_consequence_shape_a_planned_leg_opens_no_leg_two(tmp_path, costs):
    rig = _consequence_rig(tmp_path, costs, shapes=(ShadowShape.ARCHITECT,))
    _script_leg_one(rig, terminal_kind='planned', plan_steps=3, plan=_CANDIDATE_PLAN)

    await _plan_entry(rig, make_task('5387'))

    assert [cell.shape for cell in _cells(rig)] == [ShadowShape.ARCHITECT]
