"""A remote verify decides against a view of the dispatcher's registry (task 9123,
plans/verify-result-cache-prd.md Amendment A5): the shipped view's wire value, how
the dispatcher builds it, how the runner decides against it, and how the
dispatcher settles what the runner reports."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _gate_fixtures import gate_argv, gated_remote_runner
from _git_fixtures import seed_repo
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _verify_cache_fixtures import (
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    clean_unscoped_gate,
    clear_markers,
    commit_edit,
    entries_for,
    head_of,
    merge_pool,
    module_legs,
    module_units,
    outcomes_of,
    ran_legs,
    record_run,
    row_units,
    task_verify,
    units_by_id,
    use_real_legs,
    workspace_git,
)
from _verify_cache_units import (
    clean_leg,
    ran_unit,
    record_entry,
    unit_decision,
    unit_key,
    verify_result,
)

from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_run import run_task_verify_on_worktree
from orchestrator.task_verify_spec import (
    build_task_verify_spec,
    spec_from_dict,
    spec_to_dict,
    task_spec_from_json,
    task_spec_to_json,
)
from orchestrator.verify_cache import keys
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import CacheReport, LegRun, UnbackedReplay, report_of
from orchestrator.verify_cache.registry import Candidate, Entry, KillSwitch, PolicyState, Registry
from orchestrator.verify_cache.remote_view import (
    RemoteCache,
    ShippedView,
    planned_unit_ids,
    ship_view,
)
from orchestrator.verify_cache.replay import CacheAccess, ExecutedModule, ReplayPlan
from orchestrator.verify_evidence import Leg, UnitOutcome, UnitTier
from orchestrator.verify_runner import (
    MergeVerifySpec,
    RemoteRunner,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyRunnerPool,
    build_merge_verify_spec,
    result_to_json,
    run_merge_verify_on_worktree,
    spec_from_json,
    spec_to_json,
)
from orchestrator.verify_spec_shared import VerifyCommand

_NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
_ALPHA, _BETA = 'module:alpha:test', 'module:beta:test'
_BOTH_ON = VerifyCacheConfig(
    task_mode=CacheMode.ON, merge_mode=CacheMode.ON,
    remote_task_mode=CacheMode.ON, remote_merge_mode=CacheMode.ON,
)


def _candidate(unit_id: str = _ALPHA, entry_id: int = 1, evidence: int = 7) -> Candidate:
    return Candidate(unit_id, unit_key(unit_id, token=str(entry_id)).key, entry_id, evidence)


def _view(**overrides) -> ShippedView:
    fields = {
        'mode': CacheMode.ON, 'disabled_until': None, 'disabled_reason': None,
        'candidates': (_candidate(),), 'truncated': False, **overrides,
    }
    return ShippedView(**fields)


def _ticking_registry(path: Path) -> Registry:
    ticks = iter(range(100_000))
    return Registry(path, clock=lambda: _NOW + timedelta(seconds=next(ticks)))


async def _record(registry: Registry, unit_id: str, count: int) -> list[Candidate]:
    """*count* greens of *unit_id*, most recent first."""
    recorded = [
        await record_entry(registry, unit_key(unit_id, token=f'{unit_id}-{n}'), evidence=n) for n in range(count)
    ]
    return [entry.candidate for entry in reversed(recorded)]


def _wire_size(view: ShippedView) -> int:
    return len(json.dumps(view.to_wire()))


class TestShippedViewWire:
    @pytest.mark.parametrize('view', [
        _view(
            candidates=(_candidate(), _candidate(_BETA, entry_id=2, evidence=9)),
            disabled_until=_NOW + timedelta(days=3), disabled_reason='two leaks', truncated=True,
        ),
        _view(mode=CacheMode.OFF, candidates=()),
    ], ids=['full', 'mode-only'])
    def test_a_view_round_trips_through_json(self, view):
        wire = json.loads(json.dumps(view.to_wire()))

        assert ShippedView.from_wire(wire) == view

    @pytest.mark.parametrize(('mutate', 'named'), [
        (lambda wire: [wire], 'ShippedView'),
        (lambda wire: {**wire, 'mode': 'maybe'}, 'mode'),
        (lambda wire: {**wire, 'candidates': [{**wire['candidates'][0], 'entry_id': '1'}]}, 'entry_id'),
        (lambda wire: {**wire, 'disabled_until': '2026-11-24T12:00:00', 'disabled_reason': 'x'}, 'disabled_until'),
        (lambda wire: {key: value for key, value in wire.items() if key != 'truncated'}, 'truncated'),
    ], ids=['not-a-dict', 'unknown-mode', 'string-entry-id', 'naive-window', 'missing-key'])
    def test_a_malformed_view_is_rejected_naming_the_field(self, mutate, named):
        with pytest.raises((TypeError, ValueError), match=named):
            ShippedView.from_wire(mutate(_view().to_wire()))

    def test_a_disable_without_a_reason_is_rejected(self):
        with pytest.raises(ValueError, match='disabled_until'):
            _view(disabled_until=_NOW, disabled_reason=None)


class TestRegistryView:
    def test_the_projection_keys_candidates_by_unit_and_key(self):
        alpha, beta = _candidate(), _candidate(_BETA, entry_id=2)
        switch = KillSwitch(_NOW, 'two leaks')
        view = _view(candidates=(alpha, beta), disabled_until=switch.disabled_until, disabled_reason=switch.reason, truncated=True)

        projected = view.registry_view()

        assert projected.entries == {(alpha.unit_id, alpha.key): alpha, (beta.unit_id, beta.key): beta}
        assert projected.policy == switch
        assert projected.truncated is True


class TestShipView:
    @pytest.mark.asyncio
    async def test_it_ships_the_effective_mode_the_recent_greens_and_the_kill_switch(self, tmp_path):
        store = _ticking_registry(tmp_path / 'verify_cache.db')
        await _record(store, _ALPHA, 3)
        await _record(store, _BETA, 1)
        until = _NOW + timedelta(days=2)
        await store.write_policy_state(PolicyState(disabled_until=until, reason='two leaks', streak=2, leaks=('l1', 'l2')))
        modes = _BOTH_ON.model_copy(update={'remote_merge_mode': CacheMode.LOG, 'remote_candidates_per_unit': 3})

        view = await ship_view(store, modes, 'merge', [_ALPHA, _BETA], fits=lambda v: True)

        assert view is not None
        assert view.mode is modes.remote_mode_for('merge') is CacheMode.LOG
        assert view.candidates == await store.candidates([_ALPHA, _BETA], per_unit=3)
        assert view.kill_switch == KillSwitch(until, 'two leaks')
        assert view.truncated is False

    @pytest.mark.asyncio
    async def test_a_unit_over_the_cap_ships_its_cap_and_marks_the_view_truncated(self, tmp_path):
        store = _ticking_registry(tmp_path / 'verify_cache.db')
        alpha = await _record(store, _ALPHA, 3)
        beta = await _record(store, _BETA, 1)
        modes = _BOTH_ON.model_copy(update={'remote_candidates_per_unit': 2})

        view = await ship_view(store, modes, 'merge', [_ALPHA, _BETA], fits=lambda v: True)

        assert view is not None
        assert view.candidates == (*alpha[:2], *beta)
        assert view.truncated is True

    @pytest.mark.asyncio
    async def test_an_effective_off_ships_the_mode_alone_and_reads_nothing(self, tmp_path):
        db_path = tmp_path / 'verify_cache.db'
        db_path.write_bytes(b'not a sqlite database, just bytes' * 64)
        modes = _BOTH_ON.model_copy(update={'remote_merge_mode': CacheMode.OFF})

        view = await ship_view(Registry(db_path), modes, 'merge', [_ALPHA], fits=lambda v: True)

        assert view == ShippedView(CacheMode.OFF, None, None, (), False)

    @pytest.mark.asyncio
    async def test_a_size_bound_keeps_the_most_recent_greens_of_every_unit(self, tmp_path):
        store = _ticking_registry(tmp_path / 'verify_cache.db')
        alpha = await _record(store, _ALPHA, 6)
        beta = await _record(store, _BETA, 6)
        unbounded = await ship_view(store, _BOTH_ON, 'merge', [_ALPHA, _BETA], fits=lambda v: True)
        assert unbounded is not None
        bound = _wire_size(unbounded) // 2

        view = await ship_view(store, _BOTH_ON, 'merge', [_ALPHA, _BETA], fits=lambda v: _wire_size(v) <= bound)

        assert view is not None
        assert _wire_size(view) <= bound
        depth = len([c for c in view.candidates if c.unit_id == _ALPHA])
        assert 0 < depth < 6
        assert view.candidates == (*alpha[:depth], *beta[:depth])
        assert view.truncated is True

    @pytest.mark.asyncio
    async def test_a_bound_nothing_meets_ships_no_candidates_and_never_raises(self, tmp_path):
        store = _ticking_registry(tmp_path / 'verify_cache.db')
        await _record(store, _ALPHA, 2)

        view = await ship_view(store, _BOTH_ON, 'merge', [_ALPHA], fits=lambda v: False)

        assert view is not None
        assert (view.candidates, view.truncated) == ((), True)

    @pytest.mark.asyncio
    async def test_an_unreadable_registry_ships_no_view_and_says_why(self, tmp_path, caplog):
        db_path = tmp_path / 'verify_cache.db'
        db_path.write_bytes(b'not a sqlite database, just bytes' * 64)

        with caplog.at_level(logging.WARNING, logger='orchestrator.verify_cache.remote_view'):
            view = await ship_view(Registry(db_path), _BOTH_ON, 'merge', [_ALPHA], fits=lambda v: True)

        assert view is None
        assert any(str(db_path) in record.getMessage() for record in caplog.records)


class TestPlannedUnitIds:
    def test_every_set_leg_of_every_command_and_the_tree(self):
        commands = [
            VerifyCommand('alpha', test_command='pytest', lint_command='ruff', type_check_command=None),
            VerifyCommand('beta', test_command='pytest', lint_command=None, type_check_command='pyright'),
        ]

        assert planned_unit_ids(commands) == (
            'module:alpha:test', 'module:alpha:lint', 'module:beta:test', 'module:beta:type', keys.TREE_UNIT_ID,
        )

    def test_it_names_units_as_the_replay_plan_does(self):
        config = OrchestratorConfig(test_command='pytest', lint_command='ruff', type_check_command='')
        alpha = ModuleConfig(prefix='alpha', test_command='pytest', lint_command='ruff')
        global_command = VerifyCommand('', test_command='pytest', lint_command='ruff')

        assert set(planned_unit_ids([VerifyCommand.from_module_config(alpha), global_command])) == {
            *ExecutedModule.of(alpha, config).unit_ids(), *ExecutedModule.of(None, config).unit_ids(), keys.TREE_UNIT_ID,
        }


_ALPHA_MODULE = ModuleConfig(prefix='alpha', test_command='pytest alpha', lint_command='ruff check alpha')
_SHIPPED = ShippedView(
    CacheMode.ON, _NOW + timedelta(days=1), 'two leaks', (_candidate(), _candidate(_BETA, entry_id=2)), True,
)


def _merge_spec(view: ShippedView | None = None) -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(VerifyCommand.from_module_config(_ALPHA_MODULE),),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=('alpha/a.py',),
        verify_env={},
        cold_timeout_secs=60.0,
        verify_cache=view,
    )


def _task_spec(config: OrchestratorConfig):
    return build_task_verify_spec(
        config, [_ALPHA_MODULE], ['alpha/a.py'], force_workspace=False, attempt_id=1, task_id='9123', tip_sha='a' * 40,
    )


class TestMergeSpecCarriesTheView:
    @pytest.mark.parametrize('view', [_SHIPPED, None], ids=['view', 'none'])
    def test_the_view_round_trips(self, view):
        assert spec_from_json(spec_to_json(_merge_spec(view))).verify_cache == view

    def test_a_legacy_spec_without_the_block_decodes_to_no_view(self):
        legacy = _merge_spec(_SHIPPED).to_dict()
        del legacy['verify_cache']

        assert MergeVerifySpec.from_dict(legacy).verify_cache is None

    def test_the_builder_ships_no_view(self):
        config = OrchestratorConfig()

        assert build_merge_verify_spec(config, [_ALPHA_MODULE], None).verify_cache is None


class TestTaskSpecCarriesTheView:
    @pytest.mark.parametrize('view', [_SHIPPED, None], ids=['view', 'none'])
    def test_the_view_round_trips(self, view):
        spec = dataclasses.replace(_task_spec(OrchestratorConfig()), verify_cache=view)

        assert task_spec_from_json(task_spec_to_json(spec)) == spec
        assert 'verify_cache' in spec_to_dict(spec)

    def test_a_spec_dict_without_the_block_is_still_rejected(self):
        wire = spec_to_dict(_task_spec(OrchestratorConfig()))
        del wire['verify_cache']

        with pytest.raises(TypeError, match='verify_cache'):
            spec_from_dict(wire)

    def test_a_view_that_is_not_a_shipped_view_is_rejected_naming_the_field(self):
        with pytest.raises(TypeError, match='TaskVerifySpec.verify_cache'):
            dataclasses.replace(_task_spec(OrchestratorConfig()), verify_cache=_SHIPPED.to_wire())

    def test_the_builder_ships_no_view(self):
        assert _task_spec(OrchestratorConfig()).verify_cache is None


_HOST_ON = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)
_TASK_FILES = ['alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py']


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', WORKSPACE_SEED)


@pytest.fixture
def markers(tmp_path: Path) -> Path:
    return tmp_path / 'markers.txt'


@pytest.fixture
def modules(markers: Path) -> list[ModuleConfig]:
    return [cache_module('alpha', markers), cache_module('beta', markers)]


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='r')


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    use_real_legs(tmp_path, monkeypatch, markers)


async def _gate_code() -> str:
    """The code identity a runner keys under: this gate checkout's HEAD, so both hosts' keys can be equal."""
    code = (await CacheAccess.on_runner()).orchestrator_code
    assert isinstance(code, str), code
    return code


async def _recorded_then_beta_edited(repo, modules, registry, store) -> dict[str, Entry]:
    """Every unit's green recorded by a local merge gate at HEAD, then a beta-only commit."""
    config = cache_config(repo, _BOTH_ON)
    pool = merge_pool(repo, config, modules, registry, store, code=await _gate_code())
    first = await pool.dispatch(head_of(repo), build_merge_verify_spec(config, modules, None))
    assert first.passed, first.summary
    recorded = await entries_for(registry, first)
    commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
    return recorded


async def _merge_spec_shipping(repo, modules, registry, modes: VerifyCacheConfig) -> MergeVerifySpec:
    spec = build_merge_verify_spec(cache_config(repo, _BOTH_ON), modules, None)
    view = await ship_view(registry, modes, 'merge', planned_unit_ids(spec.verify_commands), fits=lambda v: True)
    return dataclasses.replace(spec, verify_cache=view)


async def _on_the_runner(repo: Path, host_modes: VerifyCacheConfig, spec: MergeVerifySpec):
    return await run_merge_verify_on_worktree(
        repo, cache_config(repo, host_modes), spec, merge_sha=head_of(repo), run_unscoped=clean_unscoped_gate,
    )


@pytest.mark.usefixtures('seam')
class TestTheRunnerDecidesAgainstTheShippedView:
    @pytest.mark.asyncio
    async def test_an_on_view_replays_alpha_and_reports_every_unit(self, repo, modules, registry, store, markers):
        recorded = await _recorded_then_beta_edited(repo, modules, registry, store)
        spec = await _merge_spec_shipping(repo, modules, registry, _BOTH_ON)
        clear_markers(markers)

        result = await _on_the_runner(repo, VerifyCacheConfig(), spec)

        assert result.passed, result.summary
        units = units_by_id(result)
        for unit_id in module_units('alpha'):
            assert units[unit_id].outcome is UnitOutcome.CACHE_HIT
            assert units[unit_id].evidence_event_id == recorded[unit_id].evidence_event_id
        assert {units[unit_id].outcome for unit_id in module_units('beta')} == {UnitOutcome.RAN_GREEN}
        assert ran_legs(markers) == module_legs('beta')
        assert result.cache_report is not None
        reported = {unit.unit.unit: unit for unit in result.cache_report.units}
        assert set(reported) == module_units('alpha') | module_units('beta')
        assert all(unit.key is not None for unit in reported.values())
        beta_admissions = [reported[unit_id].admission for unit_id in module_units('beta')]
        assert all(admission is not None and admission.admitted for admission in beta_admissions)

    @pytest.mark.asyncio
    async def test_a_log_view_cannot_be_raised_by_the_runners_own_config(self, repo, modules, registry, store, markers):
        await _recorded_then_beta_edited(repo, modules, registry, store)
        spec = await _merge_spec_shipping(
            repo, modules, registry, _BOTH_ON.model_copy(update={'remote_merge_mode': CacheMode.LOG}),
        )
        clear_markers(markers)

        result = await _on_the_runner(repo, _HOST_ON, spec)

        assert result.passed, result.summary
        units = units_by_id(result)
        assert {units[unit_id].outcome for unit_id in module_units('alpha')} == {UnitOutcome.WOULD_HIT}
        assert ran_legs(markers) == module_legs('alpha') | module_legs('beta')

    @pytest.mark.asyncio
    async def test_no_view_settles_every_unit_unavailable_as_before(self, repo, modules, registry, store):
        await _recorded_then_beta_edited(repo, modules, registry, store)
        spec = build_merge_verify_spec(cache_config(repo, _BOTH_ON), modules, None)

        result = await _on_the_runner(repo, _HOST_ON, spec)

        assert result.passed, result.summary
        assert result.units
        for unit in result.units:
            assert unit.outcome is UnitOutcome.UNAVAILABLE
            assert unit.reason is not None and 'no_registry_on_runner' in unit.reason

    @pytest.mark.asyncio
    async def test_an_off_view_keys_nothing(self, repo, modules, registry, store):
        await _recorded_then_beta_edited(repo, modules, registry, store)
        spec = await _merge_spec_shipping(
            repo, modules, registry, _BOTH_ON.model_copy(update={'remote_merge_mode': CacheMode.OFF}),
        )

        result = await _on_the_runner(repo, _HOST_ON, spec)

        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.OFF
        assert all(unit.key is None for unit in result.units or ())

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.asyncio
    async def test_a_task_runner_replays_alpha_from_an_on_view(self, repo, modules, registry, markers):
        config = cache_config(repo, _BOTH_ON)
        (repo / '.task').mkdir()
        access = CacheAccess(registry, await _gate_code())
        recorded = await record_run(registry, await task_verify(repo, config, modules, access, _TASK_FILES))
        assert module_units('alpha') <= set(recorded)
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        tip = head_of(repo)
        spec = build_task_verify_spec(
            config, modules, _TASK_FILES, force_workspace=False, attempt_id=1, task_id='5653', tip_sha=tip,
        )
        view = await ship_view(registry, _BOTH_ON, 'task', planned_unit_ids(spec.verify_commands), fits=lambda v: True)
        git_ops = GitOps(config.git, repo)
        worktree = await git_ops.create_ephemeral_verify_worktree(tip)
        clear_markers(markers)
        try:
            result = await run_task_verify_on_worktree(
                worktree, cache_config(repo, VerifyCacheConfig()), dataclasses.replace(spec, verify_cache=view),
            )
        finally:
            await git_ops.cleanup_merge_worktree(worktree)

        assert result.passed, result.summary
        units = units_by_id(result)
        for unit_id in module_units('alpha'):
            assert units[unit_id].outcome is UnitOutcome.CACHE_HIT
            assert units[unit_id].evidence_event_id == recorded[unit_id].evidence_event_id
        assert not {leg for leg in ran_legs(markers) if leg.startswith('alpha:')}
        assert result.cache_report is not None


_ALPHA_LINT = 'module:alpha:lint'


@dataclasses.dataclass(frozen=True)
class _RunnerResult:
    """What a runner hands back: its verdict's units and its report, beside the view it decided against."""

    view: ShippedView
    units: tuple
    report: CacheReport | None
    replayed: dict[str, Entry]


async def _runner_replayed(registry: Registry, role: VerifyRole, *, replays: bool = True) -> _RunnerResult:
    """A runner's green under an ON view: alpha's test and lint replay the greens the
    dispatcher shipped (or run, unless *replays*), and beta's test runs clean."""
    keys_ = {unit_id: unit_key(unit_id) for unit_id in (_ALPHA, _ALPHA_LINT, _BETA)}
    replayed = {
        unit_id: await record_entry(registry, keys_[unit_id], evidence=evidence)
        for evidence, unit_id in enumerate((_ALPHA, _ALPHA_LINT), start=5)
    }
    view = ShippedView(CacheMode.ON, None, None, tuple(entry.candidate for entry in replayed.values()), False)
    def decided(unit_id: str, entry: Entry):
        if replays:
            return unit_decision(Decision.HIT, entry.candidate, key=keys_[unit_id])
        return unit_decision(Decision.MISS, key=keys_[unit_id])

    replay = ReplayPlan(
        role=role, modes=VerifyCacheConfig().with_mode(role, CacheMode.ON),
        tree=unit_decision(Decision.MISS, key=unit_key('tree', tier=UnitTier.TREE)),
        modules={
            **{unit_id: decided(unit_id, entry) for unit_id, entry in replayed.items()},
            _BETA: unit_decision(Decision.MISS, key=keys_[_BETA]),
        },
    )
    ran = (_BETA,) if replays else (_ALPHA, _ALPHA_LINT, _BETA)
    units, record = replay.settle(
        tuple(ran_unit(unit_id) for unit_id in ran),
        tuple(LegRun(unit_id, clean_leg(Leg.LINT if unit_id == _ALPHA_LINT else Leg.TEST)) for unit_id in ran),
    )
    result = verify_result(units=units, cache_report=report_of(record, flake_suppressed=False))
    assert result.units is not None
    return _RunnerResult(view, result.units, result.cache_report, replayed)


def _garbled(tmp_path: Path) -> Registry:
    db_path = tmp_path / 'garbled.db'
    db_path.write_bytes(b'garbage, not sqlite' * 64)
    return Registry(db_path)


class TestTheDispatcherSettlesWhatTheRunnerReports:
    @pytest.mark.asyncio
    async def test_a_merge_landing_consumes_every_replayed_entry(self, registry):
        ran = await _runner_replayed(registry, 'merge')

        settled = await RemoteCache(registry, _BOTH_ON).settle(ran.view, 'merge', ran.units, ran.report, landing_sha='land-1')

        assert settled.outcome.backs_every_hit, settled.outcome.reason
        assert set(await registry.consumed_for('land-1')) == set(ran.replayed.values())

    @pytest.mark.asyncio
    async def test_a_task_verify_checks_backing_and_consumes_nothing(self, registry):
        ran = await _runner_replayed(registry, 'task')

        settled = await RemoteCache(registry, _BOTH_ON).settle(ran.view, 'task', ran.units, ran.report, landing_sha=None)

        assert settled.outcome.backs_every_hit, settled.outcome.reason
        assert settled.outcome.consumed == ()

    @pytest.mark.asyncio
    async def test_an_entry_evicted_after_shipping_is_unbacked_by_unit_and_key(self, registry):
        ran = await _runner_replayed(registry, 'merge')
        evicted = ran.replayed[_ALPHA_LINT]
        await registry.evict_key(evicted.key, 'test eviction')

        settled = await RemoteCache(registry, _BOTH_ON).settle(ran.view, 'merge', ran.units, ran.report, landing_sha='land-1')

        assert settled.outcome.unbacked == (UnbackedReplay(_ALPHA_LINT, (_ALPHA_LINT, evicted.key)),)
        assert _ALPHA_LINT in settled.outcome.reason and evicted.key in settled.outcome.reason
        assert await registry.consumed_for('land-1') == ()

    @pytest.mark.asyncio
    async def test_a_replay_under_a_log_view_is_unbacked_whatever_the_registry_holds(self, registry):
        ran = await _runner_replayed(registry, 'merge')
        logged = dataclasses.replace(ran.view, mode=CacheMode.LOG)

        settled = await RemoteCache(registry, _BOTH_ON).settle(logged, 'merge', ran.units, ran.report, landing_sha='land-1')

        assert {unbacked.unit_id for unbacked in settled.outcome.unbacked} == {_ALPHA, _ALPHA_LINT}
        assert await registry.consumed_for('land-1') == ()

    @pytest.mark.asyncio
    async def test_a_garbled_registry_is_an_outcome_not_an_exception(self, registry, tmp_path):
        ran = await _runner_replayed(registry, 'merge')

        settled = await RemoteCache(_garbled(tmp_path), _BOTH_ON).settle(
            ran.view, 'merge', ran.units, ran.report, landing_sha='land-1',
        )

        assert settled.outcome.error is not None
        assert not settled.outcome.backs_every_hit

    @pytest.mark.asyncio
    async def test_the_settled_record_is_the_report_under_the_mode_the_dispatcher_shipped(self, registry):
        ran = await _runner_replayed(registry, 'merge', replays=False)
        logged = dataclasses.replace(ran.view, mode=CacheMode.LOG)
        assert ran.report is not None

        settled = await RemoteCache(registry, _BOTH_ON).settle(logged, 'merge', ran.units, ran.report, landing_sha='land-1')

        assert settled.record == ran.report.record('merge', _BOTH_ON.with_mode('merge', CacheMode.LOG))
        assert settled.record is not None and settled.record.modes.mode_for('merge') is CacheMode.LOG
        assert settled.outcome.backs_every_hit

    @pytest.mark.asyncio
    async def test_no_report_and_no_replay_settles_no_record(self, registry):
        ran = await _runner_replayed(registry, 'merge', replays=False)

        settled = await RemoteCache(registry, _BOTH_ON).settle(ran.view, 'merge', ran.units, None, landing_sha='land-1')

        assert settled.record is None
        assert settled.outcome.backs_every_hit

    @pytest.mark.asyncio
    async def test_a_replay_without_a_report_is_unbacked(self, registry):
        ran = await _runner_replayed(registry, 'merge')

        settled = await RemoteCache(registry, _BOTH_ON).settle(ran.view, 'merge', ran.units, None, landing_sha='land-1')

        assert set(settled.outcome.unbacked) == {UnbackedReplay(_ALPHA, None), UnbackedReplay(_ALPHA_LINT, None)}
        assert await registry.consumed_for('land-1') == ()

    @pytest.mark.asyncio
    async def test_no_shipped_view_records_nothing_and_backs_no_replay(self, registry):
        ran = await _runner_replayed(registry, 'merge')

        settled = await RemoteCache(registry, _BOTH_ON).settle(None, 'merge', ran.units, ran.report, landing_sha='land-1')

        assert settled.record is None
        assert {unbacked.unit_id for unbacked in settled.outcome.unbacked} == {_ALPHA, _ALPHA_LINT}
        assert await registry.consumed_for('land-1') == ()

    def test_a_dispatcher_without_a_registry_settles_nothing_remotely(self, registry):
        assert RemoteCache.of(None, _BOTH_ON) is None
        assert RemoteCache.of(registry, _BOTH_ON) == RemoteCache(registry, _BOTH_ON)


_LAPTOP = 'laptop'
#: Linux MAX_ARG_STRLEN: the remote shell is handed the whole command as one ``-c`` argument.
_MAX_ARG_STRLEN = 131_072


def _flag(words: Sequence[str], flag: str) -> str:
    return words[words.index(flag) + 1]


def _shipped_view(command: str) -> ShippedView | None:
    """The view the remote verify *command* carries in its ``--spec``."""
    return spec_from_json(_flag(gate_argv(command), '--spec')).verify_cache


class _Laptop:
    """Host R: a second checkout of the seeded repo that answers RemoteRunner's verify ssh
    with the real runner entry, in-process, under its own host *modes*. Every push it is
    sent succeeds; *before_answer* runs after R has decided and before its verdict returns."""

    def __init__(self, checkout: Path, modes: VerifyCacheConfig) -> None:
        self.checkout = checkout
        self.modes = modes
        self.commands: list[str] = []
        self.before_answer: Callable[[], Awaitable[None]] | None = None

    def runner(self, dispatcher_repo: Path) -> RemoteRunner:
        return gated_remote_runner(_LAPTOP, 'laptop.local', 'origin', str(dispatcher_repo), run=self.run)

    async def run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        if argv[:2] == ['git', 'push']:
            return (0, '', '')
        assert argv[0] == 'ssh', argv
        self.commands.append(argv[-1])
        words = gate_argv(argv[-1])
        assert words[0] == 'verify-merge', words
        sha = _flag(words, '--sha')
        workspace_git(self.checkout, 'checkout', '-q', '--detach', sha)
        result = await run_merge_verify_on_worktree(
            self.checkout, cache_config(self.checkout, self.modes), spec_from_json(_flag(words, '--spec')),
            merge_sha=sha, run_unscoped=clean_unscoped_gate,
        )
        if self.before_answer is not None:
            await self.before_answer()
        return (0, result_to_json(result), '')


@pytest.fixture
def laptop_checkout(repo: Path, tmp_path: Path) -> Path:
    checkout = tmp_path / 'laptop'
    workspace_git(repo, 'worktree', 'add', '-q', '--detach', str(checkout))
    return checkout


def _pin_interpreter(checkout: Path, version_info: str) -> None:
    """Give *checkout* a venv of interpreter *version_info*, excluded so its working copy stays clean."""
    exclude = Path(workspace_git(checkout, 'rev-parse', '--path-format=absolute', '--git-path', 'info/exclude'))
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open('a') as excluded:
        excluded.write('.venv/\n')
    (checkout / '.venv').mkdir()
    (checkout / '.venv' / 'pyvenv.cfg').write_text(
        f'home = /usr/bin\nimplementation = CPython\nversion_info = {version_info}\n',
    )


def _remote_pool(
    laptop: _Laptop, repo: Path, registry: Registry, store: EventStore, modes: VerifyCacheConfig,
) -> VerifyRunnerPool:
    """A pool of R alone, deciding through and recording into *registry* under the dispatcher's *modes*."""
    return VerifyRunnerPool(
        [laptop.runner(repo)], event_store=store, task_id='9123',
        verify_cache_registry=registry, verify_cache_modes=modes,
    )


def _dispatcher_spec(repo: Path, modules: list[ModuleConfig]) -> MergeVerifySpec:
    return build_merge_verify_spec(cache_config(repo, _BOTH_ON), modules, None)


@pytest.mark.usefixtures('seam')
class TestARemoteMergeGateThroughTheRealRemoteRunner:
    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.asyncio
    async def test_row_21_a_remote_green_is_recorded_then_replayed_and_consumed(
        self, repo, modules, registry, store, laptop_checkout, markers,
    ):
        pool = _remote_pool(_Laptop(laptop_checkout, VerifyCacheConfig()), repo, registry, store, _BOTH_ON)
        first = await pool.dispatch(head_of(repo), _dispatcher_spec(repo, modules))
        assert first.passed, first.summary
        first_entries = await entries_for(registry, first)
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        merge_sha = head_of(repo)
        clear_markers(markers)

        result = await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert result.passed, result.summary
        first_row, row = store.fetch_events_by_type(EventType.merge_verify)
        assert (first_row['data']['runner'], row['data']['runner']) == (_LAPTOP, _LAPTOP)
        units = row_units(row)
        for unit_id in module_units('alpha'):
            assert units[unit_id]['outcome'] == 'cache_hit'
            assert units[unit_id]['evidence_event_id'] == first_entries[unit_id].evidence_event_id == first_row['id']
        assert {entry.unit_id: entry.entry_id for entry in await registry.consumed_for(merge_sha)} == {
            unit_id: first_entries[unit_id].entry_id for unit_id in module_units('alpha')
        }
        assert ran_legs(markers) == module_legs('beta')
        recorded = await entries_for(registry, result)
        for unit_id in module_units('beta'):
            assert (recorded[unit_id].evidence_event_id, recorded[unit_id].sha) == (row['id'], merge_sha)

    @pytest.mark.asyncio
    async def test_row_22_a_local_green_replays_on_a_remote_whose_keyed_inputs_agree(
        self, repo, modules, registry, store, laptop_checkout,
    ):
        recorded = await _recorded_then_beta_edited(repo, modules, registry, store)
        merge_sha = head_of(repo)
        pool = _remote_pool(_Laptop(laptop_checkout, VerifyCacheConfig()), repo, registry, store, _BOTH_ON)

        result = await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert result.passed, result.summary
        assert outcomes_of(result, module_units('alpha')) == {UnitOutcome.CACHE_HIT}
        assert {entry.unit_id: entry.entry_id for entry in await registry.consumed_for(merge_sha)} == {
            unit_id: recorded[unit_id].entry_id for unit_id in module_units('alpha')
        }

    @pytest.mark.asyncio
    async def test_row_22_a_local_green_misses_on_a_remote_whose_interpreter_differs(
        self, repo, modules, registry, store, laptop_checkout,
    ):
        await _recorded_then_beta_edited(repo, modules, registry, store)
        _pin_interpreter(laptop_checkout, '3.99.0.final.0')
        merge_sha = head_of(repo)
        pool = _remote_pool(_Laptop(laptop_checkout, VerifyCacheConfig()), repo, registry, store, _BOTH_ON)

        result = await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert result.passed, result.summary
        assert outcomes_of(result, module_units('alpha')) == {UnitOutcome.RAN_GREEN}
        assert await registry.consumed_for(merge_sha) == ()
        assert len(await registry.candidates([_ALPHA], per_unit=16)) == 2

    @pytest.mark.asyncio
    async def test_row_23_a_log_role_runs_every_unit_on_the_remote_and_consumes_nothing(
        self, repo, modules, registry, store, laptop_checkout, markers,
    ):
        recorded = await _recorded_then_beta_edited(repo, modules, registry, store)
        merge_sha = head_of(repo)
        modes = _BOTH_ON.model_copy(update={'merge_mode': CacheMode.LOG})
        pool = _remote_pool(_Laptop(laptop_checkout, _HOST_ON), repo, registry, store, modes)
        clear_markers(markers)

        result = await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert result.passed, result.summary
        assert ran_legs(markers) == module_legs('alpha') | module_legs('beta')
        local_row, row = store.fetch_events_by_type(EventType.merge_verify)
        units = row_units(row)
        for unit_id in module_units('alpha'):
            assert units[unit_id]['outcome'] == 'would_hit'
            assert units[unit_id]['evidence_event_id'] == recorded[unit_id].evidence_event_id == local_row['id']
        assert local_row['data']['runner'] != row['data']['runner'] == _LAPTOP
        assert await registry.consumed_for(merge_sha) == ()

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.asyncio
    async def test_row_24_a_green_evicted_between_ship_and_answer_is_no_verdict_and_never_reshipped(
        self, repo, modules, registry, store, laptop_checkout,
    ):
        recorded = await _recorded_then_beta_edited(repo, modules, registry, store)
        evicted = recorded[_ALPHA]
        merge_sha = head_of(repo)
        laptop = _Laptop(laptop_checkout, VerifyCacheConfig())

        async def evict_once() -> None:
            laptop.before_answer = None
            await registry.evict_key(evicted.key, 'row 24: evicted between ship and answer')

        laptop.before_answer = evict_once
        pool = _remote_pool(laptop, repo, registry, store, _BOTH_ON)

        with pytest.raises(RunnerUnavailable) as raised:
            await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert raised.value.kind == 'no_verdict'
        assert _ALPHA in str(raised.value) and evicted.key in str(raised.value)
        shipped = _shipped_view(laptop.commands[0])
        assert shipped is not None and evicted.candidate in shipped.candidates

        result = await pool.dispatch(merge_sha, _dispatcher_spec(repo, modules))

        assert result.passed, result.summary
        reshipped = _shipped_view(laptop.commands[-1])
        assert reshipped is not None
        assert evicted.key not in {candidate.key for candidate in reshipped.candidates}
        assert units_by_id(result)[_ALPHA].outcome is UnitOutcome.RAN_GREEN


class _HeldGreens(Registry):
    """A registry holding *count* greens of each of *unit_ids*, served without writing
    them: recording them one durable transaction at a time costs ~40 ms each."""

    def __init__(self, path: Path, unit_ids: Sequence[str], count: int) -> None:
        super().__init__(path)
        self._greens = {
            unit_id: tuple(
                Candidate(
                    unit_id, hashlib.sha256(f'{unit_id}#{n}'.encode()).hexdigest(),
                    1_000_000 + index * count + n, 9_000_000 + index * count + n,
                )
                for n in range(count)
            )
            for index, unit_id in enumerate(unit_ids)
        }

    async def candidates(self, unit_ids: Sequence[str], *, per_unit: int) -> tuple[Candidate, ...]:
        return tuple(candidate for unit_id in unit_ids for candidate in self._greens.get(unit_id, ())[:per_unit])

    async def policy_state(self) -> PolicyState:
        return PolicyState()


class _AnswersGreen:
    """A remote that keeps every verify command it is sent and answers each green."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def run(self, argv: list[str], *, cwd: Any = None) -> tuple[int, str, str]:
        if argv[0] != 'ssh':
            return (0, '', '')
        self.commands.append(argv[-1])
        return (0, result_to_json(verify_result()), '')


_THIRTY_UNITS_SPEC = MergeVerifySpec(
    verify_commands=tuple(
        VerifyCommand(f'module{n:02d}', test_command='pytest', lint_command='ruff check', type_check_command='pyright')
        for n in range(10)
    ),
    unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
    task_files=('module00/a.py',),
    verify_env={},
    cold_timeout_secs=60.0,
)


class TestTheShippedCommandFitsOneArgument:
    async def _shipped(self, tmp_path: Path, greens: int, modes: VerifyCacheConfig) -> tuple[str, ShippedView]:
        units = [unit_id for unit_id in planned_unit_ids(_THIRTY_UNITS_SPEC.verify_commands) if unit_id != keys.TREE_UNIT_ID]
        assert len(units) == 30
        remote = _AnswersGreen()
        runner = gated_remote_runner(_LAPTOP, 'laptop.local', 'origin', '/repo', run=remote.run)

        result = await runner.run_merge_verify(
            'a' * 40, _THIRTY_UNITS_SPEC, verify_cache=RemoteCache(_HeldGreens(tmp_path / 'held.db', units, greens), modes),
        )

        assert result.passed, result.summary
        (command,) = remote.commands
        view = _shipped_view(command)
        assert view is not None
        return command, view

    @pytest.mark.asyncio
    async def test_thirty_units_of_twenty_greens_ship_their_cap_of_sixteen(self, tmp_path):
        command, view = await self._shipped(tmp_path, 20, _BOTH_ON)

        assert len(command.encode()) <= _MAX_ARG_STRLEN
        assert view.truncated is True
        depths = Counter(candidate.unit_id for candidate in view.candidates)
        assert len(depths) == 30
        assert set(depths.values()) == {16}

    @pytest.mark.asyncio
    async def test_a_cap_the_command_cannot_hold_is_trimmed_to_fit_and_says_truncated(self, tmp_path):
        command, view = await self._shipped(
            tmp_path, 2000, _BOTH_ON.model_copy(update={'remote_candidates_per_unit': 2000}),
        )

        assert len(command.encode()) <= _MAX_ARG_STRLEN
        assert view.truncated is True
        assert 0 < len(view.candidates) < 30 * 2000
