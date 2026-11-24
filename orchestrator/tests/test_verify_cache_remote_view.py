"""A remote verify decides against a view of the dispatcher's registry (task 9123,
plans/verify-result-cache-prd.md Amendment A5): the shipped view's wire value, how
the dispatcher builds it, how the runner decides against it, and how the
dispatcher settles what the runner reports."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _verify_cache_units import record_entry, unit_key

from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.task_verify_spec import (
    build_task_verify_spec,
    spec_from_dict,
    spec_to_dict,
    task_spec_from_json,
    task_spec_to_json,
)
from orchestrator.verify_cache import keys
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.registry import Candidate, KillSwitch, PolicyState, Registry
from orchestrator.verify_cache.remote_view import ShippedView, planned_unit_ids, ship_view
from orchestrator.verify_cache.replay import ExecutedModule
from orchestrator.verify_runner import (
    MergeVerifySpec,
    UnscopedTypecheckSpec,
    build_merge_verify_spec,
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
