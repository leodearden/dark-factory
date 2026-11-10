"""Whether a verify unit may replay a recorded green, and whether a green may be recorded (task 5652)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache import keys, policy, registry
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_evidence import Leg, UnitTier

_NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
_KEY = keys.UnitKey.of('module:alpha:test', UnitTier.MODULE, 's', (('spec', 's'), ('tree:alpha', 'a' * 40)))
_ENTRY = registry.Entry(
    entry_id=1, unit_id=_KEY.unit_id, key=_KEY.key, tier=_KEY.tier, spec_hash=_KEY.spec_hash,
    inputs=_KEY.inputs, evidence_event_id=7, run_id='r1', sha='abc', recorded_at=_NOW,
)
_ENABLED = registry.PolicyState()
_WITH_ENTRY = registry.RegistryView({(_KEY.unit_id, _KEY.key): _ENTRY}, _ENABLED)
_EMPTY = registry.RegistryView({}, _ENABLED)
_TASK_ON = VerifyCacheConfig(task_mode=CacheMode.ON)
_MERGE_ON = VerifyCacheConfig(merge_mode=CacheMode.ON)
_BOTH_ON = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)


def _decide(modes, role, view, *, never_cache: bool = False) -> policy.CacheDecision:
    return policy.decide(modes, role, _KEY, view, never_cache=never_cache, now=_NOW)


def _disabled_until(until: datetime) -> registry.RegistryView:
    state = registry.PolicyState(disabled_until=until, reason='two leaks', streak=2, leaks=('l1', 'l2'))
    return registry.RegistryView({(_KEY.unit_id, _KEY.key): _ENTRY}, state)


def _assert(decision: policy.CacheDecision, expected: policy.Decision, *named: str) -> None:
    assert decision.decision is expected
    assert decision.reason
    for word in named:
        assert word in decision.reason


class _ExplodingEntries(Mapping):
    def __getitem__(self, unit: object) -> registry.Entry:
        raise RuntimeError('boom')

    def __iter__(self) -> Iterator:
        return iter(())

    def __len__(self) -> int:
        return 0


class TestDecide:
    def test_an_on_role_with_an_entry_replays_it(self):
        decision = _decide(_TASK_ON, 'task', _WITH_ENTRY)

        _assert(decision, policy.Decision.HIT, _KEY.unit_id, _KEY.key, 'evidence event 7')
        assert decision.entry is not None
        assert decision.entry == _ENTRY
        assert decision.entry.evidence_event_id == 7

    def test_task_on_leaves_merge_off(self):
        _assert(_decide(_TASK_ON, 'merge', _WITH_ENTRY), policy.Decision.MISS, 'merge', 'off')

    def test_merge_on_leaves_task_off(self):
        _assert(_decide(_MERGE_ON, 'merge', _WITH_ENTRY), policy.Decision.HIT)
        _assert(_decide(_MERGE_ON, 'task', _WITH_ENTRY), policy.Decision.MISS, 'task', 'off')

    def test_a_log_role_with_an_entry_would_hit(self):
        decision = _decide(VerifyCacheConfig(task_mode=CacheMode.LOG), 'task', _WITH_ENTRY)

        _assert(decision, policy.Decision.WOULD_HIT)
        assert decision.entry == _ENTRY

    @pytest.mark.parametrize('mode', [CacheMode.ON, CacheMode.LOG])
    def test_no_entry_is_a_miss(self, mode):
        decision = _decide(VerifyCacheConfig(task_mode=mode), 'task', _EMPTY)

        _assert(decision, policy.Decision.MISS)
        assert decision.entry is None

    def test_a_never_cache_module_misses_even_with_an_entry(self):
        _assert(_decide(_TASK_ON, 'task', _WITH_ENTRY, never_cache=True), policy.Decision.MISS, 'never')

    def test_an_unavailable_registry_is_unavailable_for_an_active_role(self):
        unavailable = policy.Unavailable('registry locked')

        _assert(_decide(_TASK_ON, 'task', unavailable), policy.Decision.UNAVAILABLE, 'registry locked')
        _assert(_decide(_TASK_ON, 'merge', unavailable), policy.Decision.MISS, 'off')

    @pytest.mark.parametrize('role', ['task', 'merge'])
    def test_the_kill_switch_disables_every_role(self, role):
        _assert(
            _decide(_BOTH_ON, role, _disabled_until(_NOW + timedelta(days=1))),
            policy.Decision.DISABLED, 'two leaks',
        )

    def test_an_expired_kill_switch_replays_again(self):
        _assert(_decide(_BOTH_ON, 'task', _disabled_until(_NOW - timedelta(days=1))), policy.Decision.HIT)

    def test_a_failing_view_is_unavailable_never_raised(self):
        exploding = registry.RegistryView(_ExplodingEntries(), _ENABLED)

        _assert(_decide(_TASK_ON, 'task', exploding), policy.Decision.UNAVAILABLE, 'boom')


class TestConsult:
    @pytest.mark.asyncio
    async def test_no_registry_is_unavailable(self):
        consulted = await policy.consult(None, [_KEY])

        assert isinstance(consulted, policy.Unavailable)
        assert 'no registry' in consulted.reason

    @pytest.mark.asyncio
    async def test_a_damaged_registry_is_unavailable_naming_it(self, tmp_path: Path):
        db_path = tmp_path / 'verify_cache.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)

        consulted = await policy.consult(registry.Registry(db_path), [_KEY])

        assert isinstance(consulted, policy.Unavailable)
        assert str(db_path) in consulted.reason

    @pytest.mark.asyncio
    async def test_a_healthy_registry_answers_its_view(self, tmp_path: Path):
        store = registry.Registry(tmp_path / 'verify_cache.db', clock=lambda: _NOW)
        entry = await store.record_green(_KEY, evidence_event_id=7, run_id='r1', sha='abc')

        consulted = await policy.consult(store, [_KEY])

        assert isinstance(consulted, registry.RegistryView)
        assert consulted.entries == {(_KEY.unit_id, _KEY.key): entry}


_CLEAN_TEST_LEG = policy.LegResult(
    leg=Leg.TEST, rc=0, timed_out=False, trivial=False, flake_suppressed=False,
    junit_tests=5, collected=5, never_cache=False,
)
_TASK_LOG = VerifyCacheConfig(task_mode=CacheMode.LOG)


class TestAdmissible:
    def test_a_clean_green_test_leg_is_admitted(self):
        admission = policy.admissible(_TASK_LOG, 'task', _CLEAN_TEST_LEG)

        assert admission.admitted is True
        assert admission.reason

    def test_an_off_role_records_nothing(self):
        admission = policy.admissible(_TASK_LOG, 'merge', _CLEAN_TEST_LEG)

        assert admission.admitted is False
        assert 'off' in admission.reason

    @pytest.mark.parametrize(('changed', 'named'), [
        ({'never_cache': True}, ('never',)),
        ({'trivial': True}, ('trivial',)),
        ({'rc': 1}, ('1',)),
        ({'timed_out': True}, ('timed out',)),
        ({'flake_suppressed': True}, ('flake',)),
        ({'junit_tests': 3}, ('3', '5')),
        ({'junit_tests': 7}, ('7', '5')),
        ({'junit_tests': None}, ('None',)),
        ({'collected': None}, ('None',)),
    ])
    def test_a_leg_that_is_not_a_clean_green_is_refused_naming_why(self, changed, named):
        admission = policy.admissible(_TASK_LOG, 'task', dataclasses.replace(_CLEAN_TEST_LEG, **changed))

        assert admission.admitted is False
        for word in named:
            assert word in admission.reason

    def test_a_lint_leg_needs_no_junit_counts(self):
        lint = dataclasses.replace(_CLEAN_TEST_LEG, leg=Leg.LINT, junit_tests=None, collected=None)

        assert policy.admissible(_TASK_LOG, 'task', lint).admitted is True
