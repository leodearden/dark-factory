"""What a verify result leaves for the verify-result cache, and writing it (task 5653)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _task_workflow_scene import (
    SCENE_TASK_ID,
    forked_task_workflow,
    scene_git,
    task_scene_config,
    task_workflow,
)
from _verify_cache_fixtures import put_pytest_on_path

from orchestrator.artifacts import ReviewAggregation
from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.verify import (
    VerifyResult,
    pytest_reported_when_complete,
    run_scoped_verification,
    run_verification,
)
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.policy import CacheDecision, Decision, LegResult
from orchestrator.verify_cache.recorder import (
    CacheRecord,
    LegRun,
    VerifyEvidenceRef,
    consume_replayed,
    record_greens,
)
from orchestrator.verify_cache.registry import Entry, Registry, registry_db_path
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan, UnitDecision
from orchestrator.verify_evidence import (
    Leg,
    TipUnresolved,
    UnitOutcome,
    UnitTier,
    VerifyUnit,
    module_leg_unit,
)
from orchestrator.verify_runner import result_from_dict, result_to_dict
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_PROGRESS_COLUMN = 73


def _progress(statuses: str, percent: int) -> str:
    return f'{statuses.ljust(_PROGRESS_COLUMN)}[{percent:3d}%]'


@pytest.fixture
def pytest_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    put_pytest_on_path(tmp_path, monkeypatch)


def _leg(leg: Leg = Leg.TEST, *, rc: int = 0) -> LegResult:
    return LegResult(
        leg=leg, rc=rc, timed_out=False, trivial=False, flake_suppressed=False,
        junit_tests=None, collected=None, never_cache=False,
    )


def _record(*legs: LegRun) -> CacheRecord:
    return CacheRecord(role='merge', modes=VerifyCacheConfig(merge_mode=CacheMode.LOG), legs=legs)


def _result(**overrides: Any) -> VerifyResult:
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='', **overrides)


class TestPytestReportedWhenComplete:
    def test_one_quiet_session_counts_its_tally(self):
        output = '\n'.join([
            _progress('...............', 100),
            '12 passed, 2 skipped, 1 xfailed in 3.1s',
        ])

        assert pytest_reported_when_complete(output) == 15

    def test_segmented_sessions_sum(self):
        output = '\n'.join([
            '===== segment 1/2 [orchestrator] cwd=/w/orchestrator rc=0 status=passed (3.1s) =====',
            'bringing up nodes...',
            _progress('........', 100),
            '8 passed in 1.20s',
            '===== segment 2/2 [shared] cwd=/w/shared rc=0 status=passed (2.0s) =====',
            'bringing up nodes...',
            _progress('...s', 100),
            '3 passed, 1 skipped in 0.90s',
        ])

        assert pytest_reported_when_complete(output) == 12

    def test_an_and_chain_of_quiet_sessions_sums(self):
        output = '\n'.join([
            'bringing up nodes...',
            _progress('...', 100),
            '3 passed in 0.5s',
            'bringing up nodes...',
            _progress('.....', 100),
            '5 passed, 16 warnings in 366.84s (0:06:06)',
        ])

        assert pytest_reported_when_complete(output) == 8

    def test_a_final_progress_line_below_full_is_no_count(self):
        output = '\n'.join([
            _progress('..........', 45),
            '10 passed in 2.0s',
        ])

        assert pytest_reported_when_complete(output) is None

    def test_a_truncated_session_after_a_complete_one_is_no_count(self):
        output = '\n'.join([
            _progress('...', 100),
            '3 passed in 0.5s',
            _progress('..', 40),
        ])

        assert pytest_reported_when_complete(output) is None

    def test_a_tally_without_a_progress_line_is_no_count(self):
        assert pytest_reported_when_complete('12 passed in 3.1s') is None

    def test_output_without_a_pytest_session_is_no_count(self):
        assert pytest_reported_when_complete('All checks passed\n') is None

    def test_deselected_and_warnings_are_not_counted(self):
        output = '\n'.join([
            _progress('...', 100),
            '3 passed, 4 deselected, 2 warnings in 0.3s',
        ])

        assert pytest_reported_when_complete(output) == 3

    def test_every_outcome_of_a_collected_test_is_counted(self):
        output = '\n'.join([
            _progress('F.EEsxX', 100),
            '1 failed, 1 passed, 1 skipped, 1 xfailed, 1 xpassed, 2 errors in 0.5s',
        ])

        assert pytest_reported_when_complete(output) == 7


class TestLegFacts:
    @pytest.mark.asyncio
    async def test_each_check_that_ran_leaves_one_leg(self, tmp_path):
        module = ModuleConfig(
            prefix='alpha', test_command='true', lint_command='false', type_check_command=None,
            verify_cache='never',
        )
        config = OrchestratorConfig()

        result = await run_verification(tmp_path, config, module, role='merge')

        record = result.cache_record
        assert record is not None
        assert (record.role, record.modes) == ('merge', config.verify_cache)
        assert record.legs == (
            LegRun('module:alpha:test', LegResult(
                leg=Leg.TEST, rc=0, timed_out=False, trivial=False, flake_suppressed=False,
                junit_tests=None, collected=None, never_cache=True,
            )),
            LegRun('module:alpha:lint', LegResult(
                leg=Leg.LINT, rc=1, timed_out=False, trivial=False, flake_suppressed=False,
                junit_tests=None, collected=None, never_cache=True,
            )),
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('pytest_on_path')
    async def test_a_full_merge_pytest_leg_counts_its_junit_and_its_collection(self, tmp_path):
        suite = tmp_path / 'suite'
        suite.mkdir()
        (suite / 'test_two.py').write_text('def test_a():\n    pass\n\n\ndef test_b():\n    pass\n')
        module = ModuleConfig(
            prefix='alpha', test_command=f'pytest {suite} -q -p no:cacheprovider',
            lint_command=None, type_check_command=None,
        )
        config = OrchestratorConfig(merge_verify_breadth='full')

        result = await run_verification(tmp_path, config, module, role='merge')

        assert result.passed, result.test_output
        assert result.cache_record is not None
        (test_leg,) = result.cache_record.legs
        assert (test_leg.leg.junit_tests, test_leg.leg.collected) == (2, 2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('beta_recorded', [True, False])
    async def test_aggregation_concatenates_legs_and_fails_closed(
        self, tmp_path, monkeypatch, beta_recorded,
    ):
        alpha_leg = LegRun('module:alpha:test', _leg())
        beta_leg = LegRun('module:beta:test', _leg(rc=1))
        by_module = {
            'alpha': _result(cache_record=_record(alpha_leg)),
            'beta': _result(cache_record=_record(beta_leg) if beta_recorded else None),
        }

        async def _module_verdict(
            _worktree: Path, _config: OrchestratorConfig, module: ModuleConfig, **_kwargs: Any,
        ) -> VerifyResult:
            return by_module[module.prefix]

        monkeypatch.setattr('orchestrator.verify.run_verification', _module_verdict)
        repo = seed_repo(tmp_path / 'repo', _TWO_MODULE_SEED)

        result = await run_scoped_verification(
            repo, OrchestratorConfig(), [_module('alpha'), _module('beta')],
            task_files=['alpha/core.py', 'beta/core.py'], role='task',
        )

        if beta_recorded:
            assert result.cache_record is not None
            assert result.cache_record.legs == (alpha_leg, beta_leg)
        else:
            assert result.cache_record is None

    def test_the_record_never_crosses_the_wire(self):
        result = _result(cache_record=_record(LegRun('module:alpha:test', _leg())))

        wire = result_to_dict(result)
        rebuilt = result_from_dict(json.loads(json.dumps(wire)))

        assert 'cache_record' not in wire
        assert rebuilt.cache_record is None
        assert rebuilt == result


def _module(prefix: str) -> ModuleConfig:
    return ModuleConfig(prefix=prefix, test_command='true', lint_command='true', type_check_command='true')


_TWO_MODULE_SEED = RepoSeed(
    files=(('alpha/core.py', 'x = 1\n'), ('beta/core.py', 'x = 1\n')),
    message='init',
)


# ---------------------------------------------------------------------------
# Settling a verify's units against its replay plan, and recording its greens.

_MERGE_ON = VerifyCacheConfig(merge_mode=CacheMode.ON)
_MERGE_LOG = VerifyCacheConfig(merge_mode=CacheMode.LOG)


def _key(unit_id: str, *, tier: UnitTier = UnitTier.MODULE, token: str = 'a') -> UnitKey:
    spec = hashlib.sha256(unit_id.encode()).hexdigest()
    return UnitKey.of(unit_id, tier, spec, [('spec', spec), ('tree:x', token)])


def _decided(decision: Decision, entry: Entry | None = None, *, key: UnitKey | None = None) -> UnitDecision:
    return UnitDecision(key, CacheDecision(decision, entry, f'{decision} because the test said so'))


def _replay(
    modules: dict[str, UnitDecision], *, tree: UnitDecision | None = None, modes: VerifyCacheConfig = _MERGE_ON,
) -> ReplayPlan:
    tree = tree or _decided(Decision.MISS, key=_key('tree', tier=UnitTier.TREE))
    return ReplayPlan(role='merge', modes=modes, tree=tree, modules=modules)


def _ran(unit_id: str, rc: int = 0) -> VerifyUnit:
    module, leg = unit_id.removeprefix('module:').rsplit(':', 1)
    return module_leg_unit(module, leg, rc)


def _clean(leg: Leg, **overrides: Any) -> LegResult:
    fields: dict[str, Any] = {
        'leg': leg, 'rc': 0, 'timed_out': False, 'trivial': False, 'flake_suppressed': False,
        'junit_tests': 2 if leg is Leg.TEST else None, 'collected': 2 if leg is Leg.TEST else None,
        'never_cache': False, **overrides,
    }
    return LegResult(**fields)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


async def _entry(registry: Registry, key: UnitKey, evidence: int = 7) -> Entry:
    return await registry.record_green(key, evidence_event_id=evidence, run_id='r0', sha='s0')


class TestSettle:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('decision', 'rc', 'outcome', 'carries_evidence'), [
        (Decision.WOULD_HIT, 0, UnitOutcome.WOULD_HIT, True),
        (Decision.WOULD_HIT, 1, UnitOutcome.RAN_RED, True),
        (Decision.UNAVAILABLE, 0, UnitOutcome.UNAVAILABLE, False),
        (Decision.UNAVAILABLE, 1, UnitOutcome.RAN_RED, False),
        (Decision.DISABLED, 0, UnitOutcome.UNAVAILABLE, False),
        (Decision.DISABLED, 1, UnitOutcome.RAN_RED, False),
        (Decision.MISS, 0, UnitOutcome.RAN_GREEN, False),
        (Decision.MISS, 1, UnitOutcome.RAN_RED, False),
    ])
    async def test_a_unit_that_ran_settles_by_its_decision_and_its_run(
        self, registry, decision, rc, outcome, carries_evidence,
    ):
        key = _key('module:alpha:test')
        entry = await _entry(registry, key) if decision is Decision.WOULD_HIT else None
        replay = _replay({'module:alpha:test': _decided(decision, entry, key=key)})
        leg = LegRun('module:alpha:test', _clean(Leg.TEST, rc=rc))

        units, record = replay.settle((_ran('module:alpha:test', rc),), (leg,))

        unit = units[0]
        assert unit.unit == 'module:alpha:test'
        assert unit.outcome is outcome
        assert unit.key == key.key
        assert unit.reason == replay.modules['module:alpha:test'].decision.reason
        assert unit.evidence_event_id == (entry.evidence_event_id if carries_evidence and entry else None)
        assert unit.would_have_leaked is (decision is Decision.WOULD_HIT and rc != 0)
        (recorded,) = record.units
        assert (recorded.unit, recorded.key, recorded.leg, recorded.entry) == (unit, key, leg.leg, entry)

    @pytest.mark.asyncio
    async def test_a_hit_that_did_not_run_settles_as_a_cache_hit(self, registry):
        key = _key('module:alpha:lint')
        entry = await _entry(registry, key, evidence=41)
        replay = _replay({'module:alpha:lint': _decided(Decision.HIT, entry, key=key)})

        units, record = replay.settle((), ())

        assert units[0] == VerifyUnit(
            unit='module:alpha:lint', tier=UnitTier.MODULE, key=key.key, outcome=UnitOutcome.CACHE_HIT,
            evidence_event_id=41, reason=replay.modules['module:alpha:lint'].decision.reason,
        )
        assert record.units[0].entry == entry
        assert record.units[0].leg is None

    def test_a_unit_outside_the_plan_passes_through_unkeyed(self):
        replay = _replay({})
        leg = LegRun('module:gamma:test', _clean(Leg.TEST))

        units, record = replay.settle((_ran('module:gamma:test'),), (leg,))

        assert units[0].outcome is UnitOutcome.RAN_GREEN
        assert units[0].key is None
        assert units[0].reason is not None
        assert units[0].reason.startswith('not keyed: ')
        assert record.units[0].leg == leg.leg

    def test_the_tree_unit_settles_last_and_records_its_own_key(self):
        tree_key = _key('tree', tier=UnitTier.TREE)
        replay = _replay({}, tree=_decided(Decision.MISS, key=tree_key))

        units, record = replay.settle((_ran('module:gamma:test'),), (LegRun('module:gamma:test', _clean(Leg.TEST)),))

        assert units[-1].unit == 'tree'
        assert units[-1].tier is UnitTier.TREE
        assert units[-1].outcome is UnitOutcome.RAN_GREEN
        assert record.tree is not None
        assert record.tree.key == tree_key
        assert (record.role, record.modes) == ('merge', _MERGE_ON)

    def test_a_red_unit_makes_the_tree_red(self):
        replay = _replay({}, tree=_decided(Decision.MISS, key=_key('tree', tier=UnitTier.TREE)))

        units, _record = replay.settle((_ran('module:gamma:test', 1),), ())

        assert units[-1].outcome is UnitOutcome.RAN_RED


def _all_admitted() -> tuple[ReplayPlan, tuple[VerifyUnit, ...], tuple[LegRun, ...]]:
    unit_ids = ('module:alpha:test', 'module:alpha:lint')
    replay = _replay(
        {unit_id: _decided(Decision.MISS, key=_key(unit_id)) for unit_id in unit_ids}, modes=_MERGE_LOG,
    )
    legs = (LegRun(unit_ids[0], _clean(Leg.TEST)), LegRun(unit_ids[1], _clean(Leg.LINT)))
    return replay, tuple(_ran(unit_id) for unit_id in unit_ids), legs


def _evidence(event_id: int = 11, *, passed: bool = True) -> VerifyEvidenceRef:
    return VerifyEvidenceRef(event_id=event_id, run_id='r1', sha='merge-sha', passed=passed)


async def _recorded(registry: Registry, *keys_: UnitKey) -> dict[str, Entry]:
    view = await registry.view(keys_)
    return {unit_id: entry for (unit_id, _key_digest), entry in view.entries.items()}


class TestRecordGreens:
    @pytest.mark.asyncio
    async def test_only_admitted_units_are_recorded(self, registry):
        unit_ids = ('module:alpha:test', 'module:alpha:lint', 'module:alpha:type', 'module:beta:test')
        keys_ = {unit_id: _key(unit_id) for unit_id in unit_ids}
        replay = _replay({unit_id: _decided(Decision.MISS, key=keys_[unit_id]) for unit_id in unit_ids})
        legs = (
            LegRun('module:alpha:test', _clean(Leg.TEST)),
            LegRun('module:alpha:lint', _clean(Leg.LINT, rc=1)),
            LegRun('module:alpha:type', _clean(Leg.TYPE, timed_out=True)),
            LegRun('module:beta:test', _clean(Leg.TEST, collected=3)),
        )
        ran = (_ran(unit_ids[0]), _ran(unit_ids[1], 1), _ran(unit_ids[2]), _ran(unit_ids[3]))
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        recorded = await _recorded(registry, *keys_.values())
        assert set(recorded) == {'module:alpha:test'}
        assert recorded['module:alpha:test'].evidence_event_id == 11
        assert (recorded['module:alpha:test'].run_id, recorded['module:alpha:test'].sha) == ('r1', 'merge-sha')
        assert outcome.reason

    @pytest.mark.asyncio
    async def test_a_never_cache_leg_is_not_recorded(self, registry):
        key = _key('module:alpha:test')
        replay = _replay({'module:alpha:test': _decided(Decision.MISS, key=key)})
        legs = (LegRun('module:alpha:test', _clean(Leg.TEST, never_cache=True)),)
        _units, record = replay.settle((_ran('module:alpha:test'),), legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        assert await _recorded(registry, key) == {}

    @pytest.mark.asyncio
    async def test_a_re_record_refreshes_the_one_row(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(11))
        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(12))

        key = replay.modules['module:alpha:test'].key
        assert key is not None
        view = await registry.view([key])
        assert [entry.evidence_event_id for entry in view.entries.values()] == [12]

    @pytest.mark.asyncio
    async def test_a_suppressed_flake_records_nothing(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=True, evidence=_evidence())

        assert await _recorded(registry, *_keys_of(replay)) == {}

    @pytest.mark.asyncio
    async def test_a_replayed_unit_is_not_new_evidence(self, registry):
        key = _key('module:alpha:lint')
        entry = await _entry(registry, key, evidence=5)
        replay = _replay({'module:alpha:lint': _decided(Decision.HIT, entry, key=key)})
        _units, record = replay.settle((), ())

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(99))

        assert (await _recorded(registry, key))['module:alpha:lint'].evidence_event_id == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize('passed', [True, False])
    async def test_the_tree_is_recorded_only_over_a_passing_verdict(self, registry, passed):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(passed=passed))

        assert replay.tree.key is not None
        recorded = await _recorded(registry, replay.tree.key, *_keys_of(replay))
        assert set(recorded) == ({'tree'} if passed else set()) | {'module:alpha:test', 'module:alpha:lint'}

    @pytest.mark.asyncio
    async def test_the_tree_is_not_recorded_beside_a_refused_unit(self, registry):
        replay, ran, legs = _all_admitted()
        refused = (legs[0], LegRun(legs[1].unit_id, _clean(Leg.LINT, timed_out=True)))
        _units, record = replay.settle(ran, refused)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        assert replay.tree.key is not None
        assert 'tree' not in await _recorded(registry, replay.tree.key)

    @pytest.mark.asyncio
    async def test_no_evidence_records_nothing_and_says_why(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(registry, record, flake_suppressed=False, evidence=None)

        assert outcome.recorded == ()
        assert 'evidence' in outcome.reason
        assert await _recorded(registry, *_keys_of(replay)) == {}

    @pytest.mark.asyncio
    async def test_an_unreadable_registry_is_an_outcome_not_an_exception(self, tmp_path):
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(Registry(db_path), record, flake_suppressed=False, evidence=_evidence())

        assert outcome.recorded == ()
        assert str(db_path) in outcome.reason


def _keys_of(replay: ReplayPlan) -> list[UnitKey]:
    return [decided.key for decided in replay.modules.values() if decided.key is not None]


class TestConsumeReplayed:
    @pytest.mark.asyncio
    async def test_the_live_entries_of_every_hit_are_consumed_for_the_landing(self, registry):
        keys_ = {unit_id: _key(unit_id) for unit_id in ('module:alpha:test', 'module:alpha:lint')}
        entries = {unit_id: await _entry(registry, key) for unit_id, key in keys_.items()}
        replay = _replay({
            unit_id: _decided(Decision.HIT, entries[unit_id], key=key) for unit_id, key in keys_.items()
        })
        _units, record = replay.settle((), ())

        outcome = await consume_replayed(registry, record, landing_sha='land-1')

        assert outcome.backs_every_hit
        assert set(await registry.consumed_for('land-1')) == set(entries.values())

    @pytest.mark.asyncio
    async def test_an_evicted_hit_is_unbacked_and_nothing_is_consumed(self, registry):
        keys_ = {unit_id: _key(unit_id) for unit_id in ('module:alpha:test', 'module:alpha:lint')}
        entries = {unit_id: await _entry(registry, key) for unit_id, key in keys_.items()}
        replay = _replay({
            unit_id: _decided(Decision.HIT, entries[unit_id], key=key) for unit_id, key in keys_.items()
        })
        _units, record = replay.settle((), ())
        await registry.evict_key(keys_['module:alpha:lint'].key, 'test eviction')

        outcome = await consume_replayed(registry, record, landing_sha='land-1')

        assert not outcome.backs_every_hit
        assert outcome.unbacked == ('module:alpha:lint',)
        assert await registry.consumed_for('land-1') == ()

    @pytest.mark.asyncio
    async def test_an_unreadable_registry_consumes_nothing(self, registry, tmp_path):
        key = _key('module:alpha:test')
        entry = await _entry(registry, key)
        _units, record = _replay({'module:alpha:test': _decided(Decision.HIT, entry, key=key)}).settle((), ())
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)

        outcome = await consume_replayed(Registry(db_path), record, landing_sha='land-1')

        assert not outcome.backs_every_hit
        assert str(db_path) in outcome.reason

    @pytest.mark.asyncio
    async def test_a_record_without_a_hit_is_a_no_op(self, tmp_path):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)
        db_path = tmp_path / 'untouched.db'

        outcome = await consume_replayed(Registry(db_path), record, landing_sha='land-1')

        assert outcome.backs_every_hit
        assert outcome.consumed == ()
        assert not db_path.exists()


# The task role records its greens where its workflow_verify row is emitted (A3).

_TASK_LOG = VerifyCacheConfig(task_mode=CacheMode.LOG)
_TASK_UNIT = 'module:alpha:test'


def _task_green() -> tuple[VerifyResult, UnitKey]:
    """A passing task-role verdict whose one test leg ran clean under a keyed, admitted unit."""
    key = _key(_TASK_UNIT)
    replay = ReplayPlan(
        role='task', modes=_TASK_LOG,
        tree=_decided(Decision.MISS, key=_key('tree', tier=UnitTier.TREE)),
        modules={_TASK_UNIT: _decided(Decision.MISS, key=key)},
    )
    units, record = replay.settle((_ran(_TASK_UNIT),), (LegRun(_TASK_UNIT, _clean(Leg.TEST)),))
    return _result(units=units, cache_record=record, duration_secs=1.0), key


class TestWorkflowRecordsTaskGreens:
    @staticmethod
    async def _scene(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, forked: bool = False,
    ) -> tuple[OrchestratorConfig, EventStore, TaskWorkflow]:
        config = task_scene_config(tmp_path)
        config.pre_merge_reverify = forked
        store = EventStore(tmp_path / 'events.db', run_id='r')
        scene = forked_task_workflow if forked else task_workflow
        workflow = await scene(config, GitOps(config.git, config.project_root), store, monkeypatch)
        monkeypatch.setattr(workflow, '_review', AsyncMock(return_value=ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={'analyst': {}},
        )))
        return config, store, workflow

    @staticmethod
    def _verifies(monkeypatch: pytest.MonkeyPatch, result: VerifyResult) -> AsyncMock:
        verify = AsyncMock(return_value=result)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)
        return verify

    @staticmethod
    async def _recorded_at(config: OrchestratorConfig, key: UnitKey) -> Entry:
        return (await _recorded(Registry(registry_db_path(config.project_root)), key))[_TASK_UNIT]

    @pytest.mark.asyncio
    async def test_the_implement_site_records_its_green_against_the_review_edge_row(self, tmp_path, monkeypatch):
        config, store, workflow = await self._scene(tmp_path, monkeypatch)
        result, key = _task_green()
        self._verifies(monkeypatch, result)

        assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        entry = await self._recorded_at(config, key)
        assert entry.evidence_event_id == row['id']
        assert (entry.run_id, entry.sha) == ('r', row['data']['tip_sha'])

    @pytest.mark.asyncio
    async def test_the_pre_merge_site_records_its_green_and_verifies_through_the_dispatchers_registry(
        self, tmp_path, monkeypatch,
    ):
        config, store, workflow = await self._scene(tmp_path, monkeypatch, forked=True)
        result, key = _task_green()
        verify = self._verifies(monkeypatch, result)

        assert await workflow._rebase_before_enqueue() is True

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        entry = await self._recorded_at(config, key)
        assert entry.evidence_event_id == row['id']
        assert (entry.run_id, entry.sha) == ('r', row['data']['tip_sha'])
        assert verify.await_args is not None
        access = verify.await_args.kwargs['verify_cache']
        assert isinstance(access, CacheAccess)
        assert access.registry is not None
        assert (await access.registry.summary()).path == registry_db_path(config.project_root)

    @pytest.mark.asyncio
    async def test_a_verdict_without_a_cache_record_records_nothing(self, tmp_path, monkeypatch):
        config, store, workflow = await self._scene(tmp_path, monkeypatch)
        self._verifies(monkeypatch, _result(duration_secs=1.0))

        assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

        assert len(store.fetch_events_by_type(EventType.workflow_verify)) == 1
        assert not registry_db_path(config.project_root).exists()

    @pytest.mark.asyncio
    async def test_an_unresolved_tip_records_nothing(self, tmp_path, monkeypatch):
        config, store, workflow = await self._scene(tmp_path, monkeypatch)
        self._verifies(monkeypatch, _task_green()[0])
        monkeypatch.setattr(workflow, '_verified_tip', AsyncMock(return_value=TipUnresolved.HEAD_UNREADABLE))

        assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

        [row] = store.fetch_events_by_type(EventType.workflow_verify)
        assert row['data']['tip_sha_unresolved'] is True
        assert not registry_db_path(config.project_root).exists()

    @pytest.mark.asyncio
    async def test_a_checkpoint_hit_records_nothing(self, tmp_path, monkeypatch):
        config, store, workflow = await self._scene(tmp_path, monkeypatch)
        assert workflow.worktree is not None
        head = scene_git(workflow.worktree, 'rev-parse', 'HEAD')
        EventStore(tmp_path / 'events.db', run_id='prior').emit(
            EventType.workflow_verify, task_id=SCENE_TASK_ID, data={'passed': True, 'tip_sha': head},
        )
        verify = self._verifies(monkeypatch, _task_green()[0])

        assert await workflow._execute_verify_review_loop() == WorkflowOutcome.DONE

        verify.assert_not_awaited()
        assert store.fetch_events_by_type(EventType.workflow_verify) == []
        assert not registry_db_path(config.project_root).exists()
