"""Which planned units of a verify replay a recorded green, and how a verify's units settle against that decision (task 5653)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _verify_cache_fixtures import (
    CODE_D1,
    CODE_D2,
    LOG_MODES,
    ON_MODES,
    TEST_CODE,
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    commit_edit,
    deployed_at,
    full_suite_plan,
    module_units,
)
from _verify_cache_units import (
    MERGE_ON,
    clean_leg,
    merge_replay,
    ran_unit,
    record_entry,
    unit_decision,
    unit_key,
)

from orchestrator import deployed_sha, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.verify import MERGE_JUNIT_DIR
from orchestrator.verify_cache.keys import TREE_UNIT_ID, UnitKey
from orchestrator.verify_cache.modes import VerifyRole
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import LegRun
from orchestrator.verify_cache.registry import Entry, Registry, registry_db_path
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan, plan_replay
from orchestrator.verify_evidence import Leg, UnitOutcome, UnitTier, VerifyUnit
from orchestrator.verify_plan import VerifyPlan


def _no_env(_mc: ModuleConfig | None) -> Mapping[str, str]:
    return {}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', WORKSPACE_SEED)


@pytest.fixture
def markers(tmp_path: Path) -> Path:
    return tmp_path / 'markers.txt'


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


@pytest.fixture
def access(registry: Registry) -> CacheAccess:
    return CacheAccess(registry=registry, orchestrator_code=TEST_CODE)


@pytest.fixture
def modules(markers: Path) -> list[ModuleConfig]:
    return [cache_module('alpha', markers), cache_module('beta', markers)]


async def _replay(
    repo: Path,
    modules: Sequence[ModuleConfig | None],
    access: CacheAccess,
    *,
    modes: VerifyCacheConfig = ON_MODES,
    role: VerifyRole = 'merge',
    plan: VerifyPlan | None = None,
    config: OrchestratorConfig | None = None,
) -> ReplayPlan:
    if plan is None and all(mc is not None for mc in modules):
        plan = full_suite_plan([mc for mc in modules if mc is not None], role)
    replay = await plan_replay(
        repo, plan, modules, config=config or cache_config(repo, modes), role=role, access=access,
        leg_env=_no_env,
    )
    assert replay is not None
    return replay


async def _record(registry: Registry, replay: ReplayPlan, unit_ids: set[str]) -> dict[str, Entry]:
    recorded = {}
    for unit_id in sorted(unit_ids):
        key = replay.modules[unit_id].key
        assert key is not None
        recorded[unit_id] = await registry.record_green(key, evidence_event_id=7, run_id='r', sha='s')
    return recorded


async def _record_tree(registry: Registry, replay: ReplayPlan) -> Entry:
    assert replay.tree.key is not None
    return await registry.record_green(replay.tree.key, evidence_event_id=9, run_id='r', sha='s')


def _decisions(replay: ReplayPlan, unit_ids: set[str]) -> set[Decision]:
    return {replay.modules[unit_id].decision.decision for unit_id in unit_ids}


class TestPlanReplay:
    @pytest.mark.asyncio
    async def test_row_1_an_empty_registry_misses_every_unit_with_a_key(self, repo, modules, access):
        replay = await _replay(repo, modules, access)

        assert set(replay.modules) == module_units('alpha') | module_units('beta')
        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.MISS
            assert isinstance(decided.key, UnitKey)
            assert decided.decision.reason
        assert replay.tree.decision.decision is Decision.MISS
        assert replay.tree.key is not None
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_row_2_an_edit_outside_alpha_replays_alpha(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), module_units('alpha'))
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        for unit_id in module_units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert _decisions(replay, module_units('beta')) == {Decision.MISS}
        assert replay.replayed_legs('alpha') == frozenset(Leg)
        assert replay.replayed_legs('beta') == frozenset()

    @pytest.mark.asyncio
    async def test_row_3_an_edit_to_alpha_misses_alpha_and_its_dependent(
        self, repo, modules, access, registry,
    ):
        first = await _replay(repo, modules, access)
        await _record(registry, first, module_units('alpha') | module_units('beta'))
        commit_edit(repo, 'alpha/src/alpha/__init__.py', 'A = 2\n')

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, module_units('alpha') | module_units('beta')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_6_a_task_role_file_scoped_green_does_not_serve_the_merge_plan(
        self, repo, modules, access, registry,
    ):
        alpha = modules[0]
        task_plan = verify_plan.derive_verify_plan(
            ['alpha/src/alpha/__init__.py'], [alpha], cache_config(repo),
            lambda path: (repo / path).read_text(), role='task',
        )
        task_replay = await _replay(repo, [alpha], access, role='task', plan=task_plan)
        keyed = {unit_id for unit_id, decided in task_replay.modules.items() if decided.key is not None}
        assert keyed
        await _record(registry, task_replay, keyed)

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, module_units('alpha')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_9_log_mode_would_hit_with_the_entry(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), module_units('alpha'))

        replay = await _replay(repo, modules, access, modes=LOG_MODES)

        for unit_id in module_units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.WOULD_HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert replay.replayed_legs('alpha') == frozenset()

    @pytest.mark.asyncio
    async def test_row_10_an_unreadable_registry_is_unavailable_naming_it(self, repo, modules, tmp_path):
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=TEST_CODE)

        replay = await _replay(repo, modules, access)

        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert str(db_path) in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_runner_consults_no_registry(self, repo, modules):
        replay = await _replay(repo, modules, await CacheAccess.on_runner())

        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert 'no_registry_on_runner' in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_working_copy_that_differs_from_head_is_unavailable(self, repo, modules, access):
        (repo / 'stray.py').write_text('x = 1\n')

        replay = await _replay(repo, modules, access)

        assert set(replay.modules) == module_units('alpha') | module_units('beta')
        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert '1 path' in decided.decision.reason

    @pytest.mark.parametrize(('stray', 'decisions'), [
        pytest.param((), {Decision.MISS}, id='only-own-outputs'),
        pytest.param(('stray.py',), {Decision.UNAVAILABLE}, id='own-outputs-and-a-stray'),
    ])
    @pytest.mark.asyncio
    async def test_the_verify_s_own_outputs_are_not_a_working_copy_change(
        self, repo, modules, access, stray, decisions,
    ):
        (repo / MERGE_JUNIT_DIR).mkdir()
        (repo / MERGE_JUNIT_DIR / 'report.alpha.xml').write_text('<testsuites/>')
        for path in stray:
            (repo / path).write_text('x = 1\n')

        replay = await plan_replay(
            repo, full_suite_plan(modules), modules, config=cache_config(repo), role='merge', access=access,
            leg_env=_no_env, verify_outputs=(MERGE_JUNIT_DIR,),
        )

        assert replay is not None
        decided = (*replay.modules.values(), replay.tree)
        assert {unit.decision.decision for unit in decided} == decisions
        if stray:
            assert {'1 path' in unit.decision.reason for unit in decided} == {True}

    @pytest.mark.asyncio
    async def test_mode_off_decides_nothing_and_touches_no_registry(self, repo, modules, tmp_path):
        db_path = tmp_path / 'never_created.db'
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=TEST_CODE)

        replay = await plan_replay(
            repo, full_suite_plan(modules), modules, config=cache_config(repo, VerifyCacheConfig()),
            role='merge', access=access, leg_env=_no_env,
        )

        assert replay is None
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_a_tree_green_replays_every_module_unit(self, repo, modules, access, registry):
        tree_entry = await _record_tree(registry, await _replay(repo, modules, access))

        replay = await _replay(repo, modules, access)

        assert replay.tree.decision.decision is Decision.HIT
        assert replay.tree_hit
        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.HIT
            assert decided.decision.entry == tree_entry

    @pytest.mark.asyncio
    async def test_a_never_cache_module_misses_and_keeps_the_tree_from_hitting(
        self, repo, markers, access, registry,
    ):
        never = [cache_module('alpha', markers, verify_cache='never'), cache_module('beta', markers)]
        first = await _replay(repo, never, access)
        await _record(registry, first, module_units('alpha') | module_units('beta'))
        await _record_tree(registry, first)

        replay = await _replay(repo, never, access)

        for unit_id in module_units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.MISS
            assert 'never-cache' in replay.modules[unit_id].decision.reason
        assert replay.tree.decision.decision is not Decision.HIT
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_a_plan_less_run_keys_its_tree_on_the_executed_commands(self, repo, access, markers):
        config = cache_config(repo, test_command='pytest -q', lint_command=f'echo lint >> {markers}')
        other = cache_config(repo, test_command='pytest -q -x', lint_command=f'echo lint >> {markers}')

        replay = await _replay(repo, [None], access, config=config)
        changed = await _replay(repo, [None], access, config=other)

        assert all(decided.key is None for decided in replay.modules.values())
        assert all('not keyed' in decided.decision.reason for decided in replay.modules.values())
        assert replay.tree.key is not None
        assert replay.tree.decision.decision is Decision.MISS
        assert changed.tree.key is not None
        assert changed.tree.key.key != replay.tree.key.key

    @pytest.mark.asyncio
    async def test_without_replayed_drops_hit_legs_and_nothing_left_is_none(
        self, repo, modules, access, registry,
    ):
        alpha, beta = modules
        await _record(registry, await _replay(repo, modules, access), module_units('alpha'))
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        assert replay.without_replayed(alpha) is None
        assert replay.without_replayed(beta) == beta


# ---------------------------------------------------------------------------
# The key's code identity: the process's DeployedSha, never wherever the code
# checkout's HEAD has moved since.


def _keys(replay: ReplayPlan) -> dict[str, UnitKey | None]:
    return {TREE_UNIT_ID: replay.tree.key, **{unit_id: decided.key for unit_id, decided in replay.modules.items()}}


class TestCodeIdentity:
    def test_the_dispatcher_access_carries_its_deployed_sha(self, repo):
        assert CacheAccess.on_dispatcher(repo, deployed=deployed_at(CODE_D1)).orchestrator_code == CODE_D1

    @pytest.mark.asyncio
    async def test_keys_follow_the_injected_identity(self, repo, modules, registry):
        d1 = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=CODE_D1)))
        d1_again = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=CODE_D1)))
        d2 = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=CODE_D2)))

        assert set(d1) == set(d2) == {TREE_UNIT_ID} | module_units('alpha') | module_units('beta')
        assert all(key is not None for key in (*d1.values(), *d2.values()))
        assert d1 == d1_again
        for unit_id, key in d1.items():
            assert key != d2[unit_id], unit_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('deployed', 'reason'), [
        pytest.param(None, ['no deployed sha'], id='none'),
        pytest.param(deployed_at(indeterminate=True), ['indeterminate', CODE_D1], id='indeterminate'),
        pytest.param(deployed_at(dirty_wire_path=True), ['uncommitted', CODE_D1], id='dirty-wire-path'),
    ])
    async def test_a_deployed_sha_that_cannot_name_the_code_makes_every_unit_unavailable(
        self, tmp_path, repo, modules, deployed, reason,
    ):
        elsewhere = tmp_path / 'elsewhere'
        access = CacheAccess.on_dispatcher(elsewhere, deployed=deployed)

        replay = await _replay(repo, modules, access)

        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            for fragment in reason:
                assert fragment in decided.decision.reason
        assert not registry_db_path(elsewhere).exists()
        assert not elsewhere.exists()

    @pytest.mark.asyncio
    async def test_a_runner_keys_on_its_pinned_gate_checkout(self):
        access = await CacheAccess.on_runner()

        assert access.registry is None
        checkout = deployed_sha.resolve_code_checkout()
        assert checkout is not None
        assert access.orchestrator_code == deployed_sha.checkout_head(checkout)


# ---------------------------------------------------------------------------
# Settling a verify's units against its replay plan.


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
        key = unit_key('module:alpha:test')
        entry = await record_entry(registry, key) if decision is Decision.WOULD_HIT else None
        replay = merge_replay({'module:alpha:test': unit_decision(decision, entry, key=key)})
        leg = LegRun('module:alpha:test', clean_leg(Leg.TEST, rc=rc))

        units, record = replay.settle((ran_unit('module:alpha:test', rc),), (leg,))

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
        key = unit_key('module:alpha:lint')
        entry = await record_entry(registry, key, evidence=41)
        replay = merge_replay({'module:alpha:lint': unit_decision(Decision.HIT, entry, key=key)})

        units, record = replay.settle((), ())

        assert units[0] == VerifyUnit(
            unit='module:alpha:lint', tier=UnitTier.MODULE, key=key.key, outcome=UnitOutcome.CACHE_HIT,
            evidence_event_id=41, reason=replay.modules['module:alpha:lint'].decision.reason,
        )
        assert record.units[0].entry == entry
        assert record.units[0].leg is None

    def test_a_unit_outside_the_plan_passes_through_unkeyed(self):
        replay = merge_replay({})
        leg = LegRun('module:gamma:test', clean_leg(Leg.TEST))

        units, record = replay.settle((ran_unit('module:gamma:test'),), (leg,))

        assert units[0].outcome is UnitOutcome.RAN_GREEN
        assert units[0].key is None
        assert units[0].reason is not None
        assert units[0].reason.startswith('not keyed: ')
        assert record.units[0].leg == leg.leg

    def test_the_tree_unit_settles_last_and_records_its_own_key(self):
        tree_key = unit_key('tree', tier=UnitTier.TREE)
        replay = merge_replay({}, tree=unit_decision(Decision.MISS, key=tree_key))

        units, record = replay.settle((ran_unit('module:gamma:test'),), (LegRun('module:gamma:test', clean_leg(Leg.TEST)),))

        assert units[-1].unit == 'tree'
        assert units[-1].tier is UnitTier.TREE
        assert units[-1].outcome is UnitOutcome.RAN_GREEN
        assert record.tree is not None
        assert record.tree.key == tree_key
        assert (record.role, record.modes) == ('merge', MERGE_ON)

    def test_a_red_unit_makes_the_tree_red(self):
        replay = merge_replay({}, tree=unit_decision(Decision.MISS, key=unit_key('tree', tier=UnitTier.TREE)))

        units, _record = replay.settle((ran_unit('module:gamma:test', 1),), ())

        assert units[-1].outcome is UnitOutcome.RAN_RED

