"""The verify-result cache decision at run_scoped_verification's fan-out seam: which legs run and how each unit reads (task 5653)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _verify_cache_fixtures import (
    CODE_D1,
    CODE_D2,
    LOG_MODES,
    TEST_CODE,
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    clear_markers,
    commit_edit,
    full_suite_plan,
    merge_gate,
    module_legs,
    module_units,
    outcomes_of,
    ran_legs,
    record_run,
    task_verify,
    units_by_id,
    use_real_legs,
)

from orchestrator.config import ModuleConfig, VerifyCacheConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_cache.keys import TREE_UNIT_ID, UnitKey
from orchestrator.verify_cache.modes import CacheMode
from orchestrator.verify_cache.recorder import VerifyEvidenceRef, record_greens
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.verify_evidence import Leg, UnitOutcome, module_leg_name, module_leg_unit


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


_TASK_ON_MERGE_OFF = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.OFF)


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    use_real_legs(tmp_path, monkeypatch, markers)


def _replay_hint(unit_id: str, entry: Entry) -> str:
    return f'replayed {unit_id} from entry {entry.entry_id} (evidence event {entry.evidence_event_id})'


def _tree_key(result: VerifyResult) -> UnitKey:
    assert result.cache_record is not None and result.cache_record.tree is not None
    key = result.cache_record.tree.key
    assert key is not None
    return key


@pytest.mark.usefixtures('seam')
class TestFanOutSeam:
    @pytest.mark.asyncio
    async def test_an_edit_outside_alpha_replays_alpha_and_runs_beta(
        self, repo, modules, access, registry, markers,
    ):
        config = cache_config(repo)
        first = await merge_gate(repo, config, modules, access)
        recorded = await record_run(registry, first)
        assert module_units('alpha') <= set(recorded)
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        clear_markers(markers)

        result = await merge_gate(repo, config, modules, access)

        assert ran_legs(markers) == module_legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.ON
        assert result.plan == first.plan == full_suite_plan(modules).to_dict()
        units = units_by_id(result)
        for unit_id in module_units('alpha'):
            assert units[unit_id].outcome is UnitOutcome.CACHE_HIT
            assert units[unit_id].evidence_event_id == 41
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert outcomes_of(result, module_units('beta')) == {UnitOutcome.RAN_GREEN}

    @pytest.mark.asyncio
    async def test_a_partly_replayed_module_runs_only_its_changed_leg_and_names_each_replay(
        self, repo, access, registry, markers,
    ):
        commit_edit(repo, 'lint/alpha.cfg', 'strict = true\n')
        alpha = dataclasses.replace(
            cache_module('alpha', markers), lint_command=f'echo alpha:lint >> {markers} && cat lint/alpha.cfg',
        )
        modules = [alpha, cache_module('beta', markers)]
        config = cache_config(repo)
        recorded = await record_run(registry, await merge_gate(repo, config, modules, access))
        commit_edit(repo, 'lint/alpha.cfg', 'strict = false\n')
        clear_markers(markers)

        result = await merge_gate(repo, config, modules, access)

        alpha_lint = module_leg_name('alpha', Leg.LINT)
        alpha_replayed = module_units('alpha') - {alpha_lint}
        assert ran_legs(markers) == {'alpha:lint'}
        assert result.passed, result.summary
        assert outcomes_of(result, {alpha_lint}) == {UnitOutcome.RAN_GREEN}
        assert outcomes_of(result, alpha_replayed | module_units('beta')) == {UnitOutcome.CACHE_HIT}
        for unit_id in alpha_replayed:
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint

    @pytest.mark.asyncio
    @pytest.mark.parametrize('alpha_partly', [False, True], ids=['alpha-wholly-replayed', 'alpha-partly-replayed'])
    async def test_a_red_verdict_keeps_the_cause_hint_a_cache_free_run_gives(
        self, repo, access, registry, markers, alpha_partly,
    ):
        commit_edit(repo, 'lint/alpha.cfg', 'strict = true\n')
        alpha = dataclasses.replace(
            cache_module('alpha', markers), lint_command=f'echo alpha:lint >> {markers} && cat lint/alpha.cfg',
        )
        modules = [alpha, cache_module('beta', markers)]
        config = cache_config(repo)
        await record_run(registry, await merge_gate(repo, config, modules, access))
        if alpha_partly:
            commit_edit(repo, 'lint/alpha.cfg', 'strict = false\n')
        commit_edit(repo, 'beta/tests/test_beta.py', 'def test_beta():\n    assert False\n')

        result = await merge_gate(repo, config, modules, access)
        cache_free = await merge_gate(repo, config, modules, None)

        assert not result.passed and not cache_free.passed
        assert UnitOutcome.CACHE_HIT in set(outcomes_of(result, module_units('alpha')))
        assert result.category == cache_free.category
        assert result.cause_hint == cache_free.cause_hint
        assert 'replayed' not in result.cause_hint

    @pytest.mark.asyncio
    async def test_a_wholly_replayed_module_reads_as_a_cache_hit_not_a_run(
        self, repo, modules, access, registry, markers,
    ):
        config = cache_config(repo)
        recorded = await record_run(registry, await merge_gate(repo, config, modules, access))
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        clear_markers(markers)

        result = await merge_gate(repo, config, modules[:1], access)

        assert ran_legs(markers) == set()
        assert result.passed, result.summary
        assert result.category == 'cache_hit'
        assert result.duration_secs == 0.0
        assert (result.test_output, result.lint_output, result.type_output) == ('', '', '')
        for unit_id in module_units('alpha'):
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert outcomes_of(result, module_units('alpha')) == {UnitOutcome.CACHE_HIT}

    @pytest.mark.asyncio
    async def test_a_recorded_tree_green_runs_no_leg_at_all(self, repo, modules, access, registry, markers):
        config = cache_config(repo)
        first = await merge_gate(repo, config, modules, access)
        assert TREE_UNIT_ID in await record_run(registry, first)
        tree_entry = await registry.record_green(_tree_key(first), evidence_event_id=99, run_id='r', sha='sha-tree')
        clear_markers(markers)

        result = await merge_gate(repo, config, modules, access)

        assert ran_legs(markers) == set()
        assert result.passed, result.summary
        units = units_by_id(result)
        assert set(units) == module_units('alpha') | module_units('beta') | {TREE_UNIT_ID}
        for unit in units.values():
            assert unit.outcome is UnitOutcome.CACHE_HIT
            assert unit.evidence_event_id == tree_entry.evidence_event_id == 99

    @pytest.mark.asyncio
    async def test_log_mode_runs_every_leg_and_reads_would_hit(self, repo, modules, access, registry, markers):
        config = cache_config(repo, LOG_MODES)
        await record_run(registry, await merge_gate(repo, config, modules, access))
        clear_markers(markers)

        result = await merge_gate(repo, config, modules, access)

        assert ran_legs(markers) == module_legs('alpha') | module_legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.LOG
        assert {unit.outcome for unit in units_by_id(result).values()} == {UnitOutcome.WOULD_HIT}

    @pytest.mark.asyncio
    async def test_outside_the_cache_the_verify_is_as_before(self, repo, modules, tmp_path, markers):
        result = await merge_gate(repo, cache_config(repo, LOG_MODES), modules, None)

        assert ran_legs(markers) == module_legs('alpha') | module_legs('beta')
        assert result.passed, result.summary
        expected = sorted(
            (module_leg_unit(prefix, leg, 0) for prefix in ('alpha', 'beta') for leg in Leg),
            key=lambda unit: unit.unit,
        )
        assert sorted(result.units or (), key=lambda unit: unit.unit) == expected
        assert {unit.reason for unit in expected} == {None}
        assert result.cache_mode is CacheMode.OFF
        assert result.cache_record is None
        assert result.plan == full_suite_plan(modules).to_dict()
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_each_role_reads_its_own_mode(self, repo, modules, access, registry, markers):
        (repo / '.task').mkdir()
        config = cache_config(repo, _TASK_ON_MERGE_OFF)
        task_files = ['alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py']
        recorded = await record_run(registry, await task_verify(repo, config, modules, access, task_files))
        alpha_units = {unit_id for unit_id in recorded if unit_id.startswith('module:alpha:')}
        assert alpha_units
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        clear_markers(markers)

        task = await task_verify(repo, config, modules, access, task_files)

        assert task.passed, task.summary
        assert task.cache_mode is CacheMode.ON
        assert outcomes_of(task, alpha_units) == {UnitOutcome.CACHE_HIT}
        assert not {leg for leg in ran_legs(markers) if leg.startswith('alpha:')}
        clear_markers(markers)

        merge = await merge_gate(repo, config, modules, access)

        assert merge.passed, merge.summary
        assert merge.cache_mode is CacheMode.OFF
        assert module_legs('alpha') <= ran_legs(markers)
        assert UnitOutcome.CACHE_HIT not in {unit.outcome for unit in units_by_id(merge).values()}

    @pytest.mark.asyncio
    async def test_a_plan_less_gate_replays_its_recorded_tree_green(self, repo, access, registry, markers):
        config = cache_config(
            repo, test_command='pytest alpha/tests beta/tests -q',
            lint_command=f'echo global:lint >> {markers}', type_check_command=f'echo global:type >> {markers}',
        )
        first = await merge_gate(repo, config, [], access)
        assert first.passed, first.summary
        assert first.plan is None
        await registry.record_green(_tree_key(first), evidence_event_id=77, run_id='r', sha='sha-first')
        clear_markers(markers)

        result = await merge_gate(repo, config, [], access)

        assert ran_legs(markers) == set()
        assert result.passed, result.summary
        units = units_by_id(result)
        assert (units[TREE_UNIT_ID].outcome, units[TREE_UNIT_ID].evidence_event_id) == (UnitOutcome.CACHE_HIT, 77)
        assert {unit.outcome for unit in units.values()} == {UnitOutcome.CACHE_HIT}


    @pytest.mark.asyncio
    async def test_a_plan_less_gate_whose_test_leg_is_not_pytest_never_records_its_tree(
        self, repo, access, registry, markers,
    ):
        config = cache_config(
            repo, test_command=f'echo global:test >> {markers}',
            lint_command=f'echo global:lint >> {markers}', type_check_command=f'echo global:type >> {markers}',
        )
        result = await merge_gate(repo, config, [], access)
        assert result.passed, result.summary
        assert result.cache_record is not None

        outcome = await record_greens(
            registry, result.cache_record, flake_suppressed=False,
            evidence=VerifyEvidenceRef(41, 'r', 'sha-first', True),
        )

        test_unit = module_leg_name(None, Leg.TEST)
        assert outcome.recorded == ()
        assert f'refused [{test_unit}: test counts unknown: junit_tests=None collected=None]' in outcome.reason
        assert f'tree: units neither admitted nor replayed: {test_unit}' in outcome.reason

    @pytest.mark.asyncio
    async def test_greens_recorded_under_one_deployed_sha_serve_only_that_sha(self, repo, modules, registry, markers):
        config = cache_config(repo)
        d1 = CacheAccess(registry=registry, orchestrator_code=CODE_D1)
        await record_run(registry, await merge_gate(repo, config, modules, d1))
        clear_markers(markers)

        replayed = await merge_gate(repo, config, modules, d1)

        assert ran_legs(markers) == set()
        assert outcomes_of(replayed, module_units('alpha') | module_units('beta')) == {UnitOutcome.CACHE_HIT}
        clear_markers(markers)

        rerun = await merge_gate(repo, config, modules, CacheAccess(registry=registry, orchestrator_code=CODE_D2))

        assert ran_legs(markers) == module_legs('alpha') | module_legs('beta')
        assert outcomes_of(rerun, module_units('alpha') | module_units('beta')) == {UnitOutcome.RAN_GREEN}
