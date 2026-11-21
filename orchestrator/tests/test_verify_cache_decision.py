"""The verify-result cache decision: which planned units replay a recorded green (task 5653)."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling
from test_verify_cache_recorder import put_pytest_on_path

from orchestrator import verify_plan
from orchestrator.config import (
    ModuleConfig,
    ModuleVerifyCache,
    OrchestratorConfig,
    VerifyCacheConfig,
)
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_cache.keys import TREE_UNIT_ID, UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import VerifyEvidenceRef, record_greens
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan, plan_replay
from orchestrator.verify_evidence import (
    Leg,
    UnitOutcome,
    VerifyUnit,
    module_leg_name,
    module_leg_unit,
)
from orchestrator.verify_plan import VerifyPlan

#: Every leg of the fixture appends ``<module>:<leg>`` to the file this names.
_MARKERS_ENV = 'DF_VERIFY_CACHE_TEST_MARKERS'


def _marking_test(prefix: str) -> str:
    return (
        f'import os\n\n\ndef test_{prefix}():\n'
        f"    with open(os.environ['{_MARKERS_ENV}'], 'a') as markers:\n"
        f"        markers.write('{prefix}:test\\n')\n"
    )


_SEED = RepoSeed(
    files=(
        ('.gitignore', '.df-verify-junit/\n.task/\n.pytest_cache/\n__pycache__/\n'),
        ('pyproject.toml', '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'),
        ('alpha/pyproject.toml', '[project]\nname = "alpha"\nversion = "0"\n'),
        ('alpha/src/alpha/__init__.py', 'A = 1\n'),
        ('alpha/tests/test_alpha.py', _marking_test('alpha')),
        ('beta/pyproject.toml', '[project]\nname = "beta"\nversion = "0"\ndependencies = ["alpha>=0"]\n'),
        ('beta/src/beta/__init__.py', 'B = 1\n'),
        ('beta/tests/test_beta.py', _marking_test('beta')),
    ),
    message='verify cache decision fixture',
)

_ON = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)
_LOG = VerifyCacheConfig(task_mode=CacheMode.LOG, merge_mode=CacheMode.LOG)
_CODE = 'c' * 40


def _units(prefix: str) -> set[str]:
    return {module_leg_name(prefix, leg) for leg in Leg}


def _module(prefix: str, markers: Path, *, verify_cache: ModuleVerifyCache = 'default') -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix,
        test_command=f'pytest {prefix}/tests -q',
        lint_command=f'echo {prefix}:lint >> {markers}',
        type_check_command=f'echo {prefix}:type >> {markers}',
        verify_cache=verify_cache,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args], cwd=repo,
        env=git_env_with_ceiling(repo), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_edit(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', f'edit {path}')


def _config(
    repo: Path,
    modes: VerifyCacheConfig = _ON,
    *,
    test_command: str = 'pytest',
    lint_command: str = 'ruff check',
    type_check_command: str = 'pyright',
) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo, merge_verify_workspace=True, merge_verify_breadth='full',
        verify_cold_preprovision_command='', verify_cache=modes,
        test_command=test_command, lint_command=lint_command, type_check_command=type_check_command,
    )


def _full_suite_plan(modules: Sequence[ModuleConfig], role: VerifyRole = 'merge') -> VerifyPlan:
    plan_role: Literal['merge', 'task'] = 'task' if role == 'task' else 'merge'
    return VerifyPlan(runs=tuple(
        run for mc in modules for run in verify_plan._derive_full_suite_runs(mc, plan_role)
    ))


def _no_env(_mc: ModuleConfig | None) -> Mapping[str, str]:
    return {}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', _SEED)


@pytest.fixture
def markers(tmp_path: Path) -> Path:
    return tmp_path / 'markers.txt'


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


@pytest.fixture
def access(registry: Registry) -> CacheAccess:
    return CacheAccess(registry=registry, orchestrator_code=_CODE)


@pytest.fixture
def modules(markers: Path) -> list[ModuleConfig]:
    return [_module('alpha', markers), _module('beta', markers)]


async def _replay(
    repo: Path,
    modules: Sequence[ModuleConfig | None],
    access: CacheAccess,
    *,
    modes: VerifyCacheConfig = _ON,
    role: VerifyRole = 'merge',
    plan: VerifyPlan | None = None,
    config: OrchestratorConfig | None = None,
) -> ReplayPlan:
    if plan is None and all(mc is not None for mc in modules):
        plan = _full_suite_plan([mc for mc in modules if mc is not None], role)
    replay = await plan_replay(
        repo, plan, modules, config=config or _config(repo, modes), role=role, access=access,
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

        assert set(replay.modules) == _units('alpha') | _units('beta')
        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.MISS
            assert isinstance(decided.key, UnitKey)
            assert decided.decision.reason
        assert replay.tree.decision.decision is Decision.MISS
        assert replay.tree.key is not None
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_row_2_an_edit_outside_alpha_replays_alpha(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), _units('alpha'))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert _decisions(replay, _units('beta')) == {Decision.MISS}
        assert replay.replayed_legs('alpha') == frozenset(Leg)
        assert replay.replayed_legs('beta') == frozenset()

    @pytest.mark.asyncio
    async def test_row_3_an_edit_to_alpha_misses_alpha_and_its_dependent(
        self, repo, modules, access, registry,
    ):
        first = await _replay(repo, modules, access)
        await _record(registry, first, _units('alpha') | _units('beta'))
        _commit_edit(repo, 'alpha/src/alpha/__init__.py', 'A = 2\n')

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, _units('alpha') | _units('beta')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_6_a_task_role_file_scoped_green_does_not_serve_the_merge_plan(
        self, repo, modules, access, registry,
    ):
        alpha = modules[0]
        task_plan = verify_plan.derive_verify_plan(
            ['alpha/src/alpha/__init__.py'], [alpha], _config(repo),
            lambda path: (repo / path).read_text(), role='task',
        )
        task_replay = await _replay(repo, [alpha], access, role='task', plan=task_plan)
        keyed = {unit_id for unit_id, decided in task_replay.modules.items() if decided.key is not None}
        assert keyed
        await _record(registry, task_replay, keyed)

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, _units('alpha')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_9_log_mode_would_hit_with_the_entry(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), _units('alpha'))

        replay = await _replay(repo, modules, access, modes=_LOG)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.WOULD_HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert replay.replayed_legs('alpha') == frozenset()

    @pytest.mark.asyncio
    async def test_row_10_an_unreadable_registry_is_unavailable_naming_it(self, repo, modules, tmp_path):
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=_CODE)

        replay = await _replay(repo, modules, access)

        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert str(db_path) in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_runner_consults_no_registry(self, repo, modules):
        replay = await _replay(repo, modules, CacheAccess.on_runner())

        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert 'no_registry_on_runner' in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_working_copy_that_differs_from_head_is_unavailable(self, repo, modules, access):
        (repo / 'stray.py').write_text('x = 1\n')

        replay = await _replay(repo, modules, access)

        assert set(replay.modules) == _units('alpha') | _units('beta')
        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert '1 path' in decided.decision.reason

    @pytest.mark.asyncio
    async def test_mode_off_decides_nothing_and_touches_no_registry(self, repo, modules, tmp_path):
        db_path = tmp_path / 'never_created.db'
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=_CODE)

        replay = await plan_replay(
            repo, _full_suite_plan(modules), modules, config=_config(repo, VerifyCacheConfig()),
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
        never = [_module('alpha', markers, verify_cache='never'), _module('beta', markers)]
        first = await _replay(repo, never, access)
        await _record(registry, first, _units('alpha') | _units('beta'))
        await _record_tree(registry, first)

        replay = await _replay(repo, never, access)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.MISS
            assert 'never-cache' in replay.modules[unit_id].decision.reason
        assert replay.tree.decision.decision is not Decision.HIT
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_a_plan_less_run_keys_its_tree_on_the_executed_commands(self, repo, access, markers):
        config = _config(repo, test_command='pytest -q', lint_command=f'echo lint >> {markers}')
        other = _config(repo, test_command='pytest -q -x', lint_command=f'echo lint >> {markers}')

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
        await _record(registry, await _replay(repo, modules, access), _units('alpha'))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        assert replay.without_replayed(alpha) is None
        assert replay.without_replayed(beta) == beta


# ---------------------------------------------------------------------------
# The decision at the real fan-out seam: run_scoped_verification's execution funnel.

_TASK_ON_MERGE_OFF = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.OFF)


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    """Real legs: a bare ``pytest`` whose junit the verify reads, and every leg leaves a marker."""
    put_pytest_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv(_MARKERS_ENV, str(markers))


def _legs(prefix: str) -> set[str]:
    return {f'{prefix}:{leg}' for leg in Leg}


def _ran_legs(markers: Path) -> set[str]:
    """The ``<module>:<leg>`` of every leg that executed since *markers* was last cleared."""
    if not markers.exists():
        return set()
    return {line.split()[0] for line in markers.read_text().splitlines() if line.strip()}


def _clear(markers: Path) -> None:
    markers.unlink(missing_ok=True)


async def _gate(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess | None,
) -> VerifyResult:
    return await run_scoped_verification(
        repo, config, list(modules), role='merge', is_merge_verify=True, force_workspace=True,
        verify_cache=access,
    )


async def _task_verify(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess,
    task_files: list[str],
) -> VerifyResult:
    return await run_scoped_verification(
        repo, config, list(modules), task_files, role='task', verify_cache=access,
    )


async def _record_run(registry: Registry, result: VerifyResult, *, event_id: int = 41) -> dict[str, Entry]:
    assert result.passed, result.summary
    assert result.cache_record is not None
    outcome = await record_greens(
        registry, result.cache_record, flake_suppressed=False,
        evidence=VerifyEvidenceRef(event_id, 'r', 'sha-first', True),
    )
    return {entry.unit_id: entry for entry in outcome.recorded}


def _by_unit(result: VerifyResult) -> dict[str, VerifyUnit]:
    assert result.units is not None
    return {unit.unit: unit for unit in result.units}


def _outcomes(result: VerifyResult, unit_ids: set[str]) -> set[UnitOutcome]:
    units = _by_unit(result)
    return {units[unit_id].outcome for unit_id in unit_ids}


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
        config = _config(repo)
        first = await _gate(repo, config, modules, access)
        recorded = await _record_run(registry, first)
        assert _units('alpha') <= set(recorded)
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == _legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.ON
        assert result.plan == first.plan == _full_suite_plan(modules).to_dict()
        units = _by_unit(result)
        for unit_id in _units('alpha'):
            assert units[unit_id].outcome is UnitOutcome.CACHE_HIT
            assert units[unit_id].evidence_event_id == 41
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert _outcomes(result, _units('beta')) == {UnitOutcome.RAN_GREEN}

    @pytest.mark.asyncio
    async def test_a_wholly_replayed_module_reads_as_a_cache_hit_not_a_run(
        self, repo, modules, access, registry, markers,
    ):
        config = _config(repo)
        recorded = await _record_run(registry, await _gate(repo, config, modules, access))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        result = await _gate(repo, config, modules[:1], access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        assert result.category == 'cache_hit'
        assert result.duration_secs == 0.0
        assert (result.test_output, result.lint_output, result.type_output) == ('', '', '')
        for unit_id in _units('alpha'):
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert _outcomes(result, _units('alpha')) == {UnitOutcome.CACHE_HIT}

    @pytest.mark.asyncio
    async def test_a_recorded_tree_green_runs_no_leg_at_all(self, repo, modules, access, registry, markers):
        config = _config(repo)
        first = await _gate(repo, config, modules, access)
        assert TREE_UNIT_ID in await _record_run(registry, first)
        tree_entry = await registry.record_green(_tree_key(first), evidence_event_id=99, run_id='r', sha='sha-tree')
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        units = _by_unit(result)
        assert set(units) == _units('alpha') | _units('beta') | {TREE_UNIT_ID}
        for unit in units.values():
            assert unit.outcome is UnitOutcome.CACHE_HIT
            assert unit.evidence_event_id == tree_entry.evidence_event_id == 99

    @pytest.mark.asyncio
    async def test_log_mode_runs_every_leg_and_reads_would_hit(self, repo, modules, access, registry, markers):
        config = _config(repo, _LOG)
        await _record_run(registry, await _gate(repo, config, modules, access))
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == _legs('alpha') | _legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.LOG
        assert {unit.outcome for unit in _by_unit(result).values()} == {UnitOutcome.WOULD_HIT}

    @pytest.mark.asyncio
    async def test_outside_the_cache_the_verify_is_as_before(self, repo, modules, tmp_path, markers):
        result = await _gate(repo, _config(repo, _LOG), modules, None)

        assert _ran_legs(markers) == _legs('alpha') | _legs('beta')
        assert result.passed, result.summary
        expected = sorted(
            (module_leg_unit(prefix, leg, 0) for prefix in ('alpha', 'beta') for leg in Leg),
            key=lambda unit: unit.unit,
        )
        assert sorted(result.units or (), key=lambda unit: unit.unit) == expected
        assert {unit.reason for unit in expected} == {None}
        assert result.cache_mode is CacheMode.OFF
        assert result.plan == _full_suite_plan(modules).to_dict()
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_each_role_reads_its_own_mode(self, repo, modules, access, registry, markers):
        (repo / '.task').mkdir()
        config = _config(repo, _TASK_ON_MERGE_OFF)
        task_files = ['alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py']
        recorded = await _record_run(registry, await _task_verify(repo, config, modules, access, task_files))
        alpha_units = {unit_id for unit_id in recorded if unit_id.startswith('module:alpha:')}
        assert alpha_units
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        task = await _task_verify(repo, config, modules, access, task_files)

        assert task.passed, task.summary
        assert task.cache_mode is CacheMode.ON
        assert _outcomes(task, alpha_units) == {UnitOutcome.CACHE_HIT}
        assert not {leg for leg in _ran_legs(markers) if leg.startswith('alpha:')}
        _clear(markers)

        merge = await _gate(repo, config, modules, access)

        assert merge.passed, merge.summary
        assert merge.cache_mode is CacheMode.OFF
        assert _legs('alpha') <= _ran_legs(markers)
        assert UnitOutcome.CACHE_HIT not in {unit.outcome for unit in _by_unit(merge).values()}

    @pytest.mark.asyncio
    async def test_a_plan_less_gate_replays_its_recorded_tree_green(self, repo, access, registry, markers):
        config = _config(
            repo, test_command='pytest alpha/tests beta/tests -q',
            lint_command=f'echo global:lint >> {markers}', type_check_command=f'echo global:type >> {markers}',
        )
        first = await _gate(repo, config, [], access)
        assert first.passed, first.summary
        assert first.plan is None
        await registry.record_green(_tree_key(first), evidence_event_id=77, run_id='r', sha='sha-first')
        _clear(markers)

        result = await _gate(repo, config, [], access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        units = _by_unit(result)
        assert (units[TREE_UNIT_ID].outcome, units[TREE_UNIT_ID].evidence_event_id) == (UnitOutcome.CACHE_HIT, 77)
        assert {unit.outcome for unit in units.values()} == {UnitOutcome.CACHE_HIT}
