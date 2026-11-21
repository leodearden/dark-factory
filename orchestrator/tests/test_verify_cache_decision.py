"""The verify-result cache decision: which planned units replay a recorded green (task 5653)."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan, plan_replay

from orchestrator import verify_plan
from orchestrator.config import (
    ModuleConfig,
    ModuleVerifyCache,
    OrchestratorConfig,
    VerifyCacheConfig,
)
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.registry import Entry, Registry
from orchestrator.verify_evidence import Leg, module_leg_name
from orchestrator.verify_plan import VerifyPlan

_SEED = RepoSeed(
    files=(
        ('.gitignore', '.df-verify-junit/\n.task/\n.pytest_cache/\n__pycache__/\n'),
        ('pyproject.toml', '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'),
        ('alpha/pyproject.toml', '[project]\nname = "alpha"\nversion = "0"\n'),
        ('alpha/src/alpha/__init__.py', 'A = 1\n'),
        ('alpha/tests/test_alpha.py', 'def test_alpha():\n    pass\n'),
        ('beta/pyproject.toml', '[project]\nname = "beta"\nversion = "0"\ndependencies = ["alpha>=0"]\n'),
        ('beta/src/beta/__init__.py', 'B = 1\n'),
        ('beta/tests/test_beta.py', 'def test_beta():\n    pass\n'),
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
    repo: Path, modes: VerifyCacheConfig = _ON, *, test_command: str = 'pytest', lint_command: str = 'ruff check',
) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo, merge_verify_workspace=True, merge_verify_breadth='full',
        verify_cold_preprovision_command='', verify_cache=modes,
        test_command=test_command, lint_command=lint_command,
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
