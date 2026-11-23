"""The composition root keys its one dispatcher verify-cache access on its own DeployedSha (task 5653)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from _git_fixtures import seed_repo
from _verify_cache_fixtures import (
    CODE_D1,
    LOG_MODES,
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    commit_edit,
    deployed_at,
    module_units,
    record_run,
    task_verify,
    use_real_legs,
)

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.deployed_sha import DeployedSha
from orchestrator.harness import Harness
from orchestrator.verify_cache.keys import TREE_UNIT_ID
from orchestrator.verify_cache.registry import Registry, registry_db_path
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.verify_evidence import UnitOutcome, UnitTier, VerifyUnit


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
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    use_real_legs(tmp_path, monkeypatch, markers)


def _harness_over(config: OrchestratorConfig, deployed: DeployedSha | None) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config, deployed_sha=deployed)


@pytest.mark.usefixtures('seam', 'code_default_config')
class TestHarnessKeysTheCacheOnItsDeployedSha:
    _TASK_FILES: ClassVar[tuple[str, ...]] = ('alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py')

    @pytest.fixture
    def config(self, repo: Path) -> OrchestratorConfig:
        """The registry lives under the project's ignored ``data/``, so recording keeps the working copy at HEAD."""
        commit_edit(repo, '.gitignore', (repo / '.gitignore').read_text() + 'data/\n')
        (repo / '.task').mkdir()
        return cache_config(repo, LOG_MODES)

    async def _seed_d1_greens(self, repo: Path, config: OrchestratorConfig, modules: list[ModuleConfig]) -> set[str]:
        registry = Registry(registry_db_path(repo))
        access = CacheAccess(registry=registry, orchestrator_code=CODE_D1)
        recorded = await record_run(registry, await task_verify(repo, config, modules, access, list(self._TASK_FILES)))
        return set(recorded) - {TREE_UNIT_ID}

    async def _module_units(
        self, harness: Harness, repo: Path, config: OrchestratorConfig, modules: list[ModuleConfig],
    ) -> dict[str, VerifyUnit]:
        result = await harness._task_verify_dispatcher(
            repo, config, modules, list(self._TASK_FILES), attempt_id=1, task_id='5653', archive_root=None,
        )
        assert result.passed, result.summary
        return {unit.unit: unit for unit in result.units or () if unit.tier is UnitTier.MODULE}

    @pytest.mark.asyncio
    async def test_a_harness_deployed_at_d1_would_replay_the_d1_greens(self, repo, config, modules):
        seeded = await self._seed_d1_greens(repo, config, modules)

        units = await self._module_units(_harness_over(config, deployed_at(CODE_D1)), repo, config, modules)

        assert set(units) == seeded == module_units('alpha') | module_units('beta')
        assert {unit.outcome for unit in units.values()} == {UnitOutcome.WOULD_HIT}

    @pytest.mark.asyncio
    async def test_a_harness_with_no_deployed_sha_keys_no_unit(self, repo, config, modules):
        await self._seed_d1_greens(repo, config, modules)

        units = await self._module_units(_harness_over(config, None), repo, config, modules)

        assert set(units) == module_units('alpha') | module_units('beta')
        for unit in units.values():
            assert unit.outcome is UnitOutcome.UNAVAILABLE, unit
            assert 'no deployed sha' in (unit.reason or ''), unit
