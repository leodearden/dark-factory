"""The main-tip sweep and its confirm gate verify in a provisioned tree (task 6917).

They must verify in a tree provisioned the way a merge worktree is. Otherwise an
environment-only failure on a green main is filed as a red main: with no
``node_modules`` the jsdom suites fail (esc-main-sweep-f002ca7d93b1-2).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from _seed_script_stubs import (
    CURRENT_LOCKING_SEED_SCRIPT,
    commit_seed_script,
    make_seed_test_repo,
    warm_pool_git_config,
)

from orchestrator import verify as verify_module
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.main_tip_sweep_cadence import SweepSeedMode
from orchestrator.verify import VerifyResult

PROVISIONED = '.provisioned'
# Stands in for the deployed `uv sync --all-packages && npm ci`.
PROVISION = f'touch {PROVISIONED}'

PASSING_RESULT = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='ok',
)


def _head(repo: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_seed_test_repo(tmp_path)


async def _sweep(
    repo: Path, *, provision: str = PROVISION, **sweep_kwargs,
) -> tuple[str, object, list[tuple[bool, bool]]]:
    """Run the sweep at HEAD; return (head, result, (provisioned, seeded) per verify)."""
    await commit_seed_script(repo, CURRENT_LOCKING_SEED_SCRIPT)
    git_ops = GitOps(warm_pool_git_config(), repo)
    head = _head(repo)
    seen: list[tuple[bool, bool]] = []

    async def _fake_full_verification(project_root: Path, cfg, **kwargs):
        seen.append((
            (project_root / PROVISIONED).exists(),
            (project_root / 'target' / 'seeded.bin').exists(),
        ))
        return PASSING_RESULT

    with patch.object(
        verify_module, 'run_full_verification', side_effect=_fake_full_verification,
    ):
        result = await verify_module.run_main_tip_sweep(
            OrchestratorConfig(project_root=repo, verify_cold_preprovision_command=provision),
            git_ops, main_sha=head, **sweep_kwargs,
        )
    return head, result, seen


@pytest.mark.asyncio
class TestMainTipSweepProvisionedEnv:
    async def test_cold_sweep_verifies_a_provisioned_unseeded_tree(self, repo: Path):
        head, result, seen = await _sweep(repo, seed_mode=SweepSeedMode.COLD)

        assert seen == [(True, False)], (
            'the cold sweep must verify a provisioned tree that is still unseeded: '
            'its ground truth is about target/, not about the lockfile install'
        )
        assert result == (head, PASSING_RESULT)

    async def test_warm_sweep_verifies_a_tree_both_seeded_and_provisioned(self, repo: Path):
        _, _, seen = await _sweep(repo, seed_mode=SweepSeedMode.WARM)

        assert seen == [(True, True)]

    async def test_a_failed_provision_still_yields_the_sweep_verdict_and_says_so(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            head, result, seen = await _sweep(repo, provision='exit 3')

        assert result == (head, PASSING_RESULT), (
            'a failed provision must be neither a None no-signal nor a red'
        )
        assert seen == [(False, False)]
        assert any(
            r.name == 'orchestrator.verify'
            and r.levelno == logging.WARNING
            and 'pre-provision' in r.getMessage()
            and 'exit 3' in r.getMessage()
            for r in caplog.records
        ), caplog.text
