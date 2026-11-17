"""The main-tip sweep and its confirm gate verify in a provisioned tree (task 6917).

They must verify in a tree provisioned the way a merge worktree is. Otherwise an
environment-only failure on a green main is filed as a red main: with no
``node_modules`` the jsdom suites fail (esc-main-sweep-f002ca7d93b1-2).
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from _git_fixtures import seed_repo
from _seed_script_stubs import (
    CURRENT_LOCKING_SEED_SCRIPT,
    commit_seed_script,
    make_seed_test_repo,
    warm_pool_git_config,
)

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.main_tip_sweep_cadence import SweepSeedMode
from orchestrator.verify import VerifyResult

PROVISIONED = '.provisioned'
# Stands in for the deployed `uv sync --all-packages && npm ci`, and records
# the verify role it was provisioned under.
PROVISION = f'printf %s "$DF_VERIFY_ROLE" > {PROVISIONED}'

PASSING_RESULT = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='ok',
)

NODE_ID = 'pkg/tests/test_env.py::test_needs_toolchain'

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output=f'FAILED {NODE_ID}',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {NODE_ID}',
    category='test_failure',
)


def _head(repo: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _provisioned_role(tree: Path) -> str | None:
    """The verify role *tree* was provisioned under, or None if it never was."""
    marker = tree / PROVISIONED
    return marker.read_text() if marker.exists() else None


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_seed_test_repo(tmp_path)


async def _sweep(
    repo: Path, *, provision: str = PROVISION, **sweep_kwargs,
) -> tuple[str, object, list[tuple[str | None, bool]]]:
    """Run the sweep at HEAD; return (head, result, (provisioned role, seeded) per verify)."""
    await commit_seed_script(repo, CURRENT_LOCKING_SEED_SCRIPT)
    git_ops = GitOps(warm_pool_git_config(), repo)
    head = _head(repo)
    seen: list[tuple[str | None, bool]] = []

    async def _fake_full_verification(project_root: Path, cfg, **kwargs):
        seen.append((
            _provisioned_role(project_root),
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

        assert seen == [('background', False)], (
            'the cold sweep must verify a tree provisioned under its own background '
            'role that is still unseeded: its ground truth is about target/, not '
            'about the lockfile install'
        )
        assert result == (head, PASSING_RESULT)

    async def test_warm_sweep_verifies_a_tree_both_seeded_and_provisioned(self, repo: Path):
        _, _, seen = await _sweep(repo, seed_mode=SweepSeedMode.WARM)

        assert seen == [('background', True)]

    async def test_the_sweep_log_reports_what_provisioning_cost(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.INFO, logger='orchestrator.verify'):
            await _sweep(repo)

        assert any(
            r.name == 'orchestrator.verify'
            and re.search(r'pre-provision finished in \d+\.\ds \(rc=0\)', r.getMessage())
            for r in caplog.records
        ), caplog.text

    async def test_a_failed_provision_still_yields_the_sweep_verdict_and_says_so(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            head, result, seen = await _sweep(repo, provision='exit 3')

        assert result == (head, PASSING_RESULT), (
            'a failed provision must be neither a None no-signal nor a red'
        )
        assert seen == [(None, False)]
        assert any(
            r.name == 'orchestrator.verify'
            and r.levelno == logging.WARNING
            and 'pre-provision' in r.getMessage()
            and 'exit 3' in r.getMessage()
            for r in caplog.records
        ), caplog.text


def _confirm_repo(tmp_path: Path) -> Path:
    """A repo whose committed subproject owns NODE_ID, so the gate can re-run it."""
    repo = seed_repo(tmp_path / 'repo')
    (repo / 'pkg' / 'tests').mkdir(parents=True)
    (repo / 'pkg' / 'orchestrator.yaml').write_text(
        'test_command: "uv run --project pkg --directory pkg pytest tests/ --tb=short -q"\n',
    )
    (repo / 'pkg' / 'tests' / 'test_env.py').write_text(
        'def test_needs_toolchain():\n    pass\n',
    )
    identity = ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid']
    subprocess.run([*identity, 'add', '-A'], cwd=repo, check=True)
    subprocess.run([*identity, 'commit', '-qm', 'pkg'], cwd=repo, check=True)
    return repo


async def _confirm(
    tmp_path: Path,
    *,
    rerun_passes: Callable[[str | None], bool],
    provision: str = PROVISION,
) -> tuple[bool, list[str | None]]:
    """Run the confirm gate on FAILING_RESULT; return (verdict, provisioned role per re-run).

    Each isolated re-run passes iff *rerun_passes* holds for the role its tree
    was provisioned under (None: never provisioned).
    """
    repo = _confirm_repo(tmp_path)
    config = OrchestratorConfig(
        project_root=repo,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
        verify_cold_preprovision_command=provision,
    )
    git_ops = GitOps(config.git, repo)
    seen: list[str | None] = []

    async def _fake_run_verification(worktree: Path, cfg, module_config=None, **kwargs):
        role = _provisioned_role(worktree)
        seen.append(role)
        return PASSING_RESULT if rerun_passes(role) else FAILING_RESULT

    with patch.object(verify_module, 'run_verification', side_effect=_fake_run_verification):
        verdict = await verify_module.confirm_main_tip_failure_is_real(
            config, git_ops, FAILING_RESULT, main_sha=_head(repo),
        )
    return verdict, seen


@pytest.mark.asyncio
class TestConfirmGateProvisionedEnv:
    async def test_an_environment_only_failure_is_suppressed_once_the_confirm_tree_is_provisioned(
        self, tmp_path: Path,
    ):
        verdict, seen = await _confirm(tmp_path, rerun_passes=lambda role: role is not None)

        assert verdict is False
        assert seen == ['task'], 'the confirm tree is provisioned under the task role'

    async def test_a_failure_that_survives_provisioning_still_files_the_alarm(
        self, tmp_path: Path,
    ):
        verdict, seen = await _confirm(tmp_path, rerun_passes=lambda role: False)

        assert verdict is True, 'the confirm gate must never mask a real red'
        assert seen and all(role == 'task' for role in seen), (
            'every isolated re-run must see the provisioned tree'
        )

    async def test_a_failed_provision_is_not_itself_an_alarm(self, tmp_path: Path):
        verdict, seen = await _confirm(
            tmp_path, rerun_passes=lambda role: True, provision='exit 3',
        )

        assert seen == [None], 'a failed provision fails open into the re-run'
        assert verdict is False, (
            'the re-run, not the failed provision, decides the verdict'
        )
