"""The main-tip sweep's seed mode, end to end through real git (task 5812).

A WARM sweep must verify a tree CoW-seeded through task 4913's held-lane-lock
path; a COLD sweep, and the default, must verify an unseeded tree; a warm sweep
whose seed cannot happen still verifies, cold, and says so in the journal.
"""

from __future__ import annotations

import logging
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
from orchestrator.git_ops import GitOps, _run
from orchestrator.main_tip_sweep_cadence import SweepSeedMode
from orchestrator.verify import VerifyResult

# The real script's disk guard refuses below 50 GiB free with rc 75.
REFUSING_SEED_SCRIPT = '#!/usr/bin/env bash\nexit 75\n'

PASSING_RESULT = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='ok',
)


@pytest.fixture
def seed_repo(tmp_path: Path) -> Path:
    return make_seed_test_repo(tmp_path)


@pytest.fixture
def seed_repo_without_warm_base(tmp_path: Path) -> Path:
    return make_seed_test_repo(tmp_path, warm_base=False)


async def _sweep(
    repo: Path, script_body: str, **sweep_kwargs,
) -> tuple[str, object, list[bool]]:
    """Run the sweep at HEAD; return (head, result, seeded-at-verify-time per call)."""
    await commit_seed_script(repo, script_body)
    git_ops = GitOps(warm_pool_git_config(), repo)
    _, head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    head = head.strip()
    seeded_at_verify: list[bool] = []

    async def _fake_full_verification(project_root: Path, cfg, **kwargs):
        seeded_at_verify.append((project_root / 'target' / 'seeded.bin').exists())
        return PASSING_RESULT

    with patch.object(
        verify_module, 'run_full_verification', side_effect=_fake_full_verification,
    ):
        result = await verify_module.run_main_tip_sweep(
            OrchestratorConfig(project_root=repo), git_ops,
            main_sha=head, **sweep_kwargs,
        )
    return head, result, seeded_at_verify


def _cold_fallback_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any(
        r.name == 'orchestrator.git_ops'
        and r.levelno >= logging.INFO
        and 'COLD' in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
class TestMainTipSweepSeedMode:
    async def test_warm_sweep_verifies_a_tree_seeded_under_its_own_lane_lock(
        self, seed_repo: Path,
    ):
        head, result, seeded = await _sweep(
            seed_repo, CURRENT_LOCKING_SEED_SCRIPT, seed_mode=SweepSeedMode.WARM,
        )

        assert seeded == [True], (
            'the warm sweep verified an UNSEEDED tree: either run_main_tip_sweep '
            'never requested warm_seed, or the CM\'s lane-lock assertion '
            '(--assume-lane-lock-held, task 4913) did not reach the seed script'
        )
        assert result == (head, PASSING_RESULT)

    async def test_cold_sweep_verifies_an_unseeded_tree(self, seed_repo: Path):
        _, _, seeded = await _sweep(
            seed_repo, CURRENT_LOCKING_SEED_SCRIPT, seed_mode=SweepSeedMode.COLD,
        )

        assert seeded == [False]

    async def test_default_seed_mode_is_cold(self, seed_repo: Path):
        _, _, seeded = await _sweep(seed_repo, CURRENT_LOCKING_SEED_SCRIPT)

        assert seeded == [False], 'run_main_tip_sweep must default to the ground truth'

    async def test_warm_sweep_whose_seed_refuses_still_verifies_cold(
        self, seed_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            _, result, seeded = await _sweep(
                seed_repo, REFUSING_SEED_SCRIPT, seed_mode=SweepSeedMode.WARM,
            )

        assert seeded == [False]
        assert result is not None
        assert _cold_fallback_logged(caplog), caplog.text

    async def test_warm_sweep_without_a_warm_base_logs_its_cold_fallback(
        self, seed_repo_without_warm_base: Path, caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            _, result, seeded = await _sweep(
                seed_repo_without_warm_base, CURRENT_LOCKING_SEED_SCRIPT,
                seed_mode=SweepSeedMode.WARM,
            )

        assert seeded == [False]
        assert result is not None
        assert _cold_fallback_logged(caplog), (
            'a warm-requested sweep that ran cold for want of a warm base must '
            f'say so at INFO; got:\n{caplog.text}'
        )
