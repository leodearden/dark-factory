"""The host's free disk never decides a test (task 7032, esc-5509-2).

conftest.py's autouse ``_hermetic_disk_usage`` installs
``_orch_helpers.py::ample_disk_usage`` as ``shutil.disk_usage``. Without it,
every lane test on the production verifier reads the real volume through the
pre-verify disk guard, and a host under 10 GiB free fails them all.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import ample_disk_usage

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane.worker import PRODUCTION_VERIFIER


def test_disk_usage_reports_the_hermetic_figure_not_the_host(tmp_path: Path) -> None:
    assert shutil.disk_usage(tmp_path) == ample_disk_usage(tmp_path)


@pytest.mark.asyncio
async def test_production_disk_guard_proceeds_at_its_default_threshold(tmp_path: Path) -> None:
    git_ops = MagicMock()
    git_ops.prune_stale_merge_worktrees = AsyncMock(
        side_effect=AssertionError('the guard pruned, so it saw a low disk'),
    )

    outcome = await PRODUCTION_VERIFIER.ensure_disk_space(
        git_ops, tmp_path, OrchestratorConfig().merge_verify_min_free_disk_bytes, 'hermetic-disk',
    )

    assert outcome.reason is None
