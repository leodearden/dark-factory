"""Integration gate: 3-crate cargo workspace fixture, train end-to-end (12 boundary scenarios).

Pattern: real git + real cargo where the workspace compile IS the observable
(scenarios 1, 5, 8-clean); orchestrator decision functions with injected /
mocked VerifyResults for the remaining scenarios (2-4, 6-7, 9-12).

Mirrors test_workflow_e2e.py: "real git operations and file I/O; agent
invocations stubbed with deterministic side-effect functions that write actual
files."  No live Claude agent is needed; component seams are tested directly.

Helper functions are folded into this file (used by this file only) so the
entire change stays inside the task's two declared module paths and avoids any
sys.modules['conftest'] collision (repo convention).

Scaffolding stubs defined here are completed in step-2.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    GroupMergeRequest,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
)
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

# Canonical location of the atomic_train cargo workspace fixture tree.
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "atomic_train"


# ---------------------------------------------------------------------------
# Fixtures: cargo availability + shared CARGO_TARGET_DIR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cargo_or_skip():
    """Skip when cargo is not installed on this machine.

    Models conftest.py:44-69 repo_root skip pattern.  skip≠fail: the gate
    exits 0 on rust-less machines while providing full coverage where cargo
    exists (cargo 1.94.1 is present in the target environment).
    """
    if shutil.which("cargo") is None:
        pytest.skip("cargo unavailable")


@pytest.fixture(scope="module")
def shared_cargo_target(tmp_path_factory):
    """Return one tmp CARGO_TARGET_DIR shared by all tests in this module.

    Compiling the trivial workspace once; incremental member builds are
    sub-second, keeping the ~6-7 cargo invocations tractable.
    """
    return tmp_path_factory.mktemp("cargo_target")


# ---------------------------------------------------------------------------
# Helper: git-init a workspace repo seeded with the cargo fixture
# ---------------------------------------------------------------------------


async def seed_workspace_repo(tmp_path: Path) -> Path:
    """Git-init a fresh repo, copy atomic_train fixture in, make initial commit.

    Returns the repo Path (same as *tmp_path*).

    Steps:
      1. git init -b main
      2. git config user.email/name
      3. shutil.copytree(_FIXTURE_ROOT → tmp_path, dirs_exist_ok=True)
      4. git add -A && git commit -m "initial workspace"
    """
    raise NotImplementedError(
        "seed_workspace_repo scaffold: complete in step-2 (make scenario 1 green)"
    )


# ---------------------------------------------------------------------------
# Helper: create one stacked train member branch
# ---------------------------------------------------------------------------


async def make_stacked_member(
    git_ops: GitOps,
    name: str,
    base_ref: str,
    edit_fn: Callable[[Path], None],
) -> tuple[Path, str]:
    """Create branch ``task/<name>`` off *base_ref*, apply *edit_fn*, commit.

    Returns ``(worktree_path, head_sha)``.

    Adapts test_merge_queue._make_stacked_train:5628 from .py-file writes to
    crate edits (edit_fn receives the worktree root and edits the relevant
    crate src file).
    """
    raise NotImplementedError(
        "make_stacked_member scaffold: complete in step-2 (make scenario 1 green)"
    )


# ---------------------------------------------------------------------------
# Helper: assemble a GroupMergeRequest from 3 stacked members
# ---------------------------------------------------------------------------


def build_group_merge_request(
    *,
    git_ops: GitOps,
    config: OrchestratorConfig,
    train_id: str,
    member_names: list[str],
    tip_name: str,
    tip_worktree: Path,
) -> GroupMergeRequest:
    """Return a GroupMergeRequest wired with AsyncMock status_check / mark_member_done.

    *member_names* must be ordered root-to-tip (inclusive).
    *tip_name* is the last entry in *member_names*.
    status_check returns ``{name: 'merge-deferred'}`` for all members.
    """
    raise NotImplementedError(
        "build_group_merge_request scaffold: complete in step-2 (make scenario 1 green)"
    )


# ---------------------------------------------------------------------------
# Helper: build an OrchestratorConfig for cargo workspace train tests
# ---------------------------------------------------------------------------


def make_train_config(repo: Path, target_dir: Path) -> OrchestratorConfig:
    """Return OrchestratorConfig with cargo test --workspace --quiet as test_command.

    Key settings:
    - push_after_advance=False  (no real remote needed for advance_main)
    - test_command='cargo test --workspace --quiet'
    - lint_command=None / type_check_command=None  (not under test here)
    - verify_env={'CARGO_TARGET_DIR': str(target_dir)}  so all verify calls
      use the shared incremental build cache
    - verify_command_timeout_secs=300  (generous for CI; trivial workspace is fast)
    """
    raise NotImplementedError(
        "make_train_config scaffold: complete in step-2 (make scenario 1 green)"
    )
