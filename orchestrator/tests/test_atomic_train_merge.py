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
    repo = tmp_path
    await _run(["git", "init", "-b", "main"], cwd=repo)
    await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
    await _run(["git", "config", "user.name", "Test"], cwd=repo)
    # Copy the cargo workspace fixture tree into the repo root.
    shutil.copytree(str(_FIXTURE_ROOT), str(repo), dirs_exist_ok=True)
    await _run(["git", "add", "-A"], cwd=repo)
    await _run(["git", "commit", "-m", "initial workspace"], cwd=repo)
    return repo


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
    full_branch = f"{git_ops.config.branch_prefix}{name}"
    wt_path = git_ops.worktree_base / name
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    await _run(
        ["git", "worktree", "add", "-b", full_branch, str(wt_path), base_ref],
        cwd=git_ops.project_root,
    )
    # Set git identity in the new worktree (required for commits).
    await _run(["git", "config", "user.email", "test@test.com"], cwd=wt_path)
    await _run(["git", "config", "user.name", "Test"], cwd=wt_path)

    # Apply the member's deterministic edit.
    edit_fn(wt_path)

    await git_ops.commit(wt_path, f"Add {name} task output")
    _, head_sha, _ = await _run(["git", "rev-parse", "HEAD"], cwd=wt_path)
    return wt_path, head_sha.strip()


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
    status_check = AsyncMock(
        return_value={name: "merge-deferred" for name in member_names}
    )
    mark_member_done = AsyncMock()

    loop = asyncio.get_event_loop()
    future: asyncio.Future[MergeOutcome] = loop.create_future()

    return GroupMergeRequest(
        task_id=tip_name,
        branch=tip_name,
        worktree=tip_worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        train_id=train_id,
        member_task_ids=list(member_names),
        tip_branch=tip_name,
        tip_task_id=tip_name,
        status_check=status_check,
        mark_member_done=mark_member_done,
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
    return OrchestratorConfig(
        project_root=repo,
        test_command="cargo test --workspace --quiet",
        # Use no-op shell builtins so lint/type_check don't interfere with
        # the Rust workspace.  'true' always exits 0.
        lint_command="true",
        type_check_command="true",
        verify_env={"CARGO_TARGET_DIR": str(target_dir)},
        verify_command_timeout_secs=300.0,
        git=GitConfig(
            main_branch="main",
            branch_prefix="task/",
            remote="origin",
            worktree_dir=".worktrees",
            push_after_advance=False,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1 (happy-path linear 3-train) — step-1 RED / step-2 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario1HappyPath:
    """PRD §10 row 1: 3-member stacked train merges atomically as one green commit.

    Uses real cargo: ``cargo test --workspace --quiet`` is the test_command
    and the post-merge verify runs without mocking.  Skips on rust-less machines
    via the ``cargo_or_skip`` session fixture.
    """

    async def test_happy_path_3_train_single_merge(
        self,
        cargo_or_skip,            # noqa: ARG002  (skip guard; value unused)
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """Happy path: seed additive 3-train, drive GroupMergeRequest, assert invariants.

        Invariants:
          (i)   Exactly one new merge commit on main.
          (ii)  All 3 mark_member_done callbacks fire with ONE shared merge_sha.
          (iii) outcome.status == 'done' and outcome.merge_sha == that sha.
          (iv)  All three crate edits present on main (git ls-tree -r --name-only).
          (v)   NO red-main window: cargo test --workspace green at final main
                tip AND at each member tip now reachable from main.
        """
        # --- seed repo ---------------------------------------------------------
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # --- stack 3 additive members -----------------------------------------
        # Each edit_fn appends a new pub fn to the relevant crate's lib.rs;
        # all edits are purely additive so every member tip compiles workspace-wide.

        def edit_alpha(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn alpha_task_output() -> u32 { 1 }\n")

        def edit_beta(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn beta_task_output() -> u32 { 2 }\n")

        def edit_gamma(wt: Path) -> None:
            lib = wt / "crate_c" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn gamma_task_output() -> u32 { 3 }\n")

        _, main_sha, _ = await _run(
            ["git", "rev-parse", "main"], cwd=repo
        )
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(git_ops, "alpha", main_sha, edit_alpha)
        wt_b, sha_b = await make_stacked_member(git_ops, "beta", sha_a, edit_beta)
        wt_c, sha_c = await make_stacked_member(git_ops, "gamma", sha_b, edit_gamma)

        # --- assert per-member workspace-green (γ₁ gate) ----------------------
        env = {**__import__("os").environ, "CARGO_TARGET_DIR": str(shared_cargo_target)}
        for label, wt in [("alpha", wt_a), ("beta", wt_b), ("gamma", wt_c)]:
            result = subprocess.run(
                ["cargo", "test", "--workspace", "--quiet"],
                cwd=wt,
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Member {label} tip is not workspace-green:\n{result.stdout}{result.stderr}"
            )

        # --- count merge commits BEFORE the train lands -----------------------
        _, before_log, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merge_commits_before = int(before_log.strip())

        # --- drive the GroupMergeRequest through real MergeWorker -------------
        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-scenario-1",
            member_names=["alpha", "beta", "gamma"],
            tip_name="gamma",
            tip_worktree=wt_c,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        outcome = await worker._do_merge(req)

        # --- (iii) outcome is done --------------------------------------------
        assert outcome is not None
        assert outcome.status == "done", f"expected done, got: {outcome!r}"
        assert outcome.merge_sha is not None

        # --- (i) exactly one new merge commit ---------------------------------
        _, after_log, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merge_commits_after = int(after_log.strip())
        assert merge_commits_after == merge_commits_before + 1, (
            "expected exactly 1 new merge commit on main, "
            f"got {merge_commits_after - merge_commits_before} new merge commits"
        )

        # --- (ii) all 3 mark_member_done callbacks with ONE shared SHA --------
        assert req.mark_member_done.call_count == 3, (  # type: ignore[union-attr]
            f"expected 3 mark_member_done calls, got {req.mark_member_done.call_count}"
        )
        called_shas = {
            call.args[1] for call in req.mark_member_done.call_args_list  # type: ignore[union-attr]
        }
        assert len(called_shas) == 1, f"all callbacks must share one SHA, got: {called_shas}"
        shared_merge_sha = next(iter(called_shas))
        assert shared_merge_sha == outcome.merge_sha

        # --- (iv) all three crate edits present on main -----------------------
        _, main_files, _ = await _run(
            ["git", "ls-tree", "-r", "--name-only", "main"], cwd=repo
        )
        # Each edit_fn wrote to crate_{a,b,c}/src/lib.rs — those files exist in fixture.
        # The additive functions are in the committed lib.rs; verify the files exist.
        assert "crate_a/src/lib.rs" in main_files
        assert "crate_b/src/lib.rs" in main_files
        assert "crate_c/src/lib.rs" in main_files

        # Check that the additive symbols appear on main
        _, lib_a_content, _ = await _run(
            ["git", "show", "main:crate_a/src/lib.rs"], cwd=repo
        )
        assert "alpha_task_output" in lib_a_content
        _, lib_c_content, _ = await _run(
            ["git", "show", "main:crate_c/src/lib.rs"], cwd=repo
        )
        assert "gamma_task_output" in lib_c_content

        # --- (v) NO red-main window: cargo green at final main tip -----------
        _, new_main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        new_main_sha = new_main_sha.strip()

        # Checkout the final main tip in a fresh worktree and run cargo
        main_wt = tmp_path / "_verify_main"
        await _run(
            ["git", "worktree", "add", "--detach", str(main_wt), new_main_sha],
            cwd=repo,
        )
        try:
            result = subprocess.run(
                ["cargo", "test", "--workspace", "--quiet"],
                cwd=main_wt,
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Final main tip is not workspace-green:\n{result.stdout}{result.stderr}"
            )
        finally:
            await _run(
                ["git", "worktree", "remove", "--force", str(main_wt)],
                cwd=repo,
            )


# ---------------------------------------------------------------------------
# Scenarios 2, 3, 4 (dispatch + worktree base) — step-3 RED / step-4 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario2WorktreeBase:
    """PRD §10 row 2: β's worktree branches off α's tip (not main).

    Uses real git_ops.create_worktree(train={order:1, ...}).
    No cargo needed.
    """

    async def test_sibling_tip_worktree_base(self, tmp_path: Path) -> None:
        """β worktree base == α tip SHA; git log task/beta..task/alpha is empty."""
        from orchestrator.git_ops import GitOps, _run

        # Seed a plain git repo (no cargo fixture needed for this scenario).
        repo = tmp_path / "repo"
        repo.mkdir()
        await _run(["git", "init", "-b", "main"], cwd=repo)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
        await _run(["git", "config", "user.name", "Test"], cwd=repo)
        (repo / "README.md").write_text("# Test\n")
        await _run(["git", "add", "-A"], cwd=repo)
        await _run(["git", "commit", "-m", "initial"], cwd=repo)

        git_config = GitConfig(
            main_branch="main",
            branch_prefix="task/",
            remote="origin",
            worktree_dir=".worktrees",
            push_after_advance=False,
        )
        git_ops = GitOps(git_config, repo)

        # Create α branch off main with one commit so it has a non-main tip.
        wt_a, alpha_sha = await make_stacked_member(
            git_ops, "alpha", (await git_ops._freshen_main())[0],
            lambda wt: (wt / "alpha.txt").write_text("alpha\n"),
        )

        # Now create β using train metadata: order=1, predecessor=alpha.
        # create_worktree branches β off α's tip (PRD §9.4).
        train_meta = {
            "id": "T-worktree",
            "order": 1,
            "members": ["alpha", "beta"],
        }
        wt_b_info = await git_ops.create_worktree("beta", train=train_meta)
        wt_b = wt_b_info.path

        # Set git identity in β's worktree.
        await _run(["git", "config", "user.email", "test@test.com"], cwd=wt_b)
        await _run(["git", "config", "user.name", "Test"], cwd=wt_b)

        # Assert: merge-base(task/beta, task/alpha) == alpha's tip SHA.
        _, merge_base_out, _ = await _run(
            ["git", "merge-base", "task/beta", "task/alpha"], cwd=repo
        )
        assert merge_base_out.strip() == alpha_sha, (
            f"expected merge-base == alpha tip {alpha_sha!r}, "
            f"got {merge_base_out.strip()!r}"
        )

        # Assert: git log task/beta..task/alpha is empty
        # (β branched exactly off α's tip, so no commits between them).
        _, log_out, _ = await _run(
            ["git", "log", "--oneline", "task/beta..task/alpha"], cwd=repo
        )
        assert log_out.strip() == "", (
            f"expected empty log between task/beta and task/alpha, "
            f"got: {log_out!r}"
        )


@pytest.mark.asyncio
class TestScenario3IntraTrainDispatch:
    """PRD §10 row 3: β dispatches when α is merge-deferred (intra-train allowance).

    Drives real scheduler._deps_satisfied with a status_map containing
    α='merge-deferred' and both tasks sharing train.id='T1'.
    """

    async def test_intra_train_dispatch(self, tmp_path: Path, caplog) -> None:
        """_deps_satisfied returns True for same-train merge-deferred predecessor."""
        import logging

        from orchestrator.scheduler import Scheduler

        config = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        scheduler = Scheduler(config)

        alpha_task = {
            "id": "alpha",
            "title": "Alpha",
            "description": "",
            "dependencies": [],
            "metadata": {"train": {"id": "T1", "order": 0, "members": ["alpha", "beta"]}},
        }
        beta_task = {
            "id": "beta",
            "title": "Beta",
            "description": "",
            "dependencies": [{"id": "alpha"}],
            "metadata": {"train": {"id": "T1", "order": 1, "members": ["alpha", "beta"]}},
        }

        status_map = {"alpha": "merge-deferred", "beta": "in-progress"}
        tasks_by_id = {"alpha": alpha_task, "beta": beta_task}

        with caplog.at_level(logging.DEBUG, logger="orchestrator.scheduler"):
            result = scheduler._deps_satisfied(beta_task, status_map, tasks_by_id)

        assert result is True, (
            "Expected _deps_satisfied to return True for intra-train merge-deferred dep"
        )
        debug_text = " ".join(r.getMessage() for r in caplog.records)
        assert "intra-train dep satisfied" in debug_text, (
            f"Expected 'intra-train dep satisfied' in logs; got: {debug_text!r}"
        )


@pytest.mark.asyncio
class TestScenario4ExtraTrainDispatchBlocked:
    """PRD §10 row 4: non-train task blocked by merge-deferred dep (no allowance).

    Regression: merge-deferred must NOT be treated as terminal for deps that
    cross train boundaries (or where the dependent has no train metadata).
    """

    async def test_extra_train_dispatch_blocked(self, tmp_path: Path) -> None:
        """_deps_satisfied returns False when dependent has no train metadata."""
        from orchestrator.scheduler import Scheduler

        config = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        scheduler = Scheduler(config)

        alpha_task = {
            "id": "alpha",
            "title": "Alpha",
            "description": "",
            "dependencies": [],
            "metadata": {"train": {"id": "T1", "order": 0, "members": ["alpha"]}},
        }
        # δ has NO train metadata — plain task depending on α.
        delta_task = {
            "id": "delta",
            "title": "Delta",
            "description": "",
            "dependencies": [{"id": "alpha"}],
            "metadata": {},  # no train key
        }

        status_map = {"alpha": "merge-deferred", "delta": "pending"}
        tasks_by_id = {"alpha": alpha_task, "delta": delta_task}

        result = scheduler._deps_satisfied(delta_task, status_map, tasks_by_id)

        assert result is False, (
            "Expected _deps_satisfied to return False for non-train task "
            "blocked by merge-deferred dep (merge-deferred is not terminal for "
            "cross-train or plain-task deps)"
        )
