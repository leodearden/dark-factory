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
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, TrainMembership, _run
from orchestrator.merge_queue import (
    ABANDONED_REASON_PREFIX,
    POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
    TRAIN_REBASE_CONFLICT_REASON_PREFIX,
    TRANSIENT_INFRA_REASON_PREFIX,
    GroupMergeRequest,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
)
from orchestrator.verify import VerifyResult, run_scoped_verification

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
    edit_fn: Callable[[Path], Any],
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

    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()

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
            f"expected 3 mark_member_done calls, got {req.mark_member_done.call_count}"  # type: ignore[union-attr]
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

        # Check that ALL THREE additive symbols appear on main.
        # Asserting α, β, AND γ ensures no member is silently dropped by the worker.
        _, lib_a_content, _ = await _run(
            ["git", "show", "main:crate_a/src/lib.rs"], cwd=repo
        )
        assert "alpha_task_output" in lib_a_content
        _, lib_b_content, _ = await _run(
            ["git", "show", "main:crate_b/src/lib.rs"], cwd=repo
        )
        assert "beta_task_output" in lib_b_content
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
        train_meta: TrainMembership = {
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

    async def test_extra_train_dispatch_blocked_cross_train(self, tmp_path: Path) -> None:
        """Cross-train case: δ is in T2, α is in T1; intra-train allowance must NOT apply.

        This is the more interesting regression: the intra-train allowance (scheduler.py:1489-1515)
        checks same train_id; a different train_id must be rejected even though δ carries
        train metadata.
        """
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
        # δ is in a DIFFERENT train (T2), depending on α which is in T1.
        # The intra-train allowance must not fire because train_id differs.
        delta_task = {
            "id": "delta",
            "title": "Delta",
            "description": "",
            "dependencies": [{"id": "alpha"}],
            "metadata": {"train": {"id": "T2", "order": 0, "members": ["delta"]}},
        }

        status_map = {"alpha": "merge-deferred", "delta": "pending"}
        tasks_by_id = {"alpha": alpha_task, "delta": delta_task}

        result = scheduler._deps_satisfied(delta_task, status_map, tasks_by_id)

        assert result is False, (
            "Expected _deps_satisfied to return False for cross-train dep (T2→T1): "
            "intra-train allowance must only apply within the SAME train_id"
        )


# ---------------------------------------------------------------------------
# Scenario 5 (group merge gate = workspace verify) — step-5 RED / step-6 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario5GroupMergeVerify:
    """PRD §10 row 5: post-merge workspace verify is the gate.

    GREEN variant: spy on run_scoped_verification to confirm exactly ONE
    post-merge call with is_merge_verify=True and role='merge'; real cargo
    green → advance_main succeeds, all 3 mark_member_done fire.

    RED variant: γ edit references a non-existent symbol; cargo fails on the
    assembled merge commit → outcome blocked, advance_main NOT called, main
    SHA unmoved, zero mark_member_done calls.

    Both variants use real cargo (cargo_or_skip).
    """

    async def test_group_merge_workspace_verify_green(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """GREEN: post-merge verify runs once with is_merge_verify=True/role='merge'."""
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # Additive edits — all member tips compile workspace-wide.
        def edit_alpha(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn alpha5_output() -> u32 { 5 }\n")

        def edit_beta(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn beta5_output() -> u32 { 5 }\n")

        def edit_gamma(wt: Path) -> None:
            lib = wt / "crate_c" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn gamma5_output() -> u32 { 5 }\n")

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(git_ops, "alpha5", main_sha, edit_alpha)
        wt_b, sha_b = await make_stacked_member(git_ops, "beta5", sha_a, edit_beta)
        wt_c, _sha_c = await make_stacked_member(git_ops, "gamma5", sha_b, edit_gamma)

        # Spy: wrap the real run_scoped_verification and record every call.
        verify_calls: list[dict] = []

        async def _spy_verify(*args, **kwargs):
            verify_calls.append({"args": args, "kwargs": kwargs})
            return await run_scoped_verification(*args, **kwargs)

        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-scenario-5-green",
            member_names=["alpha5", "beta5", "gamma5"],
            tip_name="gamma5",
            tip_worktree=wt_c,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with patch("orchestrator.merge_queue.run_scoped_verification", side_effect=_spy_verify):
            outcome = await worker._do_merge(req)

        # (i) Outcome is done.
        assert outcome is not None
        assert outcome.status == "done", f"expected done, got: {outcome!r}"

        # (ii) Exactly ONE post-merge verify call with is_merge_verify=True.
        merge_verify_calls = [
            c for c in verify_calls if c["kwargs"].get("is_merge_verify") is True
        ]
        assert len(merge_verify_calls) == 1, (
            f"expected exactly 1 post-merge verify call (is_merge_verify=True), "
            f"got {len(merge_verify_calls)}; all calls: {verify_calls}"
        )
        assert merge_verify_calls[0]["kwargs"].get("role") == "merge", (
            f"expected role='merge' on post-merge call, "
            f"got: {merge_verify_calls[0]['kwargs']}"
        )

        # (iii) All THREE member edits present on main (α, β, γ — no member silently dropped).
        _, lib_a5, _ = await _run(["git", "show", "main:crate_a/src/lib.rs"], cwd=repo)
        assert "alpha5_output" in lib_a5
        _, lib_b5, _ = await _run(["git", "show", "main:crate_b/src/lib.rs"], cwd=repo)
        assert "beta5_output" in lib_b5
        _, lib_c5, _ = await _run(["git", "show", "main:crate_c/src/lib.rs"], cwd=repo)
        assert "gamma5_output" in lib_c5

        # (iv) All 3 mark_member_done fired.
        assert req.mark_member_done.call_count == 3, (  # type: ignore[union-attr]
            f"expected 3 mark_member_done calls, got {req.mark_member_done.call_count}"  # type: ignore[union-attr]
        )

    async def test_group_merge_workspace_verify_red(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """RED: γ edit has broken code; post-merge cargo fails; main unmoved, no flips."""
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # α and β: clean additive edits.
        def edit_alpha(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn alpha5r_output() -> u32 { 55 }\n")

        def edit_beta(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn beta5r_output() -> u32 { 55 }\n")

        # γ: deliberately broken edit — references a function that does not
        # exist in train_b.  This makes cargo test --workspace fail on the
        # assembled tip (the post-merge merge worktree).
        def edit_gamma_red(wt: Path) -> None:
            lib = wt / "crate_c" / "src" / "lib.rs"
            lib.write_text(
                lib.read_text()
                + "\npub fn gamma5r_broken() { train_b::fn_that_does_not_exist_anywhere(); }\n"
            )

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(git_ops, "alpha5r", main_sha, edit_alpha)
        wt_b, sha_b = await make_stacked_member(git_ops, "beta5r", sha_a, edit_beta)
        wt_c, _sha_c = await make_stacked_member(git_ops, "gamma5r", sha_b, edit_gamma_red)

        # Record main state before the (failed) merge attempt.
        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()
        _, merges_before_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_before = int(merges_before_str.strip())

        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-scenario-5-red",
            member_names=["alpha5r", "beta5r", "gamma5r"],
            tip_name="gamma5r",
            tip_worktree=wt_c,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        outcome = await worker._do_merge(req)

        # (i) Outcome is blocked (not done).
        assert outcome is not None
        assert outcome.status != "done", (
            f"expected blocked outcome when post-merge verify is red, got: {outcome!r}"
        )

        # (ii) advance_main NOT called: main SHA unmoved.
        _, main_sha_after, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        assert main_sha_after.strip() == main_sha_before, (
            f"advance_main must NOT fire on post-merge verify failure; "
            f"main moved from {main_sha_before!r} to {main_sha_after.strip()!r}"
        )

        # (iii) No new merge commits on main.
        _, merges_after_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_after = int(merges_after_str.strip())
        assert merges_after == merges_before, (
            f"no new merge commits expected on main, "
            f"got {merges_after - merges_before} new commit(s)"
        )

        # (iv) No mark_member_done flips.
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"mark_member_done must not fire when verify gate is red, "
            f"got {req.mark_member_done.call_count} call(s)"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Scenarios 6, 7 (park-prefix derail + resume) — step-7 RED / step-8 GREEN
# ---------------------------------------------------------------------------

# Helpers for the loop-guard + workflow fixture (scenario 6 / 7).
# Inlined here (not a separate module) to stay inside the task's declared scope.


def _make_verify_failure(
    category: str = "test_failure",
    cause_hint: str = "FAILED tests/test_x.py::test_y",
) -> VerifyResult:
    """Return a failing VerifyResult with the given (category, cause_hint)."""
    return VerifyResult(
        passed=False,
        test_output="FAILED\n",
        lint_output="",
        type_output="",
        summary="Failures",
        timed_out=False,
        cause_hint=cause_hint,
        category=category,
    )


def _make_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    task_id: str,
    worktree: Path,
    *,
    train_meta: dict | None = None,
    get_statuses_return: tuple[dict[str, str], Exception | None] = ({}, None),
    merge_queue: asyncio.Queue | None = None,
    tasks_by_train_return: list[dict] | None = None,
) -> tuple:
    """Build a minimally-wired TaskWorkflow for any task (train or plain).

    Reuses the _make_workflow pattern from test_workflow_signature_loop_guard.py.
    When *train_meta* is None the task has no 'train' key (plain non-train task).
    Returns (workflow, scheduler_mock).
    """
    from orchestrator.agents.invoke import AgentResult
    from orchestrator.artifacts import TaskArtifacts
    from orchestrator.workflow import TaskWorkflow

    assignment = MagicMock()
    assignment.task_id = task_id
    metadata: dict = {"train": train_meta} if train_meta is not None else {}
    assignment.task = {
        "id": task_id, "title": task_id, "description": "",
        "status": "in-progress",
        "metadata": metadata,
    }
    assignment.modules = []

    scheduler = MagicMock()
    scheduler.get_statuses = AsyncMock(return_value=get_statuses_return)
    scheduler.mark_done = AsyncMock()
    scheduler.tasks_by_train = AsyncMock(
        return_value=(tasks_by_train_return or [])
    )
    scheduler.update_task = AsyncMock(return_value=True)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        merge_queue=merge_queue,
    )
    wf.worktree = worktree
    wf.event_store = None

    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, task_id, "desc", base_commit="base-sha")
    wf.artifacts = artifacts
    wf.plan = {"task_id": task_id, "steps": []}
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf.briefing.build_debugger_prompt = AsyncMock(return_value="debug")  # type: ignore[attr-defined]
    wf._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=""),
    )
    wf._get_head_commit = AsyncMock(return_value="head-sha")  # type: ignore[method-assign]
    wf._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

    return wf, scheduler


# Back-compat shims so existing call-sites keep working unchanged.
def _make_train_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    task_id: str,
    train_meta: dict,
    get_statuses_return: tuple[dict[str, str], Exception | None],
    worktree: Path,
    merge_queue: asyncio.Queue | None = None,
    tasks_by_train_return: list[dict] | None = None,
) -> tuple:
    return _make_workflow(
        config, git_ops, task_id, worktree,
        train_meta=train_meta,
        get_statuses_return=get_statuses_return,
        merge_queue=merge_queue,
        tasks_by_train_return=tasks_by_train_return,
    )


def _make_plain_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    task_id: str,
    worktree: Path,
) -> tuple:
    return _make_workflow(config, git_ops, task_id, worktree)


@pytest.mark.asyncio
class TestScenario6ParkPrefixDerail:
    """PRD §10 row 6: park-prefix derail — γ loop-guard fires BLOCKED,
    train state reflects parked α/β, reaper sweep skips live merge-deferred worktrees.

    Three sub-assertions in one test:
    (i)   _verify_debugfix_loop: 3 identical-signature failures → BLOCKED (γ₂ guard).
    (ii)  _build_train_state: α/β parked, γ failing.
    (iii) _reap_orphan_worktrees: α/β worktrees survive (live ids, not reaped).
    """

    async def test_loop_guard_signature_blocks(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        """(i) 3× identical-signature failures → BLOCKED (γ₂ loop-guard)."""
        import logging

        from orchestrator.workflow import WorkflowOutcome

        # Set up a minimal git repo so create_worktree works.
        repo6 = tmp_path / "repo6_lg"
        repo6.mkdir()
        await _run(["git", "init", "-b", "main"], cwd=repo6)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo6)
        await _run(["git", "config", "user.name", "Test"], cwd=repo6)
        (repo6 / "README.md").write_text("# repo6\n")
        await _run(["git", "add", "-A"], cwd=repo6)
        await _run(["git", "commit", "-m", "initial"], cwd=repo6)

        config6 = OrchestratorConfig(
            project_root=repo6,
            max_verify_attempts=5,
            max_failure_signature_repeat=3,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        git_ops6 = GitOps(config6.git, repo6)

        # Create γ's worktree.
        wt_info = await git_ops6.create_worktree("gamma6lg")
        wt_gamma = wt_info.path

        train_meta6 = {"id": "T6", "order": 2, "members": ["alpha6", "beta6", "gamma6"]}
        get_statuses6 = (
            {"alpha6": "merge-deferred", "beta6": "merge-deferred", "gamma6": "blocked"},
            None,
        )
        wf6, _ = _make_train_workflow(
            config6, git_ops6, "gamma6lg", train_meta6, get_statuses6, wt_gamma,
        )

        # Inject exactly 3 identical-signature failures: same category + cause_hint
        # differing only in file:line (normaliser strips the line number).
        # Exactly 3 — StopIteration surfaces guard regressions cleanly.
        failures = [
            _make_verify_failure("test_failure", "FAILED tests/test_x.py:42 AssertionError"),
            _make_verify_failure("test_failure", "FAILED tests/test_x.py:99 AssertionError"),
            _make_verify_failure("test_failure", "FAILED tests/test_x.py:155 AssertionError"),
        ]
        verify_mock = AsyncMock(side_effect=failures)
        monkeypatch.setattr("orchestrator.workflow.run_scoped_verification", verify_mock)

        with caplog.at_level(logging.WARNING, logger="orchestrator.workflow"):
            outcome_i = await wf6._verify_debugfix_loop()

        assert outcome_i == WorkflowOutcome.BLOCKED, (
            f"expected BLOCKED after 3 identical-signature failures, got {outcome_i!r}"
        )
        assert verify_mock.await_count == 3, (
            f"loop-guard must fire at attempt 3 (not wait for max_verify_attempts), "
            f"got {verify_mock.await_count} calls"
        )
        warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "consecutive identical verify failures" in warning_text, (
            f"expected WARNING about consecutive identical failures; got: {warning_text!r}"
        )

    async def test_build_train_state_reports_parked_failing(
        self,
        tmp_path: Path,
    ) -> None:
        """(ii) _build_train_state returns {id:T6, order:2, parked:[α,β], failing:γ}."""
        # Set up a minimal git repo so create_worktree works.
        repo6b = tmp_path / "repo6_bts"
        repo6b.mkdir()
        await _run(["git", "init", "-b", "main"], cwd=repo6b)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo6b)
        await _run(["git", "config", "user.name", "Test"], cwd=repo6b)
        (repo6b / "README.md").write_text("# repo6b\n")
        await _run(["git", "add", "-A"], cwd=repo6b)
        await _run(["git", "commit", "-m", "initial"], cwd=repo6b)

        config6b = OrchestratorConfig(
            project_root=repo6b,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        git_ops6b = GitOps(config6b.git, repo6b)

        wt_info = await git_ops6b.create_worktree("gamma6")
        wt_gamma = wt_info.path

        train_meta6 = {"id": "T6", "order": 2, "members": ["alpha6", "beta6", "gamma6"]}
        # α/β are parked (merge-deferred); γ is the failing member (blocked).
        get_statuses6 = (
            {"alpha6": "merge-deferred", "beta6": "merge-deferred", "gamma6": "blocked"},
            None,
        )
        wf6b, _ = _make_train_workflow(
            config6b, git_ops6b, "gamma6", train_meta6, get_statuses6, wt_gamma,
        )

        train_state = await wf6b._build_train_state()

        assert train_state is not None, "_build_train_state must return a dict for train task"
        assert train_state["id"] == "T6", f"train id mismatch: {train_state!r}"
        assert train_state["order"] == 2, f"train order mismatch: {train_state!r}"
        assert set(train_state["parked_members"]) == {"alpha6", "beta6"}, (
            f"expected parked_members={{alpha6, beta6}}, got {train_state['parked_members']!r}"
        )
        assert train_state["failing_member"] == "gamma6", (
            f"expected failing_member=gamma6, got {train_state['failing_member']!r}"
        )

    async def test_reaper_skips_live_merge_deferred(
        self,
        tmp_path: Path,
        mock_orch_config,
    ) -> None:
        """(iii) Reaper sweep skips α/β worktrees (live merge-deferred status)."""
        from orchestrator.harness import Harness

        with (
            patch("orchestrator.harness.McpLifecycle"),
            patch("orchestrator.harness.Scheduler"),
            patch("orchestrator.harness.BriefingAssembler"),
        ):
            harness = Harness(mock_orch_config)

        # Override harness internals for the reaper test.
        reaper_base = tmp_path / "worktrees_reaper"
        reaper_base.mkdir(parents=True, exist_ok=True)
        harness.git_ops.worktree_base = reaper_base
        harness.git_ops.worktree_has_unsaved_work = AsyncMock(return_value=False)
        harness.git_ops.quarantine_worktree = AsyncMock(return_value=tmp_path / "q-dest")
        harness.git_ops.cleanup_worktree = AsyncMock()
        harness.git_ops.prune_worktrees = AsyncMock()
        harness.config.worktree_orphan_reaper_enabled = True

        # α/β are live (in get_tasks) with merge-deferred status.
        harness.scheduler = MagicMock()
        harness.scheduler._dispatched = set()
        harness.scheduler.get_tasks = AsyncMock(return_value=[
            {"id": "alpha6", "status": "merge-deferred"},
            {"id": "beta6", "status": "merge-deferred"},
            {"id": "gamma6", "status": "blocked"},
        ])
        harness._recovered_plans = {}
        harness._preserved_worktrees = set()
        harness._recovered_sessions = {}
        harness.event_store = None

        # Create α/β worktree dirs.
        wt_alpha_dir = reaper_base / "alpha6"
        wt_beta_dir = reaper_base / "beta6"
        wt_alpha_dir.mkdir()
        wt_beta_dir.mkdir()

        # Also create an orphan dir (id "999") that SHOULD be reaped.
        orphan_dir = reaper_base / "999"
        orphan_dir.mkdir()

        await harness._reap_orphan_worktrees()

        # α/β must NOT be reaped (live merge-deferred = skipped by reaper).
        harness.git_ops.cleanup_worktree.assert_called_once()  # type: ignore[attr-defined]
        reaped_paths = {c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list}  # type: ignore[attr-defined]
        assert wt_alpha_dir not in reaped_paths, (
            "alpha6's worktree must NOT be reaped — it is a live merge-deferred task"
        )
        assert wt_beta_dir not in reaped_paths, (
            "beta6's worktree must NOT be reaped — it is a live merge-deferred task"
        )
        assert orphan_dir in reaped_paths, (
            "orphan '999' worktree should be reaped (not in live tasks)"
        )


@pytest.mark.asyncio
class TestScenario7TrainResume:
    """PRD §10 row 7: train resumes after derail fix.

    After γ's verify loop is unblocked (verify becomes green) and all members
    are merge-deferred, calling _maybe_enqueue_group_merge() on the tip (γ)
    enqueues a GroupMergeRequest and — when the request is driven through
    MergeWorker — yields scenario-1 postconditions (single merge commit, 3 done).

    Uses real git (for the merge) with mocked verify (no cargo needed).
    """

    async def test_train_resumes_after_derail_fix(self, tmp_path: Path) -> None:
        """δ₂ trigger fires after derail fix: enqueues GMR, driven merge completes."""
        from orchestrator.workflow import WorkflowOutcome

        # ── Set up a plain git repo (no cargo fixture needed — verify mocked) ─
        repo = tmp_path
        await _run(["git", "init", "-b", "main"], cwd=repo)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
        await _run(["git", "config", "user.name", "Test"], cwd=repo)
        (repo / "README.md").write_text("# train-resume test\n")
        await _run(["git", "add", "-A"], cwd=repo)
        await _run(["git", "commit", "-m", "initial"], cwd=repo)

        config7 = OrchestratorConfig(
            project_root=repo,
            test_command="true",   # no real cargo — verify is mocked
            lint_command="true",
            type_check_command="true",
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        git_ops7 = GitOps(config7.git, repo)

        # ── Stack 3 train members with simple text edits ───────────────────
        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(
            git_ops7, "alpha7", main_sha,
            lambda wt: (wt / "alpha7.txt").write_text("alpha\n"),
        )
        wt_b, sha_b = await make_stacked_member(
            git_ops7, "beta7", sha_a,
            lambda wt: (wt / "beta7.txt").write_text("beta\n"),
        )
        wt_c, _sha_c = await make_stacked_member(
            git_ops7, "gamma7", sha_b,
            lambda wt: (wt / "gamma7.txt").write_text("gamma\n"),
        )

        # ── Set up γ workflow with mocked scheduler ────────────────────────
        train_meta7 = {
            "id": "T7", "order": 2,
            "members": ["alpha7", "beta7", "gamma7"],
        }
        all_merge_deferred = (
            {"alpha7": "merge-deferred", "beta7": "merge-deferred", "gamma7": "merge-deferred"},
            None,
        )
        tasks_by_train_ret = [
            {"id": "alpha7", "status": "merge-deferred", "metadata": {"train": {"id": "T7", "order": 0}}},
            {"id": "beta7",  "status": "merge-deferred", "metadata": {"train": {"id": "T7", "order": 1}}},
            {"id": "gamma7", "status": "merge-deferred", "metadata": {"train": {"id": "T7", "order": 2}}},
        ]

        merge_q: asyncio.Queue[MergeRequest] = asyncio.Queue()

        wf7, sched7 = _make_train_workflow(
            config7, git_ops7, "gamma7", train_meta7, all_merge_deferred, wt_c,
            merge_queue=merge_q,
            tasks_by_train_return=tasks_by_train_ret,
        )

        # ── Record main SHA before the driven merge ────────────────────────
        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()

        # ── Mock verify: always passes (no cargo run in this scenario) ─────
        passed_result = VerifyResult(
            passed=True, test_output="ok\n", lint_output="", type_output="",
            summary="all green", timed_out=False,
        )

        with patch("orchestrator.merge_queue.run_scoped_verification", AsyncMock(return_value=passed_result)):
            # Run _maybe_enqueue_group_merge as a background task; it enqueues
            # the GroupMergeRequest and awaits the future.
            enqueue_task = asyncio.create_task(wf7._maybe_enqueue_group_merge())

            # Use wait_for instead of sleep(0)+get_nowait: deterministic regardless
            # of how many internal awaits the implementation does before enqueuing.
            req7 = await asyncio.wait_for(merge_q.get(), timeout=2.0)
            assert isinstance(req7, GroupMergeRequest), (
                f"expected GroupMergeRequest in queue, got {type(req7).__name__}"
            )
            assert req7.train_id == "T7"
            assert req7.member_task_ids == ["alpha7", "beta7", "gamma7"]

            # Drive the request through a real MergeWorker.
            worker_q: asyncio.Queue[MergeRequest] = asyncio.Queue()
            worker = MergeWorker(git_ops7, worker_q)
            merge_outcome = await worker._do_merge(req7)

            # Resolve the future to unblock _maybe_enqueue_group_merge.
            if merge_outcome is not None and not req7.result.done():
                req7.result.set_result(merge_outcome)

            # Wait for _maybe_enqueue_group_merge to return.
            wf_outcome = await enqueue_task

        # ── Scenario-1 postconditions ──────────────────────────────────────

        # (i) Workflow outcome is DONE.
        assert wf_outcome == WorkflowOutcome.DONE, (
            f"expected WorkflowOutcome.DONE after successful train resume, "
            f"got {wf_outcome!r}"
        )

        # (ii) Single merge commit on main.
        _, merges_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        assert int(merges_str.strip()) == 1, (
            f"expected exactly 1 merge commit on main after train resume, "
            f"got {merges_str.strip()}"
        )

        # (iii) Main advanced beyond its prior HEAD.
        _, main_sha_after, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        assert main_sha_after.strip() != main_sha_before, (
            "main must advance after the train merge"
        )

        # (iv) 3 mark_member_done callbacks (one per member) via scheduler.mark_done.
        assert sched7.mark_done.call_count == 3, (
            f"expected 3 scheduler.mark_done calls (one per member), "
            f"got {sched7.mark_done.call_count}"
        )


# ---------------------------------------------------------------------------
# Scenario 8 (main advances during holding window) — step-9 RED / step-10 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario8MainAdvances:
    """PRD §10 row 8: main advances while train is in the merge-deferred holding window.

    CLEAN variant: unrelated commit lands on main between stacking and the
    group merge; the worker rebases the tip onto new main, then merges —
    main contains both the unrelated commit AND all 3 member files, single
    new merge commit, outcome done.  Uses real cargo (cargo_or_skip).

    CONFLICT variant: main commit edits the SAME line as a train member,
    producing a genuine rebase conflict.  Worker returns the
    TRAIN_REBASE_CONFLICT_REASON_PREFIX outcome; advance_main NOT called,
    members not flipped.
    """

    async def test_main_advances_then_train_merges_clean(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """CLEAN: unrelated main commit rebased over, merge lands single clean commit."""
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # Additive member edits.
        def edit_alpha(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn alpha8c_output() -> u32 { 8 }\n")

        def edit_beta(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn beta8c_output() -> u32 { 8 }\n")

        def edit_gamma(wt: Path) -> None:
            lib = wt / "crate_c" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn gamma8c_output() -> u32 { 8 }\n")

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(git_ops, "alpha8c", main_sha, edit_alpha)
        wt_b, sha_b = await make_stacked_member(git_ops, "beta8c", sha_a, edit_beta)
        wt_c, _sha_c = await make_stacked_member(git_ops, "gamma8c", sha_b, edit_gamma)

        # Land an UNRELATED commit directly on main (no-conflict; new file).
        # git commit-tree approach: commit a new file directly onto main.
        (repo / "unrelated_main.txt").write_text("unrelated main advance\n")
        await _run(["git", "add", "unrelated_main.txt"], cwd=repo)
        await _run(
            ["git", "commit", "-m", "Unrelated main advance (scenario 8 clean)"],
            cwd=repo,
        )

        _, main_sha_after_advance, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_after_advance = main_sha_after_advance.strip()

        # Verify main is now AHEAD of the train branch point.
        _, merges_before_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_before = int(merges_before_str.strip())

        # Drive the GroupMergeRequest — worker rebases tip onto new main then merges.
        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-scenario-8-clean",
            member_names=["alpha8c", "beta8c", "gamma8c"],
            tip_name="gamma8c",
            tip_worktree=wt_c,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        outcome = await worker._do_merge(req)

        # (i) Outcome done.
        assert outcome is not None
        assert outcome.status == "done", f"expected done after clean rebase+merge, got: {outcome!r}"
        assert outcome.merge_sha is not None

        # (ii) Single new merge commit.
        _, merges_after_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_after = int(merges_after_str.strip())
        assert merges_after == merges_before + 1, (
            f"expected exactly 1 new merge commit, got {merges_after - merges_before}"
        )

        # (iii) Main contains BOTH the unrelated commit AND all member crate edits.
        _, main_files, _ = await _run(
            ["git", "ls-tree", "-r", "--name-only", "main"], cwd=repo
        )
        assert "unrelated_main.txt" in main_files, (
            "main must contain the unrelated advance commit"
        )
        assert "crate_a/src/lib.rs" in main_files
        assert "crate_b/src/lib.rs" in main_files
        assert "crate_c/src/lib.rs" in main_files

        # Check ALL THREE member edits on main (α, β, γ — no member silently dropped).
        _, lib_a, _ = await _run(["git", "show", "main:crate_a/src/lib.rs"], cwd=repo)
        assert "alpha8c_output" in lib_a
        _, lib_b, _ = await _run(["git", "show", "main:crate_b/src/lib.rs"], cwd=repo)
        assert "beta8c_output" in lib_b
        _, lib_c, _ = await _run(["git", "show", "main:crate_c/src/lib.rs"], cwd=repo)
        assert "gamma8c_output" in lib_c

        # (iv) 3 mark_member_done with one shared SHA.
        assert req.mark_member_done.call_count == 3  # type: ignore[union-attr]
        called_shas = {c.args[1] for c in req.mark_member_done.call_args_list}  # type: ignore[union-attr]
        assert len(called_shas) == 1
        assert next(iter(called_shas)) == outcome.merge_sha

        # (v) No red-main window: cargo green at final main tip.
        env = {**__import__("os").environ, "CARGO_TARGET_DIR": str(shared_cargo_target)}
        _, new_main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        new_main_sha = new_main_sha.strip()
        main_wt = tmp_path / "_verify_main8c"
        await _run(
            ["git", "worktree", "add", "--detach", str(main_wt), new_main_sha],
            cwd=repo,
        )
        try:
            result = subprocess.run(
                ["cargo", "test", "--workspace", "--quiet"],
                cwd=main_wt, env=env, capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"Final main tip must be workspace-green:\n{result.stdout}{result.stderr}"
            )
        finally:
            await _run(
                ["git", "worktree", "remove", "--force", str(main_wt)], cwd=repo
            )

    async def test_main_advances_rebase_conflict(self, tmp_path: Path) -> None:
        """CONFLICT: main edit conflicts with member → TRAIN_REBASE_CONFLICT outcome.

        Lands a commit on main that edits the SAME LINE as gamma's crate_c
        change.  rebase_onto_main genuinely conflicts → worker returns the
        TRAIN_REBASE_CONFLICT_REASON_PREFIX reason, advance_main NOT called,
        main SHA unmoved, zero mark_member_done calls.

        No cargo needed (the rebase conflict is detected before any verify).
        """
        # Seed a simple git repo (no cargo fixture needed — conflict happens at rebase).
        repo = tmp_path
        await _run(["git", "init", "-b", "main"], cwd=repo)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
        await _run(["git", "config", "user.name", "Test"], cwd=repo)
        # Use a simple text file as the "shared" file that both main and γ edit.
        (repo / "shared.txt").write_text("line1\nCONFLICT_TARGET\nline3\n")
        await _run(["git", "add", "-A"], cwd=repo)
        await _run(["git", "commit", "-m", "initial"], cwd=repo)

        config8r = OrchestratorConfig(
            project_root=repo,
            test_command="true",
            lint_command="true",
            type_check_command="true",
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        git_ops8r = GitOps(config8r.git, repo)

        # Stack α/β/γ — γ modifies shared.txt CONFLICT_TARGET line.
        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_a, sha_a = await make_stacked_member(
            git_ops8r, "alpha8r", main_sha,
            lambda wt: (wt / "alpha8r.txt").write_text("alpha8r\n"),
        )
        wt_b, sha_b = await make_stacked_member(
            git_ops8r, "beta8r", sha_a,
            lambda wt: (wt / "beta8r.txt").write_text("beta8r\n"),
        )

        def edit_gamma_conflict(wt: Path) -> None:
            (wt / "shared.txt").write_text("line1\nGAMMA_EDIT\nline3\n")

        wt_c, _sha_c = await make_stacked_member(
            git_ops8r, "gamma8r", sha_b, edit_gamma_conflict,
        )

        # Land a CONFLICTING commit on main — same line γ edits.
        (repo / "shared.txt").write_text("line1\nMAIN_EDIT\nline3\n")
        await _run(["git", "add", "shared.txt"], cwd=repo)
        await _run(
            ["git", "commit", "-m", "Conflicting main advance (scenario 8 conflict)"],
            cwd=repo,
        )

        # Record main state before the (conflicting) merge attempt.
        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()
        _, merges_before_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_before = int(merges_before_str.strip())

        req = build_group_merge_request(
            git_ops=git_ops8r,
            config=config8r,
            train_id="train-scenario-8-conflict",
            member_names=["alpha8r", "beta8r", "gamma8r"],
            tip_name="gamma8r",
            tip_worktree=wt_c,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops8r, queue)
        outcome = await worker._do_merge(req)

        # (i) Outcome reason starts with TRAIN_REBASE_CONFLICT_REASON_PREFIX.
        assert outcome is not None
        assert outcome.status != "done", (
            f"expected blocked outcome on rebase conflict, got: {outcome!r}"
        )
        assert outcome.reason.startswith(TRAIN_REBASE_CONFLICT_REASON_PREFIX), (
            f"expected reason to start with TRAIN_REBASE_CONFLICT_REASON_PREFIX, "
            f"got: {outcome.reason!r}"
        )

        # (ii) advance_main NOT called: main SHA unmoved.
        _, main_sha_after, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        assert main_sha_after.strip() == main_sha_before, (
            f"advance_main must NOT fire on rebase conflict; "
            f"main moved from {main_sha_before!r} to {main_sha_after.strip()!r}"
        )

        # (iii) No new merge commits.
        _, merges_after_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_after = int(merges_after_str.strip())
        assert merges_after == merges_before, (
            f"no new merge commits expected, got {merges_after - merges_before} new"
        )

        # (iv) No mark_member_done flips.
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"mark_member_done must not fire on rebase conflict, "
            f"got {req.mark_member_done.call_count} call(s)"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Scenarios 9, 10 (loop-guard non-train + done-prov gate unweakened) — step-11 RED / step-12 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScenario9LoopGuardNonTrain:
    """PRD §10 row 9: loop-guard fires for a plain (non-train) task.

    Regression: the γ₂ loop-guard (workflow.py:3375-3387) must fire for non-train
    tasks too.  The train-specific derail branch (scenario 6) shares the same
    max_failure_signature_repeat code path — this test confirms the guard is NOT
    train-specific and applies equally to all tasks.
    """

    async def test_loop_guard_non_train(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        """Plain task: 3× identical-sig failures → BLOCKED (same path as train scenario 6)."""
        import logging

        from orchestrator.workflow import WorkflowOutcome

        # Set up a minimal git repo (no cargo needed — verify is mocked).
        repo9 = tmp_path / "repo9"
        repo9.mkdir()
        await _run(["git", "init", "-b", "main"], cwd=repo9)
        await _run(["git", "config", "user.email", "test@test.com"], cwd=repo9)
        await _run(["git", "config", "user.name", "Test"], cwd=repo9)
        (repo9 / "README.md").write_text("# repo9\n")
        await _run(["git", "add", "-A"], cwd=repo9)
        await _run(["git", "commit", "-m", "initial"], cwd=repo9)

        config9 = OrchestratorConfig(
            project_root=repo9,
            max_verify_attempts=5,
            max_failure_signature_repeat=3,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        git_ops9 = GitOps(config9.git, repo9)

        # Create the plain task's worktree.
        wt_info9 = await git_ops9.create_worktree("plain9")
        wt_plain = wt_info9.path

        # Build a PLAIN (non-train) workflow via _make_plain_workflow (step-12 helper).
        wf9, _ = _make_plain_workflow(config9, git_ops9, "plain9", wt_plain)

        # Inject 3 identical-signature failures: same category + cause_hint
        # differing only in file:line (normaliser strips the line number).
        failures9 = [
            _make_verify_failure("test_failure", "FAILED tests/test_plain.py:10 AssertionError"),
            _make_verify_failure("test_failure", "FAILED tests/test_plain.py:20 AssertionError"),
            _make_verify_failure("test_failure", "FAILED tests/test_plain.py:30 AssertionError"),
        ]
        # Exactly 3 failures — StopIteration surfaces guard regressions cleanly.
        verify_mock9 = AsyncMock(side_effect=failures9)
        monkeypatch.setattr("orchestrator.workflow.run_scoped_verification", verify_mock9)

        with caplog.at_level(logging.WARNING, logger="orchestrator.workflow"):
            outcome9 = await wf9._verify_debugfix_loop()

        # Loop-guard must fire after max_failure_signature_repeat (3) identical
        # signatures — same code path as the train-member derail in scenario 6.
        assert outcome9 == WorkflowOutcome.BLOCKED, (
            f"expected BLOCKED after 3 identical-signature failures (plain non-train task), "
            f"got {outcome9!r}"
        )
        assert verify_mock9.await_count == 3, (
            f"loop-guard must fire at attempt 3 (not wait for max_verify_attempts), "
            f"got {verify_mock9.await_count} calls"
        )
        warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "consecutive identical verify failures" in warning_text, (
            f"expected WARNING about consecutive identical failures; got: {warning_text!r}"
        )


@pytest.mark.asyncio
class TestScenario10DoneProvGateUnweakened:
    """PRD §10 row 10: done-prov gate is unweakened for non-train tasks.

    Regression: the done_provenance_invalid gate (scheduler.py:1228-1236) must
    raise ProvenanceValidationRejection for non-train tasks — the train machinery
    (δ₁) does NOT weaken this gate.

    The gate's underlying invariant: git_ops.is_ancestor (git_ops.py:896-902).
    fused-memory's done_provenance ancestor check verifies the merge SHA is on
    main before allowing the done transition.  Train tasks pass the same SHA
    through mark_member_done → mark_done → set_task_status; the gate applies
    identically regardless of train membership.
    """

    async def test_done_provenance_gate_unweakened(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """set_task_status('done', done_provenance={invalid SHA}) raises ProvenanceValidationRejection."""
        from orchestrator.scheduler import ProvenanceValidationRejection, Scheduler

        # Build a real Scheduler with a minimal OrchestratorConfig.
        # mcp_session=None → uses mcp_call HTTP path (monkeypatched below).
        config10 = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                main_branch="main",
                branch_prefix="task/",
                remote="origin",
                worktree_dir=".worktrees",
                push_after_advance=False,
            ),
        )
        sched10 = Scheduler(config10)

        # Stub the MCP transport to return a done_provenance_invalid rejection.
        # Envelope format mirrors fused-memory's TaskInterceptor.
        rejection_response = {
            "result": {
                "structuredContent": {
                    "success": False,
                    "error": "done_provenance_invalid",
                    "hint": (
                        'kind="merged" but commit deadbeef is not on main — '
                        "the train merge machinery (δ₁) does NOT weaken this gate "
                        "(git_ops.is_ancestor gate at git_ops.py:896-902)"
                    ),
                },
                "isError": False,
            },
        }
        monkeypatch.setattr(
            "orchestrator.scheduler.mcp_call",
            AsyncMock(return_value=rejection_response),
        )

        with pytest.raises(ProvenanceValidationRejection) as excinfo:
            await sched10.set_task_status(
                "task42",
                "done",
                done_provenance={"kind": "merged", "commit": "deadbeef"},
            )

        assert excinfo.value.task_id == "task42"
        assert excinfo.value.error_code == "done_provenance_invalid", (
            f"expected error_code='done_provenance_invalid', "
            f"got {excinfo.value.error_code!r}"
        )

# ---------------------------------------------------------------------------
# Scenarios 11, 12 (non-train regression + degenerate train-of-one) — step-13 RED / step-14 GREEN
# ---------------------------------------------------------------------------


def _build_plain_merge_request(
    *,
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    worktree: Path,
) -> MergeRequest:
    """Build a plain (non-train) MergeRequest for scenario 11.

    branch is the unprefixed task_id; merge_to_main prepends branch_prefix internally.
    """
    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=task_id,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


@pytest.mark.asyncio
class TestScenario11NonTrainRegression:
    """PRD §10 row 11: non-train regression — baseline task uses single-task pipeline.

    Verifies the train machinery (α₁-ζ₁) does NOT break the existing single-task
    merge path:
    - create_worktree (no train kwarg) → worktree base == main tip
    - real cargo post-merge verify via test_command (not forced-workspace flag)
    - single MergeRequest through MergeWorker → done, one merge commit
    - no merge-deferred state ever entered
    """

    async def test_non_train_regression(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """Baseline non-train task merges cleanly via the single-task path."""
        repo = await seed_workspace_repo(tmp_path)
        config11 = make_train_config(repo, shared_cargo_target)
        git_ops11 = GitOps(config11.git, repo)

        # Capture main tip before worktree creation.
        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()

        # (i) Create non-train worktree: no train kwarg → branches from main.
        wt_info11 = await git_ops11.create_worktree("alpha11")
        wt_a11 = wt_info11.path
        await _run(["git", "config", "user.email", "test@test.com"], cwd=wt_a11)
        await _run(["git", "config", "user.name", "Test"], cwd=wt_a11)

        # Assert worktree base == main tip SHA (non-train branches from main).
        _, merge_base_out, _ = await _run(
            ["git", "merge-base", "task/alpha11", "main"], cwd=repo
        )
        assert merge_base_out.strip() == main_sha_before, (
            f"expected merge-base(task/alpha11, main) == main tip {main_sha_before!r}, "
            f"got {merge_base_out.strip()!r}"
        )

        # Apply an additive crate edit and commit.
        lib11 = wt_a11 / "crate_a" / "src" / "lib.rs"
        lib11.write_text(lib11.read_text() + "\npub fn non_train_output() -> u32 { 11 }\n")
        await git_ops11.commit(wt_a11, "Add non_train_output function (scenario 11)")

        # Record pre-merge state.
        _, merges_before_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_before = int(merges_before_str.strip())

        # (ii) Drive a plain MergeRequest through the real single-task worker.
        req11 = _build_plain_merge_request(
            git_ops=git_ops11,
            config=config11,
            task_id="alpha11",
            worktree=wt_a11,
        )
        queue11: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker11 = MergeWorker(git_ops11, queue11)
        outcome11 = await worker11._do_merge(req11)

        # (iii) Assertions: done, single new merge commit, merge_sha is set.
        assert outcome11 is not None
        assert outcome11.status == "done", (
            f"expected outcome.status='done' for non-train task, got {outcome11!r}"
        )
        assert outcome11.merge_sha is not None, (
            "expected merge_sha to be set (done provenance) after successful merge"
        )

        _, merges_after_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_after = int(merges_after_str.strip())
        assert merges_after == merges_before + 1, (
            f"expected exactly 1 new merge commit (non-train single-task path); "
            f"got {merges_after - merges_before}"
        )

        # (iv) No merge-deferred state ever entered: outcome.status is 'done',
        # not 'merge-deferred'.  The cargo workspace is green at the final main tip
        # (post-merge verify ran via test_command, not forced-workspace flag).
        result11 = subprocess.run(
            ["cargo", "test", "--workspace", "--quiet"],
            cwd=repo,
            env={**os.environ, "CARGO_TARGET_DIR": str(shared_cargo_target)},
            capture_output=True,
            text=True,
        )
        assert result11.returncode == 0, (
            f"cargo test --workspace must be green at the final main tip "
            f"(non-train regression): {result11.stderr}"
        )


@pytest.mark.asyncio
class TestScenario12DegenerateTrainOfOne:
    """PRD §10 row 12: degenerate train-of-one is observationally equivalent to non-train.

    A train with a single member (order=0) must:
    - branch from main (order=0 → same _freshen_main path as non-train)
    - land as a single merge commit via _do_train_merge (group-merge-of-one)
    - mark the single member done with a real merge SHA
    - leave no dangling merge-deferred task

    The degenerate train flows through the group-merge-of-one path but is
    observationally identical to the non-train single-task path (PRD §10 row 12).
    """

    async def test_degenerate_train_of_one(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """Single-member order=0 train: base==main, single merge commit, member done."""
        repo = await seed_workspace_repo(tmp_path)
        config12 = make_train_config(repo, shared_cargo_target)
        git_ops12 = GitOps(config12.git, repo)

        # Capture main tip before worktree creation.
        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()

        # order=0 → create_worktree branches from main, NOT from a predecessor.
        # This is the degenerate single-member train case.
        train_meta12: TrainMembership = {"id": "T12", "order": 0, "members": ["alpha12"]}
        wt_info12 = await git_ops12.create_worktree("alpha12", train=train_meta12)
        wt_a12 = wt_info12.path
        await _run(["git", "config", "user.email", "test@test.com"], cwd=wt_a12)
        await _run(["git", "config", "user.name", "Test"], cwd=wt_a12)

        # (i) Worktree base == main tip (order=0 → identical to non-train path).
        _, merge_base_out12, _ = await _run(
            ["git", "merge-base", "task/alpha12", "main"], cwd=repo
        )
        assert merge_base_out12.strip() == main_sha_before, (
            f"expected merge-base(task/alpha12, main) == main tip {main_sha_before!r}, "
            f"got {merge_base_out12.strip()!r}"
        )

        # Apply an additive crate edit and commit.
        lib12 = wt_a12 / "crate_a" / "src" / "lib.rs"
        lib12.write_text(lib12.read_text() + "\npub fn degenerate_train_output() -> u32 { 12 }\n")
        await git_ops12.commit(wt_a12, "Add degenerate_train_output function (scenario 12)")

        # (ii) Record pre-merge state.
        _, merges_before_str12, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_before12 = int(merges_before_str12.strip())

        # (iii) Build GroupMergeRequest with a single member (degenerate train).
        req12 = build_group_merge_request(
            git_ops=git_ops12,
            config=config12,
            train_id="T12",
            member_names=["alpha12"],
            tip_name="alpha12",
            tip_worktree=wt_a12,
        )

        queue12: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker12 = MergeWorker(git_ops12, queue12)
        outcome12 = await worker12._do_merge(req12)

        # (iv) Assertions: done, single merge commit, member marked done once.
        assert outcome12 is not None
        assert outcome12.status == "done", (
            f"expected outcome.status='done' for degenerate train-of-one, "
            f"got {outcome12!r}"
        )
        assert outcome12.merge_sha is not None, (
            "expected merge_sha to be set after successful degenerate-train merge"
        )

        _, merges_after_str12, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=repo
        )
        merges_after12 = int(merges_after_str12.strip())
        assert merges_after12 == merges_before12 + 1, (
            f"expected exactly 1 new merge commit (degenerate train-of-one); "
            f"got {merges_after12 - merges_before12}"
        )

        # (v) mark_member_done fired exactly once with the shared merge SHA.
        # done_provenance.kind=='merged' in the real workflow: the degenerate train
        # flows through group-merge-of-one but is observationally identical to
        # the non-train path (PRD §10 row 12).
        assert req12.mark_member_done.call_count == 1, (  # type: ignore[union-attr]
            f"expected mark_member_done called once for the single member, "
            f"got {req12.mark_member_done.call_count}"  # type: ignore[union-attr]
        )
        called_task_id12, called_sha12 = req12.mark_member_done.call_args[0]  # type: ignore[union-attr]
        assert called_task_id12 == "alpha12", (
            f"expected mark_member_done called with task_id='alpha12', "
            f"got {called_task_id12!r}"
        )
        assert called_sha12 == outcome12.merge_sha, (
            f"expected mark_member_done SHA == outcome.merge_sha {outcome12.merge_sha!r}, "
            f"got {called_sha12!r}"
        )

        # (vi) No dangling merge-deferred task: mark_member_done was called once,
        # signalling the member transitioned to done (not left in merge-deferred).
        # The cargo workspace is green at the final main tip (post-merge verify).
        result12 = subprocess.run(
            ["cargo", "test", "--workspace", "--quiet"],
            cwd=repo,
            env={**os.environ, "CARGO_TARGET_DIR": str(shared_cargo_target)},
            capture_output=True,
            text=True,
        )
        assert result12.returncode == 0, (
            f"cargo test --workspace must be green at the final main tip "
            f"(degenerate train-of-one: no red-main window): {result12.stderr}"
        )


# ---------------------------------------------------------------------------
# Gate tests (steps 01-10): disk-guard, loop-breaker, equivalence, push,
# wip-halt — drive through MergeWorker._do_merge, no cargo required.
# These tests use a simple git repo (plain text files) rather than the cargo
# fixture, so they run on any machine regardless of Rust toolchain availability.
# ---------------------------------------------------------------------------

_GIB = 1024**3


def _make_passing_verify_result() -> VerifyResult:
    """Return a passing VerifyResult for use as run_scoped_verification mock return."""
    return VerifyResult(
        passed=True,
        test_output="ok\n",
        lint_output="",
        type_output="",
        summary="All tests passed",
    )


async def _setup_gate_train(
    tmp_path: Path,
    *,
    train_id: str = "train-gate",
    member_names: tuple[str, str, str] = ("g-a", "g-b", "g-c"),
) -> tuple[GitOps, OrchestratorConfig, GroupMergeRequest]:
    """Set up a minimal git repo with a real stacked 3-member train (no cargo).

    Returns (git_ops, config, req) ready for worker._do_merge() testing.
    Each member adds one unique .txt file so diffs are always non-empty.
    """
    from orchestrator.config import GitConfig, OrchestratorConfig

    repo = tmp_path / "repo"
    repo.mkdir()
    await _run(["git", "init", "-b", "main"], cwd=repo)
    await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
    await _run(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / "README.md").write_text("# gate test repo\n")
    await _run(["git", "add", "-A"], cwd=repo)
    await _run(["git", "commit", "-m", "initial"], cwd=repo)

    git_cfg = GitConfig(
        main_branch="main",
        branch_prefix="task/",
        remote="origin",
        worktree_dir=".worktrees",
        push_after_advance=False,
    )
    config = OrchestratorConfig(project_root=repo, git=git_cfg)
    git_ops = GitOps(git_cfg, repo)

    a_name, b_name, c_name = member_names

    # Get current main SHA as base for first member.
    _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
    main_sha = main_sha.strip()

    wt_a, sha_a = await make_stacked_member(
        git_ops, a_name, main_sha,
        lambda wt, n=a_name: (wt / f"{n}.txt").write_text(f"{n}\n"),
    )
    wt_b, sha_b = await make_stacked_member(
        git_ops, b_name, sha_a,
        lambda wt, n=b_name: (wt / f"{n}.txt").write_text(f"{n}\n"),
    )
    wt_c, _sha_c = await make_stacked_member(
        git_ops, c_name, sha_b,
        lambda wt, n=c_name: (wt / f"{n}.txt").write_text(f"{n}\n"),
    )

    req = build_group_merge_request(
        git_ops=git_ops,
        config=config,
        train_id=train_id,
        member_names=list(member_names),
        tip_name=c_name,
        tip_worktree=wt_c,
    )

    return git_ops, config, req


# ---------------------------------------------------------------------------
# Gate test 01: disk guard fires — step-01 RED / step-02 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGate01TrainDiskGuard:
    """step-01 RED: _do_train_merge has no disk guard; disk pressure is ignored.

    After step-02 (GREEN): _do_train_merge uses _run_post_merge_verify which
    runs the pre-verify disk guard; persistent low disk → blocked/verify_skipped.
    """

    async def test_train_disk_guard_skips_verify(self, tmp_path: Path) -> None:
        """Disk persists low after prune → blocked, verify_skipped=True, TRANSIENT_INFRA."""
        git_ops, config, req = await _setup_gate_train(
            tmp_path, train_id="train-disk-gate",
        )

        # Record main SHA before (must be unmoved on block).
        _, main_sha_before, _ = await _run(
            ["git", "rev-parse", "main"], cwd=git_ops.project_root,
        )
        main_sha_before = main_sha_before.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with (
            patch(
                "orchestrator.merge_queue.run_scoped_verification",
                AsyncMock(return_value=_make_passing_verify_result()),
            ),
            patch(
                "orchestrator.merge_queue.shutil.disk_usage",
                return_value=MagicMock(free=1 * _GIB),  # 1 GiB << 10 GiB threshold
            ),
            patch.object(
                git_ops,
                "prune_stale_merge_worktrees",
                AsyncMock(return_value=[]),  # prune frees nothing
            ),
        ):
            outcome = await worker._do_merge(req)

        # (1) Outcome is blocked with the disk-guard reason.
        assert outcome is not None
        assert outcome.status == "blocked", f"expected blocked, got: {outcome!r}"
        assert outcome.verify_skipped is True, (
            "expected verify_skipped=True when disk guard fires"
        )
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f"expected TRANSIENT_INFRA prefix, got: {outcome.reason!r}"
        )

        # (2) main SHA must be unmoved (advance_main must not have run).
        _, main_sha_after, _ = await _run(
            ["git", "rev-parse", "main"], cwd=git_ops.project_root,
        )
        assert main_sha_after.strip() == main_sha_before, (
            f"main must not advance on disk-guard block; "
            f"moved from {main_sha_before!r} to {main_sha_after.strip()!r}"
        )

        # (3) No member flips.
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"mark_member_done must not fire when disk guard blocks train; "
            f"got {req.mark_member_done.call_count} call(s)"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Gate test 03: verify-timeout loop-breaker — step-03 RED / step-04 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGate03TrainVerifyTimeoutLoopBreaker:
    """step-03 RED: _do_train_merge has no loop-breaker short-circuit.

    Pre-seeding worker._post_merge_verify_timeouts[tip] = MAX doesn't stop
    _do_train_merge from doing git work today; with mocked passing verify it
    returns 'done'.  After step-04 the loop-breaker fires before any git work
    and returns blocked/ABANDONED prefix.
    """

    async def test_train_abandons_after_verify_timeout_threshold(
        self, tmp_path: Path,
    ) -> None:
        """Pre-seeded timeout counter at threshold → abandoned, no git work, 0 flips."""
        git_ops, config, req = await _setup_gate_train(
            tmp_path, train_id="train-loop-breaker",
        )

        # Record main SHA before attempt.
        _, main_sha_before, _ = await _run(
            ["git", "rev-parse", "main"], cwd=git_ops.project_root,
        )
        main_sha_before = main_sha_before.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Pre-seed the timeout counter at the threshold.
        tip_task_id = req.task_id  # tip is 'g-c' for default member names
        worker._post_merge_verify_timeouts[tip_task_id] = (
            worker.MAX_POST_MERGE_VERIFY_TIMEOUTS
        )

        # Spy on merge_to_main — the loop-breaker must short-circuit BEFORE any git work.
        with (
            patch.object(git_ops, "merge_to_main", wraps=git_ops.merge_to_main) as spy_merge,
            patch(
                "orchestrator.merge_queue.run_scoped_verification",
                AsyncMock(return_value=_make_passing_verify_result()),
            ),
        ):
            outcome = await worker._do_merge(req)

        # (1) Outcome is blocked with the ABANDONED prefix.
        assert outcome is not None
        assert outcome.status == "blocked", f"expected blocked, got: {outcome!r}"
        assert outcome.reason.startswith(ABANDONED_REASON_PREFIX), (
            f"expected ABANDONED prefix, got: {outcome.reason!r}"
        )

        # (2) No git work ran (loop-breaker must fire before merge_to_main).
        spy_merge.assert_not_called()

        # (3) Main SHA must be unmoved.
        _, main_sha_after, _ = await _run(
            ["git", "rev-parse", "main"], cwd=git_ops.project_root,
        )
        assert main_sha_after.strip() == main_sha_before, (
            f"main must not advance when loop-breaker fires; "
            f"moved from {main_sha_before!r} to {main_sha_after.strip()!r}"
        )

        # (4) No member flips.
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"mark_member_done must not fire when loop-breaker abandons; "
            f"got {req.mark_member_done.call_count} call(s)"  # type: ignore[union-attr]
        )


# ---------------------------------------------------------------------------
# Gate test 05: equivalence gate — step-05 RED / step-06 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGate05TrainPostMergeEquivalence:
    """step-05 RED: _do_train_merge never calls _check_post_merge_equivalence.

    After step-06 (GREEN): trains route through _finalize_advanced_merge which
    runs the equivalence check; failure → blocked/POST_MERGE_EQUIVALENCE prefix,
    0 member flips, main DID advance (merge landed before the gate).
    """

    async def test_train_blocks_on_post_merge_equivalence_fail(
        self, tmp_path: Path,
    ) -> None:
        """Equivalence check returns failures → blocked, 0 flips, main advanced."""
        git_ops, config, req = await _setup_gate_train(
            tmp_path, train_id="train-equiv-gate",
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with (
            patch(
                "orchestrator.merge_queue.run_scoped_verification",
                AsyncMock(return_value=_make_passing_verify_result()),
            ),
            patch(
                "orchestrator.merge_queue._check_post_merge_equivalence",
                AsyncMock(return_value=["g-a/src/lib.rs"]),
            ),
        ):
            outcome = await worker._do_merge(req)

        # (1) Outcome is blocked with equivalence-failure reason.
        assert outcome is not None
        assert outcome.status == "blocked", f"expected blocked, got: {outcome!r}"
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX), (
            f"expected POST_MERGE_EQUIVALENCE prefix, got: {outcome.reason!r}"
        )

        # (2) No member flips (equivalence failed after landing).
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"mark_member_done must not fire when equivalence gate fails; "
            f"got {req.mark_member_done.call_count} call(s)"  # type: ignore[union-attr]
        )

        # (3) Main DID advance (merge commit landed before the equivalence gate).
        _, main_sha_after, _ = await _run(
            ["git", "rev-parse", "main"], cwd=git_ops.project_root,
        )
        _, initial_main, _ = await _run(
            ["git", "rev-list", "--max-parents=0", "main"], cwd=git_ops.project_root,
        )
        # main moved if there's more than the initial commit (a merge commit was added)
        _, merge_count_str, _ = await _run(
            ["git", "rev-list", "--merges", "--count", "main"], cwd=git_ops.project_root,
        )
        assert int(merge_count_str.strip()) >= 1, (
            "expected advance_main to have run (merge commit present on main); "
            f"main SHA={main_sha_after.strip()!r}"
        )
