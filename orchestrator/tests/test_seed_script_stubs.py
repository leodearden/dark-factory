"""Contract tests for ``_seed_script_stubs.seed_default_warm_base``: the warm base
and ``.pool-root`` sentinel it writes are the ones GitOps itself reads.

GitOps exposes no public base-health seam, so these pin the base's path and
that it is non-empty, not the ``WarmBaseHealth.OK`` predicate itself.
"""
from __future__ import annotations

from pathlib import Path

from _seed_script_stubs import seed_default_warm_base

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / 'proj'
    root.mkdir()
    return root


def _pool_git_ops(root: Path) -> GitOps:
    return GitOps(GitConfig(warm_lane_pool=True), root, warm_lane_pool_size=1)


def _assert_non_empty_default_base(git_ops: GitOps) -> None:
    base = git_ops.warm_lane_base_target_path
    assert base.is_dir()
    assert any(base.iterdir())


def test_default_seed_gives_a_non_empty_default_base_and_present_pool_storage(
    tmp_path: Path,
) -> None:
    root = _project_root(tmp_path)

    seed_default_warm_base(root)

    git_ops = _pool_git_ops(root)
    _assert_non_empty_default_base(git_ops)
    assert git_ops.pool_storage_present() is True


def test_pool_root_false_seeds_the_base_but_leaves_pool_storage_absent(
    tmp_path: Path,
) -> None:
    root = _project_root(tmp_path)

    seed_default_warm_base(root, pool_root=False)

    git_ops = _pool_git_ops(root)
    _assert_non_empty_default_base(git_ops)
    assert git_ops.pool_storage_present() is False


def test_seeding_twice_is_harmless(tmp_path: Path) -> None:
    root = _project_root(tmp_path)

    seed_default_warm_base(root)
    seed_default_warm_base(root)

    git_ops = _pool_git_ops(root)
    _assert_non_empty_default_base(git_ops)
    assert git_ops.pool_storage_present() is True
