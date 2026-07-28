"""Warm-lane script resolution: project override first, dark-factory second.

Task 3072 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α, Phase 1).

Companion to ``test_warm_lane_scripts_shipped.py``, which pins that
dark-factory's own copies exist under ``orchestrator/scripts/warm-lane/``.
This module pins how :class:`~orchestrator.git_ops.GitOps` *chooses* between
those copies and a project's own ``<project_root>/scripts/<name>`` override,
and pins that the choice is observable:

* **B7** — a project override always wins, and the resolved path + origin is
  logged at INFO on every invocation.
* **B8** — "no implementation at either location" is a WARNING naming both
  tried paths, never a DEBUG line naming one.

The dark-factory fallback root is seamed through ``ORCH_WARM_LANE_SCRIPT_DIR``,
read at call time.  ``orchestrator/tests/conftest.py`` pins it at a
guaranteed-absent directory for the whole suite (autouse) so the ~200 existing
tests that assert "no scripts/ dir → fail-soft sentinel" never execute the real
host scripts against a ``tmp_path``.  Tests here opt back in explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import orchestrator.git_ops as git_ops_mod
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

#: Env seam for the dark-factory fallback root.  Test hermeticity only —
#: production resolution is repo-relative and never reads this.
DF_DIR_ENV = 'ORCH_WARM_LANE_SCRIPT_DIR'


def _pin_df_dir(monkeypatch: pytest.MonkeyPatch, path: Path) -> Path:
    """Point the dark-factory fallback root at ``path`` for this test."""
    monkeypatch.setenv(DF_DIR_ENV, str(path))
    return path


def _make_config(**overrides: Any) -> GitConfig:
    """Canonical warm-lane GitConfig (mirrors test_warm_lane_disk_guard.py)."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_disk_guard=True,
        warm_lane_min_free_gib=50,
        warm_lane_min_free_inodes=500_000,
        **overrides,
    )


def _make_git_ops(project_root: Path, **overrides: Any) -> GitOps:
    """Build a GitOps rooted at ``project_root`` (no real repo needed here)."""
    return GitOps(_make_config(**overrides), project_root, warm_lane_pool_size=1)


def _write_stub(path: Path, body: str = 'exit 0\n', mode: int = 0o755) -> Path:
    """Write an executable stub script at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n' + body)
    path.chmod(mode)
    return path


class TestResolveWarmLaneScript:
    """``GitOps._resolve_warm_lane_script`` preference order.

    Contract: ``(path, origin)`` where origin is ``'project'`` or
    ``'dark-factory'``, or ``None`` when neither location has the script.
    """

    NAME = 'warm-lane-gc.sh'

    def test_project_copy_wins_when_both_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both present → the project override, tagged 'project' (B7)."""
        project_root = tmp_path / 'proj'
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        project_script = _write_stub(project_root / 'scripts' / self.NAME)
        _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (project_script, 'project'), (
            'A project that carries its own warm-lane tooling must keep it '
            f'(PRD D3); got {resolved!r}'
        )

    def test_dark_factory_copy_used_when_project_lacks_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the dark-factory copy present → it is used, tagged 'dark-factory'."""
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        df_script = _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (df_script, 'dark-factory'), (
            'A project with no warm-lane tooling must fall back to '
            f"dark-factory's shipped copy; got {resolved!r}"
        )

    def test_neither_location_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Neither present → None, so callers can emit the both-paths WARNING."""
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        _pin_df_dir(monkeypatch, tmp_path / 'absent-df-dir')

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved is None, (
            f'No implementation at either location must resolve to None; got {resolved!r}'
        )

    def test_project_copy_wins_even_when_not_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existence, not mode, is the predicate — mirrors today's ``script.exists()``.

        The pre-relocation call sites branched on ``script.exists()`` alone; a
        non-executable project copy therefore reached the subprocess spawn and
        failed there.  Preserving that keeps the failure attributable to the
        project's own broken script rather than silently substituting
        dark-factory's, which would mask it.
        """
        project_root = tmp_path / 'proj'
        df_dir = _pin_df_dir(monkeypatch, tmp_path / 'df')
        project_script = _write_stub(
            project_root / 'scripts' / self.NAME, mode=0o644,
        )
        _write_stub(df_dir / self.NAME)

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        assert resolved == (project_script, 'project'), (
            'Resolution must key on existence, not the execute bit, so a broken '
            f'project override stays attributable; got {resolved!r}'
        )

    def test_unset_env_falls_back_to_the_repo_relative_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the seam UNSET, the fallback root is orchestrator/scripts/warm-lane.

        This is the case that actually ships: production never sets
        ORCH_WARM_LANE_SCRIPT_DIR, so the repo-relative default is what resolves.
        Pinned explicitly because the autouse hermeticity fixture masks it
        everywhere else in the suite.
        """
        monkeypatch.delenv(DF_DIR_ENV, raising=False)
        project_root = tmp_path / 'proj'
        project_root.mkdir()

        resolved = _make_git_ops(project_root)._resolve_warm_lane_script(self.NAME)

        expected_dir = (
            Path(git_ops_mod.__file__).resolve().parents[2] / 'scripts' / 'warm-lane'
        )
        assert resolved is not None, (
            'With the env seam unset the repo-relative dark-factory copy must '
            'resolve — production depends on it and sets no env var'
        )
        path, origin = resolved
        assert path == expected_dir / self.NAME, (
            f'Expected the repo-relative copy at {expected_dir / self.NAME}; got {path}'
        )
        assert origin == 'dark-factory'

    def test_env_seam_is_read_at_call_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-pointing the seam takes effect without a module reload.

        The autouse conftest fixture and the opt-in override both rely on
        ``monkeypatch.setenv`` winning over an import-time snapshot.
        """
        project_root = tmp_path / 'proj'
        project_root.mkdir()
        git_ops = _make_git_ops(project_root)
        _pin_df_dir(monkeypatch, tmp_path / 'absent-df-dir')

        assert git_ops._resolve_warm_lane_script(self.NAME) is None

        later_dir = tmp_path / 'df-later'
        df_script = _write_stub(later_dir / self.NAME)
        _pin_df_dir(monkeypatch, later_dir)

        assert git_ops._resolve_warm_lane_script(self.NAME) == (
            df_script, 'dark-factory',
        )
