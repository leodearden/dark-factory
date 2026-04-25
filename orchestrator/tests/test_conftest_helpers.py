"""Contract tests for shared conftest fixtures.

Verifies the `mock_orch_config` fixture's default attribute contract so that
callers can rely on it without reading conftest.py implementation details.
"""

from unittest.mock import MagicMock

import pytest

from orchestrator.config import GitConfig


class TestMockOrchConfigFixture:
    """Focused per-attribute contract tests for mock_orch_config."""

    def test_returns_magic_mock_instance(self, mock_orch_config):
        assert isinstance(mock_orch_config, MagicMock)

    def test_git_is_gitconfig(self, mock_orch_config):
        assert isinstance(mock_orch_config.git, GitConfig)

    def test_git_main_branch(self, mock_orch_config):
        assert mock_orch_config.git.main_branch == 'main'

    def test_git_branch_prefix(self, mock_orch_config):
        assert mock_orch_config.git.branch_prefix == 'task/'

    def test_git_remote(self, mock_orch_config):
        assert mock_orch_config.git.remote == 'origin'

    def test_git_worktree_dir(self, mock_orch_config):
        assert mock_orch_config.git.worktree_dir == '.worktrees'

    def test_project_root_is_tmp_path(self, mock_orch_config, tmp_path):
        assert mock_orch_config.project_root == tmp_path

    def test_usage_cap_enabled_is_false(self, mock_orch_config):
        assert mock_orch_config.usage_cap.enabled is False

    def test_review_enabled_is_false(self, mock_orch_config):
        assert mock_orch_config.review.enabled is False

    def test_sandbox_backend_is_auto(self, mock_orch_config):
        assert mock_orch_config.sandbox.backend == 'auto'
