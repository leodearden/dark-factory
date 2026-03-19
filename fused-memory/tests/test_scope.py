"""Tests for resolve_project_id() — converts filesystem project_root to logical project_id."""

import pytest

from fused_memory.models.scope import resolve_project_id


class TestResolveProjectId:
    """Test resolve_project_id derivation logic."""

    def test_full_path_extracts_basename_and_normalizes(self):
        """'/home/leo/src/dark-factory' -> 'dark_factory'"""
        assert resolve_project_id('/home/leo/src/dark-factory') == 'dark_factory'

    def test_trailing_slash_stripped(self):
        """'/project/' -> 'project'"""
        assert resolve_project_id('/project/') == 'project'

    def test_simple_path(self):
        """'/project' -> 'project'"""
        assert resolve_project_id('/project') == 'project'

    def test_already_clean_id_passthrough(self):
        """'my_project' (no slashes, no hyphens) passes through unchanged."""
        assert resolve_project_id('my_project') == 'my_project'

    def test_multiple_hyphens(self):
        """'/foo/my-cool-project' -> 'my_cool_project'"""
        assert resolve_project_id('/foo/my-cool-project') == 'my_cool_project'

    def test_explicit_mapping_overrides_derivation(self):
        """Mapping dict takes precedence over derivation."""
        mapping = {'/home/leo/src/dark-factory': 'custom_id'}
        assert resolve_project_id('/home/leo/src/dark-factory', mapping=mapping) == 'custom_id'

    def test_mapping_miss_falls_back_to_derivation(self):
        """When path not in mapping, derive from basename."""
        mapping = {'/other/path': 'other_id'}
        assert resolve_project_id('/home/leo/src/dark-factory', mapping=mapping) == 'dark_factory'

    def test_lowercased(self):
        """Mixed-case basename is lowercased."""
        assert resolve_project_id('/home/user/MyProject') == 'myproject'

    def test_hyphens_and_case_combined(self):
        """Hyphens replaced and lowercased."""
        assert resolve_project_id('/srv/My-Cool-App') == 'my_cool_app'
