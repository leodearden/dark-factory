"""Tests for resolve_project_id() — converts filesystem project_root to logical project_id."""

from fused_memory.models.scope import (
    KNOWN_PROJECT_ROOTS_ENV,
    build_known_projects_map,
    known_project_roots_from_env,
    resolve_project_id,
)


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


class TestKnownProjectRootsFromEnv:
    """known_project_roots_from_env parses the configured comma-separated env var."""

    def test_unset_env_var_yields_empty(self, monkeypatch):
        monkeypatch.delenv(KNOWN_PROJECT_ROOTS_ENV, raising=False)
        assert known_project_roots_from_env() == []

    def test_empty_env_var_yields_empty(self, monkeypatch):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, '')
        assert known_project_roots_from_env() == []

    def test_single_root(self, monkeypatch):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, '/home/leo/src/reify')
        assert known_project_roots_from_env() == ['/home/leo/src/reify']

    def test_multiple_roots_comma_separated(self, monkeypatch):
        monkeypatch.setenv(
            KNOWN_PROJECT_ROOTS_ENV,
            '/home/leo/src/reify,/home/leo/src/dark-factory',
        )
        assert known_project_roots_from_env() == [
            '/home/leo/src/reify',
            '/home/leo/src/dark-factory',
        ]

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv(
            KNOWN_PROJECT_ROOTS_ENV,
            '  /home/leo/src/reify ,  /home/leo/src/dark-factory ',
        )
        assert known_project_roots_from_env() == [
            '/home/leo/src/reify',
            '/home/leo/src/dark-factory',
        ]

    def test_empty_entries_skipped(self, monkeypatch):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, '/a,,,/b,')
        assert known_project_roots_from_env() == ['/a', '/b']

    def test_custom_env_var(self, monkeypatch):
        monkeypatch.setenv('SOME_OTHER_VAR', '/x,/y')
        assert known_project_roots_from_env('SOME_OTHER_VAR') == ['/x', '/y']


class TestBuildKnownProjectsMap:
    """build_known_projects_map composes a {project_id → project_root} mapping."""

    def test_primary_root_only(self, tmp_path):
        # primary root path doesn't need to exist for this helper.
        d = tmp_path / 'reify'
        d.mkdir()
        result = build_known_projects_map(str(d), extra_roots=[])
        assert result == {'reify': str(d.resolve())}

    def test_primary_plus_extras(self, tmp_path):
        a = tmp_path / 'reify'
        b = tmp_path / 'dark-factory'
        a.mkdir()
        b.mkdir()
        result = build_known_projects_map(str(a), extra_roots=[str(b)])
        assert result == {
            'reify': str(a.resolve()),
            'dark_factory': str(b.resolve()),
        }

    def test_empty_primary_dropped(self, tmp_path):
        b = tmp_path / 'dark-factory'
        b.mkdir()
        result = build_known_projects_map('', extra_roots=[str(b)])
        assert result == {'dark_factory': str(b.resolve())}

    def test_duplicate_project_id_first_wins(self, tmp_path):
        a = tmp_path / 'project_x'
        a.mkdir()
        sub = tmp_path / 'sub'
        sub.mkdir()
        b = sub / 'project_x'  # different parent, same basename → same project_id
        b.mkdir()
        result = build_known_projects_map(str(a), extra_roots=[str(b)])
        assert result == {'project_x': str(a.resolve())}

    def test_extra_roots_default_to_env_var(self, tmp_path, monkeypatch):
        a = tmp_path / 'reify'
        b = tmp_path / 'dark-factory'
        a.mkdir()
        b.mkdir()
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(b))
        result = build_known_projects_map(str(a))  # no extra_roots arg
        assert result == {
            'reify': str(a.resolve()),
            'dark_factory': str(b.resolve()),
        }
