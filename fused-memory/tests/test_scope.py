"""Tests for resolve_project_id() — converts filesystem project_root to logical project_id."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from fused_memory.models.scope import (
    KNOWN_PROJECT_ROOTS_ENV,
    Scope,
    build_known_projects_map,
    known_project_roots_from_env,
    read_declared_project_id,
    resolve_project_id,
    resolve_project_id_for_root,
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


class TestReadDeclaredProjectId:
    """read_declared_project_id reads the manifest-declared project_id from
    `<root>/dark-factory-orchestrator.yaml`, normalizing it with the same
    underscore-canonical rule as resolve_project_id, and fails open (returns
    None, never raises) on any error or absent/ill-typed key.

    This is what makes project_id rename-stable: a directory whose basename
    differs from its declared manifest id (the exact shape produced by a
    project-dir rename) resolves to the DECLARED id, not the basename.
    """

    def _write_manifest(self, root: Path, body: str) -> None:
        (root / 'dark-factory-orchestrator.yaml').write_text(body)

    def test_declared_id_wins_over_basename(self, tmp_path):
        """A dir named `solar-challenge` whose manifest declares
        project_id my_solar_challenge resolves to 'my_solar_challenge'
        (declared id), NOT the basename-derived 'solar_challenge'."""
        d = tmp_path / 'solar-challenge'
        d.mkdir()
        self._write_manifest(d, 'project_id: "my_solar_challenge"\n')
        assert read_declared_project_id(str(d)) == 'my_solar_challenge'
        # sanity: this is genuinely different from the basename derivation.
        assert resolve_project_id(str(d)) == 'solar_challenge'

    def test_accepts_path_argument(self, tmp_path):
        """Signature is `str | Path` — a Path argument works too."""
        d = tmp_path / 'solar-challenge'
        d.mkdir()
        self._write_manifest(d, 'project_id: "my_solar_challenge"\n')
        assert read_declared_project_id(d) == 'my_solar_challenge'

    def test_hyphenated_declared_value_is_normalized(self, tmp_path):
        """A hyphenated declared value normalizes with the same
        _to_underscore_canonical rule resolve_project_id uses."""
        d = tmp_path / 'solar-challenge'
        d.mkdir()
        self._write_manifest(d, 'project_id: "my-solar-challenge"\n')
        assert read_declared_project_id(str(d)) == 'my_solar_challenge'

    # --- fail-open cases: each returns None, never raises ---

    def test_missing_manifest_returns_none(self, tmp_path):
        d = tmp_path / 'no-manifest'
        d.mkdir()
        assert read_declared_project_id(str(d)) is None

    def test_missing_directory_returns_none(self, tmp_path):
        """A project_root that does not exist at all fails open to None."""
        assert read_declared_project_id(str(tmp_path / 'does_not_exist')) is None

    def test_malformed_yaml_returns_none(self, tmp_path):
        d = tmp_path / 'bad-yaml'
        d.mkdir()
        # Unclosed flow mapping — yaml.safe_load raises yaml.YAMLError.
        self._write_manifest(d, '{ project_id: "x"\n')
        assert read_declared_project_id(str(d)) is None

    def test_not_a_mapping_returns_none(self, tmp_path):
        d = tmp_path / 'list-yaml'
        d.mkdir()
        # Valid YAML, but the top-level document is a list, not a mapping.
        self._write_manifest(d, '- a\n- b\n')
        assert read_declared_project_id(str(d)) is None

    def test_scalar_yaml_returns_none(self, tmp_path):
        d = tmp_path / 'scalar-yaml'
        d.mkdir()
        # Valid YAML that parses to a bare scalar string, not a mapping.
        self._write_manifest(d, 'just a scalar\n')
        assert read_declared_project_id(str(d)) is None

    def test_empty_yaml_returns_none(self, tmp_path):
        d = tmp_path / 'empty-yaml'
        d.mkdir()
        # Empty file → yaml.safe_load returns None (not a mapping).
        self._write_manifest(d, '')
        assert read_declared_project_id(str(d)) is None

    def test_no_project_id_key_returns_none(self, tmp_path):
        d = tmp_path / 'no-key'
        d.mkdir()
        self._write_manifest(d, 'some_other_key: value\n')
        assert read_declared_project_id(str(d)) is None

    def test_non_string_project_id_returns_none(self, tmp_path):
        d = tmp_path / 'int-id'
        d.mkdir()
        self._write_manifest(d, 'project_id: 123\n')
        assert read_declared_project_id(str(d)) is None

    def test_null_project_id_returns_none(self, tmp_path):
        d = tmp_path / 'null-id'
        d.mkdir()
        self._write_manifest(d, 'project_id: null\n')
        assert read_declared_project_id(str(d)) is None

    def test_empty_string_project_id_returns_none(self, tmp_path):
        d = tmp_path / 'empty-id'
        d.mkdir()
        self._write_manifest(d, 'project_id: ""\n')
        assert read_declared_project_id(str(d)) is None


class TestResolveProjectIdForRoot:
    """resolve_project_id_for_root is the rename-stable resolver:
    manifest-declared id first, pure basename derivation as the fallback."""

    def test_declared_id_wins(self, tmp_path):
        """A dir named `solar-challenge` whose manifest declares
        project_id my_solar_challenge resolves to 'my_solar_challenge'."""
        d = tmp_path / 'solar-challenge'
        d.mkdir()
        (d / 'dark-factory-orchestrator.yaml').write_text(
            'project_id: "my_solar_challenge"\n',
        )
        assert resolve_project_id_for_root(str(d)) == 'my_solar_challenge'

    def test_basename_fallback_no_manifest(self, tmp_path):
        """A dir with NO manifest resolves to the basename derivation,
        identical to resolve_project_id — back-compat for every project
        that does not declare an id."""
        d = tmp_path / 'foo-bar'
        d.mkdir()
        assert resolve_project_id_for_root(str(d)) == 'foo_bar'
        assert resolve_project_id_for_root(str(d)) == resolve_project_id(str(d))

    def test_basename_fallback_manifest_without_project_id(self, tmp_path):
        """A manifest present but lacking a project_id key still falls back
        to the basename derivation."""
        d = tmp_path / 'foo-bar'
        d.mkdir()
        (d / 'dark-factory-orchestrator.yaml').write_text('other_key: 1\n')
        assert resolve_project_id_for_root(str(d)) == 'foo_bar'


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

    def test_renamed_dir_registers_under_declared_manifest_id(self, tmp_path):
        """Core guard: a directory whose basename (`solar-challenge`) differs
        from the project_id declared in its manifest (`my_solar_challenge`)
        registers under the DECLARED id — re-pointing the recon registry to
        the new path under the SAME id after a rename, instead of orphaning
        the records under the basename-derived 'solar_challenge'.
        """
        renamed = tmp_path / 'solar-challenge'
        renamed.mkdir()
        (renamed / 'dark-factory-orchestrator.yaml').write_text(
            'project_id: "my_solar_challenge"\n',
        )
        # Back-compat: a project that declares no id still keys by basename.
        plain = tmp_path / 'plain-proj'
        plain.mkdir()

        result = build_known_projects_map(
            '', extra_roots=[str(renamed), str(plain)],
        )
        assert result['my_solar_challenge'] == str(renamed.resolve())
        assert 'solar_challenge' not in result
        assert result['plain_proj'] == str(plain.resolve())


class TestScopeCanonicalizesProjectId:
    """Scope.project_id is canonicalized at construction, so every field
    derived from it (graphiti_group_id, mem0_collection_name,
    mem0_user_id) is always canonical too."""

    def test_project_id_stored_canonical(self):
        assert Scope(project_id='dark-factory').project_id == 'dark_factory'

    def test_graphiti_group_id_canonical(self):
        assert Scope(project_id='dark-factory').graphiti_group_id == 'dark_factory'

    def test_mem0_collection_name_canonical(self):
        scope = Scope(project_id='dark-factory')
        assert scope.mem0_collection_name('fused') == 'fused_dark_factory'

    def test_mem0_user_id_canonical(self):
        assert Scope(project_id='dark-factory').mem0_user_id == 'dark_factory'

    def test_uppercase_and_hyphen_combined(self):
        assert Scope(project_id='KNOW-live').project_id == 'know_live'

    def test_construction_idempotent(self):
        """Constructing a Scope from an already-canonical project_id (i.e.
        the project_id read back off a previously-constructed Scope) yields
        the same canonical value again."""
        once = Scope(project_id='dark-factory').project_id
        twice = Scope(project_id=once).project_id
        assert twice == 'dark_factory'

    def test_path_shaped_project_id_raises_validation_error(self):
        """A path-shaped project_id is LOUD-rejected as a pydantic
        ValidationError at construction, not silently normalized."""
        bad = '-home-leo-src-x'
        with pytest.raises(ValidationError) as exc_info:
            Scope(project_id=bad)
        assert bad in str(exc_info.value), (
            'PathShapedProjectIdError message must be preserved through '
            f"pydantic's wrapping; got: {exc_info.value}"
        )
