"""Tests for ProjectPrefixRegistry — multi-project prefix discovery + collision handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    """Create a fake project under *parent* with the given top-level *dirs*."""
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


# ---------------------------------------------------------------------------
# from_roots — basic discovery
# ---------------------------------------------------------------------------


class TestFromRootsBasic:
    def test_empty_root_list_yields_empty_registry(self):
        r = ProjectPrefixRegistry.from_roots([])
        assert r.project_to_root == {}
        assert r.project_to_prefixes == {}
        assert r.prefix_to_project == {}
        assert not r  # __bool__ → False

    def test_single_project_with_one_unique_dir(self, tmp_path):
        p = _mkproj(tmp_path, 'reify', ['crates'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_root == {'reify': str(p.resolve())}
        assert r.project_to_prefixes == {'reify': ('crates/',)}
        assert r.prefix_to_project == {'crates/': 'reify'}
        assert bool(r) is True

    def test_hyphenated_dir_gets_underscore_alias(self, tmp_path):
        p = _mkproj(tmp_path, 'dark-factory', ['fused-memory'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['dark_factory'] == ('fused-memory/', 'fused_memory/')
        assert r.prefix_to_project['fused-memory/'] == 'dark_factory'
        assert r.prefix_to_project['fused_memory/'] == 'dark_factory'

    def test_generic_dirs_are_filtered_out(self, tmp_path):
        # src/, tests/, docs/, scripts/, hooks/, review/ should all be skipped.
        p = _mkproj(
            tmp_path, 'project',
            ['src', 'tests', 'docs', 'scripts', 'hooks', 'review', 'real_subproject'],
        )
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real_subproject/',)

    def test_hidden_dirs_filtered_out(self, tmp_path):
        p = _mkproj(tmp_path, 'project', ['.git', '.venv', 'real'])
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real/',)

    def test_files_at_top_level_ignored(self, tmp_path):
        p = _mkproj(tmp_path, 'project', ['real_dir'])
        (p / 'README.md').write_text('hi')
        (p / 'pyproject.toml').write_text('[project]')
        r = ProjectPrefixRegistry.from_roots([str(p)])
        assert r.project_to_prefixes['project'] == ('real_dir/',)


# ---------------------------------------------------------------------------
# from_roots — collisions
# ---------------------------------------------------------------------------


class TestFromRootsCollisions:
    def test_shared_prefix_dropped_from_both_projects(self, tmp_path):
        """When two projects both have a `tools/` dir, that prefix is dropped."""
        a = _mkproj(tmp_path, 'reify', ['crates', 'tools'])
        b = _mkproj(tmp_path, 'autopilot-video', ['pipeline', 'tools'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        # tools/ removed from both project_to_prefixes entries
        assert 'tools/' not in r.prefix_to_project
        assert 'tools/' not in r.project_to_prefixes['reify']
        assert 'tools/' not in r.project_to_prefixes['autopilot_video']
        # Unique prefixes survive
        assert 'crates/' in r.prefix_to_project
        assert r.prefix_to_project['crates/'] == 'reify'
        assert 'pipeline/' in r.prefix_to_project
        assert r.prefix_to_project['pipeline/'] == 'autopilot_video'

    def test_underscore_alias_collision_treated_independently(self, tmp_path):
        """fused-memory/ and fused_memory/ are independent prefix entries —
        a collision on one doesn't drop the other.
        """
        # Project A has 'fused-memory' (hyphen alias generated) — yields both fused-memory/ and fused_memory/
        a = _mkproj(tmp_path, 'a', ['fused-memory'])
        # Project B has only 'fused_memory' (no hyphen → no alias) — yields just fused_memory/
        b = _mkproj(tmp_path, 'b', ['fused_memory'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        # fused_memory/ collides → dropped from both
        assert 'fused_memory/' not in r.prefix_to_project
        # fused-memory/ unique to a → survives
        assert r.prefix_to_project.get('fused-memory/') == 'a'

    def test_duplicate_project_id_keeps_first(self, tmp_path):
        """Two roots resolving to the same project_id: first wins; second skipped."""
        a = _mkproj(tmp_path, 'project_x', ['unique_a'])
        # Different parent, same name → same project_id.
        sub = tmp_path / 'sub'
        sub.mkdir()
        b = _mkproj(sub, 'project_x', ['unique_b'])
        r = ProjectPrefixRegistry.from_roots([str(a), str(b)])
        assert r.project_to_root == {'project_x': str(a.resolve())}
        # Only first project's prefixes registered.
        assert r.prefix_to_project == {'unique_a/': 'project_x'}


# ---------------------------------------------------------------------------
# Lookups + helpers
# ---------------------------------------------------------------------------


class TestRegistryLookups:
    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_project_for_prefix(self, registry):
        assert registry.project_for_prefix('crates/') == 'reify'
        assert registry.project_for_prefix('fused-memory/') == 'dark_factory'
        assert registry.project_for_prefix('fused_memory/') == 'dark_factory'
        assert registry.project_for_prefix('unknown/') is None

    def test_root_for_project(self, registry, tmp_path):
        assert registry.root_for_project('reify') == str((tmp_path / 'reify').resolve())
        assert registry.root_for_project('dark_factory') == str(
            (tmp_path / 'dark-factory').resolve(),
        )
        assert registry.root_for_project('nope') is None

    def test_is_known(self, registry):
        assert registry.is_known('reify')
        assert registry.is_known('dark_factory')
        assert not registry.is_known('autopilot_video')

    def test_all_prefixes_sorted(self, registry):
        prefixes = registry.all_prefixes()
        assert prefixes == tuple(sorted(prefixes))
        # Spot-check membership.
        assert 'crates/' in prefixes
        assert 'fused-memory/' in prefixes


# ---------------------------------------------------------------------------
# project_for_path — exact leading-path-component ownership lookup
# ---------------------------------------------------------------------------


class TestProjectForPath:
    """Tests for ProjectPrefixRegistry.project_for_path.

    Unlike project_for_prefix (exact prefix-string lookup) or the guard's
    find_paths (regex-over-prose), this resolves an arbitrary file path to
    its owning project via leading-path-component matching — the CERTAIN
    signal that check_files_for_scope (task 2206) relies on for rejection.
    """

    @pytest.fixture
    def registry(self, tmp_path):
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator'])
        return ProjectPrefixRegistry.from_roots([str(a), str(b)])

    def test_owning_project_for_simple_path(self, registry):
        assert registry.project_for_path('orchestrator/foo.py') == 'dark_factory'

    def test_hyphenated_prefix_and_underscore_alias_both_resolve(self, registry):
        assert registry.project_for_path('fused-memory/src/x.py') == 'dark_factory'
        assert registry.project_for_path('fused_memory/x.py') == 'dark_factory'

    def test_exact_dir_with_no_trailing_content(self, registry):
        assert registry.project_for_path('crates') == 'reify'

    def test_unowned_leading_component_returns_none(self, registry):
        assert registry.project_for_path('unknown_dir/x') is None

    def test_empty_string_returns_none(self, registry):
        assert registry.project_for_path('') is None

    def test_none_ish_returns_none(self, registry):
        """An actual None (defensive — callers should only pass str) is None-ish too."""
        assert registry.project_for_path(None) is None

    def test_startswith_but_different_component_returns_none(self, registry):
        """'cratesfoo/x' merely starts with the string 'crates' but is a
        DIFFERENT leading component — must not match the 'crates/' prefix."""
        assert registry.project_for_path('cratesfoo/x') is None

    def test_prefix_mid_path_returns_none(self, registry):
        """A prefix appearing as a non-leading segment must not match —
        leading-component only, not substring/regex."""
        assert registry.project_for_path('vendor/crates/x') is None

    def test_leading_dot_slash_is_stripped(self, registry):
        assert registry.project_for_path('./orchestrator/x') == 'dark_factory'

    def test_longest_prefix_wins(self):
        """When more than one registered prefix leads file_path, the longest wins.

        Hand-built (not via from_roots/_mkproj): real directory-derived
        prefixes are always single path components, so two of them can never
        both be a leading run of the same file_path — this exercises the
        tie-break directly.
        """
        registry = ProjectPrefixRegistry(
            prefix_to_project={'a/': 'short_owner', 'a/b/': 'long_owner'},
        )
        assert registry.project_for_path('a/b/c.py') == 'long_owner'


# ---------------------------------------------------------------------------
# Edge: non-existent root does not crash
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_nonexistent_root_yields_empty_prefixes_for_that_project(self, tmp_path):
        ghost = tmp_path / 'does_not_exist'
        r = ProjectPrefixRegistry.from_roots([str(ghost)])
        # The project_id is registered (we resolve the path even if it doesn't exist),
        # but with no prefixes.
        assert 'does_not_exist' in r.project_to_root
        assert r.project_to_prefixes['does_not_exist'] == ()

    def test_root_pointing_at_a_file_yields_empty_prefixes(self, tmp_path):
        f = tmp_path / 'just_a_file'
        f.write_text('not a dir')
        r = ProjectPrefixRegistry.from_roots([str(f)])
        assert r.project_to_prefixes['just_a_file'] == ()


# ---------------------------------------------------------------------------
# default() — built-in, filesystem-independent dark_factory registry.
#
# Folds in the retired back-compat shim's hard-coded constants (task 2208 /
# PRD D2) so the guard always classifies dark-factory paths even when no
# known_project_roots are configured.
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    _EXPECTED_PREFIXES = (
        'orchestrator/',
        'fused-memory/',
        'fused_memory/',
        'mem0/',
        'graphiti/',
        'dashboard/',
    )

    def test_default_registry_is_truthy(self):
        assert bool(ProjectPrefixRegistry.default()) is True

    def test_default_registry_carries_dark_factory_prefixes(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_to_prefixes['dark_factory'] == self._EXPECTED_PREFIXES
        for prefix in self._EXPECTED_PREFIXES:
            assert prefix in r.all_prefixes()

    def test_project_for_prefix_resolves_dark_factory(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_for_prefix('orchestrator/') == 'dark_factory'

    def test_project_for_path_resolves_dark_factory(self):
        r = ProjectPrefixRegistry.default()
        assert r.project_for_path('fused-memory/src/x.py') == 'dark_factory'
        assert r.project_for_path('mem0/foo.py') == 'dark_factory'

    def test_root_for_project(self):
        r = ProjectPrefixRegistry.default()
        assert r.root_for_project('dark_factory') == '/home/leo/src/dark-factory'

    def test_is_known(self):
        assert ProjectPrefixRegistry.default().is_known('dark_factory')

    def test_module_exposes_moved_constants(self):
        from fused_memory.middleware import project_prefix_registry as mod

        assert mod.DARK_FACTORY_PROJECT_ID == 'dark_factory'
        assert mod.DARK_FACTORY_ROOT == '/home/leo/src/dark-factory'
        assert mod.DARK_FACTORY_PATH_PREFIXES == self._EXPECTED_PREFIXES
