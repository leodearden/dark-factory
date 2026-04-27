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
