"""Tests for cargo_scope.py — workspace-crate discovery and file→crate mapping."""

from pathlib import Path

from orchestrator.cargo_scope import (
    _clear_cache,
    discover_workspace_crates,
    files_to_crates,
)


def _write_workspace(root: Path, crates: dict[str, str]) -> None:
    """Create a minimal cargo workspace rooted at *root*."""
    members = sorted({d.split('/', 1)[0] + '/*' for d in crates})
    (root / 'Cargo.toml').write_text(
        '[workspace]\nmembers = [' + ', '.join(f'"{m}"' for m in members) + ']\n'
    )
    for rel, name in crates.items():
        crate_dir = root / rel
        crate_dir.mkdir(parents=True, exist_ok=True)
        (crate_dir / 'Cargo.toml').write_text(
            f'[package]\nname = "{name}"\nversion = "0.1.0"\n'
        )


class TestDiscoverWorkspaceCrates:
    def test_single_crate(self, tmp_path: Path):
        _clear_cache()
        _write_workspace(tmp_path, {'crates/reify-eval': 'reify-eval'})
        crates = discover_workspace_crates(tmp_path)
        assert crates == {'crates/reify-eval': 'reify-eval'}

    def test_multiple_crates(self, tmp_path: Path):
        _clear_cache()
        _write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
            'crates/reify-lsp': 'reify-lsp',
            'crates/reify-cli': 'reify-cli',
        })
        crates = discover_workspace_crates(tmp_path)
        assert crates == {
            'crates/reify-eval': 'reify-eval',
            'crates/reify-lsp': 'reify-lsp',
            'crates/reify-cli': 'reify-cli',
        }

    def test_crate_name_differs_from_dir(self, tmp_path: Path):
        _clear_cache()
        _write_workspace(tmp_path, {'crates/my-crate-dir': 'my_crate_pkg'})
        crates = discover_workspace_crates(tmp_path)
        assert crates == {'crates/my-crate-dir': 'my_crate_pkg'}

    def test_no_cargo_toml_returns_empty(self, tmp_path: Path):
        _clear_cache()
        assert discover_workspace_crates(tmp_path) == {}

    def test_no_workspace_section_returns_empty(self, tmp_path: Path):
        _clear_cache()
        (tmp_path / 'Cargo.toml').write_text('[package]\nname = "solo"\nversion = "0.1.0"\n')
        assert discover_workspace_crates(tmp_path) == {}

    def test_cache_hit_on_repeat(self, tmp_path: Path):
        _clear_cache()
        _write_workspace(tmp_path, {'crates/a': 'a'})
        first = discover_workspace_crates(tmp_path)
        # Delete the Cargo.toml — if cache works, we still see the old result
        (tmp_path / 'Cargo.toml').unlink()
        second = discover_workspace_crates(tmp_path)
        assert first == second == {'crates/a': 'a'}

    def test_clear_cache_refreshes(self, tmp_path: Path):
        _clear_cache()
        _write_workspace(tmp_path, {'crates/a': 'a'})
        discover_workspace_crates(tmp_path)
        _clear_cache()
        (tmp_path / 'Cargo.toml').unlink()
        assert discover_workspace_crates(tmp_path) == {}


class TestFilesToCrates:
    def test_single_file_single_crate(self):
        crates = {'crates/reify-eval': 'reify-eval'}
        assert files_to_crates(
            ['crates/reify-eval/src/foo.rs'], crates,
        ) == ['reify-eval']

    def test_multiple_files_same_crate_deduped(self):
        crates = {'crates/reify-eval': 'reify-eval'}
        result = files_to_crates(
            ['crates/reify-eval/src/foo.rs', 'crates/reify-eval/src/bar.rs'],
            crates,
        )
        assert result == ['reify-eval']

    def test_multi_crate_unique_in_order(self):
        crates = {
            'crates/reify-eval': 'reify-eval',
            'crates/reify-lsp': 'reify-lsp',
        }
        result = files_to_crates(
            [
                'crates/reify-eval/src/a.rs',
                'crates/reify-lsp/src/b.rs',
                'crates/reify-eval/src/c.rs',
            ],
            crates,
        )
        assert result == ['reify-eval', 'reify-lsp']

    def test_file_outside_any_crate_returns_none(self):
        crates = {'crates/reify-eval': 'reify-eval'}
        # scripts/ is outside all crates → signals fallthrough to --workspace
        assert files_to_crates(['scripts/foo.sh'], crates) is None

    def test_root_file_returns_none(self):
        crates = {'crates/reify-eval': 'reify-eval'}
        assert files_to_crates(['build.rs'], crates) is None

    def test_empty_crates_returns_none(self):
        assert files_to_crates(['src/a.rs'], {}) is None

    def test_deeply_nested_file_walks_up(self):
        crates = {'crates/reify-eval': 'reify-eval'}
        result = files_to_crates(
            ['crates/reify-eval/src/sub/module/deep.rs'],
            crates,
        )
        assert result == ['reify-eval']
