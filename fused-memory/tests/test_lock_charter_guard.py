"""Unit tests for fused_memory.middleware.lock_charter_guard.

Step 1 (RED → step-2 GREEN): predicate + drift guard
Step 3 (RED → step-4 GREEN): list-gate helpers
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.lock_charter_guard import (
    CODE_EXTENSIONS,
    directory_locks,
    extract_files,
    is_file_path,
    lock_charter_error,
)

# ---------------------------------------------------------------------------
# Drift guard — pins sorted(CODE_EXTENSIONS) to the shared α/γ test vector.
# Any divergence from reify's --list-extensions output fails loudly here.
# ---------------------------------------------------------------------------

_CANONICAL_EXTENSIONS = [
    'c', 'cc', 'cjs', 'cpp', 'css', 'cts', 'cxx', 'gcode',
    'h', 'hh', 'hpp', 'html',
    'js', 'json', 'jsonc', 'jsx',
    'lock', 'md', 'mjs', 'mts', 'png', 'py',
    'ri', 'rs', 'scss', 'service', 'sh', 'step', 'stl', 'svg',
    'toml', 'ts', 'tsx', 'txt',
    'yaml', 'yml',
]


def test_extension_drift_guard():
    """sorted(CODE_EXTENSIONS) must exactly match the shared α/γ test vector."""
    assert sorted(CODE_EXTENSIONS) == _CANONICAL_EXTENSIONS


# ---------------------------------------------------------------------------
# REJECT corpus (α/γ shared vector — all must return False)
# ---------------------------------------------------------------------------

_REJECT_PATHS = [
    'crates/',
    'crates/reify-eval/src',
    'crates/reify-eval/tests',
    'examples',
    'compute_targets',
    'modal',
    'crates/reify-eval/src/',
    'a/b/c/',
    '/',
]


@pytest.mark.parametrize('path', _REJECT_PATHS)
def test_is_file_path_rejects_directories(path):
    """Directory-style paths must be classified as directories (False)."""
    assert is_file_path(path) is False


# ---------------------------------------------------------------------------
# ACCEPT corpus (all must return True)
# ---------------------------------------------------------------------------

_ACCEPT_PATHS = [
    # Standard rust file in a deep path
    'crates/foo/src/bar.rs',
    # C-P4: deep file whose parent dir name looks like an extension-less token
    'a/b/compute_targets/foo.rs',
    # C-P3: no-stat — ghost path (not on disk) must still be accepted
    'no/such/path/ghost.rs',
    # One path per canonical extension
    'examples/foo.ri',
    'crates/x/Cargo.toml',
    'notes.md',
    'logo.png',
    'units/orchestrator.service',
    'out/part.gcode',
    'src/lib.c',
    'src/lib.cc',
    'src/lib.cxx',
    'src/lib.cpp',
    'include/lib.h',
    'include/lib.hh',
    'include/lib.hpp',
    'src/index.html',
    'src/main.js',
    'src/data.json',
    'src/data.jsonc',
    'src/comp.jsx',
    'yarn.lock',
    'src/mod.mjs',
    'src/mod.mts',
    'src/mod.ts',
    'src/mod.tsx',
    'src/comp.cjs',
    'src/mod.cts',
    'src/styles.css',
    'src/styles.scss',
    'src/icon.svg',
    'Cargo.toml',
    'script.sh',
    'model/part.step',
    'model/object.stl',
    'README.txt',
    'config.yaml',
    'config.yml',
    'src/main.py',
]


@pytest.mark.parametrize('path', _ACCEPT_PATHS)
def test_is_file_path_accepts_files(path):
    """File-level paths must be classified as files (True)."""
    assert is_file_path(path) is True


# ---------------------------------------------------------------------------
# Conservative-reject edge cases matching α's case-sensitive bash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('path', ['f.PY', '.gitignore'])
def test_is_file_path_conservative_rejects(path):
    """Upper-case extensions and dotfiles without extension are rejected (C-P3)."""
    assert is_file_path(path) is False


# ---------------------------------------------------------------------------
# Step 3: list-gate helpers — directory_locks, extract_files, lock_charter_error
# ---------------------------------------------------------------------------

class TestDirectoryLocks:
    def test_empty_list_returns_empty(self):
        assert directory_locks([]) == []

    def test_all_files_returns_empty(self):
        assert directory_locks(['a/b.rs', 'examples/c.ri']) == []

    def test_mixed_returns_only_dirs(self):
        result = directory_locks(['crates/x/src/a.rs', 'crates/', 'compute_targets'])
        assert result == ['crates/', 'compute_targets']

    def test_order_preserved(self):
        result = directory_locks(['modal', 'a/b.py', 'crates/'])
        assert result == ['modal', 'crates/']

    def test_dedup(self):
        result = directory_locks(['crates/', 'crates/', 'modal'])
        assert result == ['crates/', 'modal']

    def test_non_str_skipped(self):
        result = directory_locks([None, 42, 'crates/', '', '   ', 'a/b.rs'])
        assert result == ['crates/']

    def test_whitespace_only_skipped(self):
        assert directory_locks(['   ', '\t']) == []


class TestExtractFiles:
    def test_dict_with_files(self):
        assert extract_files({'files': ['x/y.rs']}) == ['x/y.rs']

    def test_json_string_with_files(self):
        assert extract_files('{"files": ["a/"]}') == ['a/']

    def test_none_returns_empty(self):
        assert extract_files(None) == []

    def test_missing_files_key_returns_empty(self):
        assert extract_files({'other': 'val'}) == []

    def test_non_list_files_returns_empty(self):
        assert extract_files({'files': 'a/b.rs'}) == []

    def test_unparseable_json_returns_empty(self):
        assert extract_files('not valid json {{{') == []

    def test_json_non_dict_returns_empty(self):
        # JSON string that parses to a list, not a dict
        assert extract_files('["a/b.rs"]') == []

    def test_filters_non_str_entries(self):
        result = extract_files({'files': ['a/b.rs', 42, None, 'c/d.py']})
        assert result == ['a/b.rs', 'c/d.py']

    def test_empty_list_files(self):
        assert extract_files({'files': []}) == []


class TestLockCharterError:
    def test_error_type_is_violation(self):
        result = lock_charter_error(['crates/'])
        assert result['error_type'] == 'LockCharterViolation'

    def test_directory_paths_preserved(self):
        result = lock_charter_error(['crates/', 'modal'])
        assert result['directory_paths'] == ['crates/', 'modal']

    def test_hint_mentions_escape_hatch(self):
        result = lock_charter_error(['crates/'])
        assert 'files=[]' in result['hint'] or '[]' in result['hint']

    def test_error_message_mentions_directories(self):
        result = lock_charter_error(['crates/'])
        assert 'crates/' in result['error']

    def test_task_id_appears_in_error(self):
        result = lock_charter_error(['orchestrator/'], task_id='43')
        assert '43' in result['error']

    def test_no_task_id_still_works(self):
        result = lock_charter_error(['orchestrator/'])
        assert 'error' in result
        assert result['error_type'] == 'LockCharterViolation'
