"""Unit tests for fused_memory.middleware.lock_charter_guard.

Step 1 (RED → step-2 GREEN): predicate + drift guard
Step 3 (RED → step-4 GREEN): list-gate helpers
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.lock_charter_guard import (
    CODE_EXTENSIONS,
    is_file_path,
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
