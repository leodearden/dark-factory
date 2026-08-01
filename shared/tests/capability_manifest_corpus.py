"""Corpus discovery + file-attributed checker for capability-manifest sidecars.

Test-support module — NOT production code; nothing under ``shared/src/``
changes for this. Consumed by ``shared/tests/test_capability_manifest.py``'s
``TestManifestCorpusDiscovery`` / ``TestCheckManifest`` /
``TestCheckedInManifestCorpus``. Importable bare
(``from capability_manifest_corpus import ...``) because
``shared/tests/conftest.py`` inserts this ``tests/`` directory onto
``sys.path`` — the same mechanism ``silent_fallthrough_scan.py`` and
``startup_completion_probe.py`` already rely on.

Why ``git ls-files``, not ``Path.rglob``
-----------------------------------------
A naive ``REPO_ROOT.rglob('*.capability-manifest.yaml')`` is a trap in this
repo. Measured from the main checkout (``/home/leo/src/dark-factory``):

    git ls-files -- '*.capability-manifest.yaml'                   -> 30 files, 0 under .worktrees/
    find .worktrees -maxdepth 3 -name '*.capability-manifest.yaml'  -> 3793 files

``.worktrees/`` is gitignored (``.gitignore:15``) and holds every other
in-flight task's checkout. An rglob-based guard would validate thousands of
unrelated working copies and go red on other tasks' in-progress edits, so
discovery is ``git ls-files``-based instead — the same shape as
``fused-memory/tests/test_lock_charter_guard.py``'s ``_tracked_paths()``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from pydantic import ValidationError

from shared.capability_manifest import load_capability_manifest

#: Mirrors fused_memory.server.manifest_stamping._SIDECAR_SUFFIX. Deliberately
#: re-declared rather than imported — shared/tests must not depend on
#: fused_memory.
MANIFEST_SUFFIX = '.capability-manifest.yaml'

#: shared/tests/capability_manifest_corpus.py -> parents[0]=shared/tests,
#: parents[1]=shared, parents[2]=repo root. Mirrors the resolution
#: TestLoader::test_committed_exemplar_sidecar_validates already uses from
#: this same tests/ directory, and fused-memory/tests/test_lock_charter_guard.py's
#: _DF_REPO_ROOT; correct inside a worktree checkout too.
REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_manifests(repo_root: Path = REPO_ROOT) -> list[Path] | None:
    """Every checked-in ``*.capability-manifest.yaml`` sidecar under *repo_root*.

    Uses ``git ls-files`` rather than ``Path.rglob`` — see module docstring
    for why. Returns ``None`` when *repo_root* is not a git checkout
    (``git ls-files`` exits non-zero), mirroring
    ``test_lock_charter_guard.py``'s ``_tracked_paths()``. Also returns
    ``None`` when the ``git`` invocation itself can't complete — binary
    missing/unrunnable (``OSError``, e.g. ``FileNotFoundError`` on an odd
    ``PATH``) or wedged past a 30s timeout (``subprocess.TimeoutExpired``,
    e.g. ``index.lock`` contention) — rather than letting either escape:
    ``test_capability_manifest.py`` calls this at *module import time*
    (``_MANIFEST_PATHS = discover_manifests() or []``), so an uncaught
    exception here would take down collection of the entire test module,
    not just the corpus-guard classes. Otherwise returns a sorted,
    duplicate-free list of absolute paths.
    """
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_root), 'ls-files', '-z', '--', f'*{MANIFEST_SUFFIX}'],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    rel_paths = sorted({p for p in result.stdout.split('\0') if p})
    return [repo_root / rel for rel in rel_paths]


def check_manifest(path: Path) -> str | None:
    """Validate one sidecar, returning ``None`` on success or one actionable line.

    A thin file-attribution wrapper around
    :func:`shared.capability_manifest.load_capability_manifest`, which already
    recurses into every task and capability and never swallows errors — this
    adds the one thing it can't: WHICH of the many checked-in sidecars a given
    failure came from. A bare ``pydantic.ValidationError`` names a field path
    (e.g. ``tasks.3.capabilities.2.delivered_check.pattern``) but never the
    file, so a caller sweeping many sidecars (``TestCheckedInManifestCorpus``)
    would otherwise get an unattributed error.

    Catches exactly ``yaml.YAMLError`` and ``pydantic.ValidationError`` and
    converts each into a single grep-able line (embedded newlines collapsed)
    prefixed with *path* relative to :data:`REPO_ROOT` (falling back to the
    absolute path if the relative computation fails, e.g. *path* lies outside
    the repo). Every other exception — notably ``FileNotFoundError`` —
    propagates untouched: a missing file is a caller bug, not manifest drift,
    and must never be silently re-reported as a check failure.
    """
    try:
        display_path: Path | str = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    try:
        load_capability_manifest(path)
    except yaml.YAMLError as exc:
        detail = ' '.join(str(exc).split())
        return f'{display_path}: YAML syntax error: {detail}'
    except ValidationError as exc:
        detail = ' '.join(str(exc).split())
        return f'{display_path}: schema validation failed: {detail}'
    return None
