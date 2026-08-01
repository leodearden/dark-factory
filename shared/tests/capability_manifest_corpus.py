"""Corpus discovery for checked-in capability-manifest sidecars.

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
    ``test_lock_charter_guard.py``'s ``_tracked_paths()``. Otherwise returns a
    sorted, duplicate-free list of absolute paths.
    """
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files', '-z', '--', f'*{MANIFEST_SUFFIX}'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    rel_paths = sorted({p for p in result.stdout.split('\0') if p})
    return [repo_root / rel for rel in rel_paths]
