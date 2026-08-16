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

import os
import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScriptCheckRef:
    """One ``kind: script`` delivered_check, attributed back to its author.

    Everything a failure message needs to name WHICH check is broken:
    the sidecar it came from, the Greek-label task block and capability
    that declared it, and the repo-relative ``script:`` string verbatim
    (i.e. exactly what ``_run_script_check`` joins onto ``project_root``
    to build ``argv[0]``).
    """

    manifest: Path
    task_label: str
    capability: str
    script: str


def iter_script_checks(path: Path) -> list[ScriptCheckRef]:
    """Every ``kind: script`` delivered_check declared in one sidecar.

    Walks ``load_capability_manifest(path).tasks[].capabilities[]`` and emits
    one :class:`ScriptCheckRef` per capability whose ``delivered_check`` is
    non-``None`` with ``kind == 'script'``. ``delivered_check.script`` is
    guaranteed non-empty by ``_check_kind_conditional_fields``' script branch
    (``script is required when kind='script'``), so it is taken verbatim with
    no re-validation. A capability with no ``delivered_check`` at all, or one
    of ``kind`` ``'grep'``/``'manual'``, contributes nothing.

    Returns ``[]`` — rather than raising — for a sidecar that fails to load
    (``yaml.YAMLError``, ``pydantic.ValidationError``, ``OSError``). This is
    NOT a fail-soft, for two reasons:

    1. No signal is lost. ``TestCheckedInManifestCorpus.
       test_checked_in_manifest_validates`` already reports an unloadable
       sidecar loudly, with a better field-path-attributed message, and a
       schema-invalid sidecar is a manifest-shape defect rather than a
       script-lifecycle one — re-reporting it here would misattribute it.
    2. The swallow is layered under two guards that go red if it ever hides
       everything: ``TestCheckedInScriptCheckTargets``' non-vacuity test and
       its both-known-anchors test.

    The swallow is also load-bearing for collection: ``test_capability_manifest.py``
    calls this at *module import time* (``_SCRIPT_CHECKS = [...]``) to feed
    ``@pytest.mark.parametrize``, so an escaping exception would take down
    collection of the entire test module — the same hazard
    :func:`discover_manifests`' docstring names.
    """
    try:
        doc = load_capability_manifest(path)
    except (yaml.YAMLError, ValidationError, OSError):
        return []
    return [
        ScriptCheckRef(
            manifest=path,
            task_label=task.label,
            capability=cap.name,
            script=cap.delivered_check.script,
        )
        for task in doc.tasks
        for cap in task.capabilities
        if cap.delivered_check is not None
        and cap.delivered_check.kind == 'script'
        and cap.delivered_check.script
    ]


def check_script_target(ref: ScriptCheckRef, repo_root: Path = REPO_ROOT) -> str | None:
    """Check one script target exists and is executable in the WORKING tree.

    Returns ``None`` on success, else one grep-able line — the same contract
    :func:`check_manifest` uses, with the same ``relative_to(repo_root)``
    display-path convention and the same no-embedded-newline discipline, so
    the two corpus checkers read identically and their failures grep alike.
    Unlike ``check_manifest``, the line is attributed by sidecar AND
    capability, because one sidecar can declare several script checks.

    This is the exact property ``orchestrator.delivered_checks._run_script_check``
    depends on: it builds ``argv = [str(Path(project_root) / meta.script), *args]``
    and execs the target DIRECTLY (``argv[0]`` is the script, not ``python``),
    so a missing or non-executable target raises ``OSError``, which
    ``run_delivered_check``'s catch-all maps to ``DeliveredCheckResult.ERRORED``
    — the check never FAILs, it ERRORs forever, silently un-gating the
    capability it was supposed to gate. Hence ``is_file()`` gates before
    ``os.access``: a *directory* at the script path is +x and would sail past
    a bare ``os.access`` check while still being unexecutable.

    Deliberately does NOT assert the target carries a valid shebang. A
    shebang-less executable fails direct exec with ``ENOEXEC`` and would ERROR
    the same way, but both current targets carry ``#!/usr/bin/env python3``
    and ``scripts/tests/test_check_method_param_wiring.py``'s
    ``TestEndToEndAgainstTheRealRepo`` already covers it for one of them by
    actually exec'ing it — the omission is weighed, not missed (task 3649).
    """
    try:
        display_manifest: Path | str = ref.manifest.relative_to(repo_root)
    except ValueError:
        display_manifest = ref.manifest
    prefix = f'{display_manifest}: capability {ref.capability!r}: {ref.script}'
    target = repo_root / ref.script
    if not target.is_file():
        return (
            f'{prefix} does not exist — _run_script_check execs it as argv[0], '
            'so the check ERRORs forever'
        )
    if not os.access(target, os.X_OK):
        return (
            f'{prefix} is not executable — _run_script_check execs it as argv[0], '
            'so a non-executable target ERRORs forever'
        )
    return None


def committed_file_mode(rel: str, repo_root: Path = REPO_ROOT) -> str | None:
    """The git INDEX mode for *rel* (e.g. ``'100755'``), or ``None`` if untracked.

    The companion to :func:`check_script_target`'s working-tree check, and not
    redundant with it: ``os.access`` is greened by a local ``chmod +x`` that is
    never staged, which would leave ``main`` exactly as broken while the guard
    reported green — the precise hazard task 3649 exists to fix, since its whole
    subject is a mode bit that never made it into the index.

    Reuses :func:`discover_manifests`' subprocess idiom verbatim — same ``-z``
    output, same 30s timeout, same ``except (OSError, subprocess.TimeoutExpired)``
    guard and returncode check — so this second git shell-out has identical
    non-checkout / missing-binary / ``index.lock``-wedge behaviour rather than a
    differently-guarded one. Returns ``None`` for empty output (path not tracked).
    """
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_root), 'ls-files', '-s', '-z', '--', rel],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    records = [r for r in result.stdout.split('\0') if r]
    if not records:
        return None
    # `git ls-files -s` record: "<mode> <object> <stage>\t<path>".
    return records[0].split(' ', 1)[0]
