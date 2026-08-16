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


class GitUnavailable(RuntimeError):
    """``git`` could not answer the question at all — an INFRA condition.

    Strictly distinct from "git answered, and the answer was 'no such
    record'". Raised for: the binary missing/unrunnable (``OSError``, e.g.
    ``FileNotFoundError`` on an odd ``PATH``), a wedge past the 30s timeout
    (``subprocess.TimeoutExpired``, e.g. ``index.lock`` contention against
    the merge worker operating on this same checkout — see CLAUDE.md
    "Working in the main checkout"), and a non-zero exit (notably *repo_root*
    not being a checkout at all).

    Keeping this separate from a ``None`` return is what stops a transient
    wedge from being rendered as a staged-mode defect: a caller that can't
    proceed without git should ``pytest.skip`` on this, not fail (code-review
    amendment, task 3649).
    """


class AmbiguousIndexEntry(GitUnavailable):
    """git answered, but with several index records for ONE literal path.

    A subclass because it is still "no single usable answer" — a caller that
    doesn't care can catch the base :class:`GitUnavailable` alone. But it is
    NOT a flake to be skipped past: it means either a conflicted index (stages
    1/2/3 for the same path) or a ``script:`` value naming a *directory*,
    whose literal pathspec matches every tracked file beneath it. Reading
    ``records[0]`` in either case would report a mode belonging to some other
    file — a green for a path that is not a runnable script (code-review
    amendment, task 3649).
    """


def _git_ls_files(repo_root: Path, *args: str) -> list[str]:
    """``git -C <repo_root> ls-files -z <args>``, NUL-split into records.

    The single owner of this module's git shell-out: the argv shape, the 30s
    timeout, the guarded exception set, the returncode check, and the NUL
    split all live here once. Both :func:`discover_manifests` and
    :func:`committed_file_mode` go through it, so their behaviour on a
    missing binary / wedged index / non-checkout is identical *structurally*
    rather than by two copies of the same code agreeing (code-review
    amendment, task 3649 — the previous duplication asserted that identity in
    prose, which the next timeout change would have silently broken).

    Raises :class:`GitUnavailable` — never returns a sentinel — so each caller
    decides how to degrade: ``discover_manifests`` swallows it into ``None``
    (it runs at module import time and must not raise), while
    ``committed_file_mode`` lets it propagate so a test can skip rather than
    report a phantom defect.
    """
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_root), 'ls-files', '-z', *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitUnavailable(f'git ls-files under {repo_root} did not run: {exc!r}') from exc
    if result.returncode != 0:
        detail = ' '.join(result.stderr.split()) or f'exit {result.returncode}'
        raise GitUnavailable(f'git ls-files under {repo_root} failed: {detail}')
    return [record for record in result.stdout.split('\0') if record]


def discover_manifests(repo_root: Path = REPO_ROOT) -> list[Path] | None:
    """Every checked-in ``*.capability-manifest.yaml`` sidecar under *repo_root*.

    Uses ``git ls-files`` rather than ``Path.rglob`` — see module docstring
    for why. Returns ``None`` for every condition :class:`GitUnavailable`
    covers: *repo_root* not being a git checkout (``git ls-files`` exits
    non-zero, mirroring ``test_lock_charter_guard.py``'s ``_tracked_paths()``),
    a missing/unrunnable binary, or a wedge past the 30s timeout.

    The collapse to ``None`` is deliberate *here* specifically:
    ``test_capability_manifest.py`` calls this at *module import time*
    (``_MANIFEST_PATHS = discover_manifests() or []``), so an escaping
    exception would take down collection of the entire test module, not just
    the corpus-guard classes. Callers that run at test time instead — see
    :func:`committed_file_mode` — keep the distinction. Otherwise returns a
    sorted, duplicate-free list of absolute paths.
    """
    try:
        rel_paths = _git_ls_files(repo_root, '--', f'*{MANIFEST_SUFFIX}')
    except GitUnavailable:
        return None
    return [repo_root / rel for rel in sorted(set(rel_paths))]


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
    2. The swallow cannot hide a dropped target. ``TestCheckedInScriptCheckTargets``
       layers three guards over it: non-vacuity (catches hiding *everything*),
       both-known-anchors (catches hiding either of today's two targets), and
       — because neither of those notices a *future* third target quietly
       falling out — a count guard that re-derives the expected number by
       regex over the sidecar TEXT, bypassing this loader entirely, and
       asserts equality (code-review amendment, task 3649).

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

    Three outcomes, deliberately NOT collapsed into one (code-review amendment,
    task 3649 — the first cut returned ``None`` for all of them, so a caller
    rendered an infra flake as "committed at mode None", sending the reader to
    ``git update-index --chmod`` for a problem that was pure ``index.lock``
    contention):

    * exactly one index record -> its mode string;
    * zero records -> ``None``, meaning git answered and *rel* is genuinely
      **not tracked** — a real, distinct defect (a script that exists on disk
      but was never ``git add``ed), not an infra failure;
    * git could not answer -> :class:`GitUnavailable` propagates. Unlike
      :func:`discover_manifests` this runs at test time, not import time, so
      raising is safe and lets the caller skip.

    The pathspec is ``:(literal)`` magic, not the bare *rel*. ``git ls-files``
    takes a **pathspec**, so a bare value containing ``*``/``?``/``[`` would be
    glob-expanded and the first match's mode reported for a file that is not the
    target. With globbing disabled, several records can still come back — a
    conflicted index (stages 1/2/3) or a *rel* naming a directory (leading-path
    matching survives ``:(literal)``; measured: ``:(literal)scripts/tests`` ->
    60 records) — and that raises :class:`AmbiguousIndexEntry` rather than
    silently narrowing to ``records[0]``.
    """
    records = _git_ls_files(repo_root, '-s', '--', f':(literal){rel}')
    if not records:
        return None
    if len(records) > 1:
        raise AmbiguousIndexEntry(
            f'{rel} matched {len(records)} index records under {repo_root} — '
            'a conflicted index (stages 1/2/3) or a path naming a directory, '
            'so there is no single committed mode'
        )
    # `git ls-files -s` record: "<mode> <object> <stage>\t<path>".
    return records[0].split(' ', 1)[0]
