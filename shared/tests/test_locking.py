"""Tests for shared.locking — module path normalization used by both the
orchestrator scheduler and the task curator."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import shared.reify_checkout as reify_checkout
from shared.locking import (
    EXTENSIONLESS_FILENAMES,
    FILE_EXTENSIONS,
    directory_locks,
    files_to_modules,
    is_file_path,
    modules_conflict,
    normalize_lock,
    strip_directory_locks,
)

# ---------------------------------------------------------------------------
# Drift guard — pins shared.locking.FILE_EXTENSIONS to the canonical vector.
#
# Update _CANONICAL_EXTENSIONS AND FILE_EXTENSIONS together when the allowlist
# changes.  Also update the verbatim copy in
# fused-memory/src/fused_memory/middleware/lock_charter_guard.py AND its
# corresponding _CANONICAL_EXTENSIONS in
# fused-memory/tests/test_lock_charter_guard.py.
# ---------------------------------------------------------------------------

_CANONICAL_EXTENSIONS = [
    'c', 'cc', 'cjs', 'conf', 'cpp', 'css', 'csv', 'cts', 'cxx',
    'diff', 'envrc', 'example', 'example-systemd-config',
    'gcode', 'gitattributes', 'gitignore', 'gitkeep', 'gitmodules', 'golden', 'grammar',
    'h', 'hh', 'hpp', 'html',
    'icns', 'ico',
    'jq', 'js', 'json', 'jsonc', 'jsonl', 'jsx',
    'lock', 'log',
    'manifest', 'md', 'mjs', 'mts',
    'npmrc', 'png', 'py', 'python-version',
    'ri', 'rs', 'scss', 'service', 'sh', 'step', 'stl', 'svg',
    'template', 'timer', 'toml', 'ts', 'tsx', 'txt', 'typed',
    'yaml', 'yml',
]

# ---------------------------------------------------------------------------
# Classification corpora — deliberate verbatim duplicates of the corpora in
# fused-memory/tests/test_lock_charter_guard.py, for the same reason the two
# FILE_EXTENSIONS frozensets are duplicated (locking.py:16-25): the two copies
# are NOT linked by a re-export, so each must be pinned INDEPENDENTLY — and
# this is the copy the orchestrator scheduler (the α enforcement point) actually
# imports.  Keeping the two lists byte-identical is what lets a reader diff them.
#
# Rejected alternative: factor the corpora into one fixture module both suites
# import.  ``shared`` and ``fused-memory`` are separate packages with separate
# virtual environments and no shared test-support package; a cross-package test
# import would couple them exactly where the design deliberately does not, and
# would defeat the point of independent pinning.  Duplication + the two drift
# guards below is the cheaper and more faithful trade.
#
# One real tracked path per canonical extension (59 extensions / 62 paths) —
# dark-factory unless the comment says reify.  Enforced by
# TestIsFilePath::test_accept_corpus_covers_every_canonical_extension.
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
    'plans/evidence/scheduler-scoring-2026-08-06/candidates_scored.csv',
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
    # --- Extensions added by the git-ls-files sweep 2026-07-28 (#5726 / #3117).
    # Real tracked paths (verified with `git ls-files`), dark-factory unless
    # marked reify — the corpus records the evidence for each entry.
    'orchestrator/src/orchestrator/evals/reviewer_trial/corpus/mined/mined_1030.diff',
    'dashboard/dark-factory-dashboard-watchdog.timer',
    '.gitignore',
    '.gitattributes',
    '.gitmodules',
    '.python-version',
    '.envrc',
    'scripts/dashboard.service.template',
    'scripts/verify-task-845-tty.log',
    'cockpit/src/cockpit/py.typed',
    '.env.example',
    'fused-memory/fused-memory.service.example-systemd-config',
    'fused-memory/tests/fixtures/write_triage_calibration.jsonl',
    # The originating incident path: declaring this in metadata.files was
    # rejected as a directory lock because 'manifest' was absent from the list.
    'tests/infra/run-all-classification.manifest',  # reify
    'deploy/systemd/orchestrator-reify.service.d/warm-lane.conf',  # reify
    'crates/reify-doc/tests/snapshots/.gitkeep',  # reify
    'crates/reify-fdm/tests/fixtures/toolpath_bracket.golden',  # reify
    'gui/src/editor/reify.grammar',  # reify
    'gui/src-tauri/icons/icon.icns',  # reify
    'gui/src-tauri/icons/icon.ico',  # reify
    'scripts/reify-audit-snapshot-filter.jq',  # reify
    'tree-sitter-reify/.npmrc',  # reify
]

# REJECT corpus (α/γ shared vector — all must return False).  Verbatim duplicate
# of _REJECT_PATHS in fused-memory/tests/test_lock_charter_guard.py.  '/' and
# 'a/b/c/' cover the all-slashes → empty-segment branch of is_file_path, which
# had no test in this copy before.
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

# Real directories whose final segment carries a leading dot and no further dot.
# These MUST stay directory-shaped — see
# TestIsFilePath::test_leading_dot_directories_stay_directories for the rejected
# "leading-dot segment => FILE" alternative this corpus pins against.
_DOTTED_DIRECTORY_PATHS = [
    '.worktrees',
    '.task',
    '.claude',
    '.cargo',
    '.taskmaster',
]

# ---------------------------------------------------------------------------
# Extension-less tracked FILES (dark_factory #3248).
#
# The second half of the classifier: real tracked files whose final segment
# carries NO extension at all, so the FILE_EXTENSIONS lookup can never reach
# them.  Before #3248 every one of these classified as a DIRECTORY, which made
# `hooks/project-checks` undeclarable in metadata.files (γ rejected it with a
# LockCharterViolation) and — worse, because it was silent — made a task
# declaring ONLY such paths strip to an empty charter and fall through to the
# per-task `task-<id>` synthetic lock, holding no lock on the file it edits.
#
# Update _CANONICAL_EXTENSIONLESS AND EXTENSIONLESS_FILENAMES together when the
# vector changes.  Also update the verbatim copies in
# fused-memory/src/fused_memory/middleware/lock_charter_guard.py AND
# fused-memory/tests/test_lock_charter_guard.py.
#
# EVIDENCE — `git ls-files -s` over BOTH repos, re-measured 2026-08-09 (and
# identical to the 2026-07-31 measurement): every tracked path whose final
# segment contains no '.'.  dark-factory contributes Dockerfile LICENSE
# pre-commit pre-merge-commit project-checks; reify adds cargo
# cargo-audit-orphans reference-transaction.  Union = the 8 below.
#
# GITLINK EXCLUSION (load-bearing, not a measurement artefact): entries with
# mode 160000 are submodule mount points — `graphiti` and `mem0` in
# dark-factory — and are deliberately NOT admitted.  They are extension-less
# and would otherwise qualify, but admitting them would let a task declare an
# entire vendored submodule as its lock charter, which is strictly worse than
# the bug being fixed.  They stay DIRECTORIES by design.  The sweep below
# asserts this filter is doing real work rather than passing vacuously.
# ---------------------------------------------------------------------------

_CANONICAL_EXTENSIONLESS = [
    'Dockerfile',
    'LICENSE',
    'cargo',
    'cargo-audit-orphans',
    'pre-commit',
    'pre-merge-commit',
    'project-checks',
    'reference-transaction',
]

# One REAL tracked path per canonical extension-less name — dark-factory unless
# the comment says reify (same convention as _ACCEPT_PATHS above).  Verified
# tracked with `git ls-files --error-unmatch` on 2026-08-09.  Enforced complete
# by TestExtensionlessFilenames::test_accept_corpus_covers_every_canonical_name.
_EXTENSIONLESS_ACCEPT_PATHS = [
    'LICENSE',
    'fused-memory/docker/Dockerfile',
    'hooks/pre-commit',
    'hooks/pre-merge-commit',
    'hooks/project-checks',
    # reify
    'hooks/reference-transaction',
    # reify
    'scripts/agent-bin/cargo',
    # reify
    'scripts/cargo-audit-orphans',
]

# ---------------------------------------------------------------------------
# _resolve_reify_checkout — marker-bound adapter over `shared.reify_checkout`,
# the single source of layout-independent reify-checkout discovery AND of the
# skip wording that goes with it (established by task 3843 in the γ suite,
# fused-memory/tests/test_lock_charter_guard.py, and promoted to shared/ by
# task 3978 so orchestrator's scripts/verify.sh gate resolves through the same
# walk).  Feeds the module-level constants below once they are rewired to it.
# ---------------------------------------------------------------------------

_REIFY_GUARD_RELPATH = Path('scripts') / 'lock-charter-guard.sh'


def _resolve_reify_checkout(start: Path | None = None) -> reify_checkout.ReifyCheckout:
    """Locate the reify checkout carrying THIS module's guard script.

    Marker-bound adapter over the shared single source.  Every semantic —
    discovery order, REIFY_ROOT precedence, the measured worktree-vs-bare-
    checkout evidence table (and why "change parents[5] to parents[3]" is a
    REGRESSION rather than a fix), and why an override absent on disk is
    honored VERBATIM — lives in `shared.reify_checkout.resolve_reify_checkout`.
    Do not restate or re-derive them here, and do not reintroduce a local copy
    of the ancestor walk.

    Returns the shared `ReifyCheckout` — root AND provenance — rather than a
    bare path, so the skip reason is formatted from the very resolution that
    produced the root, instead of re-deriving provenance separately and
    risking the two disagreeing.

    *start* defaults to THIS file, and that default belongs here rather than
    in the shared helper: the walk has to begin at the CALL SITE, and shared's
    own ``__file__`` would start it in ``shared/src/shared/`` instead of here.
    """
    # Called through the module attribute on purpose: the delegation pin in
    # TestReifyCheckoutAdapter patches `reify_checkout.resolve_reify_checkout`,
    # which a `from ... import resolve_reify_checkout` binding would put out of
    # its reach.
    return reify_checkout.resolve_reify_checkout(
        _REIFY_GUARD_RELPATH, start=start or Path(__file__)
    )


# Repo roots for the tracked-corpus self-audit sweep.  dark-factory is
# resolved from this file (works in a worktree too: <root>/shared/tests/ ->
# parents[2] — deliberately UNCHANGED here, since parents[2] is correct in
# BOTH a bare checkout and a worktree: it names the checkout THIS test is
# running inside, main root or worktree root, either of which is a valid
# `git ls-files` target).  reify is a sibling checkout, resolved
# layout-independently via `shared.reify_checkout` (through the
# `_resolve_reify_checkout` adapter above) rather than a fixed parents[N]
# index — see that module's docstring for the measured worktree-vs-bare
# evidence table.  These constants are evaluated at IMPORT time, so
# REIFY_ROOT must be exported BEFORE pytest starts to steer them; a
# mid-session `monkeypatch.setenv` only affects direct
# `_resolve_reify_checkout()` calls (exactly what TestReifyCheckoutAdapter
# does above; it never touches these constants).
_DF_REPO_ROOT = Path(__file__).parents[2]
_REIFY_CHECKOUT = _resolve_reify_checkout()
_REIFY_REPO_ROOT: Path | None = _REIFY_CHECKOUT.root
_REIFY_GUARD_SCRIPT: Path | None = (
    _REIFY_REPO_ROOT / _REIFY_GUARD_RELPATH if _REIFY_REPO_ROOT is not None else None
)
# The skip wording comes from the shared builder too, not a hand-rolled
# string: it separates a discovery MISS (nobody has reify checked out here —
# benign) from a REIFY_ROOT naming a path that is not there (an operator
# typo, which must NAME the bad path).  Matches the wording the γ suite and
# orchestrator's verify gate emit for the same condition.
_REIFY_SKIP_REASON: str | None = reify_checkout.reify_skip_reason(
    _REIFY_GUARD_RELPATH, _REIFY_REPO_ROOT, named_by_env=_REIFY_CHECKOUT.named_by_env
)


def _reify_guard_vector(flag: str) -> list[str]:
    """Return the sorted vector emitted by the reify guard script's *flag*.

    Verbatim counterpart of the γ suite's helper of the same name, duplicated
    for the same reason the corpora are: the two suites are not linked by an
    import, and each must be able to pin its own copy of the predicate against
    bash on its own.

    LOUD on failure.  The callers skipif on the script's ABSENCE — a standalone
    checkout legitimately has no reify sibling — but a script that EXISTS and
    then fails is real signal and is raised, never skipped.  Measured
    2026-08-09: an unrecognised flag prints usage and exits 2, so a
    skip-on-any-non-zero would retire the guard the moment reify renamed an
    emitter, leaving it green forever while enforcing nothing.
    """
    result = subprocess.run(
        ['bash', str(_REIFY_GUARD_SCRIPT), flag],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'reify {_REIFY_GUARD_SCRIPT.name} {flag} exited {result.returncode}, '
        f'expected 0 — the emitter this cross-source drift guard depends on is '
        f'gone, renamed, or broken.  Check `bash {_REIFY_GUARD_SCRIPT} --help`.  '
        f'Hard failure, not a skip: silently retiring a cross-source guard would '
        f'let the three copies of the predicate diverge with every test green.\n'
        f'  stdout: {result.stdout[:400]!r}\n'
        f'  stderr: {result.stderr[:400]!r}'
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _tracked_entries(repo_root: Path) -> tuple[list[str], list[str]] | None:
    """``(non_gitlink_paths, gitlink_paths)`` for *repo_root*, or None if not a checkout.

    ``git ls-files -s`` (not plain ``ls-files``) is what makes the mode column
    available.  Mode ``160000`` means a submodule gitlink; those entries are
    returned SEPARATELY rather than discarded, because the sweeps need them to
    prove the exclusion is doing real work.  See the GITLINK EXCLUSION note
    above for why the filter is load-bearing rather than cosmetic.
    """
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files', '-s', '-z'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    paths: list[str] = []
    gitlinks: list[str] = []
    for entry in result.stdout.split('\0'):
        if not entry:
            continue
        # Format: "<mode> <sha> <stage>\t<path>"
        meta, _, path = entry.partition('\t')
        if not path:
            continue
        if meta.split(' ', 1)[0] == '160000':
            gitlinks.append(path)
        else:
            paths.append(path)
    return paths, gitlinks


def _tracked_file_paths(repo_root: Path) -> list[str] | None:
    """Tracked NON-GITLINK paths under *repo_root*, or None if not a checkout."""
    entries = _tracked_entries(repo_root)
    return None if entries is None else entries[0]


class TestReifyCheckoutAdapter:
    """`_resolve_reify_checkout` — this module's binding to the shared resolver."""

    def test_resolver_delegates_to_the_shared_single_source(self, tmp_path, monkeypatch):
        """The ancestor walk must live in ONE place — shared.reify_checkout.

        Deliberately white-box, and deliberately the only genuinely-RED
        assertion available for what is otherwise a behaviour-preserving
        refactor: in the ``.worktrees/<id>`` layout this suite normally runs
        in, ``parents[5]`` already resolves correctly, so every BLACK-BOX
        assertion in this file stays green both before and after this module
        grows its own copy of the ancestor walk — a faithful local duplicate
        would satisfy them all while defeating the single-sourcing this task
        exists to establish (the task's "do not each grow a private copy of
        the resolver" directive). What this pin buys is that a SECOND copy of
        the walk cannot quietly reappear in this file.
        """
        sentinel = tmp_path / 'sentinel-reify'
        start = tmp_path / 'dark-factory' / 'shared' / 'tests' / 'test_x.py'
        seen = {}

        def _stub(marker, start=None):
            seen['marker'] = Path(marker)
            seen['start'] = start
            return reify_checkout.ReifyCheckout(sentinel, False)

        monkeypatch.setattr(reify_checkout, 'resolve_reify_checkout', _stub)

        assert _resolve_reify_checkout(start).root == sentinel, (
            'this module must DELEGATE to '
            'shared.reify_checkout.resolve_reify_checkout rather than keep a '
            'private copy of the ancestor walk'
        )
        assert seen['marker'] == _REIFY_GUARD_RELPATH, (
            'the shared resolver must be bound to THIS call site marker '
            f'(expected {_REIFY_GUARD_RELPATH}, got {seen.get("marker")})'
        )
        assert seen['start'] == start

    def test_module_constants_track_the_resolver(self):
        """Single-source-of-truth pin against the call sites re-diverging.

        Covers the skip reason too: it must be the SHARED builder's output
        for this module's own resolution, not a hand-rolled string that can
        drift from the wording the γ suite or orchestrator's verify gate use
        for the same condition.

        RED today (before the constants are rewired): ``_REIFY_SKIP_REASON``
        does not exist yet in this module, so referencing it below raises
        ``NameError`` regardless of whether the first assertion happens to
        agree by coincidence in this developer's worktree.
        """
        checkout = _resolve_reify_checkout()

        assert checkout.root == _REIFY_REPO_ROOT
        if _REIFY_REPO_ROOT is not None:
            assert _REIFY_GUARD_SCRIPT == _REIFY_REPO_ROOT / _REIFY_GUARD_RELPATH
        shared_reason = reify_checkout.reify_skip_reason(
            _REIFY_GUARD_RELPATH, checkout.root, named_by_env=checkout.named_by_env
        )
        assert shared_reason == _REIFY_SKIP_REASON


# ---------------------------------------------------------------------------
# Planted-synthetic-layout harness — ported from
# orchestrator/tests/test_verify_role_integration.py:423-472 (task 3978),
# which reused the same technique from fused-memory/tests/test_lock_charter_guard.py
# (task 3843).  A real reify checkout is not enough to express the defect:
# in the ``.worktrees/<id>`` layout this suite normally runs in, `parents[5]`
# already resolves correctly (measured baseline: 135 passed, 0 skipped), so
# every black-box assertion against THIS machine's real checkout stays green
# both before and after the fix.  Only a synthetic ancestry — a module COPY
# imported from a planted tree — can express what a bare (non-worktree)
# checkout sees.
#
# The RED is host-independent in both directions: a parents[5] answer can
# never equal a tmp_path, and a tmp_path with no reify in its ancestry must
# resolve to None even on a machine where a real reify checkout exists.
# ---------------------------------------------------------------------------


@pytest.fixture
def planted_env(monkeypatch):
    """Neutralize the ambient env that would otherwise steer a module copy.

    REIFY_ROOT would override the discovery the planted-layout cases exist to
    exercise.  Returns monkeypatch so a case can set REIFY_ROOT back to a
    value it chose itself.
    """
    monkeypatch.delenv('REIFY_ROOT', raising=False)
    return monkeypatch


def _load_module_copy(tmp_path: Path, tests_relpath: str, *, plant_guard: bool = True):
    """Import a COPY of THIS module from a synthetic checkout layout.

    Plants ``<tmp>/src/reify/scripts/lock-charter-guard.sh`` (a real file;
    content irrelevant) when requested, creates ``<tmp>/src/<tests_relpath>/``,
    copies this module in, and loads it via ``spec_from_file_location``.

    The copy's ``shared.locking`` / ``shared.reify_checkout`` imports resolve
    from the parent process's sys.path, while its ``__file__`` is the PLANTED
    path — so its import-time constant resolution walks the SYNTHETIC
    ancestry, which is the whole point.  The temporary sys.modules entry is
    popped in a finally block so no copy outlives the call.
    """
    src = tmp_path / 'src'
    if plant_guard:
        planted = src / 'reify' / 'scripts' / 'lock-charter-guard.sh'
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text('#!/bin/sh\necho stub\n')

    tests_dir = src / tests_relpath
    tests_dir.mkdir(parents=True, exist_ok=True)
    copied = tests_dir / 'test_copy_probe.py'
    shutil.copy2(__file__, copied)

    module_name = '_reify_checkout_probe_' + re.sub(r'\W+', '_', tests_relpath)
    spec = importlib.util.spec_from_file_location(module_name, copied)
    assert spec is not None and spec.loader is not None, f'could not load {copied}'
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(module_name, None)
    return mod


_BARE_LAYOUT = 'dark-factory/shared/tests'
_WORKTREE_LAYOUT = 'dark-factory/.worktrees/4080/shared/tests'


class TestReifyCheckoutPlantedLayout:
    """The four cross-repo guards, proven ARMED from a bare-checkout layout.

    Every case here loads a COPY of this very module from a synthetic
    ancestry — see the harness comment above for why a real checkout cannot
    express the defect.
    """

    def test_gate_resolves_against_the_planted_checkout_not_this_machine(
        self, tmp_path, planted_env
    ):
        """The guard must find the reify sibling of the tree it is RUNNING in.

        Not of this developer's machine.  RED today: the bare-layout copy's
        constants are still ``parents[5]``-derived, which lands off-tree for
        this layout (measured: the ancestor five levels up from a file at
        ``<tmp>/src/dark-factory/shared/tests/`` is outside ``<tmp>`` itself).
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

        expected = tmp_path / 'src' / 'reify'
        assert expected == mod._REIFY_REPO_ROOT, (
            f'resolved {mod._REIFY_REPO_ROOT!r} instead of the reify checkout '
            f'planted beside it at {expected!r} — a fixed parents[N] index '
            f"answers for this developer's machine rather than for the tree "
            f'under test'
        )
        assert expected / 'scripts' / 'lock-charter-guard.sh' == mod._REIFY_GUARD_SCRIPT

    def test_worktree_and_bare_layouts_resolve_to_the_same_planted_checkout(
        self, tmp_path, planted_env
    ):
        """No fixed parents[N] index can satisfy both layouts.

        RED today: the worktree-layout copy's ``parents[5]`` happens to be
        correct here (mirroring why the real worktree suite sees 0 skips),
        but the bare-layout copy's does not, so the two disagree.
        """
        bare = _load_module_copy(tmp_path, _BARE_LAYOUT)
        worktree = _load_module_copy(tmp_path, _WORKTREE_LAYOUT)

        expected = tmp_path / 'src' / 'reify'
        assert expected == worktree._REIFY_REPO_ROOT
        assert worktree._REIFY_REPO_ROOT == bare._REIFY_REPO_ROOT, (
            f"'.worktrees/<id>' adds exactly two path segments, so a fixed "
            f'parents[N] cannot be correct in both layouts (got '
            f'worktree={worktree._REIFY_REPO_ROOT!r} vs bare={bare._REIFY_REPO_ROOT!r})'
        )

    def test_gate_admits_the_run_from_a_planted_checkout(self, tmp_path, planted_env):
        """Resolution is not enough — the skip decision must ADMIT the run.

        RED today: ``_REIFY_SKIP_REASON`` does not exist in this module yet.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

        assert mod._REIFY_SKIP_REASON is None, (
            f'the gate would SKIP against a planted checkout that really '
            f'carries scripts/lock-charter-guard.sh: {mod._REIFY_SKIP_REASON!r}'
        )

    def test_four_cross_repo_guards_are_armed_in_a_bare_checkout(self, tmp_path, planted_env):
        """The direct expression of "silently skipping 4 cross-repo lock guards".

        Reads the DECORATED functions' evaluated ``pytestmark`` on the
        bare-layout module copy: decorators evaluate their arguments at
        IMPORT time, which is exactly the moment this bug silently flips a
        guard off. The two Tier-2 ``skipif`` conditions must evaluate False
        (armed, not silently retired), and the two parametrized sweeps'
        ``reify`` argvalue must carry the checkout planted BESIDE the bare
        copy, not an off-tree answer.

        RED today: the bare copy's ``_REIFY_GUARD_SCRIPT`` is an off-tree,
        non-existent path, so both skipif conditions evaluate True (SKIPPED),
        and both sweeps' ``reify`` argvalue is that same off-tree
        ``_REIFY_REPO_ROOT`` rather than the planted checkout.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT)
        expected_root = tmp_path / 'src' / 'reify'

        def _skipif_condition(func):
            marks = [m for m in func.pytestmark if m.name == 'skipif']
            assert marks, f'{func.__name__} must carry a skipif mark'
            return marks[0].args[0]

        def _parametrize_reify_argvalue(func):
            marks = [m for m in func.pytestmark if m.name == 'parametrize']
            assert marks, f'{func.__name__} must carry a parametrize mark'
            return dict(marks[0].args[1])['reify']

        assert (
            _skipif_condition(
                mod.TestExtensionlessFilenamesDriftGuard.test_extensionless_drift_guard_vs_reify_script
            )
            is False
        ), 'guard 1/4: extensionless drift Tier-2 skipif must be ARMED (condition False) in a bare checkout'
        assert (
            _skipif_condition(
                mod.TestFileExtensionsDriftGuard.test_extension_drift_guard_vs_reify_script
            )
            is False
        ), 'guard 2/4: extension drift Tier-2 skipif must be ARMED (condition False) in a bare checkout'
        assert (
            _parametrize_reify_argvalue(
                mod.TestExtensionlessFilenames.test_no_canonical_name_is_also_a_directory
            )
            == expected_root
        ), 'guard 3/4: reify parametrization must carry the planted checkout, not an off-tree answer'
        assert (
            _parametrize_reify_argvalue(
                mod.TestExtensionlessFilenames.test_every_tracked_extensionless_file_is_allowlisted
            )
            == expected_root
        ), 'guard 4/4: reify parametrization must carry the planted checkout, not an off-tree answer'

    def test_discovery_miss_skips_instead_of_answering_from_this_machine(
        self, tmp_path, planted_env
    ):
        """THE structural pin against a hardcoded/relative default.

        Nothing named reify exists anywhere in the planted ancestry, so the
        honest answer is None + a skip.  A fixed parents[N] index instead
        always produces SOME path — possibly one that exists on this
        developer's machine — so the guard would silently run (or silently
        retire) against a checkout unrelated to the tree under test.

        RED today, structurally: the current ``_REIFY_REPO_ROOT`` is a raw
        ``parents[N] / 'reify'`` expression that is never None regardless of
        whether anything exists on disk at that path, and
        ``_REIFY_SKIP_REASON`` does not exist yet.
        """
        mod = _load_module_copy(tmp_path, _BARE_LAYOUT, plant_guard=False)

        assert mod._REIFY_REPO_ROOT is None, (
            f'no reify checkout exists in the planted ancestry, but the '
            f'module resolved {mod._REIFY_REPO_ROOT!r} — an off-tree answer'
        )
        reason = mod._REIFY_SKIP_REASON
        assert isinstance(reason, str) and reason
        assert 'REIFY_ROOT' in reason, f'the skip must name the override: {reason!r}'
        assert 'scripts/lock-charter-guard.sh' in reason, (
            f'the skip must name the marker: {reason!r}'
        )


class TestSkipUnlessCheckout:
    """`_skip_unless_checkout` — the reachability precondition step-6 wires
    both reify-parametrized sweeps through, so a None or set-but-absent reify
    root SKIPS honestly instead of raising ``AttributeError`` at the sweeps'
    inline ``repo_root.is_dir()``.

    RED today: `_skip_unless_checkout` does not exist in this module, so
    every case below fails with ``NameError``.  Ported in shape from
    fused-memory/tests/test_lock_charter_guard.py:721-762's helper of the
    same name, but pinned MORE thoroughly than that suite pins its own copy:
    the γ suite only exercises the helper indirectly through its own
    parametrized sweeps, which never observe the ``None`` arm on a host where
    reify IS discoverable.  These cases call the helper directly, so the
    ``None`` arm is pinned host-independently regardless of what any given
    machine has checked out.
    """

    def test_discovery_miss_skips_with_the_shared_wording(self):
        """A None root is the discovery-miss arm — nobody has a reify sibling.

        The wording must come from the shared builder, not a local string, so
        it cannot drift from what the Tier-2 skipifs above and orchestrator's
        verify gate say about the same condition.  ``pytest.skip.Exception``
        is pytest's own public handle for the outcome ``pytest.skip`` raises
        (documented on ``pytest.skip`` itself), so capturing it here needs no
        dependency on the private ``_pytest.outcomes`` import path.
        """
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_unless_checkout('reify', None)

        reason = str(excinfo.value)
        assert isinstance(reason, str) and reason, (
            f'a None root is the discovery-miss arm, which always yields a '
            f'reason from the shared builder — a falsy one here would turn '
            f'this skip into a phantom pass, got {reason!r}'
        )
        assert reason == reify_checkout.reify_skip_reason(
            _REIFY_GUARD_RELPATH, None, named_by_env=False
        ), 'the discovery-miss wording must come from the shared builder, not a local string'

    def test_set_but_absent_root_skips_naming_the_path(self, tmp_path):
        """A REIFY_ROOT-shaped path that is not on disk must be NAMED in the
        skip reason, so an operator's typo is self-evident in `pytest -rs`
        output instead of silently falling back to discovery.
        """
        missing = tmp_path / 'no-such-reify-checkout'

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _skip_unless_checkout('reify', missing)

        reason = str(excinfo.value)
        assert str(missing) in reason, (
            f'the skip reason must name the bad path so a REIFY_ROOT typo is '
            f'self-evident: {reason!r}'
        )

    def test_present_directory_is_returned_unchanged(self, tmp_path):
        """A real directory must be RETURNED, not skipped.

        Otherwise the helper could skip everything vacuously and the sweeps
        it guards would never run against a real checkout.
        """
        assert _skip_unless_checkout('reify', tmp_path) == tmp_path


class TestReifyGuardVectorNoneGuard:
    """`_reify_guard_vector` must fail LOUDLY on a None guard script.

    Its docstring already commits the helper to LOUD-on-failure for a script
    that exists and then fails; this pins the same commitment for the
    ``None`` arm the step-4 Optional widening opened up (project
    no-silent-fail-soft invariant), so a caller that lost its skipif guard
    fails with a clear AssertionError instead of shelling out to `bash None`.
    """

    def test_fails_loudly_when_guard_script_is_none(self, monkeypatch):
        """A None `_REIFY_GUARD_SCRIPT` must raise an AssertionError naming
        the constant, never shell out to `bash None` and let bash's resulting
        non-zero exit read as a real cross-source drift failure.

        RED today: calling this with `_REIFY_GUARD_SCRIPT` patched to None
        instead raises `AttributeError` (`None.name` inside the message that
        formats the non-zero-exit assertion) — confusing rather than loud,
        and not the AssertionError this case pins.
        """
        monkeypatch.setattr(sys.modules[__name__], '_REIFY_GUARD_SCRIPT', None)

        with pytest.raises(AssertionError, match='_REIFY_GUARD_SCRIPT'):
            _reify_guard_vector('--list-extensions')


class TestNormalizeLock:
    def test_default_depth_is_two(self):
        assert normalize_lock('crates/reify-types/src/persistent.rs') == 'crates/reify-types'

    def test_depth_three(self):
        assert (
            normalize_lock('crates/reify-compiler/src/foo.rs', depth=3)
            == 'crates/reify-compiler/src'
        )

    def test_depth_one(self):
        assert normalize_lock('crates/reify-types/src/persistent.rs', depth=1) == 'crates'

    def test_leading_slash_stripped(self):
        assert normalize_lock('/crates/reify-types/src/foo.rs') == 'crates/reify-types'

    def test_trailing_slash_stripped(self):
        assert normalize_lock('crates/reify-types/src/') == 'crates/reify-types'

    def test_short_path_returned_as_is(self):
        # Path with fewer segments than depth should return whatever is there
        assert normalize_lock('crates', depth=3) == 'crates'
        assert normalize_lock('crates/foo', depth=3) == 'crates/foo'

    def test_empty_returns_empty(self):
        assert normalize_lock('') == ''

    def test_single_component(self):
        assert normalize_lock('foo.py', depth=2) == 'foo.py'


class TestFilesToModules:
    def test_dedupes_same_module(self):
        files = [
            'crates/reify-compiler/src/foo.rs',
            'crates/reify-compiler/src/bar.rs',
            'crates/reify-compiler/src/sub/baz.rs',
        ]
        # depth=3 normalizes to crates/reify-compiler/src; all collapse to one key
        assert files_to_modules(files, depth=3) == ['crates/reify-compiler/src']

    def test_distinct_modules_preserved(self):
        files = [
            'crates/reify-compiler/src/foo.rs',
            'crates/reify-eval/src/bar.rs',
            'crates/reify-types/src/persistent.rs',
        ]
        result = files_to_modules(files, depth=3)
        assert result == [
            'crates/reify-compiler/src',
            'crates/reify-eval/src',
            'crates/reify-types/src',
        ]

    def test_sorted_output(self):
        files = [
            'z/module/foo.rs',
            'a/module/foo.rs',
            'm/module/foo.rs',
        ]
        assert files_to_modules(files, depth=2) == ['a/module', 'm/module', 'z/module']

    def test_empty_input(self):
        assert files_to_modules([], depth=2) == []

    def test_empty_strings_skipped(self):
        assert files_to_modules(['', 'foo/bar.py', ''], depth=2) == ['foo/bar.py']

    def test_accepts_any_iterable(self):
        # generator
        gen = (p for p in ['foo/bar.py', 'foo/baz.py'])
        assert files_to_modules(gen, depth=2) == ['foo/bar.py', 'foo/baz.py']


class TestModulesConflict:
    def test_exact_match_conflicts(self):
        assert modules_conflict('crates/reify-types', 'crates/reify-types')

    def test_parent_prefix_of_child_conflicts(self):
        # A sub-lock_depth parent ('foo') conflicts with a deeper child.
        assert modules_conflict('foo', 'foo/bar')
        assert modules_conflict('foo/bar', 'foo')

    def test_symmetric(self):
        assert modules_conflict('a/b', 'a/b/c') == modules_conflict('a/b/c', 'a/b')

    def test_sibling_paths_do_not_conflict(self):
        assert not modules_conflict('crates/reify-types', 'crates/reify-eval')

    def test_shared_string_prefix_without_slash_does_not_conflict(self):
        # 'foo' must not be treated as a prefix of 'foobar' (no path boundary).
        assert not modules_conflict('foo', 'foobar')

    def test_disjoint_paths_do_not_conflict(self):
        assert not modules_conflict('a/b', 'c/d')


class TestIsFilePath:
    """is_file_path — pure-string file vs directory classifier."""

    def test_py_file_is_file(self):
        assert is_file_path('src/app.py')

    def test_rs_file_is_file(self):
        assert is_file_path('foo/bar.rs')

    def test_extension_less_segment_is_directory(self):
        assert not is_file_path('backend')

    def test_directory_path_no_extension_is_directory(self):
        assert not is_file_path('crates/reify-eval/src')

    def test_trailing_slash_is_directory(self):
        assert not is_file_path('src/server/')

    def test_uppercase_extension_is_not_file_case_sensitive(self):
        # Case-sensitive allowlist: 'MD' not in FILE_EXTENSIONS
        assert not is_file_path('README.MD')

    def test_file_extensions_is_frozenset(self):
        assert isinstance(FILE_EXTENSIONS, frozenset)
        # Spot-check canonical members
        assert 'py' in FILE_EXTENSIONS
        assert 'rs' in FILE_EXTENSIONS
        assert 'ts' in FILE_EXTENSIONS
        assert 'md' in FILE_EXTENSIONS

    @pytest.mark.parametrize('path', _ACCEPT_PATHS)
    def test_accept_corpus_paths_are_files(self, path: str):
        """Every canonical extension is run through the α copy of the predicate.

        One representative path per canonical extension (all 59, not just the 22
        added by the 2026-07-28 sweep).  Before this amendment only the 22 new
        extensions had behavioural coverage here while the original 36 had nothing
        but four spot-checks in ``test_file_extensions_is_frozenset`` — leaving
        the copy the orchestrator scheduler actually imports pinned strictly
        weaker than the fused-memory copy, which defeats the point of
        independently pinning two deliberately-duplicated definitions.

        Paths added by the sweep are genuinely tracked files in dark-factory or
        reify (verified with ``git ls-files``) — not synthetic filenames — so the
        corpus documents the actual evidence that motivated each entry rather than
        restating the allowlist.  ``run-all-classification.manifest`` is the path
        from the originating incident: declaring it in ``metadata.files`` was
        rejected as a directory lock because ``manifest`` was absent.
        """
        assert is_file_path(path) is True, (
            f'{path!r} must classify as a file — its extension is on the '
            f'canonical allowlist (59 entries as of dark_factory #3715 hotfix)'
        )

    @pytest.mark.parametrize('path', _REJECT_PATHS)
    def test_reject_corpus_paths_are_directories(self, path: str):
        """Directory-style paths must classify as directories (False)."""
        assert is_file_path(path) is False, (
            f'{path!r} is directory-shaped and must NOT classify as a file'
        )

    def test_accept_corpus_covers_every_canonical_extension(self):
        """_ACCEPT_PATHS must exercise every entry in _CANONICAL_EXTENSIONS.

        Turns the corpus comment's "one path per canonical extension" claim into
        an enforced invariant, so an extension cannot be added to FILE_EXTENSIONS
        and pinned in the drift guard while never being run through an actual
        classification assertion.  Mirrors the assertion of the same name in
        fused-memory/tests/test_lock_charter_guard.py, which also carries the
        complementary corpus→allowlist sweep over both repos' tracked files.
        That sweep is deliberately NOT duplicated here: it checks a repo-level
        property of the tree (no tracked extension is missing), and this copy's
        vector is already pinned identical to the γ copy's by the two drift
        guards, so a second subprocess sweep would add cost without adding
        detection power.

        Coverage is derived from ``is_file_path`` itself rather than a
        hand-written re-implementation of its segment parsing: a path counts for
        ``ext`` only if the predicate ACCEPTS it and it ends in ``.<ext>``.
        Because every canonical extension is dot-free (asserted first),
        ``endswith('.' + ext)`` holds iff the substring after the LAST dot equals
        ``ext`` — exactly the lookup ``is_file_path`` performs.
        """
        dotted = sorted(e for e in _CANONICAL_EXTENSIONS if '.' in e)
        assert not dotted, (
            f'canonical extensions must be dot-free for the endswith() '
            f'equivalence this test relies on; got {dotted!r}'
        )

        accepted = [p for p in _ACCEPT_PATHS if is_file_path(p)]
        uncovered = sorted(
            ext
            for ext in _CANONICAL_EXTENSIONS
            if not any(p.endswith(f'.{ext}') for p in accepted)
        )
        assert not uncovered, (
            f'{len(uncovered)} allowlisted extension(s) are pinned in '
            f'_CANONICAL_EXTENSIONS but never classified by any _ACCEPT_PATHS '
            f'entry: {uncovered!r}\n'
            f'Add one real representative path per extension to _ACCEPT_PATHS.'
        )

    @pytest.mark.parametrize('path', _DOTTED_DIRECTORY_PATHS)
    def test_leading_dot_directories_stay_directories(self, path: str):
        """Rejected alternative: a blanket "leading-dot segment => FILE" rule.

        Seven of the 22 extensions added by the 2026-07-28 sweep are dotfiles
        (.gitignore .gitkeep .envrc .npmrc .gitattributes .gitmodules
        .python-version), so a one-line predicate rule "a segment starting with
        '.' is a file" looks like an attractive simplification of seven list
        entries.  It was considered and REJECTED: these paths are real
        DIRECTORIES, and such a rule would flip every one of them to FILE —
        letting a task declare ``.worktrees`` (the orchestrator's entire
        worktree pool) or ``.task`` as its lock charter.  That is precisely the
        over-wide-charter failure this guard exists to prevent.

        The allowlist must therefore stay ENUMERATED: that is the property
        making an unknown dotted segment default to directory.  Verified against
        reify's α implementation — ``lock-charter-guard.sh classify .worktrees``
        (and .task/.claude/.cargo/.taskmaster) returns REJECT.

        These assertions pass both before and after the widening; they are
        regression pins for a rejected design, not a behaviour change.
        """
        assert not is_file_path(path), (
            f'{path!r} is a real directory and must NOT classify as a file; a '
            f'blanket leading-dot=>file rule was rejected for exactly this reason'
        )


class TestExtensionlessFilenamesDriftGuard:
    """Pins shared.locking.EXTENSIONLESS_FILENAMES to the canonical vector."""

    def test_matches_canonical_vector(self):
        assert sorted(EXTENSIONLESS_FILENAMES) == _CANONICAL_EXTENSIONLESS

    @pytest.mark.skipif(_REIFY_SKIP_REASON is not None, reason=_REIFY_SKIP_REASON or '')
    def test_extensionless_drift_guard_vs_reify_script(self):
        """Tier-2 (cross-source): this α copy must match reify --list-extensionless.

        WHY THIS EXISTS ON THE α SIDE TOO, rather than being left to the γ
        suite's identical test.  Tier-1 above compares this frozenset against
        ``_CANONICAL_EXTENSIONLESS``, a literal in THIS file.  An edit that
        changes both together — the natural way to "add a name" — therefore
        passes every guard in the shared tree while silently diverging from the
        γ copy and from bash.  The whole hole was that nothing in this package
        could observe the other two copies.

        With both suites pinned against the same bash emitter the α↔γ agreement
        follows transitively: α == bash here and γ == bash there implies α == γ,
        which is what lets the two Python copies stay verbatim duplicates
        without a re-export linking them.  Neither Tier-2 alone gives that; both
        together do, and that is the precise sense in which three-way agreement
        is ENFORCED.

        A present-but-failing script is a hard failure, not a skip — see
        _reify_guard_vector.
        """
        script_names = _reify_guard_vector('--list-extensionless')
        assert script_names == sorted(EXTENSIONLESS_FILENAMES), (
            f'α/γ/bash drift detected on the extension-less vector!\n'
            f'  reify --list-extensionless        : {script_names!r}\n'
            f'  α EXTENSIONLESS_FILENAMES         : {sorted(EXTENSIONLESS_FILENAMES)!r}\n'
            f'Update EXTENSIONLESS_FILENAMES here, in '
            f'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            f"and reify's own scripts/lock-charter-guard.sh _EXTLESS, plus "
            f'_CANONICAL_EXTENSIONLESS in both test copies — all five places.'
        )

    def test_is_frozenset(self):
        assert isinstance(EXTENSIONLESS_FILENAMES, frozenset)

    def test_members_are_bare_final_segments(self):
        """Every member must be a non-empty, dot-free, slash-free name.

        Structural invariant, not style policing.  A member is compared against
        a path's FINAL SEGMENT, so:
          - a name containing '.' is in the wrong allowlist — its post-dot token
            belongs in FILE_EXTENSIONS, and putting it here would shadow that
            lookup with an exact-name match that almost never fires;
          - a name containing '/' could never equal a final segment, so it would
            be permanently dead weight that reads as coverage.
        """
        for name in sorted(EXTENSIONLESS_FILENAMES):
            assert isinstance(name, str) and name, f'{name!r} must be a non-empty str'
            assert '.' not in name, (
                f'{name!r} contains a dot — a dotted name is an EXTENSION and '
                f'belongs in FILE_EXTENSIONS, not EXTENSIONLESS_FILENAMES'
            )
            assert '/' not in name, (
                f'{name!r} contains a slash — members are matched against a '
                f"path's final segment, which can never contain '/'"
            )


class TestExtensionlessFilenames:
    """Extension-less tracked FILES classify as files (dark_factory #3248).

    The hazard these pin, stated concretely: two tasks editing
    ``hooks/project-checks`` concurrently could both hold NO lock on it, because
    a charter that strips to empty degrades to a per-task ``task-<id>`` synthetic
    lock that conflicts with nothing.
    """

    @pytest.mark.parametrize('path', _EXTENSIONLESS_ACCEPT_PATHS)
    def test_extensionless_accept_corpus_paths_are_files(self, path: str):
        assert is_file_path(path) is True, (
            f'{path!r} is a real tracked FILE whose final segment carries no '
            f'extension; it must classify as a file via EXTENSIONLESS_FILENAMES'
        )

    def test_accept_corpus_covers_every_canonical_name(self):
        """_EXTENSIONLESS_ACCEPT_PATHS must exercise every canonical name.

        The GAMMA-4 trap guard, mirroring
        test_accept_corpus_covers_every_canonical_extension.  Adding a name to
        the allowlist without a matching real accept path would otherwise swap
        one red test for a different red test rather than closing the gap.

        Coverage is derived from ``is_file_path`` itself rather than a
        re-implementation of its segment parsing: a path counts for ``name``
        only if the predicate ACCEPTS it and its final segment IS ``name``.
        """
        accepted = [p for p in _EXTENSIONLESS_ACCEPT_PATHS if is_file_path(p)]
        uncovered = sorted(
            name
            for name in _CANONICAL_EXTENSIONLESS
            if not any(p.rsplit('/', 1)[-1] == name for p in accepted)
        )
        assert not uncovered, (
            f'{len(uncovered)} allowlisted name(s) are pinned in '
            f'_CANONICAL_EXTENSIONLESS but never classified by any '
            f'_EXTENSIONLESS_ACCEPT_PATHS entry: {uncovered!r}\n'
            f'Add one real tracked representative path per name.'
        )

    @pytest.mark.parametrize(
        ('repo', 'repo_root'),
        [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
    )
    def test_no_canonical_name_is_also_a_directory(self, repo: str, repo_root: Path):
        """No member may appear as a NON-FINAL component of any tracked path.

        This is what bounds the risk of matching on a bare basename.  A member
        that were ALSO a directory name somewhere in the tree would make that
        directory declarable as a lock charter — the exact over-wide-charter
        failure the guard exists to prevent.  Measured zero collisions across
        both repos on 2026-08-09; this pins that property permanently rather
        than leaving it as a one-time observation.

        SCOPE, stated plainly so the guard is not over-read: it bounds the risk
        for these TWO checkouts only, while the predicate governs lock charters
        for every project the orchestrator targets.  Several members are generic
        basenames (`cargo`, `Dockerfile`, `LICENSE`, `pre-commit`), so a third
        project with a real `tools/cargo/` DIRECTORY would have it classified as
        a file and retained as a subtree-wide prefix lock, unswept by this test.
        That is why admitting a generic basename is a cross-project commitment —
        see the CROSS-PROJECT SCOPE note on the EXTENSIONLESS_FILENAMES vector.
        """
        if not repo_root.is_dir():
            pytest.skip(f'{repo} checkout not present at {repo_root}')
        paths = _tracked_file_paths(repo_root)
        if paths is None:
            pytest.skip(f'{repo_root} is not a git checkout (git ls-files failed)')
        assert paths, f'git ls-files returned no tracked paths for {repo_root}'

        collisions: dict[str, str] = {}
        for path in paths:
            for component in path.split('/')[:-1]:
                if component in EXTENSIONLESS_FILENAMES:
                    collisions.setdefault(component, path)

        assert not collisions, (
            f'{len(collisions)} EXTENSIONLESS_FILENAMES member(s) are also '
            f'DIRECTORY names in {repo}, so declaring that directory would now '
            f'be accepted as a file-level lock charter:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(collisions.items()))
            + '\nRemove the name from the allowlist, or rename the directory.'
        )

    @pytest.mark.parametrize(
        ('repo', 'repo_root'),
        [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
    )
    def test_every_tracked_extensionless_file_is_allowlisted(
        self, repo: str, repo_root: Path
    ):
        """Corpus->allowlist self-audit: the 9th extensionless file must go RED.

        Turns "add 8 names" into a self-auditing invariant.  When someone adds a
        new extension-less tracked file to either repo, this test names it here
        rather than letting it surface later as a LockCharterViolation at
        submit time — or, worse, as a silent under-lock.

        This sweep IS duplicated in the γ suite, unlike the extension-corpus
        sweep which is deliberately α-free (see
        test_accept_corpus_covers_every_canonical_extension for that rationale).
        The difference is the failure mode: the extension vector is fully pinned
        by the two drift guards, whereas this one gates a REPO-level property —
        a newly-added tracked file changes the correct answer without any test
        vector changing, so whichever suite runs must be able to see it.

        Gitlink mount points are excluded by _tracked_file_paths and must NOT be
        admitted; that exclusion is asserted below.  The gitlink set is DERIVED
        from git's own mode column rather than hardcoded, which matters in both
        directions: hardcoding ``('graphiti', 'mem0')`` made the assertion
        vacuously true for the reify parametrization (reify vendors no
        submodules), and would have gone red for no real defect if dark-factory
        ever de-vendored them.  A repo with no submodules now skips the
        sub-assertion explicitly instead of passing it silently — so read this
        guard as non-vacuous exactly where gitlinks actually exist, which today
        is the dark-factory parametrization.
        """
        if not repo_root.is_dir():
            pytest.skip(f'{repo} checkout not present at {repo_root}')
        entries = _tracked_entries(repo_root)
        if entries is None:
            pytest.skip(f'{repo_root} is not a git checkout (git ls-files failed)')
        paths, gitlinks = entries
        assert paths, f'git ls-files returned no tracked paths for {repo_root}'

        # The gitlink filter must be doing real work, not passing vacuously.
        if gitlinks:
            overlap = sorted(set(gitlinks) & set(paths))
            assert not overlap, (
                f'{overlap!r} appear as BOTH mode-160000 gitlinks and ordinary '
                f'tracked paths in {repo}; _tracked_entries must partition the '
                f'corpus, and admitting a submodule mount point would let a task '
                f'declare an entire vendored submodule as its lock charter'
            )
            for gitlink in gitlinks:
                assert not is_file_path(gitlink), (
                    f'{gitlink!r} is a submodule mount point (mode 160000) and '
                    f'must stay a DIRECTORY — see the GITLINK EXCLUSION note on '
                    f'the corpus.  Admitting it would make an entire vendored '
                    f'submodule declarable as a lock charter, strictly worse '
                    f'than the bug #3248 fixed.'
                )

        unknown: dict[str, str] = {}
        for path in paths:
            seg = path.rsplit('/', 1)[-1]
            if '.' in seg:
                continue
            if not is_file_path(path):
                unknown.setdefault(seg, path)

        assert not unknown, (
            f'{len(unknown)} extension-less tracked {repo} file(s) classify as '
            f'DIRECTORIES, so any task declaring one in metadata.files is either '
            f'rejected with a LockCharterViolation or — silently — stripped to an '
            f'empty charter and given a task-<id> synthetic lock that conflicts '
            f'with nothing:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(unknown.items()))
            + '\nAdd each name to EXTENSIONLESS_FILENAMES in '
            'shared/src/shared/locking.py AND '
            'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            'and to reify scripts/lock-charter-guard.sh _EXTLESS (all three must '
            'agree — omitting reify reds '
            'test_extensionless_drift_guard_vs_reify_script), '
            'plus _CANONICAL_EXTENSIONLESS and _EXTENSIONLESS_ACCEPT_PATHS in '
            'both test copies.'
        )


class TestDirectoryLocks:
    """directory_locks — returns ordered directory-like entries, drops files."""

    def test_returns_directory_entries_only(self):
        files = ['crates/reify-eval/src', 'a/b.py', 'crates/reify-eval/tests']
        assert directory_locks(files) == ['crates/reify-eval/src', 'crates/reify-eval/tests']

    def test_preserves_order(self):
        files = ['z/dir', 'a/dir', 'm/dir']
        assert directory_locks(files) == ['z/dir', 'a/dir', 'm/dir']

    def test_deduplicates(self):
        files = ['dir', 'dir', 'a/b.py']
        assert directory_locks(files) == ['dir']

    def test_skips_non_str_tokens(self):
        files = [None, 42, 'backend', 'src/foo.py']  # type: ignore[list-item]
        assert directory_locks(files) == ['backend']

    def test_skips_empty_and_whitespace(self):
        files = ['', '   ', 'backend']
        assert directory_locks(files) == ['backend']

    def test_empty_input(self):
        assert directory_locks([]) == []


class TestStripDirectoryLocks:
    """strip_directory_locks — inverse of directory_locks; keeps only file entries."""

    def test_strips_directories_keeps_files(self):
        files = ['crates/reify-eval/src', 'a/b.py', 'crates/reify-eval/tests', 'c.rs']
        assert strip_directory_locks(files) == ['a/b.py', 'c.rs']

    def test_empty_input(self):
        assert strip_directory_locks([]) == []

    def test_all_directories_returns_empty(self):
        assert strip_directory_locks(['backend', 'crates/reify-eval/src']) == []

    def test_all_files_returns_all(self):
        files = ['src/foo.py', 'src/bar.rs']
        assert strip_directory_locks(files) == ['src/foo.py', 'src/bar.rs']

    def test_skips_non_str_tokens(self):
        files = [None, 42, 'backend', 'src/foo.py']  # type: ignore[list-item]
        assert strip_directory_locks(files) == ['src/foo.py']

    def test_skips_empty_and_whitespace(self):
        files = ['', '   ', 'src/foo.py']
        assert strip_directory_locks(files) == ['src/foo.py']


class TestFileExtensionsDriftGuard:
    """Pin shared.locking.FILE_EXTENSIONS to the canonical extension vector.

    This is the α-copy drift guard for the shared.locking canonical source.
    The corresponding γ-copy guard lives in
    fused-memory/tests/test_lock_charter_guard.py::test_extension_drift_guard
    and uses the same ``_CANONICAL_EXTENSIONS`` vector.  Both must be updated
    together whenever the allowlist changes.
    """

    def test_extension_drift_guard(self):
        """sorted(FILE_EXTENSIONS) must match _CANONICAL_EXTENSIONS.

        Update _CANONICAL_EXTENSIONS AND FILE_EXTENSIONS together when the
        allowlist changes.  Also update the verbatim copy in
        lock_charter_guard.py and its _CANONICAL_EXTENSIONS list.
        """
        assert sorted(FILE_EXTENSIONS) == _CANONICAL_EXTENSIONS, (
            f'shared.locking.FILE_EXTENSIONS has drifted from the canonical vector.\n'
            f'  canonical : {_CANONICAL_EXTENSIONS!r}\n'
            f'  actual    : {sorted(FILE_EXTENSIONS)!r}\n'
            f'Update FILE_EXTENSIONS and _CANONICAL_EXTENSIONS together; also update '
            f'lock_charter_guard.py.'
        )

    @pytest.mark.skipif(_REIFY_SKIP_REASON is not None, reason=_REIFY_SKIP_REASON or '')
    def test_extension_drift_guard_vs_reify_script(self):
        """Tier-2 (cross-source): this α copy must match reify --list-extensions.

        The extension-half counterpart of
        TestExtensionlessFilenamesDriftGuard::test_extensionless_drift_guard_vs_reify_script,
        added for the same reason and closing the same structural hole: before
        this, ONLY the γ suite invoked the bash emitter, so an α-only edit that
        moved FILE_EXTENSIONS and the local _CANONICAL_EXTENSIONS together
        passed everything in this package while diverging from the other two
        copies.  That hole predates #3248 — the extension vector has been
        α-unpinned against bash since the two Python copies were split — and is
        fixed here rather than left as the one asymmetry between the suites.
        """
        script_exts = _reify_guard_vector('--list-extensions')
        assert script_exts == sorted(FILE_EXTENSIONS), (
            f'α/γ/bash drift detected on the extension vector!\n'
            f'  reify --list-extensions : {script_exts!r}\n'
            f'  α FILE_EXTENSIONS       : {sorted(FILE_EXTENSIONS)!r}\n'
            f'Update FILE_EXTENSIONS here, in lock_charter_guard.py, and in '
            f"reify's scripts/lock-charter-guard.sh _EXTS, plus "
            f'_CANONICAL_EXTENSIONS in both test copies.'
        )
