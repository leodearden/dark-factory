"""Unit tests for fused_memory.middleware.lock_charter_guard.

Step 1 (RED → step-2 GREEN): predicate + drift guard
Step 3 (RED → step-4 GREEN): list-gate helpers
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
import shared.locking as shared_locking
from _census_helpers import census_lines
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware.gitignored_deliverable_guard import (
    GitignoredDeliverableFinding,
    gitignored_deliverable_reject,
)
from fused_memory.middleware.lock_charter_guard import (
    EXTENSIONLESS_FILENAMES,
    FILE_EXTENSIONS,
    directory_locks,
    extract_files,
    is_file_path,
    lock_charter_error,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_SOURCE = 'lock_charter_guard.extract_files'


# ---------------------------------------------------------------------------
# Drift guards — what this suite pins, and what it deliberately does not:
#
# Tier 1 (test_extension_drift_guard, test_extensionless_drift_guard): each
#   vector against a canonical literal in this file.  Update BOTH together when
#   the allowlist changes.  This does NOT catch α/γ drift on its own.
#
# α == γ (test_gamma_vectors_match_shared_locking): this copy against
#   shared.locking's.  Both copies are committed in this repo, so the
#   comparison depends only on the commit under verify.
#
# Cross-repo (reify's scripts/lock-charter-guard.sh and tracked corpus): NOT
#   in this suite.  A sibling checkout's working state must not sit on the
#   merge gate (task 11623), so `scripts/check_lock_charter_drift.py` compares
#   them nightly via lock-charter-drift-check.timer and files one heal task per
#   drift.  OPERATIONS.md section 9a.
# ---------------------------------------------------------------------------

# The canonical α/γ vector — update this list AND FILE_EXTENSIONS together.
# Widened 36 -> 58 by the git-ls-files sweep 2026-07-28
# (reify #5726 / dark_factory #3117); generated from α's --list-extensions.
# Widened 58 -> 59 on 2026-08-06: `csv`, flagged by the corpus->allowlist guard
# below once plans/evidence/scheduler-scoring-2026-08-06/*.csv landed on main.
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
    'template', 'timer', 'tombstones', 'toml', 'ts', 'tsv', 'tsx', 'txt', 'typed',
    'yaml', 'yml',
]

def test_extension_drift_guard():
    """Tier-1 (same-file): sorted(FILE_EXTENSIONS) must match _CANONICAL_EXTENSIONS.

    This is a same-file consistency check — FILE_EXTENSIONS and _CANONICAL_EXTENSIONS
    must be updated together.  For α/γ drift see
    test_gamma_vectors_match_shared_locking.
    """
    assert sorted(FILE_EXTENSIONS) == _CANONICAL_EXTENSIONS


def test_gamma_vectors_match_shared_locking():
    """α == γ: both of this module's vectors must equal shared.locking's.

    Each Tier-1 guard compares one copy only with a literal in its own test
    file, so an edit to one Python copy plus its own literal would pass both.
    Until task 11623 the cross-repo guards gave this transitively (each copy
    equal to reify's emitter); this pins it directly, without leaving the repo.
    """
    assert FILE_EXTENSIONS == shared_locking.FILE_EXTENSIONS
    assert EXTENSIONLESS_FILENAMES == shared_locking.EXTENSIONLESS_FILENAMES


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
    # Widened 59 -> 60 for reify #6758 (spec-conformance α): the first tracked
    # reify file with a `tombstones` extension; filename is PRD-normative
    # (reify docs/prds/v0_6/spec-conformance-suite.md D3).
    'docs/reify-language-spec.tombstones',  # reify
    # reify #9058 (α-first): the first tracked reify `.tsv`, landed by reify #8662.
    'tests/infra/affected-crates-direct-sets.tsv',  # reify
]


@pytest.mark.parametrize('path', _ACCEPT_PATHS)
def test_is_file_path_accepts_files(path):
    """File-level paths must be classified as files (True)."""
    assert is_file_path(path) is True


# ---------------------------------------------------------------------------
# Conservative-reject edge cases matching α's case-sensitive bash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('path', ['f.PY'])
def test_is_file_path_conservative_rejects(path):
    """Upper-case extensions are rejected — the allowlist is lowercase (C-P3).

    '.gitignore' used to be pinned here too, under the generalisation "dotfiles
    without extension are rejected".  That generalisation was never the rule and
    is now visibly false: the predicate looks up the substring after the LAST
    dot, and for '.gitignore' that substring is 'gitignore' — allowlisted since
    the 2026-07-28 sweep (#5726 / #3117), so it correctly classifies as a file
    and is covered in _ACCEPT_PATHS.  Verified against α: `classify .gitignore`
    returns ACCEPT, so the flip is convergence with the source of truth rather
    than a regression, and special-casing it back to False would reintroduce the
    exact α/γ divergence this task closes.

    'f.PY' stays: case sensitivity is genuinely unchanged.
    """
    assert is_file_path(path) is False


# ---------------------------------------------------------------------------
# Rejected alternative: a blanket "leading-dot segment => FILE" predicate rule
# ---------------------------------------------------------------------------

# Real directories whose final segment carries a leading dot and no further dot.
_DOTTED_DIRECTORY_PATHS = [
    '.worktrees',
    '.task',
    '.claude',
    '.cargo',
    '.taskmaster',
]


@pytest.mark.parametrize('path', _DOTTED_DIRECTORY_PATHS)
def test_leading_dot_directories_stay_directories(path):
    """Seven of the 22 additions are dotfiles — a dotfile RULE was rejected.

    .gitignore .gitkeep .envrc .npmrc .gitattributes .gitmodules .python-version
    could all be covered by one predicate line ("a segment starting with '.' is a
    file") instead of seven list entries.  That simplification was considered and
    REJECTED: the paths below are real DIRECTORIES which correctly reject today
    (verified against α — `lock-charter-guard.sh classify .worktrees` returns
    REJECT), and the rule would flip every one to FILE, letting a task declare
    '.worktrees' — the orchestrator's entire worktree pool — as its lock charter.
    That is precisely the over-wide-charter failure this guard exists to prevent.

    Keeping the allowlist ENUMERATED is what makes an unknown dotted segment
    default to directory.  These assertions pass both before and after the
    widening: they are regression pins for a rejected design, not a behaviour
    change, and they live here so the reasoning travels with the code.
    """
    assert is_file_path(path) is False, (
        f'{path!r} is a real directory and must NOT classify as a file; a blanket '
        f'leading-dot=>file rule was rejected for exactly this reason'
    )


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
# shared/src/shared/locking.py AND shared/tests/test_locking.py.
#
# The corpora below are VERBATIM DUPLICATES of the α copies in
# shared/tests/test_locking.py, for the same reason _CANONICAL_EXTENSIONS is:
# the two predicate copies are NOT linked by a re-export, so each must be
# pinned independently, and byte-identical corpora let a reader diff the two
# suites and see at a glance that they agree.
#
# EVIDENCE — `git ls-files -s` over BOTH repos, re-measured 2026-08-09 (and
# identical to the 2026-07-31 measurement): every tracked path whose final
# segment contains no '.'.  dark-factory contributes Dockerfile LICENSE
# pre-commit pre-merge-commit project-checks; reify adds cargo
# cargo-audit-orphans reference-transaction, and sccache
# (scripts/canon-bin/guard/sccache, reify task 10232, measured on reify main
# 2026-11-23).  Union = the 9 below.
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
    'sccache',
]

# One REAL tracked path per canonical extension-less name — dark-factory unless
# the comment says reify (same convention as _ACCEPT_PATHS above).  Verified
# tracked with `git ls-files --error-unmatch` on 2026-08-09 (the sccache entry
# on 2026-11-23).  Enforced complete
# by test_extensionless_accept_corpus_covers_every_canonical_name.
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
    'scripts/canon-bin/guard/sccache',
    # reify
    'scripts/cargo-audit-orphans',
]


def test_extensionless_drift_guard():
    """Tier-1 (same-file): sorted(EXTENSIONLESS_FILENAMES) == _CANONICAL_EXTENSIONLESS.

    Same shape and same reason as test_extension_drift_guard above; its α/γ
    counterpart is test_gamma_vectors_match_shared_locking.
    """
    assert sorted(EXTENSIONLESS_FILENAMES) == _CANONICAL_EXTENSIONLESS


def test_extensionless_filenames_is_frozenset():
    assert isinstance(EXTENSIONLESS_FILENAMES, frozenset)


def test_extensionless_members_are_bare_final_segments():
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


@pytest.mark.parametrize('path', _EXTENSIONLESS_ACCEPT_PATHS)
def test_extensionless_accept_corpus_paths_are_files(path: str):
    """Extension-less tracked FILES classify as files (dark_factory #3248).

    The hazard these pin, stated concretely: two tasks editing
    ``hooks/project-checks`` concurrently could both hold NO lock on it, because
    a charter that strips to empty degrades to a per-task ``task-<id>`` synthetic
    lock that conflicts with nothing.
    """
    assert is_file_path(path) is True, (
        f'{path!r} is a real tracked FILE whose final segment carries no '
        f'extension; it must classify as a file via EXTENSIONLESS_FILENAMES'
    )


def test_extensionless_accept_corpus_covers_every_canonical_name():
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


# ---------------------------------------------------------------------------
# Corpus completeness — every allowlisted extension must be exercised
# ---------------------------------------------------------------------------


def test_accept_corpus_covers_every_canonical_extension():
    """_ACCEPT_PATHS must exercise every entry in _CANONICAL_EXTENSIONS.

    Direction: **allowlist → corpus**.  The corpus comment above claims "one path
    per canonical extension"; this turns that claim into an enforced invariant.
    Without it an extension can be added to the allowlist and pinned in the drift
    guards while never being run through an actual classification assertion.

    The opposite direction — **corpus → allowlist**, i.e. a real tracked file
    whose extension is MISSING from the allowlist — is the direction the
    originating incident failed in, and is guarded separately by
    test_every_tracked_extension_is_allowlisted below.  The two are complementary
    and neither implies the other.

    Measured precondition at the time this was added: the pre-existing 39
    _ACCEPT_PATHS covered exactly all 36 then-canonical extensions — no gaps and
    no extras — so this assertion carries zero pre-existing debt.

    Coverage is derived from ``is_file_path`` itself rather than from a
    hand-written re-implementation of its segment parsing (there are already
    three real implementations: this module, ``shared.locking``, and α's bash).
    A path counts for ``ext`` only if the predicate under test ACCEPTS it and it
    ends in ``.<ext>``; because every canonical extension is dot-free (asserted
    first), ``path.endswith('.' + ext)`` holds iff the substring after the path's
    LAST dot equals ``ext`` — exactly the lookup ``is_file_path`` performs.  So
    a future change to segment parsing cannot leave this test silently measuring
    a different extension than the predicate does.
    """
    dotted = sorted(e for e in _CANONICAL_EXTENSIONS if '.' in e)
    assert not dotted, (
        f'canonical extensions must be dot-free for the endswith() equivalence '
        f'this test relies on; got {dotted!r}'
    )

    accepted = [p for p in _ACCEPT_PATHS if is_file_path(p)]
    uncovered = sorted(
        ext
        for ext in _CANONICAL_EXTENSIONS
        if not any(p.endswith(f'.{ext}') for p in accepted)
    )
    assert not uncovered, (
        f'{len(uncovered)} allowlisted extension(s) are pinned in '
        f'_CANONICAL_EXTENSIONS but never classified by any _ACCEPT_PATHS entry: '
        f'{uncovered!r}\n'
        f'Add one real representative path per extension to _ACCEPT_PATHS.'
    )


# ---------------------------------------------------------------------------
# Corpus → allowlist guard — the direction the originating incident failed in
# ---------------------------------------------------------------------------

# Repo root, resolved from this test file's location.  __file__ lives at
# <repo>/fused-memory/tests/, so parents[2] is the dark-factory checkout root
# — in a worktree too.  These sweeps read THIS checkout only; reify's tracked
# corpus is swept off the gate by scripts/check_lock_charter_drift.py.
_DF_REPO_ROOT = Path(__file__).parents[2]


def _tracked_entries(repo_root: Path) -> tuple[list[str], list[str]] | None:
    """``(non_gitlink_paths, gitlink_paths)`` for *repo_root*, or None if not a checkout.

    ONE parse of the tracked corpus, split by git's own mode column, so the two
    sweeps below cannot drift on what "the corpus" means.  This replaced a pair
    of near-identical helpers (``_tracked_paths`` / ``_tracked_file_paths``)
    that differed only by ``-s`` and the mode filter; keeping both invited their
    call sites to disagree about whether gitlinks were in scope.

    ``git ls-files -s`` (not plain ``ls-files``) is what makes the mode column
    available.  Mode ``160000`` means a submodule gitlink; those entries are
    returned SEPARATELY rather than discarded, because the sweeps need them to
    prove the exclusion is doing real work.  See the GITLINK EXCLUSION note on
    _CANONICAL_EXTENSIONLESS for why the filter is load-bearing.
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


def test_every_tracked_extension_is_allowlisted():
    """No REAL tracked file may classify as a directory because of a missing ext.

    Direction: **corpus → allowlist**.  This is the failure mode that produced
    the originating 2026-07-28 incident, and the one the allowlist→corpus test
    above structurally cannot catch: ``tests/infra/run-all-classification.manifest``
    was a genuinely tracked file whose extension (``manifest``) was simply absent
    from the allowlist, so ``is_file_path`` called it a directory and
    ``update_task`` rejected the declaration with a LockCharterViolation.  Adding
    22 entries plus a corpus-completeness check does nothing to detect the 23rd
    omission — the next time someone commits a ``.proto``, ``.sql`` or ``.rules``
    file the same silent misclassification recurs and again only surfaces as an
    incident.  This test makes the allowlist self-auditing against the actual
    tracked corpus instead.

    Only THIS checkout is swept here, and that is not the whole property.  The
    allowlist is a single α/γ-shared vector governing declarations for both
    projects, and the two corpora barely overlap: measured 2026-07-28, 19
    extensions are tracked in both, 12 only in dark-factory and 17 only in
    reify.  ``manifest`` is in the reify-only group, so this sweep alone would
    NOT have caught the very incident that motivated it.  Reify's corpus is
    swept by ``scripts/check_lock_charter_drift.py::check_corpus``, nightly and
    off the merge gate, because a reify commit must never red a dark-factory
    merge (task 11623; OPERATIONS.md section 9a).

    Measured precondition when added: dark-factory 2124 tracked paths and reify
    3778, with ZERO non-allowlisted extensions in either — so this carries no
    pre-existing debt, exactly the property the allowlist→corpus test claims for
    its own direction.  (Those counts were taken over the gitlink-INCLUSIVE
    corpus this sweep formerly used; it now shares _tracked_file_paths with the
    extension-less sweep, which is 2 paths smaller in dark-factory.  The result
    is unchanged by construction: a gitlink mount point carries no extension, so
    the dotted-segment filter below could never have reached one.)

    The gate is ``is_file_path`` itself, so a flagged path is by construction one
    the predicate really does misclassify.  A dotted final segment that the
    predicate rejects means either the extension is missing from the allowlist
    (add it — and to α's ``scripts/lock-charter-guard.sh`` and
    ``shared.locking``) or its case differs (the allowlist is deliberately
    lowercase-only; a tracked ``FOO.PNG`` is a naming problem, not an allowlist
    gap).  Extension-less tracked files (``LICENSE``, ``Dockerfile``,
    ``hooks/pre-commit``) are out of scope HERE — they have no extension to
    allowlist, so this sweep skips them — but they are NOT out of scope
    generally: as of dark_factory #3248 they are recognised by name via
    ``EXTENSIONLESS_FILENAMES`` and are swept by the sibling
    ``test_every_tracked_extensionless_file_is_allowlisted`` below.  This
    paragraph previously claimed they were "correctly out of scope … handled by
    the escape hatch named in ``lock_charter_error``"; that was the bug, not the
    design — the escape hatch is ``files=[]``, i.e. declaring NO charter at all,
    which is exactly the silent under-locking #3248 fixes.
    """
    paths = _tracked_file_paths(_DF_REPO_ROOT)
    if paths is None:
        pytest.skip(f'{_DF_REPO_ROOT} is not a git checkout (git ls-files failed)')
    assert paths, f'git ls-files returned no tracked paths for {_DF_REPO_ROOT}'

    # ext -> first tracked path exhibiting it, for an actionable failure message.
    unknown: dict[str, str] = {}
    for path in paths:
        if is_file_path(path):
            continue
        seg = path.rsplit('/', 1)[-1]
        if '.' in seg:
            unknown.setdefault(seg.rsplit('.', 1)[1], path)

    assert not unknown, (
        f'{len(unknown)} extension(s) on real tracked dark-factory files are absent '
        f'from FILE_EXTENSIONS, so those files classify as DIRECTORIES and any '
        f'task declaring one in metadata.files is rejected with a '
        f'LockCharterViolation:\n'
        + '\n'.join(f'  .{ext} — e.g. {path}' for ext, path in sorted(unknown.items()))
        + '\nAdd each to FILE_EXTENSIONS here, to shared/src/shared/locking.py, '
        'and to reify scripts/lock-charter-guard.sh _EXTS (all three must agree), '
        'plus _CANONICAL_EXTENSIONS and _ACCEPT_PATHS in both test copies.'
    )


def test_no_canonical_extensionless_name_is_also_a_directory():
    """No member may appear as a NON-FINAL component of any tracked path.

    This is what bounds the risk of matching on a bare basename.  A member
    that were ALSO a directory name somewhere in the tree would make that
    directory declarable as a lock charter — the exact over-wide-charter
    failure the guard exists to prevent.  Measured zero collisions across
    both repos on 2026-08-09; this pins that property permanently rather
    than leaving it as a one-time observation.

    SCOPE, stated plainly so the guard is not over-read: it bounds the risk for
    THIS checkout only (reify's tree is swept off the gate by
    ``scripts/check_lock_charter_drift.py::check_extensionless``), while the
    predicate governs lock charters for every project the orchestrator targets.  Several members are generic
    basenames (`cargo`, `Dockerfile`, `LICENSE`, `pre-commit`), so a third
    project with a real `tools/cargo/` DIRECTORY would have it classified as a
    file and retained as a subtree-wide prefix lock, unswept by this test.  That
    is why admitting a generic basename is a cross-project commitment — see the
    CROSS-PROJECT SCOPE note on the EXTENSIONLESS_FILENAMES vector.
    """
    paths = _tracked_file_paths(_DF_REPO_ROOT)
    if paths is None:
        pytest.skip(f'{_DF_REPO_ROOT} is not a git checkout (git ls-files failed)')
    assert paths, f'git ls-files returned no tracked paths for {_DF_REPO_ROOT}'

    collisions: dict[str, str] = {}
    for path in paths:
        for component in path.split('/')[:-1]:
            if component in EXTENSIONLESS_FILENAMES:
                collisions.setdefault(component, path)

    assert not collisions, (
        f'{len(collisions)} EXTENSIONLESS_FILENAMES member(s) are also '
        f'DIRECTORY names in dark-factory, so declaring that directory would now '
        f'be accepted as a file-level lock charter:\n'
        + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(collisions.items()))
        + '\nRemove the name from the allowlist, or rename the directory.'
    )


def test_every_tracked_extensionless_file_is_allowlisted():
    """Corpus->allowlist self-audit: a newly tracked extensionless file must go RED.

    The extension-less sibling of test_every_tracked_extension_is_allowlisted,
    and the same direction — **corpus → allowlist**.  Turns "add 8 names" into a
    self-auditing invariant: when someone adds a new extension-less tracked file
    to this repo, this test names it here rather than letting it surface later
    as a LockCharterViolation at submit time — or, worse, as a silent under-lock.
    Reify's extension-less corpus is swept off the gate, by
    ``scripts/check_lock_charter_drift.py::check_extensionless``.

    This sweep IS duplicated between the α and γ suites, unlike the
    extension-corpus vectors which are pinned twice for drift detection.  The
    difference is the failure mode: the extension vector is fully pinned by the
    two drift guards, whereas this one gates a REPO-level property — a
    newly-added tracked file changes the correct answer without any test vector
    changing, so whichever suite runs must be able to see it.

    Gitlink mount points are excluded by _tracked_file_paths and must NOT be
    admitted; that exclusion is asserted below.  The gitlink set is DERIVED from
    git's own mode column rather than hardcoded, which matters in both
    directions: hardcoding ``('graphiti', 'mem0')`` made the assertion vacuously
    true for a repo that vendors no submodules, and would have gone red for no
    real defect if dark-factory ever de-vendored them.  A corpus with no
    submodules skips the sub-assertion explicitly instead of passing it
    silently.
    """
    entries = _tracked_entries(_DF_REPO_ROOT)
    if entries is None:
        pytest.skip(f'{_DF_REPO_ROOT} is not a git checkout (git ls-files failed)')
    paths, gitlinks = entries
    assert paths, f'git ls-files returned no tracked paths for {_DF_REPO_ROOT}'

    # The gitlink filter must be doing real work, not passing vacuously.
    if gitlinks:
        overlap = sorted(set(gitlinks) & set(paths))
        assert not overlap, (
            f'{overlap!r} appear as BOTH mode-160000 gitlinks and ordinary '
            f'tracked paths in dark-factory; _tracked_entries must partition the '
            f'corpus, and admitting a submodule mount point would let a task '
            f'declare an entire vendored submodule as its lock charter'
        )
        for gitlink in gitlinks:
            assert not is_file_path(gitlink), (
                f'{gitlink!r} is a submodule mount point (mode 160000) and must '
                f'stay a DIRECTORY — see the GITLINK EXCLUSION note on the '
                f'corpus.  Admitting it would make an entire vendored submodule '
                f'declarable as a lock charter, strictly worse than the bug '
                f'#3248 fixed.'
            )

    unknown: dict[str, str] = {}
    for path in paths:
        seg = path.rsplit('/', 1)[-1]
        if '.' in seg:
            continue
        if not is_file_path(path):
            unknown.setdefault(seg, path)

    assert not unknown, (
        f'{len(unknown)} extension-less tracked dark-factory file(s) classify as '
        f'DIRECTORIES, so any task declaring one in metadata.files is either '
        f'rejected with a LockCharterViolation or — silently — stripped to an '
        f'empty charter and given a task-<id> synthetic lock that conflicts '
        f'with nothing:\n'
        + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(unknown.items()))
        + '\nAdd each name to EXTENSIONLESS_FILENAMES in '
        'shared/src/shared/locking.py AND '
        'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
        'and to reify scripts/lock-charter-guard.sh _EXTLESS (all three must '
        'agree; the nightly scripts/check_lock_charter_drift.py reports reify), '
        'plus _CANONICAL_EXTENSIONLESS and _EXTENSIONLESS_ACCEPT_PATHS in '
        'both test copies.'
    )


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

    def test_extensionless_tracked_file_is_not_a_directory_lock(self):
        """Face-1 reproduction: the exact list `tools.py::submit_task` gates on.

        `submit_task` calls `directory_locks(extract_files(metadata))` and, if
        the result is non-empty, rejects the whole submission via
        `lock_charter_error`.  A task declaring `hooks/project-checks` — a real
        tracked FILE — was therefore rejected at submit time with, verbatim:

            error_type=LockCharterViolation
            directory_paths=["hooks/project-checks"]

        so an EMPTY result here IS the end-to-end fix, exercised through the
        same code path rather than through a `submit_task` round-trip (the
        server module is out of this task's charter and needs no diff — it
        changes behaviour purely by importing this predicate).

        The second path deliberately need not exist on disk: C-P3 keeps the
        predicate pure-string with no stat, already pinned by the
        'no/such/path/ghost.rs' accept-corpus entry.
        """
        assert directory_locks([
            'hooks/project-checks',
            'orchestrator/tests/test_project_checks_pyright_scope.py',
        ]) == []


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

    def test_unparseable_json_warns_and_returns_empty(self, caplog):
        """RED (task 2166 step-9): the str/json branch currently discards silently."""
        with caplog.at_level(logging.WARNING):
            result = extract_files('not valid json {{{')
        assert result == []
        assert len(census_lines(caplog, _SOURCE)) == 1

    @pytest.mark.parametrize('metadata', [42, ['a/b.rs']], ids=repr)
    def test_non_str_unreadable_metadata_warns_and_returns_empty(self, metadata, caplog):
        with caplog.at_level(logging.WARNING):
            result = extract_files(metadata)
        assert result == []
        assert len(census_lines(caplog, _SOURCE)) == 1

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

    def test_hint_names_the_extensionless_allowlist(self):
        """The hint must be followable for an unrecognised extension-less name.

        The generic advice — "Replace each directory with the specific file
        paths it contains" — is UNFOLLOWABLE for `hooks/project-checks`: that IS
        the specific file, and it contains nothing.  The remedy is to add the
        name to EXTENSIONLESS_FILENAMES, so the hint has to say so.

        The two classification assertions below are load-bearing, not scene
        setting.  `lock_charter_error`'s hint is a CONSTANT string that does not
        vary with its argument, so the substring check alone proved nothing
        about extension-less classification — `lock_charter_error(['crates/'])`,
        an ordinary directory, satisfied it identically.  Pinning the
        precondition makes the test depend on the behaviour actually under
        review: an UNRECOGNISED extension-less name still reaches this error
        path, while a RECOGNISED one no longer does — which is precisely what
        makes "add the name to EXTENSIONLESS_FILENAMES" the real remedy rather
        than one more string in a fixed message.
        """
        # Unrecognised extension-less name → still classified a directory, so it
        # genuinely reaches lock_charter_error and needs a followable hint.
        assert directory_locks(['hooks/some-new-hook']) == ['hooks/some-new-hook']
        # Recognised one → not a directory, so it never reaches the error at all.
        assert directory_locks(['hooks/pre-commit']) == []

        result = lock_charter_error(['hooks/some-new-hook'])
        assert 'EXTENSIONLESS_FILENAMES' in result['hint']

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

    @pytest.mark.parametrize(
        'path',
        ['tasks.db', '.taskmaster/tasks/tasks.db', 'data/x.sqlite', '.env'],
    )
    def test_unrecognised_extension_rejection_names_deterministic_routing(self, path):
        assert directory_locks([path]) == [path]

        result = lock_charter_error([path])

        assert result['unrecognised_extension_paths'] == [path]
        assert "task_kind='deterministic'" in result['error']
        assert "task_kind='deterministic'" in result['hint']
        assert path in result['hint']
        assert 'FILE_EXTENSIONS' in result['hint']

    def test_unrecognised_extension_hint_carries_the_gitignored_guard_routing(self):
        gitignored_hint = gitignored_deliverable_reject(
            GitignoredDeliverableFinding(ignored_paths=('tasks.db',))
        )['hint']

        assert gitignored_hint in lock_charter_error(['tasks.db'])['hint']

    @pytest.mark.parametrize(
        'path',
        ['crates/', 'orchestrator/', 'src', 'crates/reify-eval/src', 'hooks/some-new-hook', 'tasks.db/'],
    )
    def test_genuine_directory_rejection_omits_deterministic_routing(self, path):
        assert directory_locks([path]) == [path]

        result = lock_charter_error([path])

        assert result['unrecognised_extension_paths'] == []
        assert 'deterministic' not in result['error']
        assert 'deterministic' not in result['hint']

    @pytest.mark.parametrize('path', ['README.MD', 'config/app.YAML'])
    def test_case_variant_of_an_allowlisted_extension_gets_only_the_lowercase_advice(self, path):
        assert directory_locks([path]) == [path]

        result = lock_charter_error([path])

        assert result['unrecognised_extension_paths'] == []
        assert 'deterministic' not in result['error'] + result['hint']
        assert 'lowercase' in result['hint']

    def test_mixed_rejection_lists_only_the_unrecognised_extension_paths(self):
        result = lock_charter_error(['crates/', 'tasks.db'])

        assert result['directory_paths'] == ['crates/', 'tasks.db']
        assert result['unrecognised_extension_paths'] == ['tasks.db']
        assert "task_kind='deterministic'" in result['hint']
