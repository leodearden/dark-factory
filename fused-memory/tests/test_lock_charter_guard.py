"""Unit tests for fused_memory.middleware.lock_charter_guard.

Step 1 (RED → step-2 GREEN): predicate + drift guard
Step 3 (RED → step-4 GREEN): list-gate helpers
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from fused_memory.middleware.lock_charter_guard import (
    EXTENSIONLESS_FILENAMES,
    FILE_EXTENSIONS,
    directory_locks,
    extract_files,
    is_file_path,
    lock_charter_error,
)

_LCG_LOGGER = 'fused_memory.middleware.lock_charter_guard'

# ---------------------------------------------------------------------------
# Drift guard — two tiers:
#
# Tier 1 (test_extension_drift_guard): same-file consistency check.
#   Asserts sorted(FILE_EXTENSIONS) == _CANONICAL_EXTENSIONS, where both are
#   defined in this file.  Catching that FILE_EXTENSIONS and _CANONICAL_EXTENSIONS
#   stay in sync with each other.  Update BOTH lists together when the allowlist
#   changes.  This does NOT catch cross-language α/γ drift on its own.
#
# Tier 2 (test_extension_drift_guard_vs_reify_script): cross-source guard.
#   Invokes the real reify/scripts/lock-charter-guard.sh --list-extensions and
#   compares its output to sorted(FILE_EXTENSIONS).  Skipped when the script
#   is not present (e.g. in a standalone fused-memory checkout).  Run this in
#   any environment that has both repos checked out side-by-side.
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
    'template', 'timer', 'toml', 'ts', 'tsx', 'txt', 'typed',
    'yaml', 'yml',
]

# Path to the reify script (siblings repos under the same src/ directory).
# Resolved from the test file location; works in the dark-factory worktree
# layout: <src>/dark-factory/.worktrees/<n>/fused-memory/tests/ → <src>/reify/
_REIFY_GUARD_SCRIPT = Path(__file__).parents[5] / 'reify' / 'scripts' / 'lock-charter-guard.sh'


def _reify_guard_vector(flag: str) -> list[str]:
    """Return the sorted vector emitted by the reify guard script's *flag*.

    LOUD on failure, by design, and the distinction is deliberate: the callers
    are guarded by a ``skipif`` on the script's ABSENCE — a standalone checkout
    legitimately has no reify sibling — but a script that EXISTS and then fails
    is real signal and is raised, never skipped.

    Measured 2026-08-09 on the live script: an unrecognised flag prints usage to
    stderr and exits 2.  So the previous skip-on-any-non-zero would have retired
    both Tier-2 guards the moment reify renamed or dropped an emitter, leaving
    them green forever while enforcing nothing — the precise silent degradation
    that the "three-way α/γ/bash agreement is ENFORCED rather than merely
    asserted in prose" claim in both module docstrings rules out.
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
        f'This is a HARD failure and not a skip on purpose: silently retiring a '
        f'cross-source guard would let the three copies of the predicate diverge '
        f'with every test in the tree still green.\n'
        f'  stdout: {result.stdout[:400]!r}\n'
        f'  stderr: {result.stderr[:400]!r}'
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def test_extension_drift_guard():
    """Tier-1 (same-file): sorted(FILE_EXTENSIONS) must match _CANONICAL_EXTENSIONS.

    This is a same-file consistency check — FILE_EXTENSIONS and _CANONICAL_EXTENSIONS
    must be updated together.  For cross-source α/γ drift detection see
    test_extension_drift_guard_vs_reify_script.
    """
    assert sorted(FILE_EXTENSIONS) == _CANONICAL_EXTENSIONS


@pytest.mark.skipif(
    not _REIFY_GUARD_SCRIPT.is_file(),
    reason='reify script not present (standalone checkout; cross-repo drift check skipped)',
)
def test_extension_drift_guard_vs_reify_script():
    """Tier-2 (cross-source): sorted(FILE_EXTENSIONS) must match reify --list-extensions.

    Invokes the real scripts/lock-charter-guard.sh --list-extensions and
    compares its output to sorted(FILE_EXTENSIONS), catching any α/γ divergence
    that the same-file Tier-1 guard would miss.

    A present-but-failing script is a hard failure, not a skip — see
    _reify_guard_vector.
    """
    script_exts = _reify_guard_vector('--list-extensions')
    assert script_exts == sorted(FILE_EXTENSIONS), (
        f'α/γ drift detected!\n'
        f'  reify --list-extensions : {script_exts!r}\n'
        f'  γ FILE_EXTENSIONS       : {sorted(FILE_EXTENSIONS)!r}\n'
        f'Update FILE_EXTENSIONS (and _CANONICAL_EXTENSIONS) to match reify.'
    )


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
    'scripts/cargo-audit-orphans',
]


def test_extensionless_drift_guard():
    """Tier-1 (same-file): sorted(EXTENSIONLESS_FILENAMES) == _CANONICAL_EXTENSIONLESS.

    Same shape and same reason as test_extension_drift_guard above; the Tier-2
    cross-source counterpart is test_extensionless_drift_guard_vs_reify_script.
    """
    assert sorted(EXTENSIONLESS_FILENAMES) == _CANONICAL_EXTENSIONLESS


@pytest.mark.skipif(
    not _REIFY_GUARD_SCRIPT.is_file(),
    reason='reify script not present (standalone checkout; cross-repo drift check skipped)',
)
def test_extensionless_drift_guard_vs_reify_script():
    """Tier-2 (cross-source): EXTENSIONLESS_FILENAMES must match --list-extensionless.

    The extension-less sibling of test_extension_drift_guard_vs_reify_script,
    and the reason the extension-half guard cannot stand in for it: that one
    compares ``--list-extensions``, a vector of EXTENSION strings, so it is
    structurally blind to this frozenset.  Without this test the three copies
    could silently disagree on the 8 names while every other guard stayed green.

    #3248 was PLANNED on the premise that this vector was a dark_factory-only
    lead and that reify's bash ``_is_file_path`` still conservative-rejected all
    8 names, with a reify follow-up to be filed.  That premise was refuted by
    measurement on 2026-08-09: reify #5890 had already landed the identical
    vector and added ``--list-extensionless`` specifically so the other side
    could pin against it.  ``classify hooks/project-checks`` returns ACCEPT.  So
    no follow-up is owed, and the seam is CONVERGED — this test is what keeps it
    that way.

    A present-but-failing script is a hard failure, not a skip — see
    _reify_guard_vector.  That matters more here than for the extension half:
    ``--list-extensionless`` is the NEWER emitter (reify #5890), so it is the
    likelier of the two to be renamed or dropped upstream.
    """
    script_names = _reify_guard_vector('--list-extensionless')
    assert script_names == sorted(EXTENSIONLESS_FILENAMES), (
        f'α/γ/bash drift detected on the extension-less vector!\n'
        f'  reify --list-extensionless : {script_names!r}\n'
        f'  γ EXTENSIONLESS_FILENAMES  : {sorted(EXTENSIONLESS_FILENAMES)!r}\n'
        f'Update EXTENSIONLESS_FILENAMES here, in shared/src/shared/locking.py, '
        f"and reify's own scripts/lock-charter-guard.sh _EXTLESS, plus "
        f'_CANONICAL_EXTENSIONLESS in both test copies — all five places.'
    )


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

# Repo roots, resolved from this test file's location.  __file__ lives at
# <repo>/fused-memory/tests/, so parents[2] is the dark-factory checkout root
# (works in a worktree too); reify is a sibling under the same src/ directory,
# same resolution _REIFY_GUARD_SCRIPT uses.
_DF_REPO_ROOT = Path(__file__).parents[2]
_REIFY_REPO_ROOT = Path(__file__).parents[5] / 'reify'


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


@pytest.mark.parametrize(
    ('repo', 'repo_root'),
    [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
)
def test_every_tracked_extension_is_allowlisted(repo: str, repo_root: Path):
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

    BOTH repos are swept, and that is load-bearing rather than thoroughness for
    its own sake — do not "simplify" this to the local repo.  The allowlist is a
    single α/γ-shared vector governing declarations for both projects, and the
    two corpora barely overlap: measured 2026-07-28, 19 extensions are tracked in
    both, 12 only in dark-factory (diff, example, example-systemd-config,
    gitattributes, gitmodules, jsonl, jsx, log, python-version, svg, template,
    typed) and 17 only in reify (c, conf, cpp, gcode, gitkeep, golden, grammar,
    h, icns, ico, jq, manifest, npmrc, png, ri, ts, tsx).  ``manifest`` is in that
    reify-only group — zero ``.manifest`` files are tracked in dark-factory — so a
    dark-factory-only sweep would NOT have caught the very incident that motivated
    this task.  Verified by mutation: removing ``manifest`` from FILE_EXTENSIONS
    fails the reify parametrization (naming
    ``tests/infra/harness-layout-baseline.manifest``) and passes the dark-factory
    one.  Each repo is skipped independently when that checkout is absent,
    mirroring the Tier-2 skipif above.

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
    if not repo_root.is_dir():
        pytest.skip(f'{repo} checkout not present at {repo_root}')
    paths = _tracked_file_paths(repo_root)
    if paths is None:
        pytest.skip(f'{repo_root} is not a git checkout (git ls-files failed)')
    assert paths, f'git ls-files returned no tracked paths for {repo_root}'

    # ext -> first tracked path exhibiting it, for an actionable failure message.
    unknown: dict[str, str] = {}
    for path in paths:
        if is_file_path(path):
            continue
        seg = path.rsplit('/', 1)[-1]
        if '.' in seg:
            unknown.setdefault(seg.rsplit('.', 1)[1], path)

    assert not unknown, (
        f'{len(unknown)} extension(s) on real tracked {repo} files are absent '
        f'from FILE_EXTENSIONS, so those files classify as DIRECTORIES and any '
        f'task declaring one in metadata.files is rejected with a '
        f'LockCharterViolation:\n'
        + '\n'.join(f'  .{ext} — e.g. {path}' for ext, path in sorted(unknown.items()))
        + '\nAdd each to FILE_EXTENSIONS here, to shared/src/shared/locking.py, '
        'and to reify scripts/lock-charter-guard.sh _EXTS (all three must agree), '
        'plus _CANONICAL_EXTENSIONS and _ACCEPT_PATHS in both test copies.'
    )


@pytest.mark.parametrize(
    ('repo', 'repo_root'),
    [('dark-factory', _DF_REPO_ROOT), ('reify', _REIFY_REPO_ROOT)],
)
def test_no_canonical_extensionless_name_is_also_a_directory(repo: str, repo_root: Path):
    """No member may appear as a NON-FINAL component of any tracked path.

    This is what bounds the risk of matching on a bare basename.  A member
    that were ALSO a directory name somewhere in the tree would make that
    directory declarable as a lock charter — the exact over-wide-charter
    failure the guard exists to prevent.  Measured zero collisions across
    both repos on 2026-08-09; this pins that property permanently rather
    than leaving it as a one-time observation.

    SCOPE, stated plainly so the guard is not over-read: it bounds the risk for
    these TWO checkouts only, while the predicate governs lock charters for
    every project the orchestrator targets.  Several members are generic
    basenames (`cargo`, `Dockerfile`, `LICENSE`, `pre-commit`), so a third
    project with a real `tools/cargo/` DIRECTORY would have it classified as a
    file and retained as a subtree-wide prefix lock, unswept by this test.  That
    is why admitting a generic basename is a cross-project commitment — see the
    CROSS-PROJECT SCOPE note on the EXTENSIONLESS_FILENAMES vector.
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
def test_every_tracked_extensionless_file_is_allowlisted(repo: str, repo_root: Path):
    """Corpus->allowlist self-audit: the 9th extensionless file must go RED.

    The extension-less sibling of test_every_tracked_extension_is_allowlisted,
    and the same direction — **corpus → allowlist**.  Turns "add 8 names" into a
    self-auditing invariant: when someone adds a new extension-less tracked file
    to either repo, this test names it here rather than letting it surface later
    as a LockCharterViolation at submit time — or, worse, as a silent under-lock.

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
    true for the reify parametrization (reify vendors no submodules), and would
    have gone red for no real defect if dark-factory ever de-vendored them.
    A repo with no submodules now skips the sub-assertion explicitly instead of
    passing it silently — so read this guard as non-vacuous exactly where
    gitlinks actually exist, which today is the dark-factory parametrization.
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
        'agree — omitting reify reds test_extensionless_drift_guard_vs_reify_script), '
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
        with caplog.at_level(logging.WARNING, logger=_LCG_LOGGER):
            result = extract_files('not valid json {{{')
        assert result == []
        warns = [
            r for r in caplog.records
            if r.name == _LCG_LOGGER and r.levelno >= logging.WARNING
        ]
        assert any('task_metadata.schema_warning' in r.message for r in warns), (
            f'expected a task_metadata.schema_warning WARNING; got '
            f'{[r.message for r in warns]!r}'
        )

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
