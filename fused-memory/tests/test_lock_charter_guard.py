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
    CODE_EXTENSIONS,
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
#   Asserts sorted(CODE_EXTENSIONS) == _CANONICAL_EXTENSIONS, where both are
#   defined in this file.  Catching that CODE_EXTENSIONS and _CANONICAL_EXTENSIONS
#   stay in sync with each other.  Update BOTH lists together when the allowlist
#   changes.  This does NOT catch cross-language α/γ drift on its own.
#
# Tier 2 (test_extension_drift_guard_vs_reify_script): cross-source guard.
#   Invokes the real reify/scripts/lock-charter-guard.sh --list-extensions and
#   compares its output to sorted(CODE_EXTENSIONS).  Skipped when the script
#   is not present (e.g. in a standalone fused-memory checkout).  Run this in
#   any environment that has both repos checked out side-by-side.
# ---------------------------------------------------------------------------

# The canonical α/γ vector — update this list AND CODE_EXTENSIONS together.
# Widened 36 -> 58 by the git-ls-files sweep 2026-07-28
# (reify #5726 / dark_factory #3117); generated from α's --list-extensions.
_CANONICAL_EXTENSIONS = [
    'c', 'cc', 'cjs', 'conf', 'cpp', 'css', 'cts', 'cxx',
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


def test_extension_drift_guard():
    """Tier-1 (same-file): sorted(CODE_EXTENSIONS) must match _CANONICAL_EXTENSIONS.

    This is a same-file consistency check — CODE_EXTENSIONS and _CANONICAL_EXTENSIONS
    must be updated together.  For cross-source α/γ drift detection see
    test_extension_drift_guard_vs_reify_script.
    """
    assert sorted(CODE_EXTENSIONS) == _CANONICAL_EXTENSIONS


@pytest.mark.skipif(
    not _REIFY_GUARD_SCRIPT.is_file(),
    reason='reify script not present (standalone checkout; cross-repo drift check skipped)',
)
def test_extension_drift_guard_vs_reify_script():
    """Tier-2 (cross-source): sorted(CODE_EXTENSIONS) must match reify --list-extensions.

    Invokes the real scripts/lock-charter-guard.sh --list-extensions and
    compares its output to sorted(CODE_EXTENSIONS), catching any α/γ divergence
    that the same-file Tier-1 guard would miss.
    """
    result = subprocess.run(
        ['bash', str(_REIFY_GUARD_SCRIPT), '--list-extensions'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            f'reify script exited with code {result.returncode}; '
            f'stderr: {result.stderr[:200]}'
        )
    script_exts = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    assert script_exts == sorted(CODE_EXTENSIONS), (
        f'α/γ drift detected!\n'
        f'  reify --list-extensions : {script_exts!r}\n'
        f'  γ CODE_EXTENSIONS       : {sorted(CODE_EXTENSIONS)!r}\n'
        f'Update CODE_EXTENSIONS (and _CANONICAL_EXTENSIONS) to match reify.'
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


def _tracked_paths(repo_root: Path) -> list[str] | None:
    """Every ``git ls-files`` path under *repo_root*, or None if not a checkout."""
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files', '-z'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [p for p in result.stdout.split('\0') if p]


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
    this task.  Verified by mutation: removing ``manifest`` from CODE_EXTENSIONS
    fails the reify parametrization (naming
    ``tests/infra/harness-layout-baseline.manifest``) and passes the dark-factory
    one.  Each repo is skipped independently when that checkout is absent,
    mirroring the Tier-2 skipif above.

    Measured precondition when added: dark-factory 2124 tracked paths and reify
    3778, with ZERO non-allowlisted extensions in either — so this carries no
    pre-existing debt, exactly the property the allowlist→corpus test claims for
    its own direction.

    The gate is ``is_file_path`` itself, so a flagged path is by construction one
    the predicate really does misclassify.  A dotted final segment that the
    predicate rejects means either the extension is missing from the allowlist
    (add it — and to α's ``scripts/lock-charter-guard.sh`` and
    ``shared.locking``) or its case differs (the allowlist is deliberately
    lowercase-only; a tracked ``FOO.PNG`` is a naming problem, not an allowlist
    gap).  Extension-less tracked files (``LICENSE``, ``Dockerfile``,
    ``hooks/pre-commit``) are correctly out of scope: they have no extension to
    allowlist and are handled by the escape hatch named in
    ``lock_charter_error``.
    """
    if not repo_root.is_dir():
        pytest.skip(f'{repo} checkout not present at {repo_root}')
    paths = _tracked_paths(repo_root)
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
        f'from CODE_EXTENSIONS, so those files classify as DIRECTORIES and any '
        f'task declaring one in metadata.files is rejected with a '
        f'LockCharterViolation:\n'
        + '\n'.join(f'  .{ext} — e.g. {path}' for ext, path in sorted(unknown.items()))
        + '\nAdd each to CODE_EXTENSIONS here, to shared/src/shared/locking.py, '
        'and to reify scripts/lock-charter-guard.sh _EXTS (all three must agree), '
        'plus _CANONICAL_EXTENSIONS and _ACCEPT_PATHS in both test copies.'
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
