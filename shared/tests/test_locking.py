"""Tests for shared.locking — module path normalization used by both the
orchestrator scheduler and the task curator."""

from __future__ import annotations

import pytest

from shared.locking import (
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
