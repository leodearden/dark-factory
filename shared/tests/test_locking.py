"""Tests for shared.locking — module path normalization used by both the
orchestrator scheduler and the task curator."""

from __future__ import annotations

import pytest

from shared.locking import (
    CODE_EXTENSIONS,
    directory_locks,
    files_to_modules,
    is_file_path,
    modules_conflict,
    normalize_lock,
    strip_directory_locks,
)

# ---------------------------------------------------------------------------
# Drift guard — pins shared.locking.CODE_EXTENSIONS to the canonical vector.
#
# Update _CANONICAL_EXTENSIONS AND CODE_EXTENSIONS together when the allowlist
# changes.  Also update the verbatim copy in
# fused-memory/src/fused_memory/middleware/lock_charter_guard.py AND its
# corresponding _CANONICAL_EXTENSIONS in
# fused-memory/tests/test_lock_charter_guard.py.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Classification corpus for the 22 extensions added by the 2026-07-28
# ``git ls-files`` sweep (reify #5726 / dark_factory #3117).  One real tracked
# path per new extension — dark-factory unless the comment says reify.
# ---------------------------------------------------------------------------

_WIDENED_ACCEPT_PATHS = [
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
    'tests/infra/run-all-classification.manifest',  # reify — the incident path
    'deploy/systemd/orchestrator-reify.service.d/warm-lane.conf',  # reify
    'crates/reify-doc/tests/snapshots/.gitkeep',  # reify
    'crates/reify-fdm/tests/fixtures/toolpath_bracket.golden',  # reify
    'gui/src/editor/reify.grammar',  # reify
    'gui/src-tauri/icons/icon.icns',  # reify
    'gui/src-tauri/icons/icon.ico',  # reify
    'scripts/reify-audit-snapshot-filter.jq',  # reify
    'tree-sitter-reify/.npmrc',  # reify
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
        # Case-sensitive allowlist: 'MD' not in CODE_EXTENSIONS
        assert not is_file_path('README.MD')

    def test_code_extensions_is_frozenset(self):
        assert isinstance(CODE_EXTENSIONS, frozenset)
        # Spot-check canonical members
        assert 'py' in CODE_EXTENSIONS
        assert 'rs' in CODE_EXTENSIONS
        assert 'ts' in CODE_EXTENSIONS
        assert 'md' in CODE_EXTENSIONS

    @pytest.mark.parametrize('path', _WIDENED_ACCEPT_PATHS)
    def test_widened_allowlist_paths_are_files(self, path: str):
        """Real tracked paths for the 22 extensions added by the 2026-07-28 sweep.

        One representative path per extension added in reify #5726 /
        dark_factory #3117.  Every path is a genuinely tracked file in
        dark-factory or reify (verified with ``git ls-files``) — not a synthetic
        filename — so the corpus documents the actual evidence that motivated
        each entry rather than restating the allowlist.

        ``tests/infra/run-all-classification.manifest`` is the path from the
        originating incident: declaring it in ``metadata.files`` was rejected as
        a directory lock because ``manifest`` was absent from the allowlist.
        """
        assert is_file_path(path), (
            f'{path!r} must classify as a file — its extension is on the '
            f'canonical allowlist (58 entries as of reify #5726 / #3117)'
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


class TestCodeExtensionsDriftGuard:
    """Pin shared.locking.CODE_EXTENSIONS to the canonical extension vector.

    This is the α-copy drift guard for the shared.locking canonical source.
    The corresponding γ-copy guard lives in
    fused-memory/tests/test_lock_charter_guard.py::test_extension_drift_guard
    and uses the same ``_CANONICAL_EXTENSIONS`` vector.  Both must be updated
    together whenever the allowlist changes.
    """

    def test_extension_drift_guard(self):
        """sorted(CODE_EXTENSIONS) must match _CANONICAL_EXTENSIONS.

        Update _CANONICAL_EXTENSIONS AND CODE_EXTENSIONS together when the
        allowlist changes.  Also update the verbatim copy in
        lock_charter_guard.py and its _CANONICAL_EXTENSIONS list.
        """
        assert sorted(CODE_EXTENSIONS) == _CANONICAL_EXTENSIONS, (
            f'shared.locking.CODE_EXTENSIONS has drifted from the canonical vector.\n'
            f'  canonical : {_CANONICAL_EXTENSIONS!r}\n'
            f'  actual    : {sorted(CODE_EXTENSIONS)!r}\n'
            f'Update CODE_EXTENSIONS and _CANONICAL_EXTENSIONS together; also update '
            f'lock_charter_guard.py.'
        )
