"""Tests for shared.locking — module path normalization used by both the
orchestrator scheduler and the task curator."""

from __future__ import annotations

from shared.locking import (
    CODE_EXTENSIONS,
    directory_locks,
    files_to_modules,
    is_file_path,
    modules_conflict,
    normalize_lock,
    strip_directory_locks,
)


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
