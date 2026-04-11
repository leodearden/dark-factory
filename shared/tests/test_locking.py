"""Tests for shared.locking — module path normalization used by both the
orchestrator scheduler and the task curator."""

from __future__ import annotations

from shared.locking import files_to_modules, normalize_lock


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
