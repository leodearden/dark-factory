"""Tests for the standalone candidate_key leaf (fm-task-dedup W8 task A1)."""

from __future__ import annotations

from fused_memory.middleware.candidate_key import compute_candidate_key


class TestComputeCandidateKey:
    def test_non_empty_title_returns_16_char_lowercase_hex(self):
        key = compute_candidate_key('Fix the bug', ['a.py'])
        assert key is not None
        assert len(key) == 16
        assert key == key.lower()
        assert all(c in '0123456789abcdef' for c in key)

    def test_none_title_returns_none(self):
        assert compute_candidate_key(None, ['a.py']) is None

    def test_empty_title_returns_none(self):
        assert compute_candidate_key('', ['a.py']) is None

    def test_whitespace_only_title_returns_none(self):
        assert compute_candidate_key('   ', ['a.py']) is None

    def test_file_order_insensitive(self):
        a = compute_candidate_key('Fix the bug', ['b.py', 'a.py'])
        b = compute_candidate_key('Fix the bug', ['a.py', 'b.py'])
        assert a == b

    def test_title_case_and_internal_whitespace_insensitive(self):
        a = compute_candidate_key('Fix   The  Bug', [])
        b = compute_candidate_key('fix the bug', [])
        assert a == b

    def test_files_none_equals_files_empty_list(self):
        a = compute_candidate_key('Fix the bug', None)
        b = compute_candidate_key('Fix the bug', [])
        assert a == b

    def test_non_string_file_entries_are_dropped_not_raised(self):
        """A caller-supplied files list with non-str entries (e.g. a dict or
        int smuggled in via malformed metadata) must never raise — every
        caller documents that key computation must not be the reason an
        insert/update fails."""
        with_junk = compute_candidate_key('Fix the bug', ['a.py', 123, {'x': 1}, None])
        strings_only = compute_candidate_key('Fix the bug', ['a.py'])
        assert with_junk == strings_only

    def test_all_non_string_files_treated_as_empty(self):
        key = compute_candidate_key('Fix the bug', [123, {'x': 1}])
        assert key == compute_candidate_key('Fix the bug', [])
