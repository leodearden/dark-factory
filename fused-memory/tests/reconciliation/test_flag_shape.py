"""Tests for fused_memory.reconciliation.flag_shape — the leaf flag-text matcher."""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.flag_shape import contains_any_casefolded


class TestContainsAnyCasefolded:
    def test_member_found_inside_text(self):
        assert contains_any_casefolded('task_STRANDED_x', ('strand',)) is True

    def test_members_are_casefolded_too(self):
        assert contains_any_casefolded('stranded', ('STRAND',)) is True

    def test_no_member_found(self):
        assert contains_any_casefolded('abc', ('x',)) is False

    @pytest.mark.parametrize('text', [None, '', 5, b'strand', ['strand']])
    def test_total_over_malformed_text(self, text):
        assert contains_any_casefolded(text, ('strand',)) is False

    def test_empty_family_never_matches(self):
        assert contains_any_casefolded('anything', ()) is False
