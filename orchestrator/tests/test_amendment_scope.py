"""Tests for partition_suggestions_by_delta (task 2750).

The pure post-aggregation filter that partitions reviewer suggestions by
whether each finding's ``location`` (``file:line``) falls within the
amendment's NEW-side changed line ranges.  In-delta suggestions stay in the
post-amendment verdict; out-of-delta suggestions are routed to the curator
by the caller.  This is the enforceable half of scoping a post-amendment
review to the amendment delta — the reviewer-prompt constraint is advisory.
"""

from __future__ import annotations


def _sugg(location: str | None, **extras) -> dict:
    """Build a suggestion dict with the fields the filter cares about.

    Mirrors the ``_sugg`` helper pattern in test_workflow_amendment.py.
    """
    s: dict = {
        'reviewer': 'test_analyst',
        'severity': 'suggestion',
        'category': 'coverage',
        'description': 'an opinion',
        'suggested_fix': 'do the thing',
    }
    if location is not None:
        s['location'] = location
    s.update(extras)
    return s


# Amendment new-side delta: src/foo.py changed lines [10, 20]; src/bar.py
# changed lines [5, 8] and [30, 35].  Keyed exactly like
# parse_diff_added_line_ranges output.
DELTA = {
    'src/foo.py': [(10, 20)],
    'src/bar.py': [(5, 8), (30, 35)],
}


def _partition():
    from orchestrator.review_suggestions.amendment_scope import (
        partition_suggestions_by_delta,
    )
    return partition_suggestions_by_delta


class TestPartitionSuggestionsByDelta:
    """partition_suggestions_by_delta(suggestions, delta_ranges, *, context_lines)."""

    def test_line_inside_delta_range_is_in_delta(self):
        """(a) file+line inside a delta range → in_delta."""
        fn = _partition()
        s = _sugg('src/foo.py:15')
        in_delta, out_of_delta = fn([s], DELTA)
        assert in_delta == [s]
        assert out_of_delta == []

    def test_line_outside_all_ranges_same_file_is_out(self):
        """(b) same delta file but line outside every range → out_of_delta."""
        fn = _partition()
        s = _sugg('src/foo.py:99')
        in_delta, out_of_delta = fn([s], DELTA)
        assert in_delta == []
        assert out_of_delta == [s]

    def test_file_not_in_delta_is_out(self):
        """(c) file not present in delta_ranges at all → out_of_delta."""
        fn = _partition()
        s = _sugg('src/other.py:15')
        in_delta, out_of_delta = fn([s], DELTA)
        assert in_delta == []
        assert out_of_delta == [s]

    def test_file_level_location_on_delta_file_is_in_delta(self):
        """(d) location with a delta file but no ':line' (file-level) → in_delta."""
        fn = _partition()
        s = _sugg('src/foo.py')
        in_delta, out_of_delta = fn([s], DELTA)
        assert in_delta == [s]
        assert out_of_delta == []

    def test_malformed_locations_are_out_and_never_raise(self):
        """(e) missing/empty/whitespace/':'-only/unparseable-line → out, no raise."""
        fn = _partition()
        malformed = [
            _sugg(None),                 # missing location key entirely
            _sugg(''),                   # empty string
            _sugg('   '),                # whitespace only
            _sugg(':'),                  # colon only → empty file part
            _sugg('src/foo.py:abc'),     # delta file but unparseable line
        ]
        in_delta, out_of_delta = fn(malformed, DELTA)
        assert in_delta == []
        assert out_of_delta == malformed

    def test_context_lines_boundary(self):
        """(f) context_lines=N includes line at end+N but excludes end+N+1."""
        fn = _partition()
        at_boundary = _sugg('src/foo.py:23')      # 20 + 3
        beyond = _sugg('src/foo.py:24')            # 20 + 4
        in_delta, out_of_delta = fn(
            [at_boundary, beyond], DELTA, context_lines=3,
        )
        assert in_delta == [at_boundary]
        assert out_of_delta == [beyond]

    def test_context_lines_lower_boundary(self):
        """(f) context tolerance also applies below the range start."""
        fn = _partition()
        at_boundary = _sugg('src/foo.py:7')        # 10 - 3
        beyond = _sugg('src/foo.py:6')             # 10 - 4
        in_delta, out_of_delta = fn(
            [at_boundary, beyond], DELTA, context_lines=3,
        )
        assert in_delta == [at_boundary]
        assert out_of_delta == [beyond]

    def test_order_preserved_within_partitions(self):
        """(g) input order is preserved within each partition."""
        fn = _partition()
        s1 = _sugg('src/foo.py:11')     # in
        s2 = _sugg('src/zzz.py:1')      # out (file not in delta)
        s3 = _sugg('src/bar.py:31')     # in (second range of bar)
        s4 = _sugg('src/foo.py:500')    # out (line outside)
        in_delta, out_of_delta = fn([s1, s2, s3, s4], DELTA)
        assert in_delta == [s1, s3]
        assert out_of_delta == [s2, s4]

    def test_empty_suggestions(self):
        """(h) empty suggestions → ([], [])."""
        fn = _partition()
        assert fn([], DELTA) == ([], [])

    def test_empty_delta_ranges_routes_all_out(self):
        """(i) empty delta_ranges → ([], all-suggestions)."""
        fn = _partition()
        s1 = _sugg('src/foo.py:15')
        s2 = _sugg('src/bar.py')
        in_delta, out_of_delta = fn([s1, s2], {})
        assert in_delta == []
        assert out_of_delta == [s1, s2]

    def test_second_range_membership(self):
        """A line in the file's SECOND range is in_delta (multi-range files)."""
        fn = _partition()
        s = _sugg('src/bar.py:33')      # within (30, 35)
        in_delta, out_of_delta = fn([s], DELTA)
        assert in_delta == [s]
        assert out_of_delta == []
