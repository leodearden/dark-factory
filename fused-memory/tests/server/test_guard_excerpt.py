"""Contract of fused_memory.server.guard_excerpt: the excerpt every pre-service guard block echoes."""

from __future__ import annotations

from fused_memory.server.guard_excerpt import GUARD_EXCERPT_CHARS, excerpt_fields
from fused_memory.utils.clip_disclosure import TRUNCATION_FACTS_KEY


def test_an_excerpt_within_the_width_is_echoed_whole_with_no_facts():
    at_width = 'x' * GUARD_EXCERPT_CHARS

    assert excerpt_fields(content_excerpt=at_width) == {'content_excerpt': at_width}


def test_a_cut_excerpt_stays_an_unmarked_prefix_and_reports_its_cut():
    content = 'x' * (GUARD_EXCERPT_CHARS + 5)

    assert excerpt_fields(content_excerpt=content) == {
        'content_excerpt': content[:GUARD_EXCERPT_CHARS],
        TRUNCATION_FACTS_KEY: [{'field': 'content_excerpt', 'cap': GUARD_EXCERPT_CHARS, 'elided': 5}],
    }


def test_a_matched_excerpt_is_echoed_and_only_its_cut_reported():
    matched = 'y' * (GUARD_EXCERPT_CHARS + 9)

    assert excerpt_fields(content_excerpt='short', matched_excerpt=matched) == {
        'content_excerpt': 'short',
        'matched_excerpt': matched[:GUARD_EXCERPT_CHARS],
        TRUNCATION_FACTS_KEY: [{'field': 'matched_excerpt', 'cap': GUARD_EXCERPT_CHARS, 'elided': 9}],
    }
