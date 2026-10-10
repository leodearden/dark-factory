"""Tests for the digest's EWA storm classification and paused-waiting visibility (task 5311)."""

from __future__ import annotations

import dataclasses

import pytest
from test_digest import _make_digest_inputs

import orchestrator.digest as digest
from orchestrator.digest import CategoryShare, EscalationStats


def _section(markdown: str, heading: str) -> str:
    start = markdown.index(f'{heading}\n')
    end = markdown.find('\n## ', start + len(heading))
    return markdown[start:] if end == -1 else markdown[start:end]


class TestDominantCategory:
    def test_sums_across_level_and_status(self) -> None:
        stats = EscalationStats(
            category_level_status_counts={
                ('retry_cap_exhausted', 1, 'pending'): 20,
                ('retry_cap_exhausted', 1, 'resolved'): 2,
                ('design_concern', 0, 'pending'): 5,
                ('infra_issue', 1, 'pending'): 2,
            },
        )
        assert stats.dominant_category() == CategoryShare(
            category='retry_cap_exhausted', count=22, total=29,
        )

    def test_string_form(self) -> None:
        share = CategoryShare(category='retry_cap_exhausted', count=22, total=29)
        assert str(share) == '22/29 retry_cap_exhausted'

    def test_empty_window_is_none(self) -> None:
        assert EscalationStats().dominant_category() is None

    def test_tie_breaks_to_smallest_category(self) -> None:
        stats = EscalationStats(
            category_level_status_counts={
                ('b_cat', 1, 'pending'): 3,
                ('a_cat', 1, 'pending'): 3,
            },
        )
        assert stats.dominant_category() == CategoryShare('a_cat', 3, 6)

    def test_share_is_frozen(self) -> None:
        share = CategoryShare('a_cat', 3, 6)
        field_name = 'count'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(share, field_name, 4)


class TestRenderDominantCategory:
    def _render(self, stats: EscalationStats) -> str:
        inputs = dataclasses.replace(_make_digest_inputs(), escalation_stats=stats)
        return digest.render_digest_markdown(inputs)

    def test_storm_window_names_dominant_category_in_ewa_section(self) -> None:
        markdown = self._render(EscalationStats(
            category_level_status_counts={
                ('retry_cap_exhausted', 1, 'pending'): 3,
                ('design_concern', 0, 'pending'): 1,
            },
        ))
        line = '- Dominant escalation category: 3/4 retry_cap_exhausted'
        assert line in _section(markdown, '## EWA').splitlines()
        assert markdown.count(line) == 1

    def test_empty_stats_render_none(self) -> None:
        markdown = self._render(EscalationStats())
        line = '- Dominant escalation category: none'
        assert line in _section(markdown, '## EWA').splitlines()
        assert markdown.count(line) == 1
