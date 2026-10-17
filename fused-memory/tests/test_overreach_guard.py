"""Tests for fused_memory.services.overreach_guard (task 6137).

The ruling-overreach guard's class vocabulary and judge core, per
docs/ruling-overreach-guard-design-2026-10-01/design.md §4 and §8.
"""
from __future__ import annotations

import pytest
from fused_memory.services.overreach_guard import (
    CLASSIFIER_NAMES,
    EPISODE_CLASSIFIERS,
    GUARDED_CATEGORIES,
    HEAD_CHARS,
    LABELS,
    OVERREACH,
    is_guarded_category,
    matching_classifiers,
)

from fused_memory.models.enums import MemoryCategory

RULING_HEADED = 'RULING (Leo, 2026-09-01, esc-1-2): x'
PLAIN_DECISION = 'Task 848 (stale merge-lane cleanup) is done as of 2026-04-20 with commit bb7101'


class TestClassPredicate:
    @pytest.mark.parametrize('category', list(MemoryCategory))
    def test_only_decision_records_are_in_class(self, category: MemoryCategory) -> None:
        expected = category is MemoryCategory.decisions_and_rationale
        assert is_guarded_category(category) is expected

    def test_an_unknown_category_is_out_of_class(self) -> None:
        assert is_guarded_category(None) is False

    def test_the_class_is_exactly_the_decisions_category(self) -> None:
        assert frozenset({MemoryCategory.decisions_and_rationale}) == GUARDED_CATEGORIES

    def test_classifier_names_are_the_report_keys_in_order(self) -> None:
        assert CLASSIFIER_NAMES == (
            'category_decisions', 'header_ruling_paren', 'header_ruling',
            'ruling_lexeme_head', 'decision_anchor_head',
        )

    def test_episode_classifiers_are_keyed_in_name_order(self) -> None:
        assert tuple(EPISODE_CLASSIFIERS) == CLASSIFIER_NAMES

    def test_episode_classifiers_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            EPISODE_CLASSIFIERS['another'] = lambda category, content: True  # type: ignore[index]

    @pytest.mark.parametrize('name', CLASSIFIER_NAMES)
    def test_each_classifier_is_a_category_content_predicate(self, name: str) -> None:
        holds = EPISODE_CLASSIFIERS[name](MemoryCategory.decisions_and_rationale, RULING_HEADED)
        assert holds is True

    def test_a_ruling_headed_decision_matches_all_five(self) -> None:
        matched = matching_classifiers(MemoryCategory.decisions_and_rationale, RULING_HEADED)
        assert matched == CLASSIFIER_NAMES

    def test_the_same_content_in_another_category_drops_only_the_category(self) -> None:
        matched = matching_classifiers(MemoryCategory.temporal_facts, RULING_HEADED)
        assert matched == CLASSIFIER_NAMES[1:]

    def test_a_plain_decision_record_matches_only_the_category(self) -> None:
        matched = matching_classifiers(MemoryCategory.decisions_and_rationale, PLAIN_DECISION)
        assert matched == ('category_decisions',)

    @pytest.mark.parametrize('name', ['ruling_lexeme_head', 'decision_anchor_head'])
    def test_a_lexeme_past_the_head_window_does_not_count(self, name: str) -> None:
        content = 'x' * 250 + ' ruling (Leo, 2026-09-01, esc-1-2)'
        assert HEAD_CHARS == 200
        assert EPISODE_CLASSIFIERS[name](MemoryCategory.decisions_and_rationale, content) is False

    def test_the_label_vocabulary(self) -> None:
        assert LABELS == (
            'holding', 'bookkeeping', 'context', 'overreach', 'misbound', 'unjudgeable',
        )
        assert OVERREACH == 'overreach'
        assert OVERREACH in LABELS
