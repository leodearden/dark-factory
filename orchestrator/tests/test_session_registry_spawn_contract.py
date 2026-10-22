"""The registry half of the /spawn contract: the result-handback outcome vocabulary and the self-exit input."""

from __future__ import annotations

import pytest

from orchestrator import session_registry as sr


class TestResultOutcomeVocabulary:
    @pytest.mark.parametrize(('text', 'expected'), [
        ('outcome: done\nchanged: x\n', 'done'),
        ('---\noutcome: handed-off\nchanged: task/5140\naction_needed: poll\n---\nprose', 'handed-off'),
        ('outcome: blocked', 'blocked'),
        ('outcome: abandoned\n', 'abandoned'),
        ('outcome: in-progress\n', 'in-progress'),
        ('---\noutcome: in-progress\nchanged: half of it\n---\nprose', 'in-progress'),
        ('outcome: maybe\n', None),
        ('', None),
        ('I finished the work.\n', None),
        ('---\nchanged: x\n---\noutcome: done\n', None),
    ])
    def test_header_outcome_or_none(self, text, expected):
        assert sr.parse_result_outcome(text) == expected

    def test_parser_returns_the_typed_outcome(self):
        assert sr.parse_result_outcome('outcome: in-progress\n') is sr.ResultOutcome.IN_PROGRESS

    def test_outcomes_are_spelled_in_trailer_order(self):
        assert sr.RESULT_OUTCOMES == ('done', 'blocked', 'abandoned', 'handed-off', 'in-progress')

    def test_owning_outcomes_are_terminal_and_only_the_checkpoint_is_not(self):
        assert sr.OWNING_OUTCOMES <= sr.TERMINAL_RESULT_OUTCOMES
        assert sr.TERMINAL_RESULT_OUTCOMES == frozenset(sr.RESULT_OUTCOMES) - {'in-progress'}
        assert sr.OWNING_OUTCOMES == {'done', 'handed-off'}
