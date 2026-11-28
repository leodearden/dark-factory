"""The human-curator gate predicate and the runner's curator-gate declarer (task 3346)."""

from __future__ import annotations

import pytest

from shared.human_curator_gate import (
    curator_gate_declarer,
    declares_only_the_curator_gate_pin,
    is_human_curator_gate,
)


class TestIsHumanCuratorGate:
    @pytest.mark.parametrize('marker', [True, 'true', 1])
    def test_a_truthy_marker_is_a_curator_gate(self, marker: object):
        assert is_human_curator_gate({'human_curator_gate': marker}) is True

    @pytest.mark.parametrize('metadata', [{}, {'human_curator_gate': False}, {'human_curator_gate': ''}])
    def test_an_absent_or_falsy_marker_is_not(self, metadata: dict):
        assert is_human_curator_gate(metadata) is False


class TestCuratorGateDeclarer:
    def test_names_the_task(self):
        assert curator_gate_declarer('3181') == 'task-3181-human-curator-gate'

    def test_a_record_with_no_declarer_has_nothing_else_to_consult(self):
        assert declares_only_the_curator_gate_pin([], '3181') is True

    def test_only_the_runner_declarer(self):
        assert declares_only_the_curator_gate_pin(
            [' task-3181-human-curator-gate', 'task-3181-human-curator-gate', ' '], '3181',
        ) is True

    def test_another_declarer_beside_it(self):
        assert declares_only_the_curator_gate_pin(
            ['task-3181-human-curator-gate', 'task-3546-second-deviation-notice'], '3181',
        ) is False

    def test_another_tasks_curator_declarer(self):
        assert declares_only_the_curator_gate_pin(['task-866-human-curator-gate'], '3181') is False
