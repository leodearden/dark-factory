"""The digest of L0s a steward already concluded, as rendered into a
fresh-session briefing (task 5819)."""

from __future__ import annotations

import dataclasses

import pytest
from shared.task_statuses import TaskStatus

from orchestrator.agents.steward_digest import (
    DIGEST_MAX_ENTRIES,
    DIGEST_RESOLUTION_MAX_CHARS,
    HandledL0,
    render_handled_digest,
)
from orchestrator.workflow_types import (
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
)


def _resolved(escalation_id: str, text: str = 'fixed it', category: str = 'task_failure') -> HandledL0:
    return HandledL0(escalation_id, category, StewardResolved(resolution_text=text))


def _entry_line(block: str, escalation_id: str) -> str:
    [line] = [line for line in block.splitlines() if escalation_id in line]
    return line


def test_nothing_handled_renders_nothing():
    assert render_handled_digest(()) == ''


def test_most_recent_entry_comes_first():
    block = render_handled_digest(
        (_resolved('esc-first'), _resolved('esc-second'), _resolved('esc-third')),
    )
    assert block.index('esc-third') < block.index('esc-second') < block.index('esc-first')


def test_each_entry_line_names_its_escalation_and_category():
    block = render_handled_digest(
        (_resolved('esc-a', category='task_failure'), _resolved('esc-b', category='merge_conflict')),
    )
    assert 'task_failure' in _entry_line(block, 'esc-a')
    assert 'merge_conflict' in _entry_line(block, 'esc-b')


class TestOutcomeDescriptions:
    def _line(self, outcome) -> str:
        block = render_handled_digest((HandledL0('esc-x', 'task_failure', outcome),))
        return _entry_line(block, 'esc-x')

    def test_resolved_shows_label_and_text(self):
        line = self._line(StewardResolved(resolution_text='fixed the import'))
        assert 'resolved' in line
        assert 'fixed the import' in line

    def test_reescalated_names_level_one_and_the_new_id(self):
        line = self._line(StewardReescalatedL1(esc_id='esc-42-9'))
        assert 'level 1' in line
        assert 'esc-42-9' in line

    @pytest.mark.parametrize(
        ('wip', 'flag'), [(True, 'WIP commits present'), (False, 'no WIP commits')],
    )
    def test_interrupted_shows_reason_and_wip_flag(self, wip, flag):
        line = self._line(StewardInterrupted(reason='attempt_cap', wip_commits_present=wip))
        assert 'interrupted' in line
        assert 'attempt_cap' in line
        assert flag in line

    def test_budget_exhausted_says_so(self):
        assert 'budget exhausted' in self._line(StewardBudgetExhausted())

    def test_terminal_decision_shows_the_status(self):
        line = self._line(StewardTerminalDecision(new_status=TaskStatus.DEFERRED))
        assert 'deferred' in line


class TestResolutionText:
    def test_a_long_resolution_is_truncated_with_an_ellipsis(self):
        text = ''.join(f'{i:05d}' for i in range(DIGEST_RESOLUTION_MAX_CHARS))
        line = _entry_line(render_handled_digest((_resolved('esc-long', text),)), 'esc-long')
        assert text not in line
        assert text[: DIGEST_RESOLUTION_MAX_CHARS // 2] in line
        assert line.endswith('…')

    def test_a_multi_line_resolution_stays_on_its_entry_line(self):
        block = render_handled_digest(
            (_resolved('esc-ml', 'line one\n\n   line two\tthree'), _resolved('esc-next')),
        )
        assert 'line one line two three' in _entry_line(block, 'esc-ml')
        assert 'line two' not in _entry_line(block, 'esc-next')


def test_only_the_most_recent_entries_are_kept_and_the_rest_are_counted():
    ids = [f'esc-{i:02d}-x' for i in range(DIGEST_MAX_ENTRIES + 3)]
    block = render_handled_digest(tuple(_resolved(i) for i in ids))
    for kept in ids[3:]:
        assert kept in block
    for dropped in ids[:3]:
        assert dropped not in block
    assert '3 older not shown' in block


def test_block_framing_and_lead_sentence():
    block = render_handled_digest((_resolved('esc-a'),))
    assert block.startswith('\n## ')
    assert block.endswith('\n')
    assert not block.endswith('\n\n')
    lowered = block.lower()
    assert 'fresh session' in lowered
    assert 'only record' in lowered


def test_handled_l0_is_frozen():
    pytest.raises(
        dataclasses.FrozenInstanceError, setattr, _resolved('esc-a'), 'escalation_id', 'esc-b',
    )
