"""The role-neutral zero-work predicate: a failed run that did no work.

Pinned on real ``AgentResult`` values (no mocks) so the predicate is tested
against the same classifier every caller reaches through it.
"""

from __future__ import annotations

import pytest
from shared.cli_invoke import AgentFailureKind, AgentResult

from orchestrator.zero_work_failure import ZERO_WORK_FAILURE_KINDS, is_zero_work_failure


def _empty_output(*, transcript_turns: int | None) -> AgentResult:
    return AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_empty_output',
        transcript_turns=transcript_turns,
    )


class TestZeroWorkShapes:
    def test_empty_output_with_zero_transcript_turns(self):
        assert is_zero_work_failure(_empty_output(transcript_turns=0))

    def test_unreadable_transcript_counts_as_zero_work(self):
        assert is_zero_work_failure(_empty_output(transcript_turns=None))

    def test_cli_pre_turn_rejection(self):
        result = AgentResult(
            success=False,
            output='',
            subtype='error_cli_input_rejected',
            transcript_turns=0,
        )
        assert is_zero_work_failure(result)


class TestNotZeroWork:
    def test_productive_run_killed_from_outside(self):
        assert not is_zero_work_failure(_empty_output(transcript_turns=5))

    @pytest.mark.parametrize(
        'result',
        [
            pytest.param(
                AgentResult(success=False, output='', timed_out=True, transcript_turns=0),
                id='timed-out-wedge',
            ),
            pytest.param(
                AgentResult(success=False, output='', subtype='error_max_turns', turns=31),
                id='turn-cap',
            ),
            pytest.param(
                AgentResult(success=False, output='API Error: 529', api_error_status=529),
                id='fast-5xx',
            ),
            pytest.param(
                AgentResult(
                    success=False,
                    output='',
                    subtype='error_max_budget_usd',
                    cost_usd=5.07,
                ),
                id='budget-cap',
            ),
        ],
    )
    def test_other_failure_shapes(self, result: AgentResult):
        assert not is_zero_work_failure(result)


def test_zero_work_kinds_are_the_two_pre_turn_failures():
    pre_turn_kinds = {AgentFailureKind.EMPTY_OUTPUT, AgentFailureKind.CLI_INPUT_REJECTED}
    assert frozenset(pre_turn_kinds) == ZERO_WORK_FAILURE_KINDS
