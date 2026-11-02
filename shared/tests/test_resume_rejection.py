"""Task 5579: only a pre-first-turn resume REJECTION retries fresh.

A resumed invocation that ran turns and then ended non-success did real work;
``invoke_with_cap_retry`` returns it to the caller like any fresh result.
"""

from __future__ import annotations

import dataclasses

import pytest
from _measured_cli_results import measured_resume_rejection

from shared.cli_invoke import is_resume_rejected


def _rejection(**changes: object):
    return dataclasses.replace(measured_resume_rejection('sess-armed'), **changes)


class TestIsResumeRejected:
    def test_the_measured_rejection_is_a_rejection(self):
        assert is_resume_rejected(measured_resume_rejection()) is True

    def test_a_success_is_not_a_rejection(self):
        assert is_resume_rejected(_rejection(success=True)) is False

    def test_a_timeout_is_not_a_rejection(self):
        assert is_resume_rejected(_rejection(timed_out=True)) is False

    @pytest.mark.parametrize(
        'changes',
        [pytest.param({'turns': 1}, id='took-a-turn'), pytest.param({'cost_usd': 0.01}, id='billed')],
    )
    def test_evidence_of_work_vetoes_the_marker(self, changes):
        assert is_resume_rejected(_rejection(**changes)) is False

    @pytest.mark.parametrize(
        'stderr',
        [pytest.param('Segmentation fault', id='other-stderr'), pytest.param('', id='empty-stderr')],
    )
    def test_a_pre_turn_failure_without_the_marker_is_not_a_rejection(self, stderr):
        result = _rejection(stderr=stderr)
        assert result.subtype == 'error_empty_output'
        assert is_resume_rejected(result) is False

    def test_the_marker_counts_only_on_stderr(self):
        marker_text = measured_resume_rejection('sess-armed').stderr
        assert is_resume_rejected(_rejection(stderr='', output=marker_text)) is False

    @pytest.mark.parametrize(
        'stderr',
        [
            pytest.param('NO CONVERSATION FOUND WITH SESSION ID: sess-armed', id='upper-case'),
            pytest.param(
                'Warning: something unrelated\nNo conversation found with session ID: sess-armed\n',
                id='one-line-of-several',
            ),
        ],
    )
    def test_the_marker_matches_case_insensitively_on_any_line(self, stderr):
        assert is_resume_rejected(_rejection(stderr=stderr)) is True
