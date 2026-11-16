"""What the merger fail-safe reports when it blocks a merge (task 5107).

Pinned on real ``AgentResult`` values through the module's public API only.
"""

from __future__ import annotations

from shared.cli_invoke import AgentFailureKind, AgentResult, classify_agent_failure

from orchestrator.merger_failure import (
    CHECK_MERGE_STATUS_BEFORE_RETRY,
    merger_failure_report,
)

OBSERVED_AT = '2026-11-12T00:00:00+00:00'


def _empty_output_failure() -> AgentResult:
    return AgentResult(
        success=False,
        output='',
        subtype='error_empty_output',
        turns=3,
        cost_usd=0.42,
        duration_ms=1234,
        session_id='sess-1',
    )


def _observations(report) -> str:
    return '\n'.join(entry['observation'] for entry in report.evidence)


class TestInvocationFailed:
    def test_empty_output_reason_names_kind_and_subtype_without_dangling_colon(self):
        report = merger_failure_report(_empty_output_failure(), observed_at=OBSERVED_AT)

        assert 'Merger invocation failed' in report.reason
        assert f'kind={AgentFailureKind.EMPTY_OUTPUT.value}' in report.reason
        assert 'error_empty_output' in report.reason
        assert not report.reason.rstrip().endswith(':')

    def test_empty_output_detail_leads_with_reason_and_carries_run_metrics(self):
        report = merger_failure_report(_empty_output_failure(), observed_at=OBSERVED_AT)

        assert report.detail.startswith(report.reason)
        assert 'turns=3' in report.detail
        assert 'cost_usd=0.42' in report.detail

    def test_empty_output_evidence_is_structured_observations(self):
        report = merger_failure_report(_empty_output_failure(), observed_at=OBSERVED_AT)

        assert report.evidence
        for entry in report.evidence:
            assert set(entry) == {'observation', 'measured_at', 'ref'}
            assert entry['measured_at'] == OBSERVED_AT
            assert 'sess-1' in entry['ref']
        observations = _observations(report)
        assert 'error_empty_output' in observations
        assert 'turns=3' in observations
        assert 'cost_usd=0.42' in observations

    def test_suggested_action_is_check_merge_status_before_retry(self):
        report = merger_failure_report(_empty_output_failure(), observed_at=OBSERVED_AT)

        assert report.suggested_action == CHECK_MERGE_STATUS_BEFORE_RETRY

    def test_whitespace_only_output_counts_as_no_output(self):
        result = AgentResult(success=False, output='  \n')

        report = merger_failure_report(result, observed_at=OBSERVED_AT)

        assert not report.reason.rstrip().endswith(':')
        kind = classify_agent_failure(result).kind.value
        assert f'kind={kind}' in report.reason

    def test_non_blank_output_is_carried_as_a_capped_excerpt(self):
        result = AgentResult(success=False, output='x' * 500, subtype='error_during_execution')

        report = merger_failure_report(result, observed_at=OBSERVED_AT)

        kind = classify_agent_failure(result).kind.value
        assert f'kind={kind}' in report.reason
        assert 'x' * 200 in report.reason
        assert 'x' * 201 not in report.reason


class TestNoDisposition:
    def test_empty_output_reason_has_no_dangling_colon(self):
        report = merger_failure_report(
            AgentResult(success=True, output=''), observed_at=OBSERVED_AT,
        )

        assert report.reason.startswith('Merger emitted no/invalid disposition')
        assert 'Merger invocation failed' not in report.reason
        assert not report.reason.rstrip().endswith(':')

    def test_output_is_carried_in_the_reason(self):
        report = merger_failure_report(
            AgentResult(success=True, output='resolved but forgot the tool'),
            observed_at=OBSERVED_AT,
        )

        assert report.reason.startswith('Merger emitted no/invalid disposition')
        assert 'resolved but forgot the tool' in report.reason

    def test_evidence_and_suggested_action_match_the_failed_arm(self):
        report = merger_failure_report(
            AgentResult(success=True, output=''), observed_at=OBSERVED_AT,
        )

        assert report.evidence
        assert report.suggested_action == CHECK_MERGE_STATUS_BEFORE_RETRY


def test_missing_session_id_still_yields_a_non_empty_ref():
    report = merger_failure_report(
        AgentResult(success=False, output='', session_id=''), observed_at=OBSERVED_AT,
    )

    assert report.evidence
    for entry in report.evidence:
        assert entry['ref'].strip()
        assert not entry['ref'].rstrip().endswith('=')
