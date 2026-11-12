"""What the merger fail-safe reports when
``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._resolve_and_resubmit``
blocks a merge on an untrustworthy disposition; the block decision itself stays
there.
"""

from __future__ import annotations

from dataclasses import dataclass

from escalation.models import EvidenceEntry
from shared.cli_invoke import AgentFailureClass, AgentResult, classify_agent_failure

CHECK_MERGE_STATUS_BEFORE_RETRY = 'check_merge_status_before_retry'

_OUTPUT_EXCERPT_CHARS = 200
_NO_SESSION_ID = '(none reported)'


@dataclass(frozen=True)
class MergerFailureReport:
    reason: str
    detail: str
    evidence: tuple[EvidenceEntry, ...]
    suggested_action: str


def merger_failure_report(result: AgentResult, *, observed_at: str) -> MergerFailureReport:
    cls = classify_agent_failure(result)
    reason = _reason(result, cls)
    return MergerFailureReport(
        reason=reason,
        detail=f'{reason}\n{cls.diagnostic_detail}',
        evidence=_evidence(result, cls, observed_at=observed_at),
        suggested_action=CHECK_MERGE_STATUS_BEFORE_RETRY,
    )


def _reason(result: AgentResult, cls: AgentFailureClass) -> str:
    excerpt = result.output.strip()[:_OUTPUT_EXCERPT_CHARS]
    if not result.success:
        head = (
            f'Merger invocation failed (kind={cls.kind.value}, '
            f'subtype={result.subtype!r}): {cls.summary}'
        )
        return f'{head} — output: {excerpt}' if excerpt else head
    if excerpt:
        return f'Merger emitted no/invalid disposition: {excerpt}'
    return 'Merger emitted no/invalid disposition (merger returned no output)'


def _evidence(
    result: AgentResult, cls: AgentFailureClass, *, observed_at: str,
) -> tuple[EvidenceEntry, ...]:
    ref = f'merger session_id={result.session_id or _NO_SESSION_ID}'
    classification = (
        f'merger success={result.success} kind={cls.kind.value} '
        f'subtype={result.subtype!r} timed_out={result.timed_out}'
    )
    run_metrics = (
        f'merger turns={result.turns} transcript_turns={result.transcript_turns} '
        f'cost_usd={result.cost_usd} duration_ms={result.duration_ms} '
        f'api_error_status={result.api_error_status} len(output)={len(result.output)}'
    )
    return tuple(
        EvidenceEntry(observation=observation, measured_at=observed_at, ref=ref)
        for observation in (classification, run_metrics)
    )
