"""Every payload the defect filer writes (plans/client-feed-prd.md §14.3, §14.2 decision 10), as pure functions of an inbox line and the policy."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict
from shared.defects import (
    DeclareClosure,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
)

from client_feed.contract import (
    CriterionId,
    DefectReported,
    PositiveInt,
    Quarantined,
    Restated,
    Token,
)
from client_feed.filer_tools import (
    CLIENT_DEFECT_KEY,
    DEFECTS_KEY,
    FILER_IDENTITY,
    GATE_FOR_KEY,
    SOURCE_KEY,
    RefusalKind,
    TaskSubmission,
)

FIX_TITLE: Final = 'Client defect {ref} against criterion {criterion_id}'
GATE_TITLE: Final = 'Human gate: client defect {ref}'
DEFECT_SUMMARY: Final = 'Client report {report} on block {block} against criterion {criterion_id}'
CRITERIA_POINTER: Final = 'acceptance/criteria.yaml'
RESTATED_HEADING: Final = "## The defect, as restated by the portal from the client's report"
LEO_WRITES: Final = 'Leo writes this description at the gate.'
_FIX_TASK_KIND: Final = 'normal'
_GATE_TASK_KIND: Final = 'deterministic'
_FILER_WITHDRAWAL: Final = Withdrawal(reason=WithdrawalReason(reason=WithdrawalCode.NOT_A_DEFECT))


class FilerFallback(BaseModel):
    """Why the filer filed a report's typed facts only: the refusal of its restated fix, by kind and code."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: RefusalKind
    code: Token


class ClientDefectFacts(BaseModel):
    """A fix task's x_client_defect: the typed facts of its report, written and read through this one model."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    block: PositiveInt
    report: PositiveInt
    criterion_id: CriterionId
    outcome_kind: Literal['restated', 'quarantined']
    restater_prompt_version: Token
    filer_fallback: FilerFallback | None = None


def facts_of(report: DefectReported, fallback: FilerFallback | None) -> ClientDefectFacts:
    return ClientDefectFacts(
        block=report.block,
        report=report.report,
        criterion_id=report.criterion_id,
        outcome_kind=report.outcome.kind,
        restater_prompt_version=report.restater.prompt_version,
        filer_fallback=fallback,
    )


def requires_gate(facts: ClientDefectFacts, *, auto_dispatch: bool) -> bool:
    """Every fix waits on a human gate except a clean restatement under auto_dispatch."""
    return not (auto_dispatch and facts.outcome_kind == 'restated' and facts.filer_fallback is None)


def client_declaration(
    ref: str, *, summary: str, declared_at: datetime, declared_by: str
) -> dict[str, Any]:
    """The fresh client defect a fix task declares at creation."""
    return DefectDeclaration(
        ref=ref,
        origin=DefectOrigin.CLIENT,
        summary=summary,
        declared_at=declared_at,
        declared_by=declared_by,
        status=DefectStatus.OPEN,
    ).model_dump(mode='json')


def withdrawal_event(ref: str) -> dict[str, Any]:
    """The filer's not_a_defect withdrawal of a ruled report's defect."""
    return DeclareClosure(ref=ref, closure=_FILER_WITHDRAWAL).model_dump(mode='json')


def operator_acceptance_event(ref: str) -> dict[str, Any]:
    """The operator's acceptance of that withdrawal: Leo ruled on the portal, and the filer records it."""
    ruling = Ruling(subject=RulingSubject.CLOSURE, accepted=True, ruled_by=RuledBy.OPERATOR)
    return Rule(ref=ref, ruling=ruling).model_dump(mode='json')


def is_filer_withdrawal(declaration: DefectDeclaration) -> bool:
    """Whether a declared defect is pending on exactly the filer's withdrawal."""
    return declaration.status is DefectStatus.PENDING and declaration.closure == _FILER_WITHDRAWAL


def _fallback_line(fallback: FilerFallback) -> str:
    return (
        f'The filer could not file the restated report: fused-memory refused it ({fallback.kind}, '
        f'{fallback.code}), so only its typed facts are filed.'
    )


def _criterion_line(criterion_id: str) -> str:
    return f'Acceptance criterion {criterion_id} ({CRITERIA_POINTER}).'


def _restated_description(report: DefectReported, outcome: Restated) -> str:
    steps = '\n'.join(f'{number}. {step}' for number, step in enumerate(outcome.steps, 1))
    return '\n\n'.join((
        _criterion_line(report.criterion_id),
        RESTATED_HEADING,
        f'Observed: {outcome.observed}',
        f'Expected: {outcome.expected}',
        f'Steps:\n{steps}' if steps else 'Steps: none given',
    ))


def _typed_facts_description(report: DefectReported, fallback: FilerFallback | None) -> str:
    if fallback is not None:
        why = _fallback_line(fallback)
    elif isinstance(report.outcome, Quarantined):
        why = 'The portal quarantined the report: ' + ', '.join(report.outcome.reasons) + '.'
    else:
        raise ValueError('a restated report with no fallback is filed with its restatement')
    return '\n\n'.join((_criterion_line(report.criterion_id), why, LEO_WRITES))


def fix_submission(
    report: DefectReported, *, ref: str, priority: str, fallback: FilerFallback | None
) -> TaskSubmission:
    """The fix task for a report; the restated fields are the only client-derived text, and only without a fallback."""
    facts = facts_of(report, fallback)
    outcome = report.outcome
    description = (
        _restated_description(report, outcome)
        if isinstance(outcome, Restated) and fallback is None
        else _typed_facts_description(report, fallback)
    )
    declaration = client_declaration(
        ref,
        summary=DEFECT_SUMMARY.format(
            report=report.report, block=report.block, criterion_id=report.criterion_id
        ),
        declared_at=report.at,
        declared_by=FILER_IDENTITY,
    )
    return TaskSubmission(
        title=FIX_TITLE.format(ref=ref, criterion_id=report.criterion_id),
        description=description,
        priority=str(priority),
        task_kind=_FIX_TASK_KIND,
        metadata={
            SOURCE_KEY: FILER_IDENTITY,
            DEFECTS_KEY: [declaration],
            CLIENT_DEFECT_KEY: facts.model_dump(mode='json', exclude_none=True),
        },
    )


def _gate_outcome_line(report: DefectReported, facts: ClientDefectFacts) -> str:
    if facts.filer_fallback is not None:
        return f'Outcome: {facts.outcome_kind}. {_fallback_line(facts.filer_fallback)}'
    if isinstance(report.outcome, Quarantined):
        return f'Outcome: {facts.outcome_kind}, for: ' + ', '.join(report.outcome.reasons) + '.'
    return f'Outcome: {facts.outcome_kind}.'


def gate_submission(
    report: DefectReported,
    *,
    ref: str,
    fix_task_id: str,
    facts: ClientDefectFacts,
    priority: str,
    project_id: str,
) -> TaskSubmission:
    """The human gate the fix waits on; it carries no client-derived text."""
    description = '\n\n'.join((
        f'Client defect {ref} is filed as task {fix_task_id}, which waits on this gate.',
        _gate_outcome_line(report, facts),
        f'Portal admin page: project {project_id}, block {report.block}, report {report.report}.',
        f'Review it there, edit task {fix_task_id} if needed, record x_client_defect_review, then resume.',
    ))
    return TaskSubmission(
        title=GATE_TITLE.format(ref=ref),
        description=description,
        priority=str(priority),
        task_kind=_GATE_TASK_KIND,
        metadata={'always_escalates': True, SOURCE_KEY: FILER_IDENTITY, GATE_FOR_KEY: ref},
    )
