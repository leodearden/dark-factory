"""C4's settle and scoring rules for a shadow cell (plans/live-shadow-eval-prd.md C4).

One purpose: given a cell's persisted ``EvalMetrics`` and the reference its row
settled against, say what the cell scores. The coordinator consumes it for the
consequence leg-2 gate, and theta1's report applies the same rules to the
persisted metrics and the row's reference, so a score is always derived and
never stored as a copy (INV-9).

``EvalMetrics.composite_score`` as ``collect_metrics`` computed it is NOT the
shadow score of a ``reference_kind=none`` cell: C4 leaves that cell's composite
``None``, never ``0.0``, which only :class:`ImplementerScore` can carry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from orchestrator.evals.judge import score_plan_structure
from orchestrator.evals.metrics import DECLINE_KINDS, EvalMetrics, compute_composite
from orchestrator.shadow_types import ReferenceKind, ShadowShape

__all__ = [
    'ArchitectScore',
    'ImplementerScore',
    'ShadowReference',
    'consequence_leg_warranted',
    'score_architect_cell',
    'score_implementer_cell',
    'scoring_rule_for',
]

_PLANNED = 'planned'


@dataclass(frozen=True)
class ShadowReference:
    """What production outcome a cell is scored against: a landed sha, or none."""

    kind: ReferenceKind
    sha: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ReferenceKind.LANDED and (self.sha is None or not self.sha.strip()):
            raise ValueError(f'a landed reference requires a non-blank sha (got {self.sha!r})')
        if self.kind is ReferenceKind.NONE and self.sha is not None:
            raise ValueError(f'a none reference carries no sha (got {self.sha!r})')

    @classmethod
    def landed(cls, sha: str) -> ShadowReference:
        return cls(ReferenceKind.LANDED, sha)

    @classmethod
    def none(cls) -> ShadowReference:
        return cls(ReferenceKind.NONE)


@dataclass(frozen=True)
class ImplementerScore:
    """C4's implementer row: gates and review always, a composite only against a landing."""

    reference_kind: ReferenceKind
    tests_pass: bool | None
    lint_clean: bool | None
    typecheck_clean: bool | None
    review_blocking_issues: int
    composite_score: float | None


@dataclass(frozen=True)
class ArchitectScore:
    """C4's architect row: the reference-judged quality if landed, else plan structure."""

    reference_kind: ReferenceKind
    terminal_kind: str | None
    plan_quality: float | None
    plan_structure: float | None


def score_implementer_cell(metrics: EvalMetrics, reference: ShadowReference) -> ImplementerScore:
    landed = reference.kind is ReferenceKind.LANDED
    return ImplementerScore(
        reference_kind=reference.kind,
        tests_pass=metrics.tests_pass,
        lint_clean=metrics.lint_clean,
        typecheck_clean=metrics.typecheck_clean,
        review_blocking_issues=metrics.review_blocking_issues,
        composite_score=compute_composite(metrics) if landed else None,
    )


def score_architect_cell(
    metrics: EvalMetrics, reference: ShadowReference, plan: dict | None,
) -> ArchitectScore:
    """Score an architect cell; *plan* is the plan it produced, ``None`` for a decline."""
    if reference.kind is ReferenceKind.LANDED:
        judged = None if metrics.judged_without_reference else metrics.plan_quality
        return ArchitectScore(reference.kind, metrics.terminal_kind, judged, None)
    return ArchitectScore(reference.kind, metrics.terminal_kind, None, score_plan_structure(plan))


def scoring_rule_for(
    shape: ShadowShape,
) -> Callable[..., ImplementerScore | ArchitectScore]:
    """C4's row for *shape*: only the architect leg is scored as a plan."""
    if shape is ShadowShape.ARCHITECT:
        return score_architect_cell
    return score_implementer_cell


def consequence_leg_warranted(metrics: EvalMetrics) -> bool:
    """Whether a leg-1 architect cell produced a plan that leg 2 can implement.

    The last terminal call wins, so a plan followed by a decline is a decline.
    """
    if metrics.terminal_kind in DECLINE_KINDS:
        return False
    return metrics.terminal_kind == _PLANNED and metrics.plan_steps > 0
