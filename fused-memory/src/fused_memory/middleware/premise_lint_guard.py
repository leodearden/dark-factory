"""Premise-lint guard (ξ).

Validates recon-authored task descriptions at the fused-memory
``submit_task`` boundary, rejecting submissions that assert a known-false
premise about recon's control-plane mechanics (the 2083/2092/2093
false-premise batch — e.g. "run_id persists across cycles") with a
structured ``ValidationError`` before any persistence happens.

## Contract (task 2231/W5-ξ)

Enforcement is scoped to recon-stage callers only: ``agent_id`` must satisfy
``isinstance(agent_id, str) and agent_id.startswith('recon-stage-')``. Every
other caller (human, ``claude-interactive``, an absent agent_id, ...) is
entirely unaffected — ``premise_lint_error`` returns ``None`` immediately.

For a recon-stage caller, ``description``, ``prompt``, ``title``, and
``details`` are ALL scanned via
:func:`fused_memory.reconciliation.recon_self_model.premise_lint`.
``submit_task``'s own docstring names ``prompt`` as "Task description for
AI generation" — its primary content field — and ``details`` as "Task
details / implementation notes", which recon-authored remediation tasks
commonly use to carry substantial mechanism prose; a false premise stated
in ANY of these four fields (with the others clean or empty) must be
caught, so linting ``description`` alone would leave three channels
unchecked. Any :class:`~fused_memory.reconciliation.recon_self_model.Violation`
returned for any field causes a hard rejection naming the violated
invariant(s) and carrying each violation's detail text — this is the
single source of truth for what counts as a known-false premise, so the
guard can never drift from ``render_marker_lifecycle_section()``'s /
``render_suppression_schema_section()``'s rendered-prompt text describing
these mechanisms.

Each field is linted INDEPENDENTLY — one ``premise_lint`` call per field —
rather than concatenated into a single string first. A match must
therefore be wholly contained within a single field; a false premise
cannot be assembled by straddling the join between two otherwise-clean
fields (e.g. a ``description`` ending "...the run_id" plus a ``details``
beginning "persists across cycles..." must NOT combine into a spurious
cross-field match).

## Known limitation: remediation framing is not exempted

``premise_lint`` matches surface phrasing, not authorial intent — it
cannot distinguish a recon agent ASSERTING a false premise as ground truth
(the 2083/2092/2093 failure mode this guard exists to close) from one
*describing* an alleged bug using the same false phrasing (e.g. "Fix the
bug where run_id persists across cycles"). This is a deliberate,
conservative tradeoff, not an oversight: because the premise is false by
construction (see ``run_id_is_fresh_per_run()`` /
``markers_deleted_only_by_gc()`` in ``recon_self_model``), a task claiming
to FIX such a "bug" is itself restating exactly the confusion this guard
is designed to reject, so remediation-style framing ("fix the bug
where...", "the bug is that...") does NOT suppress a match. A legitimate
recon-authored task in this area should instead name the real defect
without restating the false premise as fact (e.g. "code at X incorrectly
assumes run_id is stable across cycles; fix X to stop relying on that") —
the rejection's ``hint`` field says as much so a rejected caller can
resubmit correctly framed.

## Distinct from recon_code_fix_premise_guard

This module is NOT related to ``recon_code_fix_premise_guard.py``, which is
a different mechanism entirely: a curator-path guard operating on
``CandidateTask`` objects against a YAML registry with live-source
re-verification. This guard is a pure regex lint over the description text
of an incoming ``submit_task`` call, mirroring ``execution_class_guard.py``'s
shape (same boundary, same recon-stage scoping, same structured-error
return) rather than that curator-path mechanism.
"""

from __future__ import annotations

from typing import Any

from fused_memory.reconciliation.recon_self_model import Violation, premise_lint

__all__ = [
    'premise_lint_error',
]


def premise_lint_error(
    description: str | None,
    agent_id: str | None,
    project_root: str,
    *,
    prompt: str | None = None,
    title: str | None = None,
    details: str | None = None,
) -> dict[str, Any] | None:
    """Validate every recon-stage free-text content field against
    known-false premises.

    Returns a structured error dict (``{'error': ..., 'error_type':
    'ValidationError', 'hint': ...}``) when a recon-stage caller's
    description, prompt, title, or details asserts a known-false premise
    about recon's control-plane mechanics, or ``None`` when all four are
    clean — including for every non-recon caller, which is never enforced.

    Args:
        description: The task description text to lint. ``None`` is
            treated as empty (never matches any premise rule).
        agent_id: The resolved caller identity. Enforcement fires only when
            this is a string starting with ``'recon-stage-'``.
        project_root: Absolute path to the project root. Unused — the lint
            is pure text matching with no filesystem check — but kept to
            mirror ``execution_class_error``'s signature (PRD §8.5) so the
            two guards read as a matched pair, wired at the same submit_task
            seam.
        prompt: The task ``prompt`` text — submit_task's own docstring
            names this "Task description for AI generation", its primary
            content field. ``None`` is treated as empty.
        title: The task ``title`` text. ``None`` is treated as empty.
        details: The task ``details`` text — submit_task's own docstring
            names this "Task details / implementation notes"; recon-authored
            remediation tasks commonly carry substantial mechanism prose
            here, so it is linted on equal footing with the other three
            fields. ``None`` is treated as empty.

    Each field is linted independently (see module docstring) so a match
    can never straddle the boundary between two fields.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None

    violations: list[Violation] = []
    for field in (description, prompt, title, details):
        violations.extend(premise_lint(field or ''))
    if not violations:
        return None

    invariants = ', '.join(sorted({v.invariant for v in violations}))
    seen_detail: set[str] = set()
    detail_texts: list[str] = []
    for v in violations:
        if v.detail not in seen_detail:
            seen_detail.add(v.detail)
            detail_texts.append(v.detail)
    detail_text = ' '.join(detail_texts)
    return {
        'error': (
            f'Task description, prompt, title, or details asserts a '
            f'known-false premise about recon internals, violating '
            f'invariant(s): {invariants}. {detail_text}'
        ),
        'error_type': 'ValidationError',
        'hint': (
            'Remove or correct the false premise before resubmitting. If '
            'this is a legitimate remediation task, name the actual defect '
            '(e.g. "code at X incorrectly assumes ...") rather than '
            'restating the false premise as fact. See '
            'recon_self_model.render_marker_lifecycle_section() and '
            'render_suppression_schema_section() for the canonical mechanics.'
        ),
    }
