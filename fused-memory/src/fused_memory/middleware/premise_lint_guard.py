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

For a recon-stage caller, BOTH ``description`` and ``prompt`` are scanned
via :func:`fused_memory.reconciliation.recon_self_model.premise_lint`.
``submit_task``'s own docstring names ``prompt`` as "Task description for
AI generation" — its primary content field — so a false premise stated
only in ``prompt`` (with a clean or empty ``description``) must be caught
too; linting ``description`` alone would leave that channel unchecked. Any
:class:`~fused_memory.reconciliation.recon_self_model.Violation` returned
for either field causes a hard rejection naming the violated invariant(s)
and carrying each violation's detail text — this is the single source of
truth for what counts as a known-false premise, so the guard can never
drift from ``render_marker_lifecycle_section()``'s /
``render_suppression_schema_section()``'s rendered-prompt text describing
these mechanisms.

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

from fused_memory.reconciliation.recon_self_model import premise_lint

__all__ = [
    'premise_lint_error',
]


def premise_lint_error(
    description: str | None,
    agent_id: str | None,
    project_root: str,
    *,
    prompt: str | None = None,
) -> dict[str, Any] | None:
    """Validate the recon-stage description AND prompt against known-false
    premises.

    Returns a structured error dict (``{'error': ..., 'error_type':
    'ValidationError', 'hint': ...}``) when a recon-stage caller's
    description or prompt asserts a known-false premise about recon's
    control-plane mechanics, or ``None`` when both are clean — including
    for every non-recon caller, which is never enforced.

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
            content field — linted alongside ``description`` so a false
            premise stated in either channel is caught. ``None`` is treated
            as empty.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None

    violations = premise_lint(f'{prompt or ""}\n{description or ""}')
    if not violations:
        return None

    invariants = ', '.join(sorted({v.invariant for v in violations}))
    details = ' '.join(v.detail for v in violations)
    return {
        'error': (
            f'Task description or prompt asserts a known-false premise '
            f'about recon internals, violating invariant(s): {invariants}. '
            f'{details}'
        ),
        'error_type': 'ValidationError',
        'hint': (
            'Remove or correct the false premise in the description/prompt '
            'before resubmitting. See recon_self_model.render_marker_lifecycle_section() '
            'and render_suppression_schema_section() for the canonical mechanics.'
        ),
    }
