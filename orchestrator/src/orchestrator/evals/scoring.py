"""Contract-agnostic P4 review scoring for eval runs (Phase-1 substrate).

PRD eval-framework-revival §ι, Invariant P4, Boundary test B8.

A reviewer's structured verdict is persisted in one of two shapes depending on
the contract era it ran under:

  - **legacy** (``--json-schema`` reviewer): the raw payload
    ``{reviewer, verdict, issues:[{severity, ...}], summary}`` written to
    ``reviews/<name>.json`` by ``TaskArtifacts.write_review``;
  - **MCP verdict-tool**: the SAME payload wrapped in a schema-versioned
    envelope ``{role, schema_version, session_id, emitted_at, verdict:<payload>}``
    written to ``verdicts/<role>.json`` by ``verdict_tools._envelope`` /
    ``write_verdict``.

``workflow._run_reviewer`` already unwraps ``envelope['verdict']`` back to the
legacy payload, so both eras reduce to ONE payload shape. This module
centralizes that reduction as a single tested function so P4 scoring is
contract-agnostic: the same review content yields the same quality score
regardless of which artifact shape it was persisted in. The quality formula
itself stays single-sourced in ``metrics.compute_composite`` — this module only
normalizes the artifact and counts issue severities (mirroring
``artifacts.aggregate_reviews``), it does not re-derive the score.

Kept deliberately minimal: it is the Phase-1 substrate the μ OFAT/matrix
scoring DRIVER consumes downstream of ι — ι does not build that driver.
"""

from __future__ import annotations

from typing import Any

from orchestrator.evals.metrics import EvalMetrics, compute_composite


def _unwrap_verdict_envelope(artifact: dict) -> dict:
    """Return the inner review payload, unwrapping the verdict-tool envelope.

    Detects the MCP verdict-tool envelope by its schema-versioned shape — a
    dict carrying both ``schema_version`` and ``role`` with a nested dict
    ``verdict`` — and returns ``artifact['verdict']`` (mirroring
    ``workflow._run_reviewer``'s ``envelope['verdict']`` unwrap). Otherwise the
    dict is already the legacy payload and is returned as-is.

    The envelope check is deliberately specific: a legacy payload's own
    ``verdict`` field is a STRING (``'PASS'`` / ``'ISSUES_FOUND'``), not a
    dict, so it never spuriously matches the envelope shape.
    """
    verdict = artifact.get('verdict')
    if 'schema_version' in artifact and 'role' in artifact and isinstance(verdict, dict):
        return verdict
    return artifact


def quality_from_review_artifact(
    artifact: dict,
    *,
    plan_steps: int,
    debug_cycles: int = 0,
) -> float:
    """Compute the composite quality score for a persisted review *artifact*.

    Contract-agnostic (P4 / B8): accepts EITHER the legacy review payload or
    the MCP verdict-tool envelope (unwrapped via
    :func:`_unwrap_verdict_envelope`), counts blocking vs suggestion issues by
    ``severity == 'blocking'`` exactly as ``artifacts.aggregate_reviews`` does
    — including that path's ERROR filter: an artifact whose top-level
    ``verdict`` is ``ERROR`` is skipped (0 blocking / 0 suggestions), never
    counted by severity. It returns ``metrics.compute_composite`` of an
    :class:`EvalMetrics` built from those counts — so the quality formula stays
    single-sourced. Reads only
    the artifact dict plus the scalar ``plan_steps`` / ``debug_cycles`` knobs;
    it never touches a transcript, worktree, or file path.

    ``tests_pass=True`` is assumed: this scorer answers "given the code passed
    its gates, how clean was the review?" — the pass/fail gate is scored
    elsewhere (``compute_composite`` returns 0 when tests fail).
    """
    payload = _unwrap_verdict_envelope(artifact)
    issues: list[Any] = payload.get('issues', []) or []

    # Mirror ``artifacts.aggregate_reviews``' ERROR filter exactly: a review
    # whose top-level ``verdict`` is ``ERROR`` is skipped there entirely (0
    # blocking / 0 suggestions), never counted by severity. Zero the issue list
    # here too, so an ERROR artifact that still carried ``issues`` entries
    # scores identically through this scorer and the aggregate path — keeping
    # the single-sourcing claim exact.
    if payload.get('verdict') == 'ERROR':
        issues = []

    blocking = 0
    suggestions = 0
    for issue in issues:
        if isinstance(issue, dict) and issue.get('severity') == 'blocking':
            blocking += 1
        else:
            suggestions += 1

    metrics = EvalMetrics(
        tests_pass=True,
        plan_steps=plan_steps,
        review_blocking_issues=blocking,
        review_suggestions=suggestions,
        debug_cycles=debug_cycles,
    )
    return compute_composite(metrics)
