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

import json
from collections.abc import Callable
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Recovery-behavior scoring (eval-revival η)
#
# A rubric DISTINCT from the base composite (PRD §η decision 6): frozen perfect
# inputs reward obedient instruction-followers, so a model's NOTICE-and-REPAIR
# behavior is scored separately. Every detector reads ONLY a persisted artifact
# — the final plan.json, the aggregated review/verdict blocking issues, or the
# final committed diff — never an agent transcript (Invariant P4 / boundary
# test B8). This keeps recovery scoring contract-agnostic AND every η test
# hermetic (synthetic artifacts, no paid LLM run).
# ---------------------------------------------------------------------------


def plan_step_deviated(plan: dict | None, params: dict) -> bool:
    """True when the final plan's planted-wrong step deviated from its sentinel.

    Maps the ``wrong_plan_step`` adversarial type: a synthesized plan is seeded
    with exactly one deliberately-wrong step carrying a unique ``wrong_marker``
    sentinel. Recovery = the model NOTICED and changed that step, so in the
    final plan artifact the step with ``step_id`` either no longer contains the
    sentinel (rewritten) or is gone entirely (removed).

    - ``params['step_id']``     — the id of the planted-wrong step.
    - ``params['wrong_marker']``— the unique sentinel token planted in it.

    Returns ``False`` on an empty/``None`` plan or an empty steps list (no
    recovery signal to read); ``False`` while the wrong step survives with the
    sentinel verbatim; ``True`` once the step is rewritten (sentinel absent) or
    absent from an otherwise-populated plan.
    """
    step_id = params['step_id']
    wrong_marker = params['wrong_marker']
    if not plan or not isinstance(plan, dict):
        return False
    steps = plan.get('steps') or []
    if not steps:
        return False
    matching = [
        s for s in steps
        if isinstance(s, dict) and s.get('id') == step_id
    ]
    if not matching:
        # The planted-wrong step was removed from a populated plan → deviated.
        return True
    # Present: recovered iff NONE of the matching steps still carry the sentinel.
    return all(
        wrong_marker not in json.dumps(step, default=str) for step in matching
    )


def regression_flagged(blocking_issues: list[dict], params: dict) -> bool:
    """True when >= ``min_matches`` blocking issues each name a planted marker.

    Maps the ``planted_regression`` adversarial type: a regression is planted in
    the reviewed diff and recovery = a reviewer NOTICED it, so a blocking review
    issue names the planted-regression markers. Each blocking-issue dict is run
    through :func:`_unwrap_verdict_envelope` first, so the detector reads either
    the legacy ``--json-schema`` payload's issue dicts OR the MCP verdict-tool
    envelope's — contract-agnostic (P4 / B8); for an already-aggregated issue
    dict the unwrap is a harmless no-op.

    - ``params['markers']``     — marker substrings a blocking issue must name.
    - ``params['min_matches']`` — how many DISTINCT blocking issues must each
      name >=1 marker (default 1). Matching is case-insensitive.
    """
    markers = [str(m).lower() for m in params['markers']]
    min_matches = int(params.get('min_matches', 1))
    count = 0
    for raw in blocking_issues or []:
        if not isinstance(raw, dict):
            continue
        issue = _unwrap_verdict_envelope(raw)
        text = json.dumps(issue, default=str).lower()
        if any(marker in text for marker in markers):
            count += 1
    return count >= min_matches


def real_cause_addressed(diff: str, params: dict) -> bool:
    """True when the committed diff touches the real root cause, not the surface.

    Maps the ``misleading_verify_failure`` adversarial type: the fixture reports
    a misleading verify failure whose surface symptom (``misleading_markers``)
    is a red herring; the real fix lives elsewhere (``real_cause_markers``).
    Recovery = the model diagnosed past the misleading message and changed the
    real root cause, so the final diff text contains a ``real_cause_marker``.

    - ``params['real_cause_markers']`` — substrings the real fix must touch.
    - ``params['misleading_markers']`` — the red-herring surface (documentary;
      a diff that touches only these, or nothing, is not recovery → ``False``).
    """
    if not diff:
        return False
    return any(str(marker) in diff for marker in params['real_cause_markers'])


# Closed registry mapping detector kind -> (callable, the persisted-artifact
# key it consumes). ``compute_recovery_score`` dispatches each rubric criterion
# through this; an unregistered kind raises loudly (code-owned fixture typo).
_RECOVERY_DETECTORS: dict[str, tuple[Callable[..., bool], str]] = {
    'plan_step_deviated': (plan_step_deviated, 'plan'),
    'regression_flagged': (regression_flagged, 'blocking_issues'),
    'real_cause_addressed': (real_cause_addressed, 'diff'),
}


def compute_recovery_score(
    rubric: dict,
    *,
    plan: dict,
    blocking_issues: list[dict],
    diff: str,
) -> float:
    """Score a recovery *rubric* against persisted artifacts, clamped to [0, 1].

    Iterates ``rubric['criteria']`` — each ``{id, detector, weight, params}`` —
    dispatches every criterion's ``detector`` (via :data:`_RECOVERY_DETECTORS`)
    against the relevant artifact (``plan`` / ``blocking_issues`` / ``diff``),
    and returns the weight-weighted fraction of satisfied criteria,
    ``round(sum(weight*satisfied) / sum(weight), 4)`` clamped to ``[0, 1]``.

    Raises ``ValueError`` on an empty/missing ``criteria`` list, an unknown
    detector kind, or a non-positive total weight — the rubric is a code-owned
    adversarial fixture, so a typo must fail loudly (structured-facts-at-failure
    / loud-over-silent). The boundary caller (``metrics.collect_metrics``)
    wraps this so a real-run failure degrades to a null column, not a nuked run.
    """
    criteria = rubric.get('criteria') or []
    if not criteria:
        raise ValueError(
            'recovery rubric has no criteria — a code-owned adversarial fixture '
            'must define >=1 criterion'
        )
    artifacts: dict[str, Any] = {
        'plan': plan,
        'blocking_issues': blocking_issues,
        'diff': diff,
    }
    total_weight = 0.0
    satisfied_weight = 0.0
    for criterion in criteria:
        kind = criterion.get('detector')
        if kind not in _RECOVERY_DETECTORS:
            raise ValueError(
                f'unknown recovery detector kind: {kind!r} '
                f'(registered: {sorted(_RECOVERY_DETECTORS)})'
            )
        detector, artifact_key = _RECOVERY_DETECTORS[kind]
        weight = float(criterion.get('weight', 1.0))
        params = criterion.get('params') or {}
        satisfied = bool(detector(artifacts[artifact_key], params))
        total_weight += weight
        if satisfied:
            satisfied_weight += weight
    if total_weight <= 0.0:
        raise ValueError('recovery rubric total weight must be > 0')
    score = satisfied_weight / total_weight
    return round(min(max(score, 0.0), 1.0), 4)


async def recovery_score_for_run(
    task: dict,
    artifacts: Any,
    worktree: Path,
    base_commit: str,
) -> float | None:
    """Recovery score for one eval run, or ``None`` for a non-adversarial task.

    Returns ``None`` when ``task.get('adversarial')`` is falsy — the C4
    ``recovery_score | null`` sentinel that keeps "populated vs null" distinct
    from "scored 0.0". For an adversarial task it reads only persisted
    artifacts — ``artifacts.read_plan()``, ``artifacts.aggregate_reviews()
    .blocking_issues``, and the committed diff via ``snapshots.get_diff`` — and
    scores them against ``task['adversarial']['recovery_rubric']``.

    ``base_commit`` is the task's ``pre_task_commit`` (the same base
    ``metrics._git_diff_stats`` uses). ``snapshots`` is imported lazily to keep
    this module import-light and mirror the local-import idiom the eval scoring
    path already uses.
    """
    adversarial = task.get('adversarial')
    if not adversarial:
        return None
    rubric = adversarial['recovery_rubric']
    plan = artifacts.read_plan() if artifacts else {}
    reviews = artifacts.aggregate_reviews() if artifacts else None
    blocking_issues = list(reviews.blocking_issues) if reviews else []
    from orchestrator.evals import snapshots
    diff = await snapshots.get_diff(worktree, base_commit)
    return compute_recovery_score(
        rubric, plan=plan, blocking_issues=blocking_issues, diff=diff,
    )
