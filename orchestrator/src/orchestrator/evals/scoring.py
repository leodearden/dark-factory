"""Contract-agnostic P4 review scoring for eval runs (Phase-1 substrate).

PRD eval-framework-revival §ι, Invariant P4, Boundary test B8.

A reviewer's verdict is persisted either as the legacy ``reviews/<name>.json``
payload or wrapped in the MCP verdict-tool envelope written to
``verdicts/<role>.json``. Every artifact is normalised by
``orchestrator/review_payload.py::normalize_review``, the same reader
``artifacts.aggregate_reviews`` uses, so the same review content scores the
same whichever shape persisted it. The quality formula itself stays
single-sourced in ``metrics.compute_composite``; this module only counts the
normalised payloads' issues.

Kept deliberately minimal: it is the Phase-1 substrate the μ OFAT/matrix
scoring DRIVER consumes downstream of ι — ι does not build that driver.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.review_payload import ReviewPayload, normalize_review


def quality_from_reviews(
    reviews: Iterable[ReviewPayload],
    *,
    plan_steps: int,
    debug_cycles: int = 0,
) -> float:
    """The composite quality score of *reviews*' summed blocking and suggestion counts.

    ``tests_pass=True`` is assumed: this scorer answers "given the code passed
    its gates, how clean was the review?" — the pass/fail gate is scored
    elsewhere (``compute_composite`` returns 0 when tests fail).
    """
    payloads = list(reviews)
    metrics = EvalMetrics(
        tests_pass=True,
        plan_steps=plan_steps,
        review_blocking_issues=sum(len(p.blocking_issues) for p in payloads),
        review_suggestions=sum(len(p.suggestions) for p in payloads),
        debug_cycles=debug_cycles,
    )
    return compute_composite(metrics)


def quality_from_review_artifact(
    artifact: Mapping[str, Any],
    *,
    plan_steps: int,
    debug_cycles: int = 0,
) -> float:
    """The composite quality score of one persisted review *artifact*.

    The artifact — legacy payload or verdict-tool envelope — is normalised by
    ``orchestrator/review_payload.py::normalize_review``, which raises
    ``ReviewPayloadError`` for a malformed one. It reads only the artifact and
    the scalar knobs, never a transcript, worktree, or file path (P4 / B8).
    """
    return quality_from_reviews(
        [normalize_review(artifact)],
        plan_steps=plan_steps,
        debug_cycles=debug_cycles,
    )


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
    issue names the planted-regression markers. The issues are the aggregated
    ones, which ``artifacts.aggregate_reviews`` reads through
    ``orchestrator/review_payload.py::normalize_review`` whichever contract
    wrote the review (P4 / B8); each issue dict's whole JSON text is matched.

    - ``params['markers']``     — marker substrings a blocking issue must name.
    - ``params['min_matches']`` — how many DISTINCT blocking issues must each
      name >=1 marker (default 1). Matching is case-insensitive.
    """
    markers = [str(m).lower() for m in params['markers']]
    min_matches = int(params.get('min_matches', 1))
    count = 0
    for issue in blocking_issues or []:
        if not isinstance(issue, dict):
            continue
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
