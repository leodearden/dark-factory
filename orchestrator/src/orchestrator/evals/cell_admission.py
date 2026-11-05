"""The plan-only admission rule for persisted eval cells: which cells enter the
plan-quality pool, the score each contributes, and the reductions every report
surface takes over that pool.
"""

from __future__ import annotations

import logging
from typing import Any

from .metrics import RoleUnderTest
from .plan_predicates import produced_a_plan

logger = logging.getLogger(__name__)


def is_plan_only(metrics: dict[str, Any]) -> bool:
    """Is this trial a PLAN-ONLY cell (task 3099)?

    ``run_architect_eval`` is the only producer of ARCHITECT cells and is
    plan-only BY CONSTRUCTION — it freezes
    implementer/debugger/reviewer/verify, so the cell never runs a test and
    carries its quality in ``plan_quality`` rather than ``composite_score``.

    Keyed on the EXISTING ``role_under_test`` stamp rather than a new persisted
    ``plan_only`` field for two reasons: the composite report builder already
    keys ``plan_quality_cap_excluded`` on exactly this test (a second field could
    drift from the first), and a new field would read back as its default on
    every result JSON written before this task — silently mis-classifying the
    whole existing corpus, including the campaign this fix exists to make
    readable.
    """
    return metrics.get('role_under_test') == RoleUnderTest.ARCHITECT


def has_plan_quality_score(metrics: dict[str, Any]) -> bool:
    """THE plan-quality admission cascade: architect / not tainted / scored.

    One function, so every surface that admits or refuses a plan-only cell asks
    the SAME question (task 3099, reviewer amendment). Both the plan-quality and
    the composite report builders call it, and the two clauses used to coincide
    only by COUPLING —
    ``run_architect_eval`` happens to taint exactly when it has no score, so a
    ``cap_tainted``-only test and a ``plan_quality is None``-only test agreed by
    accident. A hand-edited result, a legacy JSON, or a future taint cause that
    preserves the structural floor would break that coincidence and feed a score
    the θ pool REFUSES into the composite pool that survivor selection ranks on.
    """
    return (
        is_plan_only(metrics)
        and not metrics.get('cap_tainted')
        and metrics.get('plan_quality') is not None
    )


def plan_quality_score(
    metrics: dict[str, Any], *, where: str = '?',
) -> float | None:
    """THE θ score of one plan-only cell — ``None`` when it must be EXCLUDED.

    ONE accessor, shared by the composite and plan-quality report builders, so
    ``mean_plan_quality`` and the composite row's ``plan_quality`` cannot drift (task 3302, extending the
    :func:`mean_plan_quality` discipline from the reduction to the value being
    reduced). It COMPOSES with :func:`has_plan_quality_score` rather than
    re-deriving its cascade.

    Three outcomes, two of them deliberately distinct:

    - ``None`` — UNMEASURABLE. Not an admissible plan-only cell at all
      (:func:`has_plan_quality_score`: not an architect run, cap-tainted, or no
      persisted score). The transport layer refused us, so every number would be
      fabricated: the cell leaves the pool entirely (task 3118, unchanged here).
    - ``0.0`` — NO PLAN WAS PRODUCED (``not produced_a_plan``). A healthy
      architect that emitted a stepless artifact is the OPPOSITE of unmeasurable:
      it is a genuine CONTENT measurement worth exactly 0.0, which is what
      :func:`judge.score_plan_structure` already returns for that artifact. This
      reproduces the 2026-07-27 campaign's own hand-computed ``meanPQ_all``
      ("scores a no-plan cell as 0") on the automated table.
    - the persisted float — a plan with steps, scored on its content.

    The persisted score is DISCARDED on the ``0.0`` path, and loudly: a nonzero
    ``plan_quality`` beside ``plan_steps=0`` IS the two-scorer disagreement
    (Graphiti e2066ec6 — ``score_plan_structure`` floors a stepless plan to 0.0
    as an anti-fabrication guard; the LLM judge used to have no such guard,
    until task 3303 closed that gap at the instrument), and this read-time
    floor still has to defend the 2026-07-29 corpus already on disk carrying
    the old, unguarded shape (the cell judge.py cites,
    ``reify_task_12__architect-opus-high__52c66767.json``: ``plan_steps=0``
    alongside ``plan_quality=0.31``) — correcting it silently would hide the
    very cells an operator needs to see. The 2026-07-27 campaign's own no-plan
    cells recorded the guarded ``plan_quality=0.0`` shape instead
    (plans/eval-architect-effort-verdict-2026-07-27.md, defect 3), so only
    07-29 is known to carry the unguarded shape this floor defends against.

    *where* names the cell (``"fixture x config"``) for that warning; the metrics
    dict carries neither, so the caller — which holds the ``EvalResult`` — passes
    it. Purely diagnostic: it never changes the returned score.

    A floored cell therefore warns ONCE PER REPORT BUILD, and the CLI builds both
    surfaces over the same results, so it appears twice per run. Deliberate: the
    accessor stays PURE, and the alternative — remembering which cells have
    already been warned about — is either module-level state that goes quiet on
    the second campaign of a long-lived process or a per-build set threaded
    through both builders. Two identical, cell-named lines are the cheaper
    failure mode; both name their ``where``, so neither reads as a second cell.
    """
    if not has_plan_quality_score(metrics):
        return None
    if not produced_a_plan(metrics):
        persisted = metrics.get('plan_quality')
        if persisted:
            logger.warning(
                'Plan-quality floor applied to %s: persisted plan_quality=%s '
                'with plan_steps=0 — a stepless artifact is not a plan, so this '
                "cell scores 0.0 (score_plan_structure's anti-fabrication "
                "answer), NOT the LLM judge's number. Two scorers disagreed "
                'here; the deterministic one wins.',
                where, persisted,
            )
        return 0.0
    return float(metrics['plan_quality'])


def is_unmeasurable(metrics: dict[str, Any]) -> bool:
    """Did this trial measure NOTHING — so it must leave EVERY pool?

    True only for a PLAN-ONLY cell with no admissible θ score
    (:func:`has_plan_quality_score`): the architect was refused, so the cell has
    no quality, and its recorded ``cost_usd`` / ``workflow_duration_ms`` are the
    $0.00 / 0 ms of a run that never happened rather than a measurement.

    Averaging those zeros into the row's cost/latency means was the MIRROR IMAGE
    of the defect task 3118 removed from ``mean_plan_quality``: it handed a
    schedule-attributable "2x cheaper, 2x faster" bonus to whichever candidate
    happened to be scheduled inside a cap window — on the very table task 3099
    makes the operator's ranking surface. A WORKFLOW trial is never unmeasurable
    here: a failing run really did burn its cost and latency.
    """
    return is_plan_only(metrics) and not has_plan_quality_score(metrics)


# The σ validity marker's key, spelled ONCE (task 3628). The sibling
# definition is ``scripts/run_fable_trial_v2_campaign.py``'s ``MARKER_KEY``;
# the two are kept in agreement by :func:`judged_without_reference` below,
# which implements that module's rule 3 verbatim.
_JUDGED_WITHOUT_REFERENCE_KEY = 'judged_without_reference'


def judged_without_reference(metrics: dict[str, Any]) -> bool | None:
    """Was this cell's ``plan_quality`` judged with NO reference diff?

    THE ONE per-cell reading of the σ validity marker (task 3628), called by
    both report surfaces so they cannot drift on the one rule that makes the
    marker mean anything: **absence is not False**.

    * ``True`` / ``False`` — the instrument measured this cell's validity.
    * ``None`` — the metrics dict predates σ, so the cell's validity was NEVER
      measured. Reading that back as ``False`` would let a rebuild over the
      2026-07-27 / 07-29 corpus — which the provenance note in
      ``docs/plan-scoring-and-judge.md`` says was judged blind on half the hard subset — render ``0`` on every surface
      and suppress the trailer entirely, byte-identically to a campaign whose
      every score was graded against a real landed diff. That is precisely the
      silent degradation σ exists to remove, re-introduced at the reader.

    The rule is deliberately the SAME one
    ``scripts/run_fable_trial_v2_campaign.py`` (task 3632) states as its rule
    3 and enforces with ``MARKER_KEY not in metrics``. That script no longer
    counts the marker itself — its per-candidate counter was retired in favour
    of this module's architect-only aggregate (see :func:`judged_blind_count`),
    so the two surfaces now share ONE implementation of the rule rather than
    two agreeing copies. Two readings of one field must not answer the same
    question differently, least of all with the report — the surface an
    operator actually reads — giving the reassuring answer.
    It is also the same discipline :func:`plan_quality_score` applies at READ
    time for the no-plan floor, and for the same reason: the old corpus is
    already on disk.
    """
    if _JUDGED_WITHOUT_REFERENCE_KEY not in (metrics or {}):
        return None
    return bool(metrics.get(_JUDGED_WITHOUT_REFERENCE_KEY))


def judged_blind_count(counted: int, unmeasured: int) -> int | None:
    """Fold a per-config tally into a count, or ``None`` when ANY cell is keyless.

    The aggregate twin of :func:`judged_without_reference`: one unmeasured
    cell makes the whole config's count unknowable. This IS the count task
    3632's campaign script consumes — its own per-candidate counter was retired
    in favour of this one, so the rule has a single home. Reporting the measured
    cells' count instead would understate the bound while looking like a
    complete answer.
    """
    return None if unmeasured else counted


def mean_plan_quality(scores: list[float]) -> float | None:
    """Reduce a pool of θ-rubric plan-quality scores to its mean (4 dp).

    THE ONE reduction, shared by BOTH the composite report's row and the
    plan-quality report's per-config aggregate (task 3099). The
    composite row's ``plan_quality`` used to be a first-untainted-TRIAL
    passthrough — a harmless diagnostic echo until this task promoted the field
    to a DECISION surface (survivor selection ranks on it and the composite
    table renders it beside the composite). A
    trial-1 score sitting next to a mean would let the two tables the CLI now prints
    ADJACENTLY — and the ``quality`` / ``plan_quality`` cells of one row — give
    contradictory answers about the same quantity, reintroducing the "two
    exclusion surfaces that disagree are worse than one" hazard on the very
    surface this task exists to make trustworthy. Sharing the reduction (rather
    than duplicating the arithmetic and asserting the two agree) makes that drift
    structurally impossible.

    An EMPTY pool is ``None``, never ``0.0``: "we measured nothing" must not read
    as "it scored nothing".
    """
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def plan_rate(n_admitted: int, no_plan: int) -> float | None:
    """The fraction of ADMITTED plan-only cells that emitted a plan at all (4 dp).

    THE ONE reduction, shared by BOTH the composite report's row and the
    plan-quality report's per-config aggregate — the
    :func:`mean_plan_quality` precedent, and for the same reason: the CLI
    prints those two tables ADJACENTLY, so a reliability figure derived twice
    from separately-maintained arithmetic is a pair of surfaces free to give
    contradictory answers about the same quantity. Sharing the reduction makes
    that drift structurally impossible rather than merely tested-against.

    THE DENOMINATOR is *n_admitted* — the ADMITTED θ pool
    (``len(plan_quality_scores)`` on the composite row, ``n`` on the θ
    aggregate) — deliberately NOT the config's ``trials``. A cap-tainted cell
    is a transport refusal (:func:`has_plan_quality_score`): we never got to
    ask the model, so it is neither a cell that planned nor a cell that failed
    to plan, and it leaves BOTH the numerator and the denominator. Ranging over
    ``trials`` instead would report a candidate as LESS RELIABLE for a question
    it never got to answer, purely because it happened to be scheduled inside a
    session-cap window — the schedule-attributable penalty tasks 3118 and 3099
    spent two rounds removing, reintroduced on the surface survivor selection
    ranks on. It also matches the campaign this
    automates: ``plans/eval-architect-effort-verdict-2026-07-27.md`` computes
    its own planRate over 19 fixtures, dropping the 3 cap-contaminated ones from
    the denominator, not over all 22.

    *no_plan* is the count of those admitted cells that produced NO plan
    (``plan_quality_no_plan`` / ``no_plan``) — floored to 0.0 by
    :func:`plan_quality_score` and KEPT in the pool, which is exactly why they
    belong in this denominator while a refused cell does not.

    An EMPTY pool is ``None``, never ``0.0`` — and never ``1.0``: here the
    fabrication is TWO-SIDED, since a workflow config with no plan-only cell at
    all would be slandered by a 0.0 ("it never planned") and flattered by a 1.0
    ("it always planned"). Only ``None`` says the true thing.
    """
    if n_admitted <= 0:
        return None
    return round((n_admitted - no_plan) / n_admitted, 4)


def cost_per_plan(total_cost: float, n_planned: int) -> float | None:
    """The spend per USABLE plan (4 dp) — a deliberately ASYMMETRIC ratio.

    *total_cost* is the spend over every ADMITTED plan-only cell INCLUDING the
    ones that produced nothing; *n_planned* counts only the cells that actually
    emitted a plan (``n_admitted - no_plan``). That asymmetry is the entire
    content of "$ per USABLE plan": a no-plan cell burns real budget and returns
    nothing (``plans/eval-architect-effort-verdict-2026-07-27.md`` measured
    $0.5–$3 against a real plan's ~$3.7), so a cheap-but-unreliable candidate's
    per-fixture cost advantage is partly ILLUSORY — the doc's own arithmetic put
    fable at $3.456/fixture but $4.731 per usable plan, i.e. no cheaper than the
    opus-max incumbent while failing to plan 5x as often. Netting the failed
    attempts out of the numerator would report exactly that illusion instead of
    removing it.

    What the numerator does NOT include is an UNMEASURABLE cell's $0.00: the
    caller accumulates it under the same admission decision that fills the θ
    pool, so a cap-tainted cell's price-of-a-run-that-never-happened stays out
    for the same reason :func:`is_unmeasurable` keeps it out of the ``cost``
    pool. Letting it in would DILUTE $/plan with a free run — reporting a
    session-cap window as a discount.

    An empty denominator is ``None``, never ``0.0`` and never a
    ``ZeroDivisionError``: a config that failed to plan on every cell is a real
    campaign outcome and the one whose row an operator most needs to read, and
    "we got no plan at any price" must not render as "plans were free".
    """
    if n_planned <= 0:
        return None
    return round(total_cost / n_planned, 4)
