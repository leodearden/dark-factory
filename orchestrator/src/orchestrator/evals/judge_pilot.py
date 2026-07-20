"""Routing κ — judge-model OFAT pilot: decide + render + analyze (task 2815).

An ADDITIVE, CONSUME-ONLY analysis surface over the eval-revival ο judge-OFAT
seam (task 2825). This module IMPORTS ``report.build_composite_report`` /
``format_composite_table`` and ``runner.load_results`` / ``run_ofat_stage`` and
MODIFIES NONE of them (PRD decision 11 forbids this PRD from editing the eval
instrument — runner.py / configs.py / the judge / the Elo machinery). It is the
consumer of ο's composite that turns a judge-OFAT run into a decide-and-act
verdict: trial ``judge-haiku`` against the ``judge-sonnet`` incumbent on the
end-to-end composite, and adopt ``models.judge=haiku`` iff the candidate is
non-inferior AND cheaper (both with sufficient trials), else keep sonnet and
escalate with a committed report.

The live ≥3-trial OFAT run (a Sonnet-max implementer × fixtures × trials) and the
on-ADOPT config flip are OPERATOR steps (see
``plans/judge-ofat-pilot-runbook.md``); this module only computes and renders the
decision over an already-built composite report, so it stays pure, deterministic,
and unit-testable on synthetic composite dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Default names for the ο judge-OFAT candidates (configs.JUDGE_EVAL_CONFIGS).
INCUMBENT_JUDGE = 'judge-sonnet'
CANDIDATE_JUDGE = 'judge-haiku'

# Non-inferiority margin (composite points, composite ∈ [0, 1]). An EXPOSED
# policy knob with a documented default — NOT an empirically-asserted achievable
# bound. The candidate is "non-inferior" when its composite CI lower bound is no
# more than this margin below the incumbent composite mean (the standard
# one-sided non-inferiority test). 0.05 == tolerate at most a 5-composite-point
# regression in exchange for a cheaper judge.
DEFAULT_NON_INFERIORITY_MARGIN = 0.05


@dataclass
class JudgeAdoptionDecision:
    """The structured judge-adoption verdict over a composite report.

    ``verdict`` ∈ {``'adopt'``, ``'marginal'``, ``'reject'``}; ``escalate`` is
    ``False`` only for ``'adopt'``. ``composite_delta`` / ``cost_delta`` /
    ``judge_cost_delta`` are candidate-minus-incumbent (a negative ``cost_delta``
    means the candidate is cheaper). ``reasons`` is a human-readable explanation
    of the verdict. The two rows are the raw ``build_composite_report`` rows for
    the incumbent / candidate configs (``None`` when a row is missing).
    """

    verdict: str
    escalate: bool
    non_inferior: bool
    cheaper: bool
    sufficient: bool
    margin: float
    incumbent: str
    candidate: str
    incumbent_row: dict[str, Any] | None
    candidate_row: dict[str, Any] | None
    composite_delta: float | None
    cost_delta: float | None
    judge_cost_delta: float | None
    reasons: list[str] = field(default_factory=list)


def _composite_mean(row: dict[str, Any]) -> float:
    return float(row.get('composite', 0.0) or 0.0)


def _composite_lo(row: dict[str, Any]) -> float:
    return float(((row.get('ci95') or {}).get('composite') or {}).get('lo', 0.0) or 0.0)


def _composite_sufficient(row: dict[str, Any]) -> bool:
    return bool(((row.get('ci95') or {}).get('composite') or {}).get('sufficient', False))


def _cost(row: dict[str, Any]) -> float:
    return float(row.get('cost_usd', 0.0) or 0.0)


def _judge_cost(row: dict[str, Any]) -> float:
    return float((row.get('judge') or {}).get('cost_usd', 0.0) or 0.0)


def decide_judge_adoption(
    composite_report: dict[str, Any],
    *,
    incumbent: str = INCUMBENT_JUDGE,
    candidate: str = CANDIDATE_JUDGE,
    margin: float = DEFAULT_NON_INFERIORITY_MARGIN,
) -> JudgeAdoptionDecision:
    """Decide whether to adopt the candidate judge over the incumbent.

    Looks up the ``incumbent`` and ``candidate`` rows in
    ``composite_report['configs']`` by config name and computes:

    * ``non_inferior`` — candidate composite CI lower bound ≥ incumbent composite
      mean − ``margin`` (one-sided non-inferiority);
    * ``cheaper`` — candidate mean ``cost_usd`` < incumbent mean ``cost_usd``;
    * ``sufficient`` — both composite CIs are ``sufficient`` (n≥3).

    ``verdict='adopt'`` (``escalate=False``) iff all three hold. This step-2
    implementation only fully resolves the ADOPT path; the reject / marginal /
    insufficient / missing-row branches are filled in by step-4.
    """
    by_name = {str(r.get('config')): r for r in composite_report.get('configs', [])}
    incumbent_row = by_name[incumbent]
    candidate_row = by_name[candidate]

    inc_comp = _composite_mean(incumbent_row)
    cand_comp = _composite_mean(candidate_row)
    cand_lo = _composite_lo(candidate_row)
    inc_cost = _cost(incumbent_row)
    cand_cost = _cost(candidate_row)

    non_inferior = cand_lo >= inc_comp - margin
    cheaper = cand_cost < inc_cost
    sufficient = _composite_sufficient(candidate_row) and _composite_sufficient(incumbent_row)

    composite_delta = cand_comp - inc_comp
    cost_delta = cand_cost - inc_cost
    judge_cost_delta = _judge_cost(candidate_row) - _judge_cost(incumbent_row)

    if non_inferior and cheaper and sufficient:
        verdict, escalate = 'adopt', False
    else:
        verdict, escalate = 'marginal', True

    return JudgeAdoptionDecision(
        verdict=verdict,
        escalate=escalate,
        non_inferior=non_inferior,
        cheaper=cheaper,
        sufficient=sufficient,
        margin=margin,
        incumbent=incumbent,
        candidate=candidate,
        incumbent_row=incumbent_row,
        candidate_row=candidate_row,
        composite_delta=composite_delta,
        cost_delta=cost_delta,
        judge_cost_delta=judge_cost_delta,
        reasons=[],
    )
