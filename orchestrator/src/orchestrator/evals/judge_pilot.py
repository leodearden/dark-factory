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
from typing import TYPE_CHECKING, Any

from .report import build_composite_report, format_composite_table

if TYPE_CHECKING:
    from .runner import EvalResult

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

    ``verdict='adopt'`` (``escalate=False``) iff all three hold. Every other
    outcome escalates with a structured ``reasons`` explanation:

    * a MISSING incumbent/candidate row → ``marginal`` (loud, no ``KeyError``);
    * a NOT-``sufficient`` composite CI (n<3 on either config) → ``marginal``
      (we cannot conclude anything from too few trials — checked before the
      non-inferiority verdict so an underpowered run never reads as a confident
      reject);
    * NOT ``non_inferior`` (and sufficient) → ``reject``;
    * ``non_inferior`` + sufficient but NOT ``cheaper`` → ``marginal``.
    """
    by_name = {str(r.get('config')): r for r in composite_report.get('configs', [])}
    incumbent_row = by_name.get(incumbent)
    candidate_row = by_name.get(candidate)

    # (d) Missing row → loud marginal, no KeyError, deltas undefined.
    if incumbent_row is None or candidate_row is None:
        missing = [
            name
            for name, row in ((incumbent, incumbent_row), (candidate, candidate_row))
            if row is None
        ]
        return JudgeAdoptionDecision(
            verdict='marginal',
            escalate=True,
            non_inferior=False,
            cheaper=False,
            sufficient=False,
            margin=margin,
            incumbent=incumbent,
            candidate=candidate,
            incumbent_row=incumbent_row,
            candidate_row=candidate_row,
            composite_delta=None,
            cost_delta=None,
            judge_cost_delta=None,
            reasons=[
                f'missing required config row(s): {", ".join(missing)} '
                f'absent from the composite report (configs present: '
                f'{", ".join(sorted(by_name)) or "none"}) — cannot decide',
            ],
        )

    inc_comp = _composite_mean(incumbent_row)
    cand_comp = _composite_mean(candidate_row)
    cand_lo = _composite_lo(candidate_row)
    inc_cost = _cost(incumbent_row)
    cand_cost = _cost(candidate_row)

    non_inferior = cand_lo >= inc_comp - margin
    cheaper = cand_cost < inc_cost
    cand_sufficient = _composite_sufficient(candidate_row)
    inc_sufficient = _composite_sufficient(incumbent_row)
    sufficient = cand_sufficient and inc_sufficient

    composite_delta = cand_comp - inc_comp
    cost_delta = cand_cost - inc_cost
    judge_cost_delta = _judge_cost(candidate_row) - _judge_cost(incumbent_row)
    threshold = inc_comp - margin

    reasons: list[str] = []
    if not sufficient:
        # (c) Too few trials on either config → cannot conclude; loud marginal.
        underpowered = [
            name
            for name, ok in ((candidate, cand_sufficient), (incumbent, inc_sufficient))
            if not ok
        ]
        verdict, escalate = 'marginal', True
        reasons.append(
            f'insufficient trials (n<3) for composite CI: {", ".join(underpowered)} '
            f'— need >=3 trials/cell to conclude; escalate for more trials',
        )
    elif not non_inferior:
        # (a) Confident regression beyond the margin → reject.
        verdict, escalate = 'reject', True
        reasons.append(
            f'not non-inferior: {candidate} composite CI lower bound {cand_lo:.4f} '
            f'< {incumbent} mean {inc_comp:.4f} - margin {margin:.4f} = {threshold:.4f}',
        )
    elif not cheaper:
        # (b) Non-inferior but no cost win → marginal.
        verdict, escalate = 'marginal', True
        reasons.append(
            f'non-inferior (CI lo {cand_lo:.4f} >= {threshold:.4f}) but NOT cheaper: '
            f'{candidate} cost_usd {cand_cost:.4f} >= {incumbent} cost_usd {inc_cost:.4f} '
            f'(delta {cost_delta:+.4f})',
        )
    else:
        verdict, escalate = 'adopt', False
        reasons.append(
            f'adopt: {candidate} non-inferior (CI lo {cand_lo:.4f} >= {threshold:.4f}) '
            f'and cheaper (cost {cand_cost:.4f} < {inc_cost:.4f}, delta {cost_delta:+.4f})',
        )

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
        reasons=reasons,
    )


def select_judge_results(results: list[EvalResult]) -> list[EvalResult]:
    """Keep only the judge-OFAT rows (``metrics['role_under_test'] == 'judge'``).

    The ο judge branch stamps ``role_under_test='judge'`` on each judge-candidate
    ``EvalResult``; every other row (implementer / architect / end_to_end) is
    dropped. Filtering BEFORE :func:`report.build_composite_report` matters for
    correctness, not just tidiness: a stray non-judge row sharing a fixture would
    otherwise shift that fixture's cost/latency normalization floor and distort
    the judge composites. Pure over *results*, order-preserving.
    """
    return [r for r in results if r.metrics.get('role_under_test') == 'judge']


def analyze_judge_ofat(
    results: list[EvalResult],
    *,
    incumbent: str = INCUMBENT_JUDGE,
    candidate: str = CANDIDATE_JUDGE,
    margin: float = DEFAULT_NON_INFERIORITY_MARGIN,
) -> tuple[JudgeAdoptionDecision, dict[str, Any]]:
    """Filter → composite → decide: the full judge-OFAT analysis over *results*.

    Filters to judge rows (:func:`select_judge_results`), builds the λ
    :func:`report.build_composite_report` over them, runs
    :func:`decide_judge_adoption`, and returns ``(decision, composite_report)`` so
    the caller can both act on the verdict and render the embedded table. Consumes
    ο's seam wholesale — invents no new per-config schema.
    """
    judge_results = select_judge_results(results)
    composite_report = build_composite_report(judge_results)
    decision = decide_judge_adoption(
        composite_report, incumbent=incumbent, candidate=candidate, margin=margin,
    )
    return decision, composite_report


def _fmt_delta(value: float | None) -> str:
    """A signed 4-dp delta cell, or ``'n/a'`` when the delta is undefined
    (a missing row leaves ``composite_delta`` / ``cost_delta`` ``None``)."""
    return 'n/a' if value is None else f'{value:+.4f}'


def _yesno(value: bool) -> str:
    return 'yes' if value else 'no'


def format_judge_ofat_report(
    decision: JudgeAdoptionDecision,
    composite_report: dict[str, Any],
) -> str:
    """Render a judge-OFAT pilot decision as a deterministic markdown report.

    A header (verdict + non_inferior/cheaper/sufficient booleans + margin +
    incumbent/candidate + composite/cost/judge-cost deltas), the structured
    ``reasons``, the EMBEDDED :func:`report.format_composite_table` data section
    (so the committed report is consistent with every other eval surface and
    stays byte-stable), and a decide-and-act footer: on ``adopt`` the recommended
    ``models.judge`` flip; on ``marginal``/``reject`` a keep-sonnet + escalate
    note. No wall-clock is rendered, so the same inputs render byte-identically.
    """
    d = decision
    lines: list[str] = [
        '# Judge-model OFAT pilot report',
        '',
        f'verdict: {d.verdict} (escalate: {_yesno(d.escalate)})',
        f'incumbent: {d.incumbent} | candidate: {d.candidate}',
        (
            f'non-inferior: {_yesno(d.non_inferior)} | cheaper: {_yesno(d.cheaper)} '
            f'| sufficient: {_yesno(d.sufficient)} | margin: {d.margin:.4f}'
        ),
        f'composite delta ({d.candidate} - {d.incumbent}): {_fmt_delta(d.composite_delta)}',
        f'cost delta ({d.candidate} - {d.incumbent}): {_fmt_delta(d.cost_delta)}',
        f'judge cost delta ({d.candidate} - {d.incumbent}): {_fmt_delta(d.judge_cost_delta)}',
        '',
        'reasons:',
    ]
    lines.extend(f'  - {reason}' for reason in d.reasons)
    lines.append('')
    lines.append(format_composite_table(composite_report))
    lines.append('')
    if d.verdict == 'adopt':
        lines.append(
            f'RECOMMENDATION: adopt — flip models.judge: sonnet -> haiku in '
            f'orchestrator.yaml, reload_config, watch the δ/2534 rollup a fixed '
            f'window for haiku judge rows, then persist to defaults.yaml.',
        )
    else:
        lines.append(
            f'RECOMMENDATION: keep models.judge=sonnet and escalate with this '
            f'report (verdict={d.verdict}); attach it to the escalation.',
        )
    return '\n'.join(lines)
