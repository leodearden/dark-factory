"""Routing κ judge-OFAT pilot layer in evals/judge_pilot.py (task 2815).

Pure-function tests over λ's ``build_composite_report``-shaped dict — the same
hermetic ``_row``/``_report`` style as ``test_eval_driver_report.py``. The pilot
module is a CONSUME-ONLY analysis surface over the ο judge-OFAT seam (task 2825):
it imports ``report.build_composite_report`` / ``format_composite_table`` and
``runner.load_results`` / ``run_ofat_stage`` and edits none of them (decision 11).

Every ``decide_judge_adoption`` / ``format_judge_ofat_report`` test is hermetic:
it constructs the composite-report dict directly with hand-set composite means +
``ci95`` sub-dicts, so the decision/render logic is exercised deterministically
without any real EvalResult, I/O, LLM, or the experiment's real numbers (the
non-inferiority margin is an exposed policy parameter, never an asserted bound).
"""

from __future__ import annotations

from typing import Any

import pytest


def _row(
    config: str,
    *,
    composite: float,
    cost_usd: float,
    lo: float | None = None,
    hi: float | None = None,
    n: int = 3,
    sufficient: bool | None = None,
    role: str = 'judge',
    judge_cost: float = 0.0,
) -> dict[str, Any]:
    """A full-fidelity ``build_composite_report`` ``configs`` row.

    Carries every field ``decide_judge_adoption`` / ``format_composite_table``
    read: the ``composite``/``cost_usd`` means, the ``ci95.composite`` sub-dict
    (``{mean, lo, hi, n, sufficient}`` — the non-inferiority + sufficiency inputs),
    and the ``judge.cost_usd`` subset. ``lo``/``hi`` default to the point estimate
    and ``sufficient`` to ``n >= 3`` (report.mean_ci95's own rule).
    """
    if lo is None:
        lo = composite
    if hi is None:
        hi = composite
    if sufficient is None:
        sufficient = n >= 3
    return {
        'config': config,
        'role_under_test': role,
        'composite': composite,
        'quality': composite,
        'tests_pass_rate': 1.0,
        'cost_usd': cost_usd,
        'cost_source': 'price_table',
        'latency_secs': 10.0,
        'trials': n,
        'fixtures': n,
        'ci95': {
            'composite': {'mean': composite, 'lo': lo, 'hi': hi, 'n': n, 'sufficient': sufficient},
            'cost': {'mean': cost_usd, 'lo': cost_usd, 'hi': cost_usd, 'n': n, 'sufficient': sufficient},
            'latency': {'mean': 10.0, 'lo': 10.0, 'hi': 10.0, 'n': n, 'sufficient': sufficient},
        },
        'judge': {'invocations': n, 'cost_usd': judge_cost},
        'recovery_score': None,
        'plan_quality': None,
    }


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'generated_at': '2026-01-01T00:00:00+00:00',
        'aggregation': 'per_fixture_normalized_mean_ci',
        'price_table': {},
        'configs': rows,
    }


# ---------------------------------------------------------------------------
# step-1/2 — decide_judge_adoption ADOPT path: the candidate (judge-haiku) is
# non-inferior (its composite CI lower bound is within `margin` of the incumbent
# judge-sonnet composite mean) AND cheaper AND both composite CIs are sufficient
# (n>=3) → verdict 'adopt', no escalation.
# ---------------------------------------------------------------------------

class TestDecideAdopt:
    def test_adopt_when_non_inferior_and_cheaper_and_sufficient(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3, judge_cost=0.40),
            _row('judge-haiku', composite=0.89, cost_usd=0.10, lo=0.88, hi=0.90, n=3, judge_cost=0.05),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        # candidate lo 0.88 >= incumbent mean 0.90 - margin 0.05 == 0.85 → non-inferior;
        # candidate cost 0.10 < incumbent 0.50 → cheaper; both CIs n>=3 → sufficient.
        assert decision.verdict == 'adopt'
        assert decision.escalate is False
        assert decision.non_inferior is True
        assert decision.cheaper is True
        # The two rows are exposed for the report renderer / operator.
        assert decision.incumbent_row['config'] == 'judge-sonnet'
        assert decision.candidate_row['config'] == 'judge-haiku'
        assert decision.incumbent == 'judge-sonnet'
        assert decision.candidate == 'judge-haiku'
        # Deltas are candidate-minus-incumbent (negative == candidate is better/cheaper).
        assert decision.composite_delta == pytest.approx(0.89 - 0.90)
        assert decision.cost_delta == pytest.approx(0.10 - 0.50)
        assert decision.judge_cost_delta == pytest.approx(0.05 - 0.40)

    def test_default_margin_is_exposed_policy_parameter(self):
        from orchestrator.evals.judge_pilot import (
            DEFAULT_NON_INFERIORITY_MARGIN,
            decide_judge_adoption,
        )

        # The margin is a documented policy knob with a default, not an
        # empirically-asserted bound: calling without `margin` uses the default.
        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
            _row('judge-haiku', composite=0.89, cost_usd=0.10, lo=0.88, hi=0.90, n=3),
        ])
        decision = decide_judge_adoption(report)
        assert decision.margin == DEFAULT_NON_INFERIORITY_MARGIN
        assert isinstance(DEFAULT_NON_INFERIORITY_MARGIN, float)


# ---------------------------------------------------------------------------
# step-3/4 — the non-adopt branches. Insufficient data (n<3) and a missing
# config row map to a LOUD `marginal` (escalate) with a structured reason — not
# a silent pass or an unhandled KeyError (loud-over-silent norm, decision 4).
# ---------------------------------------------------------------------------

class TestDecideNonAdopt:
    def test_reject_when_candidate_composite_ci_lo_below_margin(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        # candidate lo 0.70 << incumbent mean 0.90 - margin 0.05 == 0.85 → NOT
        # non-inferior; both CIs sufficient (n=3) so we can confidently reject.
        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
            _row('judge-haiku', composite=0.72, cost_usd=0.10, lo=0.70, hi=0.74, n=3),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        assert decision.verdict == 'reject'
        assert decision.escalate is True
        assert decision.non_inferior is False
        # Human-readable reason explains the non-inferiority failure.
        assert decision.reasons
        assert any('inferior' in r.lower() for r in decision.reasons)

    def test_marginal_when_non_inferior_but_not_cheaper(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        # non-inferior (lo 0.88 >= 0.85) and sufficient, but candidate is MORE
        # expensive (0.60 >= 0.50) → marginal + escalate, not adopt.
        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
            _row('judge-haiku', composite=0.89, cost_usd=0.60, lo=0.88, hi=0.90, n=3),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        assert decision.verdict == 'marginal'
        assert decision.escalate is True
        assert decision.non_inferior is True
        assert decision.cheaper is False
        assert decision.reasons
        assert any(('cheaper' in r.lower() or 'cost' in r.lower()) for r in decision.reasons)

    def test_marginal_when_ci_insufficient_n_below_3(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        # Candidate would otherwise adopt (non-inferior + cheaper) but only n=2
        # trials → composite CI not `sufficient` → LOUD marginal + escalate.
        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
            _row('judge-haiku', composite=0.89, cost_usd=0.10, lo=0.87, hi=0.91, n=2, sufficient=False),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        assert decision.verdict == 'marginal'
        assert decision.escalate is True
        assert decision.sufficient is False
        assert decision.reasons
        assert any(
            ('insufficient' in r.lower() or 'trials' in r.lower() or 'n<3' in r.lower())
            for r in decision.reasons
        )

    def test_marginal_when_a_required_row_is_missing(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        # candidate row absent from the report → no KeyError; a loud structured
        # marginal verdict naming the missing config instead.
        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        assert decision.verdict == 'marginal'
        assert decision.escalate is True
        assert decision.candidate_row is None
        assert decision.reasons
        assert any(
            ('missing' in r.lower() or 'judge-haiku' in r.lower() or 'absent' in r.lower())
            for r in decision.reasons
        )

    def test_marginal_when_incumbent_row_is_missing(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        # Symmetric: incumbent absent → loud marginal, no KeyError.
        report = _report([
            _row('judge-haiku', composite=0.89, cost_usd=0.10, lo=0.88, hi=0.90, n=3),
        ])

        decision = decide_judge_adoption(report, margin=0.05)

        assert decision.verdict == 'marginal'
        assert decision.escalate is True
        assert decision.incumbent_row is None
        assert decision.reasons


# ---------------------------------------------------------------------------
# step-5/6 — format_judge_ofat_report: a deterministic markdown report that
# carries the verdict + deltas, EMBEDS report.format_composite_table verbatim
# (so the committed trial report stays consistent with every other eval surface
# and byte-stable), and renders the decide-and-act footer (the models.judge flip
# on adopt; keep-sonnet + escalate otherwise).
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _adopt(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3, judge_cost=0.40),
            _row('judge-haiku', composite=0.89, cost_usd=0.10, lo=0.88, hi=0.90, n=3, judge_cost=0.05),
        ])
        return decide_judge_adoption(report, margin=0.05), report

    def _reject(self):
        from orchestrator.evals.judge_pilot import decide_judge_adoption

        report = _report([
            _row('judge-sonnet', composite=0.90, cost_usd=0.50, lo=0.88, hi=0.92, n=3),
            _row('judge-haiku', composite=0.72, cost_usd=0.10, lo=0.70, hi=0.74, n=3),
        ])
        return decide_judge_adoption(report, margin=0.05), report

    def test_adopt_report_has_verdict_deltas_table_and_flip_line(self):
        from orchestrator.evals.judge_pilot import format_judge_ofat_report
        from orchestrator.evals.report import format_composite_table

        decision, report = self._adopt()
        text = format_judge_ofat_report(decision, report)

        # Verdict + non_inferior/cheaper booleans + margin.
        assert 'adopt' in text
        assert 'margin' in text.lower()
        # Composite delta and cost delta are surfaced (candidate - incumbent).
        assert 'composite delta' in text.lower()
        assert 'cost delta' in text.lower()
        assert f'{decision.composite_delta:+.4f}' in text
        assert f'{decision.cost_delta:+.4f}' in text
        # The composite table is embedded verbatim.
        assert format_composite_table(report) in text
        # On adopt: the recommended flip line.
        assert 'models.judge: sonnet -> haiku' in text

    def test_reject_report_renders_keep_sonnet_and_no_flip(self):
        from orchestrator.evals.judge_pilot import format_judge_ofat_report
        from orchestrator.evals.report import format_composite_table

        decision, report = self._reject()
        text = format_judge_ofat_report(decision, report)

        assert 'reject' in text
        # Keep-sonnet + escalate note instead of the flip.
        assert 'keep models.judge=sonnet' in text
        assert 'escalat' in text.lower()
        # The adopt flip recommendation must NOT be rendered on a reject.
        assert 'models.judge: sonnet -> haiku' not in text
        # Table still embedded.
        assert format_composite_table(report) in text

    def test_report_is_byte_deterministic(self):
        from orchestrator.evals.judge_pilot import format_judge_ofat_report

        decision, report = self._adopt()
        # No wall-clock in the rendered body: same inputs → byte-identical output.
        assert format_judge_ofat_report(decision, report) == format_judge_ofat_report(decision, report)
