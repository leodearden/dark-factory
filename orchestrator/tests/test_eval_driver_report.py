"""μ methodology report layer in evals/report.py (task 2478).

Pure-function tests over λ's ``build_composite_report``-shaped dict:
  step-13/14  select_survivors (top-K config names per role_under_test)
  step-15/16  build_methodology_report / format_methodology_report (nested stages)

Every test is hermetic: it constructs the composite-report dict directly (the
only fields select_survivors consumes are ``configs[].config`` /
``role_under_test`` / ``composite``) — no EvalResult, no I/O, no LLM.
"""

from __future__ import annotations

from typing import Any


def _row(config: str, role: str, composite: float) -> dict[str, Any]:
    """A minimal build_composite_report ``configs`` row (the fields ranking reads)."""
    return {'config': config, 'role_under_test': role, 'composite': composite}


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'generated_at': '2026-01-01T00:00:00+00:00',
        'aggregation': 'per_fixture_normalized_mean_ci',
        'price_table': {},
        'configs': rows,
    }


# ---------------------------------------------------------------------------
# step-13/14 — select_survivors: the top-K config names per role_under_test,
# ranked by DESCENDING composite mean, config-name tiebreak. The OFAT screen's
# survivor gate: it feeds run_matrix_stage the winning architects × implementers.
# ---------------------------------------------------------------------------

class TestSelectSurvivors:
    def test_top_k_per_role_ranked_by_descending_composite(self):
        from orchestrator.evals.report import select_survivors

        report = _report([
            _row('impl-a', 'implementer', 0.90),
            _row('impl-b', 'implementer', 0.50),
            _row('impl-c', 'implementer', 0.70),
            _row('arch-x', 'architect', 0.80),
            _row('arch-y', 'architect', 0.60),
        ])

        survivors = select_survivors(report, top_k=2, roles=['implementer', 'architect'])

        # Top-2 per role by descending composite: impl a(0.9) > c(0.7) > b(0.5).
        assert survivors == {
            'implementer': ['impl-a', 'impl-c'],
            'architect': ['arch-x', 'arch-y'],
        }

    def test_ties_broken_deterministically_by_config_name(self):
        from orchestrator.evals.report import select_survivors

        # Equal composites → deterministic ascending config-name tiebreak, so the
        # alphabetically-first name outranks. Rows given out of order to prove the
        # ranking (not input order) decides.
        report = _report([
            _row('zeta', 'implementer', 0.50),
            _row('alpha', 'implementer', 0.50),
            _row('mid', 'implementer', 0.50),
        ])

        assert select_survivors(report, top_k=1, roles=['implementer']) == {
            'implementer': ['alpha'],
        }
        assert select_survivors(report, top_k=2, roles=['implementer']) == {
            'implementer': ['alpha', 'mid'],
        }

    def test_fewer_than_k_rows_returns_all_present(self):
        from orchestrator.evals.report import select_survivors

        report = _report([_row('impl-a', 'implementer', 0.9)])

        # top_k exceeds the available rows → all present, no padding/error.
        assert select_survivors(report, top_k=3, roles=['implementer']) == {
            'implementer': ['impl-a'],
        }

    def test_role_with_no_rows_returns_empty_list(self):
        from orchestrator.evals.report import select_survivors

        report = _report([_row('impl-a', 'implementer', 0.9)])

        # A requested role with no matching rows returns [] (not KeyError/missing).
        survivors = select_survivors(report, top_k=2, roles=['implementer', 'reviewer'])
        assert survivors['reviewer'] == []
        assert survivors['implementer'] == ['impl-a']

    def test_only_requested_roles_are_returned(self):
        from orchestrator.evals.report import select_survivors

        report = _report([
            _row('impl-a', 'implementer', 0.9),
            _row('arch-x', 'architect', 0.8),
        ])

        # architect rows exist but were not requested → excluded from the result.
        survivors = select_survivors(report, top_k=2, roles=['implementer'])
        assert set(survivors) == {'implementer'}
        assert survivors['implementer'] == ['impl-a']
