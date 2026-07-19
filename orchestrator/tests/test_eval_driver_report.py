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


# ---------------------------------------------------------------------------
# step-15/16 — build_methodology_report / format_methodology_report: the μ
# top-level artifact. It NESTS three λ build_composite_report sub-reports under
# stage keys ('ofat'/'matrix'/'confirm') + survivors + winner + the echoed
# price_table (inventing no new per-config schema), and renders byte-stably by
# reusing format_composite_table per stage under a survivors/winner header.
# ---------------------------------------------------------------------------

def _res(task_id: str, config: str, role: str, composite_score: float):
    from orchestrator.evals.runner import EvalResult

    return EvalResult(
        task_id, config, 'done',
        {
            'composite_score': composite_score,
            'tests_pass': True,
            'role_under_test': role,
            'cost_usd': 1.0,
            'workflow_duration_ms': 1000,
        },
        '/tmp/wt',
    )


class TestMethodologyReport:
    def test_build_nests_three_stages_plus_survivors_winner_price(self):
        from orchestrator.evals.report import build_methodology_report

        ofat = [
            _res('fix1', 'impl-a', 'implementer', 0.9),
            _res('fix1', 'arch-x', 'architect', 0.8),
        ]
        matrix = [_res('fix1', 'arch-x+impl-a', 'end_to_end', 0.85)]
        confirm = [_res('fix1', 'arch-x+impl-a', 'end_to_end', 0.88)]
        price_table = {'impl-a': {'implementer': {'input_per_1m': 3.0, 'output_per_1m': 15.0}}}
        survivors = {'implementer': ['impl-a'], 'architect': ['arch-x']}
        winner = 'arch-x+impl-a'

        report = build_methodology_report(
            ofat, matrix, confirm,
            price_table=price_table, survivors=survivors, winner=winner,
        )

        # Three build_composite_report sub-reports nested under the stage keys.
        assert set(report['stages']) == {'ofat', 'matrix', 'confirm'}
        for stage in ('ofat', 'matrix', 'confirm'):
            sub = report['stages'][stage]
            assert sub['aggregation'] == 'per_fixture_normalized_mean_ci'  # genuine λ report
            assert isinstance(sub['configs'], list)
        # ofat stage carries BOTH the implementer and the architect OFAT rows.
        ofat_configs = {c['config'] for c in report['stages']['ofat']['configs']}
        assert ofat_configs == {'impl-a', 'arch-x'}
        # survivors + winner + echoed price_table at the methodology top level.
        assert report['survivors'] == survivors
        assert report['winner'] == winner
        assert report['price_table'] == price_table

    def test_build_is_deterministic_modulo_generated_at(self):
        from orchestrator.evals.report import build_methodology_report

        def _mk():
            return build_methodology_report(
                [_res('fix1', 'impl-a', 'implementer', 0.9)], [], [],
                price_table={}, survivors={'implementer': ['impl-a']}, winner='w',
            )

        r1, r2 = _mk(), _mk()
        # Normalize the only wall-clock fields, then require full structural equality.
        for r in (r1, r2):
            r['generated_at'] = 'X'
            for sub in r['stages'].values():
                sub['generated_at'] = 'X'
        assert r1 == r2

    def test_format_renders_stages_header_and_price(self):
        from orchestrator.evals.report import (
            build_methodology_report,
            format_methodology_report,
        )

        report = build_methodology_report(
            [_res('fix1', 'impl-a', 'implementer', 0.9)],
            [_res('fix1', 'arch-x+impl-a', 'end_to_end', 0.85)],
            [_res('fix1', 'arch-x+impl-a', 'end_to_end', 0.88)],
            price_table={'impl-a': {'implementer': {'input_per_1m': 3.0, 'output_per_1m': 15.0}}},
            survivors={'implementer': ['impl-a'], 'architect': ['arch-x']},
            winner='arch-x+impl-a',
        )

        text = format_methodology_report(report)

        # Survivors/winner header.
        assert 'arch-x+impl-a' in text     # winner combo
        assert 'impl-a' in text            # survivor + a config row
        # All three stage sections labelled.
        assert 'ofat' in text
        assert 'matrix' in text
        assert 'confirm' in text
        # The price-table section is rendered.
        assert 'price table:' in text

    def test_format_is_byte_stable_across_wall_clock(self):
        from orchestrator.evals.report import (
            build_methodology_report,
            format_methodology_report,
        )

        def _mk():
            return build_methodology_report(
                [_res('fix1', 'impl-a', 'implementer', 0.9)],
                [_res('fix1', 'arch-x+impl-a', 'end_to_end', 0.85)],
                [],
                price_table={'impl-a': {'implementer': {'input_per_1m': 3.0, 'output_per_1m': 15.0}}},
                survivors={'implementer': ['impl-a']},
                winner='arch-x+impl-a',
            )

        # Two independently-built reports (distinct generated_at) render
        # byte-identically: format renders no wall-clock, sorts rows/sections,
        # and fixes float precision.
        assert format_methodology_report(_mk()) == format_methodology_report(_mk())
