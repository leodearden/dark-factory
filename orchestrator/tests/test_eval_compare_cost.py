"""Tests for compare.format_comparison_markdown P5 cost provenance (task 2477 λ).

The per-task metric summary must state, per model, WHICH cost source produced
the displayed cost (Invariant P5: cost_source ∈ price_table|cli|unpriced_proxy)
and a latency value in seconds — rather than an unlabeled CLI cost.
"""

from __future__ import annotations


def _report_with_metrics(metrics_a, metrics_b):
    from orchestrator.evals.compare import ComparisonReport, TaskAssessment

    assessment = TaskAssessment(
        task_id='t1',
        task_name='Task One',
        winner='A',
        confidence=0.8,
        summary='A did better.',
        metrics_a=metrics_a,
        metrics_b=metrics_b,
    )
    return ComparisonReport(
        group_a_name='grpA',
        group_b_name='grpB',
        group_a_configs=['cfgA'],
        group_b_configs=['cfgB'],
        assessments=[assessment],
    )


def _metric_line(out, model_name):
    """The per-model metric summary line (the one carrying the score)."""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith(f'{model_name}:') and 'score' in stripped:
            return stripped
    raise AssertionError(f'no metric line for {model_name!r} in:\n{out}')


class TestCompareCostProvenance:
    def test_cost_source_and_latency_surfaced_per_model(self):
        from orchestrator.evals.compare import format_comparison_markdown

        metrics_a = {
            'composite_score': 0.9, 'cost_usd': 2.0, 'cost_source': 'price_table',
            'workflow_duration_ms': 3000, 'iterations': 1, 'debug_cycles': 0,
            'lines_changed': 10,
        }
        metrics_b = {
            'composite_score': 0.7, 'cost_usd': 5.0,
            'cost_source': 'unpriced_proxy', 'workflow_duration_ms': 12000,
            'iterations': 2, 'debug_cycles': 1, 'lines_changed': 20,
        }
        out = format_comparison_markdown(_report_with_metrics(metrics_a, metrics_b))

        a_line = _metric_line(out, 'grpA')
        b_line = _metric_line(out, 'grpB')

        # P5 cost-source label appears, per model, on the same line as the cost.
        assert 'price_table' in a_line
        assert 'unpriced_proxy' in b_line
        # Latency (seconds) appears per model: 3000ms → 3.0s, 12000ms → 12.0s.
        assert '3.0s' in a_line
        assert '12.0s' in b_line
        # The existing cost figure is still shown.
        assert '$2.00' in a_line
        assert '$5.00' in b_line

    def test_absent_cost_source_defaults_to_cli(self):
        from orchestrator.evals.compare import format_comparison_markdown

        # A pre-P5 report has no cost_source key: it must still render, labeled
        # as the trustworthy native-cloud default 'cli' (backward compat).
        metrics = {
            'composite_score': 0.5, 'cost_usd': 1.0,
            'workflow_duration_ms': 1000, 'iterations': 0,
            'debug_cycles': 0, 'lines_changed': 0,
        }
        out = format_comparison_markdown(_report_with_metrics(metrics, metrics))
        a_line = _metric_line(out, 'grpA')
        assert 'cli' in a_line
        assert '1.0s' in a_line
