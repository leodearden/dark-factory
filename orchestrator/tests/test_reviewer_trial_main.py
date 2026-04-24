"""Tests for reviewer trial CLI (corpus-sanity cost display).

Verifies that per-diff and aggregate cost lines include the haiku matcher
cost (match_cost_usd), not just the reviewer-panel cost (cost_usd).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from orchestrator.evals.reviewer_trial.__main__ import cli
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import ScoringResult


class TestCorpusSanityCostDisplay:
    """corpus-sanity per-diff table and totals must reflect panel + matcher cost."""

    def test_per_diff_and_total_include_match_cost(self) -> None:
        """Cost columns show panel+match combined, not panel-only.

        Before the fix (step 10):
          - d1 row shows ``$  1.00`` (panel only) → assertion fails
          - d2 row shows ``$  2.00`` (panel only) → assertion fails
          - Total line shows ``$3.00``            → assertion fails

        After the fix:
          - d1 row shows ``$  1.30`` (1.0 panel + 0.3 match) → passes
          - d2 row shows ``$  2.50`` (2.0 panel + 0.5 match) → passes
          - Total line shows ``$3.80`` (1.3 + 2.5)           → passes
        """
        # Two synthetic diffs so both contribute to threshold calculations
        # (mean_recall=0.8 > 0.6 and mean_blocking_recall=0.8 > 0.5 → exit 0)
        manifest = CorpusManifest(diffs=[
            CorpusDiff(
                diff_id='d1', language='python', source='synthetic',
                diff_text='', description='test diff 1', ground_truth=[],
            ),
            CorpusDiff(
                diff_id='d2', language='python', source='synthetic',
                diff_text='', description='test diff 2', ground_truth=[],
            ),
        ])

        panel_results = [
            PanelRunResult(
                variant_name='baseline', diff_id='d1',
                reviews={}, total_cost_usd=1.0, wall_clock_ms=100,
            ),
            PanelRunResult(
                variant_name='baseline', diff_id='d2',
                reviews={}, total_cost_usd=2.0, wall_clock_ms=100,
            ),
        ]

        # recall=0.8 / blocking_recall=0.8 so both threshold checks pass
        score_d1 = ScoringResult(
            variant_name='baseline', diff_id='d1',
            cost_usd=1.0, match_cost_usd=0.3,
            recall=0.8, precision=0.8, f1=0.8, blocking_recall=0.8,
        )
        score_d2 = ScoringResult(
            variant_name='baseline', diff_id='d2',
            cost_usd=2.0, match_cost_usd=0.5,
            recall=0.8, precision=0.8, f1=0.8, blocking_recall=0.8,
        )

        with (
            patch(
                'orchestrator.evals.reviewer_trial.__main__._load_corpus',
                return_value=manifest,
            ),
            patch(
                'orchestrator.evals.reviewer_trial.runner.run_trial',
                new_callable=AsyncMock,
                return_value=panel_results,
            ),
            patch(
                'orchestrator.evals.reviewer_trial.scorer.score_panel_run',
                new_callable=AsyncMock,
                side_effect=[score_d1, score_d2],
            ),
            patch(
                'orchestrator.evals.reviewer_trial.report.build_trial_report',
                return_value=MagicMock(),
            ),
            patch(
                'orchestrator.evals.reviewer_trial.report.save_report',
                return_value=Path('/tmp/sanity_report.json'),
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ['corpus-sanity'])

        assert result.exit_code == 0, (
            f'Command exited with code {result.exit_code}.\nOutput:\n{result.output}'
            + (f'\nException: {result.exception}' if result.exception else '')
        )

        # (a) per-diff row for d1: combined cost $1.30 (not panel-only $1.00)
        assert '$  1.30' in result.output, (
            f'Expected "$  1.30" in d1 cost column (panel 1.0 + match 0.3).\n'
            f'Output:\n{result.output}'
        )

        # (b) per-diff row for d2: combined cost $2.50 (not panel-only $2.00)
        assert '$  2.50' in result.output, (
            f'Expected "$  2.50" in d2 cost column (panel 2.0 + match 0.5).\n'
            f'Output:\n{result.output}'
        )

        # (c) aggregate "Total cost:" line: $3.80 = 1.30 + 2.50 (not panel-only $3.00)
        assert '$3.80' in result.output, (
            f'Expected "$3.80" in Total cost line (combined 1.3 + 2.5).\n'
            f'Output:\n{result.output}'
        )

