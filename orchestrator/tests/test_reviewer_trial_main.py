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
from orchestrator.evals.reviewer_trial.mining import AuditReport
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


class TestMineMinDiffsThreading:
    """`mine --min-diffs N` must thread N into the post-run audit call."""

    def test_mine_threads_min_diffs_into_post_run_audit(self) -> None:
        """The CLI-supplied --min-diffs floor must reach audit_corpus's
        post-run call, not a hardcoded default.

        Hermetic: the candidate pool is forced empty (mine_fn_candidates ->
        [], mine_escalation_refs -> {}) so the pipeline reaches the audit
        call at the end of `mine` without touching the real runs.db, git,
        or a frontier LLM. The synthetic corpus's diffs are all
        source='mined' so the hand-authored backfill loop skips them and
        `_resave()` (which would rewrite the real committed corpus) is
        never invoked. `audit_corpus` is mocked so its return value doesn't
        depend on the real (bypassed) checks -- we only care what it was
        CALLED with.

        Before the fix (step 20): line ~802 hardcodes
        ``audit_corpus(manifest, log, min_diffs=50)``, silently ignoring
        the CLI flag, so the actual call carries 50 (not 80) and this
        assertion fails.
        """
        manifest = CorpusManifest(diffs=[
            CorpusDiff(
                diff_id='mined_1', language='python', source='mined',
                diff_text='', description='synthetic mined diff 1', ground_truth=[],
                split='test',
            ),
            CorpusDiff(
                diff_id='mined_2', language='python', source='mined',
                diff_text='', description='synthetic mined diff 2', ground_truth=[],
                split='train',
            ),
        ])
        stub_report = AuditReport(ok=True, diff_count=len(manifest.diffs), failures=[])

        with (
            patch(
                'orchestrator.evals.reviewer_trial.__main__._load_corpus',
                return_value=manifest,
            ),
            patch(
                'orchestrator.evals.reviewer_trial.mining.mine_fn_candidates',
                return_value=[],
            ),
            patch(
                'orchestrator.evals.reviewer_trial.mining.mine_escalation_refs',
                return_value={},
            ),
            patch(
                'orchestrator.evals.reviewer_trial.__main__._fetch_titles',
                return_value={},
            ),
            patch(
                'orchestrator.evals.reviewer_trial.mining.audit_corpus',
                return_value=stub_report,
            ) as audit_spy,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ['mine', '--min-diffs', '80'])

        assert result.exit_code == 0, (
            f'Command exited with code {result.exit_code}.\nOutput:\n{result.output}'
            + (f'\nException: {result.exception}' if result.exception else '')
        )

        audit_spy.assert_called_once()
        call_args = audit_spy.call_args
        if 'min_diffs' in call_args.kwargs:
            actual_min_diffs = call_args.kwargs['min_diffs']
        else:
            # Accept a positional form too: audit_corpus(manifest, log, min_diffs).
            actual_min_diffs = call_args.args[2]

        assert actual_min_diffs == 80, (
            "mine --min-diffs 80 should thread 80 into audit_corpus's min_diffs kwarg, "
            f'got {actual_min_diffs!r} (call_args={call_args!r}).\n'
            f'Output:\n{result.output}'
        )

