"""Tests for reviewer trial CLI (corpus-sanity cost display).

Verifies that per-diff and aggregate cost lines include the haiku matcher
cost (match_cost_usd), not just the reviewer-panel cost (cost_usd).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from orchestrator.config import default_price_table
from orchestrator.evals.reviewer_trial.__main__ import _select_original_corpus, cli
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest
from orchestrator.evals.reviewer_trial.mining import AuditReport
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import ScoringResult
from orchestrator.evals.reviewer_trial.variants import REVIEWER_REFRESH_VARIANTS


class TestSelectOriginalCorpus:
    """_select_original_corpus keeps the original hand-authored (non-mined)
    diffs and drops the mined ones (task 2476: refresh the original 15)."""

    def test_keeps_only_non_mined_diffs(self) -> None:
        manifest = CorpusManifest(
            version='2.0',
            split_seed='seed-xyz',
            diffs=[
                CorpusDiff(diff_id='syn_1', language='python', source='synthetic',
                           diff_text='', description='s1', ground_truth=[]),
                CorpusDiff(diff_id='syn_2', language='rust', source='synthetic',
                           diff_text='', description='s2', ground_truth=[]),
                CorpusDiff(diff_id='rw_1', language='typescript', source='real_world',
                           diff_text='', description='rw1', ground_truth=[]),
                CorpusDiff(diff_id='mined_1', language='python', source='mined',
                           diff_text='', description='m1', ground_truth=[]),
                CorpusDiff(diff_id='mined_2', language='python', source='mined',
                           diff_text='', description='m2', ground_truth=[]),
            ],
        )

        selected = _select_original_corpus(manifest)

        assert isinstance(selected, CorpusManifest)
        # Exactly the three non-mined diffs, diff_ids preserved.
        assert {d.diff_id for d in selected.diffs} == {'syn_1', 'syn_2', 'rw_1'}
        assert all(d.source in ('synthetic', 'real_world') for d in selected.diffs)
        assert not any(d.source == 'mined' for d in selected.diffs)
        # version/split_seed carried over from the source manifest.
        assert selected.version == '2.0'
        assert selected.split_seed == 'seed-xyz'


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


class TestRefreshCommand:
    """`refresh` (task 2476, eval-revival κ) runs the 1x Opus incumbent +
    Sonnet-5 + cross-family candidate set over just the original
    hand-authored (non-mined) diffs, and ranks them on quality (F1) AND
    cost."""

    def test_refresh_runs_original_corpus_and_ranks_by_f1_and_cost(
        self, tmp_path: Path,
    ) -> None:
        # Mixed manifest: two non-mined diffs + one mined diff. Only the
        # non-mined pair should ever reach run_trial/score_panel_run.
        manifest = CorpusManifest(
            version='2.0',
            split_seed='seed-abc',
            diffs=[
                CorpusDiff(diff_id='syn_1', language='python', source='synthetic',
                           diff_text='diff1', description='s1', ground_truth=[]),
                CorpusDiff(diff_id='rw_1', language='typescript', source='real_world',
                           diff_text='diff2', description='rw1', ground_truth=[]),
                CorpusDiff(diff_id='mined_1', language='python', source='mined',
                           diff_text='diff3', description='m1', ground_truth=[]),
            ],
        )
        non_mined_diff_ids = {'syn_1', 'rw_1'}

        # One fake PanelRunResult per (refresh variant x non-mined diff).
        panel_results = [
            PanelRunResult(
                variant_name=variant.name, diff_id=diff_id,
                reviews={}, total_cost_usd=1.0, wall_clock_ms=100,
            )
            for variant in REVIEWER_REFRESH_VARIANTS
            for diff_id in sorted(non_mined_diff_ids)
        ]

        # Distinct F1 (quality) and cost per variant -- Sonnet-5 "wins" on
        # quality, so a correct leaderboard must NOT just preserve
        # REVIEWER_REFRESH_VARIANTS order (which puts the opus incumbent
        # first).
        f1_by_variant = {
            'variant_a': 0.5,
            'variant_sonnet5_solo': 0.9,
            'variant_cross_family': 0.7,
        }
        cost_by_variant = {
            'variant_a': 2.0,
            'variant_sonnet5_solo': 1.0,
            'variant_cross_family': 1.5,
        }

        async def fake_score_panel_run(panel_result: PanelRunResult, diff: CorpusDiff) -> ScoringResult:
            f1 = f1_by_variant[panel_result.variant_name]
            cost = cost_by_variant[panel_result.variant_name]
            return ScoringResult(
                variant_name=panel_result.variant_name,
                diff_id=panel_result.diff_id,
                recall=f1, precision=f1, f1=f1, blocking_recall=f1,
                cost_usd=cost, match_cost_usd=0.0,
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
            ) as run_trial_mock,
            patch(
                'orchestrator.evals.reviewer_trial.scorer.score_panel_run',
                new_callable=AsyncMock,
                side_effect=fake_score_panel_run,
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ['refresh', '--report-dir', str(tmp_path)])

        assert result.exit_code == 0, (
            f'Command exited with code {result.exit_code}.\nOutput:\n{result.output}'
            + (f'\nException: {result.exception}' if result.exception else '')
        )

        # run_trial must run the refresh variant set over a corpus
        # containing ONLY the non-mined diffs, priced with the default
        # price table (so the cross-family candidate's cost is non-sentinel).
        run_trial_mock.assert_awaited_once()
        call_args = run_trial_mock.call_args
        called_variants = call_args.args[0]
        called_corpus = call_args.args[1]
        assert called_variants == REVIEWER_REFRESH_VARIANTS, (
            f'Expected run_trial to be called with REVIEWER_REFRESH_VARIANTS, got {called_variants!r}'
        )
        assert {d.diff_id for d in called_corpus.diffs} == non_mined_diff_ids, (
            f'Expected run_trial corpus to contain only {non_mined_diff_ids}, '
            f'got {[d.diff_id for d in called_corpus.diffs]!r}'
        )
        assert call_args.kwargs.get('prices') == default_price_table(), (
            f'Expected run_trial to be called with prices=default_price_table(), '
            f'got {call_args.kwargs.get("prices")!r}'
        )

        # The refreshed report is written to --report-dir.
        report_path = tmp_path / 'refresh_report.md'
        assert report_path.exists(), (
            f'Expected {report_path} to be written.\nOutput:\n{result.output}'
        )
        content = report_path.read_text()

        # Leaderboard rows ordered by descending mean F1 (quality): Sonnet-5
        # (0.9) > cross-family (0.7) > the opus incumbent (0.5).
        leaderboard = content.split('## Variant Leaderboard')[1].split('##')[0]
        idx_sonnet5 = leaderboard.index('variant_sonnet5_solo')
        idx_cross = leaderboard.index('variant_cross_family')
        idx_opus = leaderboard.index('variant_a')
        assert idx_sonnet5 < idx_cross < idx_opus, (
            'Expected leaderboard ordered variant_sonnet5_solo > variant_cross_family '
            f'> variant_a by descending F1.\n{leaderboard}'
        )

        # Each variant's row carries its own (distinct) cost figure (cost).
        assert '$2.00' in leaderboard, f'Expected sonnet5 cost $2.00 (2 diffs x $1.00).\n{leaderboard}'
        assert '$3.00' in leaderboard, f'Expected cross_family cost $3.00 (2 diffs x $1.50).\n{leaderboard}'
        assert '$4.00' in leaderboard, f'Expected variant_a cost $4.00 (2 diffs x $2.00).\n{leaderboard}'

