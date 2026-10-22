"""Tests for reviewer trial CLI (corpus-sanity cost display).

Verifies that per-diff and aggregate cost lines include the haiku matcher
cost (match_cost_usd), not just the reviewer-panel cost (cost_usd).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _eval_refusal import answered, refused_agent_result
from click.testing import CliRunner
from shared.cli_invoke import AgentResult

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.config import default_price_table
from orchestrator.evals.measurement import MeasurementStage, Unmeasured, UnmeasuredCause
from orchestrator.evals.reviewer_trial.__main__ import _select_original_corpus, cli
from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    GroundTruthIssue,
)
from orchestrator.evals.reviewer_trial.mining import AuditReport
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.scorer import ScoringResult
from orchestrator.evals.reviewer_trial.variants import (
    FABLE_EVIDENCE_VARIANTS,
    REVIEWER_REFRESH_VARIANTS,
)


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

    def test_mine_threads_min_diffs_into_post_run_audit(self, tmp_path: Path) -> None:
        """The CLI-supplied --min-diffs floor must reach audit_corpus's
        post-run call, not a hardcoded default.

        Hermetic: the candidate pool is forced empty (mine_fn_candidates ->
        [], mine_escalation_refs -> {}) so the pipeline reaches the audit
        call at the end of `mine` without touching the real runs.db, git,
        or a frontier LLM. ``--runs-db`` points at an empty file under
        ``tmp_path`` so the command's is-file guard passes on any host (the
        live runs.db is gitignored and exists only on the main checkout's
        host); the mocked miners never open it. The synthetic corpus's diffs are
        all source='mined' so the hand-authored backfill loop skips them and
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
        runs_db = tmp_path / 'runs.db'
        runs_db.touch()

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
            result = runner.invoke(
                cli, ['mine', '--min-diffs', '80', '--runs-db', str(runs_db)],
            )

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



_ARM = FABLE_EVIDENCE_VARIANTS[0].name
_ISSUE = {'location': 'a.py:1', 'category': 'bug', 'severity': 'blocking', 'description': 'x'}


def _gt(gt_id: str) -> GroundTruthIssue:
    return GroundTruthIssue(
        id=gt_id, location='a.py:1', category='bug', severity='blocking',
        description='a bug', mutation_type='logic',
    )


class TestCampaignUnmeasured:
    """A refused matcher cell is counted, never averaged as a zero, and never cached."""

    def _invoke(self, tmp_path: Path, prompts: list[str]):
        manifest = CorpusManifest(diffs=[
            CorpusDiff(diff_id='d_ok', language='python', source='synthetic',
                       diff_text='OK_MARKER', description='ok', ground_truth=[_gt('ok_1')]),
            CorpusDiff(diff_id='d_refused', language='python', source='synthetic',
                       diff_text='REFUSED_MARKER', description='refused', ground_truth=[_gt('ref_1')]),
        ])
        panel_results = [
            PanelRunResult(
                variant_name=_ARM, diff_id=diff_id, total_cost_usd=cost, wall_clock_ms=100,
                reviews={'r1': {'verdict': 'NEEDS_CHANGES', 'issues': [_ISSUE]}},
            )
            for diff_id, cost in (('d_ok', 1.0), ('d_refused', 2.0))
        ]

        async def matcher(**kwargs: object) -> AgentResult:
            prompt = str(kwargs['prompt'])
            prompts.append(prompt)
            if 'REFUSED_MARKER' in prompt:
                return refused_agent_result()
            return answered({'matches': [{
                'reviewer_issue_index': 0, 'ground_truth_id': 'ok_1',
                'confidence': 0.9, 'reasoning': 'r',
            }]}, cost_usd=0.1)

        with (
            patch('orchestrator.evals.reviewer_trial.__main__._load_corpus', return_value=manifest),
            patch('orchestrator.evals.reviewer_trial.runner.run_trial',
                  new_callable=AsyncMock, return_value=panel_results),
            patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', matcher),
        ):
            return CliRunner().invoke(cli, [
                'campaign', '--results-dir', str(tmp_path / 'camp'), '--trials', '1',
                '--split', 'all', '--no-pool', '--arms', _ARM,
            ])

    def test_a_refused_matcher_cell_is_counted_excluded_and_not_cached(self, tmp_path: Path) -> None:
        prompts: list[str] = []

        result = self._invoke(tmp_path, prompts)

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        assert f'`{_ARM}` unmeasured: 1' in result.output
        refused_line = next(line for line in result.output.splitlines() if 'd_refused' in line)
        assert 'UNMEASURED' in refused_line and 'matcher:cap_hit' in refused_line, refused_line
        report = json.loads((tmp_path / 'camp' / 'campaign_report.json').read_text())
        arm = next(a for a in report['arms'] if a['name'] == _ARM)
        assert arm['unmeasured'] == 1
        assert arm['reviews'] == 1
        assert arm['mean_f1'] == 1.0
        scores = tmp_path / 'camp' / 'trial-1' / 'scores'
        assert (scores / f'{_ARM}__d_ok.json').exists()
        assert not (scores / f'{_ARM}__d_refused.json').exists()
        total_line = next(
            line for line in result.output.splitlines()
            if line.startswith('Total campaign cost (panels + matcher)')
        )
        assert '$3.41' in total_line, total_line  # 1.0 + 0.1 measured, 2.0 + 0.31 refused

    def test_a_rerun_re_measures_the_refused_cell(self, tmp_path: Path) -> None:
        prompts: list[str] = []

        self._invoke(tmp_path, prompts)
        rerun = self._invoke(tmp_path, prompts)

        assert rerun.exit_code == 0, f'{rerun.output}\n{rerun.exception!r}'
        assert sum('REFUSED_MARKER' in p for p in prompts) == 2
        assert sum('OK_MARKER' in p for p in prompts) == 1


class TestCampaignBriefing:
    """`campaign --briefing` crosses the chosen arms with each reviewer briefing generation."""

    def _invoke(self, tmp_path: Path, *options: str):
        manifest = CorpusManifest(diffs=[
            CorpusDiff(diff_id='d1', language='python', source='synthetic',
                       diff_text='d', description='d1', ground_truth=[_gt('g1')]),
        ])
        run_trial = AsyncMock(side_effect=lambda variants, corpus, **_kw: [
            PanelRunResult(variant_name=v.name, diff_id='d1', total_cost_usd=1.0) for v in variants
        ])
        score = AsyncMock(side_effect=lambda result, diff, **_kw: ScoringResult(
            variant_name=result.variant_name, diff_id=result.diff_id, cost_usd=1.0,
            recall=1.0, precision=1.0, f1=1.0, blocking_recall=1.0,
        ))
        with (
            patch('orchestrator.evals.reviewer_trial.__main__._load_corpus', return_value=manifest),
            patch('orchestrator.evals.reviewer_trial.runner.run_trial', run_trial),
            patch('orchestrator.evals.reviewer_trial.scorer.score_panel_run', score),
        ):
            result = CliRunner().invoke(cli, [
                'campaign', '--results-dir', str(tmp_path / 'camp'), '--trials', '1',
                '--split', 'all', '--no-pool', *options,
            ])
        return result, run_trial

    def test_d2_adds_a_context_complete_arm_after_each_diff_only_arm(self, tmp_path: Path) -> None:
        result, run_trial = self._invoke(
            tmp_path, '--arms', 'variant_a', '--briefing', 'diff-only,d2',
        )

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        arms = run_trial.call_args.args[0]
        assert [arm.name for arm in arms] == ['variant_a', 'variant_a_d2']
        assert [arm.briefing for arm in arms] == [
            ReviewBriefing.DIFF_ONLY, ReviewBriefing.CONTEXT_COMPLETE,
        ]

    def test_d2_alone_runs_only_each_arms_context_complete_arm(self, tmp_path: Path) -> None:
        result, run_trial = self._invoke(
            tmp_path, '--arms', 'variant_a,variant_fable51_solo,variant_a_xhigh', '--briefing', 'd2',
        )

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        arms = run_trial.call_args.args[0]
        assert [arm.name for arm in arms] == [
            'variant_a_d2', 'variant_fable51_solo_d2', 'variant_a_xhigh_d2',
        ]
        assert {arm.briefing for arm in arms} == {ReviewBriefing.CONTEXT_COMPLETE}

    @pytest.mark.parametrize(('briefings', 'incumbent', 'candidates'), [
        ('diff-only', 'variant_a', {'variant_a_xhigh'}),
        ('diff-only,d2', 'variant_a', {'variant_a_d2', 'variant_a_xhigh', 'variant_a_xhigh_d2'}),
        ('d2', 'variant_a_d2', {'variant_a_xhigh_d2'}),
    ])
    def test_the_report_pairs_every_arm_against_the_first_arms_first_generation(
        self, tmp_path: Path, briefings: str, incumbent: str, candidates: set[str],
    ) -> None:
        result, _ = self._invoke(
            tmp_path, '--arms', 'variant_a,variant_a_xhigh', '--briefing', briefings,
        )

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        report = json.loads((tmp_path / 'camp' / 'campaign_report.json').read_text())
        assert report['decisions']['incumbent'] == incumbent
        assert report['arms'][0]['name'] == incumbent
        assert set(report['decisions']['strict_f1_improvement']) == candidates
        md = (tmp_path / 'camp' / 'campaign_report.md').read_text()
        assert f'Incumbent: `{incumbent}`' in md

    def test_without_briefing_only_diff_only_arms_run(self, tmp_path: Path) -> None:
        result, run_trial = self._invoke(tmp_path, '--arms', 'variant_a,variant_a_xhigh')

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        arms = run_trial.call_args.args[0]
        assert [arm.name for arm in arms] == ['variant_a', 'variant_a_xhigh']
        assert {arm.briefing for arm in arms} == {ReviewBriefing.DIFF_ONLY}

    def test_an_unknown_briefing_is_a_usage_error_naming_the_valid_values(
        self, tmp_path: Path,
    ) -> None:
        result, run_trial = self._invoke(
            tmp_path, '--arms', 'variant_a', '--briefing', 'diff-only,bogus',
        )

        assert result.exit_code == 2, f'{result.output}\n{result.exception!r}'
        assert 'bogus' in result.output
        assert 'diff-only' in result.output and 'd2' in result.output
        run_trial.assert_not_called()

    @pytest.mark.parametrize('briefings', ['diff-only,d2,d2', 'diff-only,diff-only', 'd2,d2'])
    def test_a_repeated_briefing_is_a_usage_error_naming_it(
        self, tmp_path: Path, briefings: str,
    ) -> None:
        result, run_trial = self._invoke(tmp_path, '--arms', 'variant_a', '--briefing', briefings)

        assert result.exit_code == 2, f'{result.output}\n{result.exception!r}'
        repeated = briefings.rsplit(',', 1)[1]
        assert f'repeated: {repeated}' in result.output
        run_trial.assert_not_called()

    @pytest.mark.parametrize('briefings', ['d2,diff-only'])
    def test_d2_before_diff_only_is_a_usage_error(
        self, tmp_path: Path, briefings: str,
    ) -> None:
        result, run_trial = self._invoke(tmp_path, '--arms', 'variant_a', '--briefing', briefings)

        assert result.exit_code == 2, f'{result.output}\n{result.exception!r}'
        assert 'diff-only' in result.output and 'first' in result.output
        run_trial.assert_not_called()


def _unmeasured_score(variant: str, diff_id: str, cost_usd: float = 2.0,
                      match_cost_usd: float = 0.31) -> ScoringResult:
    return ScoringResult(
        variant_name=variant, diff_id=diff_id, cost_usd=cost_usd, match_cost_usd=match_cost_usd,
        unmeasured=Unmeasured(
            MeasurementStage.MATCHER, UnmeasuredCause.CAP_HIT, 'session limit', cost_usd=match_cost_usd,
        ),
    )


class TestCorpusSanityUnmeasured:
    """corpus-sanity reduces its own scores: an unmeasured cell is excluded and fails the run."""

    def _invoke(self, tmp_path: Path, scores: list[ScoringResult]):
        manifest = CorpusManifest(diffs=[
            CorpusDiff(diff_id=s.diff_id, language='python', source='synthetic',
                       diff_text='', description=s.diff_id, ground_truth=[])
            for s in scores
        ])
        panel_results = [
            PanelRunResult(variant_name='baseline', diff_id=s.diff_id, total_cost_usd=s.cost_usd)
            for s in scores
        ]
        with (
            patch('orchestrator.evals.reviewer_trial.__main__._load_corpus', return_value=manifest),
            patch('orchestrator.evals.reviewer_trial.runner.run_trial',
                  new_callable=AsyncMock, return_value=panel_results),
            patch('orchestrator.evals.reviewer_trial.scorer.score_panel_run',
                  new_callable=AsyncMock, side_effect=scores),
        ):
            return CliRunner().invoke(cli, ['corpus-sanity', '--report-dir', str(tmp_path)])

    def test_an_unmeasured_cell_is_excluded_from_the_aggregates_and_fails_the_run(
        self, tmp_path: Path,
    ) -> None:
        measured = ScoringResult(
            variant_name='baseline', diff_id='d1', cost_usd=1.0, match_cost_usd=0.3,
            recall=0.8, precision=0.8, f1=0.8, blocking_recall=0.8,
        )

        result = self._invoke(tmp_path, [measured, _unmeasured_score('baseline', 'd2')])

        assert result.exit_code == 1, f'{result.output}\n{result.exception!r}'
        lines = result.output.splitlines()
        row = next(line for line in lines if line.startswith('d2 '))
        assert 'UNMEASURED' in row and '0.000' not in row, row
        assert '$  2.31' in row, row
        assert not any('d2' in line and 'recall=0.000' in line for line in lines)
        mean_line = next(line for line in lines if 'Mean recall (synthetic):' in line)
        assert '0.800' in mean_line, mean_line
        assert '  - d2' not in result.output
        assert 'Unmeasured: 1' in result.output

    def test_nothing_measured_prints_n_a_and_fails_both_thresholds(self, tmp_path: Path) -> None:
        result = self._invoke(
            tmp_path, [_unmeasured_score('baseline', 'd1'), _unmeasured_score('baseline', 'd2')],
        )

        assert result.exit_code == 1, f'{result.output}\n{result.exception!r}'
        mean_line = next(line for line in result.output.splitlines() if 'Mean recall (synthetic):' in line)
        assert 'n/a' in mean_line, mean_line
        assert result.output.count('[FAIL]') == 2, result.output


class TestRefreshUnmeasured:
    def test_an_unmeasured_cell_prints_unmeasured_and_is_counted_in_the_leaderboard(
        self, tmp_path: Path,
    ) -> None:
        manifest = CorpusManifest(diffs=[
            CorpusDiff(diff_id='syn_1', language='python', source='synthetic',
                       diff_text='diff1', description='s1', ground_truth=[]),
            CorpusDiff(diff_id='rw_1', language='typescript', source='real_world',
                       diff_text='diff2', description='rw1', ground_truth=[]),
        ])
        panel_results = [
            PanelRunResult(variant_name=v.name, diff_id=d, total_cost_usd=1.0)
            for v in REVIEWER_REFRESH_VARIANTS for d in ('syn_1', 'rw_1')
        ]

        async def fake_score_panel_run(panel_result: PanelRunResult, diff: CorpusDiff) -> ScoringResult:
            if (panel_result.variant_name, panel_result.diff_id) == ('variant_cross_family', 'syn_1'):
                return _unmeasured_score('variant_cross_family', 'syn_1')
            return ScoringResult(
                variant_name=panel_result.variant_name, diff_id=panel_result.diff_id,
                recall=0.5, precision=0.5, f1=0.5, blocking_recall=0.5, cost_usd=1.0,
            )

        with (
            patch('orchestrator.evals.reviewer_trial.__main__._load_corpus', return_value=manifest),
            patch('orchestrator.evals.reviewer_trial.runner.run_trial',
                  new_callable=AsyncMock, return_value=panel_results),
            patch('orchestrator.evals.reviewer_trial.scorer.score_panel_run',
                  new_callable=AsyncMock, side_effect=fake_score_panel_run),
        ):
            result = CliRunner().invoke(cli, ['refresh', '--report-dir', str(tmp_path)])

        assert result.exit_code == 0, f'{result.output}\n{result.exception!r}'
        cell = next(
            line for line in result.output.splitlines()
            if 'variant_cross_family' in line and ' x syn_1' in line
        )
        assert 'UNMEASURED' in cell and 'matcher:cap_hit' in cell, cell
        assert 'F1=0.000' not in cell, cell
        md = (tmp_path / 'refresh_report.md').read_text().splitlines()
        header = next(line for line in md if line.startswith('| Rank |'))
        row = next(line for line in md if line.startswith('|') and '| variant_cross_family |' in line)
        columns = [c.strip() for c in header.strip('|').split('|')]
        cells = [c.strip() for c in row.strip('|').split('|')]
        assert cells[columns.index('Unmeasured')] == '1'
