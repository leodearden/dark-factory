"""CLI for reviewer panel trial verification steps.

Usage:
    cd orchestrator && uv run python -m orchestrator.evals.reviewer_trial <command>

Commands:
    unit-tests     Run the 52 unit tests (step 1)
    smoke          Run baseline panel on 1 diff (step 2)
    calibrate      Run scorer on 3 diffs for human inspection (step 3)
    corpus-sanity  Run baseline on all 15 diffs and score (step 4)
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import click

_PKG_DIR = Path(__file__).parent
_CORPUS_PATH = _PKG_DIR / 'corpus' / 'manifest.json'
_RESULTS_DIR = _PKG_DIR / 'results'
_ORCHESTRATOR_DIR = _PKG_DIR.parents[3]  # orchestrator/

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_check(passed: bool, message: str) -> bool:
    tag = click.style('[PASS]', fg='green') if passed else click.style('[FAIL]', fg='red')
    click.echo(f'{tag} {message}')
    return passed


async def _load_or_run_panel(variant, diff, stagger_secs: float = 2.0):
    """Load cached result or run panel and save.

    Mirrors the caching logic in runner.py _run_and_save() so results
    from one step are reused by later steps.
    """
    from .runner import PanelRunResult, run_panel

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _RESULTS_DIR / f'{variant.name}__{diff.diff_id}.json'

    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            click.echo(f'  [cached] {variant.name}__{diff.diff_id}')
            return PanelRunResult(
                variant_name=data['variant_name'],
                diff_id=data['diff_id'],
                reviews=data.get('reviews', {}),
                total_cost_usd=data.get('total_cost_usd', 0.0),
                wall_clock_ms=data.get('wall_clock_ms', 0),
                errors=data.get('errors', []),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    click.echo(f'  [running] {variant.name}__{diff.diff_id}')
    result = await run_panel(variant, diff, stagger_secs=stagger_secs)
    result_path.write_text(json.dumps(asdict(result), indent=2))
    return result


def _load_corpus():
    """Load corpus, exit with error if not found."""
    from .corpus import CorpusManifest

    if not _CORPUS_PATH.exists():
        click.echo(f'Corpus manifest not found: {_CORPUS_PATH}', err=True)
        sys.exit(1)
    return CorpusManifest.load(_CORPUS_PATH)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option('--verbose', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Reviewer panel trial verification steps."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt='%H:%M:%S', stream=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1: Unit tests
# ---------------------------------------------------------------------------

@cli.command('unit-tests')
def cmd_unit_tests() -> None:
    """Step 1: Run reviewer trial unit tests."""
    click.echo(click.style('Step 1: Unit Tests', bold=True))
    click.echo('=' * 40)

    # Find test files (glob doesn't expand in subprocess args)
    test_dir = _ORCHESTRATOR_DIR / 'tests'
    test_files = sorted(str(f) for f in test_dir.glob('test_reviewer_trial_*.py'))
    if not test_files:
        click.echo('No test files found matching tests/test_reviewer_trial_*.py', err=True)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, '-m', 'pytest', *test_files, '-x', '--tb=short', '-v'],
        cwd=str(_ORCHESTRATOR_DIR),
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Step 2: Smoke test
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--diff-id', default='py_missing_await', help='Corpus diff to test')
@click.option('--stagger', default=2.0, type=float, help='Seconds between reviewer launches')
def smoke(diff_id: str, stagger: float) -> None:
    """Step 2: Smoke test -- run baseline panel on 1 diff."""

    async def _run() -> int:
        from .runner import REVIEW_SCHEMA
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()
        diff = corpus.get_diff(diff_id)
        if diff is None:
            click.echo(f'Diff not found: {diff_id}', err=True)
            return 1

        click.echo(click.style(f'Smoke Test: baseline x {diff_id}', bold=True))
        click.echo('=' * 50)
        click.echo()

        result = await _load_or_run_panel(VARIANT_BASELINE, diff, stagger_secs=stagger)
        click.echo()

        checks_passed = 0
        total_checks = 5
        expected = len(VARIANT_BASELINE.reviewers)

        # Check A: All reviewers returned
        ok = _print_check(
            len(result.reviews) == expected,
            f'{len(result.reviews)}/{expected} reviewers returned valid JSON',
        )
        if ok:
            checks_passed += 1

        # Check B: Required fields
        required = set(REVIEW_SCHEMA['required'])
        all_have_fields = all(
            required <= set(rev.keys())
            for rev in result.reviews.values()
        )
        if _print_check(all_have_fields, f'All reviews have required fields ({", ".join(sorted(required))})'):
            checks_passed += 1

        # Check C: No ERROR verdicts
        error_count = sum(1 for rev in result.reviews.values() if rev.get('verdict') == 'ERROR')
        if _print_check(error_count == 0, f'{error_count} ERROR verdicts'):
            checks_passed += 1

        # Check D: At least 1 blocking issue found
        blocking_count = sum(
            1 for rev in result.reviews.values()
            for issue in rev.get('issues', [])
            if issue.get('severity') == 'blocking'
        )
        reviewers_with_blocking = sum(
            1 for rev in result.reviews.values()
            if any(i.get('severity') == 'blocking' for i in rev.get('issues', []))
        )
        if _print_check(blocking_count > 0, f'{reviewers_with_blocking}/{expected} reviewers found blocking issues ({blocking_count} total)'):
            checks_passed += 1

        # Check E: Cost in range
        cost = result.total_cost_usd
        in_range = 0.50 <= cost <= 15.00
        if _print_check(in_range, f'Total cost: ${cost:.2f} (within $0.50-$15.00)'):
            checks_passed += 1

        # Reviewer summary
        click.echo()
        click.echo(click.style('Reviewer Summary:', bold=True))
        for name, rev in sorted(result.reviews.items()):
            verdict = rev.get('verdict', '?')
            issues = rev.get('issues', [])
            blocking = sum(1 for i in issues if i.get('severity') == 'blocking')
            click.echo(f'  {name:25s} {verdict:15s} ({len(issues)} issues, {blocking} blocking)')

        if result.errors:
            click.echo()
            click.echo(click.style('Errors:', fg='red'))
            for err in result.errors:
                click.echo(f'  - {err}')

        click.echo()
        color = 'green' if checks_passed == total_checks else 'red'
        click.echo(click.style(f'Result: {checks_passed}/{total_checks} checks passed', fg=color, bold=True))
        click.echo(f'Wall clock: {result.wall_clock_ms / 1000:.1f}s | Cost: ${result.total_cost_usd:.2f}')
        return 0 if checks_passed == total_checks else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Step 3: Scorer calibration
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--diff-ids', default='py_missing_await,rs_off_by_one,ts_shallow_mutate',
              help='Comma-separated diff IDs (1 per language)')
@click.option('--stagger', default=2.0, type=float)
def calibrate(diff_ids: str, stagger: float) -> None:
    """Step 3: Scorer calibration -- run scorer on 3 diffs for human inspection."""

    async def _run() -> None:
        from .scorer import score_panel_run
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()
        ids = [d.strip() for d in diff_ids.split(',')]

        click.echo(click.style(f'Scorer Calibration: {len(ids)} diffs', bold=True))
        click.echo('=' * 50)

        for diff_id in ids:
            diff = corpus.get_diff(diff_id)
            if diff is None:
                click.echo(f'\nDiff not found: {diff_id}', err=True)
                continue

            click.echo(f'\n{"=" * 50}')
            click.echo(click.style(f'Calibration: {diff_id} ({diff.language}, {diff.source})', bold=True))
            click.echo(f'{"=" * 50}')

            # Run panel
            result = await _load_or_run_panel(VARIANT_BASELINE, diff, stagger_secs=stagger)

            # Score
            click.echo('\n  Scoring with haiku...')
            score = await score_panel_run(result, diff)

            # Display ground truth
            click.echo(f'\n{click.style("Ground Truth", bold=True)} ({len(diff.ground_truth)} issues):')
            for gt in diff.ground_truth:
                sev_color = 'red' if gt.severity == 'blocking' else 'yellow'
                click.echo(f'  [{gt.id}] {click.style(gt.severity.upper(), fg=sev_color)}  {gt.category}  {gt.location}')
                click.echo(f'    {gt.description[:120]}')

            # Display matches
            click.echo(f'\n{click.style("Matched Issues", bold=True)} ({len(score.matches)}):')
            if score.matches:
                for m in score.matches:
                    ri = m.reviewer_issue
                    click.echo(f'  [{m.ground_truth_id}] <-> {ri.get("reviewer", "?")} (confidence: {m.match_confidence:.2f})')
                    click.echo(f'    Found: {ri.get("description", "?")[:100]}')
                    click.echo(f'    Reasoning: {m.match_reasoning[:100]}')
            else:
                click.echo('  (none)')

            # Display unmatched GT
            if score.unmatched_gt:
                click.echo(f'\n{click.style("Unmatched Ground Truth", fg="red")} ({len(score.unmatched_gt)}):')
                for gt_id in score.unmatched_gt:
                    click.echo(f'  - {gt_id}')

            # Display false positives
            if score.false_positives:
                click.echo(f'\n{click.style("False Positives", fg="yellow")} ({len(score.false_positives)}):')
                for fp in score.false_positives:
                    click.echo(f'  - {fp.get("reviewer", "?")}: {fp.get("description", "?")[:80]} ({fp.get("severity", "?")})')

            # Metrics
            click.echo(f'\nMetrics: recall={score.recall:.3f}  precision={score.precision:.3f}  F1={score.f1:.3f}  blocking_recall={score.blocking_recall:.3f}')

        click.echo(f'\n{"=" * 50}')
        click.echo(click.style('ACTION: Review matches above. If accuracy < 0.85, scorer prompts need tuning.', fg='cyan', bold=True))

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Step 4: Corpus sanity
# ---------------------------------------------------------------------------

@cli.command('corpus-sanity')
@click.option('--max-parallel', default=3, type=int, help='Max concurrent panel runs')
@click.option('--stagger', default=2.0, type=float)
@click.option('--report-dir', default=None, type=click.Path(path_type=Path),
              help='Directory for report output (default: results/)')
def corpus_sanity(max_parallel: int, stagger: float, report_dir: Path | None) -> None:
    """Step 4: Corpus sanity -- run baseline on all 15 diffs, score, and report."""

    async def _run() -> int:
        from .report import build_trial_report, save_report
        from .runner import run_trial
        from .scorer import score_panel_run
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()

        click.echo(click.style(f'Corpus Sanity: baseline x {len(corpus.diffs)} diffs', bold=True))
        click.echo('=' * 50)
        click.echo()

        # Run baseline against all diffs
        click.echo('Running panels...')
        results = await run_trial([VARIANT_BASELINE], corpus, max_parallel_panels=max_parallel)
        click.echo(f'  Completed {len(results)} panel runs.')

        # Score all results
        click.echo('\nScoring results...')
        scores = []
        for result in results:
            diff = corpus.get_diff(result.diff_id)
            if diff is None:
                click.echo(f'  Warning: no corpus diff for {result.diff_id}', err=True)
                continue
            score = await score_panel_run(result, diff)
            scores.append(score)
            click.echo(f'  {result.diff_id}: recall={score.recall:.3f} blocking_recall={score.blocking_recall:.3f} F1={score.f1:.3f}')

        # Build report
        report = build_trial_report(scores, [VARIANT_BASELINE], corpus)
        out_dir = report_dir or _RESULTS_DIR
        report_path = save_report(report, out_dir / 'sanity_report')
        click.echo(f'\nReport saved to: {report_path}')

        # Per-diff results table
        click.echo(f'\n{click.style("Per-Diff Results", bold=True)}:')
        click.echo(f'{"Diff":<26s} {"Lang":<5s} {"Source":<10s} {"Recall":>7s} {"BlkRcl":>7s} {"F1":>7s} {"Cost":>7s}')
        click.echo('-' * 76)

        synthetic_recalls: list[float] = []
        synthetic_blocking_recalls: list[float] = []
        zero_recall_diffs: list[str] = []

        for score in sorted(scores, key=lambda s: s.diff_id):
            diff = corpus.get_diff(score.diff_id)
            lang = diff.language[:5] if diff else '?'
            source = diff.source if diff else '?'
            click.echo(
                f'{score.diff_id:<26s} {lang:<5s} {source:<10s} '
                f'{score.recall:>7.3f} {score.blocking_recall:>7.3f} {score.f1:>7.3f} '
                f'${score.cost_usd:>6.2f}'
            )
            if diff and diff.source == 'synthetic':
                synthetic_recalls.append(score.recall)
                synthetic_blocking_recalls.append(score.blocking_recall)
            if score.recall == 0.0:
                zero_recall_diffs.append(score.diff_id)

        # Aggregate metrics
        mean_recall = sum(synthetic_recalls) / len(synthetic_recalls) if synthetic_recalls else 0.0
        mean_blocking = sum(synthetic_blocking_recalls) / len(synthetic_blocking_recalls) if synthetic_blocking_recalls else 0.0
        all_recalls = [s.recall for s in scores]
        all_f1s = [s.f1 for s in scores]
        total_cost = sum(s.cost_usd for s in scores)

        click.echo()
        click.echo(click.style('Aggregate Metrics:', bold=True))
        click.echo(f'  Mean recall (synthetic):          {mean_recall:.3f}')
        click.echo(f'  Mean blocking recall (synthetic):  {mean_blocking:.3f}')
        click.echo(f'  Mean recall (all):                 {sum(all_recalls) / len(all_recalls):.3f}' if all_recalls else '')
        click.echo(f'  Mean F1 (all):                     {sum(all_f1s) / len(all_f1s):.3f}' if all_f1s else '')
        click.echo(f'  Total cost:                        ${total_cost:.2f}')

        # Threshold checks
        click.echo()
        checks_passed = 0
        total_checks = 2

        if _print_check(mean_recall > 0.6, f'Mean recall (synthetic) = {mean_recall:.3f} > 0.6'):
            checks_passed += 1
        if _print_check(mean_blocking > 0.5, f'Mean blocking recall (synthetic) = {mean_blocking:.3f} > 0.5'):
            checks_passed += 1

        if zero_recall_diffs:
            click.echo()
            click.echo(click.style('Zero-Recall Diffs (annotation may be too specific):', fg='yellow'))
            for d in zero_recall_diffs:
                click.echo(f'  - {d}')

        click.echo()
        color = 'green' if checks_passed == total_checks else 'red'
        click.echo(click.style(f'Result: {checks_passed}/{total_checks} thresholds met', fg=color, bold=True))
        return 0 if checks_passed == total_checks and not zero_recall_diffs else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    cli()
