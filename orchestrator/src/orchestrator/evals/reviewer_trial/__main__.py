"""CLI for reviewer panel trial verification steps.

Usage:
    cd orchestrator && uv run python -m orchestrator.evals.reviewer_trial <command>

Commands:
    unit-tests     Run the 52 unit tests (step 1)
    smoke          Run baseline panel on 1 diff (step 2)
    calibrate      Run scorer on 3 diffs for human inspection (step 3)
    corpus-sanity  Run baseline on all 15 diffs and score (step 4)
    campaign       Run arms x fixtures x N trials and reduce them; ``--briefing
                   diff-only,d2`` adds a context-complete arm per variant,
                   ``--briefing d2`` runs only the context-complete arms
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
from shared import project_layout

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.main_checkout_defaults import cli_default
from orchestrator.review_payload import Verdict, normalize_review

if TYPE_CHECKING:
    from .corpus import CorpusDiff, CorpusManifest
    from .scorer import ScoringResult

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


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt_mean(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.3f}'


def _metrics_text(f1: float, blocking_recall: float, cost_usd: float) -> str:
    return f'F1={f1:.3f} BR={blocking_recall:.3f} ${cost_usd:.2f}'


def _cell_text(score: ScoringResult) -> str:
    """One scored cell for a progress line: its metrics and spend, or why it has none."""
    if score.unmeasured is None:
        return _metrics_text(score.f1, score.blocking_recall, score.total_cost_usd)
    from orchestrator.evals.measurement import format_invocation_errors

    why = format_invocation_errors([score.unmeasured.to_record()])
    return f'UNMEASURED {why} ${score.total_cost_usd:.2f}'


def _echo_cell(score: ScoringResult) -> None:
    click.echo(f'  {score.variant_name:22s} x {score.diff_id:<28s} {_cell_text(score)}')


def _load_corpus():
    """Load corpus, exit with error if not found."""
    from .corpus import CorpusManifest

    if not _CORPUS_PATH.exists():
        click.echo(f'Corpus manifest not found: {_CORPUS_PATH}', err=True)
        sys.exit(1)
    return CorpusManifest.load(_CORPUS_PATH)


def _select_original_corpus(manifest: CorpusManifest) -> CorpusManifest:
    """Return a new manifest of just the original hand-authored (non-mined) diffs.

    Task 2495's FN-mining grew the on-disk manifest to 53 diffs (12 synthetic
    + 3 real_world + 38 mined). The eval-revival κ refresh (task 2476) is
    scoped to the original NON-mined 15 (source in {synthetic, real_world}) to
    stay apples-to-apples with the Apr-8 1×Opus trial and to bound live-run
    cost. version/split_seed are carried over unchanged.
    """
    from .corpus import CorpusManifest

    return CorpusManifest(
        diffs=[
            *manifest.filter_by_source('synthetic'),
            *manifest.filter_by_source('real_world'),
        ],
        version=manifest.version,
        split_seed=manifest.split_seed,
    )


def _select_split(manifest: CorpusManifest, split: str) -> CorpusManifest:
    """Return a manifest holding only the diffs assigned to *split*."""
    from .corpus import CorpusManifest

    return CorpusManifest(
        diffs=[d for d in manifest.diffs if d.split == split],
        version=manifest.version,
        split_seed=manifest.split_seed,
    )


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
        from .runner import REVIEW_SCHEMA, run_panel_cached
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()
        diff = corpus.get_diff(diff_id)
        if diff is None:
            click.echo(f'Diff not found: {diff_id}', err=True)
            return 1

        click.echo(click.style(f'Smoke Test: baseline x {diff_id}', bold=True))
        click.echo('=' * 50)
        click.echo()

        result = await run_panel_cached(VARIANT_BASELINE, diff, stagger_secs=stagger)
        click.echo()
        payloads = {name: normalize_review(rev) for name, rev in result.reviews.items()}

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
        error_count = sum(1 for p in payloads.values() if p.verdict is Verdict.ERROR)
        if _print_check(error_count == 0, f'{error_count} ERROR verdicts'):
            checks_passed += 1

        # Check D: At least 1 blocking issue found
        blocking_count = sum(len(p.blocking_issues) for p in payloads.values())
        reviewers_with_blocking = sum(1 for p in payloads.values() if p.blocking_issues)
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
        for name, p in sorted(payloads.items()):
            click.echo(
                f'  {name:25s} {p.verdict:15s} '
                f'({len(p.issues)} issues, {len(p.blocking_issues)} blocking)'
            )

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

def _echo_calibration(diff: CorpusDiff, score: ScoringResult) -> None:
    """*score* beside *diff*'s ground truth: what matched, what was missed, what was spurious."""
    click.echo(f'\n{"=" * 50}')
    click.echo(click.style(f'Calibration: {diff.diff_id} ({diff.language}, {diff.source})', bold=True))
    click.echo(f'{"=" * 50}')
    if score.unmeasured is not None:
        click.echo(f'  Scorer: {_cell_text(score)}')
        return

    click.echo(f'\n{click.style("Ground Truth", bold=True)} ({len(diff.ground_truth)} issues):')
    for gt in diff.ground_truth:
        sev_color = 'red' if gt.severity == 'blocking' else 'yellow'
        click.echo(f'  [{gt.id}] {click.style(gt.severity.upper(), fg=sev_color)}  {gt.category}  {gt.location}')
        click.echo(f'    {gt.description[:120]}')

    click.echo(f'\n{click.style("Matched Issues", bold=True)} ({len(score.matches)}):')
    if score.matches:
        for m in score.matches:
            ri = m.reviewer_issue
            click.echo(f'  [{m.ground_truth_id}] <-> {ri.get("reviewer", "?")} (confidence: {m.match_confidence:.2f})')
            click.echo(f'    Found: {ri.get("description", "?")[:100]}')
            click.echo(f'    Reasoning: {m.match_reasoning[:100]}')
    else:
        click.echo('  (none)')

    if score.unmatched_gt:
        click.echo(f'\n{click.style("Unmatched Ground Truth", fg="red")} ({len(score.unmatched_gt)}):')
        for gt_id in score.unmatched_gt:
            click.echo(f'  - {gt_id}')

    if score.false_positives:
        click.echo(f'\n{click.style("False Positives", fg="yellow")} ({len(score.false_positives)}):')
        for fp in score.false_positives:
            click.echo(f'  - {fp.get("reviewer", "?")}: {fp.get("description", "?")[:80]} ({fp.get("severity", "?")})')

    click.echo(f'\nMetrics: recall={score.recall:.3f}  precision={score.precision:.3f}  F1={score.f1:.3f}  blocking_recall={score.blocking_recall:.3f}')


@cli.command()
@click.option('--diff-ids', default='py_missing_await,rs_off_by_one,ts_shallow_mutate',
              help='Comma-separated diff IDs (1 per language)')
@click.option('--stagger', default=2.0, type=float)
def calibrate(diff_ids: str, stagger: float) -> None:
    """Step 3: Scorer calibration -- run scorer on 3 diffs for human inspection."""

    async def _run() -> None:
        from .corpus import CorpusManifest
        from .runner import run_and_score
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()
        ids = [d.strip() for d in diff_ids.split(',')]

        click.echo(click.style(f'Scorer Calibration: {len(ids)} diffs', bold=True))
        click.echo('=' * 50)

        found = []
        for diff_id in ids:
            diff = corpus.get_diff(diff_id)
            if diff is None:
                click.echo(f'\nDiff not found: {diff_id}', err=True)
            else:
                found.append(diff)

        click.echo('\nRunning panels and scoring with haiku...')
        scores = await run_and_score(
            [VARIANT_BASELINE], CorpusManifest(diffs=found), stagger_secs=stagger, on_scored=_echo_cell,
        )
        score_by_id = {score.diff_id: score for score in scores}
        for diff in found:
            score = score_by_id.get(diff.diff_id)
            if score is None:
                click.echo(f'\nNo score for {diff.diff_id}: its fixture was skipped (see the log)', err=True)
            else:
                _echo_calibration(diff, score)

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
        from .runner import run_and_score
        from .variants import VARIANT_BASELINE

        corpus = _load_corpus()

        click.echo(click.style(f'Corpus Sanity: baseline x {len(corpus.diffs)} diffs', bold=True))
        click.echo('=' * 50)
        click.echo()

        click.echo('Running and scoring panels...')
        scores = await run_and_score(
            [VARIANT_BASELINE], corpus, max_parallel_panels=max_parallel, stagger_secs=stagger,
            on_scored=_echo_cell,
        )
        click.echo(f'  Scored {len(scores)} panel runs.')

        # Build report
        report = build_trial_report(scores, [VARIANT_BASELINE], corpus)
        out_dir = report_dir or _RESULTS_DIR
        report_path = save_report(report, out_dir / 'sanity_report')
        click.echo(f'\nReport saved to: {report_path}')

        # Per-diff results table
        click.echo(f'\n{click.style("Per-Diff Results", bold=True)}:')
        click.echo(f'{"Diff":<26s} {"Lang":<5s} {"Source":<10s} {"Recall":>7s} {"BlkRcl":>7s} {"F1":>7s} {"TotCost":>7s}')
        click.echo('-' * 76)

        synthetic_recalls: list[float] = []
        synthetic_blocking_recalls: list[float] = []
        zero_recall_diffs: list[str] = []
        measured = [s for s in scores if s.unmeasured is None]
        unmeasured = [s for s in scores if s.unmeasured is not None]

        for score in sorted(scores, key=lambda s: s.diff_id):
            diff = corpus.get_diff(score.diff_id)
            lang = diff.language[:5] if diff else '?'
            source = diff.source if diff else '?'
            metric_columns = (
                f'{score.recall:>7.3f} {score.blocking_recall:>7.3f} {score.f1:>7.3f}'
                if score.unmeasured is None else f'{"UNMEASURED":>23s}'
            )
            click.echo(
                f'{score.diff_id:<26s} {lang:<5s} {source:<10s} '
                f'{metric_columns} ${score.total_cost_usd:>6.2f}'
            )
            if score.unmeasured is not None:
                continue
            if diff and diff.source == 'synthetic':
                synthetic_recalls.append(score.recall)
                synthetic_blocking_recalls.append(score.blocking_recall)
            if score.recall == 0.0:
                zero_recall_diffs.append(score.diff_id)

        # Aggregate metrics, over measured cells only
        mean_recall = _mean_or_none(synthetic_recalls)
        mean_blocking = _mean_or_none(synthetic_blocking_recalls)
        total_cost = sum(s.total_cost_usd for s in scores)

        click.echo()
        click.echo(click.style('Aggregate Metrics:', bold=True))
        click.echo(f'  Mean recall (synthetic):          {_fmt_mean(mean_recall)}')
        click.echo(f'  Mean blocking recall (synthetic):  {_fmt_mean(mean_blocking)}')
        click.echo(f'  Mean recall (all):                 {_fmt_mean(_mean_or_none([s.recall for s in measured]))}')
        click.echo(f'  Mean F1 (all):                     {_fmt_mean(_mean_or_none([s.f1 for s in measured]))}')
        click.echo(f'  Unmeasured: {len(unmeasured)} (excluded from every mean)')
        click.echo(f'  Total cost:                        ${total_cost:.2f}')

        # Threshold checks
        click.echo()
        checks_passed = 0
        total_checks = 2

        if _print_check(
            mean_recall is not None and mean_recall > 0.6,
            f'Mean recall (synthetic) = {_fmt_mean(mean_recall)} > 0.6',
        ):
            checks_passed += 1
        if _print_check(
            mean_blocking is not None and mean_blocking > 0.5,
            f'Mean blocking recall (synthetic) = {_fmt_mean(mean_blocking)} > 0.5',
        ):
            checks_passed += 1

        if zero_recall_diffs:
            click.echo()
            click.echo(click.style('Zero-Recall Diffs (annotation may be too specific):', fg='yellow'))
            for d in zero_recall_diffs:
                click.echo(f'  - {d}')

        if unmeasured:
            click.echo()
            click.echo(click.style(
                f'Sanity run incomplete: {len(unmeasured)} cell(s) unmeasured — rerun to measure them.',
                fg='red',
            ))

        click.echo()
        color = 'green' if checks_passed == total_checks else 'red'
        click.echo(click.style(f'Result: {checks_passed}/{total_checks} thresholds met', fg=color, bold=True))
        complete = checks_passed == total_checks and not zero_recall_diffs and not unmeasured
        return 0 if complete else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Corpus audit: integrity signals over the (expanded) corpus
# ---------------------------------------------------------------------------

_ADJUDICATION_LOG_PATH = _PKG_DIR / 'corpus' / 'adjudication_log.jsonl'

# Shared default floor for both `corpus-audit --min-diffs` and `mine`'s
# post-run audit, so the two paths can't silently disagree on what "enough
# diffs" means.
_DEFAULT_MIN_DIFFS = 50

# (check key, human-readable description) — order matches mining.audit_corpus's
# check ordering so PASS/FAIL output reads top-to-bottom in the same sequence
# the report's `failures` reasons would appear in.
_AUDIT_CHECKS = (
    ('diff_count', 'diff count meets the minimum floor'),
    ('missing_split', 'every diff has a split assigned'),
    ('split_ratio', 'train/selection/test ratio approximates 2:1:7'),
    ('missing_provenance', 'every mined-source diff has non-empty provenance'),
    ('adjudication_coverage', 'adjudication_log covers every diff_id'),
    ('spot_check_subset_empty', 'a non-empty documented spot-check subset exists'),
)


@cli.command('corpus-audit')
@click.option('--min-diffs', default=_DEFAULT_MIN_DIFFS, type=int, help='Minimum required diff count')
def corpus_audit(min_diffs: int) -> None:
    """Audit corpus integrity: size, split ratios, provenance, adjudication coverage, spot-check subset."""
    from .adjudication import AdjudicationLog
    from .mining import audit_corpus

    corpus = _load_corpus()
    log = AdjudicationLog.load(_ADJUDICATION_LOG_PATH)
    report = audit_corpus(corpus, log, min_diffs=min_diffs)

    click.echo(click.style(f'Corpus Audit: {report.diff_count} diffs (min {min_diffs})', bold=True))
    click.echo('=' * 50)

    checks_passed = 0
    for key, message in _AUDIT_CHECKS:
        if _print_check(key not in report.failures, message):
            checks_passed += 1

    click.echo()
    color = 'green' if report.ok else 'red'
    status = 'PASS' if report.ok else 'FAIL'
    click.echo(click.style(
        f'Result: {status} ({checks_passed}/{len(_AUDIT_CHECKS)} checks passed)', fg=color, bold=True,
    ))
    sys.exit(0 if report.ok else 1)


# ---------------------------------------------------------------------------
# Mine: FN-candidate mining + frontier labeling (task 2495 / PRD D-6)
# ---------------------------------------------------------------------------

_MINE_SPLIT_SEED = 'reviewer_trial-2495-fn-mining'

@dataclass
class _MineTally:
    """How far one `mine` run's labelling has got, across its concurrent workers."""

    attempted: int
    completed: int = 0
    added: int = 0
    cost: float = 0.0


@cli.command('mine')
@click.option('--runs-db', default=None, type=click.Path(path_type=Path),
              help='Path to the offline orchestrator runs.db (env: REVIEWER_TRIAL_RUNS_DB; '
                   "default: the main checkout's runs.db)")
@click.option('--escalations-dir', default=None, type=click.Path(path_type=Path),
              help='Path to the offline escalations directory (env: REVIEWER_TRIAL_ESCALATIONS_DIR; '
                   "default: the main checkout's escalation queue)")
@click.option('--repo-path', default=None, type=click.Path(path_type=Path),
              help='Git repo to recover mined diffs from (env: REVIEWER_TRIAL_REPO_ROOT; '
                   'default: the main checkout)')
@click.option('--target-total', default=100, type=int, help='Aim for this many total diffs (existing + mined)')
@click.option('--max-parallel', default=8, type=int, help='Max concurrent frontier-labeling calls')
@click.option('--model', default='opus', help='Frontier model for propose_labels_frontier')
@click.option('--max-turns', default=15, type=int, help='Turn budget per frontier call (effort=high needs >3)')
@click.option('--seed', default=_MINE_SPLIT_SEED, help='Deterministic train/selection/test split seed')
@click.option('--oauth-token-env', default='CLAUDE_OAUTH_TOKEN_A',
              help='Env var holding the OAuth token for frontier calls (multi-account pool)')
@click.option('--limit', default=None, type=int, help='Cap the number of NEW candidates attempted this run')
@click.option('--min-diffs', default=_DEFAULT_MIN_DIFFS, type=int,
              help="Minimum required diff count for the post-run audit (kept consistent with corpus-audit's --min-diffs)")
def mine(
    runs_db: Path | None, escalations_dir: Path | None, repo_path: Path | None, target_total: int,
    max_parallel: int, model: str, max_turns: int, seed: str, oauth_token_env: str,
    limit: int | None, min_diffs: int,
) -> None:
    """Mine FN-candidate diffs from runs.db/escalations, frontier-label them, and
    expand + re-split the committed corpus (task 2495 / PRD D-6).

    Idempotent and resumable: re-running skips diff_ids already present in the
    corpus, and persists (manifest + adjudication log, re-split every time)
    after every successfully-labeled diff, so an interrupted run only loses
    whatever was still in flight.
    """
    runs_db = runs_db or cli_default(
        '--runs-db', project_layout.runs_db, env_var='REVIEWER_TRIAL_RUNS_DB',
    )
    escalations_dir = escalations_dir or cli_default(
        '--escalations-dir', project_layout.escalation_queue_dir,
        env_var='REVIEWER_TRIAL_ESCALATIONS_DIR',
    )
    repo_path = repo_path or cli_default('--repo-path', Path, env_var='REVIEWER_TRIAL_REPO_ROOT')

    async def _run() -> int:
        from orchestrator.config import GitConfig

        from .adjudication import AdjudicationLog
        from .mining import (
            RecoveredCandidate,
            admit_mined_diff,
            audit_corpus,
            backfill_hand_authored,
            candidate_pool,
            fetch_task_titles,
            frontier_description,
            mine_escalation_refs,
            mine_fn_candidates,
            mined_corpus_diff,
            propose_labels_frontier,
            recover_candidates,
            save_corpus,
        )

        if not runs_db.is_file():
            click.echo(
                f'runs.db not found at {runs_db}\n'
                '  This is an offline generation tool that reads the gitignored '
                'orchestrator run history, which lives in the MAIN repo checkout, not '
                "a task worktree (see corpus/README.md). Pass --runs-db, "
                'or set REVIEWER_TRIAL_RUNS_DB, to point at a real runs.db -- otherwise '
                "sqlite3 silently creates an empty one and this fails later with "
                "'no such table: task_results'.",
                err=True,
            )
            return 1

        oauth_token = os.environ.get(oauth_token_env)
        if not oauth_token:
            click.echo(
                f'Warning: {oauth_token_env} not set; falling back to ambient Claude CLI credentials.',
                err=True,
            )

        manifest = _load_corpus()
        log = AdjudicationLog.load(_ADJUDICATION_LOG_PATH)

        click.echo(click.style(
            f'Mine: FN-candidate mining (corpus: {len(manifest.diffs)} diffs on disk)', bold=True,
        ))
        click.echo('=' * 60)

        backfilled = backfill_hand_authored(manifest, log)
        if backfilled:
            click.echo(f'  Backfilled {backfilled} adjudication entries for pre-existing hand-authored diffs.')
            save_corpus(manifest, log, seed=seed, manifest_path=_CORPUS_PATH, log_path=_ADJUDICATION_LOG_PATH)

        candidates = mine_fn_candidates(runs_db)
        esc_refs = mine_escalation_refs(escalations_dir)
        click.echo(f'  {len(candidates)} FN candidates mined from runs.db')

        titles = fetch_task_titles(runs_db, [c.task_id for c in candidates])
        git = GitConfig()
        recovered = recover_candidates(
            candidates, existing_ids={d.diff_id for d in manifest.diffs}, repo_path=repo_path,
            branch_prefix=git.branch_prefix, main_branch=git.main_branch,
        )
        pool = candidate_pool(
            recovered, corpus_size=len(manifest.diffs), target_total=target_total, limit=limit,
        )

        click.echo(f'  {len(recovered)} candidates have a recoverable diff not already in the corpus')
        click.echo(f'  Attempting {len(pool)} this run (target_total={target_total}, limit={limit})')
        click.echo()

        semaphore = asyncio.Semaphore(max(1, max_parallel))
        tally = _MineTally(attempted=len(pool))

        async def _label(item: RecoveredCandidate) -> None:
            candidate = item.candidate
            description = frontier_description(candidate, titles.get(candidate.task_id, ''))
            async with semaphore:
                try:
                    issues, cost = await propose_labels_frontier(
                        item.diff_text, model=model, description=description,
                        oauth_token=oauth_token, max_turns=max_turns,
                    )
                except Exception as exc:  # noqa: BLE001 -- offline generation script; one bad candidate must not sink the batch
                    logging.getLogger(__name__).warning(
                        'propose_labels_frontier failed for task %s: %s', candidate.task_id, exc,
                    )
                    issues, cost = [], 0.0

            tally.completed += 1
            tally.cost += cost
            progress = f'  [{tally.completed}/{tally.attempted}] {candidate.diff_id}:'
            if not issues:
                click.echo(f'{progress} no issues proposed -- skipping (cost=${cost:.3f})')
                return

            diff = mined_corpus_diff(
                item, issues=issues, description=description,
                escalation_refs=esc_refs.get(candidate.task_id, []), runs_db=runs_db,
            )
            admit_mined_diff(manifest, log, diff, frontier_model=model)
            save_corpus(manifest, log, seed=seed, manifest_path=_CORPUS_PATH, log_path=_ADJUDICATION_LOG_PATH)
            tally.added += 1
            click.echo(
                f'{progress} {len(issues)} issues added (cost=${cost:.3f}, corpus now {len(manifest.diffs)})'
            )

        await asyncio.gather(*(_label(item) for item in pool))

        click.echo()
        click.echo(
            f'Mined {tally.added}/{tally.attempted} new diffs this run (frontier cost: ${tally.cost:.2f}).'
        )
        click.echo(f'Corpus: {len(manifest.diffs)} diffs total.')

        report = audit_corpus(manifest, log, min_diffs=min_diffs)
        click.echo()
        click.echo(click.style(f'Corpus Audit: {report.diff_count} diffs', bold=True))
        for key, message in _AUDIT_CHECKS:
            _print_check(key not in report.failures, message)
        return 0 if report.ok else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted -- progress up to the last completed diff was saved; rerun `mine` to resume.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Full trial: all variants x all diffs
# ---------------------------------------------------------------------------

@cli.command('full')
@click.option('--max-parallel', default=3, type=int, help='Max concurrent panel runs')
@click.option('--stagger', default=2.0, type=float)
@click.option('--report-dir', default=None, type=click.Path(path_type=Path),
              help='Directory for report output (default: results/)')
def full_trial(max_parallel: int, stagger: float, report_dir: Path | None) -> None:
    """Full trial: run ALL variants against ALL corpus diffs, score, and report."""

    async def _run() -> int:
        from .report import build_trial_report, format_markdown, save_report
        from .runner import run_and_score
        from .variants import ALL_VARIANTS

        corpus = _load_corpus()

        click.echo(click.style(f'Full Trial: {len(ALL_VARIANTS)} variants x {len(corpus.diffs)} diffs', bold=True))
        click.echo('=' * 60)
        for v in ALL_VARIANTS:
            click.echo(f'  {v.name:15s} {v.description} ({len(v.reviewers)} reviewers)')
        click.echo()

        click.echo('Running and scoring panels...')
        scores = await run_and_score(
            ALL_VARIANTS, corpus, max_parallel_panels=max_parallel, stagger_secs=stagger,
            on_scored=_echo_cell,
        )

        # Build and save report
        report = build_trial_report(scores, ALL_VARIANTS, corpus)
        out_dir = report_dir or _RESULTS_DIR
        report_path = save_report(report, out_dir / 'full_trial_report')

        # Print the full markdown report
        click.echo()
        click.echo(format_markdown(report))
        click.echo(f'\nReport saved to: {report_path}')

        total_cost = sum(s.cost_usd for s in scores)
        click.echo(f'Total panel cost: ${total_cost:.2f}')
        return 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Effort sweep: 1x opus at medium/high/max thinking
# ---------------------------------------------------------------------------

@cli.command('sweep')
@click.option('--max-parallel', default=3, type=int, help='Max concurrent panel runs')
@click.option('--stagger', default=2.0, type=float)
def effort_sweep(max_parallel: int, stagger: float) -> None:
    """Effort sweep: 1x opus at medium/high/max thinking levels."""

    async def _run() -> int:
        from .report import build_trial_report, format_markdown, save_report
        from .runner import run_and_score
        from .variants import EFFORT_SWEEP_VARIANTS

        corpus = _load_corpus()

        click.echo(click.style(
            f'Effort Sweep: {len(EFFORT_SWEEP_VARIANTS)} variants x {len(corpus.diffs)} diffs',
            bold=True,
        ))
        click.echo('=' * 60)
        for v in EFFORT_SWEEP_VARIANTS:
            effort = v.reviewers[0].effort
            click.echo(f'  {v.name:18s} effort={effort:6s} {v.description}')
        click.echo()

        click.echo('Running and scoring panels...')
        scores = await run_and_score(
            EFFORT_SWEEP_VARIANTS, corpus, max_parallel_panels=max_parallel, stagger_secs=stagger,
            on_scored=_echo_cell,
        )

        report = build_trial_report(scores, EFFORT_SWEEP_VARIANTS, corpus)
        report_path = save_report(report, _RESULTS_DIR / 'effort_sweep_report')

        click.echo()
        click.echo(format_markdown(report))
        click.echo(f'\nReport saved to: {report_path}')

        total_cost = sum(s.cost_usd for s in scores)
        click.echo(f'Total panel cost: ${total_cost:.2f}')
        return 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Eval-revival refresh: 1x opus incumbent vs Sonnet-5 vs cross-family,
# scored on the original (non-mined) 15-diff corpus (task 2476)
# ---------------------------------------------------------------------------

@cli.command('refresh')
@click.option('--max-parallel', default=3, type=int, help='Max concurrent panel runs')
@click.option('--stagger', default=2.0, type=float)
@click.option('--report-dir', default=None, type=click.Path(path_type=Path),
              help='Directory for report output (default: results/)')
def refresh_trial(max_parallel: int, stagger: float, report_dir: Path | None) -> None:
    """Eval-revival refresh: 1x opus incumbent vs a Sonnet-5 and a
    cross-family single-generalist candidate, ranked on quality (F1) AND
    cost over the original hand-authored (non-mined) 15-diff corpus.

    Bypasses the FN-mined diffs added by task 2495 so the comparison stays
    apples-to-apples with the Apr-8 1x Opus trial and bounds live-run cost.
    """

    async def _run() -> int:
        from orchestrator.config import default_price_table

        from .report import build_trial_report, format_markdown, save_report
        from .runner import run_and_score
        from .variants import REVIEWER_REFRESH_VARIANTS

        corpus = _select_original_corpus(_load_corpus())
        prices = default_price_table()

        click.echo(click.style(
            f'Eval-Revival Refresh: {len(REVIEWER_REFRESH_VARIANTS)} variants x {len(corpus.diffs)} diffs',
            bold=True,
        ))
        click.echo('=' * 60)
        for v in REVIEWER_REFRESH_VARIANTS:
            click.echo(f'  {v.name:20s} {v.description} ({len(v.reviewers)} reviewers)')
        click.echo()

        click.echo('Running and scoring panels...')
        scores = await run_and_score(
            REVIEWER_REFRESH_VARIANTS, corpus, max_parallel_panels=max_parallel, stagger_secs=stagger,
            prices=prices, on_scored=_echo_cell,
        )

        # Build and save report
        report = build_trial_report(scores, REVIEWER_REFRESH_VARIANTS, corpus)
        out_dir = report_dir or _RESULTS_DIR
        report_path = save_report(report, out_dir / 'refresh_report')

        # Print the full markdown report
        click.echo()
        click.echo(format_markdown(report))
        click.echo(f'\nReport saved to: {report_path}')

        total_cost = sum(s.cost_usd for s in scores)
        click.echo(f'Total panel cost: ${total_cost:.2f}')
        return 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


# ---------------------------------------------------------------------------
# Fable admission evidence campaign (ruling D5, 2026-09-10): every arm over
# the same fixtures for N trials, scored, reduced by campaign.py.
# ---------------------------------------------------------------------------

def _parse_briefings(
    _ctx: click.Context, _param: click.Parameter, value: str,
) -> list[ReviewBriefing]:
    """The generations to run, each once; diff-only, when listed, leads.

    Every arm is paired against the first arm's first generation, so ``d2``
    alone pairs every d2 arm against the first arm's d2 arm, while d2 listed
    before diff-only would invert the gate. A repeat would collide on its
    result paths.
    """
    valid = [b.value for b in ReviewBriefing]
    names = [name.strip() for name in value.split(',') if name.strip()]
    unknown = [name for name in names if name not in valid]
    if unknown or not names:
        raise click.BadParameter(
            f'{", ".join(unknown) or "(none)"}: each must be one of {", ".join(valid)}',
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise click.BadParameter(f'each generation runs once; repeated: {", ".join(repeated)}')
    briefings = [ReviewBriefing(name) for name in names]
    if ReviewBriefing.DIFF_ONLY in briefings and briefings[0] is not ReviewBriefing.DIFF_ONLY:
        raise click.BadParameter(
            f'{ReviewBriefing.DIFF_ONLY.value} must come first when listed: it is the incumbent '
            f'{ReviewBriefing.CONTEXT_COMPLETE.value} is paired against',
        )
    return briefings


@cli.command('campaign')
@click.option('--results-dir', required=True, type=click.Path(path_type=Path),
              help='Campaign root; per-trial panel results and scores checkpoint under it')
@click.option('--trials', default=3, type=int, help='Independent repeats per arm (replication by fixture)')
@click.option('--split', default='test', help='Corpus split to run (train|selection|test|all)')
@click.option('--arms', default=None, help='Comma-separated variant names (default: FABLE_EVIDENCE_VARIANTS)')
@click.option('--briefing', 'briefings', default=ReviewBriefing.DIFF_ONLY.value, callback=_parse_briefings,
              help='Comma-separated reviewer briefing generations, each run once for every arm '
                   '(diff-only | d2; default: diff-only; diff-only first when both are listed; '
                   "d2 alone pairs every arm against the first arm's d2 arm)")
@click.option('--diff-cap', default=None, type=int,
              help='Diff truncation in chars (default: runner.DIFF_CAP_CHARS; 0 lifts the cap)')
@click.option('--max-parallel', default=3, type=int, help='Max concurrent panel runs')
@click.option('--pool/--no-pool', default=True,
              help='Draw on the shared OAuth account pool via a UsageGate (default) or the CLI default login')
@click.option('--config', 'config_path', default=None, type=click.Path(path_type=Path),
              help="Orchestrator config supplying usage_cap accounts (default: the main checkout's)")
def campaign(
    results_dir: Path, trials: int, split: str, arms: str | None,
    briefings: list[ReviewBriefing], diff_cap: int | None,
    max_parallel: int, pool: bool, config_path: Path | None,
) -> None:
    """Run the Fable evidence campaign: every arm x every fixture x N trials.

    Each arm runs once per ``--briefing`` generation; the first arm's first
    generation is the incumbent every other arm is paired against.

    Panel results checkpoint to ``<results-dir>/trial-<n>/`` and scores to
    ``<results-dir>/trial-<n>/scores/``, so a killed campaign resumes
    without re-spending. Re-running with the same arguments only reduces.
    """
    from orchestrator.config import PROJECT_CONFIG_FILENAME, default_price_table, load_config

    config_path = config_path or cli_default('--config', lambda root: root / PROJECT_CONFIG_FILENAME)

    async def _run() -> int:
        from orchestrator.evals.eval_gate import campaign_usage_gate

        from . import campaign as reduce
        from .runner import DIFF_CAP_CHARS, run_and_score
        from .variants import FABLE_EVIDENCE_VARIANTS, campaign_arms

        manifest = _load_corpus()
        corpus = manifest if split == 'all' else _select_split(manifest, split)
        by_name = {v.name: v for v in FABLE_EVIDENCE_VARIANTS}
        variants = campaign_arms(
            [by_name[a.strip()] for a in arms.split(',') if a.strip()]
            if arms else FABLE_EVIDENCE_VARIANTS,
            briefings,
        )
        incumbent = variants[0].name
        cap = DIFF_CAP_CHARS if diff_cap is None else diff_cap
        prices = default_price_table()

        async with campaign_usage_gate(load_config(config_path) if pool else None) as usage_gate:
            if usage_gate is not None:
                await usage_gate.check_at_startup()
            elif pool:
                raise click.ClickException(
                    '--pool was requested but no account-pool gate could be built from '
                    f'{config_path} (usage_cap disabled, the gate failed to construct, or '
                    'no accounts resolved; see the log). Pass --no-pool to run on the CLI '
                    'default login instead.'
                )

            click.echo(click.style(
                f'Campaign: {len(variants)} arms x {len(corpus.diffs)} diffs ({split}) x {trials} trials, '
                f'briefings {",".join(briefings)}, diff cap {cap or "lifted"}, '
                f'pool={"on" if usage_gate else "off"}',
                bold=True,
            ))
            blocking_gt = {d.diff_id: {g.id for g in d.blocking_issues()} for d in corpus.diffs}
            records: list[dict] = []
            for trial in range(1, trials + 1):
                trial_dir = results_dir / f'trial-{trial}'
                click.echo(f'\n== trial {trial}/{trials}')
                scores = await run_and_score(
                    variants, corpus, results_dir=trial_dir, score_dir=trial_dir / 'scores',
                    max_parallel_panels=max_parallel, prices=prices, usage_gate=usage_gate,
                    diff_cap_chars=cap, on_scored=_echo_cell,
                )
                records.extend({**asdict(s), 'trial': trial} for s in scores)

            arms_out, decisions = reduce.summarize_campaign(records, incumbent, blocking_gt)
            md = reduce.format_markdown(arms_out, decisions)
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / 'campaign_report.md').write_text(md + '\n')
            (results_dir / 'campaign_report.json').write_text(
                json.dumps(reduce.to_json(arms_out, decisions), indent=2),
            )
            click.echo()
            click.echo(md)
            spent = sum(a.total_cost_usd + a.unmeasured_cost_usd for a in arms_out)
            click.echo(f'\nTotal campaign cost (panels + matcher): ${spent:.2f}')
            click.echo(f'Report saved to: {results_dir / "campaign_report.md"}')
            return 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    cli()
