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
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
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
        click.echo(f'{"Diff":<26s} {"Lang":<5s} {"Source":<10s} {"Recall":>7s} {"BlkRcl":>7s} {"F1":>7s} {"TotCost":>7s}')
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
                f'${score.total_cost_usd:>6.2f}'
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
        total_cost = sum(s.total_cost_usd for s in scores)

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

_MINE_DEFAULT_DB = Path('/home/leo/src/dark-factory/data/orchestrator/runs.db')
_MINE_DEFAULT_ESC_DIR = Path('/home/leo/src/dark-factory/data/escalations')
_MINE_DEFAULT_REPO = Path('/home/leo/src/dark-factory')
_MINE_SPLIT_SEED = 'reviewer_trial-2495-fn-mining'
_RESAVE_BATCH_SIZE = 10  # persist every K successfully-labeled diffs (not every 1) to bound
                          # the O(n^2) full-corpus manifest+adjudication-log rewrite cost

_LANG_EXTENSIONS = {
    'python': {'.py', '.pyi'},
    'rust': {'.rs'},
    'typescript': {'.ts', '.tsx'},
}


def _infer_language(diff_text: str) -> str:
    """Best-effort language sniff from a unified diff's ``diff --git`` headers.

    Tallies changed-file extensions against the corpus's three known
    languages and returns the majority; defaults to 'python' (dark_factory
    is an all-Python monorepo) when no file matches or the diff carries no
    recognizable header.
    """
    paths = re.findall(r'^diff --git a/(\S+) b/\S+', diff_text, re.M)
    counts: Counter[str] = Counter()
    for p in paths:
        suffix = Path(p).suffix
        for lang, exts in _LANG_EXTENSIONS.items():
            if suffix in exts:
                counts[lang] += 1
                break
    if not counts:
        return 'python'
    return counts.most_common(1)[0][0]


_FETCH_TITLES_CHUNK_SIZE = 500  # stay well under SQLite's SQLITE_MAX_VARIABLE_NUMBER
                                  # (999 on older builds, 32766 on modern ones) -- the
                                  # candidate set scales with run history and is unbounded.


def _fetch_titles(db_path: Path, task_ids: list[str]) -> dict[str, str]:
    """Supplementary read-only lookup of ``task_results.title``, keyed by task_id.

    Kept separate from ``mining.mine_fn_candidates`` (whose FnCandidate field
    set is unit-tested) -- this is CLI-only convenience for building a
    human-readable frontier-prompt description.

    Queries in chunks of ``_FETCH_TITLES_CHUNK_SIZE`` task_ids rather than one
    giant ``IN (...)`` so an unbounded candidate set never risks a
    ``sqlite3.OperationalError`` from exceeding SQLite's placeholder limit.
    """
    if not task_ids:
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        titles: dict[str, str] = {}
        for i in range(0, len(task_ids), _FETCH_TITLES_CHUNK_SIZE):
            chunk = task_ids[i:i + _FETCH_TITLES_CHUNK_SIZE]
            placeholders = ','.join('?' for _ in chunk)
            rows = conn.execute(
                f'SELECT task_id, title FROM task_results WHERE task_id IN ({placeholders})',
                chunk,
            ).fetchall()
            titles.update((row[0], row[1] or '') for row in rows)
        return titles
    finally:
        conn.close()


def _select_spot_check_subset(diff_ids: list[str], fraction: float = 0.1, minimum: int = 5) -> set[str]:
    """Deterministic, evenly-spread sample of *diff_ids* for the documented
    human spot-check subset (~10% of mined diffs, floor 5, capped at N)."""
    ordered = sorted(diff_ids)
    n = len(ordered)
    if n == 0:
        return set()
    k = min(n, max(minimum, round(n * fraction)))
    step = n / k
    indices = sorted({int(i * step) for i in range(k)})
    return {ordered[i] for i in indices}


@cli.command('mine')
@click.option('--runs-db', default=str(_MINE_DEFAULT_DB), type=click.Path(path_type=Path),
              help='Path to the offline orchestrator runs.db')
@click.option('--escalations-dir', default=str(_MINE_DEFAULT_ESC_DIR), type=click.Path(path_type=Path),
              help='Path to the offline escalations directory')
@click.option('--repo-path', default=str(_MINE_DEFAULT_REPO), type=click.Path(path_type=Path),
              help='Git repo to recover mined diffs from')
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
    runs_db: Path, escalations_dir: Path, repo_path: Path, target_total: int,
    max_parallel: int, model: str, max_turns: int, seed: str, oauth_token_env: str,
    limit: int | None, min_diffs: int,
) -> None:
    """Mine FN-candidate diffs from runs.db/escalations, frontier-label them, and
    expand + re-split the committed corpus (task 2495 / PRD D-6).

    Idempotent and resumable: re-running skips diff_ids already present in the
    corpus, and persists (manifest + adjudication log, re-split every time)
    every ``_RESAVE_BATCH_SIZE`` successfully-labeled diffs (plus once more at
    the end of the run) -- bounding the full-corpus rewrite cost to roughly
    ``total_added / _RESAVE_BATCH_SIZE`` saves instead of one per diff. An
    interrupted run loses at most the current in-progress batch.
    """

    async def _run() -> int:
        from .adjudication import AdjudicationLog
        from .corpus import CorpusDiff
        from .mining import (
            FnCandidate,
            assign_split,
            audit_corpus,
            mine_escalation_refs,
            mine_fn_candidates,
            propose_labels_frontier,
            recover_diff,
            resolve_merge_sha_by_task_id,
        )

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

        def _resave() -> None:
            """Persist manifest + adjudication log, re-deriving split + the
            spot-check subset from the CURRENT full diff set. Synchronous (no
            ``await`` inside) so it's safe to call from within a worker's
            critical section -- called every ``_RESAVE_BATCH_SIZE`` successful
            mines (plus a final flush after the batch completes, including on
            interrupt/error) rather than after every single one, bounding the
            full-corpus rewrite cost; an interrupted run leaves a fully
            consistent, resumable on-disk corpus as of the last save."""
            mined_ids = sorted(d.diff_id for d in manifest.diffs if d.source == 'mined')
            subset = _select_spot_check_subset(mined_ids) if mined_ids else set()
            entries_by_id = {e.diff_id: e for e in log.entries}
            for diff_id in mined_ids:
                entry = entries_by_id.get(diff_id)
                if entry is not None:
                    entry.in_spot_check_subset = diff_id in subset

            assignment = assign_split([d.diff_id for d in manifest.diffs], seed=seed)
            for d in manifest.diffs:
                d.split = assignment.get(d.diff_id)
            manifest.split_seed = seed

            manifest.save(_CORPUS_PATH)
            log.save(_ADJUDICATION_LOG_PATH)

        # Backfill adjudication coverage for pre-existing hand-authored diffs
        # (never fabricated as "frontier" -- logged with their own original
        # ground truth, tagged with a frontier_model that names them as
        # hand-authored so the provenance is honest).
        logged_ids = {e.diff_id for e in log.entries}
        backfilled = 0
        for diff in manifest.diffs:
            if diff.diff_id in logged_ids or diff.source == 'mined':
                continue
            log.append(
                diff.diff_id,
                frontier_proposal=diff.ground_truth,
                in_spot_check_subset=False,
                frontier_model='hand-authored',
                notes=(
                    'Pre-existing hand-authored ground truth from the original '
                    '15-diff corpus (predates task 2495 FN mining); not frontier-proposed.'
                ),
            )
            logged_ids.add(diff.diff_id)
            backfilled += 1
        if backfilled:
            click.echo(f'  Backfilled {backfilled} adjudication entries for pre-existing hand-authored diffs.')
            _resave()

        candidates = mine_fn_candidates(runs_db)
        esc_refs = mine_escalation_refs(escalations_dir)
        click.echo(f'  {len(candidates)} FN candidates mined from runs.db')

        titles = _fetch_titles(runs_db, [c.task_id for c in candidates])

        existing_ids = {d.diff_id for d in manifest.diffs}
        recovered: list[tuple[FnCandidate, str, str]] = []
        for c in candidates:
            diff_id = f'mined_{c.task_id}'
            if diff_id in existing_ids:
                continue
            sha = c.merge_sha or resolve_merge_sha_by_task_id(c.task_id, repo_path)
            if not sha:
                continue
            text = recover_diff(sha, repo_path)
            if not text or not text.strip():
                continue
            recovered.append((c, sha, text))

        # Smallest diffs first: propose_labels_frontier truncates its prompt
        # context to 10K chars, so prioritizing smaller (fully-visible) diffs
        # both biases toward higher-fidelity frontier labels and, since most
        # recovered merge-commit diffs are far larger than 10K chars, keeps
        # per-call prompt size (and cost) down across the batch.
        recovered.sort(key=lambda item: len(item[2]))

        needed = max(0, target_total - len(manifest.diffs))
        if needed == 0:
            pool = []
        else:
            pool_size = needed + max(5, needed // 4)  # buffer for attrition
            pool = recovered[:pool_size]
        if limit is not None:
            pool = pool[:limit]

        click.echo(f'  {len(recovered)} candidates have a recoverable diff not already in the corpus')
        click.echo(f'  Attempting {len(pool)} this run (target_total={target_total}, limit={limit})')
        click.echo()

        semaphore = asyncio.Semaphore(max(1, max_parallel))
        total = len(pool)
        state = {'completed': 0, 'added': 0, 'cost': 0.0}

        async def _worker(candidate: FnCandidate, sha: str, diff_text: str) -> None:
            async with semaphore:
                title = titles.get(candidate.task_id, '')
                description = (
                    f'Task {candidate.task_id} ({candidate.project_id}): {title}. '
                    f'Flagged as a false-negative candidate -- review passed but '
                    f'{"; ".join(candidate.signal_reason)}.'
                )
                try:
                    issues, cost = await propose_labels_frontier(
                        diff_text, model=model, description=description,
                        oauth_token=oauth_token, max_turns=max_turns,
                    )
                except Exception as exc:  # noqa: BLE001 -- offline generation script; one bad candidate must not sink the batch
                    logging.getLogger(__name__).warning(
                        'propose_labels_frontier failed for task %s: %s', candidate.task_id, exc,
                    )
                    issues, cost = [], 0.0

            state['completed'] += 1
            state['cost'] += cost
            diff_id = f'mined_{candidate.task_id}'
            if not issues:
                click.echo(f'  [{state["completed"]}/{total}] {diff_id}: no issues proposed -- skipping (cost=${cost:.3f})')
                return

            provenance = {
                'kind': 'mined',
                'task_id': candidate.task_id,
                'project_id': candidate.project_id,
                'outcome': candidate.outcome,
                'review_cycles': candidate.review_cycles,
                'verify_attempts': candidate.verify_attempts,
                'signal_reason': candidate.signal_reason,
                'merge_sha': sha,
                'runs_db_path': str(runs_db),
                'runs_db_query': (
                    'SELECT task_id, project_id, outcome, review_cycles, verify_attempts FROM '
                    'task_results WHERE review_cycles >= 1, cross-referenced with events '
                    '(escalation_created, merge_finalized) -- see mining.mine_fn_candidates.'
                ),
                'escalation_refs': [
                    {
                        'task_id': ref.task_id,
                        'category': ref.category,
                        'severity': ref.severity,
                        'summary': ref.summary,
                        'level': ref.level,
                        'path': str(ref.path),
                    }
                    for ref in esc_refs.get(candidate.task_id, [])
                ],
            }
            diff = CorpusDiff(
                diff_id=diff_id,
                language=_infer_language(diff_text),
                source='mined',
                diff_text=diff_text,
                description=description,
                ground_truth=issues,
                project='dark-factory',
                split=None,
                provenance=provenance,
            )
            manifest.diffs.append(diff)
            log.append(
                diff.diff_id,
                frontier_proposal=diff.ground_truth,
                in_spot_check_subset=False,  # corrected by _resave()'s stratified pass
                frontier_model=model,
                notes=f'Frontier-proposed labels for FN candidate task {candidate.task_id}.',
            )
            _resave()
            state['added'] += 1
            click.echo(
                f'  [{state["completed"]}/{total}] {diff_id}: {len(issues)} issues added '
                f'(cost=${cost:.3f}, corpus now {len(manifest.diffs)})'
            )

        await asyncio.gather(*(_worker(c, sha, text) for c, sha, text in pool))

        click.echo()
        click.echo(f'Mined {state["added"]}/{total} new diffs this run (frontier cost: ${state["cost"]:.2f}).')
        click.echo(f'Corpus: {len(manifest.diffs)} diffs total.')

        report = audit_corpus(manifest, log, min_diffs=50)
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
        from .runner import run_trial
        from .scorer import score_panel_run
        from .variants import ALL_VARIANTS

        corpus = _load_corpus()

        click.echo(click.style(f'Full Trial: {len(ALL_VARIANTS)} variants x {len(corpus.diffs)} diffs', bold=True))
        click.echo('=' * 60)
        for v in ALL_VARIANTS:
            click.echo(f'  {v.name:15s} {v.description} ({len(v.reviewers)} reviewers)')
        click.echo()

        # Run all variants
        click.echo('Running panels...')
        results = await run_trial(ALL_VARIANTS, corpus, max_parallel_panels=max_parallel)
        click.echo(f'  Completed {len(results)} panel runs.')

        # Score all results
        click.echo('\nScoring results...')
        scores = []
        for result in results:
            diff = corpus.get_diff(result.diff_id)
            if diff is None:
                continue
            score = await score_panel_run(result, diff)
            scores.append(score)
            click.echo(f'  {result.variant_name:15s} x {result.diff_id:<25s} F1={score.f1:.3f} BR={score.blocking_recall:.3f}')

        # Build and save report
        report = build_trial_report(scores, ALL_VARIANTS, corpus)
        out_dir = report_dir or _RESULTS_DIR
        report_path = save_report(report, out_dir / 'full_trial_report')

        # Print the full markdown report
        click.echo()
        click.echo(format_markdown(report))
        click.echo(f'\nReport saved to: {report_path}')

        total_cost = sum(r.total_cost_usd for r in results)
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
        from .runner import run_trial
        from .scorer import score_panel_run
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

        click.echo('Running panels...')
        results = await run_trial(EFFORT_SWEEP_VARIANTS, corpus, max_parallel_panels=max_parallel)
        click.echo(f'  Completed {len(results)} panel runs.')

        click.echo('\nScoring results...')
        scores = []
        for result in results:
            diff = corpus.get_diff(result.diff_id)
            if diff is None:
                continue
            score = await score_panel_run(result, diff)
            scores.append(score)
            click.echo(f'  {result.variant_name:18s} x {result.diff_id:<25s} F1={score.f1:.3f} BR={score.blocking_recall:.3f}')

        report = build_trial_report(scores, EFFORT_SWEEP_VARIANTS, corpus)
        report_path = save_report(report, _RESULTS_DIR / 'effort_sweep_report')

        click.echo()
        click.echo(format_markdown(report))
        click.echo(f'\nReport saved to: {report_path}')

        total_cost = sum(r.total_cost_usd for r in results)
        click.echo(f'Total panel cost: ${total_cost:.2f}')
        return 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    cli()
