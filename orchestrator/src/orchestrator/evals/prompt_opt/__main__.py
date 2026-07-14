"""CLI for the T8 prompt-variant canary (runs.db metric-comparison).

See plans/tier1-prompt-optimization-prd.md T8 / D-7 and
plans/tier1-prompt-optimization-runbook.md for the pin -> watch canary ->
unpin-on-regress operator flow this command serves.

Usage:
    cd orchestrator && uv run python -m orchestrator.evals.prompt_opt canary \
        --deploy-at 2026-07-01T00:00:00+00:00 --project-id dark_factory

Commands:
    canary   Compare a post-deploy runs.db window against a rolling
             pre-deploy baseline window and emit a pass/regress verdict.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import click

from orchestrator.evals.prompt_opt.canary import (
    CanaryThresholds,
    MetricComparison,
    deploy_at_from_provenance,
    run_canary,
)

# Overridable via env var for portability across checkouts -- runs.db is
# offline, gitignored data that lives in the MAIN repo, not a task worktree
# (mirrors reviewer_trial.__main__'s REVIEWER_TRIAL_RUNS_DB idiom).
_CANARY_DEFAULT_DB = Path(os.environ.get(
    'PROMPT_OPT_RUNS_DB', '/home/leo/src/dark-factory/data/orchestrator/runs.db',
))

# Documented PRD §9 calibration STARTING points -- both the window lengths
# below and the per-metric threshold defaults sourced from CanaryThresholds()
# are starting points to be tuned against real runs.db baseline variance,
# never asserted as numeric guarantees (see canary.CanaryThresholds's
# docstring and the plan's design decisions).
_DEFAULT_BASELINE_DAYS = 7
_DEFAULT_POST_DAYS = 7
_DEFAULT_THRESHOLDS = CanaryThresholds()

# regress/insufficient_data get distinct nonzero codes so a caller can tell
# "the deploy looks bad" apart from "we don't know yet" without scraping
# stdout; a pre-flight usage problem (bad flag combo, missing runs.db) is a
# separate concern and uses the conventional usage-error code (2).
_VERDICT_EXIT_CODES = {'pass': 0, 'regress': 1, 'insufficient_data': 3}
_VERDICT_COLORS = {'pass': 'green', 'regress': 'red', 'insufficient_data': 'yellow'}
_USAGE_ERROR_EXIT_CODE = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_metric(comparison: MetricComparison) -> None:
    """One PASS/REGRESS/SKIP line per metric (mirrors reviewer_trial.
    __main__._print_check's tag convention)."""
    if comparison.baseline is None or comparison.post is None:
        tag = click.style('[SKIP]', fg='yellow')
        click.echo(f'{tag} {comparison.metric}: incomparable (no done rows in one window)')
        return

    tag = (
        click.style('[REGRESS]', fg='red')
        if comparison.regressed
        else click.style('[PASS]', fg='green')
    )
    delta_rel = f'{comparison.delta_rel:+.1%}' if comparison.delta_rel is not None else 'n/a'
    click.echo(
        f'{tag} {comparison.metric}: baseline={comparison.baseline:.4g} '
        f'post={comparison.post:.4g} delta={delta_rel} threshold={comparison.threshold:.4g}'
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """T8 prompt-variant canary: runs.db metric-comparison + deploy verdict."""


@cli.command('canary')
@click.option('--runs-db', default=str(_CANARY_DEFAULT_DB), type=click.Path(path_type=Path),
              help='Path to the offline orchestrator runs.db (env: PROMPT_OPT_RUNS_DB)')
@click.option('--deploy-at', default=None,
              help='ISO-8601 deploy timestamp. Required unless --prompt-id/--executor-model/'
                   '--harness-version are all given (resolves it from T1 provenance instead)')
@click.option('--prompt-id', default=None,
              help='T1 pinned prompt_id -- resolves --deploy-at via the T1 provenance sidecar '
                   '(requires --executor-model and --harness-version too)')
@click.option('--executor-model', default=None, help='T1 key component, used with --prompt-id')
@click.option('--harness-version', default=None, help='T1 key component, used with --prompt-id')
@click.option('--artifacts-root', default=None, type=click.Path(path_type=Path),
              help='Override the T1 PromptArtifactStore root (default: default_artifacts_root()); '
                   'only used with --prompt-id')
@click.option('--project-id', required=True, help='project_id to filter runs.db rows by')
@click.option('--baseline-days', default=_DEFAULT_BASELINE_DAYS, type=int,
              help='Rolling pre-deploy baseline window length in days')
@click.option('--post-days', default=_DEFAULT_POST_DAYS, type=int,
              help='Post-deploy window length in days')
@click.option('--cost-per-done-task-rel-tol', type=float,
              default=_DEFAULT_THRESHOLDS.cost_per_done_task_rel_tol)
@click.option('--cost-per-done-task-abs-floor', type=float,
              default=_DEFAULT_THRESHOLDS.cost_per_done_task_abs_floor)
@click.option('--requeue-rate-rel-tol', type=float,
              default=_DEFAULT_THRESHOLDS.requeue_rate_rel_tol)
@click.option('--requeue-rate-abs-floor', type=float,
              default=_DEFAULT_THRESHOLDS.requeue_rate_abs_floor)
@click.option('--mean-review-cycles-rel-tol', type=float,
              default=_DEFAULT_THRESHOLDS.mean_review_cycles_rel_tol)
@click.option('--mean-review-cycles-abs-floor', type=float,
              default=_DEFAULT_THRESHOLDS.mean_review_cycles_abs_floor)
@click.option('--mean-verify-attempts-rel-tol', type=float,
              default=_DEFAULT_THRESHOLDS.mean_verify_attempts_rel_tol)
@click.option('--mean-verify-attempts-abs-floor', type=float,
              default=_DEFAULT_THRESHOLDS.mean_verify_attempts_abs_floor)
@click.option('--min-samples', type=int, default=_DEFAULT_THRESHOLDS.min_samples,
              help="Minimum rows required in EACH window, else verdict='insufficient_data'")
def canary(
    runs_db: Path,
    deploy_at: str | None,
    prompt_id: str | None,
    executor_model: str | None,
    harness_version: str | None,
    artifacts_root: Path | None,
    project_id: str,
    baseline_days: int,
    post_days: int,
    cost_per_done_task_rel_tol: float,
    cost_per_done_task_abs_floor: float,
    requeue_rate_rel_tol: float,
    requeue_rate_abs_floor: float,
    mean_review_cycles_rel_tol: float,
    mean_review_cycles_abs_floor: float,
    mean_verify_attempts_rel_tol: float,
    mean_verify_attempts_abs_floor: float,
    min_samples: int,
) -> None:
    """Compare pipeline metrics over a post-deploy window vs a rolling
    pre-deploy baseline window read from runs.db, and emit a pass/regress
    verdict (PRD T8 / D-7).

    Exit code is 0 on 'pass', 1 on 'regress', 3 on 'insufficient_data' (2 is
    reserved for a pre-flight usage problem) -- see
    plans/tier1-prompt-optimization-runbook.md for the operator flow this
    feeds: pin -> watch this command -> unpin (T1
    PromptArtifactStore.unpin) on regress.
    """
    if deploy_at is None:
        if not (prompt_id and executor_model and harness_version):
            click.echo(
                'Must pass either --deploy-at, or all three of --prompt-id / '
                '--executor-model / --harness-version to resolve it from the T1 '
                'provenance sidecar.',
                err=True,
            )
            sys.exit(_USAGE_ERROR_EXIT_CODE)
        resolved = deploy_at_from_provenance(
            prompt_id, executor_model, harness_version, root=artifacts_root,
        )
        if resolved is None:
            click.echo(
                f'No pinned T1 provenance found for prompt_id={prompt_id!r} '
                f'executor_model={executor_model!r} harness_version={harness_version!r}.',
                err=True,
            )
            sys.exit(_USAGE_ERROR_EXIT_CODE)
        deploy_at = resolved
        click.echo(f'Resolved --deploy-at={deploy_at} from T1 provenance.')

    try:
        datetime.fromisoformat(deploy_at)
    except ValueError:
        click.echo(
            f'Invalid --deploy-at value: {deploy_at!r} -- must be an ISO-8601 datetime '
            '(e.g. 2026-07-01T00:00:00+00:00) parseable by datetime.fromisoformat.',
            err=True,
        )
        sys.exit(_USAGE_ERROR_EXIT_CODE)

    if not runs_db.is_file():
        click.echo(
            f'runs.db not found at {runs_db}\n'
            '  This reads the gitignored orchestrator run history '
            '(data/orchestrator/runs.db in the MAIN repo checkout, not a task '
            'worktree). Pass --runs-db, or set PROMPT_OPT_RUNS_DB.',
            err=True,
        )
        sys.exit(_USAGE_ERROR_EXIT_CODE)

    thresholds = CanaryThresholds(
        cost_per_done_task_rel_tol=cost_per_done_task_rel_tol,
        cost_per_done_task_abs_floor=cost_per_done_task_abs_floor,
        requeue_rate_rel_tol=requeue_rate_rel_tol,
        requeue_rate_abs_floor=requeue_rate_abs_floor,
        mean_review_cycles_rel_tol=mean_review_cycles_rel_tol,
        mean_review_cycles_abs_floor=mean_review_cycles_abs_floor,
        mean_verify_attempts_rel_tol=mean_verify_attempts_rel_tol,
        mean_verify_attempts_abs_floor=mean_verify_attempts_abs_floor,
        min_samples=min_samples,
    )

    click.echo(click.style(f'Canary: project_id={project_id} deploy_at={deploy_at}', bold=True))
    click.echo(f'  runs_db={runs_db}  baseline_days={baseline_days}  post_days={post_days}')
    click.echo('=' * 60)

    verdict = run_canary(
        runs_db, deploy_at,
        baseline_days=baseline_days, post_days=post_days,
        project_id=project_id, thresholds=thresholds,
    )

    click.echo()
    click.echo(f'Baseline window: {verdict.baseline_n} rows   Post window: {verdict.post_n} rows')
    zero_done_windows = [
        label for label, n_done in (
            ('baseline', verdict.baseline_n_done), ('post', verdict.post_n_done),
        )
        if n_done == 0
    ]
    if zero_done_windows:
        click.echo(click.style(
            f'  NOTE: {" and ".join(zero_done_windows)} window has ZERO done rows -- '
            'cost_per_done_task/mean_review_cycles/mean_verify_attempts carry no signal '
            'there (incomparable / 0.0 fallback); only requeue_rate reflects real deploy '
            'behavior for that window. A verdict computed this way understates risk.',
            fg='yellow',
        ))
    click.echo()
    for comparison in verdict.comparisons:
        _print_metric(comparison)

    click.echo()
    click.echo(click.style(
        f'Verdict: {verdict.verdict.upper()}', fg=_VERDICT_COLORS[verdict.verdict], bold=True,
    ))

    if verdict.verdict == 'insufficient_data':
        click.echo(
            f'  Need >= {min_samples} rows in EACH window (baseline={verdict.baseline_n}, '
            f'post={verdict.post_n}). Widen --baseline-days/--post-days, or wait for more '
            'post-deploy data and re-run.'
        )
    elif verdict.verdict == 'regress':
        click.echo(f'  Regressed metrics: {", ".join(verdict.regressed_metrics)}')
        if prompt_id and executor_model and harness_version:
            click.echo(
                f'  ACTION: PromptArtifactStore(...).unpin({prompt_id!r}, {executor_model!r}, '
                f'{harness_version!r}) -- see plans/tier1-prompt-optimization-runbook.md.'
            )
        else:
            click.echo(
                '  ACTION: unpin the deployed artifact -- '
                'shared.prompt_artifact.PromptArtifactStore.unpin(prompt_id, executor_model, '
                'harness_version) -- see plans/tier1-prompt-optimization-runbook.md.'
            )

    sys.exit(_VERDICT_EXIT_CODES[verdict.verdict])


if __name__ == '__main__':
    cli()
