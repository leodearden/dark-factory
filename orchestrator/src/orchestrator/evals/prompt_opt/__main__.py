"""CLI for the prompt-optimization operator commands.

Usage:
    cd orchestrator && uv run python -m orchestrator.evals.prompt_opt <command>

Commands:
    build-curator-corpus   Build the labeled T5 curator replay corpus from
                            tickets.db (task 2496 / PRD D-6). See
                            curator_replay_corpus/README.md for the schema,
                            split, and human spot-check sign-off protocol.
    canary                 Compare a post-deploy runs.db window against a
                            rolling pre-deploy baseline window and emit a
                            pass/regress verdict (PRD T8 / D-7). See
                            plans/tier1-prompt-optimization-runbook.md for the
                            pin -> watch canary -> unpin-on-regress flow.
    smoke                  Run the hermetic end-to-end acceptance smoke (PRD
                            T7): the whole reviewer prompt-opt stack (loader ->
                            HEURISTICS -> loop -> scorer -> report) on a
                            <=3-diff synthetic fixture corpus, with NO real
                            LLM/DB/network. Exit 0 iff the run completes and the
                            reviewer CONTRACT stays byte-identical -- the cheap
                            gate that de-risks the $300-800 real loops (§8).

Mirrors orchestrator/evals/reviewer_trial/__main__.py's click pattern. Both
commands are operator-facing thin wiring over already-tested machinery
(curator_corpus.py / canary.py) -- every function they call is unit-tested
hermetically against synthetic fixtures, never the real DB or a real LLM.
This file adds only real I/O (a real tickets.db or runs.db path, and for
build-curator-corpus real frontier-model calls) plus, for build-curator-corpus,
a CLI-local cost-accumulator closure (sums FrontierLabel.cost_usd across the
run for the printed summary).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from shared.prompt_artifact import default_artifacts_root

from orchestrator.evals.prompt_opt.canary import (
    CanaryThresholds,
    MetricComparison,
    deploy_at_from_provenance,
    run_canary,
)

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'

_PKG_DIR = Path(__file__).parent
_CURATOR_CORPUS_DIR = _PKG_DIR / 'curator_replay_corpus'
_CURATOR_MANIFEST_PATH = _CURATOR_CORPUS_DIR / 'manifest.json'

# Overridable via env var for portability across checkouts -- tickets.db is
# offline, gitignored operational data that lives in the MAIN repo, not a
# task worktree (mirrors reviewer_trial's REVIEWER_TRIAL_RUNS_DB pattern;
# see corpus/README.md's "Regenerating / extending the corpus").
_DEFAULT_TICKETS_DB = Path(os.environ.get(
    'CURATOR_TICKETS_DB', '/home/leo/src/dark-factory/data/reconciliation/tickets.db',
))
# ~100 labeled decisions matches D-6's "make a 2:1:7 split meaningful"
# floor for the reviewer corpus (target ~100, >=~50 minimum); curator
# labels are cheaper per-item (one frontier call, no diff text to read) so
# 100 is a conservative default an operator can raise via --n.
_DEFAULT_N = 100
_DEFAULT_SEED = 2496
# Mirrors curator_corpus.py's own _DEFAULT_SPOT_CHECK_CAP -- the fraction/
# minimum defaults inside select_spot_check_subset already bound the
# per-stratum sample; this cap only bites for a much larger --n than the
# default above.
_DEFAULT_SPOT_CHECK_SIZE = 200

# (check key, human-readable description) -- order matches
# curator_corpus.audit_curator_corpus's check ordering so PASS/FAIL output
# reads top-to-bottom in the same sequence the report's `failures` would.
_AUDIT_CHECKS = (
    ('item_count', 'item count meets the minimum floor'),
    ('missing_split', 'every item has a split assigned'),
    ('split_ratio', 'train/selection/test ratio approximates 2:1:7'),
    ('missing_gold_label', 'every item carries a gold label'),
    ('adjudication_coverage', 'adjudication_log covers every ticket_id'),
    ('spot_check_subset_empty', 'a non-empty documented spot-check subset exists'),
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

# Deterministic 2:1:7 corpus split seed for the acceptance smoke -- mirrors
# smoke._DEFAULT_SPLIT_SEED (the task number). A local literal (like
# _DEFAULT_SEED above) so the light `smoke` command need not import the heavy
# smoke module -- which pulls in agents.roles + the loop engine -- just to read
# a decorator default; the real run imports it lazily inside the command body.
_SMOKE_DEFAULT_SPLIT_SEED = 2498


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_check(passed: bool, message: str) -> bool:
    tag = click.style('[PASS]', fg='green') if passed else click.style('[FAIL]', fg='red')
    click.echo(f'{tag} {message}')
    return passed


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
@click.option('--verbose', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Prompt-optimization operator CLI: T5 curator replay corpus + T8 canary."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt='%H:%M:%S', stream=sys.stderr)


@cli.command('build-curator-corpus')
@click.option('--db-path', default=str(_DEFAULT_TICKETS_DB), type=click.Path(path_type=Path),
              help='Path to the offline tickets.db (env: CURATOR_TICKETS_DB)')
@click.option('--n', default=_DEFAULT_N, type=int,
              help='Max labeled decisions to sample (stratified by recorded action)')
@click.option('--seed', default=_DEFAULT_SEED, type=int, help='Deterministic sampling/split seed')
@click.option('--spot-check-size', default=_DEFAULT_SPOT_CHECK_SIZE, type=int,
              help='Cap on the human spot-check subset size')
@click.option('--out', default=str(_CURATOR_MANIFEST_PATH), type=click.Path(path_type=Path),
              help="Where to write manifest.json ('annotations/' and adjudication_log.jsonl "
                   'are written alongside it)')
def build_curator_corpus_cmd(db_path: Path, n: int, seed: int, spot_check_size: int, out: Path) -> None:
    """Build the T5 curator replay corpus from tickets.db (task 2496 / PRD D-6).

    Thin wiring over already-tested machinery (curator_corpus.py): reads
    recorded curator decisions from *db_path*, samples up to *n* stratified
    by recorded action, obtains a GOLD label for each sampled candidate from
    a real frontier-model call (propose_curator_label_frontier -- NEVER the
    recorded decision; PRD D-6: decisions != ground truth), flags a
    deterministic human spot-check subset capped at *spot_check_size*,
    assigns a 2:1:7 split, and writes manifest.json + annotations/*.json +
    adjudication_log.jsonl under *out*'s directory. Prints an
    audit_curator_corpus PASS/FAIL summary and exits non-zero on FAIL.
    """
    from .curator_corpus import (
        FrontierLabel,
        audit_curator_corpus,
        build_curator_corpus,
        propose_curator_label_frontier,
    )

    async def _run() -> int:
        if not db_path.is_file():
            click.echo(
                f'tickets.db not found at {db_path}\n'
                '  This is an offline generation tool that reads the gitignored '
                'reconciliation ticket store (data/reconciliation/tickets.db in the '
                'MAIN repo checkout, not a task worktree -- see '
                'curator_replay_corpus/README.md). Pass --db-path, or set '
                'CURATOR_TICKETS_DB, to point at a real tickets.db.',
                err=True,
            )
            return 1

        click.echo(click.style(f'Build Curator Corpus: n={n} seed={seed} db={db_path}', bold=True))
        click.echo('=' * 60)

        # propose_curator_label_frontier reports spend on FrontierLabel.cost_usd
        # rather than as a second return value (FrontierProposerFn's contract is
        # a plain Callable[[dict], Awaitable[FrontierLabel]]) -- wrap it here to
        # accumulate the total across every sampled candidate so operator spend
        # is visible, not just implicitly billed.
        cost_state = {'usd': 0.0}

        async def _proposer_with_cost_tracking(candidate: dict) -> FrontierLabel:
            label = await propose_curator_label_frontier(candidate)
            cost_state['usd'] += label.cost_usd
            return label

        manifest, log = await build_curator_corpus(
            db_path, n=n, seed=seed, spot_check_size=spot_check_size,
            frontier_proposer=_proposer_with_cost_tracking,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(out)
        adjudication_log_path = out.parent / 'adjudication_log.jsonl'
        log.save(adjudication_log_path)

        click.echo(f'  {len(manifest.items)} items labeled and written to {out}')
        click.echo(f'  Adjudication log: {adjudication_log_path}')
        click.echo(f'  Frontier labeling cost: ${cost_state["usd"]:.2f}')

        report = audit_curator_corpus(manifest, log)
        click.echo()
        click.echo(click.style(f'Corpus Audit: {report.item_count} items', bold=True))
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
        return 0 if report.ok else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


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
        if verdict.regressed_metrics:
            click.echo(f'  Regressed metrics: {", ".join(verdict.regressed_metrics)}')
        else:
            # Forced regress with no individual metric over threshold --
            # the baseline-had-done/post-has-none rule (see the NOTE
            # printed above and compare_windows's docstring).
            click.echo(
                '  Regressed: post window completed ZERO tasks despite the '
                'baseline completing work -- forced regress even though no '
                'individual metric crossed its threshold (see NOTE above).'
            )
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


@cli.command('smoke')
@click.option(
    '--artifacts-root', default=lambda: str(default_artifacts_root()),
    type=click.Path(path_type=Path),
    help='Isolated PromptArtifactStore root to pin the smoke artifact into '
         '(default: default_artifacts_root(); pass a throwaway/tmp root for CI). '
         'The pinned heuristics carry the hermetic SMOKE_QUALITY sentinel, so '
         'point this at a scratch root, never a production artifacts root.',
)
@click.option('--split-seed', default=_SMOKE_DEFAULT_SPLIT_SEED, type=int,
              help='Deterministic 2:1:7 corpus split seed for the held-out TEST verdict')
def smoke(artifacts_root: Path, split_seed: int) -> None:
    """Run the hermetic end-to-end Tier-1 prompt-opt acceptance smoke (PRD T7).

    Fully hermetic -- dependency-injects the three LLM-touching seams
    (rollout_fn / Scorer / propose_fn), so it makes NO real LLM call and
    touches no runs.db/tickets.db/network. Runs the REAL
    REVIEWER_COMPREHENSIVE.prompt_spec through the REAL run_optimization_loop
    + PromptArtifactStore loader over a <=3-diff synthetic fixture corpus,
    pins the loop's final heuristics, resolves them back, and proves the
    reviewer CONTRACT prefix is byte-identical.

    Thin wiring over already-tested machinery (smoke.py): prints
    format_markdown(report) plus a PASS/FAIL banner carrying the held-out
    test_score, the provenance sidecar fields, and the CONTRACT-untouched
    proof. Exit 0 iff the run completes AND the CONTRACT is untouched; exit 1
    if the run raises or the CONTRACT prefix is not intact. This is the cheap
    acceptance gate that de-risks the $300-800 real reviewer/curator loops
    (plans/tier1-prompt-optimization-runbook.md §8).
    """
    # Lazy import: smoke.py pulls in agents.roles + the loop engine, so keep it
    # out of module import so the other subcommands stay import-light.
    from .smoke import format_markdown, run_acceptance_smoke

    click.echo(click.style(
        f'Acceptance Smoke: artifacts_root={artifacts_root} split_seed={split_seed}', bold=True,
    ))
    click.echo('=' * 60)

    try:
        report = asyncio.run(run_acceptance_smoke(
            store_root=artifacts_root, split_seed=split_seed,
        ))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 -- surface ANY run failure as a non-zero gate
        click.echo(click.style(
            f'Acceptance smoke FAILED to run: {exc!r}', fg='red', bold=True,
        ), err=True)
        sys.exit(1)

    click.echo(format_markdown(report))
    click.echo()

    prov = report.provenance
    intact = report.contract_prefix_intact
    color = 'green' if intact else 'red'
    status = 'PASS' if intact else 'FAIL'
    click.echo(click.style(
        f'Result: {status} -- held_out_TEST_score={report.test_score:.4f} '
        f'harness_version={prov.harness_version} optimizer_model={report.optimizer_model} '
        f'accepted={report.accepted_count} rejected={report.rejected_count} '
        f'CONTRACT untouched={intact} (resolved source={report.resolved_source})',
        fg=color, bold=True,
    ))
    # Exit 0 only when the run completed AND the reviewer CONTRACT stayed
    # byte-identical -- the gate the operator watches before a real run.
    sys.exit(0 if intact else 1)


if __name__ == '__main__':
    cli()
