"""Every ``orchestrator eval*`` command and the drivers behind them."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from shared.project_layout import runs_db, tasks_db

from orchestrator.cli_config import CONFIG_OPTION_HELP_PREFIX, pass_config
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.main_checkout_defaults import cli_default
from orchestrator.routing import RoleDefaults, RouteInputs, RoutingDecision, resolve_route
from orchestrator.shadow_types import ShadowShape

PACKAGED_TASKS_DIR = Path(__file__).parent / 'tasks'

eval_cli = click.Group('eval-cli')


@eval_cli.command('eval')
@click.option('--task', 'task_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to a single task JSON file')
@click.option('--config-name', default=None,
              help='Eval config name (e.g. claude-opus-high) or "all"')
@click.option('--matrix', is_flag=True, help='Run full eval matrix (all tasks × all configs)')
@click.option('--judge', is_flag=True, help='Run Elo-based LLM judge on existing results')
@click.option('--plan-only', is_flag=True, help='Generate plans for tasks (no execution)')
@pass_config()
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--trials', type=int, default=1,
              help='Number of trials per (task, config) pair')
@click.option('--force', is_flag=True, help='Re-run even if results exist')
@click.option('--cleanup', is_flag=True, help='Remove eval worktrees')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
@click.option('--max-rounds', type=int, default=50,
              help='Max judge invocations per task (default: 50)')
@click.option('--reset', is_flag=True, help='Clear judge state and start fresh')
@click.option('--report', 'report_only', is_flag=True,
              help='Generate report from existing state (no new judge calls)')
@click.option('--vllm-url', default=None,
              help='vLLM endpoint URL (e.g. http://workstation:8000)')
@click.option('--worktree', 'worktree_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Reuse an existing eval worktree (skip create + use its .task state)')
@click.option('--compare', nargs=2, default=None,
              help='Compare two model groups head-to-head via LLM assessment. '
                   'Each arg is a config name (use canonical name from '
                   '--combine-runs if merging).')
@click.option('--combine-runs', multiple=True,
              help='Merge config names as one model (comma-separated, first is '
                   'canonical). Can be supplied multiple times.')
def eval_cmd(
    task_path: Path | None,
    config_name: str | None,
    matrix: bool,
    judge: bool,
    plan_only: bool,
    config: OrchestratorConfig,
    max_parallel: int | None,
    trials: int,
    force: bool,
    cleanup: bool,
    timeout: int | None,
    max_rounds: int,
    reset: bool,
    report_only: bool,
    vllm_url: str | None,
    worktree_path: Path | None,
    compare: tuple[str, str] | None,
    combine_runs: tuple[str, ...],
):
    """Run multi-provider implementor evaluations."""
    # Inject ANTHROPIC_BASE_URL into vLLM configs when --vllm-url is set
    if vllm_url:
        from orchestrator.evals.configs import FINAL_RUN_CONFIGS, VLLM_EVAL_CONFIGS
        for cfg in [*VLLM_EVAL_CONFIGS, *FINAL_RUN_CONFIGS]:
            if cfg.env_overrides:  # only vLLM configs have env_overrides
                cfg.env_overrides['ANTHROPIC_BASE_URL'] = vllm_url

    if cleanup:
        _run_cleanup(config)
        return

    if report_only:
        _run_report_cmd()
        return

    if compare:
        _run_compare_cmd(compare, combine_runs)
        return

    if judge:
        _run_judge_cmd(max_rounds=max_rounds, reset=reset)
        return

    if plan_only:
        _run_plan_only(task_path, config)
        return

    if matrix:
        _run_matrix_cmd(
            config,
            max_parallel=max_parallel, trials=trials,
            force=force, timeout=timeout,
        )
        return

    if task_path is None:
        click.echo('Error: --task is required (or use --matrix / --judge / --plan-only)', err=True)
        sys.exit(1)

    _run_single_eval(
        task_path, config_name, config,
        force=force, timeout=timeout, worktree_path=worktree_path,
    )


def _load_fixture_dir(tasks_dir: Path) -> list[dict]:
    """Load every ``*.json`` eval fixture in *tasks_dir* (empty list if absent).

    A malformed or truncated fixture (e.g. an interrupted ``eval-sample`` write)
    is surfaced loudly with the offending file NAMED — a clean CLI error rather
    than an opaque ``JSONDecodeError`` traceback with no filename context.
    """
    fixtures: list[dict] = []
    if tasks_dir.exists():
        for tp in sorted(tasks_dir.glob('*.json')):
            try:
                fixtures.append(json.loads(tp.read_text()))
            except json.JSONDecodeError as exc:
                raise click.ClickException(
                    f'malformed eval fixture {tp}: {exc}'
                ) from exc
    return fixtures


@eval_cli.command('eval-list-fixtures')
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir to list (default: orchestrator/evals/tasks)')
@click.option('--cohort', default=None,
              help='Scope the stratification to one cohort (e.g. revival-zeta)')
@click.option('--audit', is_flag=True,
              help='Also run the corpus audit (band + per-fixture completeness) '
                   'over ONE cohort; exits non-zero on any audit failure. '
                   'Scopes to --cohort when given, else defaults to the '
                   'revival-zeta band cohort — so a bare --audit never flags '
                   'the retained legacy fixtures (band overflow / unpinned '
                   'branches) as failures.')
def eval_list_fixtures_cmd(tasks_dir: Path | None, cohort: str | None, audit: bool):
    """Print the eval-fixture stratification (repo×kind×path) counts.

    Reads only the fixtures dir — needs no orchestrator config.
    """
    from orchestrator.evals.task_sampler import (
        DEFAULT_COHORT,
        audit_fixture_corpus,
        format_stratification_table,
        git_ref_exists,
        stratification_counts,
    )
    tasks_dir = tasks_dir or PACKAGED_TASKS_DIR
    fixtures = _load_fixture_dir(tasks_dir)
    counts = stratification_counts(fixtures, cohort=cohort)
    click.echo(format_stratification_table(counts))

    if audit:
        # The band [10,14] + evals/<id> branch-pin checks are revival-ζ
        # invariants. A bare --audit over ALL cohorts would falsely fail on the
        # retained legacy fixtures (df_task_12/13/18, reify_task_12/27): band
        # overflow (>14 total) + missing_branch for every unpinned legacy id.
        # So scope the audit to the requested cohort, defaulting to revival-ζ.
        audit_cohort = cohort if cohort is not None else DEFAULT_COHORT
        scoped = [f for f in fixtures if f.get('cohort') == audit_cohort]
        report = audit_fixture_corpus(scoped, ref_exists=git_ref_exists)
        click.echo('')
        click.echo(f'audit ({audit_cohort}): ok={report.ok} count={report.count}')
        for failure in report.failures:
            click.echo(f'  FAIL {failure}')
        if not report.ok:
            sys.exit(1)


@eval_cli.command('eval-sample')
@click.option('--since', default='6 weeks ago',
              help="git --since discovery window (default: '6 weeks ago')")
@click.option('--target-low', type=int, default=10, help='Band floor (default 10)')
@click.option('--target-high', type=int, default=14, help='Band ceiling (default 14)')
@click.option('--seed', type=int, default=0, help='Deterministic sampling seed')
@click.option('--cohort', default='revival-zeta',
              help='Cohort marker stamped on each cut fixture')
@click.option('--dry-run', is_flag=True,
              help='Print the intended stratified selection without pinning '
                   'branches or writing fixtures')
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Where fixture JSONs are written (default: orchestrator/evals/tasks)')
@click.option('--df-root', type=click.Path(path_type=Path), default=None,
              help='dark_factory checkout to discover merges in (default: the main checkout)')
@click.option('--reify-root', type=click.Path(path_type=Path),
              default=Path('/home/leo/src/reify'),
              help='reify checkout to discover merges in')
@click.option('--sampled-at', default=None,
              help='ISO-8601 provenance timestamp (default: now, UTC)')
def eval_sample_cmd(since: str, target_low: int, target_high: int, seed: int,
                    cohort: str, dry_run: bool, tasks_dir: Path | None,
                    df_root: Path | None, reify_root: Path, sampled_at: str | None):
    """Cut a stratified near-HEAD eval-fixture corpus from both repos.

    Discovers completed-task merges in the df + reify checkouts, samples them
    down to the [target_low, target_high] band round-robin across
    repo×kind×path cells, and (unless --dry-run) captures each reference diff,
    pins its ``evals/<id>`` branch, and writes the fixture JSON. Needs no
    orchestrator config — the seed + sampled_at are supplied here at the CLI
    boundary so the sampler library stays wall-clock-free.
    """
    asyncio.run(_run_eval_sample(
        since=since, target_low=target_low, target_high=target_high, seed=seed,
        cohort=cohort, dry_run=dry_run, tasks_dir=tasks_dir,
        df_root=df_root or cli_default('--df-root', Path), reify_root=Path(reify_root),
        sampled_at=sampled_at,
    ))


async def _run_eval_sample(*, since: str, target_low: int, target_high: int,
                           seed: int, cohort: str, dry_run: bool,
                           tasks_dir: Path | None, df_root: Path,
                           reify_root: Path, sampled_at: str | None):
    """Async worker for ``eval-sample`` (discover → sample → capture/pin/write)."""
    from datetime import UTC, datetime

    from orchestrator.evals.task_sampler import (
        build_fixture_record,
        capture_reference,
        default_verify_commands,
        discover_completed_tasks,
        enrich_candidates_from_task_db,
        pin_eval_branch,
        repo_of,
        sample_stratified,
    )

    tasks_dir = tasks_dir or PACKAGED_TASKS_DIR

    git = GitConfig()
    candidates = []
    for project, root in (('dark_factory', df_root), ('reify', reify_root)):
        if not root.exists():
            click.echo(f'WARNING: {project} checkout {root} not found; skipping',
                       err=True)
            continue
        found = await discover_completed_tasks(
            root, since=since, project=project, project_root=str(root),
            branch_prefix=git.branch_prefix, main_branch=git.main_branch,
        )
        found = enrich_candidates_from_task_db(found, tasks_db(root))
        click.echo(f'discovered {len(found)} completed-task merge(s) in {project}')
        candidates.extend(found)

    result = sample_stratified(
        candidates, target_low=target_low, target_high=target_high, seed=seed,
    )
    click.echo(f'selected {len(result.selected)} of {len(candidates)} candidate(s):')
    for cell, count in sorted(result.cell_counts.items()):
        click.echo(f'  {cell[0]}/{cell[1]}/{cell[2]}: {count}')
    for note in result.notes:
        click.echo(f'  note: {note}')

    if dry_run:
        click.echo('(dry-run) no branches pinned, no fixtures written')
        return

    if sampled_at is None:
        sampled_at = datetime.now(UTC).isoformat()
    tasks_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cand in result.selected:
        repo = repo_of(cand)
        reference = await capture_reference(
            cand.project_root, cand.pre_commit, cand.post_commit,
        )
        record = build_fixture_record(
            cand, reference, default_verify_commands(repo), plan=None,
            cohort=cohort, sampled_at=sampled_at, seed=seed,
        )
        branch = await pin_eval_branch(
            cand.project_root, record['id'], cand.post_commit,
        )
        (tasks_dir / f'{record["id"]}.json').write_text(
            json.dumps(record, indent=2) + '\n',
        )
        click.echo(f'  wrote {record["id"]}.json + pinned {branch}')
        written += 1
    click.echo(f'wrote {written} fixture(s) to {tasks_dir}')


def _run_single_eval(
    task_path: Path, config_name: str | None, base_config,
    force: bool = False, timeout: int | None = None,
    worktree_path: Path | None = None,
):
    """Run eval for a single task with one or all configs.

    A candidate whose role is ARCHITECT (eval-revival θ) is dispatched to
    ``run_architect_eval`` — the plan-only architect eval that scores the
    produced plan against the real landed diff and FREEZES every downstream
    role — and its per-fixture ``plan_quality`` score is surfaced (plus a
    plan-quality table across all architect runs). Ordinary implementer configs
    still route to ``run_eval``, unchanged.
    """
    from orchestrator.evals.configs import EVAL_CONFIGS, get_config_by_name
    from orchestrator.evals.eval_gate import campaign_usage_gate
    from orchestrator.evals.measurement import format_invocation_errors
    from orchestrator.evals.metrics import RoleUnderTest
    from orchestrator.evals.plan_quality_report import (
        build_plan_quality_report,
        format_plan_quality_table,
    )
    from orchestrator.evals.runner import run_architect_eval, run_eval

    all_configs = EVAL_CONFIGS

    if config_name and config_name != 'all':
        cfg = get_config_by_name(config_name)
        if not cfg:
            click.echo(f'Unknown config: {config_name}', err=True)
            click.echo(f'Available: {", ".join(c.name for c in all_configs)}', err=True)
            sys.exit(1)
        configs = [cfg]
    else:
        configs = all_configs

    async def _run():
        # ONE gate for the whole config loop (task 4427) — see
        # campaign_usage_gate for the argument. Specific to this site: the loop
        # is a campaign the μ stage functions never pass through, so it owns a
        # gate of its own, and BOTH dispatch branches share it so cap state a
        # config proved is inherited by every config after it, architect and
        # implementer alike.
        async with campaign_usage_gate(base_config) as gate:
            architect_results = []
            for cfg in configs:
                if cfg.role is RoleUnderTest.ARCHITECT:
                    # θ: plan-only architect eval — downstream roles frozen.
                    # run_architect_eval manages its own eval worktree at the
                    # fixture's pre_task_commit, so it takes no worktree_path.
                    result = await run_architect_eval(
                        task_path, cfg, base_config, timeout_override=timeout,
                        usage_gate=gate,
                    )
                    architect_results.append(result)
                    plan_quality = result.metrics.get('plan_quality')
                    # A cap-tainted cell names its infra failure inline, so an
                    # operator watching the run sees it LIVE rather than a bare
                    # `plan_quality=None` that reads like a scoring quirk. Healthy
                    # cells echo exactly as before.
                    #
                    # 'unmeasurable', not 'cap-tainted': the flag covers every cause
                    # that left no model content (cap hit, auth failure,
                    # model-not-found, wedge, harness error), and a PERMANENT config
                    # error must not read to the operator as a transient cap window.
                    # The marker that follows always names the actual cause.
                    taint = (
                        ' unmeasurable: '
                        f'{format_invocation_errors(result.metrics.get("invocation_errors"))}'
                        if result.metrics.get('cap_tainted') else ''
                    )
                    # `steps=` is echoed BESIDE the score (task 3302) because it is
                    # the plan-production predicate the whole pipeline now keys on:
                    # `steps=0` beside any plan_quality means the architect produced
                    # nothing, which the final table floors to 0.0. Showing it live
                    # is what stops a no-plan candidate from looking healthy for the
                    # length of a campaign.
                    click.echo(
                        f'{result.task_id} × {result.config_name}: '
                        f'{result.outcome} plan_quality={plan_quality} '
                        f'steps={result.metrics.get("plan_steps")}{taint} '
                        f'({result.wall_clock_ms / 1000:.1f}s)'
                    )
                else:
                    result = await run_eval(
                        task_path, cfg, base_config, timeout_override=timeout,
                        worktree_path=worktree_path, usage_gate=gate,
                    )
                    click.echo(
                        f'{result.task_id} × {result.config_name}: '
                        f'{result.outcome} ({result.wall_clock_ms / 1000:.1f}s)'
                    )

            # Surface the θ plan-quality table across all architect runs.
            if architect_results:
                click.echo('')
                click.echo(
                    format_plan_quality_table(
                        build_plan_quality_report(architect_results)
                    )
                )

    _require_zai_coding_plan(configs)
    asyncio.run(_run())


def _run_matrix_cmd(
    base_config,
    max_parallel: int | None = None,
    trials: int = 1,
    force: bool = False,
    timeout: int | None = None,
):
    """Backward-compatible ``eval --matrix`` alias → the μ matrix driver (task 2478).

    Open-Q4 resolution: the legacy ``--matrix`` flag now routes to the both-live
    architect×implementer matrix driver (:func:`_run_matrix_driver` →
    ``run_matrix_stage``) and emits the C4 composite report, instead of the old
    all-configs ``run_eval_matrix`` summary. The signature is preserved verbatim
    so the ``eval --matrix`` dispatch (and the vLLM-injection CLI test that pins
    this routing) is unchanged; ``force`` is retained for kwarg back-compat — the
    μ screen/matrix stages always run fresh, so it is not consulted.
    """
    _run_matrix_driver(
        base_config, tasks_dir=None,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


def _resolve_eval_task_paths(base_config, tasks_dir: Path | None) -> list[Path]:
    """Resolve the eval-fixture dir → the sorted ``*.json`` task paths (μ driver).

    Defaults to the packaged ``evals/tasks`` dir (falling back to the one under
    *base_config.project_root* when the packaged dir is absent, mirroring
    :func:`_run_single_eval`). :func:`_load_fixture_dir` validates the corpus
    first — a malformed fixture is surfaced loudly with the offending file NAMED
    — and an empty corpus is a clean CLI error, not a silent no-op.
    """
    tasks_dir = tasks_dir or PACKAGED_TASKS_DIR
    if not tasks_dir.exists():
        tasks_dir = Path(base_config.project_root) / 'orchestrator' / 'evals' / 'tasks'
    # Loud validation (malformed fixture → named ClickException) + emptiness guard.
    if not _load_fixture_dir(tasks_dir):
        click.echo(f'No eval fixtures found in {tasks_dir}', err=True)
        sys.exit(1)
    return sorted(tasks_dir.glob('*.json'))


def _require_zai_coding_plan(configs) -> None:
    """Probe the Z.ai Coding Plan once if any candidate targets it; an unusable plan fails the command."""
    from orchestrator.evals.zai_probe import (
        ZaiCodingPlanUnavailable,
        require_zai_coding_endpoint_for,
    )

    try:
        require_zai_coding_endpoint_for(configs)
    except ZaiCodingPlanUnavailable as exc:
        raise click.ClickException(str(exc)) from exc


def _eval_prices(base_config) -> dict[str, Any]:
    """The μ driver's price map: ``base_config.prices`` or the packaged
    :func:`default_price_table` fallback. Single-sourced so every stage seeds its
    price table (individual or combined-name) from the same map.
    """
    from orchestrator.config import default_price_table

    return base_config.prices or default_price_table()


def _emit_composite_report(results, price_table) -> None:
    """Emit the C4 composite report for *results* (the μ driver stages' surface).

    *price_table* is the pre-built ``{config_name: {role: entry}}`` table the
    caller seeds — λ's :func:`build_price_table` (OFAT's individual configs) or
    :func:`build_pairwise_price_table` (the matrix/confirm end-to-end stages,
    whose report rows are keyed by the combined ``arch+impl`` name, so the price
    section must be keyed the same way to stay aligned). Builds λ's per-config
    composite/cost/latency/CI95/judge report and prints
    :func:`format_composite_table`. The quality figure is single-sourced in λ's
    ``compute_composite`` — the driver never re-derives a score.

    When *results* contain any PLAN-ONLY architect run, the θ plan-quality table
    is emitted after it (task 3099), mirroring the precedent already in
    :func:`_run_single_eval` rather than inventing a second rendering path. The
    two tables are complementary, not redundant: the composite row reports the
    cap-exclusion as a COUNT, while only the plan-quality table breaks it out BY
    CAUSE — and that is what tells an operator reading an OFAT run whether a
    missing architect cell is a transient cap window (rerun it) or a permanent
    model-not-found (that candidate can never run at all). A result set with no
    architect rows emits the composite table alone, so the existing
    ``eval-matrix`` / ``eval-confirm`` end-to-end surfaces are unchanged.
    """
    from orchestrator.evals.composite_report import (
        build_composite_report,
        format_composite_table,
    )
    from orchestrator.evals.metrics import RoleUnderTest
    from orchestrator.evals.plan_quality_report import (
        build_plan_quality_report,
        format_plan_quality_table,
    )

    report = build_composite_report(results, price_table=price_table)
    click.echo(format_composite_table(report))

    if any(r.metrics.get('role_under_test') == RoleUnderTest.ARCHITECT for r in results):
        click.echo('')
        click.echo(format_plan_quality_table(build_plan_quality_report(results)))


def _run_ofat_driver(
    base_config, *, tasks_dir: Path | None,
    max_parallel: int | None, trials: int, timeout: int | None,
) -> None:
    """OFAT screen: ``run_ofat_stage`` over ``ofat_candidates()`` → composite (μ)."""
    from orchestrator.evals.configs import ofat_candidates
    from orchestrator.evals.prices import build_price_table
    from orchestrator.evals.runner import run_ofat_stage

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
    candidates = ofat_candidates()
    _require_zai_coding_plan(candidates)
    results = asyncio.run(run_ofat_stage(
        task_paths, candidates, base_config,
        max_parallel=max_parallel, trials=trials, timeout_override=timeout,
    ))
    # OFAT rows are keyed by the individual candidate name → individual price table.
    _emit_composite_report(
        results, build_price_table(candidates, _eval_prices(base_config)),
    )


def _run_matrix_driver(
    base_config, *, tasks_dir: Path | None,
    max_parallel: int | None, trials: int, timeout: int | None,
) -> None:
    """Matrix stage: both-live architect×implementer cross product → composite (μ).

    Splits ``ofat_candidates()`` by role into the architect and implementer
    survivor inputs; ``run_matrix_stage`` expands the FULL cross product
    (INCLUDING same-family diagonals) via ``configs.matrix_pairs``.
    """
    from orchestrator.evals.configs import matrix_pairs, ofat_candidates
    from orchestrator.evals.metrics import RoleUnderTest
    from orchestrator.evals.prices import build_pairwise_price_table
    from orchestrator.evals.runner import run_matrix_stage

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
    candidates = ofat_candidates()
    _require_zai_coding_plan(candidates)
    arch_cfgs = [c for c in candidates if c.role is RoleUnderTest.ARCHITECT]
    impl_cfgs = [c for c in candidates if c.role is RoleUnderTest.IMPLEMENTER]
    results = asyncio.run(run_matrix_stage(
        task_paths, arch_cfgs, impl_cfgs, base_config,
        max_parallel=max_parallel, trials=trials, timeout_override=timeout,
    ))
    # End-to-end rows are keyed by the combined arch+impl name → a combined-name
    # price table over the FULL cross product, so the price section aligns.
    _emit_composite_report(
        results,
        build_pairwise_price_table(
            matrix_pairs(arch_cfgs, impl_cfgs), _eval_prices(base_config),
        ),
    )


def _run_confirm_driver(
    base_config, *, arch: str, impl: str, tasks_dir: Path | None,
    max_parallel: int | None, trials: int, timeout: int | None,
) -> None:
    """Confirmation batch: the single winning ``(arch, impl)`` combo → composite (μ)."""
    from orchestrator.evals.configs import get_config_by_name
    from orchestrator.evals.metrics import RoleUnderTest
    from orchestrator.evals.prices import build_pairwise_price_table
    from orchestrator.evals.runner import run_confirm_stage

    arch_cfg = get_config_by_name(arch)
    impl_cfg = get_config_by_name(impl)
    if arch_cfg is None or impl_cfg is None:
        missing = arch if arch_cfg is None else impl
        click.echo(f'Unknown config: {missing}', err=True)
        sys.exit(1)
    # Loud-over-silent: --arch must resolve to an ARCHITECT config and --impl to
    # an IMPLEMENTER one. Swapping the flags otherwise resolves fine and silently
    # builds a nonsensical both-live combo (implementer pinned as the architect
    # and vice-versa) — reject the mismatch NAMING the offending flag + role.
    if arch_cfg.role is not RoleUnderTest.ARCHITECT:
        click.echo(
            f'--arch: {arch!r} is a {arch_cfg.role} config, expected an architect',
            err=True,
        )
        sys.exit(1)
    if impl_cfg.role is not RoleUnderTest.IMPLEMENTER:
        click.echo(
            f'--impl: {impl!r} is a {impl_cfg.role} config, expected an implementer',
            err=True,
        )
        sys.exit(1)

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
    _require_zai_coding_plan([arch_cfg, impl_cfg])
    results = asyncio.run(run_confirm_stage(
        task_paths, arch_cfg, impl_cfg, base_config,
        max_parallel=max_parallel, trials=trials, timeout_override=timeout,
    ))
    # The single winning combo's rows are keyed by the combined arch+impl name.
    _emit_composite_report(
        results,
        build_pairwise_price_table(
            [(arch_cfg, impl_cfg)], _eval_prices(base_config),
        ),
    )


@eval_cli.command('eval-ofat')
@pass_config(help=CONFIG_OPTION_HELP_PREFIX)
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir (default: orchestrator/evals/tasks)')
@click.option('--trials', type=int, default=1,
              help='Trials per (fixture, candidate) cell (default: 1)')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
def eval_ofat_cmd(config, tasks_dir, trials, max_parallel, timeout):
    """OFAT screen (μ): vary ONE role per candidate, pin the rest to incumbents.

    Each candidate varies exactly one role — implementer incumbents (Opus AND
    Sonnet, the G2 >=2-config floor) drive ``run_eval`` with the plan frozen;
    architect candidates drive ``run_architect_eval`` live with downstream roles
    frozen. Emits the C4 composite report.
    """
    _run_ofat_driver(
        config, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


@eval_cli.command('eval-matrix')
@pass_config(help=CONFIG_OPTION_HELP_PREFIX)
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir (default: orchestrator/evals/tasks)')
@click.option('--trials', type=int, default=1,
              help='Trials per (fixture, pair) cell (default: 1)')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
def eval_matrix_cmd(config, tasks_dir, trials, max_parallel, timeout):
    """Matrix stage (μ): both-live architect×implementer cross product.

    Runs the FULL architect×implementer cross product over the OFAT survivors,
    INCLUDING same-family diagonals (the plan-style/implementer coupling
    hypothesis, PRD decision 9). Emits the C4 composite report.
    """
    _run_matrix_driver(
        config, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


@eval_cli.command('eval-confirm')
@pass_config(help=CONFIG_OPTION_HELP_PREFIX)
@click.option('--arch', required=True,
              help='Winning architect config name (e.g. architect-opus-high)')
@click.option('--impl', required=True,
              help='Winning implementer config name (e.g. claude-opus-high)')
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir (default: orchestrator/evals/tasks)')
@click.option('--trials', type=int, default=3,
              help='Confirmation trials per fixture (default: 3, decision 10 floor)')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
def eval_confirm_cmd(config, arch, impl, tasks_dir, trials, max_parallel, timeout):
    """Confirmation batch (μ): one end-to-end batch of the winning combo.

    Runs the single winning (architect, implementer) combo across all fixtures ×
    N trials (default 3 — decision 10's statistics floor, enough repeats for a
    CI95 on the winner). Emits the C4 composite report.
    """
    _run_confirm_driver(
        config, arch=arch, impl=impl, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


@eval_cli.group('eval-shadow')
def eval_shadow():
    """Live paired shadow evaluation (plans/live-shadow-eval-prd.md)."""


@eval_shadow.command('report')
@pass_config(help=CONFIG_OPTION_HELP_PREFIX)
@click.option('--shape', type=click.Choice([s.value for s in ShadowShape]),
              default=ShadowShape.IMPLEMENTER.value, show_default=True,
              help='Cell shape to report.')
@click.option('--candidate', default=None, help='Report only this candidate.')
@click.option('--json', 'as_json', is_flag=True, help='Emit the report as JSON.')
def eval_shadow_report_cmd(config, shape, candidate, as_json):
    """C5 report of one project's shadow cells: one paired line per candidate.

    Exits 1 when a settled cell's result is missing, in either form, so a
    short table never reads as a complete measurement.
    """
    from orchestrator.evals.shadow_report import (
        RunsDbMissingError,
        UnsupportedShapeError,
        read_shadow_report,
        render_text,
    )
    try:
        report = asyncio.run(read_shadow_report(
            runs_db(config.project_root),
            project_id=config.fused_memory.project_id,
            shape=ShadowShape(shape),
            candidate=candidate,
            n_min=config.shadow_eval.n_min,
        ))
    except RunsDbMissingError as e:
        click.echo(f'Error: {e}. Does --config name the intended project?', err=True)
        sys.exit(1)
    except UnsupportedShapeError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(render_text(report), nl=False)
    if not report.complete:
        sys.exit(1)


def _run_judge_cmd(max_rounds: int = 50, reset: bool = False):
    """Run Elo-based judge on existing results."""
    from orchestrator.evals.elo import JudgeState, TaskPool, load_state, save_state
    from orchestrator.evals.elo_report import build_report, format_markdown, save_report
    from orchestrator.evals.judge import run_elo_tournament
    from orchestrator.evals.results import ResultStore
    from orchestrator.evals.runner import load_task
    from orchestrator.evals.snapshots import get_diff_between_commits

    # Load or reset state
    if reset:
        state = JudgeState()
        click.echo('Judge state reset.')
    else:
        state = load_state()
        if state.per_task:
            click.echo(f'Resuming from existing state ({len(state.per_task)} tasks)')

    results = ResultStore.packaged().load_all()
    if not results:
        click.echo('No existing results found in evals/results/', err=True)
        sys.exit(1)

    # Group by task, filter to passing with existing worktrees
    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    passing: dict[str, list[dict]] = {}
    for task_id, task_results in by_task.items():
        p = [r.to_dict() for r in task_results
             if r.metrics.get('tests_pass', False)
             and Path(r.worktree_path).exists()]
        if p:
            passing[task_id] = p
            click.echo(f'  {task_id}: {len(p)} contenders with worktrees')

    if not passing:
        click.echo('No passing results with existing worktrees found', err=True)
        sys.exit(1)

    tasks_dir = PACKAGED_TASKS_DIR

    async def _run():
        for task_id, result_dicts in passing.items():
            task_file = tasks_dir / f'{task_id}.json'
            if not task_file.exists():
                click.echo(f'Skipping {task_id}: task file not found')
                continue

            task = load_task(task_file)

            # Add reference implementation if post_task_commit exists
            pre = task.get('pre_task_commit')
            post = task.get('post_task_commit')
            if pre and post:
                try:
                    project_root = Path(task['project_root'])
                    ref_diff = await get_diff_between_commits(project_root, pre, post)
                    if ref_diff.strip():
                        result_dicts.append({
                            'config_name': 'reference',
                            'diff': ref_diff,
                            'worktree_path': '',
                        })
                        click.echo(f'  {task_id}: added reference implementation')
                except Exception as e:
                    click.echo(f'  {task_id}: could not compute reference diff: {e}')

            if len(result_dicts) < 2:
                click.echo(f'Skipping {task_id}: need at least 2 contenders')
                continue

            # Get or create task pool
            if task_id not in state.per_task:
                state.per_task[task_id] = TaskPool()
            pool = state.per_task[task_id]

            click.echo(
                f'\nJudging {task_id} '
                f'({len(result_dicts)} contenders, max {max_rounds} rounds)...'
            )
            tally = await run_elo_tournament(
                result_dicts, task, pool, max_rounds,
            )
            click.echo(
                f'  {task_id}: {tally.judged} judged, '
                f'{tally.unmeasured} unmeasured (excluded from Elo)'
            )

            # Save state after each task (crash resilience)
            save_state(state)

        # Generate and print report
        report = build_report(state)
        report_path = save_report(report)
        click.echo(f'\nReport saved to {report_path}')
        click.echo('\n' + format_markdown(report))

    asyncio.run(_run())


def _run_report_cmd():
    """Generate report from existing judge state (no new judge calls)."""
    from orchestrator.evals.elo import load_state
    from orchestrator.evals.elo_report import build_report, format_markdown, save_report

    state = load_state()
    if not state.per_task:
        click.echo('No judge state found. Run --judge first.', err=True)
        sys.exit(1)

    report = build_report(state)
    report_path = save_report(report)
    click.echo(f'Report saved to {report_path}')
    click.echo('\n' + format_markdown(report))


def _run_compare_cmd(
    compare: tuple[str, str],
    combine_runs: tuple[str, ...],
):
    """Compare two model groups via LLM-powered qualitative assessment."""
    from orchestrator.evals.compare import (
        apply_combine_runs,
        compare_models,
        format_comparison_markdown,
    )
    from orchestrator.evals.results import ResultStore
    from orchestrator.evals.runner import load_task

    results = ResultStore.packaged().load_all()
    if not results:
        click.echo('No existing results found in evals/results/', err=True)
        sys.exit(1)

    # Apply combine-runs aliasing
    combine_groups = [g.split(',') for g in combine_runs]
    if combine_groups:
        results = apply_combine_runs(results, combine_groups)

    group_a_configs = compare[0].split(',')
    group_b_configs = compare[1].split(',')

    # Load task definitions for descriptions
    tasks_dir = PACKAGED_TASKS_DIR
    tasks: dict[str, dict] = {}
    if tasks_dir.exists():
        for tp in sorted(tasks_dir.glob('*.json')):
            task = load_task(tp)
            tasks[task['id']] = task

    # Determine canonical group names
    group_a_name = group_a_configs[0]
    group_b_name = group_b_configs[0]

    click.echo(
        f'Comparing {group_a_name} vs {group_b_name} '
        f'({len(results)} total results loaded)'
    )

    async def _run():
        report = await compare_models(
            results, group_a_configs, group_b_configs, tasks,
            group_a_name=group_a_name,
            group_b_name=group_b_name,
        )
        click.echo('\n' + format_comparison_markdown(report))

    asyncio.run(_run())


def _run_cleanup(base_config):
    """Remove all eval worktrees."""
    from orchestrator.evals.snapshots import (
        cleanup_eval_worktree,
        eval_worktree_root,
    )

    # Eval worktrees live OUTSIDE project_root (a sibling of the repo) so a
    # nested pytest/pyright/ruff/cargo run cannot collect the main repo's
    # ancestor config (Defect B, task 2881). Discover them at that same
    # relocated root — sharing eval_worktree_root with create_eval_worktree so
    # placement and cleanup cannot drift.
    worktree_root = eval_worktree_root(base_config.project_root)
    if not worktree_root.exists():
        click.echo('No eval worktrees found.')
        return

    async def _cleanup():
        count = 0
        for task_dir in sorted(worktree_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for run_dir in sorted(task_dir.iterdir()):
                if run_dir.is_dir():
                    await cleanup_eval_worktree(base_config.project_root, run_dir)
                    count += 1
        click.echo(f'Cleaned up {count} eval worktrees.')

    asyncio.run(_cleanup())


def _format_routing_decision(role_name: str, decision: RoutingDecision) -> str:
    """The stdout line naming the route one eval invocation runs on."""
    line = (
        f'    routing_decision role={role_name} model={decision.model} '
        f'effort={decision.effort} budget_usd={decision.budget_usd} '
        f'max_turns={decision.max_turns} source_layer={decision.source_layer}'
    )
    if decision.rule_id is not None:
        line += f' rule_id={decision.rule_id}'
    if decision.rejected:
        line += f' rejected={",".join(decision.rejected)}'
    return line


def _run_plan_only(task_path: Path | None, base_config):
    """Generate plans for eval tasks with the architect on its resolved route.

    Runs the architect against the pre-task commit for each task, on the
    route ``resolve_route`` gives role 'architect' under *base_config* and on
    that config's architect backend, and saves the resulting plan into the
    task JSON file.
    """
    from orchestrator.evals.runner import load_task
    from orchestrator.evals.snapshots import cleanup_eval_worktree, create_eval_worktree

    tasks_dir = PACKAGED_TASKS_DIR

    task_paths = [task_path] if task_path else sorted(tasks_dir.glob('*.json'))

    if not task_paths:
        click.echo('No task files found', err=True)
        sys.exit(1)

    async def _run():
        from orchestrator.agents.invoke import invoke_agent
        from orchestrator.agents.roles import ARCHITECT
        from orchestrator.artifacts import TaskArtifacts
        from orchestrator.evals.replay_frame import (
            REPLAY_FRAME_ID,
            ReplayFramedBriefingAssembler,
        )

        for tp in task_paths:
            task = load_task(tp)
            task_id = task['id']

            if task.get('plan'):
                click.echo(f'  {task_id}: already has plan ({len(task["plan"].get("steps", []))} steps), skipping')
                continue

            click.echo(f'  {task_id}: generating plan...')
            project_root = Path(task['project_root'])

            # Create worktree at pre-task commit
            worktree, _run_id = await create_eval_worktree(
                project_root, task_id, task['pre_task_commit'],
                setup_commands=task.get('setup_commands'),
            )

            try:
                # Init artifacts so architect has a place to write
                artifacts = TaskArtifacts(worktree)
                artifacts.init(
                    task_id,
                    task.get('task_definition', {}).get('title', ''),
                    task.get('task_definition', {}).get('description', ''),
                    base_commit=task['pre_task_commit'],
                )

                # Build architect prompt
                task_def = task.get('task_definition', {})
                briefing = ReplayFramedBriefingAssembler(
                    base_config, base_commit=task['pre_task_commit'],
                )
                prompt = await briefing.build_architect_prompt(task_def, worktree=worktree)

                decision = resolve_route(RouteInputs(
                    role_name=ARCHITECT.name, task_id=task_id, task_metadata={},
                    plan_shape=None, routing_tier=0, dispatch_count=0,
                    role_defaults=RoleDefaults(
                        model=ARCHITECT.default_model, effort='high',
                        budget_usd=ARCHITECT.default_budget,
                        max_turns=ARCHITECT.default_max_turns,
                    ),
                ), base_config)
                click.echo(_format_routing_decision(ARCHITECT.name, decision))
                result = await invoke_agent(
                    prompt=prompt,
                    system_prompt=ARCHITECT.system_prompt,
                    cwd=worktree,
                    model=decision.model,
                    max_turns=decision.max_turns,
                    max_budget_usd=decision.budget_usd,
                    allowed_tools=ARCHITECT.allowed_tools or None,
                    disallowed_tools=ARCHITECT.disallowed_tools or None,
                    effort=decision.effort,
                    backend=base_config.backends.architect,
                )

                if not result.success:
                    click.echo(f'    FAILED: {result.output[:200]}', err=True)
                    continue

                # Read the plan the architect wrote
                plan = artifacts.read_plan()
                if not plan:
                    click.echo('    FAILED: architect produced no plan.json', err=True)
                    continue

                # Save plan into task JSON
                task['plan'] = plan
                task['plan_replay_frame'] = REPLAY_FRAME_ID
                with open(tp, 'w') as f:
                    json.dump(task, f, indent=2)
                    f.write('\n')

                step_count = len(plan.get('steps', []))
                click.echo(
                    f'    OK: {step_count} steps, '
                    f'cost=${result.cost_usd:.2f}'
                )

            finally:
                await cleanup_eval_worktree(project_root, worktree)

    asyncio.run(_run())
