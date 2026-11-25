"""CLI entry point: `orchestrator run [--prd X]`."""

import asyncio
import contextlib
import functools
import importlib
import json
import logging
import os
import signal
import socket
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import click
from shared.spawn_temp_files import sweep_orphaned_spawn_temp_files

from orchestrator.cli_config import CONFIG_OPTION_HELP_PREFIX, load_program_dotenv, pass_config
from orchestrator.config import OrchestratorConfig
from orchestrator.config_census_ignore import audit_census_ignore_entries
from orchestrator.evals.cli import eval_cli
from orchestrator.verify_cancel import (
    WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
    WATCHDOG_KILL_GRACE_SECS,
    WatchdogTrigger,
    acquire_merge_verify_flock,
    cancel_request,
    fire_watchdog_kill,
    lane_lock_path,
    merge_verify_lock_path,
    pgid_file,
    read_lock_holder_pgid,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    remove_pgid_file,
    remove_worktree_path,
    start_own_process_group,
    start_stdin_watchdog,
    write_lock_holder_pgid,
    write_pgid_file,
    write_worktree_path,
)

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps
    from orchestrator.task_verify_spec import TaskVerifySpec
    from orchestrator.verify import VerifyResult
    from orchestrator.verify_runner import MergeVerifySpec

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# How long after asyncio.run() returns to wait before force-exiting.
# The watchdog is armed AFTER asyncio.run() returns (not before) so that
# long orchestration runs are never affected. After this deadline a diagnostic
# dump is written to stderr and os._exit(137) fires. asyncio.run()'s own
# teardown is bounded separately, by the teardown watchdog below.
SHUTDOWN_WATCHDOG_TIMEOUT_SECS = 30

# Bounds asyncio.run()'s own teardown, which starts once the main coroutine has ended.
TEARDOWN_WATCHDOG_TIMEOUT_SECS = 10

# Bounded wait (task 2306 α) for acquiring the laptop persistent-worktree
# merge-verify fcntl.flock (verify_cancel.acquire_merge_verify_flock) when
# git.persistent_merge_worktree is on. On timeout, verify-merge emits a
# distinguished contention VerifyResult instead of ever falling back to an
# ephemeral worktree. Monkeypatchable so tests run the bounded wait fast.
MERGE_VERIFY_FLOCK_WAIT_SECS = 10.0


def _env_float(name: str, default: float) -> float:
    """Read an optional env-var float override, falling back to *default*.

    Task 2309 boundary gate (PRD §11 Q1/Q2 tunability): the remote-side
    timing constants (flock wait, watchdog heartbeat timeout/grace) are
    module-level and otherwise untunable from outside the spawned
    ``verify-merge`` child, making an integration gate wall-clock-bound on
    the production windows (since task 4195, a 90.0s heartbeat timeout plus a
    5.0s kill grace). Unset, unparseable, or non-positive
    values fall back to *default* so production behavior is byte-identical
    when the env var is absent. The non-positive guard matters beyond
    "byte-identical": every current caller feeds a timing window
    (``select.select`` timeout / ``time.sleep`` duration) where <= 0 would
    otherwise reach the watchdog's daemon thread as an invalid argument and
    silently kill it instead of arming it.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


class WatchdogHandle(NamedTuple):
    """Return value of :func:`_force_exit_after_delay`."""
    disarm: Callable[[], None]
    thread: threading.Thread


def _force_exit_after_delay(
    timeout_secs: float,
    exit_code: int = 137,
    *,
    stream=None,
    _exit: 'Callable[[int], None] | None' = None,
    phase: str = 'after asyncio.run() returned',
) -> 'WatchdogHandle':
    """Arm a daemon-thread watchdog that force-exits if not disarmed in time.

    Spawns a daemon thread (cannot block interpreter shutdown) that waits
    ``timeout_secs`` seconds.  If the ``disarm`` callable returned by this
    function is NOT called before the timeout, the thread writes a diagnostic
    dump of live threads/frames to *stream* (defaulting to ``sys.stderr`` at
    fire time) then calls ``_exit(exit_code)`` (defaulting to ``os._exit``).

    ``os._exit`` is used (not ``sys.exit``) so that ``threading._shutdown``
    and atexit callbacks are bypassed entirely — exactly the stuck path we
    need to escape.

    Exit code 137 = 128 + 9 (SIGKILL convention); distinguishable from
    130 (SIGINT) and 143 (SIGTERM) in operator logs.

    Returns a ``WatchdogHandle`` (NamedTuple of ``disarm: Callable[[], None]``
    and ``thread: threading.Thread``).  Calling ``handle.disarm()`` (even
    multiple times) is safe and prevents the watchdog from firing.
    ``handle.thread`` is exposed for tests that want to ``join()`` after
    assertions to guarantee the watchdog thread has exited. The orchestrator
    intentionally never disarms its post-run watchdog, so it guards interpreter
    shutdown (atexit callbacks + ``threading._shutdown()`` joining non-daemon
    threads). *phase* names the window being guarded in the dump header.

    The ``_exit`` parameter is an injectable exit callable (default
    ``os._exit``). Injecting a stub in tests means a leaked/late-firing daemon
    thread can NEVER reach the REAL global ``os._exit`` — so it cannot abruptly
    kill an xdist worker process. The closure captures ``_exit`` at arm-time;
    the production default keeps behaviour byte-identical.
    """
    # Resolve the exit callable once at arm-time so the closure captures it.
    # A late-firing thread calls the stub (or the real os._exit default) that
    # was live when the watchdog was armed — no global lookup at fire-time.
    _exit_fn: Callable[[int], None] = _exit if _exit is not None else os._exit

    _event = threading.Event()

    def _watchdog() -> None:
        fired = not _event.wait(timeout_secs)
        if not fired:
            # Disarmed normally — exit without doing anything.
            return
        # Timeout reached without disarm: write diagnostic and force-exit.
        out = stream if stream is not None else sys.stderr
        try:
            import traceback

            lines = [f'SHUTDOWN WATCHDOG FIRED — process hung {phase}\n']
            frames = sys._current_frames()
            for t in threading.enumerate():
                frame = frames.get(t.ident) if t.ident is not None else None
                lines.append(f'\n--- Thread {t.name!r} (daemon={t.daemon}, ident={t.ident}) ---\n')
                if frame is not None:
                    lines.extend(traceback.format_stack(frame))
            out.write(''.join(lines))
            out.flush()
        except Exception:
            # Diagnostic dump failed (e.g. traceback or sys._current_frames
            # torn down during interpreter shutdown). Emit a fallback sentinel
            # so operators still see a log line before exit 137. Wrapped in
            # its own try/except so a stream-write failure still falls through
            # to _exit_fn — the force-exit guarantee must never be weakened.
            try:
                out.write('SHUTDOWN WATCHDOG FIRED (diagnostic dump failed)\n')
                out.flush()
            except Exception:
                pass
        _exit_fn(exit_code)

    thread = threading.Thread(target=_watchdog, name='shutdown-watchdog', daemon=True)
    thread.start()

    def disarm() -> None:
        """Signal the watchdog not to fire."""
        _event.set()

    return WatchdogHandle(disarm=disarm, thread=thread)


_T = TypeVar('_T')


def _run_with_bounded_teardown(
    main: Callable[[], Awaitable[_T]],
    *,
    teardown_secs: float,
    _exit: Callable[[int], None] | None = None,
) -> _T:
    """``asyncio.run(main())``, force-exiting if asyncio.run's teardown hangs.

    asyncio.run cancels every task left when *main* ends and waits on them with
    no timeout, so one task that ignores its cancel would hold the process
    forever (task 7247). The watchdog is armed only once *main* has ended and
    is disarmed when asyncio.run returns.
    """
    armed: list[WatchdogHandle] = []

    async def _main_then_arm() -> _T:
        try:
            return await main()
        finally:
            armed.append(_force_exit_after_delay(
                teardown_secs, _exit=_exit, phase="in asyncio.run()'s teardown",
            ))
            _log_tasks_left_for_teardown()

    try:
        return asyncio.run(_main_then_arm())
    finally:
        for handle in armed:
            handle.disarm()


def _log_tasks_left_for_teardown() -> None:
    """Name, at WARNING, every task asyncio.run()'s teardown will cancel and wait on.

    The teardown watchdog's dump shows thread stacks only, so this line is what
    names the task that ignored its cancel.
    """
    current = asyncio.current_task()
    leftovers = sorted(t.get_name() for t in asyncio.all_tasks() if t is not current)
    if leftovers:
        logging.getLogger(__name__).warning(
            "asyncio.run()'s teardown will cancel and wait on %d task(s): %s", len(leftovers), leftovers,
        )


def _make_cancel_handler(main_task, logger):
    """Build an idempotent SIGTERM/SIGINT handler.

    The first signal cancels the main task; subsequent signals are logged
    and ignored so they cannot re-cancel the task mid-cleanup. SIGKILL
    remains the operator escape hatch if cleanup itself ever wedges.
    """
    # Mutable single-element list so the nested closure can mutate it
    # without needing nonlocal (simplifies making this a module-level
    # factory that's testable in isolation).
    fired = [False]

    def _cancel(sig_name: str) -> None:
        if fired[0]:
            logger.info(
                f'{sig_name} received — shutdown already in progress, ignoring'
            )
            return
        fired[0] = True
        logger.warning(f'{sig_name} received — cancelling main task')
        main_task.cancel()

    return _cancel


class _DotenvProgramGroup(click.Group):
    """Root group that loads .env when run as a program (console script, python -m), never on import or CliRunner.invoke."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        load_program_dotenv()
        return super().__call__(*args, **kwargs)


@click.group(cls=_DotenvProgramGroup)
@click.option('--verbose', is_flag=True, help='Enable debug logging')
def main(verbose: bool):
    """Dark Factory agent orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stderr,
    )


@main.command()
@click.option('--prd', type=click.Path(exists=True, path_type=Path), default=None,
              help='Path to PRD markdown file (omit to run existing tasks)')
@pass_config()
@click.option('--dry-run', is_flag=True, help='Populate tasks only, do not execute')
@click.option('--delay', default=None,
              help='Delay before executing tasks (e.g. 4h, 30m, 90s). '
                   'Escalation server starts immediately.')
@click.option('--force-dirty-start', is_flag=True,
              help='project_root with uncommitted changes always starts; this '
                   'silently skips filing the born-at-L2 cleanup escalation')
@click.option('--retag-modules', is_flag=True,
              help='Force re-tag all non-done/cancelled tasks with code modules')
@click.option('--until-idle', is_flag=True,
              help='Exit when the task queue drains (default: run forever, '
                   'idling for newly-scheduled tasks)')
def run(prd: Path | None, config: OrchestratorConfig, dry_run: bool, delay: str | None,
        force_dirty_start: bool, retag_modules: bool, until_idle: bool):
    """Run the orchestrator against a PRD, or execute existing tasks if no PRD given."""
    from orchestrator.deployed_sha import capture_deployed_sha, resolve_code_checkout

    # Task 6565 (D1): importing the harness loads the whole verify wire path, so
    # bracketing this one import records which commit the process runs.
    deployed_sha = capture_deployed_sha(
        resolve_code_checkout(), lambda: importlib.import_module('orchestrator.harness'),
    )
    from orchestrator.harness import Harness

    delay_secs = _parse_duration(delay) if delay else 0
    if deployed_sha is None and config.enabled_verify_runners:
        click.echo(
            'Error: a remote verify runner is enabled but this process cannot read the '
            'commit it runs (no readable git checkout above its code); the remote gate '
            'needs that sha (task 6565). Run from a git checkout or disable the runner.',
            err=True,
        )
        sys.exit(1)
    logger = logging.getLogger(__name__)
    # Process start, before this process spawns any agent; the sweep is host-wide, so it
    # also reclaims fused-memory's and scripts' orphans. See
    # shared/src/shared/spawn_temp_files.py::sweep_orphaned_spawn_temp_files.
    try:
        sweep_orphaned_spawn_temp_files()
    except Exception:
        logger.warning(
            'Startup sweep of orphaned agent-spawn temp files failed (non-fatal, startup continues)',
            exc_info=True,
        )
    harness = Harness(config, deployed_sha=deployed_sha)

    async def _main():
        # Route SIGTERM/SIGINT through asyncio so CancelledError flows into
        # harness.run() between scheduling steps rather than raising at an
        # arbitrary bytecode inside the loop machinery. This guarantees the
        # finally block in harness.run() runs to completion.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        assert main_task is not None

        _cancel = _make_cancel_handler(main_task, logger)

        for sig_name in ('SIGTERM', 'SIGINT'):
            sig = getattr(signal, sig_name)
            try:
                loop.add_signal_handler(sig, _cancel, sig_name)
            except (NotImplementedError, RuntimeError):
                # Fallback for platforms where add_signal_handler is unsupported
                signal.signal(sig, lambda *_: _cancel('signal'))

        return await harness.run(
            prd, dry_run=dry_run, delay_secs=delay_secs,
            force_dirty_start=force_dirty_start,
            retag_modules=retag_modules,
            until_idle=until_idle,
            apply_project_slice_weight=True,
        )

    try:
        report = _run_with_bounded_teardown(_main, teardown_secs=TEARDOWN_WATCHDOG_TIMEOUT_SECS)
    except asyncio.CancelledError:
        click.echo('Orchestrator cancelled', err=True)
        # asyncio.run() has returned, its teardown bounded by the teardown
        # watchdog and that watchdog disarmed — arm this one NOW to guard interpreter
        # shutdown (atexit callbacks + threading._shutdown() joining non-daemon
        # threads). The watchdog is intentionally NOT disarmed: if shutdown
        # completes cleanly, the daemon thread is killed with the process;
        # if shutdown hangs, the daemon thread fires os._exit(137).
        # Same unbounded-echo tradeoff applies to the `Orchestrator cancelled`
        # message above — a stuck stderr can hang it indefinitely outside the
        # watchdog window. Accepted for the same reason as the normal path.
        _force_exit_after_delay(SHUTDOWN_WATCHDOG_TIMEOUT_SECS)
        sys.exit(130)

    click.echo(report.summary())

    # asyncio.run() has returned and the report has been emitted — arm the watchdog
    # NOW to guard interpreter shutdown (atexit callbacks + threading._shutdown()
    # joining non-daemon threads). asyncio.run()'s own teardown was bounded
    # separately, by the teardown watchdog, which is disarmed before the report
    # echo. Placed AFTER click.echo so user-visible work is
    # not covered by the timer; placed BEFORE the sys.exit(1) branch so both the
    # clean and blocked-exit paths are guarded — non-daemon threads from
    # harness.run() can linger regardless of whether tasks were blocked.
    # Intentionally left armed: if shutdown completes cleanly the daemon thread
    # is killed with the process; if shutdown hangs it fires os._exit(137).
    # Note the deliberate tradeoff — placing the arm AFTER click.echo(report.summary())
    # means a stuck stdout (full pipe, blocked terminal, slow tty) can hang the report
    # write INDEFINITELY, outside the 30-second watchdog window. We accept that:
    # hanging visibly in click.echo while the operator sees partial output is preferable
    # to force-killing the process before the report is emitted at all. The watchdog's
    # scope is interpreter shutdown (atexit + threading._shutdown joining non-daemon
    # threads), not user-visible I/O.
    _force_exit_after_delay(SHUTDOWN_WATCHDOG_TIMEOUT_SECS)

    if report.blocked > 0:
        sys.exit(1)


@main.command()
@pass_config()
def status(config: OrchestratorConfig):
    """Show current task tree and status."""
    from orchestrator.overrides import OverrideStore
    from orchestrator.scheduler import Scheduler

    scheduler = Scheduler(config, override_store=OverrideStore.from_config(config))

    async def _show():
        tasks = await scheduler.get_tasks()
        if not tasks:
            click.echo('No tasks found.')
            return
        for t in tasks:
            tid = t.get('id', '?')
            title = t.get('title', 'Untitled')
            status = t.get('status', 'unknown')
            modules = t.get('metadata', {}).get('modules', [])
            mod_str = f' [{", ".join(modules)}]' if modules else ''
            click.echo(f'  [{status:12s}] {tid}: {title}{mod_str}')

    asyncio.run(_show())


@main.command('flake-ledger')
@pass_config()
def flake_ledger_cmd(config: OrchestratorConfig):
    """Print the flake ledger report: open debt, recurrence chains, health counters.

    READ ONLY. This command opens no debt, files no task, resolves nothing and
    escalates nothing, and it will NOT create a ledger DB for a project that has
    none — it reports the absence instead. It also reads each open row's owner
    status live from fused-memory, and says READ FAILED when it cannot.
    """
    # Named flake_ledger_cmd so the function does not shadow the flake_ledger module.
    from orchestrator.flake_ledger import ledger_db_path
    from orchestrator.flake_report import build_report, corroborate_owners, render_report

    # Built only when there are owners to read, so an absent ledger never builds one.
    async def read_statuses(ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        from orchestrator.chronic_flake import SchedulerChronicFlakeTaskClient
        from orchestrator.scheduler import Scheduler

        client = SchedulerChronicFlakeTaskClient(Scheduler(config), config.project_root)
        return await client.get_statuses(ids)

    report = build_report(ledger_db_path(Path(config.project_root)))
    report = asyncio.run(corroborate_owners(report, read_statuses))
    click.echo(render_report(report))


@main.group('verify-cache')
def verify_cache_group():
    """The verify-result cache (plans/verify-result-cache-prd.md)."""


@verify_cache_group.command('status')
@pass_config()
def verify_cache_status_cmd(config: OrchestratorConfig):
    """Print each role's cache mode, the registry's greens by tier, and the kill switch.

    READ ONLY: it will NOT create a registry for a project that has none.
    """
    from datetime import UTC, datetime

    from orchestrator.verify_cache.registry import (
        Registry,
        RegistryError,
        RegistrySummary,
        registry_db_path,
    )
    from orchestrator.verify_cache.status import render_status

    summary: RegistrySummary | RegistryError
    try:
        summary = asyncio.run(Registry(registry_db_path(Path(config.project_root))).summary())
    except RegistryError as e:
        summary = e
    for line in render_status(config.verify_cache, summary, datetime.now(UTC)):
        click.echo(line)


@verify_cache_group.command('audit-run')
@pass_config(help=f'{CONFIG_OPTION_HELP_PREFIX} Its project_root is the tree audited.')
@click.option('--log-dir', type=click.Path(path_type=Path), default=None,
              help='Where each module\'s raw output is written '
                   '(default: <project_root>/.df-verify-cache-audit).')
def verify_cache_audit_run_cmd(config: OrchestratorConfig, log_dir: Path | None):
    """Run every module's test leg uncached, printing failures as <module>::<node id>.

    The offline lane's verify-cache-audit command (plans/verify-result-cache-prd.md § Audit).
    """
    from orchestrator.verify_cache.audit_run import DEFAULT_LOG_DIR, drop_runner_only_env, run_audit

    drop_runner_only_env(os.environ)
    tree = Path(config.project_root)
    sys.exit(asyncio.run(run_audit(tree, config, log_dir or tree / DEFAULT_LOG_DIR)))


@verify_cache_group.command('enable')
@click.option('--reason', required=True, help='Why the cache is safe to re-enable; recorded on the policy row.')
@pass_config(help=f'{CONFIG_OPTION_HELP_PREFIX} Selects the target project.')
def verify_cache_enable_cmd(reason: str, config: OrchestratorConfig):
    """Clear the verify cache's kill switch (an operator write, plans/verify-result-cache-prd.md § Audit).

    Also resolves the pending kill-switch L2 blockers, whose disable this ends.
    """
    from datetime import UTC, datetime

    from escalation.queue import EscalationQueue
    from shared.project_layout import escalation_queue_dir

    from orchestrator.verify_cache.audit import enable
    from orchestrator.verify_cache.audit_hook import resolve_ended_blockers
    from orchestrator.verify_cache.registry import Registry, RegistryError, registry_db_path
    from orchestrator.verify_cache.status import render_status

    project_root = Path(config.project_root)
    registry = Registry(registry_db_path(project_root))
    now = datetime.now(UTC)
    try:
        policy = asyncio.run(enable(registry, reason, now))
        summary = asyncio.run(registry.summary())
    except RegistryError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint='--reason') from e
    for line in render_status(config.verify_cache, summary, now):
        click.echo(line)
    queue_dir = escalation_queue_dir(project_root)
    if not queue_dir.is_dir():
        return
    try:
        resolved = resolve_ended_blockers(
            EscalationQueue(queue_dir), policy, now,
            resolved_by='interactive', resolution_class='actionable',
            resolution=f'Verify cache re-enabled by the operator: {policy.reason}',
        )
    except OSError as e:
        click.echo(
            f'Error: the cache is re-enabled, but resolving its kill-switch blocker in {queue_dir} failed: {e}; '
            'resolve it with resolve_issue',
            err=True,
        )
        sys.exit(1)
    for escalation_id in resolved:
        click.echo(f'resolved kill-switch blocker {escalation_id}')


@main.command('probe-models')
@pass_config()
@click.option('--models', 'models_csv', default=None,
              help='Comma-separated model list to probe, overriding the default '
                   '(config.routing.allowed_models plus the fable candidate model).')
@click.option('--output', 'output_path', type=click.Path(path_type=Path),
              default=None,
              help='Where to write the rendered probe artifact YAML (default: '
                   'routing.DEFAULT_PROBE_ARTIFACT_PATH, i.e. config/model-availability.yaml).')
@click.option('--budget-usd', 'budget_usd',
              type=click.FloatRange(min=0.0, min_open=True),
              default=None,
              help='Per-invocation USD ceiling for each probe turn (default: '
                   'routing.DEFAULT_PROBE_BUDGET_USD). Must exceed one turn of the '
                   'most expensive probed model — undershooting it reports '
                   'budget_too_low, not unavailability.')
def probe_models(config: OrchestratorConfig, models_csv: str | None, output_path: Path | None,
                 budget_usd: float | None):
    """Probe every configured pool account x candidate model for availability
    and write the rendered status artifact (default config/model-availability.yaml).

    The probed model set defaults to config.routing.allowed_models plus the
    fable candidate model (see routing.probe_models); pass --models to
    override with an explicit comma-separated list.

    The per-turn budget defaults to routing.DEFAULT_PROBE_BUDGET_USD and must
    clear one turn of the most expensive probed model; a pair that aborts on
    that ceiling is recorded as budget_too_low, never as unavailability. Such
    pairs are counted and warned about on stderr, and a run in which EVERY
    probed pair aborted exits non-zero -- it produced no availability
    evidence at all, so it must not read as a successful probe.
    """
    from datetime import UTC, datetime

    from orchestrator import routing

    models = (
        [m.strip() for m in models_csv.split(',') if m.strip()] or None
        if models_csv
        else None
    )

    budget = routing.DEFAULT_PROBE_BUDGET_USD if budget_usd is None else budget_usd
    report = asyncio.run(routing.probe_models(
        config.usage_cap.accounts, config.routing.allowed_models, models=models,
        budget_usd=budget,
    ))

    generated_at = datetime.now(UTC).isoformat()
    artifact = routing.render_probe_artifact(report, generated_at)

    out_path = output_path or Path(routing.DEFAULT_PROBE_ARTIFACT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(artifact)

    click.echo(f'Wrote model availability artifact to {out_path}')

    # A POSITIVE but mis-sized ceiling passes the parse-time FloatRange check
    # and still yields an artifact that is uniformly and plausibly wrong, so
    # the artifact alone cannot be the only signal: an operator who never
    # opens the YAML would read exit 0 as evidence the models are available.
    # The artifact is written first either way -- budget_too_low rows are
    # honest evidence about the BUDGET, and must not be discarded.
    statuses = [status for row in report.accounts.values() for status in row.values()]
    aborted = sum(1 for status in statuses
                  if status == routing.PROBE_BUDGET_TOO_LOW_STATUS)
    if aborted:
        click.echo(
            f'WARNING: {aborted} of {len(statuses)} (account, model) pairs aborted '
            f'on the budget ceiling in force (--budget-usd {budget}); those rows are '
            f'NOT availability evidence. Re-run with a higher --budget-usd.',
            err=True,
        )
        if aborted == len(statuses):
            click.echo(
                'Error: every probed pair aborted on the budget ceiling — this run '
                'produced no availability evidence.',
                err=True,
            )
            sys.exit(1)


@main.command('check-config')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to the project orchestrator config YAML to lint (REQUIRED '
                   'unless ORCH_CONFIG_PATH is set).')
def check_config(config_path: Path | None):
    """Lint a project config YAML for unknown keys that pydantic silently drops.

    OrchestratorConfig uses ``extra='ignore'``, so any key with no matching model
    field is DISCARDED before validation with no error — the 2026-07-22 incident
    where a top-level ``spare_warm_lanes: 8`` (the field lives on ``git.``) was
    dropped for weeks.  This offline gate walks the RAW project YAML against the
    schema via ``census_config_keys`` DIRECTLY (not a full validated load), so it
    still reports phantom keys even when the config has an unrelated value-level
    validation error.

    A key deliberately present for NON-OrchestratorConfig consumers (e.g. one the
    project's own scripts read) can be excused two ways, and is then listed in an
    INFORMATIONAL section that never affects the exit code:

    \b
      * name it with the reserved ``x_``/``x-`` prefix (works at any depth, no
        config ceremony) — the preferred form for a NEW knob;
      * add it to ``config_key_census.ignore`` in the same YAML (fnmatch globs,
        so ``cpu_governance.*`` opts out a whole namespace) — for existing
        names other tooling already greps for.

    An ignore entry is an ASSERTION that some non-orchestrator consumer reads
    the key, so it should carry a justification:

    \b
      config_key_census:
        ignore:
          - path: cpu_governance.weights
            reason: read verbatim by scripts/cpu-governed-exec.sh
          - path: warm_lane_pool
            reason: temporary — pending #5908, which deletes this entry

    A bare string still works but reports as un-reasoned DEBT.  If the consumer
    has NOT landed yet, the reason must cite its tracking task as ``#NNNN``:
    an uncited "pending" claim has no expiry, so nothing will ever re-check it.

    \b
    Exit codes:
      0 — no unknown keys and no HARD ignore-entry findings (advisory findings
          are reported but stay exit-neutral).  An EMPTY project YAML is a
          legitimate clean result — it means "use all defaults" — and exits 0;
      1 — at least one GENUINELY-unknown key, OR the file could not be read or
          parsed at all (a census of nothing is not a clean census, task 4124);
          dominates 2;
      2 — no unknown keys, but at least one HARD ignore-entry finding
          (self-refuting / missing-cite / orphaned).
    """
    from orchestrator.config import census_config_keys
    from orchestrator.config_census_ignore import HARD_KINDS

    # Resolve the config path (arg wins, then ORCH_CONFIG_PATH) without
    # constructing a validated config — census only needs the raw YAML path.
    if config_path is None:
        env_path = os.environ.get('ORCH_CONFIG_PATH')
        if not env_path:
            click.echo(
                'Error: --config is required (or set ORCH_CONFIG_PATH).', err=True
            )
            sys.exit(1)
        config_path = Path(env_path)
        if not config_path.exists():
            click.echo(f'Error: Config file not found: {config_path}', err=True)
            sys.exit(1)

    census = census_config_keys(config_path)

    # An empty census has two very different causes.  Fail CLOSED when nothing
    # could be parsed: the census's own fail-open contract (config.py) makes
    # `unknown` empty for an unreadable/unparseable file, so printing the
    # affirmative OK below would tell an operator a config is safe precisely
    # when it could not be inspected at all.  The ignored/OK sections are both
    # vacuous in this case, so return before either — and so is the
    # ignore-entry audit below, which re-reads the same unparseable file.
    if census.parse_error:
        click.echo(f'Error: {census.parse_error}', err=True)
        click.echo(
            'Could not lint this file, so its config keys are UNKNOWN — this is '
            'NOT a clean result. Fix the file and re-run.',
            err=True,
        )
        sys.exit(1)

    # A broken lint must never turn a working gate into a crash, so the audit
    # (which reaches the filesystem and a sqlite store) degrades to "no
    # findings" rather than taking check-config down with it.
    try:
        findings = audit_census_ignore_entries(config_path)
    except Exception as exc:  # reported, never swallowed
        click.echo(f'WARNING: ignore-entry audit failed ({exc}) — skipped.', err=True)
        findings = []

    # Informational FIRST, and explicitly marked as such: these keys were
    # deliberately excused, so listing them keeps an over-broad glob auditable
    # without ever reading as a failure or touching the exit code.
    if census.ignored:
        _REASONS = {
            'reserved_prefix': 'ignored: reserved prefix',
            'allowlist': 'ignored: config_key_census.ignore',
        }
        click.echo(
            f'{len(census.ignored)} key(s) excused from the census '
            '(informational — does not affect the exit code):'
        )
        for ik in census.ignored:
            label = _REASONS.get(ik.reason, f'ignored: {ik.reason}')
            # An allowlisted key with no operator justification is DEBT, and
            # saying so is the whole point: an unexplained entry is an
            # unfalsifiable assertion about a consumer nobody can check.
            if ik.note:
                suffix = f'  — {ik.note}'
            elif ik.reason == 'allowlist':
                suffix = '  — no reason given (undocumented debt)'
            else:
                suffix = ''
            click.echo(f'  {ik.path}  ({label}){suffix}')
        click.echo('')

    hard = [f for f in findings if f.kind in HARD_KINDS]
    if findings:
        advisory = [f for f in findings if f.kind not in HARD_KINDS]
        click.echo(f'{len(findings)} finding(s) in config_key_census.ignore entries:')
        for f in hard + advisory:
            click.echo(f'  [{f.severity}] {f.kind}: {f.detail}')
        click.echo(
            f'  ({len(advisory)} advisory finding(s) do not affect the exit code.)'
        )
        click.echo('')

    if not census.unknown:
        if hard:
            click.echo(
                f'FAIL: {config_path} has {len(hard)} hard ignore-entry '
                'finding(s) (no unknown config keys).'
            )
            sys.exit(2)
        click.echo(f'OK: {config_path} has no unknown config keys.')
        sys.exit(0)

    click.echo(f'Found {len(census.unknown)} unknown config key(s) in {config_path}:')
    for uk in census.unknown:
        if uk.shadow_hint:
            # Advisory ONLY: a shadow hint is a NAME match against the model
            # tree and may be a coincidental collision, so it stays phrased as a
            # question rather than an instruction to move the key.
            click.echo(f'  {uk.path}  → did you mean {uk.shadow_hint}?')
        else:
            click.echo(f'  {uk.path}')
    sys.exit(1)


def _hand_exit_to_fired_watchdog(
    fired: threading.Event | None, thread: threading.Thread | None,
) -> None:
    """If the stdin watchdog has fired, block on it; it owns the remote verify's exit.

    A fired watchdog has already tree-killed the build and is mid-way through
    its own SIGTERM -> grace_secs sleep -> SIGKILL escalation (see
    fire_watchdog_kill), so never print a (misleading) VerifyResult for a
    build we just killed out from under ourselves. Block on the watchdog
    thread instead of exiting immediately: a bare sys.exit(1) begins
    interpreter shutdown right away, and since the watchdog thread is a
    daemon thread, shutdown does not wait for it -- it can be torn down
    mid-sleep, before the SIGKILL escalation that exists specifically to reap
    start_new_session grandchildren (cargo/rustc) that survived SIGTERM.
    Joining lets fire_watchdog_kill's own unconditional os._exit(1) be the
    authoritative exit once the full escalation has run to completion.

    The watchdog's SIGTERM routinely makes the body RAISE rather than return
    (e.g. it kills ``git worktree add``, rc=-15), so the shared prologue
    (:func:`_run_remote_verify`) calls this from its ``finally``: every exit
    path must hand off, not just a return.
    """
    if fired is None or not fired.is_set():
        return
    if thread is not None:
        thread.join()
    sys.exit(1)


class _RemoteVerifyContext(NamedTuple):
    """One remote verify invocation's host state, shared by its preflight and body."""

    config: OrchestratorConfig
    git_ops: 'GitOps'
    pgid_path: Path | None


_SpecT = TypeVar('_SpecT')


def _run_remote_verify(
    config: OrchestratorConfig,
    spec_json: str,
    request_id: str | None,
    *,
    parse_spec: Callable[[str], _SpecT],
    body: Callable[[_RemoteVerifyContext, _SpecT], Coroutine[Any, Any, 'VerifyResult']],
    preflight: Callable[
        [_RemoteVerifyContext], AbstractContextManager['VerifyResult | None']
    ] | None = None,
) -> None:
    """The prologue and exit contract every remote verify subcommand shares.

    Parses ``--spec``, joins its own process group and writes
    the pgid file (with ``--request-id``), enters *preflight*, starts the
    stdin watchdog, then runs *body*.  Prints one VerifyResult JSON to stdout
    and exits 0 whenever it has one (even ``passed=False``); exits 1 on bad
    input or an infrastructure error.  *preflight* yields a VerifyResult to
    emit instead of running *body*, or None to proceed; whatever it holds is
    released first on the way out.
    """
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_runner import result_to_json

    try:
        spec = parse_spec(spec_json)
    except Exception as e:
        click.echo(f'Error: invalid --spec: {e}', err=True)
        sys.exit(1)

    # Construct GitOps once; reused for both pgid-path derivation (when --request-id
    # is set) and the body's worktree work.  GitOps.__init__ is side-effect-free so
    # constructing it here is safe in all paths.
    git_ops = GitOps(config.git, config.project_root)

    # Cancellation support: write pgid file before starting work so that a
    # concurrent `cancel-verify --request-id X` can locate and kill this process.
    pgf: Path | None = None
    if request_id is not None:
        pgf = pgid_file(git_ops.worktree_base, request_id)
        pgid = start_own_process_group()
        write_pgid_file(pgf, pgid)
    ctx = _RemoteVerifyContext(config, git_ops, pgf)

    preflight_hold = contextlib.ExitStack()
    preflight_result = (
        preflight_hold.enter_context(preflight(ctx)) if preflight is not None else None
    )
    if preflight_result is not None:
        preflight_hold.close()
        # Always remove the pgid file so cancel-verify knows this run is done.
        if pgf is not None:
            remove_pgid_file(pgf)
        # Any preflight result exits 0: stdout is the only channel to
        # RemoteRunner, which reads a non-zero exit as RunnerUnavailable and
        # discards the result.
        click.echo(result_to_json(preflight_result))
        return

    # Connection-death watchdog (task 2308 γ): ties this process's lifetime to
    # the ssh dispatch channel on fd 0. Fires on stdin EOF (channel closed) or
    # heartbeat starvation (hard partition), killing the build subtree and
    # self-exiting instead of surviving as a setsid orphan. Same pgid as the
    # pgid file, so a concurrent cancel-verify still tree-kills coherently.
    #
    # watchdog_fired is set as the FIRST action of the fire callback --
    # strictly before any signal is sent -- so it always happens-before the
    # kill that unblocks _run_cmd's awaited subprocess. Without this flag,
    # a killed build command returns a normal (if failed) VerifyResult up
    # through the body, and the main thread can reach the click.echo/return
    # below and exit 0 in a race against fire_watchdog_kill's own grace-period
    # sleep + os._exit(1) -- observed as flaky exit codes on the same
    # watchdog-fired outcome (task 2309 boundary gate).
    watchdog_fired: threading.Event | None = None
    watchdog_thread: threading.Thread | None = None
    if request_id is not None:
        heartbeat_timeout = _env_float(
            'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS', WATCHDOG_HEARTBEAT_TIMEOUT_SECS
        )
        grace_secs = _env_float('ORCH_WATCHDOG_KILL_GRACE_SECS', WATCHDOG_KILL_GRACE_SECS)
        watchdog_fired = threading.Event()

        def _on_watchdog_fire(trigger: WatchdogTrigger) -> None:
            watchdog_fired.set()
            fire_watchdog_kill(pgid, trigger=trigger, grace_secs=grace_secs)

        watchdog_thread = start_stdin_watchdog(
            pgid, heartbeat_timeout=heartbeat_timeout, fire=_on_watchdog_fire
        )

    try:
        result = asyncio.run(body(ctx, spec))
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    finally:
        preflight_hold.close()
        # Always remove the pgid file so cancel-verify knows this run is done.
        if pgf is not None:
            remove_pgid_file(pgf)
        _hand_exit_to_fired_watchdog(watchdog_fired, watchdog_thread)

    click.echo(result_to_json(result))


@contextlib.contextmanager
def _merge_lane_gate(ctx: _RemoteVerifyContext) -> Iterator['VerifyResult | None']:
    """verify-merge's preflight: the laptop lane-lock contention gate.

    Yields a distinguished contention VerifyResult when a lane actor holds the
    lane (the caller emits it without touching the tree), else None while
    holding the compat co-lock until exit.  The contention result is a
    passed=False verdict that exits 0, so beta receives its discriminant
    rather than a RunnerUnavailable.
    """
    from orchestrator.verify_runner import make_flock_contention_result

    git_ops = ctx.git_ops
    # Flock guard (task 2306 α; converged onto the shared lane lock in task 2830):
    # serialize the verify span under a laptop-side exclusive fcntl.flock when the
    # persistent-worktree knob is on — the per-host serial invariant that
    # _bump_host_verify_attempt_count relies on is only supplied at WORKSTATION
    # startup, not on the laptop.
    #
    # DUAL-LOCK with SPLIT lane-lock lifetime (task 2830, esc-2830-1): the PRIMARY
    # lock is the SHARED <lane_dir>.lock (lane_lock_path(persistent_merge_worktree_path)
    # = <worktree_base>/_merge-verify.lock) — the SAME lock GitOps.merge_verify_lease,
    # GitOps.reset_persistent_merge_worktree, and reify's seed/thin/gc take (task
    # 2685) — so a laptop lane actor is mutually excluded from a live laptop verify.
    # Because reset_persistent_merge_worktree re-acquires this SAME lane lock across
    # its own tree mutation, the CLI CANNOT hold it continuously across the body (an
    # in-process second flock on the same inode self-conflicts — esc-2830-1). The
    # lane lock therefore has a SPLIT lifetime, mirroring the LOCAL flow's sequential
    # reset-hold-then-lease-hold (merge_queue.py):
    #   (a) here, PRE-FLIGHT — a bounded-wait acquire used purely as a contention
    #       GATE (a live lane actor -> distinguished result, NO tree touch, PRD
    #       Invariant 5), then RELEASED before the body so reset can take it;
    #   (b) inside _verify_merge_body, re-acquired around the BUILD only.
    #
    # The divergent .merge_verify.lock (merge_verify_lock_path) is RETAINED as a
    # transitional rollout CO-lock, held across the WHOLE span: it is a DIFFERENT
    # inode (no self-conflict with reset), so its continuous hold both (i) closes
    # the momentary lane-release gaps between the gate, reset, and the build against
    # a concurrent waiter, and (ii) still mutually excludes an in-flight OLD
    # verify-merge (holding only that lock, before checkout-sync ships this code)
    # during rollout. A post-rollout follow-up drops it, leaving the CLI on the lane
    # lock's split lifetime alone (matching merge_verify_lease).
    #
    # Acquire LANE-first (the gate), then COMPAT. The shared holder-pgid is written
    # by the body's build-scoped hold, NOT here: an in-flight OLD caller wrote that
    # holder when it took the old lock, so a NEW waiter that loses the bounded wait
    # must still read the CORRECT (un-clobbered) holder on the contention path below.
    # acquire_merge_verify_flock is a bounded-wait poll (returns None on timeout,
    # never blocks), so the two-lock acquire cannot deadlock — worst case is a
    # timeout -> contention.
    #
    # Knob OFF -> lane_fd/compat_fd stay None -> byte-identical back-compat (no lock).
    lane_fd: int | None = None
    compat_fd: int | None = None
    contention_result = None
    if ctx.config.git.persistent_merge_worktree:
        flock_wait_secs = _env_float(
            'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS', MERGE_VERIFY_FLOCK_WAIT_SECS
        )
        # (1) PRIMARY: the shared <lane_dir>.lock (task 2830 — the fix).
        lane_fd = acquire_merge_verify_flock(
            lane_lock_path(git_ops.persistent_merge_worktree_path), flock_wait_secs
        )
        if lane_fd is None:
            # Bounded wait timed out: a laptop lane actor (or another verify-merge)
            # holds the shared lane lock. Emit the distinguished contention result
            # WITHOUT ever touching the tree (no acquire_host_verify_worktree, no
            # ephemeral _merge-<uuid> fallback — PRD Invariant 5).
            #
            # holder_pgid can legitimately be None here — a DIAGNOSTIC-only window,
            # not a correctness gap. The shared holder is written by the body's
            # build-scoped hold, NOT at this gate (see the DUAL-LOCK note above). So
            # when the current holder is another NEW-code verify-merge that has
            # passed its own gate, released the lane lock, and is mid-reset inside
            # acquire_host_verify_worktree — reset_persistent_merge_worktree takes
            # the lane lock but never writes the holder-pgid (git_ops.py) — this read
            # returns None for that window. Mutual exclusion and the gate contract
            # are intact; make_flock_contention_result tolerates holder_pgid=None
            # (only the diagnostic's holder_pgid field is null). A follow-up could
            # restore always-present holder visibility by also writing the shared
            # holder at this gate, at the cost of extra cleanup on the body's
            # fail-open (build_lane_fd is None) path.
            holder_pgid = read_lock_holder_pgid(git_ops.worktree_base)
            contention_result = make_flock_contention_result(
                host=socket.gethostname(),
                holder_pgid=holder_pgid,
                waiter_pgid=os.getpgrp(),
            )
        else:
            # (2) COMPAT rollout co-lock: the divergent .merge_verify.lock, so an
            # in-flight OLD verify-merge (holding only that lock) still mutually
            # excludes during rollout.
            compat_fd = acquire_merge_verify_flock(
                merge_verify_lock_path(git_ops.worktree_base), flock_wait_secs
            )
            if compat_fd is None:
                # An in-flight OLD caller holds .merge_verify.lock. Read the holder
                # it recorded (NOT yet clobbered — we write ours only after both
                # locks are held), emit the distinguished result, and release the
                # already-held lane lock before bailing so the primary lock is not
                # leaked on this fail-closed path. holder_pgid may also be None here
                # (a NEW-code holder mid-reset — see the lane-timeout path above).
                holder_pgid = read_lock_holder_pgid(git_ops.worktree_base)
                contention_result = make_flock_contention_result(
                    host=socket.gethostname(),
                    holder_pgid=holder_pgid,
                    waiter_pgid=os.getpgrp(),
                )
                release_merge_verify_flock(lane_fd)
                lane_fd = None
            else:
                # (3) Gate passed — no laptop lane actor is present. RELEASE the
                # lane lock now: reset_persistent_merge_worktree (inside
                # acquire_host_verify_worktree) re-acquires this SAME lane lock
                # across its tree mutation, and an in-process second flock on the
                # same inode would self-conflict (esc-2830-1). The body re-takes the
                # lane lock and writes the shared holder-pgid for the build's
                # duration only. The compat co-lock (compat_fd) stays held across
                # the WHOLE span — a DIFFERENT inode, so no self-conflict — closing
                # the momentary lane-release gaps against a concurrent waiter and
                # preserving rollout back-compat exclusion.
                release_merge_verify_flock(lane_fd)
                lane_fd = None

    try:
        yield contention_result
    finally:
        # Past the contention gate the lane lock is already released (the gate
        # holds it only pre-flight; the body's build-scoped hold manages the lane
        # lock and the shared holder-pgid — task 2830 split lifetime, esc-2830-1).
        # Only the compat co-lock is held across the whole span; release it.
        if compat_fd is not None:
            release_merge_verify_flock(compat_fd)


@contextlib.asynccontextmanager
async def _verify_worktree_lifetime(
    ctx: _RemoteVerifyContext, wt: Path,
) -> AsyncIterator[Path]:
    """Own *wt* for one remote verify: record it beside the pgid file, always remove it.

    The record lets a cancel-verify, whose SIGKILL skips this ``finally``, reap
    *wt* instead.
    """
    try:
        if ctx.pgid_path is not None:
            write_worktree_path(ctx.pgid_path, wt)
        yield wt
    finally:
        await ctx.git_ops.cleanup_merge_worktree(wt)
        if ctx.pgid_path is not None:
            remove_worktree_path(ctx.pgid_path)


async def _verify_merge_body(
    sha: str, ctx: _RemoteVerifyContext, spec: 'MergeVerifySpec',
) -> 'VerifyResult':
    """verify-merge's body: the host's verify worktree at *sha*, then the merge bundle."""
    from orchestrator.verify_runner import run_merge_verify_on_worktree

    git_ops = ctx.git_ops
    wt = await git_ops.acquire_host_verify_worktree(sha)
    async with _verify_worktree_lifetime(ctx, wt):
        # LANE-lock hold scoped to the BUILD only (task 2830, esc-2830-1).
        #
        # acquire_host_verify_worktree -> reset_persistent_merge_worktree
        # (git_ops.py) itself acquires the SAME lane lock across its tree
        # mutation and releases it before returning. The CLI span therefore
        # must NOT hold the lane lock across the reset: a second in-process
        # fcntl.flock on the same inode via an independent fd self-conflicts
        # (flock(2): independent fds are treated independently and deny each
        # other), so reset's bounded lane-lock wait could never succeed —
        # 30s timeout -> RuntimeError -> verify exits without ever building.
        #
        # Instead mirror the LOCAL flow (merge_queue.py: reset-hold released,
        # THEN merge_verify_lease re-acquires the same lane lock around the
        # build): re-take the lane lock HERE, after the reset returns, for the
        # build's duration only, and record the shared holder-pgid so a
        # concurrent lane actor / waiter reads the correct holder. Fail-OPEN
        # on a contended acquire (proceed unrecorded, mirroring
        # GitOps.merge_verify_lease) — any live actor is already excluded by
        # the pre-flight gate + the compat co-lock held across the whole span.
        build_lane_fd: int | None = None
        if ctx.config.git.persistent_merge_worktree:
            build_wait = _env_float(
                'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS', MERGE_VERIFY_FLOCK_WAIT_SECS
            )
            build_lane_fd = await asyncio.to_thread(
                acquire_merge_verify_flock,
                lane_lock_path(git_ops.persistent_merge_worktree_path),
                build_wait,
            )
            if build_lane_fd is not None:
                write_lock_holder_pgid(git_ops.worktree_base, os.getpgrp())
        try:
            return await run_merge_verify_on_worktree(wt, ctx.config, spec, merge_sha=sha)
        finally:
            if build_lane_fd is not None:
                remove_lock_holder_pgid(git_ops.worktree_base)
                release_merge_verify_flock(build_lane_fd)


@main.command('verify-merge')
@click.option('--sha', required=True, help='Merge commit SHA to verify (must be present in the local repo)')
@click.option('--spec', 'spec_json', required=True, help='MergeVerifySpec as a JSON string (from RemoteRunner dispatch)')
@pass_config()
@click.option('--request-id', default=None,
              help='Optional request ID for cancellation support (task 1732 α).  '
                   'When provided: joins a new process group via setsid, writes the pgid to '
                   '<worktree_base>/.merge_verify_pgids/<request-id> at startup, and removes '
                   'it on exit (normal or exceptional).  A concurrent '
                   '``orchestrator cancel-verify --request-id X`` can then kill the entire '
                   'descendant tree (including start_new_session build escapes).  '
                   'Absent → today\'s exact behavior, back-compat.  '
                   'RemoteRunner wiring is deferred to tasks β/γ.')
def verify_merge(sha: str, spec_json: str, config: OrchestratorConfig, request_id: str | None):
    """Run the merge-verify bundle at a given SHA and emit a VerifyResult JSON to stdout.

    Materialises a detached worktree at --sha, runs the same scoped + unscoped
    verify bundle the merge queue uses (fidelity by construction), and emits a
    single VerifyResult JSON document to stdout.  All logs go to stderr.  Exit
    code is 0 on success (even when passed=False); non-zero only on bad input or
    infrastructure errors.

    When ``git.persistent_merge_worktree`` is on (PRD §8 η), the subcommand
    reuses the host's own fixed-path warm worktree (``_merge-verify``) across
    invocations via :meth:`~orchestrator.git_ops.GitOps.acquire_host_verify_worktree`,
    mirroring κ invariants 1–6 on the laptop host.  The periodic from-scratch
    safety valve (invariant 6) is driven by a disk-persistent per-host attempt
    counter so it fires correctly even across stateless CLI invocations.

    When ``--request-id`` is supplied, the subcommand also calls ``os.setsid``
    (via :func:`~orchestrator.verify_cancel.start_own_process_group`) to join a
    new process group and writes its pgid to
    ``<worktree_base>/.merge_verify_pgids/<request-id>``, recording its
    worktree beside it.  Both are removed on exit.  A concurrent
    ``orchestrator cancel-verify --request-id X`` reads the file, kills the
    full descendant tree (capturing ``start_new_session`` build escapes via a
    ``/proc`` PPID walk) plus a ``killpg`` backstop, and reaps the worktree.

    **pgid-file directory**: ``<worktree_base>/.merge_verify_pgids`` —
    host-side, per-project, never pruned or git-cleaned (mirrors the
    ``.merge_verify_host_attempts`` counter precedent at the same path).
    ``worktree_base`` is ``GitOps(config.git, config.project_root).worktree_base``.
    Both hosts run the ``df`` checkout, so landing ``--request-id`` support on
    ``main`` ships the contract to the laptop via its normal checkout sync.

    Consumer: RemoteRunner (δ) parses stdout as a VerifyResult.
    """
    from orchestrator.verify_runner import spec_from_json

    _run_remote_verify(
        config, spec_json, request_id,
        parse_spec=spec_from_json,
        body=functools.partial(_verify_merge_body, sha),
        preflight=_merge_lane_gate,
    )


def _task_spec_for_sha(sha: str, spec_json: str) -> 'TaskVerifySpec':
    """Parse ``--spec``, refusing a spec whose tip_sha is not ``--sha`` (INV-3, receiving end)."""
    from orchestrator.task_verify_spec import task_spec_from_json

    spec = task_spec_from_json(spec_json)
    if spec.tip_sha != sha:
        raise ValueError(f'spec tip_sha {spec.tip_sha} is not --sha {sha}')
    return spec


async def _verify_task_body(
    sha: str, ctx: _RemoteVerifyContext, spec: 'TaskVerifySpec',
) -> 'VerifyResult':
    """verify-task's body: a fresh ephemeral worktree at *sha*, then the task-role verify."""
    from orchestrator.task_verify_run import run_task_verify_on_worktree

    wt = await ctx.git_ops.create_ephemeral_verify_worktree(sha)
    async with _verify_worktree_lifetime(ctx, wt):
        return await run_task_verify_on_worktree(wt, ctx.config, spec)


@main.command('verify-task')
@click.option('--sha', required=True, help='Task tip SHA to verify (must be present in the local repo)')
@click.option('--spec', 'spec_json', required=True,
              help='TaskVerifySpec as a JSON string (from RemoteRunner.run_task_verify)')
@pass_config()
@click.option('--request-id', default=None,
              help='Optional cancellation key: joins a new process group and records the '
                   'pgid and worktree under <worktree_base>/.merge_verify_pgids/ for '
                   '``orchestrator cancel-verify --request-id X``.')
def verify_task(sha: str, spec_json: str, config: OrchestratorConfig, request_id: str | None):
    """Run a task-role verify of --sha on a fresh ephemeral worktree; print one VerifyResult JSON.

    Exit 0 on any verdict; exit 1 on a config, spec or infrastructure error, or
    when the stdin watchdog fires.  The worktree is recorded beside the pgid
    file so ``cancel-verify`` reaps it.  Always ephemeral: no lane flock, no
    unscoped typecheck, no flake suppression.  PRD:
    plans/remote-task-leg-verify-prd.md §5.3.
    """
    _run_remote_verify(
        config, spec_json, request_id,
        parse_spec=functools.partial(_task_spec_for_sha, sha),
        body=functools.partial(_verify_task_body, sha),
    )


@main.command('verify-host-preflight')
@click.option('--runner', required=True, help='A declared verify runner, enabled or not')
@click.option('--sha', default=None, help='Commit to verify (default: the tip of git.main_branch)')
@click.option('--reference', default=None,
              help='A second declared runner verifying the same commit, for skip parity')
@click.option('--json', 'as_json', is_flag=True, help="Print the runner's record as JSON")
@pass_config()
def verify_host_preflight(
    runner: str, sha: str | None, reference: str | None, as_json: bool, config: OrchestratorConfig,
):
    """Run the full merge gate on one verify host through the production remote path.

    Exit 0 when the gate passed and, with --reference, skip parity holds; 1
    when the gate is red, vacuous or produced no verdict, or parity is broken
    or undetermined; 2 when refused before any dispatch.  The record lands in
    data/verify-logs/preflight/<runner>-<sha>.json; see
    docs/verify-host-preflight.md.
    """
    from orchestrator.deployed_sha import capture_deployed_sha, resolve_code_checkout

    # The gate the preflight builds is this process's own wire path (task 6565 D1).
    deployed_sha = capture_deployed_sha(
        resolve_code_checkout(),
        lambda: importlib.import_module('orchestrator.verify_host_preflight'),
    )
    from orchestrator import verify_host_preflight as preflight

    if deployed_sha is None:
        click.echo(
            'Error: this process cannot read the commit it runs (no readable git checkout '
            'above its code), and the remote gate is built from that sha (task 6565)',
            err=True,
        )
        sys.exit(2)
    try:
        outcome = asyncio.run(preflight.run_preflight(
            config, runner=runner, sha=sha, reference=reference, deployed_sha=deployed_sha,
            build_runner=preflight.production_runner_builder(config, deployed_sha),
        ))
    except (preflight.PreflightRefused, preflight.SkipParityExceptionsError) as exc:
        click.echo(f'Error: {exc}', err=True)
        sys.exit(2)
    if as_json:
        click.echo(json.dumps(preflight.record_dict(outcome.primary, outcome.parity), indent=2, sort_keys=True))
    else:
        click.echo(preflight.render_summary(outcome))
    sys.exit(0 if outcome.ok else 1)


def _recorded_merge_worktree_remover(git_ops: 'GitOps') -> Callable[[Path], None]:
    """Return cancel-verify's remover for the worktree a killed run recorded.

    It reaps only a ``_merge-`` band entry directly under ``worktree_base``,
    through the single guarded remover: ``cleanup_merge_worktree``'s primary
    path does not band-check, so a corrupt record must not reach it with a
    task worktree.
    """
    from orchestrator.git_ops import classify_worktree_entry

    def remove(worktree: Path) -> None:
        in_merge_band = (
            worktree.resolve().parent == git_ops.worktree_base.resolve()
            and classify_worktree_entry(worktree.name) == 'merge'
        )
        if not in_merge_band:
            click.echo(
                f'cancel-verify: not removing recorded worktree {worktree}: it is '
                f'not a _merge- entry directly under {git_ops.worktree_base}',
                err=True,
            )
            return
        asyncio.run(git_ops.cleanup_merge_worktree(worktree))

    return remove


@main.command('gate-identity')
def gate_identity():
    """Print this checkout's gate identity as one JSON line (task 6565, D3).

    ``{"code_sha": <HEAD of the checkout holding this code>,
    "wire_fingerprint": <remote_gate.wire_fingerprint of the wire classes>}``.
    A remote gate answers this before any verify is dispatched to it; the
    dispatcher proceeds only when both values equal its own.  Importing the
    wire classes to fingerprint them also proves the gate's codec imports.
    Exits 1 with the cause on stderr when the checkout's HEAD cannot be read.
    """
    import subprocess

    from orchestrator.deployed_sha import checkout_head, resolve_code_checkout
    from orchestrator.remote_gate import local_gate_identity

    checkout = resolve_code_checkout()
    if checkout is None:
        click.echo('Error: no git checkout holds this orchestrator code', err=True)
        sys.exit(1)
    try:
        code_sha = checkout_head(checkout)
    except (OSError, subprocess.CalledProcessError) as exc:
        click.echo(f'Error: cannot read HEAD of {checkout}: {exc}', err=True)
        sys.exit(1)
    click.echo(json.dumps(local_gate_identity(code_sha).to_wire()))


@main.command('cancel-verify')
@click.option('--request-id', required=True,
              help='Request ID passed to the verify-merge or verify-task invocation '
                   'to cancel; the worktree that run recorded is reaped too.')
@pass_config(help=f'{CONFIG_OPTION_HELP_PREFIX} Selects the target project — determines '
                  'worktree_base for the pgid-file lookup.')
def cancel_verify(request_id: str, config: OrchestratorConfig):
    """Kill the verify-merge or verify-task tree identified by --request-id and exit 0.

    Reads the pgid file at
    ``<worktree_base>/.merge_verify_pgids/<request-id>`` (written by
    ``verify-merge --request-id``), snapshots the ``/proc`` PPID map, and
    ``SIGKILL``\\s every descendant plus a ``killpg`` backstop.  This reaps the
    entire process tree including ``start_new_session`` build-command escapes
    (which leave ``verify-merge``'s process group but remain its descendants in
    the ``/proc`` parent chain).  The SIGKILL skips that run's own ``finally``,
    so the ``_merge-`` worktree it recorded beside the pgid file is reaped here
    (:func:`_recorded_merge_worktree_remover`).

    **Idempotent**: exits 0 when the file is absent (already cancelled or never
    started), when the content is corrupt, or when all processes are already
    dead.  Exits non-zero only when a live process could not be killed
    (``SIGKILL`` raised ``PermissionError``); in that case the pgid file is
    retained so a retry can act on it.

    pgid-file directory: ``<worktree_base>/.merge_verify_pgids`` — host-side,
    per-project, never pruned/git-cleaned (mirrors the
    ``.merge_verify_host_attempts`` counter precedent).  ``worktree_base`` is
    derived via ``GitOps(config.git, config.project_root).worktree_base``.

    Cross-host rollout: both the server and the laptop host run the ``df``
    checkout, so landing this command on ``main`` ships the contract to the
    laptop via its normal checkout sync.
    """
    from orchestrator.git_ops import GitOps

    git_ops = GitOps(config.git, config.project_root)
    pgf = pgid_file(git_ops.worktree_base, request_id)
    failed_pids: list[int] = []
    killed_pgid: list[int] = []
    rc = cancel_request(
        pgf, failed_pids_out=failed_pids, killed_pgid_out=killed_pgid,
        remove_worktree=_recorded_merge_worktree_remover(git_ops),
    )
    # ── task 3186 (PRD δ): CLEAR THE FIXED-KEY HOLDER RENDEZVOUS ────────────
    # `cancel_request` SIGKILLs the verify-merge tree, which skips that
    # process's own `finally` — the one that would have called
    # `remove_lock_holder_pgid`.  The per-request pgid file it removes itself
    # is harmless when leaked (request ids are uuid4 and never revisited), but
    # the FIXED key is a different animal: every run overwrites it, and
    # `GitOps._merge_verify_lease_active` probes it with `killpg(pgid, 0)`.  A
    # leaked — or, once the pid counter wraps, RECYCLED — entry there reads as
    # a LIVE holder, and `reset_persistent_merge_worktree` consumes that
    # predicate FAIL-CLOSED: it raises `MergeVerifyLeaseHeld` and the warm
    # `_merge-verify` lane is wedged until some later run happens to overwrite
    # the key.  See verify_cancel.py's stale-file / PID-reuse note for the
    # measured picture.
    #
    # GATED ON IDENTITY, NOT ON rc.  Two facts make an unconditional
    # `if rc == 0` clear unsafe:
    #
    #   * rc == 0 does not mean anything was killed.  Three of `cancel_request`'s
    #     four return-0 paths never send a signal (absent file, corrupt content,
    #     `pgid <= 0`), and the absent-file one is the COMMON case — a
    #     verify-merge that completes normally removes its own per-request pgid
    #     file in its `finally`, so a cancel racing normal completion (exactly
    #     the race δ's head teardown creates) kills nothing at all.
    #   * The key is SHARED.  Its owner writes it only when it won the build-lane
    #     flock and clears it only in its own `finally` (see the `verify-merge`
    #     span above), so it routinely names a DIFFERENT, live verify than the
    #     request being cancelled.
    #
    # Clearing it in either of those situations makes the lease read fail-OPEN
    # (`read_lock_holder_pgid` -> None -> "not held"): the typed
    # `MergeVerifyLeaseHeld` diagnosis is lost and DF-3071's admission guard
    # reads `_merge-verify` as IDLE while a verify is live, so the fleet
    # REDEPLOYS over it instead of deferring — trading the wedged-lane risk
    # above for a strictly worse one.  So clear only when this cancel actually
    # swept a pgid AND the key names that same pgid.
    #
    # The rc != 0 case is subsumed and stays fail-closed for its own reason: a
    # LIVE process refused SIGKILL, so it plausibly still holds both lease axes
    # and `cancel_request` reports no kill.  Same reasoning as its retention of
    # the per-request pgid file on that path.
    #
    # SCOPE: this closes the merge-worker-initiated cancel — the route δ's
    # head-verify teardown takes for a REMOTE lease.  The `fire_watchdog_kill`
    # `os._exit(1)` leak (verify_cancel.py's stale-file note) is a different
    # route and is deliberately NOT addressed here.
    if killed_pgid and read_lock_holder_pgid(git_ops.worktree_base) == killed_pgid[0]:
        remove_lock_holder_pgid(git_ops.worktree_base)
    for pid in failed_pids:
        click.echo(
            f'cancel-verify: PermissionError: could not SIGKILL pid {pid} '
            f'(process alive but kill refused) — retry or escalate manually',
            err=True,
        )
    sys.exit(rc)


for command in eval_cli.commands.values():
    main.add_command(command)


def _parse_duration(s: str) -> int:
    """Parse a duration string like '4h', '30m', '90s', or bare seconds."""
    s = s.strip().lower()
    if s.endswith('h'):
        return int(s[:-1]) * 3600
    if s.endswith('m'):
        return int(s[:-1]) * 60
    if s.endswith('s'):
        return int(s[:-1])
    return int(s)


if __name__ == '__main__':
    main()
