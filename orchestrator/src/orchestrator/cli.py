"""CLI entry point: `orchestrator run [--prd X]`."""

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import click
from dotenv import load_dotenv

from orchestrator.config import ConfigRequiredError, load_config
from orchestrator.verify_cancel import (
    WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
    WATCHDOG_KILL_GRACE_SECS,
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
    start_own_process_group,
    start_stdin_watchdog,
    write_lock_holder_pgid,
    write_pgid_file,
)

load_dotenv()  # loads .env into os.environ (e.g. CLAUDE_OAUTH_TOKEN_A/B)

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# How long after asyncio.run() returns to wait before force-exiting.
# The watchdog is armed AFTER asyncio.run() returns (not before) so that
# long orchestration runs are never affected. After this deadline a diagnostic
# dump is written to stderr and os._exit(137) fires.
SHUTDOWN_WATCHDOG_TIMEOUT_SECS = 30

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
    the 10-15s production windows. Unset, unparseable, or non-positive
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
    assertions to guarantee the watchdog thread has exited; the orchestrator
    intentionally never disarms so the watchdog guards interpreter shutdown
    (atexit callbacks + ``threading._shutdown()`` joining non-daemon threads).

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

            lines = ['SHUTDOWN WATCHDOG FIRED — process hung after asyncio.run() returned\n']
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


@click.group()
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
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
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
def run(prd: Path | None, config_path: Path | None, dry_run: bool, delay: str | None,
        force_dirty_start: bool, retag_modules: bool, until_idle: bool):
    """Run the orchestrator against a PRD, or execute existing tasks if no PRD given."""
    from orchestrator.harness import Harness

    delay_secs = _parse_duration(delay) if delay else 0
    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    harness = Harness(config)
    logger = logging.getLogger(__name__)

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
        )

    try:
        report = asyncio.run(_main())
    except asyncio.CancelledError:
        click.echo('Orchestrator cancelled', err=True)
        # asyncio.run() has returned — arm the watchdog NOW to guard interpreter
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
    # joining non-daemon threads). Placed AFTER click.echo so user-visible work is
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
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
def status(config_path: Path | None):
    """Show current task tree and status."""
    from orchestrator.overrides import OverrideStore
    from orchestrator.scheduler import Scheduler

    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
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


@main.command('probe-models')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
@click.option('--models', 'models_csv', default=None,
              help='Comma-separated model list to probe, overriding the default '
                   '(config.routing.allowed_models plus the fable candidate model).')
@click.option('--output', 'output_path', type=click.Path(path_type=Path),
              default=None,
              help='Where to write the rendered probe artifact YAML (default: '
                   'routing.DEFAULT_PROBE_ARTIFACT_PATH, i.e. config/model-availability.yaml).')
def probe_models(config_path: Path | None, models_csv: str | None, output_path: Path | None):
    """Probe every configured pool account x candidate model for availability
    and write the rendered status artifact (default config/model-availability.yaml).

    The probed model set defaults to config.routing.allowed_models plus the
    fable candidate model (see routing.probe_models); pass --models to
    override with an explicit comma-separated list.
    """
    from datetime import UTC, datetime

    from orchestrator import routing

    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    models = (
        [m.strip() for m in models_csv.split(',') if m.strip()] or None
        if models_csv
        else None
    )

    report = asyncio.run(routing.probe_models(
        config.usage_cap.accounts, config.routing.allowed_models, models=models,
    ))

    generated_at = datetime.now(UTC).isoformat()
    artifact = routing.render_probe_artifact(report, generated_at)

    out_path = output_path or Path(routing.DEFAULT_PROBE_ARTIFACT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(artifact)

    click.echo(f'Wrote model availability artifact to {out_path}')


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
      * add its dotted path to ``config_key_census.ignore`` in the same YAML
        (fnmatch globs, so ``cpu_governance.*`` opts out a whole namespace) —
        for existing names other tooling already greps for.

    Exits 1 if any GENUINELY-unknown key is found, else 0.
    """
    from orchestrator.config import census_config_keys

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
            click.echo(f'  {ik.path}  ({_REASONS.get(ik.reason, f"ignored: {ik.reason}")})')
        click.echo('')

    if not census.unknown:
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


@main.command('verify-merge')
@click.option('--sha', required=True, help='Merge commit SHA to verify (must be present in the local repo)')
@click.option('--spec', 'spec_json', required=True, help='MergeVerifySpec as a JSON string (from RemoteRunner dispatch)')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
@click.option('--request-id', default=None,
              help='Optional request ID for cancellation support (task 1732 α).  '
                   'When provided: joins a new process group via setsid, writes the pgid to '
                   '<worktree_base>/.merge_verify_pgids/<request-id> at startup, and removes '
                   'it on exit (normal or exceptional).  A concurrent '
                   '``orchestrator cancel-verify --request-id X`` can then kill the entire '
                   'descendant tree (including start_new_session build escapes).  '
                   'Absent → today\'s exact behavior, back-compat.  '
                   'RemoteRunner wiring is deferred to tasks β/γ.')
def verify_merge(sha: str, spec_json: str, config_path: Path | None, request_id: str | None):
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
    ``<worktree_base>/.merge_verify_pgids/<request-id>``.  The file is removed
    on exit.  A concurrent ``orchestrator cancel-verify --request-id X`` reads
    the file and kills the full descendant tree (capturing ``start_new_session``
    build escapes via a ``/proc`` PPID walk) plus a ``killpg`` backstop.

    **pgid-file directory**: ``<worktree_base>/.merge_verify_pgids`` —
    host-side, per-project, never pruned or git-cleaned (mirrors the
    ``.merge_verify_host_attempts`` counter precedent at the same path).
    ``worktree_base`` is ``GitOps(config.git, config.project_root).worktree_base``.
    Both hosts run the ``df`` checkout, so landing ``--request-id`` support on
    ``main`` ships the contract to the laptop via its normal checkout sync.

    Consumer: RemoteRunner (δ) parses stdout as a VerifyResult.
    """
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_runner import (
        make_flock_contention_result,
        result_to_json,
        run_merge_verify_on_worktree,
        spec_from_json,
    )

    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    try:
        spec = spec_from_json(spec_json)
    except Exception as e:
        click.echo(f'Error: invalid --spec: {e}', err=True)
        sys.exit(1)

    # Construct GitOps once; reused for both pgid-path derivation (when --request-id
    # is set) and the async worktree acquisition inside _run().  GitOps.__init__ is
    # side-effect-free so constructing it here is safe in all paths.
    git_ops = GitOps(config.git, config.project_root)

    # Cancellation support: write pgid file before starting work so that a
    # concurrent `cancel-verify --request-id X` can locate and kill this process.
    pgf: Path | None = None
    if request_id is not None:
        pgf = pgid_file(git_ops.worktree_base, request_id)
        pgid = start_own_process_group()
        write_pgid_file(pgf, pgid)

    async def _run():
        # git_ops closed over from outer scope — same instance, no redundant construction.
        wt = await git_ops.acquire_host_verify_worktree(sha)
        try:
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
            if config.git.persistent_merge_worktree:
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
                return await run_merge_verify_on_worktree(wt, config, spec, merge_sha=sha)
            finally:
                if build_lane_fd is not None:
                    remove_lock_holder_pgid(git_ops.worktree_base)
                    release_merge_verify_flock(build_lane_fd)
        finally:
            await git_ops.cleanup_merge_worktree(wt)

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
    # its own tree mutation, the CLI CANNOT hold it continuously across _run() (an
    # in-process second flock on the same inode self-conflicts — esc-2830-1). The
    # lane lock therefore has a SPLIT lifetime, mirroring the LOCAL flow's sequential
    # reset-hold-then-lease-hold (merge_queue.py):
    #   (a) here, PRE-FLIGHT — a bounded-wait acquire used purely as a contention
    #       GATE (a live lane actor -> distinguished result, NO tree touch, PRD
    #       Invariant 5), then RELEASED before _run() so reset can take it;
    #   (b) inside _run(), re-acquired around the BUILD only (see _run above).
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
    # by _run()'s build-scoped hold, NOT here: an in-flight OLD caller wrote that
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
    if config.git.persistent_merge_worktree:
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
            # not a correctness gap. The shared holder is written by _run()'s
            # build-scoped hold, NOT at this gate (see the DUAL-LOCK note above). So
            # when the current holder is another NEW-code verify-merge that has
            # passed its own gate, released the lane lock, and is mid-reset inside
            # acquire_host_verify_worktree — reset_persistent_merge_worktree takes
            # the lane lock but never writes the holder-pgid (git_ops.py) — this read
            # returns None for that window. Mutual exclusion and the gate contract
            # are intact; make_flock_contention_result tolerates holder_pgid=None
            # (only the diagnostic's holder_pgid field is null). A follow-up could
            # restore always-present holder visibility by also writing the shared
            # holder at this gate, at the cost of extra cleanup on _run()'s fail-open
            # (build_lane_fd is None) path.
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
                # same inode would self-conflict (esc-2830-1). _run() re-takes the
                # lane lock and writes the shared holder-pgid for the build's
                # duration only. The compat co-lock (compat_fd) stays held across
                # the WHOLE span — a DIFFERENT inode, so no self-conflict — closing
                # the momentary lane-release gaps against a concurrent waiter and
                # preserving rollout back-compat exclusion.
                release_merge_verify_flock(lane_fd)
                lane_fd = None

    if contention_result is not None:
        # Always remove the pgid file so cancel-verify knows this run is done.
        if pgf is not None:
            remove_pgid_file(pgf)
        # Contention must exit 0: a passed=False VerifyResult on stdout is the
        # only delivery channel to beta — a non-zero exit makes RemoteRunner
        # treat this as RunnerUnavailable and beta never sees the discriminant.
        click.echo(result_to_json(contention_result))
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
    # through _run(), and the main thread can reach the click.echo/return
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

        def _on_watchdog_fire() -> None:
            watchdog_fired.set()
            fire_watchdog_kill(pgid, grace_secs=grace_secs)

        watchdog_thread = start_stdin_watchdog(
            pgid, heartbeat_timeout=heartbeat_timeout, fire=_on_watchdog_fire
        )

    try:
        result = asyncio.run(_run())
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    finally:
        # Past the contention gate the lane lock is already released (the gate
        # holds it only pre-flight; _run()'s build-scoped hold manages the lane
        # lock and the shared holder-pgid — task 2830 split lifetime, esc-2830-1).
        # Only the compat co-lock is held across the whole span here; release it.
        if compat_fd is not None:
            release_merge_verify_flock(compat_fd)
        # Always remove the pgid file so cancel-verify knows this run is done.
        if pgf is not None:
            remove_pgid_file(pgf)

    if watchdog_fired is not None and watchdog_fired.is_set():
        # The watchdog already tree-killed the build and is mid-way through
        # its own SIGTERM -> grace_secs sleep -> SIGKILL escalation (see
        # fire_watchdog_kill) -- never print a (misleading) VerifyResult for
        # a build we just killed out from under ourselves. Block on the
        # watchdog thread itself here instead of exiting immediately: a bare
        # sys.exit(1) begins interpreter shutdown right away, and since the
        # watchdog thread is a daemon thread, shutdown does not wait for it --
        # it can be torn down mid-sleep, before the SIGKILL escalation that
        # exists specifically to reap start_new_session grandchildren
        # (cargo/rustc) that survived SIGTERM. Joining lets
        # fire_watchdog_kill's own unconditional os._exit(1) be the
        # authoritative exit once the full escalation has run to completion.
        if watchdog_thread is not None:
            watchdog_thread.join()
        sys.exit(1)  # pragma: no cover - fire_watchdog_kill os._exit()s first

    click.echo(result_to_json(result))


@main.command('cancel-verify')
@click.option('--request-id', required=True,
              help='Request ID passed to the verify-merge invocation to cancel.')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — determines worktree_base for '
                   'the pgid-file lookup.')
def cancel_verify(request_id: str, config_path: Path | None):
    """Kill the verify-merge process tree identified by --request-id and exit 0.

    Reads the pgid file at
    ``<worktree_base>/.merge_verify_pgids/<request-id>`` (written by
    ``verify-merge --request-id``), snapshots the ``/proc`` PPID map, and
    ``SIGKILL``\\s every descendant plus a ``killpg`` backstop.  This reaps the
    entire process tree including ``start_new_session`` build-command escapes
    (which leave ``verify-merge``'s process group but remain its descendants in
    the ``/proc`` parent chain).

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

    try:
        config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    git_ops = GitOps(config.git, config.project_root)
    pgf = pgid_file(git_ops.worktree_base, request_id)
    failed_pids: list[int] = []
    rc = cancel_request(pgf, failed_pids_out=failed_pids)
    for pid in failed_pids:
        click.echo(
            f'cancel-verify: PermissionError: could not SIGKILL pid {pid} '
            f'(process alive but kill refused) — retry or escalate manually',
            err=True,
        )
    sys.exit(rc)


@main.command('eval')
@click.option('--task', 'task_path', type=click.Path(exists=True, path_type=Path),
              default=None, help='Path to a single task JSON file')
@click.option('--config-name', default=None,
              help='Eval config name (e.g. claude-opus-high) or "all"')
@click.option('--matrix', is_flag=True, help='Run full eval matrix (all tasks × all configs)')
@click.option('--judge', is_flag=True, help='Run Elo-based LLM judge on existing results')
@click.option('--plan-only', is_flag=True, help='Generate plans for tasks (no execution)')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless ORCH_CONFIG_PATH '
                   'is set). Selects the target project — sets project_root and '
                   'fused_memory.project_id.')
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
    config_path: Path | None,
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
    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    # Inject ANTHROPIC_BASE_URL into vLLM configs when --vllm-url is set
    if vllm_url:
        from orchestrator.evals.configs import FINAL_RUN_CONFIGS, VLLM_EVAL_CONFIGS
        for cfg in [*VLLM_EVAL_CONFIGS, *FINAL_RUN_CONFIGS]:
            if cfg.env_overrides:  # only vLLM configs have env_overrides
                cfg.env_overrides['ANTHROPIC_BASE_URL'] = vllm_url

    if cleanup:
        _run_cleanup(base_config)
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
        _run_plan_only(task_path, base_config)
        return

    if matrix:
        _run_matrix_cmd(
            base_config,
            max_parallel=max_parallel, trials=trials,
            force=force, timeout=timeout,
        )
        return

    if task_path is None:
        click.echo('Error: --task is required (or use --matrix / --judge / --plan-only)', err=True)
        sys.exit(1)

    _run_single_eval(
        task_path, config_name, base_config,
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


@main.command('eval-list-fixtures')
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
    tasks_dir = tasks_dir or (Path(__file__).parent / 'evals' / 'tasks')
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


@main.command('eval-sample')
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
@click.option('--df-root', type=click.Path(path_type=Path),
              default=Path('/home/leo/src/dark-factory'),
              help='dark_factory checkout to discover merges in')
@click.option('--reify-root', type=click.Path(path_type=Path),
              default=Path('/home/leo/src/reify'),
              help='reify checkout to discover merges in')
@click.option('--sampled-at', default=None,
              help='ISO-8601 provenance timestamp (default: now, UTC)')
def eval_sample_cmd(since: str, target_low: int, target_high: int, seed: int,
                    cohort: str, dry_run: bool, tasks_dir: Path | None,
                    df_root: Path, reify_root: Path, sampled_at: str | None):
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
        df_root=Path(df_root), reify_root=Path(reify_root), sampled_at=sampled_at,
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

    tasks_dir = tasks_dir or (Path(__file__).parent / 'evals' / 'tasks')

    candidates = []
    for project, root in (('dark_factory', df_root), ('reify', reify_root)):
        if not root.exists():
            click.echo(f'WARNING: {project} checkout {root} not found; skipping',
                       err=True)
            continue
        found = await discover_completed_tasks(
            root, since=since, project=project, project_root=str(root),
        )
        found = enrich_candidates_from_task_db(
            found, root / '.taskmaster' / 'tasks' / 'tasks.db',
        )
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

    A candidate whose ``role == 'architect'`` (eval-revival θ) is dispatched to
    ``run_architect_eval`` — the plan-only architect eval that scores the
    produced plan against the real landed diff and FREEZES every downstream
    role — and its per-fixture ``plan_quality`` score is surfaced (plus a
    plan-quality table across all architect runs). Ordinary implementer configs
    still route to ``run_eval``, unchanged.
    """
    from orchestrator.evals.configs import EVAL_CONFIGS, get_config_by_name
    from orchestrator.evals.report import (
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
        architect_results = []
        for cfg in configs:
            if cfg.role == 'architect':
                # θ: plan-only architect eval — downstream roles frozen.
                # run_architect_eval manages its own eval worktree at the
                # fixture's pre_task_commit, so it takes no worktree_path.
                result = await run_architect_eval(
                    task_path, cfg, base_config, timeout_override=timeout,
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
                    f' unmeasurable: {result.metrics.get("invocation_error")}'
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
                    worktree_path=worktree_path,
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
    tasks_dir = tasks_dir or (Path(__file__).parent / 'evals' / 'tasks')
    if not tasks_dir.exists():
        tasks_dir = Path(base_config.project_root) / 'orchestrator' / 'evals' / 'tasks'
    # Loud validation (malformed fixture → named ClickException) + emptiness guard.
    if not _load_fixture_dir(tasks_dir):
        click.echo(f'No eval fixtures found in {tasks_dir}', err=True)
        sys.exit(1)
    return sorted(tasks_dir.glob('*.json'))


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
    from orchestrator.evals.report import (
        build_composite_report,
        build_plan_quality_report,
        format_composite_table,
        format_plan_quality_table,
    )

    report = build_composite_report(results, price_table=price_table)
    click.echo(format_composite_table(report))

    if any(r.metrics.get('role_under_test') == 'architect' for r in results):
        click.echo('')
        click.echo(format_plan_quality_table(build_plan_quality_report(results)))


def _run_ofat_driver(
    base_config, *, tasks_dir: Path | None,
    max_parallel: int | None, trials: int, timeout: int | None,
) -> None:
    """OFAT screen: ``run_ofat_stage`` over ``ofat_candidates()`` → composite (μ)."""
    from orchestrator.evals.configs import ofat_candidates
    from orchestrator.evals.report import build_price_table
    from orchestrator.evals.runner import run_ofat_stage

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
    candidates = ofat_candidates()
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
    from orchestrator.evals.report import build_pairwise_price_table
    from orchestrator.evals.runner import run_matrix_stage

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
    candidates = ofat_candidates()
    arch_cfgs = [c for c in candidates if c.role == 'architect']
    impl_cfgs = [c for c in candidates if c.role == 'implementer']
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
    from orchestrator.evals.report import build_pairwise_price_table
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
    if arch_cfg.role != 'architect':
        click.echo(
            f'--arch: {arch!r} is a {arch_cfg.role} config, expected an architect',
            err=True,
        )
        sys.exit(1)
    if impl_cfg.role != 'implementer':
        click.echo(
            f'--impl: {impl!r} is a {impl_cfg.role} config, expected an implementer',
            err=True,
        )
        sys.exit(1)

    task_paths = _resolve_eval_task_paths(base_config, tasks_dir)
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


@main.command('eval-ofat')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless '
                   'ORCH_CONFIG_PATH is set).')
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir (default: orchestrator/evals/tasks)')
@click.option('--trials', type=int, default=1,
              help='Trials per (fixture, candidate) cell (default: 1)')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
def eval_ofat_cmd(config_path, tasks_dir, trials, max_parallel, timeout):
    """OFAT screen (μ): vary ONE role per candidate, pin the rest to incumbents.

    Each candidate varies exactly one role — implementer incumbents (Opus AND
    Sonnet, the G2 >=2-config floor) drive ``run_eval`` with the plan frozen;
    architect candidates drive ``run_architect_eval`` live with downstream roles
    frozen. Emits the C4 composite report.
    """
    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    _run_ofat_driver(
        base_config, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


@main.command('eval-matrix')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless '
                   'ORCH_CONFIG_PATH is set).')
@click.option('--tasks-dir', type=click.Path(path_type=Path), default=None,
              help='Fixture dir (default: orchestrator/evals/tasks)')
@click.option('--trials', type=int, default=1,
              help='Trials per (fixture, pair) cell (default: 1)')
@click.option('--max-parallel', type=int, default=None,
              help='Max concurrent eval runs (default: unlimited)')
@click.option('--timeout', type=int, default=None,
              help='Timeout in minutes per eval run (overrides task JSON)')
def eval_matrix_cmd(config_path, tasks_dir, trials, max_parallel, timeout):
    """Matrix stage (μ): both-live architect×implementer cross product.

    Runs the FULL architect×implementer cross product over the OFAT survivors,
    INCLUDING same-family diagonals (the plan-style/implementer coupling
    hypothesis, PRD decision 9). Emits the C4 composite report.
    """
    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    _run_matrix_driver(
        base_config, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


@main.command('eval-confirm')
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to orchestrator config YAML (REQUIRED unless '
                   'ORCH_CONFIG_PATH is set).')
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
def eval_confirm_cmd(config_path, arch, impl, tasks_dir, trials, max_parallel, timeout):
    """Confirmation batch (μ): one end-to-end batch of the winning combo.

    Runs the single winning (architect, implementer) combo across all fixtures ×
    N trials (default 3 — decision 10's statistics floor, enough repeats for a
    CI95 on the winner). Emits the C4 composite report.
    """
    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    _run_confirm_driver(
        base_config, arch=arch, impl=impl, tasks_dir=tasks_dir,
        max_parallel=max_parallel, trials=trials, timeout=timeout,
    )


def _run_judge_cmd(max_rounds: int = 50, reset: bool = False):
    """Run Elo-based judge on existing results."""
    from orchestrator.evals.elo import JudgeState, TaskPool, load_state, save_state
    from orchestrator.evals.judge import run_elo_tournament
    from orchestrator.evals.report import build_report, format_markdown, save_report
    from orchestrator.evals.runner import load_results, load_task
    from orchestrator.evals.snapshots import get_diff_between_commits

    # Load or reset state
    if reset:
        state = JudgeState()
        click.echo('Judge state reset.')
    else:
        state = load_state()
        if state.per_task:
            click.echo(f'Resuming from existing state ({len(state.per_task)} tasks)')

    results = load_results()
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

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'

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
            rounds_used = await run_elo_tournament(
                result_dicts, task, pool, max_rounds,
            )
            click.echo(f'  {task_id}: {rounds_used} judge calls')

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
    from orchestrator.evals.report import build_report, format_markdown, save_report

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
    from orchestrator.evals.runner import load_results, load_task

    results = load_results()
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
    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'
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


def _run_plan_only(task_path: Path | None, base_config):
    """Generate plans for eval tasks using the architect (opus-high).

    Runs the architect against the pre-task commit for each task,
    saves the resulting plan into the task JSON file.
    """
    from orchestrator.evals.runner import load_task
    from orchestrator.evals.snapshots import cleanup_eval_worktree, create_eval_worktree

    tasks_dir = Path(__file__).parent / 'evals' / 'tasks'

    task_paths = [task_path] if task_path else sorted(tasks_dir.glob('*.json'))

    if not task_paths:
        click.echo('No task files found', err=True)
        sys.exit(1)

    async def _run():
        from orchestrator.agents.briefing import BriefingAssembler
        from orchestrator.agents.invoke import invoke_agent
        from orchestrator.agents.roles import ARCHITECT
        from orchestrator.artifacts import TaskArtifacts

        briefing = BriefingAssembler(base_config)

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
                prompt = await briefing.build_architect_prompt(task_def, worktree=worktree)

                # Invoke architect (opus-high, always Claude)
                result = await invoke_agent(
                    prompt=prompt,
                    system_prompt=ARCHITECT.system_prompt,
                    cwd=worktree,
                    model='opus',
                    max_turns=50,
                    max_budget_usd=5.0,
                    allowed_tools=ARCHITECT.allowed_tools or None,
                    disallowed_tools=ARCHITECT.disallowed_tools or None,
                    effort='high',
                    backend='claude',
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
