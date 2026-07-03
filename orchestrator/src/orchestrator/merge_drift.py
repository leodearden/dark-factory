"""Lever-C drift-check detective (MQ-refactor task γ).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the drift-check
runner (dispatches a local + remote pair through
:class:`~orchestrator.verify_runner.DriftDetector` in a throwaway worktree)
and its land-hook cadence gate.  ``merge_queue`` re-exports both names
through a top-level shim so existing importers
(``from orchestrator.merge_queue import X``, etc.) keep working unchanged.

A moved function that calls a merge_queue-resident sibling — whether that
sibling stays permanently (``_run_unscoped_typechecks``, the module-level
``_build_remote_runners`` legacy-pool builder) or is monkeypatched by the
existing test suite via the string path ``orchestrator.merge_queue.<name>``
(``run_scoped_verification``, ``build_merge_verify_spec``, ``LocalRunner``,
``VerifyRunnerPool``, ``_derive_task_files_from_git``, ``_run_drift_check``)
— resolves it through a function-local (deferred) import from
:mod:`orchestrator.merge_queue` rather than a direct intra-module reference.
This mirrors the ``_main_health_fingerprint`` convention in
``merge_queue.py`` and keeps this module free of any top-level import of
``merge_queue`` (which would deadlock module load, since merge_queue's shim
needs this module fully defined first).

Note: despite both being off-serial-lane detective controls spawned from the
same 'done'-land hook, :func:`_run_drift_check` does NOT call
``_run_cold_shadow_verify`` or ``_run_shadow_compare`` (in
:mod:`orchestrator.merge_shadow`) — drift-check and shadow-compare are
independent sibling detectives, not caller/callee.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_types import MergeRequest

# run_scoped_verification is not referenced directly in this module —
# _run_drift_check always reaches back to the orchestrator.merge_queue-
# resident binding (see its body).  Imported here only so TestReachBackRouting
# has a "naive" orchestrator.merge_drift.run_scoped_verification patch target
# to assert is NOT what governs.  _derive_task_files_from_git is deliberately
# NOT imported here: unlike run_scoped_verification, nothing needs to prove a
# merge_drift-local patch is inert for it, and _run_drift_check reaches back
# to orchestrator.merge_queue for it exclusively (see its body).
from orchestrator.verify import run_scoped_verification  # noqa: F401

# LocalRunner / VerifyRunnerPool / build_merge_verify_spec: same reasoning —
# _run_drift_check always reaches back to orchestrator.merge_queue for these;
# DriftDetector / HostAllocator are used directly (not reached back).
from orchestrator.verify_runner import (
    DriftDetector,
    HostAllocator,
    LocalRunner,  # noqa: F401
    VerifyRunnerPool,  # noqa: F401
    build_merge_verify_spec,  # noqa: F401
)

if TYPE_CHECKING:
    from orchestrator.merge_queue import SpeculativeMergeWorker

logger = logging.getLogger('orchestrator.merge_queue')


async def _run_drift_check(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    escalation_queue: Any,
    event_store: EventStore | None,
    quarantine_set: set[str],
    *,
    allocator: HostAllocator | None = None,
) -> None:
    """Drift detective control: run DriftDetector.check in a throwaway worktree.

    Mirrors ``_run_cold_shadow_verify`` / ``_run_shadow_compare`` as an
    off-serial-lane detective control (spawned via :func:`_maybe_run_drift_check`).
    Creates a throwaway verify worktree, builds a 2-host pool (local trust-anchor
    + eligible remote), runs :meth:`~orchestrator.verify_runner.DriftDetector.check`,
    and propagates any quarantined remote name into *quarantine_set* (the worker-
    level shared set consulted by ``_run_post_merge_verify``).

    Exceptions are caught and logged so this detective control never crashes the
    worker.  ``cleanup_merge_worktree`` and allocator slot releases are always
    called in the ``finally`` block.

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed (config, task_files, …).
        merge_commit: The just-landed merge commit SHA.
        escalation_queue: Live escalation queue (None-safe: passed to DriftDetector).
        event_store: Optional event store (None-safe: passed to DriftDetector).
        quarantine_set: Worker-level mutable set; quarantined remote names are
            added here so subsequent dispatches skip the diverged remote.
        allocator: Worker-lifetime :class:`~orchestrator.verify_runner.HostAllocator`
            (β decision 5).  When provided, both host slots are acquired via the
            allocator and released in the finally block.  Fallback to the legacy
            ``_build_remote_runners`` pool when ``None`` (backward-compatible).
    """
    # Reach-back (deferred import): the existing test suite patches these
    # dependencies by string path at orchestrator.merge_queue.<name> (this
    # function used to live in merge_queue.py).  Resolving them dynamically
    # from merge_queue's namespace at call time keeps those patches
    # effective post-extraction — see the module docstring's reach-back
    # convention.  Attribute access (rather than `from ... import <name>`)
    # is used here because build_merge_verify_spec / VerifyRunnerPool /
    # LocalRunner / run_scoped_verification also have a module-level "naive"
    # import above (kept solely as a TestReachBackRouting patch target); a
    # `from ... import` reach-back would shadow-and-thus-dead-code that
    # naive import, which ruff flags (F811).  _derive_task_files_from_git has
    # no merge_drift-local "naive" import at all (there is nothing to prove
    # inert) but is reached back to for the same reason as the others:
    # merge_queue.py imports it at module level from orchestrator.verify and
    # the test suite patches it on that namespace.  _run_unscoped_typechecks
    # and _build_remote_runners have no merge_drift-local copy at all (both
    # stay permanently in merge_queue.py) but are accessed the same way for
    # consistency.
    import orchestrator.merge_queue as _mq

    # wt is initialised before the try so the finally guard (`if wt is not None`)
    # is safe even when create_throwaway_verify_worktree itself raises.  Moving the
    # creation inside the try ensures the docstring contract ("Exceptions are caught
    # and logged") holds for all failure modes — disk-full / git errors included.
    wt = None
    local_lease = None
    remote_lease = None
    try:
        wt = await git_ops.create_throwaway_verify_worktree(merge_commit)
        task_files_tuple = tuple(req.task_files) if req.task_files is not None else None
        # Derive task_files on the dispatching host (fresh main) when not supplied
        # and Lever C is on — mirrors the same gate in _run_post_merge_verify.
        if task_files_tuple is None and req.config.enabled_verify_runners:
            derived = await _mq._derive_task_files_from_git(wt, req.config)
            if derived:
                task_files_tuple = tuple(derived)
        spec = _mq.build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)

        if allocator is not None:
            # β decision 5: acquire both hosts through the allocator so slot
            # accounting is respected (≤1 verify-merge per host at any time).
            # The throwaway worktree (wt) was created unconditionally above; only
            # the LocalRunner *object* is deferred to the factory so it is not
            # constructed if the local slot is unavailable.
            _ttf = task_files_tuple  # capture for closure

            def _local_factory() -> _mq.LocalRunner:
                return _mq.LocalRunner(
                    wt, req.config, req.module_configs, _ttf,
                    run_scoped=_mq.run_scoped_verification,
                    run_unscoped=_mq._run_unscoped_typechecks,
                    task_id=req.task_id,
                )

            local_lease = allocator.acquire_local(_local_factory)
            remote_lease = allocator.acquire_remote()
            if local_lease is None or remote_lease is None:
                logger.debug(
                    'Drift check skipped for task %s: host slots unavailable '
                    '(local_free=%s, remote_free=%s)',
                    req.task_id,
                    local_lease is not None,
                    remote_lease is not None,
                )
                return
            pool = _mq.VerifyRunnerPool(
                [local_lease.runner, remote_lease.runner],
                event_store=event_store,
                task_id=req.task_id,
            )
        else:
            # Legacy fallback: build fresh pool via _build_remote_runners (no
            # slot accounting — used when allocator is not threaded in yet).
            pool = _mq.VerifyRunnerPool(
                [_mq.LocalRunner(
                    wt, req.config, req.module_configs, task_files_tuple,
                    run_scoped=_mq.run_scoped_verification,
                    run_unscoped=_mq._run_unscoped_typechecks,
                    task_id=req.task_id,
                ), *_mq._build_remote_runners(req.config, wt, quarantine=quarantine_set)],
                event_store=event_store,
                task_id=req.task_id,
            )

        # Cadence is already enforced upstream in _maybe_run_drift_check
        # (_drift_land_count % every_n gate).  DriftDetector.should_sample is
        # never called on this path, so omitting every_n_lands here keeps a
        # single source of truth and avoids a dead duplicate cadence value.
        detector = DriftDetector(
            pool,
            event_store=event_store,
            escalation_queue=escalation_queue,
            task_id=req.task_id,
        )
        # Capture the eligible remote BEFORE check() so we can propagate its
        # name even after the pool quarantines it (eligible_remote() returns
        # None after quarantine).
        eligible_before = pool.eligible_remote()
        result = await detector.check(merge_commit, spec)
        if result.quarantined and eligible_before is not None:
            quarantine_set.add(eligible_before.name)
    except Exception:
        logger.warning(
            'Drift check failed for task %s / commit %s; ignoring (detective control)',
            req.task_id, merge_commit,
            exc_info=True,
        )
    finally:
        if allocator is not None:
            if local_lease is not None:
                await allocator.release(local_lease)
            if remote_lease is not None:
                await allocator.release(remote_lease)
        if wt is not None:
            await git_ops.cleanup_merge_worktree(wt)


async def _maybe_run_drift_check(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
) -> None:
    """Cadence gate + off-serial-lane spawn for the Lever C drift detective.

    Called from :meth:`SpeculativeMergeWorker._verify_and_advance` on every
    successful ('done') land, immediately after ``_maybe_schedule_shadow_compare``.
    Returns **immediately** — spawns a background task when cadence is met,
    otherwise is a no-op.

    Cadence: every ``config.verify_drift_check_every_n_lands`` successful lands.
    No-op when ``config.enabled_verify_runners`` is empty (Lever C off).

    State is tracked in worker-level attrs ``_drift_land_count`` (incremented
    here) and ``_runner_quarantine`` (propagated from :func:`_run_drift_check`).

    Args:
        worker: Live :class:`SpeculativeMergeWorker` instance.
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed.
        merge_commit: The just-landed merge commit SHA.
    """
    # Reach-back (deferred import): the existing test suite patches the
    # spawned coroutine by string path at
    # orchestrator.merge_queue._run_drift_check (this function used to live
    # in merge_queue.py, alongside its sibling).  Resolving it dynamically
    # from merge_queue's namespace at call time keeps those patches
    # effective post-extraction.
    from orchestrator.merge_queue import _run_drift_check

    if not req.config.enabled_verify_runners:
        return
    every_n = req.config.verify_drift_check_every_n_lands
    worker._drift_land_count += 1
    if worker._drift_land_count % every_n == 0:
        _coro = _run_drift_check(
            git_ops, req, merge_commit,
            worker._escalation_queue, worker._event_store,
            worker._runner_quarantine,
            allocator=worker._ensure_host_allocator(req.config),
        )
        _task = asyncio.create_task(_coro)
        if not isinstance(_task, asyncio.Task):
            # create_task was intercepted (e.g. by a test mock) and did not
            # schedule _coro.  Close it here to prevent
            # "coroutine was never awaited" RuntimeWarning from leaking into
            # unrelated tests (pyproject.toml converts those warnings to errors).
            _coro.close()
        else:
            # Hold a strong reference so the event loop cannot GC the drift
            # detective mid-run (asyncio keeps only a weak ref to Tasks).
            # Mirrors the _shadow_compare_tasks pattern at :func:`_maybe_schedule_shadow_compare`.
            worker._drift_check_tasks.add(_task)
            _task.add_done_callback(
                lambda t: worker._drift_check_tasks.discard(t)
            )
