"""Test-owned drive harness for the single-item trust-anchor verify path
(task 2180 / W1 xi).

``drive_verify_and_advance(worker, item)`` replicates — byte-for-byte in
behavior — the body of the retired ``SpeculativeMergeWorker._verify_and_advance``
compat shim (originally orchestrator/src/orchestrator/merge_queue.py:13164-13224):
force-local lease → ``_run_inflight_verify`` → ``InflightEntry(was_speculative=
False)`` → ``_finalize_inflight``, returning ``True`` iff main was advanced.

The shim existed ONLY so the ~18 tests that drove that flat single-item path
directly stayed green.  Relocating its body here lets the production shim be
deleted, removing one axis of the test↔production coupling (β decision 5,
γ design decision 5) — the drive path now lives in test code, the ~18 tests
call this helper, and merge_queue.py no longer carries a test-only method.

This module lives in orchestrator/tests/ and is imported by bare module name
(``from _merge_queue_harness import drive_verify_and_advance``), matching the
flat-test-helper convention (see ``_serial_merge_worker.py``, ``_orch_helpers.py``
— orchestrator/tests/ has no ``__init__.py``).

CRITICAL — dynamic module lookup, not from-import: every merge_queue
module-level symbol is referenced as ``_mq.<name>`` (attribute lookup on the
module object, resolved at call time), NEVER ``from orchestrator.merge_queue
import <name>``.  ~15 of the migrated tests do
``patch('orchestrator.merge_queue.run_scoped_verification', …)`` (and similar
on the other symbols); a from-import would bind the name in THIS module's
namespace at import time, so the patch on the merge_queue attribute would
silently fail to reach the helper's reference and the pass-mock would be
ignored.  Attribute lookup resolves against the live module exactly as the
in-class shim's module-global reads did.  Worker methods
(``worker._ensure_host_allocator`` / ``_run_inflight_verify`` /
``_finalize_inflight``) are already call-time-dynamic through the instance.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import orchestrator.merge_queue as _mq

if TYPE_CHECKING:
    from orchestrator.merge_queue import (
        LocalRunner,
        RealMergeItem,
        SpeculativeMergeWorker,
    )


async def drive_verify_and_advance(
    worker: SpeculativeMergeWorker, item: RealMergeItem,
) -> bool:
    """Acquire a LOCAL lease → _run_inflight_verify → _finalize_inflight.

    Returns True iff main was advanced (matches the original
    _verify_and_advance return contract).  ``was_speculative=False``: this
    flat drive path never manages the speculation semaphore — direct-call
    tests that care about speculation exercise the full loop instead.
    """
    req = item.request
    allocator = worker._ensure_host_allocator(req.config)

    _item_for_factory = item
    _req_for_factory = req

    def _local_factory() -> LocalRunner:
        assert _item_for_factory.merge_wt is not None, \
            'harness path: merge_wt must be non-None for a real item'
        return _mq.LocalRunner(
            _item_for_factory.merge_wt,
            _req_for_factory.config,
            _req_for_factory.module_configs,
            None,
            run_scoped=_mq.run_scoped_verification,
            run_unscoped=_mq._run_unscoped_typechecks,
            task_id=_req_for_factory.task_id,
        )

    lease = await allocator.acquire(_local_factory)
    if lease is None:
        # Fallback: force-acquire the local slot (harness path; no competing
        # acquirers in direct-call tests).
        lease = allocator.acquire_local(_local_factory)
    if lease is None:
        # Still None: all slots parked.  Resolve as blocked and return.
        if not req.result.done():
            req.result.set_result(_mq.MergeOutcome(
                'blocked', reason='No verify host available (harness path)',
            ))
        return False

    verify_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
        worker._run_inflight_verify(item, lease)
    )

    entry = _mq.InflightEntry(
        item=item,
        lease=lease,
        verify_task=verify_task,
        merge_wt=item.merge_wt,
        was_speculative=False,  # harness does not manage the speculation slot
        started_at=time.time(),
    )

    return await worker._finalize_inflight(entry)
