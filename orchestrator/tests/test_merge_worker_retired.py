"""Guard tests for MQ-refactor task ν: MergeWorker retirement (R7b).

The legacy serial ``MergeWorker`` is retired from the production
``orchestrator.merge_queue`` package.  Its behavior is preserved verbatim as
a test-local reference fixture (``orchestrator/tests/_serial_merge_worker.py``)
so the ~89 existing constructions across the test suite keep exercising the
same serial-specific surface (``_dequeue``/``_process``/``_do_merge``/
``_urgent`` CAS re-enqueue) without being rewritten against
``SpeculativeMergeWorker``.

Two guards bracket the mechanical relocation:

  test_serial_reference_fixture_available   — the fixture module exists and
      shape-matches the historical ``MergeWorker`` (step-1 RED / step-2 GREEN).
  test_merge_worker_absent_from_production   — the class is actually gone
      from the production module, not just duplicated (step-4 RED / step-5
      GREEN).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import orchestrator.merge_queue as mq


def test_serial_reference_fixture_available() -> None:
    """The relocated serial reference is importable and shape-compatible.

    Asserts the test-local ``_serial_merge_worker.MergeWorker``:
      - is importable by bare module name (flat orchestrator/tests/ convention)
      - still carries the shared WIP-halt contract
      - constructs with the historical signature
        ``MergeWorker(git_ops, queue, event_store=None)``
      - exposes the serial-worker surface the ported tests rely on
      - carries the historical MAX_POST_MERGE_VERIFY_* class constants
    """
    from _serial_merge_worker import MergeWorker

    git_ops = MagicMock()
    worker = MergeWorker(git_ops, asyncio.Queue(), event_store=None)

    # The halt contract, read off the PUBLIC surface the mixin exists to
    # provide rather than off the mixin class itself: which base supplies
    # ``halt_for_wip`` / ``set_halt_owner`` / ``is_halt_owner`` / ``unhalt_wip``
    # / ``is_wip_halted`` / ``halt_owner_esc_id`` is an implementation detail,
    # and the pinned fact is that a constructed serial worker still answers all
    # six.
    for member in (
        'halt_for_wip', 'set_halt_owner', 'is_halt_owner', 'unhalt_wip',
        'is_wip_halted', 'halt_owner_esc_id',
    ):
        assert hasattr(worker, member), (
            f'MergeWorker must still carry the shared WIP-halt contract; '
            f'missing {member!r}'
        )
    assert worker.is_wip_halted is False, (
        'a freshly constructed serial worker starts unhalted'
    )
    assert worker.halt_owner_esc_id is None, (
        'a freshly constructed serial worker owns no halt escalation'
    )

    for attr in ('_dequeue', '_process', '_do_merge', '_urgent', '_queue'):
        assert hasattr(worker, attr), (
            f'MergeWorker fixture is missing serial-worker surface: {attr!r}'
        )

    assert MergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
    assert MergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES == 1


def test_merge_worker_absent_from_production() -> None:
    """The serial class is gone from production — moved, not vanished.

    Structural runtime invariant (not a docstring/annotation meta-test):
    grep showing no MergeWorker class in orchestrator/src is the task's
    headline user-observable signal; this guard turns it into a durable
    regression barrier against re-introduction.
    """
    assert not hasattr(mq, 'MergeWorker'), (
        'MergeWorker must be removed from the production orchestrator.merge_queue '
        'module — the serial reference now lives only in tests/_serial_merge_worker.py'
    )

    import _serial_merge_worker

    assert _serial_merge_worker.MergeWorker is not None, (
        'the serial reference must still exist as a test-local fixture — '
        'it moved, it did not vanish'
    )

    # Smoke: harness.py must still import cleanly after the union/import
    # simplification (TYPE_CHECKING import, annotation, casts, docstring).
    import orchestrator.harness  # noqa: F401
