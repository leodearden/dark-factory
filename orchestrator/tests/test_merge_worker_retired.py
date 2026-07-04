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
      - subclasses ``orchestrator.merge_queue._WipHaltMixin``
      - constructs with the historical signature
        ``MergeWorker(git_ops, queue, event_store=None)``
      - exposes the serial-worker surface the ported tests rely on
      - carries the historical MAX_POST_MERGE_VERIFY_* class constants
    """
    from _serial_merge_worker import MergeWorker

    assert issubclass(MergeWorker, mq._WipHaltMixin), (
        'MergeWorker must still inherit the shared halt-state-machine mixin'
    )

    git_ops = MagicMock()
    worker = MergeWorker(git_ops, asyncio.Queue(), event_store=None)

    for attr in ('_dequeue', '_process', '_do_merge', '_urgent', '_queue'):
        assert hasattr(worker, attr), (
            f'MergeWorker fixture is missing serial-worker surface: {attr!r}'
        )

    assert MergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
    assert MergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES == 1
