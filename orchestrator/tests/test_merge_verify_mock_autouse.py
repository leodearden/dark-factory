"""Pin the autouse mock for orchestrator.merge_queue.run_scoped_verification.

Task 1829: the autouse ``_mock_merge_queue_verification`` fixture in
``orchestrator/tests/conftest.py`` monkeypatches
``orchestrator.merge_queue.run_scoped_verification`` to return
``VerifyResult(passed=True)`` for every test.  This is correct for the
majority of MergeWorker tests that want a fast, deterministic verify result,
but SILENTLY defeats tests whose purpose IS to assert that a compile-broken
member is CAUGHT by the post-merge gate (outcome blocked, not done).

The fixture honours an ``exercise_merge_verify`` marker opt-out (mirroring
``exercise_dry_run_unblock`` / ``_no_dry_run_unblock``).  When a test carries
the marker, the fixture returns early and the REAL
``orchestrator.merge_queue.run_scoped_verification`` binding (real cargo) runs.

This file:
  (a) pins the default-mock behaviour so a regression shows up deterministically,
  (b) gives the marker opt-out a cargo-INDEPENDENT RED→GREEN gate that works on
      any host (no cargo required).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import orchestrator.merge_queue as mq
from orchestrator.verify import run_scoped_verification as _real_verify


def test_merge_verify_mocked_by_default():
    """Default autouse patch: merge_queue.run_scoped_verification is an AsyncMock.

    Regression guard: confirms the autouse fixture is active and the binding is
    NOT the real orchestrator.verify.run_scoped_verification.  This must remain
    GREEN before AND after the task-1829 narrowing (the mock is only skipped for
    tests that carry the exercise_merge_verify marker).
    """
    binding = mq.run_scoped_verification
    assert isinstance(binding, AsyncMock), (
        f'Expected merge_queue.run_scoped_verification to be an AsyncMock '
        f'(autouse fixture active), got: {binding!r}'
    )
    assert binding is not _real_verify, (
        'merge_queue.run_scoped_verification must NOT be the real function when '
        'the autouse mock is in place'
    )


@pytest.mark.exercise_merge_verify
def test_exercise_merge_verify_marker_disables_mock():
    """With the marker, the REAL run_scoped_verification binding is restored.

    We assert identity (``is``) rather than calling the function — actually
    invoking it would require cargo and a real workspace.

    This is RED until step-2 narrows ``_mock_merge_queue_verification`` to
    honour the ``exercise_merge_verify`` marker.
    """
    binding = mq.run_scoped_verification
    assert binding is _real_verify, (
        f'Expected merge_queue.run_scoped_verification to be the REAL function '
        f'when exercise_merge_verify marker is present, got: {binding!r}'
    )
    assert not isinstance(binding, AsyncMock), (
        'merge_queue.run_scoped_verification must NOT be an AsyncMock when '
        'exercise_merge_verify marker is present'
    )
