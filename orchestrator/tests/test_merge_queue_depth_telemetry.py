"""Tests for merge-verify depth telemetry (task 2340).

Covers:
  step-9  RED   — SpeculativeMergeWorker._verify_frontier_depth() pure unit test
  step-10 GREEN — _verify_frontier_depth() implementation
  step-11 RED   — depth/speculative plumbing: _run_post_merge_verify's
                  pool.dispatch forwarding + _run_inflight_verify caller wiring
  step-12 GREEN — thread depth/speculative through _run_post_merge_verify /
                  _run_inflight_verify / _dispatch_item
  step-13 RED   — speculative_merge event carries depth
  step-14 GREEN — classify_and_merge threads worker._verify_frontier_depth()
                  into the speculative_merge _emit_speculative call

DEPTH DEFINITION (ε=1890 verify-frontier stack height): depth 0 = a head
verify against real main; depth d = d speculated items already
frozen/verifying ahead of the item joining the frontier.  See
_verify_frontier_depth()'s docstring and test_merge_queue_frozen_prefix.py
for the underlying frozen-prefix model this helper reuses.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from orchestrator.merge_queue import SpeculativeMergeWorker


def _make_bare_worker() -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for pure unit tests.

    No event loop or real git_ops required — mirrors test_halt_owner.py's
    ``SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())``
    construction style.
    """
    return SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: _verify_frontier_depth()
# ---------------------------------------------------------------------------


class TestVerifyFrontierDepth:
    """_verify_frontier_depth() == len(_frozen_inflight_entries()) (ε=1890).

    Pure/synchronous delegation test — lightly stubs _frozen_inflight_entries()
    so this test is isolated from the frozen-prefix computation itself
    (already covered by test_merge_queue_frozen_prefix.py) and asserts only
    the depth helper's own wiring.

    RED until step-10 GREEN adds the method.
    """

    def test_empty_frontier_returns_zero(self) -> None:
        """No frozen/verifying entries ahead -> depth 0 (head verify vs real main)."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        assert worker._verify_frontier_depth() == 0

    def test_one_frozen_entry_returns_one(self) -> None:
        """One speculated item ahead -> depth 1."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [object()]
        assert worker._verify_frontier_depth() == 1

    def test_three_frozen_entries_returns_three(self) -> None:
        """Three speculated items ahead -> depth 3."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [object(), object(), object()]
        assert worker._verify_frontier_depth() == 3
