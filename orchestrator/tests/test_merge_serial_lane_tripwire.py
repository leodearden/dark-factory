"""Tests for the merge serial-lane C4 tripwire (task 2930, PRD η).

docs/prds/merge-worktree-lifecycle-integrity.md §4 C4 / §9 row 10: dispatching
a SECOND concurrent LOCAL merge verify while the ``_MERGE_AHEAD_BOUND``-derived
per-host in-flight bound is 1 must log a WARNING and emit a telemetry event
(``EventType.merge_serial_lane_breached``).  **No hard block** — this is a
cheap DETECTION net for a future request-identity leak of the task/5326 class
(two journal entries rehydrated for one branch, both enqueued, bypassing the
``InFlightMergeRegistry`` coalesce gate).

Each test class imports the symbols under test LOCALLY inside its test methods
(not at module scope) so a not-yet-implemented name never breaks collection of
the rest of this file during earlier RED steps — mirrors the convention
documented in test_merge_skew_tripwire.py:11-14.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig

# ---------------------------------------------------------------------------
# Config builder (per-file duplication — PRD D9: no cross-test-module imports
# of private fixtures; mirrors test_merge_queue_persistent_worktree.py:59+)
# ---------------------------------------------------------------------------


def _make_persistent_config(root: Path, *, persistent: bool) -> OrchestratorConfig:
    """Build OrchestratorConfig with the persistent_merge_worktree knob set."""
    git = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=persistent,
    )
    return OrchestratorConfig(project_root=root, git=git)


# ---------------------------------------------------------------------------
# Step 01 — per_host_inflight_bound (INV-5 extraction)
# ---------------------------------------------------------------------------


class TestPerHostInflightBound:
    """``per_host_inflight_bound(merge_ahead_bound, num_hosts)`` is THE formula.

    One definition of ``ceil(max(1, bound) / max(1, hosts))``, shared by the
    fail-closed startup guard (``enforce_persistent_worktree_serial_lane``) and
    the C4 runtime tripwire — INV-5 (no-lockstep-duplication): two sites that
    must agree byte-for-byte are one site plus a call.
    """

    @pytest.mark.parametrize(
        ('bound', 'hosts', 'expected'),
        [
            (1, 1, 1),
            (2, 1, 2),
            (2, 2, 1),
            (3, 2, 2),  # ceil semantics — an uneven split rounds UP
            (4, 4, 1),
            (4, 2, 2),
        ],
    )
    def test_ceil_semantics(self, bound: int, hosts: int, expected: int) -> None:
        """The worst-case per-host in-flight count is ceil(bound / hosts)."""
        from orchestrator.merge_liveness import per_host_inflight_bound  # noqa: PLC0415

        assert per_host_inflight_bound(bound, hosts) == expected

    @pytest.mark.parametrize(
        ('bound', 'hosts'),
        [
            (0, 1),
            (-5, 1),
            (1, 0),
            (1, -3),
            (0, 0),
        ],
    )
    def test_degenerate_inputs_clamp_to_at_least_one(self, bound: int, hosts: int) -> None:
        """Degenerate inputs fail SAFE: >= 1, never ZeroDivisionError, never 0.

        Matches the clamp behaviour the guard has always had inline.  A
        spuriously permissive 0 would make the fail-closed startup guard stop
        refusing (and the tripwire stop firing) on a nonsense config.
        """
        from orchestrator.merge_liveness import per_host_inflight_bound  # noqa: PLC0415

        assert per_host_inflight_bound(bound, hosts) == 1


class TestSerialLaneGuardUsesSharedBound:
    """INV-5 anti-drift pin: the startup guard ROUTES THROUGH the helper.

    The guard's verdict must be genuinely derived from
    ``per_host_inflight_bound``'s return value, not from a duplicate inline
    expression that could silently drift away from the tripwire's copy.
    """

    def test_guard_delegates_to_per_host_inflight_bound(self, tmp_path: Path) -> None:
        """Patching the helper to return 1 makes the guard accept bound=2/hosts=1.

        Un-refactored, ``ceil(2/1) == 2`` would raise.  A no-raise here proves
        the raise decision reads the helper's return value.
        """
        from orchestrator.merge_liveness import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_persistent_config(tmp_path, persistent=True)
        fake = MagicMock(return_value=1)
        with patch('orchestrator.merge_liveness.per_host_inflight_bound', fake):
            result = enforce_persistent_worktree_serial_lane(
                cfg, merge_ahead_bound=2, num_hosts=1
            )

        assert result is None, (
            'guard must not raise when the shared bound helper returns 1 — '
            'a raise here means the guard still recomputes the formula inline'
        )
        fake.assert_called_once_with(2, 1)
