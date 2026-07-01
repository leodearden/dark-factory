"""Tests for the offline deep-test lane singleton worker (task 1953, β2).

Covers ``orchestrator.offline_lane.OfflineLaneWorker`` in isolation: the
``on_post_merge`` enqueue-and-return trigger seam, the always-from-head
``_run_once`` snapshot, the coalescing ``run()`` loop (trigger-driven +
poll-backstop, fail-open), the lockfile singleton, and the default
``run-offline-deep.sh`` suite-runner seam.

See also ``test_harness_offline_lane.py`` for the harness-side
launch/stop/registration wiring, and
``test_harness_offline_lane_trigger.py`` (task 1951, β1) for the
``_offline_lane_notifiee`` fan-out this worker registers into.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import GitConfig

# ---------------------------------------------------------------------------
# GitConfig knobs (step-1/2)
# ---------------------------------------------------------------------------


def test_git_config_offline_lane_knobs():
    """GitConfig exposes the three offline-lane knobs with correct defaults.

    Step 1 (RED): the fields do not yet exist — must fail before impl.
    """
    cfg_default = GitConfig()
    assert cfg_default.offline_lane_enabled is False, (
        'offline_lane_enabled must default to False (feature off)'
    )
    assert cfg_default.offline_lane_test_threads == 6, (
        'offline_lane_test_threads must default to 6 (§11.2 small fixed N)'
    )
    assert cfg_default.offline_lane_poll_interval_secs == 120.0, (
        'offline_lane_poll_interval_secs must default to 120.0'
    )

    cfg_set = GitConfig(
        offline_lane_enabled=True,
        offline_lane_test_threads=4,
        offline_lane_poll_interval_secs=30.0,
    )
    assert cfg_set.offline_lane_enabled is True
    assert cfg_set.offline_lane_test_threads == 4
    assert cfg_set.offline_lane_poll_interval_secs == 30.0


def test_git_config_offline_lane_knobs_validation():
    """offline_lane_test_threads is ge=1; offline_lane_poll_interval_secs is gt=0."""
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_test_threads=0)
    with pytest.raises(ValidationError):
        GitConfig(offline_lane_poll_interval_secs=0.0)
