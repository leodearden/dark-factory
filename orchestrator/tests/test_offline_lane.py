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
