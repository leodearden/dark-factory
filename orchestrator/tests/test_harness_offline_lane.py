"""Tests for Harness-side offline deep-test lane wiring (task 1953, β2).

Covers ``Harness._start_offline_lane`` / ``Harness._stop_offline_lane``: the
enable-gated construction of an ``OfflineLaneWorker``, its registration into
the ``_offline_lane_notifiee`` slot (task 1951, β1), and the background
``_offline_lane_task`` lifecycle.

See also ``test_offline_lane.py`` for the worker's own control-flow
contract, and ``test_harness_offline_lane_trigger.py`` (β1) for the
``_note_merge_all`` fan-out this wiring plugs into.
"""

from __future__ import annotations
