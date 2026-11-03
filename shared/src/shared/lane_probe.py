"""The exit status an offline-lane outage probe and the lane agree on (plans/offline-lane-red-path-prd.md D7).

The lane (``orchestrator/src/orchestrator/offline_lane_runs.py::run_probe``) and a probe
(``fused-memory/scripts/lane_dependency_probe.py``) live in packages that cannot import each other.
"""

from typing import Final

OUTAGE_EXIT_STATUS: Final = 75
"""A probe exits this, with a first stdout line naming the dependency, when that dependency is unusable.

75 is sysexits' EX_TEMPFAIL, the codebase's backpressure code
(``orchestrator/scripts/warm-lane/warm-lane-disk-guard.sh``).
"""
