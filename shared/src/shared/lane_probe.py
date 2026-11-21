"""What the offline lane and the code it runs agree on (plans/offline-lane-red-path-prd.md D7).

The lane (``orchestrator/src/orchestrator/offline_lane_runs.py``) and the code it runs, a probe
(``fused-memory/scripts/lane_dependency_probe.py``) or a test session (``fused-memory/tests/_fm_falkor_live.py``),
live in packages that cannot import each other.
"""

from collections.abc import Mapping
from typing import Final

OUTAGE_EXIT_STATUS: Final = 75
"""A probe exits this, with a first stdout line naming the dependency, when that dependency is unusable.

75 is sysexits' EX_TEMPFAIL, the codebase's backpressure code
(``orchestrator/scripts/warm-lane/warm-lane-disk-guard.sh``).
"""

LANE_ROLE_VARIABLE: Final = 'DF_VERIFY_ROLE'
OFFLINE_LANE_ROLE: Final = 'offline'
"""The value of ``LANE_ROLE_VARIABLE`` in every run the offline lane launches: detect, confirm and probe."""


def runs_under_offline_lane(environ: Mapping[str, str]) -> bool:
    return environ.get(LANE_ROLE_VARIABLE) == OFFLINE_LANE_ROLE
