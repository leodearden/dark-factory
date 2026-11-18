"""Records a landed-outbox row the way a merge worker journals a landing.

Imported by bare module name, like ``_startup_harness.py``, so a test that reads
harness privates can seed a row without itself importing a merge-lane module.
"""

from __future__ import annotations

import time

from orchestrator.merge_lane.landed_outbox import LandedOutbox, LandedRow


def record_landing(outbox: LandedOutbox, task_id: str, *, advanced_sha: str) -> None:
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha=advanced_sha, advanced_sha=advanced_sha,
        landed_at=time.time(),
    ))
