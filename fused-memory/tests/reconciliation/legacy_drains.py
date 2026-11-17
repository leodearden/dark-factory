"""Seed drained ``event_buffer`` rows that carry no run attribution.

``EventBuffer`` has no drain path that leaves a row without
``drained_by_run_id`` (task 6784), but ``restore_drained(include_unattributed=True)``
exists for rows a pre-task-2711 process left behind. Tests of that upgrade
boundary seed such rows here, through a second connection to the buffer's
database file, exactly as that older process wrote them. Imported as
``from reconciliation.legacy_drains import drain_unattributed``.
"""

import sqlite3
from contextlib import closing
from pathlib import Path


def drain_unattributed(db_path: Path, project_id: str) -> int:
    """Mark every buffered row of *project_id* drained with no run or instance stamp; return how many."""
    with closing(sqlite3.connect(db_path)) as conn, conn:
        return conn.execute(
            "UPDATE event_buffer SET status = 'drained', "
            "drained_by_run_id = NULL, drained_by_instance_id = NULL "
            "WHERE project_id = ? AND status = 'buffered'",
            (project_id,),
        ).rowcount
