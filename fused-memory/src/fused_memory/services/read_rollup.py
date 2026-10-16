"""The hourly read-count rollup that replaces per-read ``write_ops`` rows (task 5405).

``WriteJournal.log_write_op`` is the live producer: it counts each successful
read that has no per-row consumer into ``read_op_counts`` instead of storing
it. ``fused-memory/scripts/roll_up_journal_reads.py`` moves the history that
predates it. This module holds only SQL constants and pure functions.
"""

from __future__ import annotations

READ_OP_COUNTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS read_op_counts (
    -- hour_start: UTC hour start as 'YYYY-MM-DDTHH:00:00+00:00', text-comparable with created_at bounds.
    -- '' in project_id / agent_id / operation means the write_ops column was NULL.
    hour_start TEXT NOT NULL,
    project_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (hour_start, project_id, agent_id, operation)
) WITHOUT ROWID;
"""


def _rollup_key(at: str, project_id: str, agent_id: str, operation: str) -> str:
    return (
        f"strftime('%Y-%m-%dT%H:00:00+00:00', {at}), "
        f"IFNULL({project_id}, ''), IFNULL({agent_id}, ''), IFNULL({operation}, '')"
    )


COUNT_READ_SQL = (
    'INSERT INTO read_op_counts (hour_start, project_id, agent_id, operation, count) '
    f"VALUES ({_rollup_key('?', '?', '?', '?')}, ?) "
    'ON CONFLICT (hour_start, project_id, agent_id, operation) '
    'DO UPDATE SET count = count + excluded.count'
)

# Consumers of per-row search detail: leaf eta/theta (WriteJournalConfig.search_retention_days),
# scripts/harvest_production_queries.py, and WriteJournal.get_ops_by_causation.
READ_OPERATIONS_KEPT_PER_ROW = frozenset({'search'})


def keeps_read_row(*, operation: str | None, success: bool) -> bool:
    return operation in READ_OPERATIONS_KEPT_PER_ROW or not success


_KEPT_OPERATIONS_SQL = ', '.join(f"'{op}'" for op in sorted(READ_OPERATIONS_KEPT_PER_ROW))

COUNT_ONLY_READ_PREDICATE = (
    "kind = 'read' AND success = 1 "
    f'AND (operation IS NULL OR operation NOT IN ({_KEPT_OPERATIONS_SQL}))'
)

# History migration (fused-memory/scripts/roll_up_journal_reads.py). Parameters
# for both: a rowid window (lo, hi), half-open. Rowids are a sequential table
# b-tree range in insertion (= time) order, so no index is built on write_ops.
_COLUMN_KEY = _rollup_key('created_at', 'project_id', 'agent_id', 'operation')

RAW_READS_BY_ROLLUP_KEY_SQL = (
    f'SELECT {_COLUMN_KEY}, COUNT(*) FROM write_ops '
    f'WHERE rowid >= ? AND rowid < ? AND {COUNT_ONLY_READ_PREDICATE} '
    'GROUP BY 1, 2, 3, 4'
)

DELETE_COUNT_ONLY_READS_SQL = (
    f'DELETE FROM write_ops WHERE rowid >= ? AND rowid < ? AND {COUNT_ONLY_READ_PREDICATE}'
)
