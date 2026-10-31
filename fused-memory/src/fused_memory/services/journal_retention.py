"""The startup retention sweep over every ``write_journal.db`` table; the periodic ``write_ops`` pass is ``services/write_ops_retention.py``."""

from __future__ import annotations

import time

from fused_memory.config.schema import WriteJournalConfig
from fused_memory.services.write_journal import WriteJournal
from fused_memory.services.write_ops_retention import prune_write_ops_pass

#: Tables the startup sweep leaves unpruned on purpose. The hourly read-count
#: rollup (task 5405) has no horizon decided yet.
TABLES_WITHOUT_RETENTION: frozenset[str] = frozenset({
    'read_op_counts',  # debt: task 7968
})


async def prune_every_table_at_startup(
    journal: WriteJournal, config: WriteJournalConfig
) -> dict[str, int]:
    """Prune every journal table once; returns the rows deleted, keyed by table name.

    Call it AFTER ``MemoryService.recover_mem0_intents``, so no still-pending
    intent is pruned.

    ``write_ops`` and ``backend_ops`` SHARE one row budget and one deadline, so
    one startup's batched pruning stays inside ``prune_max_seconds`` rather than
    doubling it against the watchdog's startup grace. ``backend_ops`` still gets
    at least one batch, because deadlines are checked between batches; a row
    budget already spent to zero yields 0 and that prune's WARNING.

    Every prune is fire-and-forget, so this never raises.
    """
    mem0_intents = await journal.prune_mem0_intents()
    idempotent_ops = await journal.prune_idempotent_ops()
    started = time.monotonic()
    write_ops = await prune_write_ops_pass(journal, config)
    backend_ops = await journal.prune_backend_ops(
        older_than_days=config.write_retention_days,
        batch_size=config.prune_batch_size,
        max_rows=config.prune_max_rows_per_run - write_ops,
        max_seconds=config.prune_max_seconds - (time.monotonic() - started),
    )
    referent_findings = await journal.prune_referent_findings(
        older_than_days=config.referent_findings_retention_days,
    )
    return {
        'write_ops': write_ops,
        'backend_ops': backend_ops,
        'mem0_intents': mem0_intents,
        'idempotent_ops': idempotent_ops,
        'referent_findings': referent_findings,
    }
