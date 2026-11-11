"""Reads the CostStore ``invocations`` ledger that the curator gate's borrowers write to.

A plain module rather than conftest.py, for the reason ``_fm_helpers.py`` gives.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite


async def invocation_rows(db_path: Path) -> list[dict]:
    """Every row of the ``invocations`` table in the ledger at *db_path*, as dicts."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('SELECT * FROM invocations') as cursor:
            return [dict(row) for row in await cursor.fetchall()]
