"""The ``make_tasks_db`` fixture's DDL may only keep columns production declares identically.

To run just this suite:

    uv run --project shared pytest scripts/tests/test_make_tasks_db_schema.py -q
"""
from __future__ import annotations

import sqlite3

from shared.task_store_schema import TASK_STORE_DDL
from tasks_db_schema import columns_outside, introspect


def test_make_tasks_db_declares_only_columns_production_declares(make_tasks_db):
    fixture_conn = sqlite3.connect(make_tasks_db([]))
    production_conn = sqlite3.connect(":memory:")
    try:
        production_conn.executescript(TASK_STORE_DDL)
        assert columns_outside(introspect(fixture_conn), introspect(production_conn)) == ()
    finally:
        fixture_conn.close()
        production_conn.close()
