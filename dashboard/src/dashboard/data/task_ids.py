"""The one rule for the task a stored task id names: a string of ASCII decimal digits.

Run ledgers, escalation records and the scheduler's lock holders all carry a
task id as text. Anything else in that field, a null, a number or a label,
names no task in the store.
"""

from __future__ import annotations


def parse_task_id(raw: object) -> int | None:
    """The task *raw* names, or None when it is not a string of ASCII decimal digits."""
    if isinstance(raw, str) and raw.isascii() and raw.isdecimal():
        return int(raw)
    return None
