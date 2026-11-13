"""The one rule for the task number a producer's TEXT task id names."""

from __future__ import annotations

import re
from typing import Final

_TASK_ID: Final = re.compile(r'[0-9]+')


def task_id_of(text: str | None) -> int | None:
    """The positive integer a TEXT task id spells in decimal digits, else None."""
    if text is None or not _TASK_ID.fullmatch(text):
        return None
    value = int(text)
    return value if value >= 1 else None
