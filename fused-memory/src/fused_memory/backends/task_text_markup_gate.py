"""Storage-layer tripwire for leaked tool-call envelope markup in task text (task 4419).

``SqliteTaskBackend.add_task`` and ``SqliteTaskBackend.update_task`` call
:func:`refuse_leaked_task_text` inside their write transaction, so every
writer is covered, including the ones that never cross MCP dispatch and so
never meet ``fused_memory/server/markup_guard.py::install_markup_guard``.

This gate uses the PRECISE stored-text detector,
``fused_memory/utils/toolcall_xml_leak.py::detect_leak``, not the recall-first
predicate the MCP boundary uses. A false positive here refuses a legitimate
write with no caller to resubmit it, and the precise detector passes prose
that quotes the leak in the repo's escaped convention. The detector and the
column list both come from that module, so this gate and the read-time sweep
(``scripts/scan_task_toolcall_leaks.py``) cannot disagree on what counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from fused_memory.backends.task_backend_errors import LeakedEnvelopeMarkupError
from fused_memory.utils.toolcall_xml_leak import SCANNED_COLUMNS, detect_leak

__all__ = ['LeakedTaskText', 'find_leaked_task_text', 'refuse_leaked_task_text']


class LeakedTaskText(NamedTuple):
    """The first task-text column found carrying a leaked fragment."""

    column: str
    fragment: str


def find_leaked_task_text(values: Mapping[str, object]) -> LeakedTaskText | None:
    """Return the first leak among *values*, a write's text keyed by column.

    Only the columns in ``SCANNED_COLUMNS`` are inspected, in that order.
    Absent, ``None`` and non-string values are clean.
    """
    for column in SCANNED_COLUMNS:
        fragment = detect_leak(values.get(column))
        if fragment is not None:
            return LeakedTaskText(column=column, fragment=fragment)
    return None


def refuse_leaked_task_text(values: Mapping[str, object]) -> None:
    """Raise :class:`LeakedEnvelopeMarkupError` when *values* carry a leak."""
    leak = find_leaked_task_text(values)
    if leak is not None:
        raise LeakedEnvelopeMarkupError(column=leak.column, fragment=leak.fragment)
