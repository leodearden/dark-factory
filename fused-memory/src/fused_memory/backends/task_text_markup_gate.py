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

A refusal names the arguments the fragment swallowed, recovered by
``shared/src/shared/toolcall_markup.py::repair``. That is the value a sink
would otherwise have silently defaulted (task 4358 asked for priority 'low'
and was stored as 'medium'), and the gate runs before any default is derived.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from shared.toolcall_markup import repair

from fused_memory.backends.task_backend_errors import LeakedEnvelopeMarkupError
from fused_memory.utils.toolcall_xml_leak import SCANNED_COLUMNS, detect_leak

__all__ = ['LeakedTaskText', 'find_leaked_task_text', 'refuse_leaked_task_text']


class LeakedTaskText(NamedTuple):
    """The first task-text column found carrying a leaked fragment."""

    column: str
    fragment: str
    #: The swallowed arguments, name -> value; empty when no boundary between
    #: the caller's text and the absorbed arguments is provable.
    recovered: dict[str, str]
    #: The column's value up to the fragment, or None under that same condition.
    clean_value: str | None


def find_leaked_task_text(
    text: Mapping[str, str | None],
    arguments: Mapping[str, object],
) -> LeakedTaskText | None:
    """Return the first leak in *text*, a write's persisted values keyed by column.

    *arguments* is the sink's full argument map as received: its names are the
    parameters a fragment may have swallowed, and its non-None entries are the
    ones the write supplied, which are never reported as swallowed. Only the
    columns in ``SCANNED_COLUMNS`` are inspected, in that order.
    """
    supplied = [name for name, value in arguments.items() if value is not None]
    for column in SCANNED_COLUMNS:
        value = text.get(column)
        fragment = detect_leak(value)
        if value is None or fragment is None:
            continue
        repaired = repair(value, column, arguments.keys(), supplied)
        return LeakedTaskText(
            column=column,
            fragment=fragment,
            recovered=repaired.recovered if repaired is not None else {},
            clean_value=repaired.clean_value if repaired is not None else None,
        )
    return None


def refuse_leaked_task_text(
    text: Mapping[str, str | None],
    arguments: Mapping[str, object],
) -> None:
    """Raise :class:`LeakedEnvelopeMarkupError` when *text* carries a leak."""
    leak = find_leaked_task_text(text, arguments)
    if leak is not None:
        raise LeakedEnvelopeMarkupError(
            column=leak.column,
            fragment=leak.fragment,
            recovered=leak.recovered,
            clean_value=leak.clean_value,
        )
