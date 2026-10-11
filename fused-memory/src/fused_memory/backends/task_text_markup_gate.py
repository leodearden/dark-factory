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
and was stored as 'medium'). Which arguments a write supplied is judged on the
sink's arguments as received, before any default or stamp fills them in.
"""

from __future__ import annotations

from collections.abc import Mapping

from shared.toolcall_markup import repair

from fused_memory.backends.task_backend_errors import LeakedEnvelopeMarkupError
from fused_memory.utils.toolcall_xml_leak import SCANNED_COLUMNS, detect_leak

__all__ = ['refuse_leaked_task_text']


def refuse_leaked_task_text(
    text: Mapping[str, str | None],
    arguments: Mapping[str, object],
) -> None:
    """Raise :class:`LeakedEnvelopeMarkupError` for the first leak in *text*.

    *text* is the write's persisted values keyed by column; only the columns in
    ``SCANNED_COLUMNS`` are inspected, in that order. *arguments* is the sink's
    full argument map as received: its names are the parameters a fragment may
    have swallowed, and its non-None entries are the ones the write supplied,
    which are never reported as swallowed.
    """
    supplied = [name for name, value in arguments.items() if value is not None]
    for column in SCANNED_COLUMNS:
        value = text.get(column)
        if value is None:
            continue
        fragment = detect_leak(value)
        if fragment is None:
            continue
        repaired = repair(value, column, arguments.keys(), supplied)
        raise LeakedEnvelopeMarkupError(
            column=column,
            fragment=fragment,
            recovered=repaired.recovered if repaired is not None else {},
            clean_value=repaired.clean_value if repaired is not None else None,
        )
