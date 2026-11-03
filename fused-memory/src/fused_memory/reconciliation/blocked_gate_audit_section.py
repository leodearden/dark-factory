"""The ``### Blocked Gate-Task Review Audit`` Stage-2 payload section.

This module owns that section: which blocked tasks count as gates awaiting a
human, the order they are listed in, and the section's rendered form.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from shared.timestamps import parse_timestamp_or_warn

from fused_memory.reconciliation.capped_task_list import render_capped_task_list
from fused_memory.reconciliation.task_filter import id_key

BLOCKED_GATE_AUDIT_HEADER = '### Blocked Gate-Task Review Audit'

#: Defensive only: expected never to bind, and a clip is never silent.
MAX_BLOCKED_GATE_AUDIT_RENDERED: int = 60

_UNSTAMPED = datetime.min.replace(tzinfo=UTC)


def render_blocked_gate_audit_section(
    active_tasks: Iterable[object], *, project_id: str, run_id: str | None
) -> str:
    """Render every blocked gate task, rendering the header even when none exist."""
    gates = select_blocked_gate_tasks(active_tasks)
    total = len(gates)
    # The kept head is the OLDEST gates only because select_blocked_gate_tasks sorts ascending.
    body = render_capped_task_list(
        gates,
        cap=MAX_BLOCKED_GATE_AUDIT_RENDERED,
        cap_name='MAX_BLOCKED_GATE_AUDIT_RENDERED',
        omitted_noun='blocked gate task(s)',
        dropped_first='the most recently escalated gates',
        log_event='reconciliation.gate_task_audit_render_capped',
        log_extra={'project_id': project_id, 'run_id': run_id, 'total_gate_tasks': total},
    )
    return f'\n{BLOCKED_GATE_AUDIT_HEADER} ({total} gate task(s) awaiting review)\n{body}'


def select_blocked_gate_tasks(active_tasks: Iterable[object]) -> list[dict]:
    """Return every blocked gate task, oldest ``gate_escalated_at`` first.

    A blocked task is a gate when EITHER arm holds; each arm covers the
    other's blind spot:

    * Declared: ``task_kind == 'deterministic'`` and ``operational_mode``
      absent or ``'gate'``. ``task_kind`` needs positive evidence while an
      absent ``operational_mode`` matches, following each field's declared
      default in ``shared/src/shared/task_metadata.py::TaskMetadata``. This
      arm alone catches deterministic tasks blocked with no stamp (the
      ScriptTimeout and infra_issue paths in the
      ``orchestrator/src/orchestrator/deterministic_runner.py`` module
      docstring).
    * Observed: a non-empty ``gate_escalated_at`` string. This arm alone
      catches the llm-mode coerced pure gates described at
      ``fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py::gate_escalated_age_secs``.

    Intended to consume the UNCAPPED ``FilteredTaskTree.active_tasks``.
    Unstamped and unparseable entries sort first so a render cap, which
    keeps the head, never clips them; ties break by task id ascending.
    """
    selected: list[tuple[datetime, int, dict]] = []
    for task in active_tasks:
        if not isinstance(task, dict) or task.get('status') != 'blocked':
            continue
        raw = task.get('metadata')
        metadata = raw if isinstance(raw, dict) else {}
        stamp = metadata.get('gate_escalated_at')
        stamped = isinstance(stamp, str) and bool(stamp)
        declared = (
            metadata.get('task_kind') == 'deterministic'
            and metadata.get('operational_mode', 'gate') == 'gate'
        )
        if declared or stamped:
            selected.append((_stamp_sort_key(stamp if stamped else None), id_key(task), task))
    selected.sort(key=lambda entry: (entry[0], entry[1]))
    return [task for *_, task in selected]


def _stamp_sort_key(stamp: str | None) -> datetime:
    if stamp is None:
        return _UNSTAMPED
    parsed, _ = parse_timestamp_or_warn(stamp, context='blocked_gate_audit.gate_escalated_at')
    return parsed
