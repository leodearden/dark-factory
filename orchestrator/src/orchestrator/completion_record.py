"""The metadata the orchestrator stamps on a task's fused-memory completion record.

This is the writer's half of a wire contract. The orchestrator does not depend
on fused_memory, so it spells the values itself; fused-memory owns the other
half (fused_memory/memory_metadata.py::ORCHESTRATOR_COMPLETION_SOURCE and the
kind registry), and fused-memory/tests/server/test_write_triage_completion_records.py
validates :func:`completion_record_metadata` against it.
"""

from __future__ import annotations

_SOURCE = 'orchestrator_completion'
#: A declared kind force-stores the record past write triage, so it is never
#: filed as a child of another task's record
#: (fused-memory/src/fused_memory/server/write_triage.py::declares_attach_keys).
_KIND = 'task_completion'


def completion_record_metadata(task_id: str | int) -> dict[str, str]:
    """The ``add_memory`` metadata for *task_id*'s completion record.

    ``task_id`` is stamped in exact-string form, the form the task_id-keyed
    audits query on.
    """
    return {'task_id': str(task_id), 'source': _SOURCE, 'kind': _KIND}
