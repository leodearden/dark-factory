"""The note-to-file escalation class: a filer-declared record that requests
nothing, so it is filed already resolved and archived, never pending
(plans/plan-deviation-recording-prd.md D2).
"""

from __future__ import annotations

from typing import Any

from escalation.models import NOTE_TO_FILE_CLASS, Escalation
from escalation.queue import EscalationQueue

NOTE_TO_FILE_RESOLUTION = 'note to file — archived unrouted at filing'
NOTE_TO_FILE_RESOLVER = 'escalation-server'


def note_to_file_refusal(severity: str) -> dict[str, Any] | None:
    """The refusal response for a note that is not info severity, else None."""
    if severity == 'info':
        return None
    return {
        'error': f'note_to_file=True requires severity=info, got {severity!r}',
        'code': 'note_to_file_requires_info',
    }


def file_note_to_file(queue: EscalationQueue, esc: Escalation) -> dict[str, Any]:
    """File *esc* resolved with close_only, so it reaches no reader and no task."""
    refusal = note_to_file_refusal(esc.severity)
    if refusal is not None:
        return refusal
    esc.resolution_action = 'close_only'
    filed = queue.submit_resolved(
        esc,
        NOTE_TO_FILE_RESOLUTION,
        resolved_by=NOTE_TO_FILE_RESOLVER,
        resolution_class=NOTE_TO_FILE_CLASS,
    )
    return {
        'id': filed.id,
        'status': filed.status,
        'resolution_class': filed.resolution_class,
        'level': filed.level,
    }
