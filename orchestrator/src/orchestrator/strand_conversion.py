"""Record a convert-to-blocked: the stamp and the event that name what pins the converted row.

When orchestrator/src/orchestrator/harness.py::Harness._reconcile_one_stranded's enforce arm moves a
pinned, unclaimed in-progress row to blocked, two durable facts must name what pins it:
``metadata.strand_converted`` (what get_task shows an operator and the L1 rotation) and one
``strand_converted`` event (what task 3546's gate step 3 and the dashboard count). Both derive from
ONE ``StrandConverted`` value, so they cannot disagree. It is called only after the blocked write
succeeded, so it is fail-open: the status write already happened, and nothing here may abort the
sweep that made it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from pydantic import ValidationError
from shared.strand_conversion import StrandConverted

from orchestrator.event_store import EventStore, EventType
from orchestrator.recovery_emission import emit_recovery_event, escalation_ages_secs, pin_buckets
from orchestrator.task_ground_truth import TruthReport, recovery_shape_str

__all__ = ['TaskMetadataWriter', 'record_strand_conversion', 'strand_converted_payload']

logger = logging.getLogger(__name__)


class TaskMetadataWriter(Protocol):
    """The ``update_task`` this module writes the stamp through; the merge mode is always explicit."""

    async def __call__(
        self, task_id: str, metadata: dict[str, dict], /, *, metadata_mode: str,
    ) -> bool: ...


def strand_converted_payload(
    stamp: StrandConverted,
    *,
    task_id: str,
    escalation_ids: dict[str, list[str]],
    ages_secs: dict[str, float | None],
    metadata_stamped: bool,
) -> dict[str, object]:
    """The ``strand_converted`` event's payload; every stamp-backed field is read from *stamp*."""
    return {
        'task_id': task_id,
        'shape': stamp.shape,
        'escalation_ids': escalation_ids,
        'pinning_escalation_ids': list(stamp.pinning_escalation_ids),
        'ages_secs': ages_secs,
        'measured_at': stamp.at.isoformat(),
        'metadata_stamped': metadata_stamped,
    }


async def record_strand_conversion(
    task_id: str,
    report: TruthReport,
    *,
    event_store: EventStore | None,
    update_task: TaskMetadataWriter,
    now: datetime,
) -> StrandConverted | None:
    """Stamp *task_id* and emit its ``strand_converted`` event; ``None`` when nothing pins it.

    Never raises.
    """
    pins = pin_buckets(
        task_id, report.open_escalations, store_unavailable=report.escalation_store_unavailable,
    )
    shape = recovery_shape_str(report)
    try:
        stamp = StrandConverted(
            at=now, shape=shape,
            pinning_escalation_ids=tuple(sorted(pins.buckets['queue_handoff'])),
        )
    except ValidationError as exc:
        logger.error(
            'strand conversion of task %s not recorded: a converted row must be pinned by at '
            'least one queue-handoff record (shape=%s, escalation_ids=%s): %s',
            task_id, shape, pins.buckets, exc,
        )
        return None
    metadata_stamped = await _write_stamp(task_id, stamp, update_task)
    emit_recovery_event(
        event_store=event_store,
        event_type=EventType.strand_converted,
        task_id=task_id,
        payload=strand_converted_payload(
            stamp,
            task_id=task_id,
            escalation_ids=pins.buckets,
            ages_secs=escalation_ages_secs(report.open_escalations, now=now),
            metadata_stamped=metadata_stamped,
        ),
    )
    return stamp


async def _write_stamp(
    task_id: str, stamp: StrandConverted, update_task: TaskMetadataWriter,
) -> bool:
    try:
        accepted = await update_task(task_id, stamp.to_metadata(), metadata_mode='merge')
    except Exception as exc:
        logger.warning(
            'strand conversion of task %s: metadata.strand_converted write raised: %s', task_id, exc,
        )
        return False
    if not accepted:
        logger.warning(
            'strand conversion of task %s: metadata.strand_converted write was refused', task_id,
        )
        return False
    return True
