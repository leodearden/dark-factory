"""The consumer side of the C1 ``metadata.pending_since`` anchor.

fused-memory stamps ``pending_since`` on every landing in ``pending``
(``sqlite_task_backend.py::stamp_pending_since``).  This module is the one
reader of that anchor (plans/scheduler-dispatch-scoring-and-lock-layer-prd.md
INV-5); the scheduler's scorer reads it here.  ``pending_age_secs`` is the
PRD's C1 reader contract, kept as the entry point for any further consumer.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from shared.task_deferral import parse_utc_timestamp


class AnchorStatus(enum.Enum):
    PRESENT = 'present'
    ABSENT = 'absent'
    MALFORMED = 'malformed'


@dataclass(frozen=True)
class AnchorReading:
    age_secs: float
    status: AnchorStatus

    @property
    def covered(self) -> bool:
        return self.status is AnchorStatus.PRESENT


def read_pending_age(task: Mapping, now: datetime) -> AnchorReading:
    """Seconds since *task* last landed in pending, and whether that anchor was usable.

    An absent or blank anchor (the writer re-stamps a blank one too) and an
    unparseable one both read as age 0; a future anchor reads as age 0 but is
    still covered, so clock skew between stamper and reader is not an outage.
    """
    metadata = task.get('metadata') or {}
    raw = metadata.get('pending_since') if isinstance(metadata, Mapping) else None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return AnchorReading(0.0, AnchorStatus.ABSENT)
    try:
        anchor = parse_utc_timestamp(raw)
    except (ValueError, TypeError):
        return AnchorReading(0.0, AnchorStatus.MALFORMED)
    return AnchorReading(max(0.0, (now - anchor).total_seconds()), AnchorStatus.PRESENT)


def pending_age_secs(task: Mapping, now: datetime) -> float:
    return read_pending_age(task, now).age_secs
