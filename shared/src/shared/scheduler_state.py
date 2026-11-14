"""The scheduler state snapshot: one typed shape for its producer and every reader.

The orchestrator scheduler builds a ``SchedulerStateSnapshot`` and writes its
``to_json()`` form to ``scheduler_state.json``; fused-memory and the dashboard
parse that form back with ``from_json``. Unknown keys from a newer producer are
ignored and keys an older producer lacks take their defaults, so a field is
added in one place: here. Each process re-serialises through its own deployed
copy of this model, so fused-memory's ``get_scheduler_state`` forwards only the
fields that copy defines: a new field reaches the dashboard only once the
orchestrator, fused-memory and the dashboard all run code that defines it.
``snapshot_at is None`` means no producer has written a snapshot.
Contract: plans/review-all-s14-shared-leaves-prd.md (decision 12, invariant I4,
INV-13).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_serializer,
)

__all__ = [
    'MalformedSnapshotError',
    'OverrideRecord',
    'ParkRecord',
    'PinQueueEntry',
    'RequeueCooldown',
    'SchedulerStateSnapshot',
]

_RECORD_CONFIG = ConfigDict(frozen=True, extra='ignore')


class MalformedSnapshotError(ValueError):
    """The input to ``SchedulerStateSnapshot.from_json`` is not a snapshot's wire form."""


class ParkRecord(BaseModel):
    """The modules one owner holds parked, and when the park was installed ('' if unknown)."""

    model_config = _RECORD_CONFIG

    modules: tuple[str, ...] = ()
    installed_at: str = ''


class PinQueueEntry(BaseModel):
    model_config = _RECORD_CONFIG

    task_id: str
    order: int | None = None


class OverrideRecord(BaseModel):
    """An operator override; the all-default record means no override."""

    model_config = _RECORD_CONFIG

    boost_tier: str | None = None
    pinned: bool = False
    reserve_now: bool = False
    ttl_until: str | None = None


class RequeueCooldown(BaseModel):
    """The requeue cooldown gating a task's re-dispatch, as armed.

    Every value is fixed at arming, never derived from now, because the
    producer content-dedups snapshot writes. ``envelope_secs`` is the
    un-jittered ceiling ``armed_secs`` was drawn from (armed lies in
    [envelope/2, envelope]); both grow across successive transient requeues
    and stay flat for a genuine one. ``armed_at`` and ``expires_at`` are
    ISO-8601 wall-clock strings. An expired entry can linger until the
    producer's next tick, so compare ``expires_at`` against now rather than
    treating presence as proof the task is still cooling.
    """

    model_config = _RECORD_CONFIG

    armed_secs: float
    envelope_secs: float
    transient: bool
    transient_count: int
    armed_at: str
    expires_at: str


class SchedulerStateSnapshot(BaseModel):
    """``parks`` maps each owner to the modules it tops; ``park_stacks`` maps each
    module to its whole stack, bottom to top, as open entry records; ``lock_depth``
    is the depth the producer normalises lock keys to. ``pause_class`` is the wire
    value of ``orchestrator/src/orchestrator/pause_policy.py::PauseClass`` (shared
    cannot import orchestrator), ``None`` when not paused.

    Frozen means no field can be reassigned. The mappings are plain dicts that
    validation copies from its input, so a snapshot never aliases its producer's
    state, but nothing stops a reader mutating one in place.
    """

    model_config = _RECORD_CONFIG

    skip_counts: Mapping[str, int] = Field(default_factory=dict)
    parks: Mapping[str, ParkRecord] = Field(default_factory=dict)
    park_stacks: Mapping[str, tuple[Mapping[str, Any], ...]] = Field(default_factory=dict)
    pin_reservations: Mapping[str, ParkRecord] = Field(default_factory=dict)
    effective_priorities: Mapping[str, str] = Field(default_factory=dict)
    pin_queue: tuple[PinQueueEntry, ...] = ()
    overrides: Mapping[str, OverrideRecord] = Field(default_factory=dict)
    current_holders: Mapping[str, str] = Field(default_factory=dict)
    lock_depth: StrictInt | None = None
    is_paused: bool = False
    pause_reason: str | None = None
    pause_class: str | None = None
    requeue_cooldowns: Mapping[str, RequeueCooldown] = Field(default_factory=dict)
    snapshot_at: AwareDatetime | None = None

    @field_serializer('snapshot_at')
    def _render_snapshot_at(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    @classmethod
    def empty(cls) -> SchedulerStateSnapshot:
        return cls()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SchedulerStateSnapshot:
        """Parse a snapshot's wire form; unknown keys are ignored.

        Raises MalformedSnapshotError when *data* does not have the snapshot's shape.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise MalformedSnapshotError(str(exc)) from exc

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode='json')
