"""Surfaces the task-verify admission queue while legs are waiting in it.

Each tick reads the gate's pure snapshot and records what a ``phase=verify``
task is waiting behind: a ``verify_admission_heartbeat`` while any waiter is
queued, and a ``verify_admission_wait_aged`` when the oldest waiter past the
alarm age first appears or changes. Age surfacing only; nothing here times a
wait out. Payload shapes live on those EventType members in event_store.py;
design: plans/verify-admission-fifo-prd.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.event_store import EventStore, EventType
from orchestrator.verify_admission_gate import (
    AdmissionOrder,
    AdmissionRequest,
    AdmissionSnapshot,
    HolderView,
    WaiterView,
    gate_for,
)

#: The merge-lane queue heartbeat's event cadence (merge_lane/worker.py::SpeculativeMergeWorker).
ADMISSION_WATCH_INTERVAL_SECS = 300.0


def _identity(request: AdmissionRequest) -> dict[str, Any]:
    return {
        'task_id': request.task_id,
        'module': request.module,
        'role': request.role,
        'plan_id': request.plan_id,
    }


def _holder(holder: HolderView) -> dict[str, Any]:
    return {
        'task_id': holder.request.task_id,
        'module': holder.request.module,
        'role': holder.request.role,
        'held_secs': holder.held_secs,
    }


def heartbeat_payload(snapshot: AdmissionSnapshot, order: AdmissionOrder) -> dict[str, Any]:
    oldest = snapshot.oldest
    return {
        'depth': snapshot.depth,
        'oldest_wait_secs': snapshot.oldest_wait_secs,
        'oldest': None if oldest is None else _identity(oldest.request),
        'holders': [_holder(holder) for holder in snapshot.holders],
        'order': order,
    }


def wait_aged_payload(waiter: WaiterView, *, aged_count: int, alarm_secs: float) -> dict[str, Any]:
    return {
        'waiter': {**_identity(waiter.request), 'wait_secs': waiter.age_secs},
        'aged_count': aged_count,
        'alarm_secs': alarm_secs,
    }


class AdmissionWatch:
    """Per-tick admission facts, alarming once per change of the oldest aged waiter."""

    def __init__(self) -> None:
        self._alarmed_seq: int | None = None

    def tick(
        self,
        event_store: EventStore,
        *,
        slots_dir: Path,
        alarm_secs: float,
        order: AdmissionOrder,
    ) -> None:
        snapshot = gate_for(slots_dir).snapshot()
        if snapshot.depth > 0:
            event_store.emit(
                EventType.verify_admission_heartbeat,
                phase='verify',
                data=heartbeat_payload(snapshot, order),
            )
        aged = [waiter for waiter in snapshot.waiters if waiter.age_secs >= alarm_secs]
        oldest = max(aged, key=lambda waiter: waiter.age_secs, default=None)
        if oldest is None or oldest.seq == self._alarmed_seq:
            return
        self._alarmed_seq = oldest.seq
        event_store.emit(
            EventType.verify_admission_wait_aged,
            task_id=oldest.request.task_id,
            phase='verify',
            role=oldest.request.role,
            data=wait_aged_payload(oldest, aged_count=len(aged), alarm_secs=alarm_secs),
        )
