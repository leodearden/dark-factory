"""Landing-evidence rejects that file no escalation, recorded as events.

Task 4496.  **This docstring is the canonical WHY**; ``EventType.
landing_evidence_rejected`` and the two harness call sites carry a pointer
here, not a copy.

Two ``validate_landing_evidence`` call sites reject without escalating, by
design: the already-landed dispatch gate's ANCESTRY arm and the reconcile
sweep's MARK_DONE_WITH_PROVENANCE arm.  Before this module both rejects left
no durable trace.  That is not harmless at the gate: its ``False`` means
DISPATCH, so a false ``effect_absent`` there sends an agent to redo work that
already landed.  The ``landing_evidence_rejected`` row makes the reject rate
countable, which is the prerequisite for deciding later whether an escalation
or a persistence threshold is warranted.

CADENCE.  One row per NEW or CHANGED signature ``(reason, citation,
effect_check_sha)`` per ``(site, task_id)``; identical repeats are silent.
Two edges release a task's memo so its next reject re-announces: an accepted
verdict at the same site, and ``LandingRejectRecorder.release``, which the
harness calls when it dispatches the task.  So the COUNTING UNIT at the gate
is one row per signature per dispatch the reject let through (the wasted-
dispatch count an escalation threshold would read), never one per tick.  The
gate runs for every dispatch candidate on every tick and the sweep re-visits
a blocked task every sweep, so unconditional rows would be the INV-4 storm
``orchestrator/src/orchestrator/recovery_emission.py::should_emit_event``
already documents for this very gate.  The signature leaves out the survival
numbers, so an ordinary main advance does not re-emit; a moved branch tip
(``effect_check_sha``) is a new fact and does.  The memo is in-memory and
non-durable (a restart re-arms it), bounded by one small tuple per task
rejected and since neither accepted nor dispatched.  It records only a row
handed to the store: no store, a payload that cannot be built or a raising
``emit`` leaves the reject unannounced, so the next identical one retries.
A write failure ``EventStore.emit`` swallows itself is invisible here.

PAYLOAD.  The verdict's probe verbatim, so its key vocabulary stays
single-sourced on ``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::
LandingVerdict.probe``, plus ``site`` and ``task_status`` (applied last, so
they win any collision).  The action taken follows from those two: the gate
(``pending``) dispatches; the sweep reverts ``in-progress`` and leaves
``blocked``.
"""

from __future__ import annotations

import enum
import json
import logging
from typing import TYPE_CHECKING, Any

from orchestrator.event_store import EventType

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.merge_lane.landing_evidence import LandingVerdict

logger = logging.getLogger(__name__)

__all__ = ['LandingRejectRecorder', 'LandingRejectSite', 'landing_reject_payload']


class LandingRejectSite(enum.StrEnum):
    already_landed_gate_ancestry = 'already_landed_gate_ancestry'
    reconcile_sweep = 'reconcile_sweep'


def landing_reject_payload(
    verdict: LandingVerdict, *, site: LandingRejectSite, task_status: str | None,
) -> dict[str, Any]:
    """The event ``data`` for *verdict*: its probe, JSON-safe, plus the site keys.

    The JSON round trip snapshots the probe and stringifies anything
    unserialisable, because ``EventStore.emit`` silently drops a row it
    cannot encode.
    """
    data = json.loads(json.dumps(verdict.probe, default=repr))
    data['site'] = str(site)
    data['task_status'] = task_status
    return data


_Signature = tuple[str, str | None, str | None]


class LandingRejectRecorder:
    """Transition-gated ``landing_evidence_rejected`` emission (see module docstring)."""

    def __init__(self) -> None:
        self._signatures: dict[tuple[str, str], _Signature] = {}

    @staticmethod
    def _key(site: LandingRejectSite, task_id: str) -> tuple[str, str]:
        return (str(site), str(task_id))

    @staticmethod
    def _signature(verdict: LandingVerdict) -> _Signature:
        return (
            str(verdict.reason),
            verdict.probe.get('citation'),
            verdict.probe.get('effect_check_sha'),
        )

    def observe(
        self,
        verdict: LandingVerdict,
        *,
        event_store: EventStore | None,
        site: LandingRejectSite,
        task_id: str,
        task_status: str | None,
    ) -> bool:
        """Record *verdict* at *site*; True only when a row was emitted.

        Never raises: this is telemetry beside a disposition, and must not
        disturb it.
        """
        try:
            key = self._key(site, task_id)
            if verdict.accepted:
                self._signatures.pop(key, None)
                return False
            signature = self._signature(verdict)
            if event_store is None or self._signatures.get(key) == signature:
                return False
            event_store.emit(
                EventType.landing_evidence_rejected,
                task_id=str(task_id),
                data=landing_reject_payload(verdict, site=site, task_status=task_status),
            )
            self._signatures[key] = signature
            return True
        except Exception:
            logger.warning(
                'landing_evidence_rejected emission failed for task %s at %s',
                task_id, site, exc_info=True,
            )
            return False

    def release(self, task_id: str) -> None:
        """Forget *task_id*'s memo at every site: its next reject re-announces."""
        for site in LandingRejectSite:
            self._signatures.pop(self._key(site, task_id), None)
