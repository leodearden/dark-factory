"""Preserved-specimen reset refusal at the ``submit_task`` boundary (task 5456).

Refuses a recon-filed submission that routing turns into an ESCALATING gate
and that asks for a deliberately preserved specimen to be moved out of
``in-progress``.  What a preserved specimen is, and why moving one destroys
evidence, is ``reconciliation/preservation_specimen_guard.py``'s module
docstring; this module reuses that module's corroboration primitive and
stranded-claim discriminator rather than restating either.

WHY THIS CHOKEPOINT.  Stage 1's ``filter_preservation_specimen_flags`` sits on
Stage 1's flag chain only.  Stage 2's own stranded-task detection files the
gate directly: tasks 5080 and 5104 asked a human to reset specimen 3105, and
mem0 6abfe67f asks whether Stage 2's detection is the generator.  Every filing
path ends in ``submit_task``, so that is where this layer sits.

A SUBJECT IS REQUIRED.  A reset gate must name the task it asks to move,
through ``recurring_gate_guard``'s incoming subject keys.  None of the five
historical stranded-reset gates (3463, 3944, 3965, 5080, 5104) declared one,
so a layer that let a subject-less filing through would have caught none of
them.  Such a filing is refused with no I/O and a hint naming the key.

THE DISCRIMINATOR.  A submission requests a reset when ``flag_asserts_stranded``
answers True for its title read as ``suggested_action`` and its description
read as ``description``.  No ``flag_type`` is passed: a title is not a flag
type, and reading it as one fires on gates that ask for no reset.

ESCALATING GATES ONLY.  The scope is whatever ``inject_operational_routing``
stamps ``always_escalates=True``, at admission.  Two named residuals are not
covered here: a recon filing that does not become an escalating gate, and an
``update_task`` amendment of an already-open carrier, the route the
recurring-gate dedupe steers a recon stage to, which can reword a reset carrier
minted before this guard with no preserved-specimen check.

FAILS OPEN.  A degraded read, a timeout or any other failure logs a WARNING and
lets the submission through.  Refusing on a backend blip would block every
recon reset gate, including the genuine ones a human needs to see.  Unlike
Stage 1's layer, which escalates an unresolved cycle, this one is handed no
escalation channel, so a persistent outage is audible here only as those
WARNINGs; the gate it admits still lands in front of a human.

IMPORT CONTRACT.  This module imports ``recurring_gate_guard``'s public subject
readers, ``operational_routing_guard.inject_operational_routing``,
``preservation_specimen_guard``'s public primitive and discriminator, and
``models.scope.resolve_project_id``.  It never imports ``task_interceptor`` or
``server/`` at module level.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fused_memory.middleware.operational_routing_guard import inject_operational_routing
from fused_memory.middleware.recurring_gate_guard import (
    GATE_SUBJECT_KEY,
    GATE_SUBJECT_SUBMISSION_ALIASES,
    extract_gate_subject,
)
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.preservation_specimen_guard import (
    corroborate_preservation,
    flag_asserts_stranded,
)

__all__ = [
    'PRESERVED_SPECIMEN_GATE_TIMEOUT_SECONDS',
    'files_an_escalating_gate',
    'preserved_specimen_gate_error',
    'preserved_specimen_reset_error',
    'reset_gate_subject_required_error',
    'submission_requests_reset',
]

logger = logging.getLogger(__name__)

#: Ceiling on one submission's corroboration.  Exceeding it fails open.
PRESERVED_SPECIMEN_GATE_TIMEOUT_SECONDS: float = 15.0


def files_an_escalating_gate(metadata: str | dict[str, Any] | None) -> bool:
    """True when routing turns *metadata* into an ``always_escalates`` gate.

    Asks the routing itself rather than copying its coerced-class set:
    ``recurring_gate_guard.is_gate_submission`` answers False for
    operational+``llm`` and for ``decision``, and routing escalates both.
    Pure apart from the routing's lazy import.
    """
    routed = inject_operational_routing(metadata)
    return isinstance(routed, dict) and routed.get('always_escalates') is True


def submission_requests_reset(title: str | None, description: str | None) -> bool:
    """True when the submission asks for a task to be moved out of ``in-progress``.

    Total over ``None``.  Pure, sync, no I/O.
    """
    return flag_asserts_stranded({'suggested_action': title, 'description': description})


def reset_gate_subject_required_error() -> dict[str, Any]:
    """The structured refusal for a recon reset gate that names no subject."""
    return {
        'error': (
            'a recon-filed gate asking for a task to be moved out of in-progress '
            'must name that task, so it can be checked against the preserved '
            'specimens; this submission names none.'
        ),
        'error_type': 'ResetGateSubjectRequired',
        'hint': (
            f'Re-file with metadata.{GATE_SUBJECT_KEY} set to the id of the task '
            f'this gate asks to move.'
        ),
    }


def preserved_specimen_reset_error(subject: str, citation: str) -> dict[str, Any]:
    """The structured refusal for a reset gate whose subject is a preserved specimen.

    Same shape as ``recurring_gate_guard.recurring_gate_error``:
    ``gate_subject`` and ``preservation_citation`` are the machine-readable
    half, so no consumer has to parse the prose.
    """
    return {
        'error': (
            f'task {subject} is a deliberately preserved specimen (cited by '
            f'{citation}); a gate asking for it to be moved out of in-progress '
            f'requests the destruction of the evidence it is kept for.'
        ),
        'error_type': 'PreservedSpecimenResetViolation',
        'gate_subject': subject,
        'preservation_citation': citation,
        'hint': (
            f'Do not file this gate, and do not reset, re-dispatch or cancel task '
            f'{subject}. The cited record says its in-progress/no-claimant state '
            f'is deliberate, and cancelling it is equally destructive. If the '
            f'citation looks stale, retiring it is for a human: cite {citation} '
            f'in your finding rather than requesting the write.'
        ),
    }


async def preserved_specimen_gate_error(
    metadata: str | dict[str, Any] | None,
    agent_id: str | None,
    project_root: str,
    *,
    title: str | None,
    description: str | None,
    memory_service: Any,
    timeout_seconds: float = PRESERVED_SPECIMEN_GATE_TIMEOUT_SECONDS,
    log: logging.Logger = logger,
) -> dict[str, Any] | None:
    """Refuse a recon reset gate that names no subject or names a preserved specimen.

    Gates run cheapest-first, so a non-recon caller costs one ``startswith``
    and memory is read only on the narrow recon reset-gate path.  Returns
    ``None`` whenever the submission may proceed, including every fail-open
    outcome.  ``asyncio.CancelledError`` propagates.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None
    if not files_an_escalating_gate(metadata):
        return None
    if not submission_requests_reset(title, description):
        return None

    subject = extract_gate_subject(metadata, aliases=GATE_SUBJECT_SUBMISSION_ALIASES)
    if subject is None:
        log.warning(
            'preserved_specimen_gate_guard: refusing recon reset gate from %s — '
            'it declares no %s',
            agent_id, GATE_SUBJECT_KEY,
        )
        return reset_gate_subject_required_error()

    project_id = resolve_project_id(project_root)
    try:
        verdict = await asyncio.wait_for(
            corroborate_preservation(memory_service, project_id, subject, log=log),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of gate_subject=%s in '
            'project %s timed out after %.2fs — unresolved, submission proceeds',
            subject, project_id, timeout_seconds,
        )
        return None
    except Exception:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of gate_subject=%s in '
            'project %s failed — unresolved, submission proceeds',
            subject, project_id, exc_info=True,
        )
        return None

    if verdict.citation is not None:
        log.warning(
            'preserved_specimen_gate_guard: refusing recon reset gate for '
            'gate_subject=%s in project %s — preserved specimen cited by %s',
            subject, project_id, verdict.citation,
        )
        return preserved_specimen_reset_error(subject, verdict.citation)

    if verdict.degraded:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of gate_subject=%s in '
            'project %s unresolved, submission proceeds',
            subject, project_id,
        )
    return None
