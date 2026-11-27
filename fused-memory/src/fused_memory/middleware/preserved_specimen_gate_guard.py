"""Preserved-specimen reset refusal at the ``submit_task`` boundary (task 5456).

Refuses a recon-filed submission that routing turns into an ESCALATING gate
and that asks for a deliberately preserved specimen to be moved out of
``in-progress``.  What a preserved specimen is, and why moving one destroys
evidence, is ``reconciliation/preservation_specimen_guard.py``'s module
docstring; this module reuses that module's corroboration primitive and
stranded-claim discriminator rather than restating either.

WHY THIS CHOKEPOINT.  Stage 1's ``filter_preservation_specimen_flags`` sits on
Stage 1's flag chain only, and Stage 2's own stranded-task detection files the
gate directly, as tasks 5080 and 5104 did for specimen 3105.  Every filing
path ends in ``submit_task``, so that is where this layer sits.

A SUBJECT IS REQUIRED.  A reset gate must name the task it asks to move,
through ``recurring_gate_guard``'s incoming subject keys, as a bare task id;
several are comma-joined, and each is corroborated.  A filing that names
none, or names one in a spelling the store cannot key on (``'#3105'``,
``'task-3105'``), is refused with no I/O and a hint asking for the bare id.
Letting either through would exempt the commonest shape: a subject-less
filing passes trivially, and an unkeyable one reads back as "not preserved".

THE DISCRIMINATOR.  A submission requests a reset when two readings agree.
Stage 1's ``flag_asserts_stranded``, over the title read as
``suggested_action`` and the description read as ``description``, bounds the
class, so the vocabulary has one home; no ``flag_type`` is passed, because a
title is not a flag type.  Inside that bound this layer also requires
:data:`_MOVE_OUT_OF_IN_PROGRESS`: a stranded-action verb that
moves a task, naming a destination status, naming ``in-progress`` as the
move's source, or taking the task as its whole object.  A verb that is
negated, used as a noun, or under a preventive lead-in requests nothing
(:data:`_NON_REQUEST_PREFIX`).  Stage 1's one-field
conjunction alone fires on a PROTECTIVE gate ("reopen the escalation pinning
task 3105 so it is not reset out of in-progress"), the twin specimen's loss
mode; at this layer a false positive is not a dropped flag but a gate no
human ever sees.

ESCALATING GATES ONLY.  The scope is whatever ``inject_operational_routing``
stamps ``always_escalates=True``, at admission.  Three named residuals are not
covered here:

1. a recon filing that does not become an escalating gate;
2. an ``update_task`` amendment of an already-open carrier, the route the
   recurring-gate dedupe steers a recon stage to, which can reword a reset
   carrier minted before this guard with no preserved-specimen check;
3. a request worded outside Stage 1's verb family: "move ... to pending",
   "set ... to pending" and "cancel" are not read as a reset.  The recon
   prompt tells the stage a refused gate must not be re-filed in other
   words; nothing here enforces it.

FAILS OPEN.  A degraded read, a timeout or any other failure logs a WARNING and
lets the submission through.  Refusing on a backend blip would block every
recon reset gate, including the genuine ones a human needs to see.  A
composite subject is read component by component under one shared ceiling,
each unread component getting an equal share of what remains; a citation
already read refuses the gate whatever a later component does, and a
component that times out or fails is unresolved on its own.  Unlike Stage 1's
layer, which escalates an unresolved cycle, this one is handed no
escalation channel, so a persistent outage is audible here only as those
WARNINGs; the gate it admits still lands in front of a human.

IMPORT CONTRACT.  This module imports ``recurring_gate_guard``'s public subject
readers, ``operational_routing_guard.inject_operational_routing``,
``preservation_specimen_guard``'s public primitive, discriminator and action
vocabulary, ``flag_identity.parse_flag_subject``,
``shared.task_statuses.TaskStatus`` and ``models.scope.resolve_project_id``.
It never imports ``task_interceptor`` or ``server/`` at module level.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from shared.task_statuses import TaskStatus

from fused_memory.middleware.operational_routing_guard import inject_operational_routing
from fused_memory.middleware.recurring_gate_guard import (
    GATE_SUBJECT_KEY,
    GATE_SUBJECT_SUBMISSION_ALIASES,
    extract_gate_subject,
)
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.flag_identity import parse_flag_subject
from fused_memory.reconciliation.preservation_specimen_guard import (
    STRANDED_ACTION_TARGET_FAMILY,
    STRANDED_ACTION_VERB_FAMILY,
    PreservationCorroboration,
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

#: Ceiling on one submission's corroboration, shared by every component of
#: its subject.  Exceeding it fails open.
PRESERVED_SPECIMEN_GATE_TIMEOUT_SECONDS: float = 15.0


def _alternation(members: tuple[str, ...]) -> str:
    return '|'.join(map(re.escape, members))


#: Longest first, so no member can shadow a longer one in an alternation.
_STATUSES_OUT_OF_IN_PROGRESS: tuple[str, ...] = tuple(
    sorted((s.value for s in TaskStatus if s is not TaskStatus.IN_PROGRESS), key=len, reverse=True)
)

_QUOTE = "['\"`]?"
_IN_PROGRESS = rf'{_QUOTE}(?:{_alternation(STRANDED_ACTION_TARGET_FAMILY)})'
_DESTINATION = rf'{_QUOTE}(?:{_alternation(_STATUSES_OUT_OF_IN_PROGRESS)})\b'
_TASK_OBJECT = r'\s+(?:it|(?:the\s+)?(?:[\w-]+\s+)?tasks?(?:\s+#?\d+)?)'
_MOVE = (
    rf'\s+(?:back\s+)?to\s+{_DESTINATION}'
    rf'|\s+(?:out\s+of|from)\s+{_IN_PROGRESS}'
    rf'|\s+{_IN_PROGRESS}{_QUOTE}\s*(?:→|->|=>|\bto\b)\s*{_DESTINATION}'
)

#: A stranded-action verb, in any inflection, that moves a task out of
#: ``in-progress``: an optional object ("it", or "task" with at most one word
#: before it and an optional id) followed by a destination status or by
#: ``in-progress`` named as the source, or the object alone ending the clause.
#: "reset stranded task 3105 out of in-progress", "resetting in-progress →
#: pending" and "reset to pending" match; "reset it to open so task 3105 stays
#: in-progress" does not.
_MOVE_OUT_OF_IN_PROGRESS: re.Pattern[str] = re.compile(
    rf'\b(?:{_alternation(STRANDED_ACTION_VERB_FAMILY)})\w*'
    rf'(?:(?:{_TASK_OBJECT})?(?:{_MOVE})|{_TASK_OBJECT}(?=\s*(?:[.;:!?)]|$)))',
    re.IGNORECASE,
)

_NEGATION = r"(?:\b(?:not|never|without|instead\s+of|rather\s+than)|n['’]t)"
_AUXILIARY_OR_ADVERB = r'(?:be|been|being|get|got|getting|again|ever|\w+ly)'
_PREVENTIVE = r'\b(?:prevent|avoid|stop|lest|otherwise|or\s+else)\w*'
_CLAUSE_WORD = r"(?!(?:and|then|but|or|so|also)\b)[\w'’-]+"

#: Text ending where the verb after it requests nothing: a negation, perhaps
#: with auxiliaries and adverbs between ("so it is not accidentally reset",
#: "never again reset", "without resetting"); an article making the verb a
#: noun ("a reset"); or a preventive or counterfactual lead-in a few words
#: back in the same clause ("to prevent a reset", "to avoid resetting it",
#: "otherwise recon will reset").  A false positive refuses a gate no human
#: then sees, so this reading leans protective.
_NON_REQUEST_PREFIX: re.Pattern[str] = re.compile(
    rf'(?:{_NEGATION}(?:\s+{_AUXILIARY_OR_ADVERB}){{0,3}}'
    r'|\b(?:a|an|the|any|another)'
    rf'|{_PREVENTIVE}(?:\s+{_CLAUSE_WORD}){{0,4}})'
    r'\s+$',
    re.IGNORECASE,
)

#: A component the corroboration reads can key on: a task id, or a subtask id.
_TASK_ID: re.Pattern[str] = re.compile(r'[1-9][0-9]*(?:\.[1-9][0-9]*)?')


def files_an_escalating_gate(metadata: str | dict[str, Any] | None) -> bool:
    """True when routing turns *metadata* into an ``always_escalates`` gate.

    Asks the routing itself rather than copying its coerced-class set:
    ``recurring_gate_guard.is_gate_submission`` answers False for
    operational+``llm`` and for ``decision``, and routing escalates both.
    Pure, sync, no I/O.
    """
    routed = inject_operational_routing(metadata)
    return isinstance(routed, dict) and routed.get('always_escalates') is True


def _moves_a_task_out_of_in_progress(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    return any(
        not _NON_REQUEST_PREFIX.search(text, 0, match.start())
        for match in _MOVE_OUT_OF_IN_PROGRESS.finditer(text)
    )


def submission_requests_reset(title: str | None, description: str | None) -> bool:
    """True when the submission asks for a task to be moved out of ``in-progress``.

    The module docstring's DISCRIMINATOR paragraph states the rule.  Total
    over ``None``.  Pure, sync, no I/O.
    """
    return flag_asserts_stranded(
        {'suggested_action': title, 'description': description}
    ) and any(_moves_a_task_out_of_in_progress(text) for text in (title, description))


def _reset_gate_task_ids(subject: str | None) -> tuple[str, ...]:
    """The task ids *subject* names, or ``()`` unless every component is one.

    Decomposed by ``flag_identity.parse_flag_subject``, the parser Stage 1
    decomposes a flag's task_id with.  One unkeyable component empties the
    answer: corroborating only the rest would check fewer tasks than the gate
    asks to move.
    """
    components = parse_flag_subject(subject).components
    if components and all(_TASK_ID.fullmatch(c) for c in components):
        return components
    return ()


def reset_gate_subject_required_error(subject: str | None) -> dict[str, Any]:
    """The structured refusal for a recon reset gate naming no task id it can be checked by."""
    named = 'names none' if subject is None else f'names {subject!r}, which is not a task id'
    return {
        'error': (
            'a recon-filed gate asking for a task to be moved out of in-progress '
            'must name that task, so it can be checked against the preserved '
            f'specimens; this submission {named}.'
        ),
        'error_type': 'ResetGateSubjectRequired',
        'hint': (
            f'Re-file with metadata.{GATE_SUBJECT_KEY} set to the bare id of the '
            f"task this gate asks to move, such as '3105'; comma-join the ids if "
            f'it asks to move several.'
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


async def _corroborate_component(
    memory_service: Any,
    project_id: str,
    task_id: str,
    *,
    timeout_seconds: float,
    log: logging.Logger,
) -> PreservationCorroboration:
    try:
        return await asyncio.wait_for(
            corroborate_preservation(memory_service, project_id, task_id, log=log),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of task %s in project %s '
            'timed out after %.2fs — unresolved',
            task_id, project_id, timeout_seconds,
        )
        return PreservationCorroboration(citation=None, degraded=True)
    except Exception:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of task %s in project %s '
            'failed — unresolved',
            task_id, project_id, exc_info=True,
        )
        return PreservationCorroboration(citation=None, degraded=True)


async def _corroborate_until_cited(
    memory_service: Any,
    project_id: str,
    task_ids: tuple[str, ...],
    *,
    timeout_seconds: float,
    log: logging.Logger,
) -> dict[str, PreservationCorroboration]:
    """Corroborate *task_ids* in order under one ceiling, stopping at the first citation.

    Each read gets an equal share of what remains, so a slow or hung component
    can neither discard a citation already read nor starve a later component
    of its read.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    verdicts: dict[str, PreservationCorroboration] = {}
    for position, task_id in enumerate(task_ids):
        share = (deadline - loop.time()) / (len(task_ids) - position)
        verdict = await _corroborate_component(
            memory_service, project_id, task_id, timeout_seconds=share, log=log,
        )
        verdicts[task_id] = verdict
        if verdict.citation is not None:
            break
    return verdicts


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
    """Refuse a recon reset gate that names no task id or names a preserved specimen.

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
    task_ids = _reset_gate_task_ids(subject)
    if not task_ids:
        log.warning(
            'preserved_specimen_gate_guard: refusing recon reset gate from %s — '
            'its %s %r names no task id',
            agent_id, GATE_SUBJECT_KEY, subject,
        )
        return reset_gate_subject_required_error(subject)

    project_id = resolve_project_id(project_root)
    verdicts = await _corroborate_until_cited(
        memory_service, project_id, task_ids, timeout_seconds=timeout_seconds, log=log,
    )

    for task_id, verdict in verdicts.items():
        if verdict.citation is not None:
            log.warning(
                'preserved_specimen_gate_guard: refusing recon reset gate for '
                'gate_subject=%s in project %s — preserved specimen cited by %s',
                task_id, project_id, verdict.citation,
            )
            return preserved_specimen_reset_error(task_id, verdict.citation)

    unresolved = [task_id for task_id, verdict in verdicts.items() if verdict.degraded]
    if unresolved:
        log.warning(
            'preserved_specimen_gate_guard: corroboration of gate_subject=%s in '
            'project %s unresolved, submission proceeds',
            ','.join(unresolved), project_id,
        )
    return None
