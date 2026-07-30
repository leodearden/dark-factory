#!/usr/bin/env python3
"""Sweep the Mem0 (Qdrant) corpus for leaked tool-call XML and repair it (task 3083).

Background
----------
The harness's tool-call XML parser terminates a string ARGUMENT early at a
literal closing tag inside that argument's value. These markers never originate
in this repository -- there is no XML parser anywhere in ``fused-memory/src/``
and no production write path emits a serialized tool-call fragment -- so their
presence in stored text is positive evidence of that harness defect. See
``fused_memory.utils.toolcall_xml_leak``, THE single source of truth for what a
leak looks like and this script's sole leak oracle.

The defect has two manifestations. Sibling-argument loss is silent and is now
refused at the write boundary by ``middleware/toolcall_xml_guard``. Content
self-duplication is the visible one, and it is already in the corpus: records
9f2d2ae6 and c759c53b are the recorded specimens. This script is how that
backlog gets found and fixed.

Until task 3083 there was no way to find them at all. ``search`` is SEMANTIC,
and a leaked XML tail carries almost no semantic signal;
``count_by_metadata``/``scroll_by_metadata`` match METADATA equality, not
payload text. The missing capability -- a literal payload-text scan -- is
``Mem0Backend.scan_payload_text`` / ``MemoryService.scan_memory_content``,
which this script consumes.

Repair is delete + re-add, never an in-place payload SET: repaired content must
be re-embedded, and an in-place write would leave a stale vector pointing at
the corrupted text.

Safety
------
Dry-run is the default: the printed JSON report IS the investigation. Only two
shapes are ever auto-repaired, and for both the repair preserves the memory TEXT
in full plus the payload metadata that is the record's only metadata-scoped
retrieval axis (carry-over rule: ``payload keys - _MEM0_OWNED_KEYS``, with the
scope identities threaded back as ``agent_id=``/``session_id=`` arguments and
anything carried nowhere NAMED per-record in ``metadata_dropped``):

  ``repairable_tail``       the fragment runs to end-of-content and carries
                            nothing after its own marker (the c759c53b shape);
  ``repairable_duplicate``  the text after the marker is a VERBATIM duplicate
                            of the text before it (the 9f2d2ae6 shape).

Anything else carrying a leak is ``manual_review`` and is NEVER mutated --
``repair_content`` returns None for it, so there is no repaired string to
write. This mirrors ``clear_malformed_empty_memory.is_malformed_empty_payload``'s
all-conditions-required fail-safe: the predicate is structurally incapable of
authorizing the destruction of real content.

``--apply`` on a corpus containing any ``manual_review`` record still repairs
the confidently-classified ones, but exits NON-ZERO so the operator cannot
mistake a partial sweep for a complete one. A truncated ``--apply`` (``--limit``
reached, or the scan otherwise capped) exits non-zero for the same reason: it
covered an unknown fraction of the corpus.

The delete/re-add is VERIFIED-PERSISTED, not assumed. ``MemoryService.add_memory``
swallows a Mem0 write failure into ``AddMemoryResponse.message`` as
``[mem0_error: ...]`` and returns NORMALLY, so a returned response is not by
itself evidence of a write. A non-raising add that did not persist is therefore
treated IDENTICALLY to a throw -- ``content_lost_in_flight``, a ``logger.error``,
``repaired`` left False, and a non-zero exit. And because a repaired record has
to land back in mem0 to replace what the delete removed, a record whose category
is not ``MEM0_PRIMARY`` is refused BEFORE the delete (``skipped_not_mem0_routed``,
also non-zero): a plain re-add would route it to Graphiti only, while
``dual_write=True`` would duplicate the Graphiti copy the mem0-scoped delete
deliberately left alive.

The printed report ALWAYS survives, because for a ``content_lost_in_flight``
record it is the only remaining copy of the original text. Each record is added
to the report BEFORE any store mutation is attempted and its repair runs under
its own ``try``, so one record's transport error is recorded as ``record_error``
on that record (non-zero exit; whether its delete landed is unknown) and the
sweep continues instead of unwinding. Should anything escape anyway, ``main``
owns the progress container and prints the PARTIAL report -- same shape, plus
``"aborted": true`` -- before exiting 2.

Scope
-----
Mem0/Qdrant ONLY. Graphiti episodes are NOT covered -- Graphiti-side discovery
is deliberately out of scope and filed as a follow-up; the residual episode
d12b0eb4 has its own content-preserving redaction path.

This script + its test suite are MOCK-unit only (AsyncMock service, no live
Qdrant), matching every precedent cleanup script in this directory. The live
``--apply --exhaustive`` run against real Qdrant is the operational close-out
that completes WORK (c).

RUN THIS SWEEP BEFORE ANY FURTHER LARGE CONSOLIDATION PASS
-----------------------------------------------------------
Routine consolidation deletes corrupted entries as a MERGE SIDE EFFECT, which
silently destroys the specimens before anyone can classify them. Both
2026-07-27 instances -- mem0 c759c53b and 9f2d2ae6 -- were lost exactly that
way. A consolidation pass run first does not just delay this sweep; it
destroys the evidence the sweep needs and makes the true incidence rate
unknowable.

``--exhaustive`` skips the server-side prefilter and paginates the entire
collection. That is the mode for the authoritative incidence-rate sweep: the
answer then rests on nothing but the shared Python detector, not on Qdrant's
un-indexed-field ``MatchText`` fallback semantics.

Usage
-----
  # Dry run (default): classify everything, mutate nothing.
  python scripts/sweep_toolcall_xml_leak.py

  # Authoritative incidence rate -- no prefilter, whole collection.
  python scripts/sweep_toolcall_xml_leak.py --exhaustive

  # Commit the repairs (refuses loudly, non-zero, if anything needs review).
  python scripts/sweep_toolcall_xml_leak.py --apply

  # Override the target project (default: dark_factory).
  python scripts/sweep_toolcall_xml_leak.py --project-id reify --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from fused_memory.models import MEM0_PRIMARY, MemoryCategory
from fused_memory.utils.toolcall_xml_leak import LEAK_TAIL

logger = logging.getLogger('sweep_toolcall_xml_leak')

CLEAN = 'clean'
REPAIRABLE_TAIL = 'repairable_tail'
REPAIRABLE_DUPLICATE = 'repairable_duplicate'
MANUAL_REVIEW = 'manual_review'

CLASSIFICATIONS = (CLEAN, REPAIRABLE_TAIL, REPAIRABLE_DUPLICATE, MANUAL_REVIEW)


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

# Payload keys tried in order when extracting a Mem0 memory's text content from
# its raw Qdrant payload dict. Reused verbatim from
# clear_malformed_empty_memory.py so "what counts as a memory's content" is
# judged identically to the existing sweeps ('data' is the canonical Qdrant
# scroll-payload content key for infer=False writes).
_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')


# Payload keys that mem0/Qdrant OWN or re-derive on write, and which must
# therefore never be forged back into an ``add_memory(metadata=...)`` call:
#   'id'                      the Qdrant point id, assigned by the new write;
#   'hash'                    mem0's content hash, recomputed from the content;
#   'data'/'memory'/'content' the memory text itself, passed as ``content=``;
#   'created_at'/'updated_at' mem0 timestamps, stamped by the new write;
#   'user_id'/'agent_id'/'run_id'/'actor_id'/'role'
#                             written from ``Scope`` (mem0_client.py:124-126),
#                             so they are threaded as ARGUMENTS instead;
#   'category'                overwritten by the service anyway
#                             (memory_service.py:2193 ``meta['category'] = ...``).
# No such constant already exists in the repo to reuse -- verified:
# ``_MEM0_CONTENT_KEYS`` (memory_service.py:96) covers only the three content
# keys. This set is therefore defined here deliberately, and must be kept in
# step with mem0's payload shape.
_MEM0_OWNED_KEYS: frozenset[str] = frozenset({
    'id', 'hash', 'data', 'memory', 'content', 'created_at', 'updated_at',
    'user_id', 'agent_id', 'run_id', 'actor_id', 'role', 'category',
})

# The subset of the owned keys that the repair DOES carry, just not as metadata:
# the content goes in as ``content=``, and these go in via ``Scope``. Reporting
# them as "dropped" would be a false alarm that buries the genuinely-lost ones.
_ARGUMENT_CARRIED_KEYS: frozenset[str] = frozenset({
    'data', 'memory', 'content', 'category', 'agent_id', 'run_id',
})


def carried_metadata(payload: dict, memory_id: Any) -> tuple[dict, list[str]]:
    """Return (metadata to re-add with, sorted keys carried nowhere). Pure.

    The carry-over rule is ``payload keys - _MEM0_OWNED_KEYS``, plus an
    ``x_repaired_from_memory_id`` provenance marker. It exists because the
    repair rebuilds the Qdrant point from scratch: anything not handed back to
    ``add_memory`` is gone. And the caller metadata is not decoration --
    ``get_memories_by_metadata``/``count_memories_by_metadata`` match payload
    KEYS by equality via ``MatchValue`` (mem0_client.py:315-359), so those keys
    are the repaired record's only metadata-scoped retrieval axis. Dropping
    them would make the memory invisible to every consumer that could
    previously find it.

    The second element names only what is carried NOWHERE -- the owned keys
    minus :data:`_ARGUMENT_CARRIED_KEYS` -- so the report flags real losses
    rather than burying them among keys that went back in as arguments.

    The caller's *payload* is never mutated.
    """
    metadata = {key: value for key, value in payload.items() if key not in _MEM0_OWNED_KEYS}
    metadata['x_repaired_from_memory_id'] = memory_id
    dropped = sorted(
        key for key in payload
        if key in _MEM0_OWNED_KEYS and key not in _ARGUMENT_CARRIED_KEYS
    )
    return metadata, dropped


def extract_content(payload: dict) -> str:
    """Return the first non-empty string among payload['data'], ['memory'],
    ['content'], trying _CONTENT_KEYS in order.

    Returns '' when no key is present or every present value is empty/not a
    string -- identical to clear_malformed_empty_memory.extract_content.
    """
    for key in _CONTENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def _split_at_leak(text: str) -> tuple[str, str] | None:
    """Split *text* into (before, remainder) around the leak marker.

    ``before`` is everything preceding the stray closing tag -- the text the
    caller actually meant to write. ``remainder`` is everything AFTER the
    continuation tag that follows it, which is what a repair would discard.

    Returns None when *text* carries no leak.

    The leak pattern itself is never re-spelled here: ``LEAK_TAIL`` from the
    shared module is the sole oracle, and this function only skips past the
    continuation tag's own closing ``>`` to find where the discarded remainder
    begins. That skip assumes a continuation tag contains no ``>`` inside its
    quoted attribute value -- true of every recorded specimen. A mis-split
    could only ever make ``remainder`` fail to match ``before``, which
    downgrades the record to ``manual_review``; it can never manufacture a
    false repair. Fail-safe in the only direction that matters.
    """
    stripped = text.rstrip()
    match = LEAK_TAIL.search(stripped)
    if match is None:
        return None
    before = stripped[: match.start()]
    continuation = match.group(1)
    tag_end = continuation.find('>')
    remainder = continuation[tag_end + 1 :] if tag_end != -1 else ''
    return before, remainder


def classify_record(payload: dict) -> str:
    """Classify a raw Qdrant payload into one of :data:`CLASSIFICATIONS`.

    Returns:
        'clean' when the extracted content carries no leak (including a
        payload with no extractable content at all -- nothing to repair);
        'repairable_tail' when the fragment carries nothing after its marker
        and dropping it leaves non-empty text;
        'repairable_duplicate' when the text after the marker is a VERBATIM
        duplicate of the text before it;
        'manual_review' for every other leak -- including a leak at offset 0,
        where no repair could preserve content because there is none to keep.
    """
    content = extract_content(payload)
    split = _split_at_leak(content)
    if split is None:
        return CLEAN

    before, remainder = split
    if not before:
        # Dropping the fragment would leave an empty memory: refuse.
        return MANUAL_REVIEW
    if not remainder.strip():
        return REPAIRABLE_TAIL
    if remainder == before:
        return REPAIRABLE_DUPLICATE
    return MANUAL_REVIEW


def repair_content(text: str | None) -> str | None:
    """Return the content-preserving repair of *text*, or None to REFUSE.

    None means exactly one thing: "do not mutate this record". It is returned
    for a leak this function cannot vouch for, and for a None input, so a
    refusal can never be mistaken for a repair.

    Clean text is returned UNCHANGED rather than as None. That is what makes
    the repair idempotent -- ``repair_content(repair_content(x)) ==
    repair_content(x)`` -- since a repaired string is itself clean.
    """
    if text is None:
        return None
    classification = classify_record({'data': text})
    if classification == CLEAN:
        return text
    if classification in (REPAIRABLE_TAIL, REPAIRABLE_DUPLICATE):
        split = _split_at_leak(text)
        assert split is not None  # noqa: S101 - guaranteed by the classification
        return split[0]
    return None


def readd_persisted(response: Any) -> tuple[bool, str]:
    """Return (True, '') only when *response* PROVES a mem0 copy was written.

    Pure, no I/O. This exists because a non-raising ``add_memory`` is NOT
    evidence of a write: ``MemoryService.add_memory`` catches ``Exception`` on
    the Mem0 path, logs it, sets ``_mem0_error``, and then returns a perfectly
    NORMAL ``AddMemoryResponse`` whose ``message`` carries
    ``[mem0_error: ...]``. Trusting the absence of an exception would therefore
    mark the record repaired AFTER the delete already removed the only copy --
    permanent content loss reported as a clean sweep.

    Three independent things must all hold, because each can fail alone:
      * ``memory_ids`` is non-empty  -- an empty result is mem0's silent
        dedup/infer no-op drop;
      * ``mem0`` appears in ``stores_written`` -- a Graphiti-only write has not
        restored the Qdrant copy the delete removed;
      * ``message`` carries no ``mem0_error`` -- the swallowed-failure marker.

    Stores are compared via ``getattr(s, 'value', s)`` so a real
    ``SourceStore`` member, a plain ``'mem0'`` string, and a test double all
    read identically.
    """
    memory_ids = getattr(response, 'memory_ids', None) or []
    if not memory_ids:
        return False, 'add_memory returned no memory_ids'

    stores = getattr(response, 'stores_written', None) or []
    if not any(getattr(store, 'value', store) == 'mem0' for store in stores):
        return False, 'add_memory did not report a mem0 write in stores_written'

    message = getattr(response, 'message', '') or ''
    if 'mem0_error' in message:
        return False, 'add_memory reported a swallowed mem0_error'

    return True, ''


def routes_to_mem0(payload: dict) -> tuple[bool, str]:
    """Return (True, '') only when this payload's category writes to mem0.

    Pure, no I/O. ``write_mem0 = resolved_category in MEM0_PRIMARY or
    dual_write`` in ``MemoryService.add_memory``, so re-adding a
    Graphiti-primary record after a mem0-scoped delete would put the repaired
    text in Graphiti ONLY and the Qdrant copy would simply be gone.
    ``dual_write=True`` is not a fix either: it would duplicate the Graphiti
    copy that ``delete_memory(store='mem0')`` deliberately left alive. Unsafe in
    both directions, so the sweep refuses to choose and leaves the record for a
    human.

    An absent or unrecognised category is refused rather than optimistically
    assumed mem0-routed -- the fail-safe direction is always "do not delete".
    """
    raw = payload.get('category')
    if not raw:
        return False, 'payload carries no category, so its store routing is unknown'
    try:
        category = MemoryCategory(raw)
    except ValueError:
        return False, f'payload category {raw!r} is not a recognised MemoryCategory'
    if category not in MEM0_PRIMARY:
        return False, (
            f'payload category {category.value!r} is not MEM0_PRIMARY, so a re-add '
            'would not restore the mem0 copy'
        )
    return True, ''


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_report(
    *,
    project_id: str,
    collection: str,
    dry_run: bool,
    exhaustive: bool,
    scanned: int,
    truncated: bool,
    limit: int | None,
    records: list[dict],
) -> dict:
    """Assemble the JSON-serializable report from already-computed fields.

    No I/O. ``records`` is carried verbatim so a dry run's printed report
    doubles as the investigation artifact the module docstring promises.
    ``scanned``/``truncated`` are reported explicitly so a capped run discloses
    its own incompleteness rather than presenting a partial count as the
    incidence rate.
    """
    counts = {name: 0 for name in CLASSIFICATIONS}
    for record in records:
        classification = record.get('classification')
        if classification in counts:
            counts[classification] += 1
    return {
        'project_id': project_id,
        'collection': collection,
        'dry_run': dry_run,
        'exhaustive': exhaustive,
        'scanned': scanned,
        'truncated': truncated,
        'limit': limit,
        'counts': counts,
        'repaired': sum(1 for record in records if record.get('repaired')),
        'records': records,
    }


def new_progress() -> dict:
    """Return a fresh caller-owned progress container for :func:`run`.

    The container exists so the report can OUTLIVE an abort. ``records`` is the
    live list :func:`run` appends to, so a caller holding this dict still has
    every record processed so far even if ``run`` raises partway through -- and
    for a ``content_lost_in_flight`` record, that in-memory text is the ONLY
    surviving copy of the original content (the point is already deleted). A
    report discarded on the way out would make that loss permanent.
    """
    return {'records': [], 'collection': '', 'scanned': 0, 'truncated': False}


def report_from_progress(args: Any, progress: dict) -> dict:
    """Build the report from whatever *progress* has accumulated so far.

    Used both for the normal return from :func:`run` and by :func:`main` on an
    abort, so a partial run is reported in EXACTLY the same shape as a complete
    one -- there is no second, lossier "emergency" format to get out of step.
    """
    return build_report(
        project_id=args.project_id,
        collection=progress.get('collection', ''),
        dry_run=not args.apply,
        exhaustive=bool(args.exhaustive),
        scanned=progress.get('scanned', 0),
        truncated=bool(progress.get('truncated')),
        limit=args.limit,
        records=progress.get('records', []),
    )


async def run(args: Any, memory_service: Any, progress: dict | None = None) -> dict:
    """Scan, classify, and (under ``--apply``) repair the leaked records.

    Discovery is ``MemoryService.scan_memory_content`` -- a LITERAL payload-text
    scan. Never ``search``: a leaked fragment carries almost no semantic signal,
    which is exactly why this corpus was unsweepable before task 3083. A scan
    ``TimeoutError`` propagates out of here rather than being reported as a
    clean corpus.

    Dry-run (``args.apply`` falsy) performs ZERO mutations. Under ``--apply``,
    only ``repairable_tail``/``repairable_duplicate`` records are touched, and
    the repair is delete-then-re-add so the corrected text is re-embedded --
    never an in-place Qdrant payload SET, which would leave a stale vector
    pointing at the corrupted string.

    Every repair is PRE-FLIGHT checked (:func:`routes_to_mem0` -- a record whose
    category would not be re-written to mem0 is never deleted, and is flagged
    ``skipped_not_mem0_routed``) and POSTCONDITION verified
    (:func:`readd_persisted` -- ``repaired`` is set only on proof of a mem0
    write).

    If the delete SUCCEEDS and the re-add does not persist -- whether it RAISED
    or merely returned a response that does not evidence a write -- the content
    is gone from the store and lives only in this report. Both are recorded
    identically and loudly (``content_lost_in_flight`` with both ids, the
    original content, the repaired content, and the reason) and never
    swallowed: a silent half-completed repair would destroy the very text this
    sweep exists to preserve.

    Two mechanisms keep that promise honest, and they are deliberately
    redundant:

      * Each record is appended to ``progress['records']`` BEFORE any mutation
        of the store is attempted, and its per-record body runs under its own
        ``try``. One record's transport error (a ``delete_memory`` timeout, a
        Qdrant outage) is recorded on THAT record as ``record_error`` and the
        sweep continues, instead of unwinding the whole run and taking every
        earlier record's report entry -- including any already-lost original
        text -- with it.
      * *progress* is caller-owned. Should anything escape anyway, the caller
        still holds every record accumulated so far and can print the partial
        report (see :func:`main`).
    """
    progress = new_progress() if progress is None else progress
    records: list[dict] = progress['records']
    dry_run = not args.apply

    scan = await memory_service.scan_memory_content(
        project_id=args.project_id,
        exhaustive=bool(args.exhaustive),
        limit=args.limit,
    )
    matches = scan.get('matches', [])
    progress['collection'] = scan.get('collection', '')
    progress['scanned'] = scan.get('scanned', len(matches))
    progress['truncated'] = bool(scan.get('truncated'))

    for match in matches:
        payload = match.get('metadata') or {}
        content = extract_content(payload)
        classification = classify_record(payload)
        record: dict = {
            'id': match.get('id'),
            'classification': classification,
            'repaired': False,
            'content': content,
            'created_at': match.get('created_at'),
        }
        # Appended BEFORE any store mutation, and mutated in place from here
        # on, so this record is in the report no matter how its repair ends.
        records.append(record)

        if classification in (REPAIRABLE_TAIL, REPAIRABLE_DUPLICATE):
            record['repaired_content'] = repair_content(content)

        try:
            if not dry_run and classification in (REPAIRABLE_TAIL, REPAIRABLE_DUPLICATE):
                await _repair_record(memory_service, args, match, payload, record)
            elif not dry_run and classification == MANUAL_REVIEW:
                logger.warning(
                    'sweep_toolcall_xml_leak: refusing to repair memory_id=%s -- the '
                    'text after the leak marker is neither absent nor a verbatim '
                    'duplicate, so no repair preserves content. Left untouched for '
                    'human review.',
                    match.get('id'),
                )
        except Exception as exc:  # noqa: BLE001 - one bad record must not void the run
            record['record_error'] = True
            record['error'] = str(exc)
            logger.exception(
                'sweep_toolcall_xml_leak: memory_id=%s failed mid-repair (%s). '
                'Recorded on this record and the sweep continues -- aborting here '
                'would discard the report that is the only copy of any '
                'already-lost content.',
                match.get('id'), exc,
            )

    return report_from_progress(args, progress)


async def _repair_record(
    memory_service: Any, args: Any, match: dict, payload: dict, record: dict
) -> None:
    """Delete the corrupted point, then re-add the repaired text.

    Mutates *record* in place with the outcome. The delete comes FIRST: a
    re-add first would leave both copies live if the delete then failed, which
    is a duplicate rather than a repair.

    PRE-FLIGHT, before anything is deleted: :func:`routes_to_mem0` must vouch
    for the payload's category. A record that would not be re-written to mem0
    is skipped entirely -- no delete, no add -- and flagged
    ``skipped_not_mem0_routed`` for a human.

    POSTCONDITION, after the re-add: :func:`readd_persisted` must vouch for the
    response. ``repaired=True`` is set ONLY then. A non-raising ``add_memory``
    that did not actually persist is treated IDENTICALLY to a throw -- same
    ``content_lost_in_flight`` flag, same ``logger.error`` -- because the
    consequence is identical: the delete landed and the text now lives only in
    this report.

    The repair is content-preserving in a precise sense: it preserves the memory
    TEXT, and it preserves the payload metadata that is the record's only
    metadata-scoped retrieval axis (:func:`carried_metadata`). The scope-carried
    identities go back in as ARGUMENTS -- ``agent_id=`` and ``session_id=``,
    since mem0 writes its ``run_id`` payload key from ``Scope.session_id``
    (mem0_client.py:124-126) -- rather than being forged as metadata. Any key it
    cannot carry is NAMED in ``metadata_dropped`` on the report rather than
    dropped silently.
    """
    memory_id = match.get('id')
    repaired = record.get('repaired_content')

    routed, routing_reason = routes_to_mem0(payload)
    if not routed:
        record['skipped_not_mem0_routed'] = True
        record['error'] = routing_reason
        logger.warning(
            'sweep_toolcall_xml_leak: refusing to repair memory_id=%s -- %s. '
            'Neither a plain re-add (mem0 copy lost) nor dual_write=True '
            '(duplicate Graphiti copy) is safe here, so the record is left '
            'entirely untouched for human review.',
            memory_id, routing_reason,
        )
        return

    metadata, dropped = carried_metadata(payload, memory_id)
    # Recorded BEFORE the delete, so the audit survives even a lost-in-flight
    # repair -- that is precisely when a hand restore needs it.
    record['metadata_preserved'] = sorted(set(payload) - _MEM0_OWNED_KEYS)
    record['metadata_dropped'] = dropped

    await memory_service.delete_memory(
        memory_id=memory_id,
        store='mem0',
        project_id=args.project_id,
    )
    try:
        response = await memory_service.add_memory(
            content=repaired,
            category=payload.get('category'),
            project_id=args.project_id,
            agent_id=payload.get('agent_id'),
            session_id=payload.get('run_id'),
            metadata=metadata,
        )
    except Exception as exc:
        _report_content_lost(record, memory_id, str(exc))
        return

    new_ids = getattr(response, 'memory_ids', None) or []
    # Record whatever ids came back BEFORE judging the response, so a human
    # chasing a loss has every handle this run ever saw.
    record['new_id'] = new_ids[0] if new_ids else None

    persisted, reason = readd_persisted(response)
    if not persisted:
        message = getattr(response, 'message', '') or ''
        _report_content_lost(record, memory_id, f'{reason} (add_memory message: {message})')
        return

    record['repaired'] = True


def _report_content_lost(record: dict, memory_id: Any, error: str) -> None:
    """Flag *record* as content-lost-in-flight and say so as loudly as possible.

    Shared by the raising and the non-raising-but-unpersisted branches of
    :func:`_repair_record`, because the two are the SAME failure: the point is
    already deleted, so the original text now exists only in this report.
    ``repaired`` is deliberately left False.
    """
    record['content_lost_in_flight'] = True
    record['error'] = error
    logger.error(
        'sweep_toolcall_xml_leak: CONTENT LOST IN FLIGHT for memory_id=%s -- '
        'the corrupted point was deleted but the repaired re-add did not persist '
        '(%s). The original and repaired text are in this run report; restore by '
        'hand. No further records were harmed.',
        memory_id, error,
    )


def resolve_exit_code(report: dict) -> int:
    """Exit-code predicate: 0 for any dry run, 1 for an incomplete ``--apply``.

    A dry run mutates nothing and is never a failure, however much it finds.
    An ``--apply`` exits non-zero when it left ``manual_review`` records behind
    or was truncated -- in both cases it covered less than the whole corpus,
    and a zero exit would let a partial sweep read as a complete one. Pure,
    sync, no I/O.

    Three per-record outcomes also force non-zero, all needing a human:
    ``content_lost_in_flight`` (the delete landed but the re-add did not
    persist, so the text survives only in the printed report),
    ``skipped_not_mem0_routed`` (a repairable record whose category does not
    route to mem0, left entirely untouched), and ``record_error`` (the record's
    repair aborted on an unexpected error -- the sweep carried on, but whether
    that record's delete landed is UNKNOWN, which is precisely the state a
    human has to adjudicate).
    """
    if report['dry_run']:
        return 0
    if report.get('counts', {}).get(MANUAL_REVIEW, 0) > 0:
        return 1
    if report.get('truncated'):
        return 1
    for record in report.get('records', []):
        if (
            record.get('content_lost_in_flight')
            or record.get('skipped_not_mem0_routed')
            or record.get('record_error')
        ):
            return 1
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build this script's argparse parser.

    Factored out of :func:`main` (mirrors clear_malformed_empty_memory.py) so
    the CLI surface is testable without any live I/O.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Scan the Mem0 (Qdrant) corpus for leaked tool-call XML and, with '
            '--apply, repair the two shapes that can be fixed without losing '
            'content.'
        ),
    )
    parser.add_argument(
        '--project-id', dest='project_id', default='dark_factory',
        help='Project scope to sweep (default: dark_factory).',
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help=(
            'Commit the repairs for confidently-classified records (default: '
            'dry-run only, prints the classification report and exits 0).'
        ),
    )
    parser.add_argument(
        '--exhaustive', action='store_true', default=False,
        help=(
            'Skip the server-side prefilter and paginate the entire '
            'collection. Slower, but the result then depends on nothing but '
            'the shared Python detector -- use this for the authoritative '
            'incidence rate.'
        ),
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help=(
            'Stop after inspecting this many records. A truncated --apply '
            'exits non-zero, since it covered an unknown fraction of the '
            'corpus.'
        ),
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    # Owned HERE, not inside run(), so the report survives an abort. For any
    # record already flagged content_lost_in_flight, the printed report is the
    # ONLY surviving copy of the original text -- discarding it on the way out
    # would turn a recoverable half-repair into permanent, silent data loss.
    progress = new_progress()

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(args, memory, progress)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    exit_code = 0
    try:
        report = asyncio.run(_run_live())
    except Exception:
        report = report_from_progress(args, progress)
        report['aborted'] = True
        logger.exception(
            'sweep_toolcall_xml_leak: fatal error during sweep -- emitting the '
            'PARTIAL report for the %d record(s) already processed. Any record '
            'flagged content_lost_in_flight has its original text here and '
            'NOWHERE else; restore it from this output before re-running.',
            len(report['records']),
        )
        exit_code = 2

    print(json.dumps(report, indent=2, default=str))
    if exit_code:
        return exit_code
    if not args.apply:
        logger.info('Dry run -- nothing was modified. Use --apply to commit the repairs.')
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
