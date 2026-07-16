#!/usr/bin/env python3
"""Clear a malformed/empty Mem0 (Qdrant) memory record by id (task 2691).

Background
----------
A malformed/empty Mem0 record (category=observations_and_summaries claimed by
the recon MCP surface, but with a null-payload category, null agent_id, and
no extractable content -- e.g. 028edb1f-299c-4755-9432-f5a9fde97677) recurs
unresolved across reconciliation cycles. The recon MCP surface (cite_memory)
exposes only a metadata fingerprint, not the raw Qdrant payload, so deletion
cannot be confirmed-safe from within the pipeline.

This script reaches the raw Qdrant transport directly via
``MemoryService.mem0._get_async_qdrant()`` (the same admin path
``consolidate_namespace_families.run`` uses), retrieves the point by id
(``AsyncQdrantClient.retrieve``), CONFIRMS the malformed fingerprint with a
pure fail-safe predicate (empty content AND null category AND null agent_id
-- ALL three, see ``is_malformed_empty_payload``), and only then, under
``--apply``, deletes it via a direct Qdrant point delete (``client.delete``
with ``PointIdsList``).

It deliberately does NOT use Mem0's ``AsyncMemory.delete()`` /
``MemoryService.delete_memory`` -- a null-payload record breaks Mem0's delete
path (it reads ``existing.payload['data']`` to build a history entry ->
``KeyError``; the exact task-86 null-payload failure mode).

Safety
------
Dry-run is the default: the printed JSON report (raw payload + classification)
IS the investigation. ``--apply`` against a record that does NOT match the
malformed fingerprint (a 'healthy' record) is refused LOUDLY (WARNING +
non-zero exit) -- honoring the loud-over-silent-degradation norm, so a
mistyped ``--memory-id`` can never delete a real memory. An absent record
(already deleted) is idempotent success (exit 0), so the recurring finding
self-heals once this tool is run against an already-cleared id.

The tool is parameterized (``--memory-id``/``--project-id``, id NOT
hardcoded) so the recurring malformed-empty-record class (precedent tasks
86/54/61/98) is handled by re-running against a different id. Collection
resolves to ``fused_dark_factory`` via ``Scope('dark_factory')
.mem0_collection_name('fused')`` (``collection_prefix`` default ``'fused'``).

Scope: this script + its test suite are MOCK-unit only (AsyncMock Qdrant
client, MagicMock points) -- no live Qdrant. The live
``--apply --memory-id 028edb1f-...`` run against live Qdrant is the
operational close-out (needs live Qdrant + OPENAI env), documented here --
mirroring every precedent cleanup script (mock-unit only; operator runs
``--apply``).

Usage
-----
  # Dry run (default): print the raw payload + classification, touch nothing.
  python scripts/clear_malformed_empty_memory.py \\
      --memory-id 028edb1f-299c-4755-9432-f5a9fde97677

  # Commit the deletion (only proceeds if the record matches the malformed
  # fingerprint; refuses loudly otherwise).
  python scripts/clear_malformed_empty_memory.py \\
      --memory-id 028edb1f-299c-4755-9432-f5a9fde97677 --apply

  # Override the target project (default: dark_factory).
  python scripts/clear_malformed_empty_memory.py \\
      --memory-id <id> --project-id reify --apply
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('clear_malformed_empty_memory')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

# Payload keys tried in order when extracting a Mem0 memory's text content
# from its raw Qdrant payload dict. Reused verbatim from
# audit_duplicate_memories.py's _CONTENT_KEYS so 'empty content' is judged
# identically to how the dedup sweep judges it ('data' is the canonical
# Qdrant scroll-payload content key for infer=False writes).
_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')


def extract_content(payload: dict) -> str:
    """Return the first non-empty string among payload['data'], ['memory'],
    ['content'], trying _CONTENT_KEYS in order.

    Returns '' when no key is present or every present value is empty/not a
    string -- mirrors audit_duplicate_memories.fetch_procedural_memories's
    content-extraction fallback exactly.
    """
    for key in _CONTENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def is_malformed_empty_payload(payload: dict) -> bool:
    """True iff *payload* matches the malformed-empty fingerprint: empty
    extracted content AND category is None AND agent_id is None -- ALL
    three conditions required.

    Requiring all three means a healthy record (non-empty content, or a set
    category, or a set agent_id) can NEVER match, so this predicate
    structurally cannot authorize deleting a real memory.
    """
    return (
        extract_content(payload) == ''
        and payload.get('category') is None
        and payload.get('agent_id') is None
    )


def classify_payload(payload: dict | None) -> str:
    """Classify a retrieved Qdrant payload (or None) for the report/gate.

    Returns:
        'absent' if *payload* is None (record not found by retrieve);
        'malformed' if *payload* matches is_malformed_empty_payload;
        'healthy' otherwise.
    """
    if payload is None:
        return 'absent'
    if is_malformed_empty_payload(payload):
        return 'malformed'
    return 'healthy'


# ---------------------------------------------------------------------------
# Thin async I/O
# ---------------------------------------------------------------------------

async def retrieve_payload(qdrant_client, collection: str, memory_id: str) -> dict | None:
    """Retrieve the raw Qdrant payload for *memory_id* in *collection*.

    Calls ``qdrant_client.retrieve(collection_name=collection,
    ids=[memory_id], with_payload=True, with_vectors=False)`` -- vectors are
    never needed here (this script only inspects/deletes, never re-upserts).

    Returns:
        The first returned record's payload as a plain dict, or None when
        retrieve returns an empty list (the record does not exist -- already
        deleted, or never existed).
    """
    records = await qdrant_client.retrieve(
        collection_name=collection,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    return dict(records[0].payload or {})


async def delete_point(qdrant_client, collection: str, memory_id: str) -> None:
    """Delete the *memory_id* point from *collection* via a direct Qdrant
    point delete (PointIdsList) -- NOT Mem0's AsyncMemory.delete(), which
    would KeyError on a null-payload record (module docstring)."""
    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    await qdrant_client.delete(
        collection_name=collection,
        points_selector=qmodels.PointIdsList(points=[memory_id]),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_report(
    *,
    memory_id: str,
    collection: str,
    classification: str,
    dry_run: bool,
    deleted: bool,
    payload: dict | None,
) -> dict:
    """Assemble the JSON-serializable report dict from already-computed
    fields. No I/O.

    ``payload`` is included verbatim (including None for an absent record)
    so the printed report doubles as the investigation artifact the module
    docstring promises for a dry run.
    """
    return {
        'memory_id': memory_id,
        'collection': collection,
        'classification': classification,
        'dry_run': dry_run,
        'deleted': deleted,
        'payload': payload,
    }


def resolve_exit_code(report: dict) -> int:
    """Exit-code predicate: 1 iff the record was 'healthy' (not the
    malformed fingerprint) and this was NOT a dry run -- the loud refusal
    of a mistargeted --apply. Every other case (malformed, absent, or any
    dry run) exits 0. Pure, sync, no I/O."""
    if report['classification'] == 'healthy' and not report['dry_run']:
        return 1
    return 0


async def run(args: Any, qdrant_client: Any, collection_name: str) -> dict:
    """Retrieve, classify, and (under --apply) delete the fingerprint-
    confirmed malformed record at args.memory_id.

    Dry-run (``args.apply`` falsy) performs ZERO mutations: the returned
    report is a preview only. Under ``--apply``:
      - classification == 'malformed': delete_point is called, deleted=True.
      - classification == 'healthy': delete is NEVER called -- a WARNING is
        logged refusing the delete (fail-safe; see module docstring).
      - classification == 'absent': no delete is attempted (nothing to
        delete) -- an INFO log notes the idempotent no-op.
    """
    payload = await retrieve_payload(qdrant_client, collection_name, args.memory_id)
    classification = classify_payload(payload)
    dry_run = not args.apply
    deleted = False

    if args.apply and classification == 'malformed':
        await delete_point(qdrant_client, collection_name, args.memory_id)
        deleted = True
    elif args.apply and classification == 'healthy':
        logger.warning(
            "clear_malformed_empty_memory: refusing to delete memory_id=%s in "
            "collection=%s -- payload does NOT match the malformed-empty "
            "fingerprint (classification='healthy'). No mutation performed.",
            args.memory_id, collection_name,
        )
    elif classification == 'absent':
        logger.info(
            'clear_malformed_empty_memory: memory_id=%s in collection=%s not '
            'found (already absent) -- idempotent no-op.',
            args.memory_id, collection_name,
        )

    return build_report(
        memory_id=args.memory_id,
        collection=collection_name,
        classification=classification,
        dry_run=dry_run,
        deleted=deleted,
        payload=payload,
    )
