#!/usr/bin/env python3
"""Retrospective read-only sweep for unverified completion claims in Graphiti.

Background
----------
Escalation ``esc-3085-1`` recorded two fabricated-completion records written in
a ~17-hour window on 2026-07-26/27:

1. An episode asserting that "task 5422's de-flake fix has been applied" while
   the task was still in-progress.
2. An episode asserting reify task 5638 "was reported unactionable and re-filed
   into dark_factory's task tree as ticket ``tkt_0RRRC5AASJ9Z630VP4PCN9H376``"
   — a ticket id that does not exist in the registry. Graphiti's extraction
   pipeline fanned that one sentence into a derived ``RELATES_TO`` edge
   (``7dbf12cf-9251-4674-b396-8eefdf651d1c``) asserting the same false fact.

Task 3142 landed :mod:`fused_memory.services.completion_claim_gate`, a
WRITE-TIME gate that catches this shape going forward. Every record written
before it landed passed through an ungated path. This script answers the
retrospective half: it re-runs that same detection vocabulary — IMPORTED, never
re-derived — over the historical Graphiti corpus and reports what a gate would
have caught.

Safety
------
This script has NO mutation path at all. There is no ``--apply``, no
``--invalidate``, no ``--delete``, no ``--repair``. It reads over
``GRAPH.RO_QUERY``, where read-only is SERVER-enforced rather than
client-promised, and it imports nothing that can write. Anything the report
indicts is left exactly as it is, for human adjudication — that is the task's
scope note ("read-only report first; do NOT auto-delete or auto-invalidate
edges on a regex verdict") and ``TestReadOnlyByConstruction`` in
``tests/test_audit_unverified_completion_claims.py`` enforces it mechanically
so a later editor cannot quietly relax it.

Usage
-----
Sweep the default project, listing only mismatches::

    uv run python scripts/audit_unverified_completion_claims.py

Sweep both trees that esc-3085-1 spanned, with the full unverifiable listing::

    uv run python scripts/audit_unverified_completion_claims.py \\
        --project dark_factory --project reify --include-unverifiable

Exit non-zero when any mismatch is found (for a CI/cron gate)::

    uv run python scripts/audit_unverified_completion_claims.py --fail-on-mismatch

Exit codes: ``0`` ran, ``1`` infra failure (nothing printed), ``2``
``--fail-on-mismatch`` with at least one mismatch.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from fused_memory.models.enums import MemoryCategory
from fused_memory.services.completion_claim_gate import (
    CompletionClaim,
    extract_completion_claims,
)

# --------------------------------------------------------------------------- #
# Category recovery from source_description
# --------------------------------------------------------------------------- #

_CATEGORY_PREFIXES: tuple[str, ...] = ('add_memory:', 'replay_from_mem0:')
"""The two in-tree episode writers that encode the category in the description.

``MemoryService`` writes ``add_memory:<category>`` on the classified-write path
(memory_service.py:2804) and ``replay_from_mem0:<category>`` on the Mem0 replay
path (:3302). Anything else is a caller-supplied ``add_episode`` description
and carries no category at all.
"""

_DESCRIPTION_PREFIX_RE: re.Pattern[str] = re.compile(
    r'^(?:\[temporal:[^\]]*\]|\[unverified_claim\])\s*'
)
"""One leading marker prefix, stripped repeatedly so any order/count works.

``graphiti_client`` prepends ``[temporal:<ctx>] `` when temporal_context is set
(:700) and ``[unverified_claim] `` when task 3142's write-time gate flagged the
episode (:702). Both can be present. Failing to strip the second would push
every already-flagged episode out of the swept population — exactly the records
this sweep most needs to read.
"""


def parse_category(source_description: str | None) -> str | None:
    """Return the memory category encoded in *source_description*, or None.

    Marker prefixes are stripped first, then a recognized writer prefix, and
    the remainder is returned only when it is a real
    :class:`~fused_memory.models.enums.MemoryCategory` member — so a typo or a
    renamed category yields ``None`` rather than minting a phantom bucket that
    would silently never match the in-scope filter.

    Pure: no I/O. Never raises on empty/odd input.
    """
    if not isinstance(source_description, str):
        return None
    text = source_description.strip()
    if not text:
        return None

    # Strip any leading run of marker prefixes, in any order.
    while True:
        stripped = _DESCRIPTION_PREFIX_RE.sub('', text, count=1)
        if stripped == text:
            break
        text = stripped.strip()

    for prefix in _CATEGORY_PREFIXES:
        if text.startswith(prefix):
            candidate = text[len(prefix):].strip()
            if candidate in tuple(MemoryCategory):
                return candidate
            return None
    return None


IN_SCOPE_CATEGORIES: frozenset[str] = frozenset({
    'temporal_facts',
    'decisions_and_rationale',
})
"""The two categories this task scopes the sweep to.

Both are GRAPHITI-primary (models/enums.py:36-40), which is exactly WHY the
label has to be recovered from ``source_description`` above rather than read
off a field: Graphiti persists no ``category`` property on Episodic nodes,
Entity nodes, or RELATES_TO edges, and ``MemoryService._search_graphiti``
hardcodes ``category=None`` (:3521) when it reads them back.
"""


# --------------------------------------------------------------------------- #
# The corpus projection + the scan (pure — no I/O, no store, no network)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """The closed projection of a swept record.

    Mirrors ``build_corpus.EpisodeRecord``'s discipline: the dataclass IS the
    statement of what this sweep is allowed to read. Nothing outside these
    fields is projected out of the store, so a reviewer can bound the read from
    the type alone.

    Attributes:
        uuid: The Graphiti node uuid.
        kind: Always ``'episode'`` — the sweep is episode-primary (only
            Episodic nodes carry the category this task scopes to).
        text: The episode ``content`` — the exact string the write-time gate
            consumes at server/tools.py:2793.
        source_description: The raw description, kept verbatim so a reader can
            re-derive ``category`` without re-running the sweep.
        category: The recovered category, or ``None`` when uncategorized.
        project_id: The graph the record was read from (Graphiti ``group_id``).
        created_at: Write timestamp, as the store reports it.
        name: The episode name.
    """

    uuid: str
    kind: str
    text: str
    source_description: str
    category: str | None
    project_id: str
    created_at: str
    name: str


@dataclass(frozen=True, slots=True)
class ScannedRecord:
    """A :class:`CorpusRecord` paired with the claims found in its text."""

    record: CorpusRecord
    claims: tuple[CompletionClaim, ...]


def scan_records(
    records: Iterable[CorpusRecord],
    *,
    default_project_id: str,
    known_project_ids: frozenset[str] | set[str],
    categories: frozenset[str] = IN_SCOPE_CATEGORIES,
) -> list[ScannedRecord]:
    """Run the IMPORTED extractor over every in-scope record that makes a claim.

    Records outside *categories* are skipped, and records yielding no claims are
    dropped entirely — this is a report of problems, so a record with nothing to
    say does not appear.

    The extractor comes from
    :mod:`fused_memory.services.completion_claim_gate`; this module contributes
    NO detection regex of its own. Re-deriving the negation/aspirational
    strippers would leave a third drifting copy of the one thing that keeps
    "has not yet landed" from reading as a completion (that module's docstring,
    :20-26).

    Returns results sorted by ``(category, created_at, uuid)`` so two runs over
    the same corpus produce byte-identical output regardless of the order the
    store happened to return rows in.

    Pure: no I/O, no store, no network.
    """
    scanned: list[ScannedRecord] = []
    for record in records:
        if record.category not in categories:
            continue
        claims = extract_completion_claims(
            record.text,
            default_project_id=record.project_id or default_project_id,
            known_project_ids=known_project_ids,
        )
        if not claims:
            continue
        scanned.append(ScannedRecord(record=record, claims=tuple(claims)))

    scanned.sort(
        key=lambda s: (s.record.category or '', s.record.created_at, s.record.uuid)
    )
    return scanned
