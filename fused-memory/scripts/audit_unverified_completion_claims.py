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

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.models.enums import MemoryCategory
from fused_memory.services.completion_claim_gate import (
    UNRESOLVABLE,
    ClaimVerdict,
    CompletionClaim,
    extract_completion_claims,
    verify_claims,
)

# UNRESOLVABLE is re-exported deliberately. The ticket probe below returns THIS
# sentinel object — identity-compared by completion_claim_gate._verify_ticket —
# when the registry cannot be consulted at all, which is what keeps a missing
# tickets.db from being reported as a fabrication accusation. Tests bind it from
# this module so they assert against the same object the sweep actually passes.
__all__ = ['UNRESOLVABLE']

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
        project_id: The record's OWN Graphiti ``group_id`` — what the record
            claims about itself.
        graph_name: WHICH GRAPH WAS QUERIED to read this record (the
            ``--project`` value / :attr:`EpisodeReader.graph_name`). Distinct
            from ``project_id``: they coincide for most rows, and a graph
            legitimately holds episodes whose ``group_id`` differs. Only
            ``graph_name`` can address the store again, so every follow-up read
            (the derived-edge enumeration in ``_run``) must key on THIS field —
            keying on ``project_id`` silently resolves to no reader and the miss
            then serializes as a measured zero.
        created_at: Write timestamp, as the store reports it.
        name: The episode name.
    """

    uuid: str
    kind: str
    text: str
    source_description: str
    category: str | None
    project_id: str
    graph_name: str
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


# --------------------------------------------------------------------------- #
# Adjudication (pure and sync — the three authorities are INJECTED probes)
# --------------------------------------------------------------------------- #

#: Verdict statuses, ordered so a mismatch always sorts ahead of an
#: unverifiable. A mismatch is evidence the writer was WRONG; an unverifiable is
#: merely unchecked. Putting the evidence at the head is what makes the part
#: worth reading the part that gets read.
_STATUS_ORDER: dict[str, int] = {'mismatch': 0, 'unverifiable': 1}


@dataclass(frozen=True, slots=True)
class Finding:
    """One non-verified claim, with everything needed to re-check it by hand.

    ``derived_edge_uuids`` names the Graphiti ``RELATES_TO`` edges extracted
    FROM this episode — the actual harm artefacts. In esc-5603-1 one false
    sentence fanned out into five false edges, so a report that named only the
    source episode would understate what is actually in the graph. It defaults
    to ``()`` because the pure layer cannot reach the store; ``_run`` attaches
    the real uuids for flagged episodes.

    ``None`` is a THIRD state, distinct from ``()``: NOT ENUMERATED. It
    serializes to JSON ``null``, which no consumer can misread as a measured
    zero, and it is what ``_run`` records when no reader resolved for the
    finding's graph or the edge query raised. ``()`` keeps its one meaning —
    queried, and no edges found. ``edges_unqueried`` carries the same fact as a
    boolean so the condition is greppable in the committed artifact without
    inferring it from a null.
    """

    record_uuid: str
    record_kind: str
    category: str | None
    project_id: str
    graph_name: str
    created_at: str
    claim_kind: str
    subject: str
    ref: str
    claim_project_id: str | None
    status: str
    observed: str
    claimed_text: str
    derived_edge_uuids: tuple[str, ...] | None = ()
    edges_unqueried: bool = False

    def to_json(self) -> dict[str, Any]:
        """Render as a plain JSON-serializable dict."""
        return {
            'record_uuid': self.record_uuid,
            'record_kind': self.record_kind,
            'category': self.category,
            'project_id': self.project_id,
            'graph_name': self.graph_name,
            'created_at': self.created_at,
            'claim_kind': self.claim_kind,
            'subject': self.subject,
            'ref': self.ref,
            'claim_project_id': self.claim_project_id,
            'status': self.status,
            'observed': self.observed,
            'claimed_text': self.claimed_text,
            # None stays None — it renders as JSON `null`, the whole point.
            'derived_edge_uuids': (
                None if self.derived_edge_uuids is None
                else list(self.derived_edge_uuids)
            ),
            'edges_unqueried': self.edges_unqueried,
        }


def adjudicate(
    scanned: Iterable[ScannedRecord],
    *,
    task_status_probe: Callable[[str, str | None], str | None],
    ticket_probe: Callable[[str], dict[str, Any] | None | Any],
    commit_probe: Callable[[str, str | None], bool | None],
) -> list[Finding]:
    """Adjudicate every scanned claim, returning only the non-verified ones.

    The verdicts come from the IMPORTED
    :func:`~fused_memory.services.completion_claim_gate.verify_claims`, with the
    three tri-state probes passed through unchanged — so this sweep and the
    write-time gate cannot disagree about what a given authority answer means.

    Verified claims are DROPPED: this is a report of problems, and listing the
    clean majority would bury the part worth reading.

    Each finding slices the claiming clause out of the record's own text by the
    claim span, the same self-containment
    :func:`~fused_memory.services.completion_claim_gate.build_unverified_flag`
    provides (:623-639) — a reader can see what was claimed, what was checked
    and what was actually there without re-running anything.

    Pure and sync. Probes are injected, so no Taskmaster, no ticket DB and no
    git are required to exercise this layer.
    """
    findings: list[Finding] = []
    for item in scanned:
        verdicts: list[ClaimVerdict] = verify_claims(
            list(item.claims),
            task_status_probe=task_status_probe,
            ticket_probe=ticket_probe,
            commit_probe=commit_probe,
        )
        for verdict in verdicts:
            if verdict.status == 'verified':
                continue
            start, end = verdict.claim.span
            findings.append(
                Finding(
                    record_uuid=item.record.uuid,
                    record_kind=item.record.kind,
                    category=item.record.category,
                    project_id=item.record.project_id,
                    graph_name=item.record.graph_name,
                    created_at=item.record.created_at,
                    claim_kind=verdict.claim.kind,
                    subject=verdict.claim.subject,
                    ref=verdict.claim.ref,
                    claim_project_id=verdict.claim.project_id,
                    status=verdict.status,
                    observed=verdict.observed,
                    claimed_text=item.record.text[start:end].strip(),
                )
            )

    findings.sort(
        key=lambda f: (
            _STATUS_ORDER.get(f.status, 99),
            f.category or '',
            f.created_at,
            f.record_uuid,
            f.ref,
        )
    )
    return findings


# --------------------------------------------------------------------------- #
# The report (pure — swept_at is passed in, never read from the wall clock)
# --------------------------------------------------------------------------- #

CAVEATS: tuple[str, ...] = (
    'RETROSPECTIVE BIAS, FALSE NEGATIVES: the write-time gate reads each '
    'authority AT WRITE TIME; this sweep reads it TODAY. A claim written while '
    'its task was still in-progress VERIFIES today if that task has since gone '
    'terminal, so the sweep systematically under-counts. A low mismatch count '
    'is therefore NOT evidence the corpus is clean.',
    "RETROSPECTIVE BIAS, FALSE POSITIVES: 'cancelled' is in "
    'shared.task_statuses.TERMINAL, so a claim about work that was later '
    'CANCELLED reads as verified here even though the work never landed.',
    'UNVERIFIABLE INDICTS NOBODY: a task deleted since, a garbage-collected '
    'commit, or an expired ticket all read as unverifiable through no fault of '
    "the writer. 'unverifiable' and 'mismatch' are different facts and are "
    'never summed in this report.',
    'COVERAGE, STORE: this sweep covers GRAPHITI EPISODES only. A '
    'MEM0-resident record is NOT covered, because the two categories in scope '
    'are Graphiti-primary and Mem0 is not read at all.',
    'COVERAGE, ORPHANED EDGES: a derived RELATES_TO edge whose source episode '
    'has been deleted is NOT covered, and neither is an episode ingested '
    'through add_episode with a caller-supplied description, which carries no '
    'category and so falls outside the category scope.',
    'COVERAGE, CROSS-GRAPH EDGES: derived_edge_uuids is enumerated ONLY in the '
    'graph the episode was READ FROM (the finding\'s graph_name). An episode '
    "whose group_id differs from that graph — see any finding where those two "
    'fields disagree — may have its derived RELATES_TO edges in ANOTHER graph, '
    'where they are NOT counted here. Measured, not hypothetical: episode '
    'a887c958-0018-4715-8817-cf048c187e8d exists only in the reify graph with '
    'group_id dark_factory, and its 8 derived edges exist only in the '
    'dark_factory graph. An empty list therefore means "none in THIS graph", '
    'not "none anywhere".',
    'EDGES NOT ENUMERATED: a finding whose derived_edge_uuids is null (and '
    'whose edges_unqueried is true) had its RELATES_TO edges NOT enumerated — '
    'no reader resolved for its graph, or the edge query failed. Its harm '
    'artefacts are UNKNOWN, not absent. An empty list means the opposite: the '
    'store WAS asked and returned nothing.',
    'DETECTION BOUND: claims are detected by the deterministic lexical '
    'vocabulary in fused_memory.services.completion_claim_gate, which requires '
    'completion PHRASING and a concrete NAMED REF (task id / commit sha / tkt_ '
    'id) to co-occur in one clause. A fabricated completion phrased without a '
    'named ref is invisible to this sweep by construction.',
)
"""The report's caveats, as DATA rather than docstring prose.

They travel with the committed artifact into investigation.md and into whatever
reads it next, which is why they are a first-class report key rather than a
comment here.

Deliberately NOT pinned word-for-word by a test. A test asserting
``report['caveats'] == CAVEATS`` is tautological — ``build_report`` returns this
very tuple — and substring-matching the prose pins cosmetic phrasing: rewording
a caveat for clarity would fail, while gutting one's meaning and keeping the
matched word would pass. The behavioural fact (the report ships a non-empty
caveats key) is asserted; the wording is left free to improve.
"""


def _tally(findings: Iterable[Finding], key: Callable[[Finding], str]) -> dict[str, int]:
    """Count *findings* by *key*, returned in sorted key order.

    Sorted, not insertion-ordered: the report must be byte-stable across runs,
    and dict insertion order here would follow whatever order the store
    returned rows in.
    """
    counts: dict[str, int] = {}
    for finding in findings:
        counts[key(finding)] = counts.get(key(finding), 0) + 1
    return dict(sorted(counts.items()))


def build_report(
    findings: Iterable[Finding],
    *,
    swept_at: str,
    scanned_count: int,
    records_with_claims: int,
    projects: list[str],
    categories: list[str],
    include_unverifiable: bool,
) -> dict[str, Any]:
    """Build the machine-readable report.

    VOLUME CONTROL WITHOUT A SILENT CAP. The write-time gate tags on
    unverifiable because a false positive costs one ``source_description``
    prefix. A batch report has inverted economics: across thousands of
    historical episodes the unverifiable bucket dominates for reasons that
    indict nobody — tasks purged, commits gc'd, tickets expired — and a report
    that lists all of it inline is unread, which is the failure mode that let
    the original incident sit unnoticed. So only mismatches are LISTED by
    default, but BOTH buckets are unconditionally COUNTED and ``truncated_by``
    names the flag that reveals the rest. A bounded listing must say what it
    dropped, or it reads as coverage it does not have.

    ``mismatch`` and ``unverifiable`` are NEVER summed into one headline
    number. They are different facts, and
    :mod:`fused_memory.services.completion_claim_gate` makes the same
    distinction load-bearing at the sentinel level (:123-127) — a report that
    collapsed it would undo that.

    Pure: *swept_at* is an argument rather than a ``datetime.now()`` call, which
    is what makes the output byte-stable across two runs on the same input.
    """
    findings = list(findings)
    mismatches = [f for f in findings if f.status == 'mismatch']
    unverifiables = [f for f in findings if f.status == 'unverifiable']

    listed = findings if include_unverifiable else mismatches
    truncated_by: dict[str, Any] | None = None
    if not include_unverifiable and unverifiables:
        truncated_by = {
            'status': 'unverifiable',
            'withheld': len(unverifiables),
            'flag': '--include-unverifiable',
            'note': (
                f'{len(unverifiables)} unverifiable finding(s) were COUNTED in '
                f'summary.unverifiable but not listed. Re-run with '
                f'--include-unverifiable to see them.'
            ),
        }

    return {
        'swept_at': swept_at,
        'projects': list(projects),
        'categories': list(categories),
        'scanned': scanned_count,
        'records_with_claims': records_with_claims,
        'summary': {
            'mismatch': len(mismatches),
            'unverifiable': len(unverifiables),
            # Always present, 0 when none: a key that appeared only when
            # nonzero would read as absent rather than as zero. Same
            # no-silent-caps discipline as `truncated_by` — a reader sees the
            # denominator of un-enumerated findings without diffing the list.
            'edges_unqueried': sum(1 for f in findings if f.edges_unqueried),
            'by_category': _tally(findings, lambda f: f.category or '(uncategorized)'),
            'by_project': _tally(findings, lambda f: f.project_id),
            'by_subject': _tally(findings, lambda f: f.subject),
        },
        'truncated_by': truncated_by,
        'caveats': list(CAVEATS),
        'findings': [f.to_json() for f in listed],
    }


# --------------------------------------------------------------------------- #
# The read seam — GRAPH.RO_QUERY, no graphiti driver
# --------------------------------------------------------------------------- #

DEFAULT_PROJECT = 'dark_factory'
"""The graph swept when --project is not given."""

RO_COMMAND = 'GRAPH.RO_QUERY'
"""The only FalkorDB command this script is permitted to issue.

Read-only here is **server-enforced**, not client-promised: build_corpus.py
(:667-675) records that a ``CREATE`` issued through this command path was
refused by a live FalkorDB and materialized no graph. For an audit whose entire
premise is "do not mutate on a regex verdict", that is the difference between a
promise and a guarantee — which is why this script reads over Cypher rather
than through ``MemoryService`` / ``GraphitiBackend``, both of which reach the
store through a handle that CAN write.

Two secondary reasons. ``MemoryService.initialize()`` brings up Qdrant and an
embedder this sweep does not need, and graphiti_core's ``FalkorDriver.__init__``
fire-and-forgets ``build_indices_and_constraints()`` — a WRITE. And
``MemoryService.get_episodes`` is unusable regardless: it drops
``source_description`` from its projection (memory_service.py:3838-3843), which
is the ONLY carrier of the category this task scopes to, and the MCP layer
clamps ``last_n`` to 1000 against a population measured at 2976 (dark_factory)
and 4547 (reify) rows.
"""

PROJECTED_FIELDS: tuple[str, ...] = (
    'uuid',
    'name',
    'group_id',
    'source_description',
    'created_at',
    'content',
)
"""The closed set of episode fields this sweep ever reads.

Load-bearing, not documentation: :data:`EPISODE_CYPHER` builds its RETURN list
from this tuple and :class:`CorpusRecord` mirrors it field for field, so the
query and the record shape cannot drift apart.
"""

EPISODE_CYPHER = 'MATCH (e:Episodic) RETURN ' + ', '.join(
    f'e.{field}' for field in PROJECTED_FIELDS
)
"""The population query, derived from :data:`PROJECTED_FIELDS`.

No ``WHERE``: the category filter runs in Python over the recovered label,
because the category is not a stored property and so cannot be filtered in
Cypher at all. No ``LIMIT`` by default — a truncated population would make
every denominator in the report wrong.
"""

DERIVED_EDGE_CYPHER = (
    'MATCH ()-[r:RELATES_TO]->() WHERE $episode_uuid IN r.episodes '
    'RETURN r.uuid, r.fact, r.valid_at, r.invalid_at'
)
"""The RELATES_TO edges extracted FROM a given episode — the harm artefacts.

``r.episodes`` is a real persisted uuid list, not merely a declared one: the
projection at ``fused_memory/maintenance/cross_graph_move.py:672`` is the
in-tree precedent, and it was confirmed populated in both live graphs before
this script was written. This is the join that yields the esc-3085-1
instance-(2) shape — episode ``02090224-7bc9-4485-9291-6748e1042ac9`` ->
edge ``7dbf12cf-9251-4674-b396-8eefdf651d1c``.
"""


class EpisodeReader:
    """Reads the ``Episodic`` population of one graph over ``GRAPH.RO_QUERY``.

    Constructed either with an explicit *graph* handle (the tests pass a
    double) or with a uri + graph name, in which case a ``falkordb.asyncio``
    client is opened lazily on first use.

    ``falkordb.FalkorDB`` — the DB client — is categorically distinct from
    ``graphiti_core.driver.falkordb_driver.FalkorDriver``, the graphiti driver
    whose ``__init__`` fire-and-forgets ``build_indices_and_constraints()``.
    This class must never construct the latter: that would make a script whose
    entire premise is "read-only" issue a write before it read anything.
    """

    def __init__(
        self,
        *,
        graph: Any | None = None,
        graph_name: str = DEFAULT_PROJECT,
        uri: str | None = None,
    ) -> None:
        self._graph = graph
        self.graph_name = graph_name
        self.uri = uri

    @staticmethod
    def assert_read_only_command(command: str) -> None:
        """Raise unless *command* is :data:`RO_COMMAND`.

        A client-side guard layered on the server-side one. The server would
        refuse a write anyway; this makes the violation a typed error at the
        seam that owns the guarantee, rather than a redis error surfacing three
        layers down where a reader would not connect it to this claim.
        """
        if command != RO_COMMAND:
            raise RuntimeError(
                f'this sweep may only issue {RO_COMMAND}, refused {command!r} '
                f'— it is strictly read-only'
            )

    def _resolve_graph(self) -> Any:
        """Return the graph handle, opening a client on first use.

        The import is local so that merely importing this module — which the
        tests do, with a double in hand — never requires falkordb to be
        installed or a store to be running.
        """
        if self._graph is None:
            from falkordb.asyncio import FalkorDB  # noqa: PLC0415

            client = (
                FalkorDB.from_url(self.uri) if self.uri else FalkorDB()
            )
            self._graph = client.select_graph(self.graph_name)
        return self._graph

    @staticmethod
    def _cell(row: list[Any], index: int) -> str:
        """Read column *index* defensively.

        A FalkorDB ``RETURN e.<prop>`` over a node lacking that property yields
        NULL, which lands here as ``None``. Degrading to ``''`` keeps one odd
        historical row from aborting a sweep whose whole job is to survey odd
        historical rows; the empty value is visible in the report either way.
        """
        if index >= len(row):
            return ''
        value = row[index]
        return '' if value is None else str(value)

    async def fetch_population(self, *, limit: int | None = None) -> list[CorpusRecord]:
        """Return every ``Episodic`` node in this graph, projected.

        One query, no ``WHERE`` — the category filter cannot run in Cypher
        because the category is not a stored property, so it runs in Python
        over the label recovered from ``source_description``.
        """
        self.assert_read_only_command(RO_COMMAND)
        query = EPISODE_CYPHER
        if limit is not None:
            query = f'{query} LIMIT {int(limit)}'
        graph = self._resolve_graph()
        response = await graph.ro_query(query)
        rows = getattr(response, 'result_set', response) or []
        population: list[CorpusRecord] = []
        for row in rows:
            source_description = self._cell(row, 3)
            population.append(
                CorpusRecord(
                    uuid=self._cell(row, 0),
                    kind='episode',
                    text=self._cell(row, 5),
                    source_description=source_description,
                    category=parse_category(source_description),
                    project_id=self._cell(row, 2) or self.graph_name,
                    graph_name=self.graph_name,
                    created_at=self._cell(row, 4),
                    name=self._cell(row, 1),
                )
            )
        return population

    async def fetch_derived_edges(self, episode_uuid: str) -> tuple[str, ...]:
        """Return the uuids of RELATES_TO edges extracted FROM *episode_uuid*.

        These are the actual harm artefacts: in esc-5603-1 one false sentence
        fanned out into five false edges, so naming only the source episode
        would understate what is in the graph.
        """
        self.assert_read_only_command(RO_COMMAND)
        graph = self._resolve_graph()
        response = await graph.ro_query(
            DERIVED_EDGE_CYPHER, {'episode_uuid': episode_uuid}
        )
        rows = getattr(response, 'result_set', response) or []
        return tuple(self._cell(row, 0) for row in rows)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Factored out (as audit_duplicate_memories.py:2796 does) precisely so
    ``TestReadOnlyByConstruction`` can enumerate ``parser._actions`` and assert
    that NO mutation affordance exists. There is deliberately no ``--apply``,
    no ``--invalidate``, no ``--delete`` and no ``--repair``: this script
    reports, and a human adjudicates.
    """
    parser = argparse.ArgumentParser(
        prog='audit_unverified_completion_claims',
        description=(
            'Read-only retrospective sweep for unverified completion claims in '
            'the Graphiti episode corpus (task 3381, esc-3085-1). Reports only '
            '— it never mutates the graph.'
        ),
    )
    parser.add_argument(
        '--project',
        action='append',
        metavar='GRAPH',
        help=(
            'Project graph to sweep. Repeatable — esc-3085-1 spanned two trees '
            f'(reify agents claiming about dark_factory work). Default: '
            f'{DEFAULT_PROJECT}.'
        ),
    )
    parser.add_argument(
        '--uri',
        default=None,
        help='FalkorDB URI (redis://host:port). Default: read from --config.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to fused-memory config.yaml. Default: the packaged config.',
    )
    parser.add_argument(
        '--categories',
        default=','.join(sorted(IN_SCOPE_CATEGORIES)),
        help=(
            'Comma-separated categories to sweep. Default: the two this task '
            'scopes to.'
        ),
    )
    parser.add_argument(
        '--include-unverifiable',
        action='store_true',
        help=(
            'List unverifiable findings too. They are always COUNTED in the '
            'summary; this flag adds them to the findings listing.'
        ),
    )
    parser.add_argument(
        '--fail-on-mismatch',
        action='store_true',
        help='Exit 2 when at least one mismatch is found (for a CI/cron gate).',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help=(
            'Cap the episodes read per graph. Diagnostics only — a capped run '
            "makes the report's denominators unrepresentative, and the cap is "
            'recorded in the report when set.'
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

logger = logging.getLogger('audit_unverified_completion_claims')


@dataclass(frozen=True, slots=True)
class Probes:
    """The three authority probes, resolved EAGERLY and passed as one facade.

    Injected exactly as ``audit_found_on_main_provenance`` injects its
    ``GitFacts`` facade, which is what lets ``_run`` be exercised end-to-end
    with no Taskmaster, no ticket DB and no git.
    """

    task_status_probe: Callable[[str, str | None], str | None]
    ticket_probe: Callable[[str], dict[str, Any] | None | Any]
    commit_probe: Callable[[str, str | None], bool | None]


def _known_projects(args: argparse.Namespace) -> list[str]:
    """The graphs to sweep, defaulting to :data:`DEFAULT_PROJECT`."""
    return list(args.project) if args.project else [DEFAULT_PROJECT]


def _load_config(config_path: str | None = None) -> Any:
    """Load a :class:`FusedMemoryConfig`.

    It is a pydantic ``BaseSettings``, so the bare constructor IS the loader
    (server/main.py:521, maintenance/_utils.py:79) — there is no ``.load()``
    or ``.from_yaml()`` classmethod. A ``--config`` path is applied through
    ``CONFIG_PATH`` — the env var ``settings_customise_sources`` already reads
    (config/schema.py:2047) — so this script does not invent a second
    config-resolution path that could drift from the server's.
    """
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if config_path:
        os.environ['CONFIG_PATH'] = str(config_path)
    return FusedMemoryConfig()


def _resolve_uri(args: argparse.Namespace) -> str | None:
    """Resolve the FalkorDB URI from --uri, then the env, then the config."""
    if args.uri:
        return str(args.uri)
    env_uri = os.environ.get('FALKORDB_URI')
    if env_uri:
        return env_uri
    try:
        config = _load_config(args.config)
        return getattr(getattr(config.graphiti, 'falkordb', None), 'uri', None)
    except Exception:
        logger.warning(
            'could not resolve a FalkorDB uri from config; falling back to the '
            'client default', exc_info=True,
        )
        return None


async def _build_ticket_probe(
    db_path: Path, refs: Iterable[str]
) -> Callable[[str], dict[str, Any] | None | Any]:
    """Build the ticket probe, probing store AVAILABILITY exactly once.

    THE DELIBERATE DIVERGENCE from the live write-path gate.
    ``TaskInterceptor.get_ticket_row`` returns ``None`` BOTH for "no such
    ticket" and for "no ticket store configured" (task_interceptor.py:
    3006-3012), and ``completion_claim_gate._verify_ticket`` maps a ``None``
    row to ``'mismatch'`` (:575). On the write path that conflation is
    contained upstream by the ``_taskmaster_configured`` guard and costs at
    most one spurious tag on one episode. In a BATCH sweep the same conflation
    is a different animal: if tickets.db is merely absent or unopenable, every
    ticket claim in the entire corpus prints as a fabrication accusation
    against a named agent.

    So availability is established ONCE, up front, and unavailability yields
    :data:`UNRESOLVABLE` for every ref — never ``None``. The gate's own module
    makes this distinction load-bearing at the sentinel level (:123-127,
    INV-2), so honouring it here follows its stated intent rather than
    departing from it.
    """
    rows: dict[str, Any] = {}
    available = False
    if not db_path.exists():
        logger.warning(
            'no ticket store at %s; every ticket claim is UNVERIFIABLE '
            '(never a mismatch)', db_path,
        )
    else:
        try:
            from fused_memory.middleware.ticket_store import TicketStore  # noqa: PLC0415

            store = TicketStore(db_path)
            await store.initialize()
            available = True
            for ref in dict.fromkeys(refs):
                try:
                    # A globally unique PK lookup over the one shared
                    # tickets.db, so it needs no project and answers a
                    # cross-project claim correctly. A key present-but-None is
                    # the genuine "no such ticket" (a mismatch); an ABSENT key
                    # stays unresolvable.
                    rows[ref] = await store.get(ref)
                except Exception:
                    logger.warning(
                        'ticket lookup failed for %r; UNVERIFIABLE', ref,
                        exc_info=True,
                    )
        except Exception:
            logger.warning(
                'ticket store at %s could not be opened; every ticket claim is '
                'UNVERIFIABLE (never a mismatch)', db_path, exc_info=True,
            )
            available = False

    def probe(ref: str) -> dict[str, Any] | None | Any:
        if not available:
            return UNRESOLVABLE
        if ref in rows:
            return rows[ref]
        return UNRESOLVABLE

    return probe


async def _build_task_status_probe(
    project_roots: dict[str, str], refs_by_project: dict[str | None, set[str]]
) -> Callable[[str, str | None], str | None]:
    """Build the task-status probe from ONE batched read per claimed project.

    Batched rather than one read per claim (which would multiply authority
    traffic by the claim count for no gain), and scoped to the CLAIMED project
    rather than the writer's (a sha or task id claimed for dark_factory is not
    answered by reify's tree). An absent key is the unresolvable signal.
    """
    resolved: dict[tuple[str | None, str], str] = {}
    for project, refs in refs_by_project.items():
        root = project_roots.get(project) if project else None
        if not root:
            logger.warning(
                'project %r is not registered; %d task claim(s) are UNVERIFIABLE',
                project, len(refs),
            )
            continue
        try:
            from fused_memory.backends.sqlite_task_backend import (  # noqa: PLC0415
                SqliteTaskBackend,
            )

            # project_root is a METHOD argument, not a constructor kwarg, and
            # get_statuses is a coroutine (backends/sqlite_task_backend.py:1662).
            backend = SqliteTaskBackend()
            statuses = await backend.get_statuses(root, ids=sorted(refs))
        except Exception:
            logger.warning(
                'get_statuses failed for project %r; %d task claim(s) are '
                'UNVERIFIABLE', project, len(refs), exc_info=True,
            )
            continue
        for key, value in (statuses or {}).items():
            if isinstance(value, str):
                resolved[(project, str(key))] = value

    def probe(ref: str, project_id: str | None) -> str | None:
        return resolved.get((project_id, ref))

    return probe


def _build_commit_probe(
    project_roots: dict[str, str],
) -> Callable[[str, str | None], bool | None]:
    """Build the commit probe over the IMPORTED :func:`make_commit_probe`.

    Rooted at the CLAIMED project's repository. An unregistered project is
    unresolvable, never a miss — reporting "no such commit" because the wrong
    repo was searched would be a false accusation.
    """
    from fused_memory.services.completion_claim_gate import (  # noqa: PLC0415
        make_commit_probe,
    )

    cache: dict[str, Callable[[str], bool | None]] = {}

    def probe(ref: str, project_id: str | None) -> bool | None:
        root = project_roots.get(project_id) if project_id else None
        if not root:
            return None
        if project_id not in cache:
            try:
                cache[str(project_id)] = make_commit_probe(root)
            except Exception:
                logger.warning(
                    'could not build a commit probe for %r; UNVERIFIABLE',
                    project_id, exc_info=True,
                )
                return None
        try:
            return cache[str(project_id)](ref)
        except Exception:
            logger.warning('commit probe failed for %r; UNVERIFIABLE', ref)
            return None

    return probe


def _default_project_roots() -> dict[str, str]:
    """Map project_id -> project_root via the server's OWN registry builder.

    :func:`build_known_projects_map` (models/scope.py:282) is the same helper
    the server uses to populate the ``known_projects`` map it hands the live
    gate, and it defaults its extra roots from ``known_project_roots_from_env``
    — which is what lets one sweep answer claims about BOTH dark_factory and
    reify. Re-deriving the mapping here would risk answering a cross-project
    claim against the wrong tree.
    """
    try:
        from fused_memory.models.scope import (  # noqa: PLC0415
            build_known_projects_map,
        )

        primary = os.environ.get('PROJECT_ROOT') or str(
            Path(__file__).resolve().parent.parent.parent
        )
        return build_known_projects_map(primary)
    except Exception:
        logger.warning('could not load the project registry', exc_info=True)
        return {}


async def _run(
    args: argparse.Namespace,
    *,
    reader_factory: Callable[[str], Any] | None = None,
    probes: Probes | None = None,
) -> int:
    """Sweep every requested graph and print one JSON report to stdout.

    Exit codes: ``0`` ran, ``1`` infra failure (NOTHING is printed — a
    truncated report that looks complete is worse than none), ``2``
    ``--fail-on-mismatch`` and at least one mismatch was found.
    """
    if probes is None or reader_factory is None:
        logging.basicConfig(
            level=logging.INFO, format='%(levelname)s %(name)s: %(message)s',
            stream=sys.stderr,
        )

    projects = _known_projects(args)
    categories = frozenset(
        c.strip() for c in str(args.categories).split(',') if c.strip()
    )
    known_project_ids = frozenset(projects)

    # ---- read (may fail; nothing is printed if it does) -------------------- #
    try:
        make_reader: Callable[[str], Any]
        if reader_factory is not None:
            make_reader = reader_factory
        else:
            uri = _resolve_uri(args)

            def _default_reader(project: str) -> Any:
                return EpisodeReader(graph_name=project, uri=uri)

            make_reader = _default_reader

        readers: dict[str, Any] = {p: make_reader(p) for p in projects}
        population: list[CorpusRecord] = []
        for project, reader in readers.items():
            records = await reader.fetch_population(limit=args.limit)
            logger.info('%s: read %d episode(s)', project, len(records))
            population.extend(records)
    except Exception:
        logger.error(
            'could not read the episode population — no report is emitted '
            'rather than a partial one that would read as complete',
            exc_info=True,
        )
        return 1

    scanned = scan_records(
        population,
        default_project_id=projects[0],
        known_project_ids=known_project_ids,
        categories=categories,
    )

    # ---- resolve the three authorities ONCE, eagerly, batched -------------- #
    if probes is None:
        project_roots = _default_project_roots()
        refs_by_project: dict[str | None, set[str]] = {}
        ticket_refs: list[str] = []
        for item in scanned:
            for claim in item.claims:
                if claim.subject == 'task':
                    refs_by_project.setdefault(claim.project_id, set()).add(claim.ref)
                elif claim.subject == 'ticket':
                    ticket_refs.append(claim.ref)
        probes = Probes(
            task_status_probe=await _build_task_status_probe(
                project_roots, refs_by_project
            ),
            ticket_probe=await _build_ticket_probe(
                Path(_default_ticket_db_path()), ticket_refs
            ),
            commit_probe=_build_commit_probe(project_roots),
        )

    findings = adjudicate(
        scanned,
        task_status_probe=probes.task_status_probe,
        ticket_probe=probes.ticket_probe,
        commit_probe=probes.commit_probe,
    )

    # ---- attach the harm artefacts to each flagged episode ----------------- #
    # None in this cache means NOT ENUMERATED, and is carried through to JSON
    # `null`. () means queried-and-empty. Collapsing the two would make an
    # un-run query indistinguishable from a measured zero on the report's
    # central harm-artefact column.
    edge_cache: dict[tuple[str, str], tuple[str, ...] | None] = {}
    enriched: list[Finding] = []
    for finding in findings:
        # Keyed on graph_name, NOT project_id: `readers` is keyed by --project
        # graph name, and a graph legitimately holds episodes whose group_id
        # differs. The uuid alone is not a safe key either — two graphs may hold
        # the same uuid, so the pair is what makes the cache correct.
        key = (finding.graph_name, finding.record_uuid)
        if key not in edge_cache:
            reader = readers.get(finding.graph_name)
            if reader is None:
                logger.warning(
                    'no reader for graph %s — derived edges NOT enumerated for '
                    'episode %s; its harm artefacts are UNKNOWN, not absent',
                    finding.graph_name, finding.record_uuid,
                )
                edge_cache[key] = None
            else:
                try:
                    edge_cache[key] = await reader.fetch_derived_edges(
                        finding.record_uuid
                    )
                except Exception:
                    logger.warning(
                        'could not enumerate derived edges for episode %s',
                        finding.record_uuid, exc_info=True,
                    )
                    edge_cache[key] = None
        edges = edge_cache[key]
        enriched.append(
            Finding(**{
                **vars_of(finding),
                'derived_edge_uuids': edges,
                'edges_unqueried': edges is None,
            })
        )

    report = build_report(
        enriched,
        swept_at=datetime.now(UTC).isoformat(),
        scanned_count=len(population),
        records_with_claims=len(scanned),
        projects=projects,
        categories=sorted(categories),
        include_unverifiable=bool(args.include_unverifiable),
    )
    if args.limit is not None:
        report['limit'] = args.limit

    print(json.dumps(report, indent=2, default=str))

    if args.fail_on_mismatch and report['summary']['mismatch']:
        return 2
    return 0


def vars_of(finding: Finding) -> dict[str, Any]:
    """Field dict for a slots dataclass (``vars()`` does not work on slots)."""
    return {
        field: getattr(finding, field)
        for field in Finding.__dataclass_fields__  # type: ignore[attr-defined]
    }


def _default_ticket_db_path() -> str:
    """The shared tickets.db path, best effort.

    Missing is a NORMAL outcome here, not an error: it lands every ticket claim
    on UNVERIFIABLE via :func:`_build_ticket_probe`.
    """
    try:
        config = _load_config()
        recon = getattr(config, 'reconciliation', None)
        data_dir = getattr(recon, 'data_dir', None) if recon else None
        # tickets.db is a sibling of reconciliation.db inside the recon data
        # dir — the construction at server/main.py:1546.
        return str(Path(data_dir or './data') / 'tickets.db')
    except Exception:
        logger.warning('could not resolve the ticket db path', exc_info=True)
    return str(Path('./data') / 'tickets.db')


def main() -> int:
    """Entry point: parse argv and run the sweep."""
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
