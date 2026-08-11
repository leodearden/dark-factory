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
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from fused_memory.models.enums import MemoryCategory
from fused_memory.services.completion_claim_gate import (
    UNRESOLVABLE,
    ClaimVerdict,
    CompletionClaim,
    extract_completion_claims,
    verify_claims,
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
    """

    record_uuid: str
    record_kind: str
    category: str | None
    project_id: str
    created_at: str
    claim_kind: str
    subject: str
    ref: str
    claim_project_id: str | None
    status: str
    observed: str
    claimed_text: str
    derived_edge_uuids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Render as a plain JSON-serializable dict."""
        return {
            'record_uuid': self.record_uuid,
            'record_kind': self.record_kind,
            'category': self.category,
            'project_id': self.project_id,
            'created_at': self.created_at,
            'claim_kind': self.claim_kind,
            'subject': self.subject,
            'ref': self.ref,
            'claim_project_id': self.claim_project_id,
            'status': self.status,
            'observed': self.observed,
            'claimed_text': self.claimed_text,
            'derived_edge_uuids': list(self.derived_edge_uuids),
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
    'DETECTION BOUND: claims are detected by the deterministic lexical '
    'vocabulary in fused_memory.services.completion_claim_gate, which requires '
    'completion PHRASING and a concrete NAMED REF (task id / commit sha / tkt_ '
    'id) to co-occur in one clause. A fabricated completion phrased without a '
    'named ref is invisible to this sweep by construction.',
)
"""The report's caveats, as DATA rather than docstring prose.

They travel with the committed artifact into investigation.md and into whatever
reads it next, and a test pins this literal tuple so a later editor cannot
quietly delete one while leaving the report looking authoritative.
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


def main() -> int:
    """Entry point: parse argv and run the sweep."""
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
