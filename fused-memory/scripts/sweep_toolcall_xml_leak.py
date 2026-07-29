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
shapes are ever auto-repaired, and both are recoverable WITHOUT losing a single
character of real content:

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

Scope
-----
Mem0/Qdrant ONLY. Graphiti episodes are NOT covered -- Graphiti-side discovery
is deliberately out of scope and filed as a follow-up; the residual episode
d12b0eb4 has its own content-preserving redaction path.

This script + its test suite are MOCK-unit only (AsyncMock service, no live
Qdrant), matching every precedent cleanup script in this directory. The live
``--apply`` run against real Qdrant is the operational close-out.

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
import json
import logging
import sys

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


def resolve_exit_code(report: dict) -> int:
    """Exit-code predicate: 0 for any dry run, 1 for an incomplete ``--apply``.

    A dry run mutates nothing and is never a failure, however much it finds.
    An ``--apply`` exits non-zero when it left ``manual_review`` records behind
    or was truncated -- in both cases it covered less than the whole corpus,
    and a zero exit would let a partial sweep read as a complete one. Pure,
    sync, no I/O.
    """
    if report['dry_run']:
        return 0
    if report.get('counts', {}).get(MANUAL_REVIEW, 0) > 0:
        return 1
    if report.get('truncated'):
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


def main() -> int:  # pragma: no cover - live wiring lands with the run() step
    raise NotImplementedError(
        'sweep_toolcall_xml_leak: the live orchestration lands in the next step.'
    )


if __name__ == '__main__':
    sys.exit(main())
