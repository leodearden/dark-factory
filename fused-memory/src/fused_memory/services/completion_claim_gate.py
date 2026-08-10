"""Completion-claim verification gate for episode ingestion (task 3142).

Reify escalation ``esc-5603-1`` recorded the motivating incident: an agent
wrote an episode asserting that a fix "has been applied" for a task that was
still in-progress, and Graphiti's extraction pipeline fanned that single
sentence out into FIVE false edges asserting completed work. ``esc-3085-1``
extended the scope — the same failure shape occurs for *filing/dispatch*
claims ("re-filed ... as ticket ``tkt_...``") naming a ticket that does not
exist, and it occurs ACROSS projects (a reify-authored claim about a
dark_factory ticket).

This module is the code-level enforcement of the "Terminal-State Pre-Check
Discipline" that until now existed only as prompt text for reconciliation
agents (:mod:`fused_memory.reconciliation.prompts.stage1`) — prose an agent
can simply not follow. It is PRD leaf pi / contract C4 of
``docs/prds/memory-write-path-convergence.md``.

Shape
-----
Detection is DETERMINISTIC lexical matching (word-boundary alternation over
closed vocabularies), never fuzzy or LLM classification — same discipline as
every sibling detector in :mod:`fused_memory.reconciliation.task_filter`,
whose regex vocabulary this module IMPORTS rather than re-derives. Re-deriving
the negation/aspirational strippers would leave a second, drifting copy of the
one thing that keeps "has not yet landed" and "will land" from false-firing.

A claim is only a claim when completion PHRASING and a concrete NAMED REF
(task id / commit sha / ``tkt_`` id) co-occur in the same clause. Bare "the
fix was applied" yields nothing: the task scopes detection to claims "that
name a task or commit", and requiring the ref is also the volume control —
an unanchored detector would tag a large fraction of ordinary agent narration,
and a tag that fires constantly stops being read.

Verification is split from detection behind INJECTED probes (mirroring
:func:`middleware.recon_claim_verification_guard.verify_attributed_claims`),
so the acceptance criterion is unit-testable with no Taskmaster, no ticket DB
and no git.

This module imports nothing from :mod:`fused_memory.server` — ``server/tools``
imports IT, and a cycle would be fatal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from fused_memory.reconciliation.task_filter import (
    _CLAUSE_SPLIT_RE,
    FUTURE_ASPIRATIONAL_RE,
    NEGATED_TERMINAL_RE,
    TASK_REF_RE,
)

logger = logging.getLogger(__name__)

__all__ = [
    'APPLIED_WORK_RE',
    'CompletionClaim',
    'extract_completion_claims',
]

ClaimKind = Literal['applied_work', 'filing_dispatch']
ClaimSubject = Literal['task', 'commit', 'ticket']


# --------------------------------------------------------------------------- #
# Phrasing vocabularies
# --------------------------------------------------------------------------- #
#
# APPLIED_WORK_RE is written in the same COPULA-GUARDED shape as
# task_filter.PRESENT_TENSE_COMPLETION_RE, and for the same reason (task 2824
# review): a transitive-capable verb placed in the bare arm false-fires on an
# ordinary object-taking sentence. "task N applied the patch to the wrong
# branch" and "task N patched over the symptom" are NOT completion claims about
# task N's own work, so `applied`/`patched` match only after a copula
# ("has been applied"). `landed`/`merged`/`shipped` are intransitive in this
# register and are safe bare, exactly as task_filter has them.
#
# THREE verbs here (`applied`/`patched`/`deployed`) are NOT in task_filter's
# completion vocabulary, so the imported NEGATED_TERMINAL_RE and
# FUTURE_ASPIRATIONAL_RE strippers have no arm for them — and an unstripped
# verb reopens exactly the negation hole those strippers exist to close ("the
# fix has not been applied" would extract as a completion claim). The two
# supplementary strippers below cover precisely that delta. INVARIANT: every
# verb added to _APPLIED_PARTICIPLES must also be reachable from
# _APPLIED_ANY_FORM, which is the strippers' (deliberately broader) vocabulary.
_APPLIED_PARTICIPLES: str = r'applied|patched|deployed'
_APPLIED_ANY_FORM: str = (
    r'appl(?:y|ied|ies|ying)|patch(?:ed|es|ing)?|deploy(?:ed|s|ing)?'
)

APPLIED_WORK_RE: re.Pattern[str] = re.compile(
    r'\b(?:'
    # bare past-tense/participle completion words that read as a completion of
    # the anchored work even without a copula ("task N landed").
    r'landed|merged|shipped|'
    # copula/auxiliary + (been/already/now/fully)* + completion word
    r'(?:is|are|was|were|has|have|had|been)\s+(?:been\s+|already\s+|now\s+|fully\s+)*'
    r'(?:landed|merged|shipped|' + _APPLIED_PARTICIPLES + r')'
    r')\b',
    re.IGNORECASE,
)

# Supplementary strippers: the delta vocabulary above, in the SAME shape as the
# task_filter originals (each swallows the verb it governs, so removing the span
# removes the completion evidence). The modal/filler prefixes are copied from
# FUTURE_ASPIRATIONAL_RE deliberately — a narrower prefix here would leave
# "task N's fix is supposed to be applied next week" reading as a completion.
_NEGATED_APPLIED_RE: re.Pattern[str] = re.compile(
    r"\b(?:not|never|hasn't|has\s+not|yet\s+to\s+be)\s+(?:yet\s+|been\s+)*"
    r'(?:' + _APPLIED_ANY_FORM + r')\b',
    re.IGNORECASE,
)
_ASPIRATIONAL_APPLIED_RE: re.Pattern[str] = re.compile(
    r'\b(?:will|going\s+to|plans?\s+to|planned\s+to|intends?\s+to|intended\s+to|'
    r'aims?\s+to|meant\s+to|hopes?\s+to|expects?\s+to|scheduled\s+to|slated\s+to|'
    r'supposed\s+to|needs?\s+to|to\s+be|should|would|shall)\b'
    r'(?:\s+(?:be|been|get|soon|also|now|already|just|then|finally|'
    r'eventually|not|yet|still)){0,3}'
    r'\s+(?:' + _APPLIED_ANY_FORM + r')\b',
    re.IGNORECASE,
)


def _iter_clauses(text: str):
    """Yield ``(clause, start_offset)`` for each clause of *text*.

    Same boundaries as ``task_filter._CLAUSE_SPLIT_RE.split`` ('.', ';',
    newline, '!', '?'), but offset-preserving: a claim's span has to point back
    into the ORIGINAL text so the flag can quote what was claimed (INV-2).
    Empty clauses are skipped, matching the sibling detectors.
    """
    pos = 0
    for match in _CLAUSE_SPLIT_RE.finditer(text):
        if match.start() > pos:
            yield text[pos:match.start()], pos
        pos = match.end()
    if pos < len(text):
        yield text[pos:], pos


def _strip_exemptions(clause: str) -> str:
    """Remove negated-terminal and future/aspirational spans from *clause*.

    Both stripper regexes deliberately swallow the completion verb they govern,
    so removing their spans removes the completion EVIDENCE — that is what
    makes "has not yet landed" and "will land" produce no claim. Substituting a
    space (rather than deleting) preserves word boundaries; offsets are not
    preserved, which is why refs are extracted from the ORIGINAL clause.
    """
    de_exempted = NEGATED_TERMINAL_RE.sub(' ', clause)
    de_exempted = _NEGATED_APPLIED_RE.sub(' ', de_exempted)
    de_exempted = FUTURE_ASPIRATIONAL_RE.sub(' ', de_exempted)
    return _ASPIRATIONAL_APPLIED_RE.sub(' ', de_exempted)


@dataclass(frozen=True, slots=True)
class CompletionClaim:
    """One extracted claim that some concrete, named thing is complete.

    Attributes:
        kind: ``'applied_work'`` (the work itself landed) or
            ``'filing_dispatch'`` (the work was filed/queued/cancelled
            somewhere) — the esc-3085-1 scope extension.
        subject: Which authority adjudicates it — ``'task'``, ``'commit'`` or
            ``'ticket'``.
        ref: The named reference, as written: a task id, a commit sha, or a
            ``tkt_`` id.
        project_id: The project whose registry answers a ``task`` claim.
            ``None`` for a ``ticket`` claim, which resolves by globally unique
            primary key and therefore needs no project at all.
        span: ``(start, end)`` offsets of the claiming clause in the original
            text.
    """

    kind: ClaimKind
    subject: ClaimSubject
    ref: str
    project_id: str | None
    span: tuple[int, int]


def extract_completion_claims(
    text: str,
    *,
    default_project_id: str,
    known_project_ids: frozenset[str] | set[str],
) -> list[CompletionClaim]:
    """Extract every completion claim in *text* that names a concrete ref.

    A clause produces claims only when BOTH a phrasing hit survives
    negation/aspirational stripping AND that same clause names a concrete ref.
    Claims are returned in text order, deduplicated on
    ``(kind, subject, ref, project_id)`` so a repeated assertion costs one
    authority read rather than several.

    Args:
        text: The episode content being ingested.
        default_project_id: The WRITING agent's project — the fallback owner of
            an unqualified task ref.
        known_project_ids: The server's project registry. A project qualifier
            adjacent to a ref is honoured only when it is a member here, which
            is what stops an arbitrary preceding word from being mistaken for a
            project name.

    Pure: no I/O, no side effects. Never raises on empty/odd input.
    """
    if not text:
        return []

    claims: list[CompletionClaim] = []
    seen: set[tuple[str, str, str, str | None]] = set()

    for clause, offset in _iter_clauses(text):
        de_exempted = _strip_exemptions(clause)
        if not APPLIED_WORK_RE.search(de_exempted):
            continue

        span = (offset, offset + len(clause))
        for ref in _ordered_unique(TASK_REF_RE.findall(clause)):
            key = ('applied_work', 'task', ref, default_project_id)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                CompletionClaim(
                    kind='applied_work',
                    subject='task',
                    ref=ref,
                    project_id=default_project_id,
                    span=span,
                )
            )

    return claims


def _ordered_unique(values: list[str]) -> list[str]:
    """De-duplicate *values* preserving first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
