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
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from shared.task_statuses import TERMINAL as TERMINAL_TASK_STATUSES

from fused_memory.middleware.recon_claim_verification_guard import (
    _GIT_PROBE_TIMEOUT_SECS,
    _resolve_git_toplevel,
)
from fused_memory.reconciliation.task_filter import (
    _CLAUSE_SPLIT_RE,
    FUTURE_ASPIRATIONAL_RE,
    NEGATED_TERMINAL_RE,
    TASK_REF_RE,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import of the optional ``escalation`` workspace package, mirroring
# server/markup_tripwire.py: when it is missing (minimal CI envs, deployments
# that have not installed it) the escalation becomes a logged no-op. This module
# sits on the MCP write path, so it must never make a write fail — the episode is
# already ingested and tagged by the time escalation is attempted.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

__all__ = [
    'APPLIED_WORK_RE',
    'COMMIT_REF_RE',
    'FILING_DISPATCH_RE',
    'TICKET_REF_RE',
    'UNRESOLVABLE',
    'UNVERIFIED_CLAIM_TAG',
    'ClaimVerdict',
    'CompletionClaim',
    'build_unverified_flag',
    'emit_unverified_claim_escalation',
    'make_commit_probe',
    'extract_completion_claims',
    'verify_claims',
]

ClaimKind = Literal['applied_work', 'filing_dispatch']
ClaimSubject = Literal['task', 'commit', 'ticket']
VerdictStatus = Literal['verified', 'mismatch', 'unverifiable']

#: The tag stamped onto an episode carrying at least one non-verified claim.
#: It rides the same channel ``temporal_context`` does — a plain
#: ``add_episode`` parameter, then a durable-queue payload key, then a
#: ``source_description`` prefix on the Graphiti episodic node and a
#: ``metadata`` key on every derived Mem0 fact. ``add_episode`` deliberately
#: never persists its ``metadata`` argument, so a metadata key could not carry
#: it; and the harm in the motivating incident was the DERIVED edges, not the
#: tool response, so a response-only tag would have labelled none of it.
UNVERIFIED_CLAIM_TAG: str = 'unverified_claim'


class _Unresolvable:
    """Sentinel type: the authority could not be consulted at all."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return 'UNRESOLVABLE'


#: Distinguishes "the registry says no such ticket" (a MISMATCH — the claim is
#: false) from "the registry could not be reached" (UNVERIFIABLE — the claim is
#: merely unchecked). Both end up tagged, but only the first is evidence that
#: the writer was wrong, and the flag must not conflate them (INV-2).
UNRESOLVABLE: _Unresolvable = _Unresolvable()


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

# The esc-3085-1 scope extension: a claim that work was FILED/DISPATCHED
# somewhere ("re-filed into dark_factory's tree as ticket tkt_...") is the same
# failure shape as a claim that it landed — it asserts a durable artefact
# exists, and the incident's artefact did not.
#
# Unlike the applied family these verbs match BARE, with no copula guard.
# Instance (2)'s own wording forces it: "...and re-filed into ..." has no
# copula before the verb. The looser guard is affordable here because a filing
# claim still needs a concrete ref in the same clause to become a claim at all,
# and "task N filed <ticket>" is a filing claim about that ticket either way —
# the transitive reading that motivates task_filter's copula guard for
# applied/resolved does not produce a WRONG subject here.
_FILING_ANY_FORM: str = (
    r're[-\s]?fil(?:e|ed|es|ing)|fil(?:e|ed|es|ing)|submit(?:ted|s|ting)?|'
    r'queu(?:e|ed|es|ing)|dispatch(?:ed|es|ing)?|cancel(?:l?ed|s|l?ing)?|'
    r'clos(?:e|ed|es|ing)'
)

FILING_DISPATCH_RE: re.Pattern[str] = re.compile(
    r'\b(?:'
    r're[-\s]?filed|filed|submitted|queued|dispatched|cancell?ed|'
    # "closed as <state>" only — a bare "closed" describes non-task things far
    # too often (task_filter drops it from TERMINAL_OUTCOME_RE for exactly this
    # reason), and the "as" is what marks a dispatch outcome.
    r'closed\s+as'
    r')\b',
    re.IGNORECASE,
)

# `tkt_` + the 26-char Crockford-base32 body minted by
# ticket_store._new_ticket_id. The bounded body length is load-bearing: it is
# what makes a bare "filed as ticket tkt_" (no id) produce no claim rather than
# a claim about an empty ref that no authority could ever resolve.
TICKET_REF_RE: re.Pattern[str] = re.compile(r'(\btkt_[0-9A-Za-z]{20,32}\b)')

# A commit sha is 7-40 hex chars — a shape that ordinary English words hit by
# accident ("deadbeef", "added", "facade"). An explicit commit/sha cue is
# therefore REQUIRED, so a hex-looking word is never mistaken for a sha and
# sent to a git probe.
COMMIT_REF_RE: re.Pattern[str] = re.compile(
    r'\b(?:commit|sha|merge[-\s]commit)\b\s*[:=]?\s*[`\'"]?([0-9a-f]{7,40})\b',
    re.IGNORECASE,
)

# A project qualifier sitting immediately before a ref — "dark_factory task
# 3142", "dark_factory's task 3142". Case-sensitive and anchored to the end of
# the preceding text, and (crucially) only HONOURED when the captured word is a
# member of the server's project registry: without that membership test "the
# merge task 3142" would read 'merge' as a project name, and every unqualified
# ref in ordinary prose would acquire a bogus owner.
_PROJECT_QUALIFIER_RE: re.Pattern[str] = re.compile(
    r"([A-Za-z][A-Za-z0-9_-]*)(?:'s)?\s+$"
)

# The `<project_id>:<task_id>` external-dependency spelling used throughout
# task metadata. TASK_REF_RE cannot see it (no task/df/# cue), and it is only
# ever accepted for a REGISTERED project, which is what keeps it from firing on
# an unrelated "label:1234" or a "host:8080".
_EXTERNAL_TASK_REF_RE: re.Pattern[str] = re.compile(
    r'\b([A-Za-z][A-Za-z0-9_-]*):(\d+)\b'
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

# Supplementary strippers: the delta vocabulary above (BOTH families), in the
# SAME shape as the task_filter originals (each swallows the verb it governs, so
# removing the span removes the completion evidence). The modal/filler prefixes
# are copied from FUTURE_ASPIRATIONAL_RE deliberately — a narrower prefix here
# would leave "the follow-up is supposed to be filed as tkt_X next week" reading
# as an accomplished filing.
_EXTENSION_ANY_FORM: str = _APPLIED_ANY_FORM + r'|' + _FILING_ANY_FORM

_NEGATED_EXTENSION_RE: re.Pattern[str] = re.compile(
    r"\b(?:not|never|hasn't|has\s+not|yet\s+to\s+be)\s+(?:yet\s+|been\s+)*"
    r'(?:' + _EXTENSION_ANY_FORM + r')\b',
    re.IGNORECASE,
)
_ASPIRATIONAL_EXTENSION_RE: re.Pattern[str] = re.compile(
    r'\b(?:will|going\s+to|plans?\s+to|planned\s+to|intends?\s+to|intended\s+to|'
    r'aims?\s+to|meant\s+to|hopes?\s+to|expects?\s+to|scheduled\s+to|slated\s+to|'
    r'supposed\s+to|needs?\s+to|to\s+be|should|would|shall)\b'
    r'(?:\s+(?:be|been|get|soon|also|now|already|just|then|finally|'
    r'eventually|not|yet|still)){0,3}'
    r'\s+(?:' + _EXTENSION_ANY_FORM + r')\b',
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
    de_exempted = _NEGATED_EXTENSION_RE.sub(' ', de_exempted)
    de_exempted = FUTURE_ASPIRATIONAL_RE.sub(' ', de_exempted)
    return _ASPIRATIONAL_EXTENSION_RE.sub(' ', de_exempted)


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
        if APPLIED_WORK_RE.search(de_exempted):
            kind: ClaimKind = 'applied_work'
        elif FILING_DISPATCH_RE.search(de_exempted):
            kind = 'filing_dispatch'
        else:
            continue

        subject, refs = _refs_in_clause(
            clause,
            default_project_id=default_project_id,
            known_project_ids=known_project_ids,
        )
        if subject is None:
            continue

        span = (offset, offset + len(clause))
        for ref, project_id in refs:
            key = (kind, subject, ref, project_id)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                CompletionClaim(
                    kind=kind,
                    subject=subject,
                    ref=ref,
                    project_id=project_id,
                    span=span,
                )
            )

    return claims


def _qualifying_project(
    clause: str, start: int, known_project_ids: frozenset[str] | set[str]
) -> str | None:
    """Return the registered project qualifying the ref at *start*, else None."""
    match = _PROJECT_QUALIFIER_RE.search(clause[:start])
    if match is None:
        return None
    candidate = match.group(1)
    return candidate if candidate in known_project_ids else None


def _refs_in_clause(
    clause: str,
    *,
    default_project_id: str,
    known_project_ids: frozenset[str] | set[str],
) -> tuple[ClaimSubject | None, list[tuple[str, str | None]]]:
    """Return the MOST SPECIFIC ref family named in *clause*, each ref paired
    with the project whose authority adjudicates it.

    Precedence is ticket > commit > task, and it is deliberately exclusive: a
    clause naming both ("task 5638 ... re-filed as ticket tkt_X") is one claim
    about the ticket, not two. The most specific ref is the one the claim is
    actually asserting into existence, and it is checked against the cheapest,
    most exact authority — a primary-key lookup rather than a status read.
    """
    tickets = _ordered_unique(TICKET_REF_RE.findall(clause))
    if tickets:
        # A ticket id is a globally unique primary key in one shared
        # tickets.db, so a ticket claim carries no project at all — that is
        # precisely what let instance (2) (a reify writer claiming a
        # dark_factory ticket) be adjudicated correctly.
        return 'ticket', [(ref, None) for ref in tickets]

    commits: list[tuple[str, str | None]] = [
        (match.group(1), _qualifying_project(clause, match.start(), known_project_ids)
         or default_project_id)
        for match in COMMIT_REF_RE.finditer(clause)
    ]
    if commits:
        return 'commit', _ordered_unique_pairs(commits)

    tasks: list[tuple[str, str | None]] = [
        (match.group(1), _qualifying_project(clause, match.start(), known_project_ids)
         or default_project_id)
        for match in TASK_REF_RE.finditer(clause)
    ]
    tasks += [
        (match.group(2), match.group(1))
        for match in _EXTERNAL_TASK_REF_RE.finditer(clause)
        if match.group(1) in known_project_ids
    ]
    if tasks:
        return 'task', _ordered_unique_pairs(tasks)
    return None, []


def _ordered_unique(values: list[str]) -> list[str]:
    """De-duplicate *values* preserving first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _ordered_unique_pairs(
    values: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """De-duplicate ``(ref, project_id)`` pairs preserving first-appearance order."""
    seen: set[tuple[str, str | None]] = set()
    out: list[tuple[str, str | None]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """One claim adjudicated against its authority.

    Attributes:
        claim: The claim that was checked.
        status: ``'verified'`` (the authority agrees), ``'mismatch'`` (the
            authority CONTRADICTS the claim) or ``'unverifiable'`` (the
            authority could not be consulted).
        observed: What was actually seen — a live status string, an absent-id
            report, a registry row summary. Recorded verbatim so a reader can
            re-check the verdict rather than take it on trust (INV-2).
    """

    claim: CompletionClaim
    status: VerdictStatus
    observed: str


def verify_claims(
    claims: list[CompletionClaim],
    *,
    task_status_probe: Callable[[str, str | None], str | None],
    ticket_probe: Callable[[str], dict[str, Any] | None | _Unresolvable],
    commit_probe: Callable[[str, str | None], bool | None],
) -> list[ClaimVerdict]:
    """Adjudicate each claim against its authority. Pure, sync, never raises
    on probe RESULTS (a probe that itself raises is the caller's to contain).

    Probe contract — TRI-STATE in every case:

    * ``task_status_probe(ref, project_id) -> str | None``: the task's live
      status, or ``None`` when it cannot be resolved. The literal ``'unknown'``
      sentinel a NULL/absent DB status maps to counts as unresolvable too.
      Terminal status (done/cancelled) verifies; any other real status is a
      mismatch.
    * ``ticket_probe(ref) -> dict | None | UNRESOLVABLE``: the registry row,
      ``None`` when the registry says NO SUCH TICKET (a mismatch — this is
      esc-3085-1 instance (2)), or :data:`UNRESOLVABLE` when the registry could
      not be consulted.
    * ``commit_probe(ref, project_id) -> bool | None``: ``True`` present,
      ``False`` absent, ``None`` unresolvable.

    THE FAIL DIRECTION IS DELIBERATELY INVERTED relative to this module's two
    closest siblings, and copying theirs would silently reproduce the hole this
    gate exists to close:

    * ``tools._premature_completion_block`` REJECTS a write, and
      ``recon_claim_verification_guard.make_source_and_history_probe`` DROPS a
      candidate. Both must fail OPEN, because an infra hiccup bouncing a
      legitimate write is worse than the stale claim it would have caught.
    * This gate only LABELS. Its worst case on a false positive is an extra
      source_description prefix and one metadata key on a kept episode; its
      worst case on a false NEGATIVE is another batch of false Graphiti edges
      like the five in reify esc-5603-1. So an unresolvable authority lands on
      ``'unverifiable'`` and gets TAGGED, never quietly passed.

    Short-circuits on an empty claim list, so the common write path never
    touches an authority at all.
    """
    if not claims:
        return []

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        if claim.subject == 'ticket':
            verdicts.append(_verify_ticket(claim, ticket_probe))
        elif claim.subject == 'commit':
            verdicts.append(_verify_commit(claim, commit_probe))
        else:
            verdicts.append(_verify_task(claim, task_status_probe))
    return verdicts


def _verify_task(
    claim: CompletionClaim,
    probe: Callable[[str, str | None], str | None],
) -> ClaimVerdict:
    status = probe(claim.ref, claim.project_id)
    if status is None:
        return ClaimVerdict(
            claim, 'unverifiable',
            f'live status for task {claim.ref} could not be resolved '
            f'(project={claim.project_id!r})',
        )
    if status in TERMINAL_TASK_STATUSES:
        return ClaimVerdict(claim, 'verified', status)
    if status == 'unknown':
        # get_statuses' documented sentinel for a NULL/absent status. It is NOT
        # a live status, so it cannot contradict the claim — but it cannot
        # confirm it either.
        return ClaimVerdict(
            claim, 'unverifiable',
            f'task {claim.ref} reports status "unknown" '
            f'(project={claim.project_id!r})',
        )
    return ClaimVerdict(claim, 'mismatch', status)


def _verify_ticket(
    claim: CompletionClaim,
    probe: Callable[[str], dict[str, Any] | None | _Unresolvable],
) -> ClaimVerdict:
    row = probe(claim.ref)
    if isinstance(row, _Unresolvable):
        return ClaimVerdict(
            claim, 'unverifiable',
            f'ticket registry could not be consulted for {claim.ref}',
        )
    if row is None:
        return ClaimVerdict(
            claim, 'mismatch', f'no ticket {claim.ref} exists in the registry',
        )
    return ClaimVerdict(
        claim, 'verified',
        f'ticket {claim.ref} exists (project_id={row.get("project_id")!r}, '
        f'status={row.get("status")!r})',
    )


def _verify_commit(
    claim: CompletionClaim,
    probe: Callable[[str, str | None], bool | None],
) -> ClaimVerdict:
    present = probe(claim.ref, claim.project_id)
    if present is None:
        return ClaimVerdict(
            claim, 'unverifiable',
            f'commit {claim.ref} could not be checked in project '
            f'{claim.project_id!r}',
        )
    if present:
        return ClaimVerdict(
            claim, 'verified',
            f'commit {claim.ref} exists in project {claim.project_id!r}',
        )
    return ClaimVerdict(
        claim, 'mismatch',
        f'no commit {claim.ref} in project {claim.project_id!r}',
    )


def build_unverified_flag(
    verdicts: list[ClaimVerdict],
    *,
    text: str,
) -> dict[str, Any] | None:
    """Build the structured flag for the non-verified verdicts, or None.

    Returns ``None`` when every verdict verified (or there are none) — the
    caller uses that to keep the clean path exactly inert: no tag, no response
    key, no log line.

    Each entry records the claim's own words (sliced from *text* by its span)
    alongside the OBSERVED authority state, so the flag is self-contained: a
    reader can see what was claimed, what was checked, and what was actually
    there, without re-running anything (INV-2).
    """
    entries = [
        {
            'kind': verdict.claim.kind,
            'subject': verdict.claim.subject,
            'ref': verdict.claim.ref,
            'project_id': verdict.claim.project_id,
            'span': [verdict.claim.span[0], verdict.claim.span[1]],
            'text': text[verdict.claim.span[0]:verdict.claim.span[1]].strip(),
            'status': verdict.status,
            'observed': verdict.observed,
        }
        for verdict in verdicts
        if verdict.status != 'verified'
    ]
    if not entries:
        return None
    return {'tag': UNVERIFIED_CLAIM_TAG, 'claims': entries}


# --------------------------------------------------------------------------- #
# Git-backed commit probe (the module's only impure export)
# --------------------------------------------------------------------------- #

#: Matches git's clean "no such object" report. `git cat-file -e` exits 128 for
#: BOTH a missing object and a missing repository, so the exit code alone
#: cannot tell "the writer named a commit that does not exist" from "git was
#: unusable here" — and reporting the second as the first would put a false
#: accusation in an escalation. The stderr text is the only discriminator git
#: offers, so it is the one used, and anything it does not match degrades to
#: UNRESOLVABLE rather than to a clean miss.
_NOT_A_VALID_OBJECT_RE: re.Pattern[str] = re.compile(
    r'not a valid object name', re.IGNORECASE,
)


def make_commit_probe(repo_root: Path | str) -> Callable[[str], bool | None]:
    """Build a ``probe(sha) -> bool | None`` backed by *repo_root*'s git objects.

    ``True`` the sha resolves to a commit, ``False`` git says no such object,
    ``None`` git could not answer (not a repository, binary missing, timeout,
    any other non-zero exit). NEVER raises.

    Rooted at *repo_root*'s resolved git top level via the sibling guard's
    :func:`~middleware.recon_claim_verification_guard._resolve_git_toplevel`, so
    a package subdirectory probes the whole repository rather than nothing.

    ``cat-file -e <sha>^{commit}`` (not ``rev-parse``) is the check: the
    ``^{commit}`` peel means a sha that names a blob or a tree — or an
    abbreviation that is ambiguous — is a miss rather than a spurious hit, and
    ``-e`` keeps the output empty on success.

    NOTE the return type differs from
    :func:`~middleware.recon_claim_verification_guard.make_source_and_history_probe`,
    which fails OPEN to ``True`` on a git error. That probe DROPS a candidate on
    a False, so a hiccup must not manufacture a fabrication flag. This one only
    labels, so its errors resolve to ``None`` -> ``'unverifiable'`` -> tagged.
    """
    resolved_root = _resolve_git_toplevel(Path(repo_root))

    def probe(sha: str) -> bool | None:
        try:
            result = subprocess.run(
                ['git', 'cat-file', '-e', f'{sha}^{{commit}}'],
                cwd=resolved_root,
                timeout=_GIT_PROBE_TIMEOUT_SECS,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                'completion_claim_gate: git cat-file failed for sha=%r under %s '
                '— commit existence is UNRESOLVABLE: %s',
                sha, resolved_root, exc,
            )
            return None

        if result.returncode == 0:
            return True

        stderr = (result.stderr or '').strip()
        if _NOT_A_VALID_OBJECT_RE.search(stderr):
            return False

        logger.warning(
            'completion_claim_gate: git cat-file errored (exit=%d) for sha=%r '
            'under %s — commit existence is UNRESOLVABLE: %s',
            result.returncode, sha, resolved_root, stderr,
        )
        return None

    return probe


# --------------------------------------------------------------------------- #
# Operator-facing escalation
# --------------------------------------------------------------------------- #
#
# Copied shape-for-shape from server/markup_tripwire.emit_markup_storm_escalation
# (task 3141), including its choice of channel. The recon_report filer is NOT
# usable here: it silently DROPS findings when no Stage-2 run is active, and an
# episode arrives at arbitrary times, so a gate that filed through it would go
# quiet exactly when nothing else is watching. Opening the project's queue
# directly is what makes the finding survive to an operator.
_QUEUE_DIRNAME: str = 'data/escalations'
_ANCHOR_PREFIX: str = 'unverified-claim'
_AGENT_ROLE: str = 'fused-memory/completion-claim-gate'
_CATEGORY: str = 'unverified_completion_claim'


def emit_unverified_claim_escalation(
    project_root: str | None,
    flag: dict[str, Any],
) -> str | None:
    """File an ``unverified_completion_claim`` escalation for *flag* (INV-4).

    Returns the escalation id — freshly filed, or the id of an already-open
    escalation for this ``(project_root, ref)`` (dedup) — or ``None`` when
    filing is impossible or fails.

    NEVER raises. The episode is already ingested and tagged by the time this
    runs, so escalation is purely ADDITIVE: every failure mode degrades to
    ``None`` plus a log line rather than changing the write's outcome.

    The tag alone is not enough. It labels the corpus, but nothing routinely
    reads the corpus hunting for tags — without a queued record the finding
    lives only in a WARNING line, which is how the motivating incident went
    unnoticed long enough to matter in the first place.

    The anchor is per-REF rather than per-project (the markup sibling's choice):
    two different false claims are two different findings and each deserves its
    own record, while a writer repeating the SAME claim collapses onto the one
    open escalation instead of minting a new one per episode.
    """
    entries = (flag or {}).get('claims') or []
    if not entries:
        return None
    ref = str(entries[0].get('ref') or '').strip()
    if not ref:
        return None
    if project_root is None:
        logger.debug(
            'completion_claim_gate: no project_root resolved; unverified claim '
            'about %r will not be escalated', ref,
        )
        return None
    if not HAS_ESCALATION:
        logger.debug(
            'completion_claim_gate: escalation package unavailable; unverified '
            'claim about %r in project_root=%r will not be escalated',
            ref, project_root,
        )
        return None

    anchor = f'{_ANCHOR_PREFIX}-{ref}'
    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'completion_claim_gate: failed to open the escalation queue for '
            'project_root=%r; unverified claim about %r not escalated',
            project_root, ref,
        )
        return None

    # Best-effort dedup: a read failure falls THROUGH to filing rather than
    # bailing out — losing duplicate-suppression is strictly better than losing
    # the finding.
    try:
        existing = queue.get_by_task(anchor, status='pending')
    except Exception:
        logger.exception(
            'completion_claim_gate: failed to check for an existing open '
            'escalation for anchor=%r in project_root=%r; proceeding to file',
            anchor, project_root,
        )
        existing = []
    if existing:
        logger.info(
            'completion_claim_gate: %s already open for anchor=%r in '
            'project_root=%r; not filing a duplicate',
            existing[0].id, anchor, project_root,
        )
        return existing[0].id

    detail = '\n'.join(
        [f'project_root={project_root!r}', '']
        + [
            line
            for entry in entries
            for line in (
                f'- subject={entry.get("subject")!r} ref={entry.get("ref")!r} '
                f'kind={entry.get("kind")!r} project_id={entry.get("project_id")!r}',
                f'  verdict={entry.get("status")!r} observed={entry.get("observed")!r}',
                f'  claimed: {entry.get("text")!r}',
            )
        ]
        + [
            '',
            'An episode was ingested carrying a completion claim that the live '
            'authority CONTRADICTS (verdict=mismatch) or could not confirm '
            '(verdict=unverifiable). The episode was TAGGED, not rejected: its '
            "Graphiti source_description is prefixed '[unverified_claim] ' and "
            "every derived Mem0 fact carries metadata['unverified_claim']=True, "
            'so the derived edges are labelled at the point of harm.',
            '',
            'Check the claim against the authority named above. If it is false, '
            'the derived facts need correcting at the source — a tag marks them, '
            'it does not retract them. If it is true, the authority was stale or '
            'unreachable and the tag is the false positive; that is the cheap '
            'direction and needs no corpus repair.',
            '',
            'Task 3142 / PRD leaf pi owns this gate '
            '(fused_memory.services.completion_claim_gate); grep the server logs '
            "for 'completion_claim_gate.unverified' for the full set.",
        ]
    )

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(anchor),
            task_id=anchor,
            agent_role=_AGENT_ROLE,
            # 'info', not 'blocking': nothing is stuck. The episode landed, the
            # tag is on it, and this record exists so the claim gets checked —
            # filing it as blocking would put routine write-path noise in front
            # of work that genuinely cannot proceed.
            severity='info',
            category=_CATEGORY,
            summary=(
                f'unverified completion claim about {entries[0].get("subject")} '
                f'{ref} ({entries[0].get("status")}: '
                f'{entries[0].get("observed")!r})'
            ),
            detail=detail,
            suggested_action=(
                'check the claim against the named authority; if it is false, '
                'correct the derived facts at the source'
            ),
        )
        esc_id = queue.submit(esc)
    except Exception:
        # A queue I/O failure must not propagate: the episode is already
        # ingested and tagged, and the WARNING at the call site has already
        # recorded the finding. The operator simply loses the queued heads-up.
        logger.exception(
            'completion_claim_gate: failed to submit the unverified-claim '
            'escalation for project_root=%r anchor=%r', project_root, anchor,
        )
        return None

    logger.warning(
        'completion_claim_gate: queued %s for project_root=%r anchor=%r',
        esc_id, project_root, anchor,
    )
    return esc_id
