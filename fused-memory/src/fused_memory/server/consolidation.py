"""Pure helpers for the ``consolidate_memories`` tool (task 3133).

Single home for the op's argument shape and (later leaves) its response
envelope construction.  Follows the :mod:`fused_memory.server.mem0_update_authz`
/ :mod:`fused_memory.server.markup_tripwire` precedent: ``server/tools.py``
registers its tools inside a closure, so a body written there would be
neither importable nor unit-testable, and these checks must be callable
without a running server, a service or an event loop.

Imports stay stdlib + :mod:`fused_memory.memory_metadata` /
:mod:`fused_memory.topic_slug` / :mod:`fused_memory.utils.validation` for
exactly that reason.

WHY VALIDATION IS A SEPARATE, FIRST STEP
----------------------------------------
``consolidate_memories`` is irreversible by construction: it writes a
canonical, patches retained peers and DELETES its supersedes.  Argument
validation is the only stage that can refuse at zero cost, so everything
decidable from the arguments alone is decided here — before the canonical
exists, before a single victim is touched.

Two properties follow from that position and are not incidental:

1. **No short-circuit.**  Every offender is collected and named in one
   error, the same bar :func:`fused_memory.memory_metadata.validate_memory_metadata`
   holds.  Returning the first problem would make a caller fix one id,
   re-run a DESTRUCTIVE operation, and learn a second fact the first call
   already knew.
2. **The delete arm requires ``run_id``.**  It becomes the tombstone's
   ``deleting_run_id`` — the field naming the run that KILLED a record,
   deliberately distinct from the victim's own ``metadata.run_id`` naming
   the run that WROTE it.  There is no ambient run id at the MCP boundary,
   and both fallbacks are worse than refusing: ``''`` mints tombstones that
   answer "who deleted this?" with silence, and reusing a session or
   causation id puts a non-run identifier in a field auditors read as a run
   id.  Enforcing it HERE rather than at the delete site is what keeps an
   unattributable delete from stranding a freshly-written canonical over
   supersedes that were never folded.
"""

from __future__ import annotations

from typing import Any

from fused_memory.memory_metadata import normalize_supersedes
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN, is_valid_topic_slug

# `_safe_repr` is module-private in `utils.validation` and this is its first
# cross-module import. Taken deliberately over a local copy: it is the repo's
# ONE repr-with-cap helper, and every sibling validator's rejection messages
# already render offending values through it. A second implementation here
# would be the duplication INV-5 forbids AND would let this tool's errors
# truncate differently from every other validator's.
from fused_memory.utils.validation import _safe_repr, is_full_uuid

__all__ = ['build_consolidation_result', 'validate_consolidate_args']

#: Pointer to the ONE topic-slug namespace (PRD D4), quoted at the wire so a
#: caller can find the rule rather than re-derive a regex that already has a
#: single home shared with ``ProceduralTopicCluster.topic_id``.
_TOPIC_HINT = (
    'A topic is lowercase alphanumeric segments joined by single hyphens '
    f'(max {TOPIC_SLUG_MAX_LEN} chars) — see `fused_memory.topic_slug`, the one '
    'namespace shared with ProceduralTopicCluster.topic_id. snake_case and '
    'Capitalised values are the common misses.'
)

#: Same guidance ``validate_full_uuid`` gives, restated here because this
#: validator reports MANY offending ids in one error and cannot return that
#: helper's per-value dict verbatim.
_UUID_HINT = (
    'Resolve each full 36-character UUID first — call search or get_memory_by_id '
    'and pass that result\'s `id` field verbatim. An 8-character hex prefix read '
    'out of a search-result snippet is not a valid id.'
)

#: The recovery procedure for a ``'partial'`` result, stated where the caller
#: that needs it is actually looking. There is deliberately NO resume arm —
#: this op takes no existing canonical id — so the ONE unsafe recovery is the
#: obvious one: re-running it writes a SECOND canonical for the same
#: ``(project, topic)``, which uniqueness only WARNS about under the shipped
#: ``enforce=False`` default and therefore admits. That is the +1-per-pass
#: ratchet the whole op exists to end, so the hint names the by-hand path
#: instead, per primitive, in the order a caller must apply them.
_PARTIAL_RECOVERY_HINT = (
    'PARTIAL IS NOT A RETRY SIGNAL. Do NOT re-run consolidate_memories for '
    'this topic: the canonical in `canonical_id` already exists, and a second '
    'call would write a SECOND canonical for the same (project, topic) — '
    'admitted rather than refused wherever canonical uniqueness is still in '
    'warn mode, which is the +1-per-pass ratchet this op exists to end. '
    'Finish the named ids by hand instead: read `failed_deletes` / '
    '`reparent_failures` / `retain_failures` for what did not happen and why, '
    'fix the cause, then call delete_memory per still-listed id (it runs the '
    'same citation gate and child guard this op does) and update_memory to '
    'tag any peer that was not retained. Ids in `survivors` are still in the '
    'corpus; ids in `survivor_check_failed` were proven neither gone nor '
    'alive — re-read those with get_memory_by_id before acting on them.'
)

_RUN_ID_HINT = (
    'Pass the reconciliation run performing this consolidation. It is stamped as '
    'the tombstone\'s `deleting_run_id` (the run that KILLED the record) and is '
    'deliberately NOT the victims\' own `metadata.run_id` (the run that WROTE '
    'them) — conflating the two is what makes a deletion audit unreadable.'
)


def _bad_id_members(values: list[Any], field: str) -> list[str]:
    """Name every member of *values* that is not a canonical full UUID.

    Reports by INDEX as well as value: two members can render identically
    under ``_safe_repr`` (``'873889a1'`` supplied twice), and a caller
    editing a list needs to know which slot to fix.
    """
    return [
        f'{field}[{i}] is not a full 36-character UUID: {_safe_repr(v)}'
        for i, v in enumerate(values)
        if not is_full_uuid(v)
    ]


def _repeated_id_members(values: list[Any], field: str) -> list[str]:
    """Name every member of *values* that repeats an earlier member.

    Reports BOTH slots by index, for the reason :func:`_bad_id_members`
    reports one: the two members render identically under ``_safe_repr``, so
    naming the value alone would not tell a caller which slot to delete.

    Hashable members only. An unhashable malformed member is already named by
    :func:`_bad_id_members`, and comparing it here would raise inside the one
    step whose entire job is to refuse without raising.
    """
    first_seen: dict[Any, int] = {}
    problems: list[str] = []
    for i, value in enumerate(values):
        if not isinstance(value, (str, bytes, int, float)):
            continue
        if value in first_seen:
            problems.append(
                f'{field}[{i}] repeats {field}[{first_seen[value]}]: '
                f'{_safe_repr(value)}'
            )
        else:
            first_seen[value] = i
    return problems


def validate_consolidate_args(
    *,
    canonical_content: Any,
    topic: Any,
    supersedes: Any,
    retain: Any,
    run_id: Any,
) -> tuple[dict[str, str] | None, list[Any], list[Any]]:
    """Check a ``consolidate_memories`` call and normalize its two id arms.

    Returns ``(error_or_None, normalized_supersedes, normalized_retain)``.
    The error is the flat ``{'error', 'error_type': 'ValidationError',
    'hint'}`` dict the MCP tool boundary RETURNS (never raises): a raised
    exception is flattened by ``@mcp_tool_errors`` to ``{'error',
    'error_type'}`` and the hint — the part that tells the caller what to do
    — would be lost.

    Both arms go through :func:`normalize_supersedes`, which supplies the
    legacy-scalar tolerance (PRD D2: 81 live records carry a scalar) for
    free and, critically, **never drops a malformed member** — so a short
    hex or a non-string survives normalization and can be rejected BY NAME
    below instead of being silently discarded along with the supersession
    edge it encoded.

    The normalized lists are returned even alongside an error so a caller
    that wants to log what it parsed can, but a non-``None`` error means
    nothing may be executed.
    """
    problems: list[str] = []
    hints: list[str] = []

    def _add_hint(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    # Normalize FIRST: every check below reasons about the list shape, and
    # `[]` vs None vs a legacy scalar must not be three different questions.
    supersedes_ids = normalize_supersedes(supersedes)
    retain_ids = normalize_supersedes(retain)

    if not isinstance(canonical_content, str) or not canonical_content.strip():
        problems.append(
            '`canonical_content` must be a non-empty string, got '
            f'{_safe_repr(canonical_content)}'
        )
        _add_hint(
            'canonical_content is the consolidated record\'s TEXT — the single '
            'claim the surviving canonical will state.'
        )

    if not is_valid_topic_slug(topic):
        problems.append(f'`topic` is not a valid topic slug: {_safe_repr(topic)}')
        _add_hint(_TOPIC_HINT)

    bad_ids = _bad_id_members(supersedes_ids, 'supersedes') + _bad_id_members(
        retain_ids, 'retain'
    )
    if bad_ids:
        problems.extend(bad_ids)
        _add_hint(_UUID_HINT)

    # An empty arm is an ABSENT arm — same rule as
    # `_update_memory_arm_presence_error`. Without this, a caller whose
    # supersedes list computed to `[]` gets a success envelope for a
    # consolidation that folded nothing.
    if not supersedes_ids and not retain_ids:
        problems.append(
            'consolidate_memories requires at least one arm: `supersedes` (fold '
            'and delete) or `retain` (tag in place). An empty list counts as no arm'
        )
        _add_hint(
            'A canonical with neither arm is just add_memory — call that instead.'
        )

    # Delete-and-retain is not a resolvable instruction, and silently
    # picking an arm would make the op's own report wrong about what it did
    # to that record. Compared on hashable members only: an unhashable
    # malformed member is already reported above.
    overlap = [
        i
        for i in supersedes_ids
        if isinstance(i, (str, bytes, int, float)) and i in retain_ids
    ]
    if overlap:
        seen: list[Any] = []
        for i in overlap:
            if i not in seen:
                seen.append(i)
        problems.append(
            'these ids appear in BOTH `supersedes` and `retain`: '
            + ', '.join(str(i) for i in seen)
        )
        _add_hint(
            'An id can be folded-and-deleted or retained-and-tagged, not both. '
            'Drop it from whichever arm you did not mean.'
        )

    # A repeated id is an ordinary authoring slip — a caller assembling one
    # cluster from two pages of search results is exactly who trips it — with
    # three durable consequences, and this is the only point at which all
    # three still cost nothing to avoid:
    #
    #   the delete arm awaits `delete_memory` TWICE for one record, emitting a
    #       second `memory_deleted` event and a second WriteJournal row for a
    #       record that is already gone;
    #   the canonical's durable `metadata.supersedes` KEEPS the repeat, because
    #       the step (7b) narrowing compares SETS and so never fires on a list
    #       that differs from the confirmed-gone set only by duplication;
    #   `tombstones_written` and `tombstones_expected` BOTH count the repeat
    #       while `ReconLedgerStore.upsert_many` collapses the two identical
    #       five-part identities to ONE row (last-write-wins, per its own
    #       docstring), so the pair the envelope advertises as its audit-trail
    #       proof overstates the ledger by one.
    #
    # That third one is why this is refused rather than tolerated: it is an
    # INFERRED count, in the op whose whole deliverable is corroborated ones
    # (INV-2 structured-facts, INV-3 corroborate-before-acting).
    #
    # REFUSED, not silently de-duplicated, for the reason the overlap check
    # above refuses rather than picking an arm, and the reason
    # `normalize_supersedes` never drops a member: rewriting the caller's set
    # would make the op's own report describe a request nobody made.
    repeats = _repeated_id_members(supersedes_ids, 'supersedes')
    repeats += _repeated_id_members(retain_ids, 'retain')
    if repeats:
        problems.extend(repeats)
        _add_hint(
            'List each id ONCE per arm. A repeat would be deleted twice, '
            'claimed twice by the canonical, and counted twice against a '
            'ledger that stores it once.'
        )

    # See the module docstring: refused HERE so an unattributable delete
    # costs zero writes rather than stranding a canonical over unfolded
    # supersedes. Retain-only calls are exempt — they never delete, so
    # nothing becomes unattributable.
    if supersedes_ids and (not isinstance(run_id, str) or not run_id.strip()):
        problems.append(
            'the delete arm requires `run_id`: a supersede cannot be deleted '
            'without an auditable `deleting_run_id` for its tombstone, got '
            f'{_safe_repr(run_id)}'
        )
        _add_hint(_RUN_ID_HINT)

    if not problems:
        return None, supersedes_ids, retain_ids

    return (
        {
            'error': 'consolidate_memories: ' + '; '.join(problems),
            'error_type': 'ValidationError',
            'hint': ' '.join(hints),
        },
        supersedes_ids,
        retain_ids,
    )


def build_consolidation_result(
    *,
    canonical_id: str,
    topic: str,
    canonical_supersedes: list[str] | None = None,
    supersedes_correction: dict[str, Any] | None = None,
    deleted: list[str],
    failed_deletes: list[dict[str, Any]],
    survivors: list[str],
    survivor_check_failed: list[dict[str, Any]] | None = None,
    retained: list[str] | None = None,
    retain_failures: list[dict[str, Any]] | None = None,
    reparented: list[dict[str, Any]] | None = None,
    reparent_failures: list[dict[str, Any]] | None = None,
    topic_members: list[Any] | None = None,
    topic_members_total: int | None = None,
    topic_members_truncated: bool = False,
    topic_members_available: bool = True,
    citation_repoint: dict[str, Any] | None = None,
    tombstones_written: int = 0,
    tombstones_expected: int = 0,
) -> dict[str, Any]:
    """The op's response envelope, and the ONE home of its status rule.

    Pure and importable so the rule can be reasoned about without a server,
    a service or an event loop — the same reason the validator above lives
    here.

    THE STATUS RULE.  ``'partial'`` whenever any disposition list is
    non-empty: a failed delete, a failed retain tag, a failed reparent, a
    SURVIVOR, or an id whose closure could not be CHECKED
    (``survivor_check_failed``).  The last two are the load-bearing cases
    and the reason the rule cannot key on failures alone — an id whose
    delete reported success but which still resolves produced no failure to
    count, and it is exactly the silent-fail-soft this op exists to
    surface.  ``'partial'`` is not a softer success: it says this cluster is
    still open, and the lists say which ids keep it open.

    ``survivor_check_failed`` is a THIRD outcome, not a variant of the other
    two: the id was neither observed alive (so it is not a survivor) nor
    proven gone (so it is not closed), because the corroborating read did
    not answer.  Folding it into either list would assert something no read
    supports, and it counts as open business for the plainest possible
    reason — corroborated closure IS this op's deliverable, so an id whose
    closure is unknown means the op did not deliver it.  That is the exact
    CONTRAST with the tombstone shortfall below: there the work completed
    and only its audit trail did not land; here the work's own proof is
    missing.

    WHAT THE STATUS RULE DELIBERATELY IGNORES: a tombstone shortfall
    (``tombstones_written < tombstones_expected``).  The memory operation
    genuinely completed — the cluster is folded and closed — and only its
    AUDIT TRAIL did not land, e.g. on a deployment running
    ``recon_ledger_enabled=False`` where the fail-safe writer returns 0.
    Reporting the two counts satisfies structured-facts and
    no-silent-fail-soft without conflating the two failures, which matters
    because ``'partial'`` is an invitation to RETRY: a caller that re-ran a
    COMPLETED consolidation would re-write a canonical whose supersedes are
    already gone, which is precisely the net-+1 ratchet this op exists to
    end.  An audit gap is fixed by looking at the logs, never by repeating
    the delete.

    WHAT ``'partial'`` IS NOT: a retry signal.  The lists say which ids keep
    the cluster open, and they are there to be FINISHED BY HAND — never by
    re-running this op.  A second call for the same ``(project, topic)``
    writes a SECOND canonical: uniqueness ships in warn mode
    (``enforce=False``), so the duplicate is censused and ADMITTED, which is
    the "+1 entry per failed pass, until the cluster contains the
    consolidator's own prior canonicals" ratchet this whole op exists to end.
    That recovery procedure is stated in the envelope itself, under ``hint``,
    because a caller reading a partial result is exactly the reader about to
    make that mistake and the docstring is not where it will look.

    ``canonical_supersedes`` is what the canonical DURABLY CLAIMS to have
    replaced once the call is done, which on a partial run is narrower than
    what was requested: the tool patches the field down to the
    corroborated-gone set rather than leaving a live record listed as
    superseded.  ``supersedes_correction`` reports that narrowing when it was
    needed — including when it FAILED, in which case ``canonical_supersedes``
    still shows the wider set the record is really carrying.

    Every disposition key is ALWAYS present, empty lists included, so a
    caller can read ``result['survivors']`` without a membership test and a
    later arm cannot quietly stop reporting by omitting its key.  Three keys
    are deliberate exceptions, each present only when the thing it describes
    actually happened: ``citation_repoint`` (an empty map would read as "the
    gate ran and found nothing" on a call where the gate never ran at all),
    ``supersedes_correction`` (absent means the canonical's claim was right
    as written, not that a correction ran and changed nothing) and ``hint``
    (recovery guidance on a clean run would be noise).
    """
    failed_deletes = list(failed_deletes)
    survivors = list(survivors)
    survivor_check_failed = list(survivor_check_failed or [])
    retain_failures = list(retain_failures or [])
    reparent_failures = list(reparent_failures or [])
    open_business = (
        failed_deletes
        or retain_failures
        or reparent_failures
        or survivors
        or survivor_check_failed
    )
    members = list(topic_members or [])
    result: dict[str, Any] = {
        'status': 'partial' if open_business else 'consolidated',
        'canonical_id': canonical_id,
        'topic': topic,
        # What the canonical's `metadata.supersedes` actually says now — a
        # claim about the corpus, distinct from `deleted` (what this call
        # did) and from the requested set (what it was asked to do).
        'canonical_supersedes': list(canonical_supersedes or []),
        'deleted': list(deleted),
        'failed_deletes': failed_deletes,
        'retained': list(retained or []),
        'retain_failures': retain_failures,
        'reparented': list(reparented or []),
        'reparent_failures': reparent_failures,
        'survivors': survivors,
        'survivor_check_failed': survivor_check_failed,
        'topic_members': members,
        # Disclosed unconditionally, next to the listing it qualifies: a
        # capped scroll that reported only its rows would read as the whole
        # closure, which is the overclaim this op exists to eliminate.
        'topic_members_total': (
            len(members) if topic_members_total is None else topic_members_total
        ),
        'topic_members_truncated': bool(topic_members_truncated),
        # The same disclosure one step further out: an empty listing from a
        # scroll that never ANSWERED is indistinguishable, in the payload
        # alone, from a topic that genuinely has no members. This flag is
        # what separates them, so `[]` is never read as a finding.
        'topic_members_available': bool(topic_members_available),
        # Reported as a PAIR, always. A bare "written" count says nothing
        # about whether it was enough, and the shortfall is only legible
        # against what was owed.
        'tombstones_written': int(tombstones_written),
        'tombstones_expected': int(tombstones_expected),
    }
    if citation_repoint:
        result['citation_repoint'] = citation_repoint
    if supersedes_correction:
        result['supersedes_correction'] = supersedes_correction
    if open_business:
        # Carried IN the envelope, not just in the docstring: the caller
        # holding a partial result is the one about to re-run the op, and a
        # re-run is the ratchet. See THE STATUS RULE above.
        result['hint'] = _PARTIAL_RECOVERY_HINT
    return result
