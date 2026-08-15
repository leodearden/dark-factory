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

__all__ = ['validate_consolidate_args']

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
