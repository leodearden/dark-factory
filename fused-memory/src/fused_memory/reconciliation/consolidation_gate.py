"""Consolidation-gate shape brief + mechanical closure predicate (task 3112).

## The two defects this module fixes

**Defect 1 — the gate-filing instruction is silent on the target END STATE.**
:func:`fused_memory.reconciliation.recon_self_model.render_source_completion_section`
is today the whole instruction a recon stage gets for filing a consolidation
gate, and it prescribes no end-state shape, no enumeration policy and no
re-search guard.  Every gate therefore invents its own (dark-factory gates
2969/2973/3011/3016/3036/3063/3092; 3036 hand-wrote its member enumeration
under an invented ``metadata.memory_ids`` key, which a later cycle then
extended 7→8 while it still defined "done").  :func:`render_end_state_brief`
and :func:`render_consolidation_gate_section` supply the missing shape.

**Defect 2 — nothing REFUSES to close a gate task over a malformed cluster.**
``consolidate_memories`` (task 3133) reports closure at *op* time, but the
user-observable signal is the gate TASK closing, and no gate stood between a
curator's claim and that transition.  :func:`evaluate_closure` is that
predicate; ``middleware/task_interceptor.py::TaskInterceptor._apply_status_transition``
is the seam that calls it.

## The end state: PRD §3 Option C, as ratified by gate 3200

A consolidated cluster is *N short single-claim peers sharing
``metadata.topic``, exactly one of which carries ``canonical: true``*.  The
surviving same-topic peers are therefore **correct**, not residue: a predicate
that treats a live same-topic peer as a defect makes every correctly executed
consolidation permanently uncloseable.  The predicate here refuses only on
cluster MALFORMEDNESS (wrong canonical count, an "absorbed" id still live, an
unstamped cluster member) or on a view too incomplete to judge — never on peer
count.

## Import-LEAF, deliberately

``middleware/task_interceptor.py`` imports this module, so this module's import
weight becomes the interceptor's.  Imports are restricted to the standard
library plus :mod:`fused_memory.memory_metadata`,
:mod:`fused_memory.utils.validation` and
``reconciliation.recon_self_model.EXECUTION_CLASSES``.  PRD D4 records a
*measured* hard import cycle from a careless import of exactly this kind
(``config/schema.py`` → ``memory_metadata`` → ``backends.mem0_client`` →
``config.schema`` raising ``ImportError: cannot import name
'FusedMemoryConfig'``), which is why ``TOPIC_SLUG_RE`` got its own stdlib-only
leaf module with a regression test.  Nothing here may import
``reconciliation.targeted``, ``reconciliation.harness``,
``services.memory_service`` or ``server.tools``;
``tests/test_consolidation_gate.py`` pins that in both import orders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fused_memory.memory_metadata import (
    EXPERIMENTAL_KEY_PREFIX,
    normalize_supersedes,
)
from fused_memory.utils.validation import is_full_uuid

__all__ = [
    'EXIT_CLOSED',
    'EXIT_NOT_CLOSED',
    'GATE_METADATA_KEY',
    'ClosureVerdict',
    'evaluate_closure',
    'render_end_state_brief',
]

# The Tier-C ``x_``-prefixed gate block under which a consolidation gate carries
# its working TOPIC (plus any inert provenance and audited waivers).  Composed
# from EXPERIMENTAL_KEY_PREFIX rather than re-spelling the ``x_`` literal — the
# precedent set by ``server/grouped_read.py::CONTESTED_METADATA_KEY``.  A Tier-C
# key passes the metadata boundary silently and generates no unknown-key census
# line, so it needs no amendment to RESERVED_VOCABULARY_KEYS.
GATE_METADATA_KEY = f'{EXPERIMENTAL_KEY_PREFIX}recon_consolidation_gate'


# --------------------------------------------------------------------------- #
# Defect 1 — the end-state brief.  Single-sourced: the filed gate's
# description, the stage prompt section and the closure predicate's docstring
# all read the SAME text, so the three cannot prescribe different targets.
# --------------------------------------------------------------------------- #


def render_end_state_brief() -> str:
    """Render the target-shape brief a filed consolidation gate carries.

    This is Defect 1's payload.  Until now
    ``recon_self_model.render_source_completion_section`` was the entire
    gate-filing instruction and it named no end state at all, so each gate
    invented its own.  Reused verbatim by
    :func:`render_consolidation_gate_section` and embedded in every gate built
    by :func:`build_consolidation_gate_task`, so the prompt, the filed gate and
    the closure predicate cannot drift apart.
    """
    return (
        'TARGET END STATE (PRD memory-metadata-vocabulary §3 Option C, '
        'ratified by gate 3200):\n'
        ' 1. Split the cluster into N SHORT single-claim peers. Each peer '
        'states ONE claim; none is a concatenation of the others.\n'
        ' 2. Every peer carries the same `metadata.topic=<slug>`. That shared '
        'topic — not a hand-written member list — is what makes the cluster '
        'findable and is the only working list the closure check reads.\n'
        ' 3. Exactly ONE peer carries `metadata.canonical: true`, and that '
        'canonical is itself SHORT: an index/summary claim pointing at its '
        'peers, never the concatenated body of the cluster.\n'
        ' 4. That canonical\'s `metadata.supersedes` lists ONLY ids genuinely '
        'deleted/absorbed. An id that is still live must not appear there.\n'
        ' 5. Execute the change through the `consolidate_memories` tool\'s '
        'ratified `retain` arm rather than by hand: retained ids are tagged '
        'in place with the cluster topic and are never deleted, never given '
        '`canonical`, never given `parent_id`. Surviving same-topic peers are '
        'the TARGET, not residue.\n\n'
        'THE "1 canonical + 1 appendix" ABSORBING END STATE IS RETIRED. Do not '
        'aim for it. PRD §3 measured the inversion twice: 168c3a6b ranked '
        '10/10 and was then deleted, and its ~9k-char replacement bbc063a7 was '
        'ABSENT from a limit=10 window while ten short siblings ranked '
        '0.66-0.76. Post-consolidation write rate measurably DOUBLED. The '
        'effect is a property of entry LENGTH, not of those particular '
        'entries — so an absorbing canonical becomes the least retrievable '
        'member of its own cluster, and writers who cannot retrieve it keep '
        'minting entry N+1. Absorbing the cluster into one long record '
        'therefore defeats the consolidation it was meant to perform.'
    )


# --------------------------------------------------------------------------- #
# Defect 2 — the closure predicate.
# --------------------------------------------------------------------------- #

#: Exit-code contract, shared by the ``set_task_status`` seam and by
#: ``scripts/check_consolidation_closure.py`` so the two can never disagree
#: about what a verdict MEANS.  Derived from ``closed`` in exactly one place
#: (:func:`_verdict`); nothing else may construct a :class:`ClosureVerdict`.
EXIT_CLOSED = 0
EXIT_NOT_CLOSED = 1


@dataclass(frozen=True)
class ClosureVerdict:
    """The result of asking "is this topic cluster in the Option-C end state?".

    ``reasons`` are STRUCTURED entries (``{'code', 'ids', 'detail'}``), not
    bare prose, so the interceptor refusal dict and the operator CLI can each
    render them their own way from one source.  ``waived`` echoes every audited
    ``considered_and_kept`` entry that actually suppressed a reason, so a
    waiver is visible in the verdict rather than silently applied.
    """

    closed: bool
    reasons: tuple[dict[str, Any], ...]
    exit_code: int
    topic: str
    message: str
    waived: tuple[dict[str, Any], ...] = ()


def _reason(code: str, *, ids: Sequence[Any] = (), detail: str = '') -> dict[str, Any]:
    """One structured refusal entry.  ``ids`` NAMES the offenders."""
    return {'code': code, 'ids': [str(i) for i in ids], 'detail': detail}


def _verdict(
    topic: str,
    reasons: Sequence[dict[str, Any]],
    *,
    waived: Sequence[dict[str, Any]] = (),
    message: str = '',
) -> ClosureVerdict:
    """THE single home of the ``closed`` → ``exit_code`` derivation.

    Kept in one place so the boolean and the exit code cannot drift apart:
    everything that builds a verdict goes through here.
    """
    frozen_reasons = tuple(reasons)
    closed = not frozen_reasons
    if not message:
        if closed:
            message = (
                f'Topic {topic!r} is in the Option-C end state: one canonical '
                'over its same-topic peers, nothing claimed-absorbed still live.'
            )
        else:
            codes = ', '.join(sorted({r['code'] for r in frozen_reasons}))
            message = (
                f'Topic {topic!r} is NOT closed: {codes}. See reasons for the '
                'offending ids.'
            )
    return ClosureVerdict(
        closed=closed,
        reasons=frozen_reasons,
        exit_code=EXIT_CLOSED if closed else EXIT_NOT_CLOSED,
        topic=topic,
        message=message,
        waived=tuple(waived),
    )


def _payload_meta(payload: Any) -> Mapping[str, Any]:
    """The metadata mapping of a scrolled payload, defensively.

    Never raises on a malformed row: this predicate's job is to REPORT
    malformedness, and a predicate that dies on bad input blocks the very
    gates it exists to adjudicate.
    """
    if not isinstance(payload, Mapping):
        return {}
    meta = payload.get('metadata')
    return meta if isinstance(meta, Mapping) else {}


def _payload_id(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ''
    value = payload.get('id')
    return str(value) if value is not None else ''


def evaluate_closure(
    gate_block: Any,
    *,
    members: Sequence[Any],
    scroll_total: int | None,
    scroll_truncated: bool,
    scroll_available: bool,
    unstamped_live_ids: Sequence[str] = (),
) -> ClosureVerdict:
    """Decide whether a consolidation gate's topic cluster may be CLOSED.

    PURE — no I/O.  *members* are payloads already scrolled by the caller
    (``{'id', 'created_at', 'metadata'}``, the shape
    ``MemoryService.get_memories_by_metadata`` returns); the completeness of
    that scroll is disclosed by *scroll_total* / *scroll_truncated* /
    *scroll_available*, which are REQUIRED arguments precisely so a caller
    cannot forget to disclose a partial view.

    THE CENTRAL CORRECTNESS PROPERTY — MEMBERSHIP.  A live entry carrying the
    gate's ``metadata.topic`` is a legitimate Option-C PEER and is NEVER a
    refusal on membership grounds.  Under PRD §3 Option C (ratified by gate
    3200) the target end state is *N short single-claim peers sharing one
    topic, exactly one canonical*, so those peers are the deliverable, not
    residue; "regrowth" of same-topic peers is a category error under C (PRD
    §3 D9a).  A predicate that refused on peer count would make every
    correctly executed consolidation permanently uncloseable — which is also
    why enforcing here subsumes task 3084's proposed auto-close by
    construction.

    It therefore refuses ONLY on cluster malformedness:

    * ``no_canonical`` / ``multiple_canonicals`` — the canonical count is not
      exactly one.  Canonical is ``meta.get('canonical') is True``, a strict
      IDENTITY check, single-sourcing the meaning with
      ``services/topic_anchor.py::select_canonical_payload``; truthiness would
      admit an int ``1`` (``1 == True`` in Python) and let the read path and
      this predicate disagree about what canonical means.

    Every offender is collected rather than short-circuited on the first,
    matching ``server/consolidation.py::validate_consolidate_args``' convention,
    so a curator gets one complete list instead of N refuse-fix-retry rounds.
    """
    block = gate_block if isinstance(gate_block, Mapping) else {}
    topic = str(block.get('topic') or '')

    reasons: list[dict[str, Any]] = []
    # Case-folded so a rendering difference between the scroll row and the
    # stored metadata cannot manufacture a false `absorbed_member_still_live`
    # (nor hide a real one): `is_full_uuid` tolerates case for the same
    # reason — casing is a rendering choice, not a different identifier.
    live_ids = {_payload_id(p).lower() for p in members if _payload_id(p)}

    # --- canonical invariant: exactly one strictly-canonical member -------- #
    canonical_ids = [
        _payload_id(p) for p in members if _payload_meta(p).get('canonical') is True
    ]
    if not canonical_ids:
        reasons.append(
            _reason(
                'no_canonical',
                detail=(
                    f'No member of topic {topic!r} carries '
                    "`metadata.canonical: true` (strict boolean). Option C "
                    'requires exactly one canonical index claim over the peers.'
                ),
            )
        )
    elif len(canonical_ids) > 1:
        reasons.append(
            _reason(
                'multiple_canonicals',
                ids=canonical_ids,
                detail=(
                    f'{len(canonical_ids)} members of topic {topic!r} carry '
                    '`metadata.canonical: true`; Option C permits exactly one. '
                    'Per-topic uniqueness (3198) ships warn-mode-first, so '
                    'duplicates land through ordinary writes.'
                ),
            )
        )

    # --- absorbed-actually-gone: the canonical's supersedes claim ---------- #
    # Only the CANONICAL's claim is the cluster's claim.  A non-canonical
    # peer's stale supersedes is not what the gate asserted, and reading it
    # would refuse gates over other clusters' history.
    if len(canonical_ids) == 1:
        canonical = next(
            p for p in members if _payload_meta(p).get('canonical') is True
        )
        reasons.extend(
            _classify_supersedes(
                _payload_meta(canonical).get('supersedes'),
                live_ids=live_ids,
                topic=topic,
            )
        )

    return _verdict(topic, reasons)


def _classify_supersedes(
    raw: Any,
    *,
    live_ids: set[str],
    topic: str,
) -> list[dict[str, Any]]:
    """Classify each member of the canonical's ``supersedes`` claim.

    Goes through the SHARED :func:`normalize_supersedes` — never a second
    ``supersedes`` parser (INV-5); that helper's own docstring names this
    closure predicate as one of its two designated readers.  It accepts
    ``None`` (nothing absorbed), the legacy SCALAR spelling (81 live records
    predate 3196's migration) as a one-member list, and a list as itself — so
    a bare 36-char uuid string is one member here, never 36 characters.

    It also deliberately never DROPS a malformed member, so this function can
    reject one BY NAME (``malformed_supersedes_member``) rather than raising.
    Raising would permanently block exactly the gates whose metadata is
    already malformed — the census counts 3 short-hex and 8 non-string live.

    Three outcomes per member:

    * not a canonical full uuid -> ``malformed_supersedes_member``
    * well-formed and STILL in the live topic scroll ->
      ``absorbed_member_still_live`` (the curator claimed it was deleted; it
      is not)
    * well-formed and absent -> correctly folded, no reason
    """
    malformed: list[Any] = []
    still_live: list[str] = []
    for member in normalize_supersedes(raw):
        if not is_full_uuid(member):
            malformed.append(member)
        elif str(member).lower() in live_ids:
            still_live.append(str(member))

    reasons: list[dict[str, Any]] = []
    if still_live:
        reasons.append(
            _reason(
                'absorbed_member_still_live',
                ids=still_live,
                detail=(
                    f"The canonical of topic {topic!r} claims these ids in "
                    '`metadata.supersedes`, but the live topic scroll still '
                    'returns them. Either they were never deleted, or they '
                    'must not be claimed as superseded.'
                ),
            )
        )
    if malformed:
        reasons.append(
            _reason(
                'malformed_supersedes_member',
                ids=malformed,
                detail=(
                    f"The canonical of topic {topic!r} carries "
                    '`metadata.supersedes` members that are not canonical '
                    '36-char dashed uuids, so their closure cannot be checked '
                    'against the scroll. Fix or drop them.'
                ),
            )
        )
    return reasons
