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
    is_valid_topic_slug,
    normalize_supersedes,
)
from fused_memory.utils.validation import is_full_uuid

__all__ = [
    'EXIT_CLOSED',
    'EXIT_NOT_CLOSED',
    'GATE_METADATA_KEY',
    'WAIVABLE_REASON_CODES',
    'ClosureVerdict',
    'ConsolidationGateSpec',
    'build_consolidation_gate_task',
    'evaluate_closure',
    'render_consolidation_gate_section',
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


def _reason(
    code: str,
    *,
    ids: Sequence[Any] = (),
    detail: str = '',
    **extra: Any,
) -> dict[str, Any]:
    """One structured refusal entry.  ``ids`` NAMES the offenders.

    *extra* carries code-specific structured facts (e.g. ``scroll_total``) as
    FIELDS rather than buried in prose, so the seam, the CLI and a human all
    read the same number instead of parsing a sentence.
    """
    return {'code': code, 'ids': [str(i) for i in ids], 'detail': detail, **extra}


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

    # --- completeness FIRST, and unconditional ----------------------------- #
    # A predicate whose entire job is refuting a false closure claim must not
    # pass on a view it knows is partial or absent (INV-3: an actor that cannot
    # corroborate must not act; PRD δ: a disagreement is never downgraded to
    # "no children").  Neither of these can be waived — you cannot waive not
    # having looked.
    if not scroll_available:
        # Nothing was SEEN, so nothing about the cluster's content may be
        # asserted and no waiver can be judged against reality.  Returning here
        # rather than falling through is what keeps this from fabricating an
        # absence-based accusation out of an empty *members*.
        return _verdict(
            topic,
            [
                _reason(
                    'scroll_unavailable',
                    detail=(
                        f'The live scroll for topic {topic!r} did not answer, so '
                        'the cluster could not be checked. This is NOT "nothing '
                        'left": an unanswered read and a genuinely empty topic '
                        'are different outcomes (the same three-way distinction '
                        '`build_consolidation_result` draws).'
                    ),
                    scroll_total=scroll_total,
                )
            ],
        )
    if scroll_truncated:
        reasons.append(
            _reason(
                'scroll_incomplete',
                detail=(
                    f'The live scroll for topic {topic!r} hit its cap after '
                    f'{scroll_total} members, and `_next_offset` is discarded, so '
                    'members beyond it were never seen. A partial view cannot '
                    'establish closure.'
                ),
                scroll_total=scroll_total,
            )
        )

    # --- canonical invariant: exactly one strictly-canonical member -------- #
    canonical_ids = [
        _payload_id(p) for p in members if _payload_meta(p).get('canonical') is True
    ]
    if not canonical_ids and not scroll_truncated:
        # ABSENCE-based, so it is only sound on a COMPLETE view: a canonical
        # sitting past the scroll cap is indistinguishable from no canonical at
        # all, and accusing the cluster of a defect that may not exist would be
        # a fabrication.  The gate still refuses — on `scroll_incomplete`, the
        # reason that is actually true.
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
        # PRESENCE-based, so truncation does not weaken it: seeing two
        # canonicals proves two canonicals, and unseen rows can only add more.
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

    # --- unstamped cluster members: the ONE thing provenance may add ------- #
    # A member the detector observed live but which never got stamped into the
    # topic is invisible to the scroll, so it can only reach the predicate this
    # way.  ABSENCE-based (not-stamped is indistinguishable from past-the-cap),
    # so it is suppressed on a truncated view for the same reason
    # `no_canonical` is.
    stray = [str(i) for i in unstamped_live_ids if str(i)]
    if stray and not scroll_truncated:
        reasons.append(
            _reason(
                'unstamped_cluster_member',
                ids=stray,
                detail=(
                    'These ids are live members of the cluster but do not carry '
                    f'`metadata.topic={topic!r}`, so the topic scroll cannot see '
                    'them. Stamp them into the topic or delete them — an '
                    'unstamped member is a member the next reader will not find.'
                ),
            )
        )

    # --- the audited escape, applied LAST ---------------------------------- #
    reasons, waived = _apply_waivers(
        block,
        reasons,
        live_universe=live_ids | {str(i).lower() for i in unstamped_live_ids},
    )
    return _verdict(topic, reasons, waived=waived)


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


#: The ONLY reason codes an audited waiver may suppress. Both are of the form
#: "this specific LIVE entry is a problem" — the judgment call a curator can
#: legitimately have already made. Deliberately NOT waivable: the completeness
#: codes (you cannot waive not having looked), `no_canonical` /
#: `multiple_canonicals` (a cluster-shape defect to FIX, not to excuse), and
#: `malformed_supersedes_member` (a metadata defect to fix). Keeping the set
#: closed and named is what stops the escape becoming a gate kill-switch.
WAIVABLE_REASON_CODES = frozenset(
    {'absorbed_member_still_live', 'unstamped_cluster_member'}
)


def _apply_waivers(
    block: Mapping[str, Any],
    reasons: list[dict[str, Any]],
    *,
    live_universe: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the gate block's ``considered_and_kept`` audit list.

    WHY THE ESCAPE EXISTS.  Gates routinely sit for days — the Stage-1
    stale-gate backlog threshold is 48h
    (``reconciliation/stages/memory_consolidator.py``) — so an entry written
    against the topic by an unrelated agent AFTER the curator made their
    judgment would otherwise leave the gate permanently uncloseable with no
    sanctioned exit.  A predicate with no exit is a predicate operators route
    around.

    WHY IT IS NOT A RUBBER STAMP.  Three properties, each pinned by a test:
    only :data:`WAIVABLE_REASON_CODES` can be suppressed at all; every entry
    needs a non-empty ``note`` or it waives nothing and says so
    (``unaudited_waiver``); and a waiver naming nothing live is reported
    (``stale_waiver``) rather than silently ignored, so a waiver list cannot
    quietly rot into a permanent bypass.  Applied waivers are echoed on
    ``ClosureVerdict.waived`` so the exit is visible in the verdict itself.

    WHY THE KEY IS ``considered_and_kept`` AND NOT ``retain``.  ``retain`` is
    already taken, with a different meaning: it is the INPUT arm of
    ``consolidate_memories`` (gate 3200's ratified default — ids tagged in
    place with the cluster topic, never deleted, never given ``canonical`` or
    ``parent_id``).  Reusing that word for a gate-closure waiver would give one
    term two jobs across two subsystems.  Living inside the Tier-C
    ``x_recon_consolidation_gate`` block also means the key needs no amendment
    to ``RESERVED_VOCABULARY_KEYS`` (a closed five-key set) and produces no
    unknown-key census noise.

    Entry shape: ``{'id', 'note', 'recorded_at', 'recorded_by'}``.
    """
    raw = block.get('considered_and_kept')
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return reasons, []

    applied: list[dict[str, Any]] = []
    unaudited: list[Any] = []
    stale: list[Any] = []
    waived_ids: set[str] = set()

    for entry in raw:
        if not isinstance(entry, Mapping):
            unaudited.append(entry)
            continue
        entry_id = entry.get('id')
        note = entry.get('note')
        if not isinstance(note, str) or not note.strip():
            unaudited.append(entry_id)
            continue
        if str(entry_id).lower() not in live_universe:
            stale.append(entry_id)
            continue
        waived_ids.add(str(entry_id).lower())
        applied.append(dict(entry))

    kept: list[dict[str, Any]] = []
    for reason in reasons:
        if reason['code'] not in WAIVABLE_REASON_CODES:
            kept.append(reason)
            continue
        remaining = [i for i in reason['ids'] if i.lower() not in waived_ids]
        if not remaining:
            continue  # every offender in this reason was audited and kept
        kept.append({**reason, 'ids': remaining})

    if unaudited:
        kept.append(
            _reason(
                'unaudited_waiver',
                ids=unaudited,
                detail=(
                    'Each `considered_and_kept` entry needs a non-empty `note` '
                    'recording WHY the entry was kept. A bare id list is not an '
                    'audit trail, so these entries waived nothing.'
                ),
            )
        )
    if stale:
        kept.append(
            _reason(
                'stale_waiver',
                ids=stale,
                detail=(
                    'These `considered_and_kept` entries name ids that are not '
                    'live, so the waiver no longer describes reality. Remove '
                    'them rather than leaving a list that silently grants '
                    'nothing.'
                ),
            )
        )
    return kept, applied


# --------------------------------------------------------------------------- #
# The recon-side filing convention — a direct sibling of
# ``predicate_contradiction.py``'s frozen-spec + as_submit_task_kwargs() +
# build_*_task() + render_*_section() template.
# --------------------------------------------------------------------------- #

_GATE_EXECUTION_CLASS = 'operational'
_GATE_OPERATIONAL_MODE = 'gate'


@dataclass(frozen=True)
class ConsolidationGateSpec:
    """An assembled, not-yet-submitted consolidation gate.

    ``as_submit_task_kwargs()`` returns a dict splattable directly into
    ``mcp__fused-memory__submit_task`` (the caller adds ``project_root``),
    matching ``predicate_contradiction.py::PredicateContradictionTask``.
    """

    title: str
    description: str
    priority: str
    task_kind: str
    metadata: dict[str, Any]

    def as_submit_task_kwargs(self) -> dict[str, Any]:
        """Return the submit_task keyword arguments for this gate."""
        return {
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'task_kind': self.task_kind,
            'metadata': self.metadata,
        }


def _default_title(topic: str) -> str:
    return f'Consolidation gate: topic {topic}'


def _default_description(topic: str, rationale: str) -> str:
    parts = [
        'A memory consolidation reached an irreversible, content-losing '
        'judgment call that must not be resolved silently. This is the human '
        f'gate for topic `{topic}`.\n',
        'The WORKING LIST is the live `metadata.topic` scroll for that topic — '
        'never a hand-written member list. Any enumeration carried in this '
        "task's metadata is inert provenance (`authoritative: false`) and "
        'cannot make this gate closeable.\n',
    ]
    if rationale.strip():
        parts.append(f'Judgment call: {rationale.strip()}\n')
    parts.append(
        'This gate CANNOT be closed while the live same-topic scroll shows the '
        'cluster is not well-formed: `set_task_status(..., done)` re-runs the '
        'check and refuses with the offending ids. Run '
        '`scripts/check_consolidation_closure.py` to see the same verdict by '
        'hand.\n'
    )
    parts.append(render_end_state_brief())
    return '\n'.join(parts)


def build_consolidation_gate_task(
    *,
    topic: str,
    rationale: str = '',
    observed_members: Sequence[str] = (),
    report_run: str | None = None,
    detector: str | None = None,
    authoritative: bool = False,
    considered_and_kept: Sequence[Mapping[str, Any]] = (),
    priority: str = 'medium',
    title: str | None = None,
    description: str | None = None,
) -> ConsolidationGateSpec:
    """Build the ``operational`` + ``operational_mode='gate'`` consolidation gate.

    PURE — no filesystem or network I/O.

    The submission is a DETERMINISTIC PURE GATE: ``task_kind='deterministic'``
    with ``always_escalates=True`` and no ``before_done``.  That combination is
    not a stylistic choice — ``deterministic_task_guard`` invariant 2 rejects a
    deterministic task that neither runs an action nor escalates, and invariant
    3b rejects ``always_escalates`` on a normal task, so it is the only legal
    shape for a pure human gate.  It is also exactly what
    ``TaskInterceptor._inject_deterministic_pure_gate`` produces, so the
    declared submission and the post-coercion task agree either way.  No
    ``before_done`` is referenced: that coercion unconditionally pops one, and
    ``docs/task-authoring.md`` records that ``before_done.kind='predicate'``
    forbids ``always_escalates=True``, so a pure gate structurally cannot carry
    one.

    *observed_members* / *report_run* / *detector* are INERT PROVENANCE in leaf
    kappa's (task 3136) report shape.  ``authoritative`` is hard-set to
    ``False`` whatever the caller passes: the live topic scroll is the sole
    working list, and provenance can only ever ADD a refusal (an id live but
    never stamped into the topic), never grant a pass.  That is what makes
    "inert" a structural property rather than a promise — pinned by the test
    asserting :func:`evaluate_closure` returns an identical verdict with and
    without it.  Contrast DF gate 3036, whose hand-written enumeration under an
    invented ``metadata.memory_ids`` key was extended 7 to 8 by a later cycle
    while it still defined "done".

    Raises:
        ValueError: if *topic* is not a well-formed topic slug.  Validated
            through the shared :func:`is_valid_topic_slug` — one namespace,
            shared with ``ProceduralTopicCluster.topic_id`` (PRD D4) — because
            a gate filed against a topic the closure scroll could never match
            would be permanently uncloseable.
    """
    if not is_valid_topic_slug(topic):
        raise ValueError(
            f'topic={topic!r} is not a well-formed topic slug, so the closure '
            'scroll could never match its cluster and the gate would be '
            'permanently uncloseable. Topics are lowercase alphanumeric '
            'hyphen-separated (see fused_memory.topic_slug.is_valid_topic_slug '
            '— one namespace shared with ProceduralTopicCluster.topic_id).'
        )

    gate_block: dict[str, Any] = {'topic': topic}
    if considered_and_kept:
        gate_block['considered_and_kept'] = [dict(e) for e in considered_and_kept]
    if observed_members or report_run or detector:
        gate_block['provenance'] = {
            'report_run': report_run,
            'observed_members': [str(m) for m in observed_members],
            'detector': detector,
            # Hard-set, never read from the caller: see the docstring.
            'authoritative': False,
        }

    metadata: dict[str, Any] = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'execution_class': _GATE_EXECUTION_CLASS,
        'operational_mode': _GATE_OPERATIONAL_MODE,
        GATE_METADATA_KEY: gate_block,
    }

    return ConsolidationGateSpec(
        title=title or _default_title(topic),
        description=description or _default_description(topic, rationale),
        priority=priority,
        task_kind='deterministic',
        metadata=metadata,
    )


def render_consolidation_gate_section(*, can_file_tasks: bool) -> str:
    """Render the consolidation-gate filing-convention prompt section (task
    3112), mirroring ``render_predicate_contradiction_section``'s style.

    Interpolated into BOTH stage system prompts. Both stages hold the
    memory-mutation tools, so either can REACH a gate-worthy judgment call —
    but only Stage 2 holds ``submit_task``, so only the HOW clause is
    parameterized on *can_file_tasks*: Stage 2 (``True``) builds and files the
    gate itself; Stage 1 (``False``) relays it to Stage 2 through the
    ``flag_for_stage2`` / ``flagged_items`` channel. ``flagged_items`` is the
    DURABLE half of that channel — the Mem0 ``flag_for_stage2=true`` write
    alone is best-effort — which is what stage1.py already tells the stage
    about flag delivery.

    ``reconciliation.cli_stage_runner::STAGE1_DISALLOWED`` is the source of
    truth for what Stage 1 cannot call; it is cited rather than restated here,
    for the reason :func:`recon_self_model.render_source_completion_section`
    gives in its own docstring — a mirrored inventory goes stale silently. The
    sibling :func:`predicate_contradiction.render_predicate_contradiction_section`
    is Stage-2-only for exactly this reason, so this is the third instance of
    one convention rather than a new one.

    Everything OTHER than the HOW clause is shared and byte-identical across
    both variants: the end state does not depend on which stage files the gate.
    It reuses :func:`render_end_state_brief` VERBATIM, so the prompt, the filed
    gate's description and the closure predicate's target are one text and
    cannot drift into disagreeing.
    """
    if can_file_tasks:
        how_clause = (
            'HOW: you hold `submit_task` in this stage, so file it yourself. '
            "Build the submission with `build_consolidation_gate_task("
            "topic='<slug>', rationale=...)` (module "
            '`fused_memory.reconciliation.consolidation_gate`) and submit its '
            '`as_submit_task_kwargs()`. It emits an '
            "`execution_class='operational'` + `operational_mode='gate'` task "
            f'whose `metadata.{GATE_METADATA_KEY}.topic` carries the cluster '
            'TOPIC.'
        )
    else:
        how_clause = (
            'HOW: you do NOT hold submit_task in this stage (task writes are '
            'disallowed here via DISALLOW_TASK_WRITES), so you cannot file the '
            'gate here. Relay it to Stage 2 via the `flag_for_stage2` / '
            '`flagged_items` channel, carrying the cluster\'s topic SLUG and '
            'the rationale, so Stage 2 can build it with '
            "`build_consolidation_gate_task(topic='<slug>', rationale=...)` "
            '(module `fused_memory.reconciliation.consolidation_gate`) and '
            "file it as an `execution_class='operational'` + "
            "`operational_mode='gate'` task whose "
            f'`metadata.{GATE_METADATA_KEY}.topic` carries that TOPIC. The '
            'topic slug and the rationale are the whole payload Stage 2 needs '
            '— relay them, do not relay a member list (see below).'
        )
    return (
        '## Consolidation Gate\n'
        'When a memory consolidation reaches an irreversible, content-losing '
        'judgment call, file it as a human gate — but file it with a TARGET '
        'and a working key, not just a description of the problem. A gate that '
        'does not say what "done" looks like gets a different answer from '
        'every cycle that reads it.\n\n' + how_clause + '\n\n'
        'EMIT THE TOPIC, NOT A MEMBER LIST. The live `metadata.topic` scroll is '
        'the working list. A hand-written enumeration is accepted only as inert '
        "provenance (`authoritative: false`) and can never make a gate "
        'closeable — a hand-written list goes stale the moment the next cycle '
        'writes to the cluster, and a stale list that still defines "done" is '
        'worse than no list.\n\n'
        'THE GATE REFUSES TO CLOSE OVER A MALFORMED CLUSTER. '
        '`set_task_status(<gate>, done)` re-runs the closure check against the '
        'live same-topic scroll and REFUSES — naming the offending ids — when '
        'the cluster is not well-formed, when an id claimed in '
        '`supersedes` is still live, or when the scroll could not be completed. '
        'Surviving same-topic PEERS are never a refusal: they are the target '
        'end state. If a live entry was considered and deliberately kept, '
        f'record it in `metadata.{GATE_METADATA_KEY}.considered_and_kept` as '
        '`{id, note, recorded_at, recorded_by}` — the `note` is mandatory, and '
        'a waiver without one waives nothing.\n\n' + render_end_state_brief()
    )
