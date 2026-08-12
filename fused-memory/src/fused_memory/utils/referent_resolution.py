"""Referent-set resolution: WHICH source is authoritative for one write.

The PRECEDENCE POLICY layer over utils/canonical_labels. That module stays THE
single normative site for the label VOCABULARY (INV-5 / PRD resolved decision
5); this one decides which of the available referent sources speaks for a given
write, and reports what the prose contradicted.

Precedence, strictly, one authoritative source per resolution::

    declared  >  metadata.task_id  >  derived scan  >  none

``.source`` is SINGULAR because exactly one source wins — this is an override
chain, not a union. A caller that declared its referents is believed over
ambient harness metadata, which is believed over what the prose happens to say.

This module compiles NO regex of its own and must never grow one. The derived
path IS :func:`~fused_memory.utils.canonical_labels.scan_content`, the declared
path builds :class:`~fused_memory.utils.canonical_labels.Referent` objects, and
the metadata bridge tries
:func:`~fused_memory.utils.canonical_labels.parse_node_name` before its single
bare-digit branch. A second copy of the label pattern here would be exactly the
lockstep duplication INV-5 forbids, and the drift would be invisible until a
destructive consumer acted on the stale half.

KNOWN BLIND SPOTS, inherited from ``scan_content`` and restated so no reader
assumes completeness (PRD resolved decision 8): a node named with bare digits
('1251'), a reference made by task TITLE rather than number, and Greek-letter
or codename aliases ('Task θ2=2184') are all invisible to the DERIVED path by
design. Precision over recall — consumers perform destructive edge surgery, so
a false positive misattributes a fact. The load-bearing consequence is in
:func:`resolve_referents`: an empty scan is UNINFORMATIVE, never contradictory,
so it can never produce a conflict and can never reject an honest write.

This module is a dependency-free leaf — stdlib plus utils/canonical_labels and
utils/validation, both themselves leaves — so leaf δ (``server/tools.py``) and
leaf ε (``services/memory_service.py``) can each import it without a cycle.
Mirrors utils/cross_project_refs.py, the pure policy consumer of
canonical_labels whose shape it copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fused_memory.utils.canonical_labels import Referent, parse_node_name, scan_content
from fused_memory.utils.validation import (
    InputValidationError,
    PathShapedProjectIdError,
    _safe_repr,
    canonicalize_project_id,
)

#: Remediation text for a rejected ``declared`` entry. Single-sourced here and
#: folded into the raised message, following ``require_full_uuid``
#: (validation.py): an exception has no room for a separate structured key, so
#: the accepted shape has to travel with the error itself.
_DECLARED_REFERENT_HINT = (
    "Each declared referent must be a dict of the shape {'kind': 'task', "
    "'id': <digits>, 'project_id': <optional project key>} — 'kind' and "
    "'project_id' are optional and default to 'task' and the local project. "
    "'id' must be the task number's digits (an int, or a string of ASCII "
    'digits), never a label like "Task 3127" and never a bool.'
)

#: The complete set of keys a declared entry may carry. Closed deliberately:
#: an unrecognized key ('projectId') would otherwise be silently ignored and
#: resolve to an OWN-project referent — a confidently wrong answer about which
#: project a fact belongs to.
_DECLARED_ENTRY_KEYS = frozenset({'kind', 'id', 'project_id'})

#: The CLOSED vocabulary of resolution sources, in precedence order (strongest
#: first). Exported as one tuple so task ι's declaration-rate telemetry
#: iterates a single site rather than re-spelling four string literals; a
#: second copy would drift the same way two copies of the label vocabulary do.
#:
#: - ``'declared'``  — the caller stated its referents explicitly, INCLUDING
#:   the empty declaration ``[]`` ("considered, none apply").
#: - ``'metadata'``  — bridged from ambient ``metadata['task_id']``.
#: - ``'derived'``   — scanned out of the content by ``scan_content``.
#: - ``'none'``      — nothing declared, bridged or derivable. A real, COUNTED
#:   outcome, not a missing value.
REFERENT_SOURCES: tuple[str, ...] = ('declared', 'metadata', 'derived', 'none')

#: The type twin of :data:`REFERENT_SOURCES`. Single-sourced against it below
#: so the constant and the type cannot drift apart.
ReferentSource = Literal['declared', 'metadata', 'derived', 'none']

#: A resolved set of referents.
#:
#: Named HERE, at the producer, because the PRD's ζ signature
#: (``_verify_episode_referents(..., referents: ReferentSet)``) names this type
#: and nothing in the tree defines it — naming it at the producer stops ζ
#: inventing a competing spelling, the same INV-5 pressure that produced β.
#:
#: A SET by content — de-duplicated on ``(kind, project_id, number)`` with
#: first-seen order preserved, the same key and discipline ``scan_content``
#: uses — and a TUPLE by type, for :class:`LabelScan`'s stated reason: a list
#: would stay ``.append()``-able on an object that is evidence for destructive
#: graph surgery.
ReferentSet = tuple[Referent, ...]


@dataclass(frozen=True, kw_only=True)
class ReferentResolution:
    """What one write's referents resolved to, and from where.

    Frozen for the same reason :class:`Referent` and :class:`LabelScan` are: a
    resolution is EVIDENCE for destructive edge surgery, not a mutable
    accumulator. ``frozen=True`` blocks attribute rebinding only, which is why
    every referent field is a :data:`ReferentSet` (a tuple) rather than a list
    — otherwise ``resolution.referents.append(...)`` would let a consumer
    quietly add a referent the resolver refused to infer.

    Keyword-only because four same-shaped referent-ish fields read as noise
    positionally, and a call site that swapped ``conflicts`` for ``ambiguous``
    would type-check fine.
    """

    #: Which source was authoritative. REQUIRED — no default — because
    #: ``.source`` must be set on EVERY resolution including the empty one
    #: (``source='none'``), and a required field makes omission structurally
    #: impossible rather than merely inconvenient. One of
    #: :data:`REFERENT_SOURCES`.
    source: ReferentSource
    #: The resolved referents, from the authoritative source only. Empty is a
    #: legitimate answer for every source; an empty set carries nothing to
    #: verify membership against, so a downstream verifier must no-op on it
    #: regardless of ``.source``.
    referents: ReferentSet = ()
    #: DECLARED referents the scanned content contradicts — a declared referent
    #: of kind K whose kind the scan did see, but which the scan did not name.
    #:
    #: Populated on the ``'declared'`` path ONLY. Ambient ``metadata.task_id``
    #: is not a claim about the prose (an agent working on task 3668
    #: legitimately writes memories about Task 2500), and the derived path IS
    #: the scan, so neither can contradict it.
    #:
    #: This module REPORTS; it never rejects and never degrades ``.referents``
    #: because of a conflict. Whether a conflict rejects the write is leaf δ's
    #: gate to decide (``_entities_gate`` in ``server/tools.py``) — a resolver
    #: that silently fell back to the scan here would produce a write the
    #: caller never asked for.
    conflicts: ReferentSet = ()
    #: Referents the CONTENT is genuinely ambiguous about — a number claimed
    #: both by a bare own-project mention and by a foreign-qualified reference.
    #: Reported verbatim from the scan whatever the winning source, and NEVER
    #: promoted into :attr:`referents`: ambiguity is recorded, not guessed.
    ambiguous: ReferentSet = ()


def _declared_number(entry: dict) -> str:
    """Return the entry's task number as verbatim digits, or raise.

    Digits are never int-normalized: ``'0132'`` is a DIFFERENT referent from
    ``'132'``, so coercing would silently repoint the caller's own assertion.
    """
    if 'id' not in entry:
        raise InputValidationError(
            f"declared referent {_safe_repr(entry)} is missing required key 'id'. "
            f'{_DECLARED_REFERENT_HINT}'
        )
    value = entry['id']
    # bool FIRST: bool subclasses int, so {'id': True} would otherwise become
    # Task 1. The lesson is already encoded in validation._is_plain_int.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InputValidationError(
            f'declared referent {_safe_repr(entry)} has a non-integer id '
            f'{_safe_repr(value)}. {_DECLARED_REFERENT_HINT}'
        )
    number = str(value).strip()
    # String predicates rather than a compiled '^\\d+$': INV-5 forbids a second
    # copy of the label vocabulary in this module. The isascii() half is
    # precision, not ceremony — str.isdigit() accepts Arabic-Indic and
    # superscript digits, and a Unicode digit is not a task id.
    if not number or not (number.isascii() and number.isdigit()):
        raise InputValidationError(
            f'declared referent {_safe_repr(entry)} has an id that is not a '
            f'run of ASCII digits: {_safe_repr(value)}. {_DECLARED_REFERENT_HINT}'
        )
    return number


def _declared_referents(declared: list[dict], *, group_id: str) -> ReferentSet:
    """Parse the caller's explicit referent entries into a referent set.

    Entry shape is ``{'kind': 'task', 'id': <digits>, 'project_id': <optional>}``.
    ``kind`` defaults to ``'task'`` — the same default :class:`Referent` itself
    carries, so canonical_labels stays the single site for it.

    A ``project_id`` equal to the canonicalized *group_id* is RECLASSIFIED to
    an own-project referent, mirroring ``scan_content``'s self-qualified
    reclassification EXACTLY. Without that, a self-qualified declaration could
    never compare equal to the scanned form and the conflict check would fire
    on a caller who was right.

    De-duplicated on ``(kind, project_id, number)`` with first-seen order
    preserved — the same key and the same discipline ``scan_content`` uses.

    A malformed entry RAISES :class:`InputValidationError`; it is never
    dropped and never smuggled into ``.conflicts``. Dropping would flip
    ``.source`` down to a lower source, lose the caller's intent invisibly and
    corrupt the declaration-rate telemetry ι reads — the silent degradation
    this repo's loud-over-silent norm forbids. Reporting it as a conflict
    would tell the agent its declaration contradicts its prose and send it
    hunting a semantic disagreement that does not exist.
    """
    if not isinstance(declared, list):
        raise InputValidationError(
            f'declared referents must be a list, got {_safe_repr(declared)}. '
            f'{_DECLARED_REFERENT_HINT}'
        )
    try:
        local_project = canonicalize_project_id(group_id)
    except PathShapedProjectIdError:
        # scan_content answers an empty scan for a path-shaped group_id; here
        # there is still a declaration to honour, so only the self-qualified
        # RECLASSIFICATION is unavailable. Comparing against a sentinel no
        # canonical project id can equal keeps a foreign declaration foreign
        # rather than silently collapsing it onto the local project.
        local_project = ''

    referents: list[Referent] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in declared:
        if not isinstance(entry, dict):
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} is not a dict. '
                f'{_DECLARED_REFERENT_HINT}'
            )
        if unknown := set(entry) - _DECLARED_ENTRY_KEYS:
            # Rejected, not ignored: 'projectId' would otherwise resolve
            # silently to an own-project referent.
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has unrecognized '
                f'key(s) {sorted(unknown)}. {_DECLARED_REFERENT_HINT}'
            )
        number = _declared_number(entry)

        raw_project = entry.get('project_id') or ''
        try:
            project_id = canonicalize_project_id(raw_project) if raw_project else ''
        except PathShapedProjectIdError as exc:
            # Never canonicalized: normalizing a mangled path would mint a
            # NEW, wrong canonical key (RCA §4), and the referent would then
            # confidently name a project that does not exist.
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has a path-shaped '
                f'project_id: {exc} {_DECLARED_REFERENT_HINT}'
            ) from exc
        if project_id and local_project and project_id == local_project:
            project_id = ''

        try:
            referent = Referent(
                kind=entry.get('kind', 'task'),
                number=number,
                project_id=project_id,
            )
        except ValueError as exc:
            # An unregistered kind. Re-raised as InputValidationError so δ's
            # boundary sees ONE exception type, preserving __post_init__'s
            # message, which already names the registered kinds and where to
            # add one.
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} is invalid: {exc} '
                f'{_DECLARED_REFERENT_HINT}'
            ) from exc
        key = (referent.kind, referent.project_id, referent.number)
        if key in seen:
            continue
        seen.add(key)
        referents.append(referent)
    return tuple(referents)


def _metadata_referents(metadata: dict) -> ReferentSet:
    """Bridge ambient ``metadata['task_id']`` into a referent set, or ``()``.

    An unusable value is DROPPED, never raised on. This is asymmetric with the
    ``declared`` path (which raises — see :func:`_declared_referents`) and the
    asymmetry is deliberate: ``declared`` is an explicit caller assertion,
    while ``metadata`` is ambient harness state the writer may not control, and
    failing a write over odd ambient metadata would lose memories for no gain.
    The degradation stays OBSERVABLE rather than silent — ι sees a lower
    'metadata' share in its source counts, and the caller still falls THROUGH
    to the derived scan rather than short-circuiting to ``'none'``.

    ``_normalize_task_id_metadata`` (services/memory_service.py) documents this
    path as carrying a SCALAR int or str: a list or tuple is out of contract
    there precisely because it would ``str()``-coerce to a Python repr, so a
    non-scalar is dropped here rather than coerced into a garbage referent
    like ``'[5040, 5149]'``.
    """
    value = metadata.get('task_id')
    # bool FIRST: bool subclasses int, so {'task_id': True} would otherwise
    # become Task 1. The lesson is already encoded in validation._is_plain_int.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return ()
    text = str(value).strip()
    if not text:
        return ()

    # β's anchored parser owns every spelling that IS a label ('Task 3127',
    # 'task #3127', 'reify:3127' with the qualifier preserved as a
    # different-project signal), so reuse costs nothing and keeps the
    # vocabulary at one site.
    referent = parse_node_name(text)
    if referent is not None:
        return (referent,)

    # Exactly ONE shape of our own: the bare digit run that metadata.task_id
    # actually carries, and the one parse_node_name is anchored to refuse.
    # String predicates rather than a compiled '^\d+$' because INV-5 forbids a
    # second copy of the label vocabulary in this module. The isascii() half is
    # precision, not ceremony: str.isdigit() accepts Arabic-Indic and
    # superscript digits, and a Unicode digit is not a task id.
    if text.isascii() and text.isdigit():
        return (Referent(kind='task', number=text),)
    return ()


def resolve_referents(
    *,
    declared: list[dict] | None,
    metadata: dict,
    content: str,
    group_id: str,
) -> ReferentResolution:
    """Resolve which referents one write is about, and from which source.

    Precedence, strictly — ``declared`` > ``metadata['task_id']`` > derived
    scan > ``none``. Exactly ONE source is authoritative per resolution, which
    is why :attr:`ReferentResolution.source` is singular: this is an override
    chain, not a union of everything available.

    An EMPTY :attr:`~ReferentResolution.referents` carries nothing to test
    membership against, so a downstream verifier must no-op on it regardless
    of ``.source`` — already the PRD's stated behaviour for the
    ``source='none'`` row ("no repair attempted"), and it extends unchanged to
    every other source that resolves empty.

    Args:
        declared: The caller's explicit referent entries, or None. TRI-STATE:
            None means "never considered" and falls through; ``[]`` means
            "considered, none apply" and is honoured as a declaration.
        metadata: The write's ambient metadata; ``task_id`` is bridged from it.
        content: The verbatim body to scan.
        group_id: The group the content belongs to (= the local project_id).

    These are the PRD's exact four parameters and no more. In particular there
    is deliberately no ``known_project_ids``: ``scan_content`` is called in its
    documented PERMISSIVE mode, and threading a live project registry is a
    wiring concern belonging to the leaf that owns the wiring (δ/ε). Adding an
    unspecified fifth parameter here would fork, mid-batch, the signature those
    siblings are being written against.
    """
    # ALWAYS scan, unconditionally, BEFORE any precedence branching.
    #
    # Do not "optimize" this into the derived branch. Two consumer-visible
    # fields need the scan whatever source wins: `.conflicts` is DEFINED as
    # declared-versus-scan, and `.ambiguous` must read identically whether the
    # effective referents came from a declaration, from metadata, or from the
    # scan. A caller forced to check `.source` before trusting `.ambiguous`
    # would have to re-derive the scan itself — a second scan site, and exactly
    # the INV-5 lockstep duplication canonical_labels exists to prevent.
    #
    # The cost is two regex passes over already-in-memory content, negligible
    # against the LLM-composed Graphiti write this precedes.
    scan = scan_content(content, group_id=group_id)

    # `.ambiguous` is the scan's verbatim answer on every path. Ambiguous
    # referents are recorded, never guessed, and never promoted into
    # `.referents`.
    ambiguous = scan.ambiguous

    # An explicit declaration is the strongest source and is honoured verbatim.
    if declared:
        return ReferentResolution(
            source='declared',
            referents=_declared_referents(declared, group_id=group_id),
            ambiguous=ambiguous,
        )

    # Ambient metadata outranks the prose but is never reconciled against it:
    # `.conflicts` stays empty on this path, because metadata is not a CLAIM
    # about the content. An agent working on task 3668 legitimately writes
    # memories about Task 2500, so reconciling the two would reject a large
    # fraction of correct writes.
    bridged = _metadata_referents(metadata)
    if bridged:
        return ReferentResolution(source='metadata', referents=bridged, ambiguous=ambiguous)

    # scan.refs already excludes the ambiguous ones, so the derived set needs
    # no further filtering here.
    if scan.refs:
        return ReferentResolution(source='derived', referents=scan.refs, ambiguous=ambiguous)

    return ReferentResolution(source='none', ambiguous=ambiguous)
