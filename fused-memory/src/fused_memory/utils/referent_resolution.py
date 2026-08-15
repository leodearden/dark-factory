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

Precedence selects the AUTHORITY, never the IDENTITY. The self-qualified
reclassification — a reference qualified with the LOCAL project is an
own-project referent, not a foreign one — is SOURCE-INVARIANT: all three paths
apply it, through the one :func:`_local_project` helper, so a given spelling
denotes the same node whichever source happens to win. A path that skipped it
would hand a consumer a ``node_name`` for a foreign node that does not exist
inside the local graph, which is the same misattribution failure arriving
through the identity axis instead of the attribution one.

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
from typing import Literal, get_args

from fused_memory.utils.canonical_labels import (
    LabelScan,
    Referent,
    is_task_vocabulary_qualifier,
    parse_node_name,
    scan_content,
)
from fused_memory.utils.validation import (
    InputValidationError,
    PathShapedProjectIdError,
    _safe_repr,
    canonicalize_project_id,
    validate_project_id,
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
ReferentSource = Literal['declared', 'metadata', 'derived', 'none']

#: The CLOSED vocabulary of resolution sources, in precedence order (strongest
#: first) — see :data:`ReferentSource` above for what each one means.
#:
#: DERIVED from the Literal rather than re-spelled beside it, so the runtime
#: constant and the static type cannot drift apart: adding a fifth source means
#: editing exactly one line. Exported so task ι's declaration-rate telemetry
#: iterates a single site instead of re-spelling four string literals — a
#: second copy would drift the same way two copies of the label vocabulary do.
REFERENT_SOURCES: tuple[str, ...] = get_args(ReferentSource)

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


def _local_project(group_id: str) -> str:
    """The canonical project id *group_id* names, or ``''`` if it names none.

    The single site for "is this qualifier actually US?", shared by
    :func:`_declared_referents` and :func:`_metadata_referents` so the
    self-qualified reclassification cannot drift between them. Two copies of
    the rule is the lockstep duplication this module's docstring forbids under
    INV-5 — and is demonstrably how the metadata path came to skip it.

    ``scan_content`` answers an EMPTY scan for a path-shaped *group_id*; here
    there is still a declaration or an explicit metadata value to honour, so
    only the self-qualified RECLASSIFICATION is unavailable. The ``''`` return
    is a sentinel no canonical project id can equal, and both call sites guard
    on it (``referent.project_id and local``) so an untrustworthy local project
    id keeps a FOREIGN referent foreign rather than silently collapsing it onto
    the local project — strictly worse than the reclassification being missing.
    """
    try:
        return canonicalize_project_id(group_id)
    except PathShapedProjectIdError:
        return ''


def _conflicting_referents(declared: ReferentSet, scan: LabelScan) -> ReferentSet:
    """The declared referents the scanned content CONTRADICTS.

    A declared referent ``d`` conflicts iff the scan found at least one
    referent of ``d.kind`` and ``d`` is not among them. Declared order is
    preserved; the result is de-duplicated.

    Three load-bearing choices, each invisible from the code alone:

    1. KIND-SCOPED, not global. ``_KIND_LABELS`` holds one entry today, so the
       scoping is a no-op and NO test can construct a second kind — the guard
       therefore has to be structural, exactly as β's kind-keyed ambiguity
       partition is. The moment 'escalation' joins the registry, a global rule
       would let a bare 'escalation 2500' mention corroborate (or contradict)
       an unrelated TASK referent.
    2. Membership is tested against ``refs | ambiguous``, so a declaration that
       names an AMBIGUOUS referent counts as RESOLVING the ambiguity rather
       than contradicting it — the single most useful thing a declaration can
       do.
    3. An EMPTY per-kind scan never conflicts. This is the load-bearing
       consequence of resolved decision 8: the scanner cannot see bare-digit
       node names, title-only references or Greek-letter aliases, so its
       SILENCE is uninformative and must not reject an honest write.

    The rejected alternative — a set-level "declared and scanned are disjoint"
    verdict — is more permissive but stops catching ``[3127, 3129]`` where 3129
    is an adjacent-number typo and 3127 corroborates, which is precisely the
    misattribution this PRD exists to prevent. A per-ref list is also strictly
    more informative: δ can derive the disjoint verdict from it, not the
    reverse.
    """
    scan_by_kind: dict[str, set[Referent]] = {}
    for referent in scan.refs + scan.ambiguous:
        scan_by_kind.setdefault(referent.kind, set()).add(referent)

    conflicts: list[Referent] = []
    seen: set[Referent] = set()
    for referent in declared:
        scanned = scan_by_kind.get(referent.kind)
        if not scanned or referent in scanned:
            continue
        if referent in seen:
            continue
        seen.add(referent)
        conflicts.append(referent)
    return tuple(conflicts)


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

    That contract is TOTAL, not best-effort: EVERY malformed entry — wrong
    container, non-dict element, unknown key, bad ``id``, bad ``kind`` TYPE,
    bad ``project_id`` TYPE, unregistered ``kind``, path-shaped ``project_id``,
    task-VOCABULARY ``project_id``, charset-invalid ``project_id``
    — leaves this function as exactly ``InputValidationError``, so δ's
    ``except InputValidationError`` gate catches all of them. ``declared`` is
    caller-supplied JSON off an MCP tool argument, so every shape is reachable.

    The ``kind``/``project_id`` type checks are deliberately SEPARATE from the
    ``except ValueError`` around the :class:`Referent` construction rather than
    folded into it. ``__post_init__``'s ``self.kind not in _KIND_LABELS``
    raises ``TypeError`` for an unhashable kind, and ``TypeError`` is not a
    ``ValueError``, so that handler structurally cannot see it. The two-layer
    defence is deliberate: a *str* kind still falls through to
    ``__post_init__``, whose message names the registered kinds and where to
    add one.
    """
    if not isinstance(declared, list):
        raise InputValidationError(
            f'declared referents must be a list, got {_safe_repr(declared)}. '
            f'{_DECLARED_REFERENT_HINT}'
        )
    local_project = _local_project(group_id)

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

        # Checked BEFORE the Referent(...) construction below, not folded into
        # the `except ValueError` around it: __post_init__'s membership test
        # `self.kind not in _KIND_LABELS` raises TypeError for an UNHASHABLE
        # kind, and TypeError is not a ValueError, so that handler structurally
        # cannot convert it. The two layers are deliberate, not redundant — and
        # a str kind must still fall through to __post_init__ so the
        # unregistered-kind message keeps naming _KIND_LABELS and where to add
        # one. (bool needs no separate check: it is not a str.)
        kind = entry.get('kind', 'task')
        if not isinstance(kind, str):
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has a non-string kind '
                f'{_safe_repr(kind)}. {_DECLARED_REFERENT_HINT}'
            )

        # Read the RAW value and type-check it BEFORE the `or ''`
        # normalization. Validating the post-`or` value would still let every
        # FALSY non-str — 0, [], {}, False — through as an OWN-project
        # referent: the identical silently-wrong-project outcome the unknown-key
        # ('projectId') rejection above already guards against, arriving through
        # a different door. None and '' stay accepted as legitimate absence.
        raw_project = entry.get('project_id')
        if raw_project is not None and not isinstance(raw_project, str):
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has a non-string '
                f'project_id {_safe_repr(raw_project)}. {_DECLARED_REFERENT_HINT}'
            )
        raw_project = raw_project or ''
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
        # canonicalize_project_id is a NORMALIZER, not a validator — it says so
        # itself and defers the charset allowlist to validate_project_id. Left
        # at just the normalize, '   ', 'foo bar', '..' and 'x;y' all passed
        # through verbatim and minted a FOREIGN referent naming a project that
        # cannot exist ('   :132'), reaching a consumer that performs
        # destructive edge surgery keyed on node_name.
        #
        # validate_project_id, not a fresh charset test here: it is already THE
        # site for the allowlist (require_project_id is a thin raise-wrapper
        # over it), and a second copy is the lockstep duplication INV-5 forbids.
        # Its error is re-raised WRAPPED rather than propagated bare so this
        # rejection names the offending entry and carries the remediation hint,
        # exactly like every other rejection in this TOTAL contract.
        #
        # Deliberately NOT length-capped: validation.py enforces charset and
        # non-emptiness only, and inventing a cap at this site would fork the
        # rule. If a cap is ever wanted it belongs there.
        if project_id and (err := validate_project_id(project_id)):
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has an invalid '
                f'project_id {_safe_repr(raw_project)}: {err["error"]} '
                f'{_DECLARED_REFERENT_HINT}'
            )
        # A task-VOCABULARY qualifier is not a project id. canonical_labels
        # refuses it in BOTH parse_node_name and scan_content, so without this
        # the declared path was the ONE source that accepted it — and
        # {'id': 2500, 'project_id': 'task'} minted a FOREIGN 'task:2500'
        # node while the identical spelling resolved to the LOCAL 'Task 2500'
        # through the other two paths. That is the source-invariance break
        # this module's docstring forbids, arriving on the identity axis: a
        # consumer keyed on node_name would perform edge surgery against a
        # node that cannot exist. Same failure shape the metadata bridge was
        # fixed for in b2cfe5518f, through the remaining path.
        #
        # REJECTED, not reclassified to the local project. On the other two
        # paths the input is an ambiguous STRING a parser must disambiguate,
        # and 'Task: 2500' is claimed as a local label BEFORE the qualified
        # pattern ever sees it. Here 'task' arrives in a field explicitly
        # named project_id — a category error, not a spelling — and this
        # path's contract is TOTAL loudness on malformed entries. Silently
        # rewriting the caller's own assertion is the silent degradation the
        # rest of this function exists to avoid.
        if project_id and is_task_vocabulary_qualifier(project_id):
            raise InputValidationError(
                f'declared referent {_safe_repr(entry)} has a task-vocabulary '
                f'word as its project_id ({_safe_repr(raw_project)}); that is '
                'task vocabulary, not a project key. Omit project_id for an '
                f'own-project referent, or name a real project. '
                f'{_DECLARED_REFERENT_HINT}'
            )
        if project_id and local_project and project_id == local_project:
            project_id = ''

        try:
            referent = Referent(
                kind=kind,
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


def _metadata_referents(metadata: dict | None, *, group_id: str) -> ReferentSet:
    """Bridge ambient ``metadata['task_id']`` into a referent set, or ``()``.

    A SELF-qualified value — one whose qualifier is the local project — is
    RECLASSIFIED to an own-project referent, exactly as ``scan_content`` and
    :func:`_declared_referents` do. All THREE paths now apply the same rule
    through the same :func:`_local_project` helper, and that is load-bearing:
    a spelling must denote the SAME node whichever source wins, so ``.source``
    selects the AUTHORITY, never the IDENTITY. Without it a downstream verifier
    or edge-surgery consumer keyed on ``node_name`` would hunt a foreign node
    ``'dark_factory:3127'`` that does not exist inside the dark_factory graph
    instead of the local ``'Task 3127'``, and conclude the fact is
    unattributable — the exact misattribution this PRD exists to prevent,
    reintroduced by the one path that skipped the rule.

    Reclassification changes WHICH node the metadata names, never whether it
    can conflict: this stays a bridge, not a claim about the prose, so the
    metadata path still never populates ``.conflicts``.

    An unusable value — or an unusable CONTAINER, ``None`` included — is
    DROPPED, never raised on. This is asymmetric with the
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
    # The CONTAINER is dropped on the same terms as the VALUE, so this
    # function has one degradation story rather than two. Not defensive
    # ceremony: every metadata-bearing signature in server/tools.py is
    # `metadata: dict | None = None` and tools.py guards it with
    # isinstance(metadata, dict) at four sites, so non-dict metadata reaches
    # that boundary in practice. `metadata.get` on None raises AttributeError,
    # which is wrong twice over here — it escapes δ's `except
    # InputValidationError` gate, and it contradicts this function's stated
    # drop-never-raise contract, losing a memory over ambient harness state
    # the writer may not even control.
    #
    # Tolerated HERE rather than at δ because the drop-not-raise asymmetry is
    # this function's documented policy; a caller-side isinstance check would
    # put a fifth copy of it at the boundary.
    if not isinstance(metadata, dict):
        return ()
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
        # parse_node_name already CANONICALIZES the qualifier it parsed
        # ('Dark-Factory:3127' -> project_id='dark_factory'), so only group_id
        # needs canonicalizing here — do not add a second canonicalize of the
        # parsed side.
        #
        # The `referent.project_id and local` guard is what keeps a foreign
        # referent foreign when group_id is path-shaped: _local_project returns
        # the '' sentinel there, which would otherwise compare equal to an
        # own-project referent's empty project_id and fire the branch
        # spuriously. Same guard, same reason, as on the declared path.
        local = _local_project(group_id)
        if referent.project_id and local and referent.project_id == local:
            # Rebuilt through the constructor rather than dataclasses.replace,
            # so the frozen vocabulary type's __post_init__ validation runs on
            # the reclassified value too.
            referent = Referent(kind=referent.kind, number=referent.number)
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
    metadata: dict | None,
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
            ``dict | None`` to match the shape δ actually holds — every
            metadata-bearing signature in server/tools.py is
            ``dict | None = None``. A non-dict is DROPPED, not raised on, like
            any other unusable ambient value.
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

    # An explicit declaration is the strongest source and is honoured verbatim,
    # INCLUDING the empty one. Two things a later reader will want to
    # "simplify", and must not:
    #
    # 1. `if declared:` collapses [] onto None and makes the PRD's tri-state a
    #    lie. The distinction is exactly what ι counts as "the agent CONSIDERED
    #    referents and none applied" (declared=[]) versus "the agent never
    #    looked" (declared=None, nothing derivable -> source='none').
    # 2. declared=[] produces NO conflict even when the scan DID find
    #    referents. That is deliberate, not an oversight: resolved decision 3
    #    rejects on conflict, never on ABSENCE, and a downstream verifier must
    #    in any case no-op on an empty referent set — there is nothing to test
    #    membership against — exactly as it already does for source='none'.
    if declared is not None:
        declared_referents = _declared_referents(declared, group_id=group_id)
        return ReferentResolution(
            source='declared',
            referents=declared_referents,
            # REPORTED, never enforced: .referents is deliberately left
            # untouched by a conflict. γ is mechanism and δ is policy — a
            # resolver that degraded to the scan here would produce a write the
            # caller never asked for.
            conflicts=_conflicting_referents(declared_referents, scan),
            ambiguous=ambiguous,
        )

    # Ambient metadata outranks the prose but is never reconciled against it:
    # `.conflicts` stays empty on this path, because metadata is not a CLAIM
    # about the content. An agent working on task 3668 legitimately writes
    # memories about Task 2500, so reconciling the two would reject a large
    # fraction of correct writes.
    bridged = _metadata_referents(metadata, group_id=group_id)
    if bridged:
        return ReferentResolution(source='metadata', referents=bridged, ambiguous=ambiguous)

    # scan.refs already excludes the ambiguous ones, so the derived set needs
    # no further filtering here. `.conflicts` stays empty on this path too:
    # conflicts are a property of what the CALLER declared, and the derived set
    # IS the scan — it cannot contradict itself.
    if scan.refs:
        return ReferentResolution(source='derived', referents=scan.refs, ambiguous=ambiguous)

    return ReferentResolution(source='none', ambiguous=ambiguous)
