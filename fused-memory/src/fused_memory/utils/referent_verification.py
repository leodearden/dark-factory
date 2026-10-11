"""The pure POLICY and RECORD layer for write-time referent verification
(task 3671, PRD leaf zeta of plans/memory-referent-fidelity-prd.md).

What lives here is everything the verification sub-pass DECIDES WITH and
RECORDS INTO, and nothing that it DOES: the closed check and axis
vocabularies, the log-volume cap, the :class:`ReferentFinding` record, the
:class:`ReferentStats` aggregate, and the pure predicates that turn one
endpoint plus one declared referent set into a candidate target, a reason it
has none, or a reason a determined one is implausible.

THE BEHAVIOUR DELIBERATELY STAYS BEHIND. ``_verify_episode_referents`` remains
a method on :class:`~fused_memory.services.memory_service.MemoryService`,
because it is not pure: it walks a live episode's edges, scans each fact, and
runs inside the per-group ``_identity_lock_for`` critical section. Splitting
the decisions out from the walk is what makes every rule below directly
unit-testable without constructing a service, and it is the reason this module
imports nothing from ``services/`` — a leaf, with no cycle back.

It sits alongside ``utils/referent_resolution.py``, the same shape one axis
over: that module decides WHICH referents a write is about, this one decides
whether an edge LANDED on one of them. Both are policy over the single label
vocabulary in ``utils/canonical_labels.py`` and neither compiles a regex.

The REPAIR layer (PRD leaf eta — ``REFERENT_REPAIR_OUTCOMES``,
``ReferentRepair``, ``ReferentRepairStats``) is NOT here. It is the WRITING
consumer of these records and stays with the service that performs the writes;
it imports :func:`_implausible_target_reason` back from this module rather than
re-deriving it.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass, field
from typing import Any

from fused_memory.utils.canonical_labels import Referent, parse_node_name
from fused_memory.utils.referent_resolution import local_referent

#: THE closed vocabulary of verification checks (task 3671, PRD leaf zeta).
#: The single normative site for the "which check fired" field the PRD's
#: §Contract requires on every repair record — a check name must be REGISTERED
#: here, never spelled as a bare string at a call site, or the two consumers
#: (leaf eta's repair, leaf iota's rate) key off vocabularies that drift.
#:
#: The C' post-LLM veto is deliberately NOT a third member. Post-write,
#: "extracted Task N was merged onto the Task M node" is observationally
#: IDENTICAL to "an edge about Task M is attached to a Task N node" — which
#: these two checks already detect. A separate veto mechanism would be two
#: sites that must agree byte-for-byte, i.e. exactly the INV-5 lockstep
#: duplication utils/canonical_labels.py exists to prevent. The PRD says so
#: outright: it "folds in", and is "not a distinct leaf".
REFERENT_CHECKS: tuple[str, ...] = ('set-membership', 'per-edge-pairing')

#: THE closed vocabulary of finding AXES — the counter buckets that are not
#: check names (task 3671, PRD leaf zeta). Registered here for the reason
#: :data:`REFERENT_CHECKS` and :data:`REFERENT_REPAIR_OUTCOMES` are: leaf iota
#: keys a rate off these names, so a bucket must be REGISTERED rather than
#: spelled as a bare string at the construction site and again at every reader.
#:
#: An AXIS answers an independent question ABOUT a finding; a CHECK names which
#: rule produced it. Axes increment ALONGSIDE whichever check fired, so
#: :meth:`MemoryService.referent_finding_counts` deliberately does not sum to
#: the finding total. The two:
#:
#: * ``'unresolvable'`` — no correct target could be determined, so the finding
#:   is recorded and LEFT ALONE. A numerator over the checks, not a third check.
#: * ``'corroborated'`` — the edge's own fact names the node it is already
#:   attached to, so :func:`_candidate_pool`'s corroboration veto emptied the
#:   pool. A NO-OBSERVABLE-DEFECT row: real and recorded, but not evidence of a
#:   scanner regression, and the dominant legitimate ``source='metadata'`` write
#:   shape produces it routinely. Registered as its own axis, and NOT folded out
#:   of the check bucket, so leaf iota can SUBTRACT it from the membership rate
#:   — which needs the denominator still there. Conflating it into the check
#:   would let ordinary ambient-task writes read as a scanner regression, the
#:   same argument ``'failed'`` makes one register down.
#:
#: DELIBERATELY NARROWER THAN "structurally unactionable". `_candidate_pool`'s
#: other two vetoes (an ambiguous endpoint, a ``source='metadata'`` fallback)
#: also produce unactionable rows, but neither is derivable from anything the
#: finding record carries — widening this axis to cover them would require a
#: stored flag, i.e. a second site that must agree with the guard in lockstep,
#: which is the drift this whole subsystem is built to avoid. Their volume is
#: bounded by `_REFERENT_FINDING_WARN_CAP` instead.
REFERENT_FINDING_AXES: tuple[str, ...] = ('unresolvable', 'corroborated')

#: Per-EPISODE ceiling on INDIVIDUALLY logged verification findings, after which
#: the operator log emits one aggregate line instead (task 3671, PRD leaf zeta).
#:
#: TEN, taken from the sibling :data:`_REFERENT_REPAIR_STREAK_THRESHOLD` in this
#: same subsystem rather than invented: both answer the same operator question
#: ("how many of these do I need to see before I have the picture?"), and two
#: unrelated magic numbers in one subsystem is a tuning surface nobody can hold
#: in their head.
#:
#: A LOG-VOLUME POLICY NUMBER, NOT AN ACCURACY THRESHOLD. Nothing downstream
#: keys off it — the counters and :attr:`ReferentStats.findings` are outside the
#: cap entirely — so retuning it cannot change a verdict, a rate, or a repair.
#: Every test computes its expectations FROM this constant for that reason, so a
#: retune can never red the suite.
#:
#: A LOWER value costs diagnostic detail: the individually-logged findings are
#: where an operator reads WHICH edges are wrong, and ten distinct payloads is
#: about the smallest sample that shows whether a storm is one repeated shape or
#: many. A HIGHER one buys nothing once the aggregate line exists — past a dozen
#: near-identical payloads the marginal line adds no diagnosis, and the storm's
#: SIZE, which is what the aggregate reports, is the thing that matters.
#:
#: IT BOUNDS THE WARN-LEVEL HALF ONLY, and the INFO half is uncapped ON PURPOSE.
#: A corroborated finding is demoted below WARNING *before* this budget is
#: applied (see the finding loop in
#: :meth:`MemoryService._verify_episode_referents`), so it never consumes it and
#: is never suppressed by it. What this constant protects is the ALERT channel —
#: the level an operator is expected to read every line of — from being buried
#: by the dominant legitimate ``source='metadata'`` shape. INFO is the OPT-IN
#: diagnostic channel: capping it would withhold exactly the per-finding detail
#: somebody switched it on to read, and it is switched on per module and
#: switched off again the same way. Uncapped is not unbounded WORK either — the
#: emission is guarded by ``logger.isEnabledFor``, so on a process that has not
#: enabled INFO for this module the demoted half costs neither a payload dict
#: nor a record. Nor can it lose the storm signal INV-4 requires: the
#: ``'corroborated'`` counter axis moves once per finding at every level, and
#: the aggregate line's per-check totals are computed over ALL of
#: :attr:`ReferentStats.findings` rather than over what was logged.
_REFERENT_FINDING_WARN_CAP: int = 10


@dataclass(frozen=True, kw_only=True)
class ReferentFinding:
    """One edge END that landed on a node the write was not about.

    The structured record INV-2 requires, carrying every field the PRD
    §Contract names — "edge uuid, old endpoint uuid, new endpoint uuid,
    referent set, which check fired" — plus what leaf eta needs to act:

    ==========================  ====================================
    PRD field                   Attribute
    ==========================  ====================================
    edge uuid                   :attr:`edge_uuid` (+ :attr:`which_end`)
    old endpoint uuid           :attr:`old_endpoint_uuid`
    new endpoint uuid           :attr:`new_endpoint_uuid`
    referent set                :attr:`referent_set`
    which check fired           :attr:`check`
    ==========================  ====================================

    FROZEN, and its collection field is a TUPLE, for the reason
    :class:`~fused_memory.utils.canonical_labels.Referent` and ``LabelScan``
    are: a finding is evidence for DESTRUCTIVE edge surgery, and ``frozen=True``
    blocks attribute rebinding only — a list field would leave
    ``finding.referent_set.append(...)`` open, letting a consumer quietly widen
    the set that justified the repair it is about to perform.

    Keyword-only because a record this wide, most of it strings, is exactly the
    shape where a positional argument silently lands in the wrong slot. (The
    count is deliberately not stated: it has grown twice already, and a docstring
    that has to be recounted on every field addition is a docstring that goes
    stale on the first one that forgets.)

    ``new_endpoint_uuid is None`` means "the node does not exist yet, or its
    name keys a duplicate-name group" — leaf eta resolves-or-mints via
    ``ensure_entity_node``, which handles both identically. It does NOT mean
    unrepairable; that is :attr:`resolvable`, which zeta defaults to ``False``
    so "recorded and left alone, never guessed at" is the structural default
    rather than something every construction site must remember.
    """

    #: The edge whose endpoint is wrong.
    edge_uuid: str
    #: Which end: ``'source'`` or ``'target'``. With :attr:`edge_uuid` this is
    #: the identity of the finding — at most one finding per (edge, end).
    which_end: str
    #: The project GRAPH this episode was written to — Graphiti's ``group_id``,
    #: and the scope in which every uuid on this record resolves.
    #:
    #: REQUIRED, and repeated per-finding rather than deduped onto the enclosing
    #: :class:`ReferentStats`, for exactly the reason
    #: :func:`_store_failure_diagnostics`' ``project_id`` Args note gives for the
    #: identical choice in this same module: an entry must be independently
    #: self-describing, so a consumer reading ONE finding off a log payload or a
    #: durable row never has to join back against the enclosing call to learn
    #: which project it came from. One process serves nine projects, and a
    #: finding read against the wrong one is not hypothetical — it produced a
    #: false conclusion in the 2026-08-31 audit. That is why this is required
    #: rather than defaulted: a finding with no project discriminator is the
    #: record that failure was made of, and it is now unconstructible.
    group_id: str
    #: The project SCOPE the write ran under.
    #:
    #: A KNOWN-ALIASED PAIR TODAY — said plainly here so nobody reads a
    #: guarantee into it. ``_verify_episode_referents`` is handed a ``group_id``
    #: and no :class:`~fused_memory.models.scope.Scope`, so the ONE production
    #: construction site fills this field from that same value, and no live
    #: payload has ever carried a pair that disagrees. There is nothing for it
    #: to disagree with yet:
    #: :attr:`~fused_memory.models.scope.Scope.graphiti_group_id` returns
    #: ``self.project_id``.
    #:
    #: Kept as its own field anyway, for a reason that is about the DURABLE row
    #: rather than the in-memory record: this payload is written verbatim to
    #: ``write_journal``'s ``referent_findings`` table and read back by a later
    #: process. :meth:`MemoryService._reconcile_episode_identity`'s own
    #: docstring already anticipates task 3335's cross-project split, after
    #: which the two need not agree — and a row that had carried only
    #: ``group_id`` would by then be permanently ambiguous about which of the
    #: two it meant, with no back-fill possible for rows already on disk. The
    #: cost is one string per finding on the ~0.2%-of-edges path.
    #:
    #: What the tests pin is accordingly that the RECORD can carry a distinct
    #: pair — that neither field is derived from or collapsed into the other —
    #: never that the system today produces one.
    project_id: str
    #: Which check fired; one of :data:`REFERENT_CHECKS`.
    check: str
    #: The node the edge is attached to today — as THIS EPISODE'S in-memory
    #: result reported it, PRE-NORMALIZATION. Recorded for the operator log,
    #: with the same caveat :attr:`old_endpoint_name` carries one field down and
    #: for the same reason: this pass is the EIGHTH sub-pass of
    #: ``_reconcile_episode_identity``, running after
    #: ``_normalize_task_node_names`` and ``_dedup_episode_nodes``, both of which
    #: call ``graphiti.merge_entities``. A node that LOST such a merge no longer
    #: exists in the graph, so an operator following this uuid out of a warning
    #: payload can query one that resolves to nothing — just as a spelling can
    #: have been normalized out from under the name beside it.
    #:
    #: Leaf eta is unaffected: ``reassign_edge`` keys on the EDGE uuid
    #: (:attr:`edge_uuid`) and never on this one, so a dead value is recorded
    #: but never acted on. The cost is operator-facing only, which is why this
    #: is DOCUMENTED rather than re-resolved live — re-resolving would add a
    #: backend round-trip per finding inside the identity-lock critical section,
    #: to refresh a field nothing reads.
    old_endpoint_uuid: str
    #: That node's name as this episode's result reported it. Recorded for the
    #: operator log; the VERDICT is keyed off :attr:`endpoint_referent`, since a
    #: spelling can have been normalized out from under this string.
    old_endpoint_name: str
    #: The parsed referent that name denotes — the thing actually compared.
    endpoint_referent: Referent
    #: The referent set the write DECLARED itself to be about, as canonical
    #: node names. The SET-MEMBERSHIP arm is what tests the endpoint against
    #: THIS; the pairing arm tests it against :attr:`cited`. Read the two
    #: together — recording only this one is what left the pairing arm's
    #: deciding input off the record (esc-3671-3).
    referent_set: tuple[str, ...]
    #: The referents THIS EDGE'S OWN FACT names, as canonical node names.
    #: The evidence that decided the PER-EDGE PAIRING arm, which fires
    #: precisely on ``cited_declared and endpoint_referent not in cited``, and
    #: — read together with :attr:`endpoint_referent` — the evidence behind
    #: :func:`_candidate_pool`'s corroboration veto: an endpoint that appears
    #: here is one the fact says the edge already belongs on.
    #:
    #: Recorded on BOTH arms, not just the one that reads it, so a consumer
    #: never has to know which arm carries which evidence. SORTED, for the
    #: reason :func:`_candidate_targets` sorts its survivors: the underlying
    #: value is a frozenset, whose iteration order is not stable across
    #: processes under hash randomization, and a finding must be stable across
    #: runs and diffable in eta's audit. Empty is a real answer rather than an
    #: absence — a fact naming no task is UNINFORMATIVE about where its edge
    #: belongs, which is exactly why the pairing guard refuses to fire on it.
    cited: tuple[str, ...] = ()
    #: The referent the edge SHOULD hang off, when exactly one candidate
    #: survives. ``None`` whenever :attr:`resolvable` is False.
    intended_referent: Referent | None = None
    #: The uuid of :attr:`intended_referent`'s node, when it resolves to
    #: exactly one live node. See the class docstring for what ``None`` means.
    new_endpoint_uuid: str | None = None
    #: Whether the :attr:`new_endpoint_uuid` lookup was DEGRADED — the backend
    #: would not answer — as opposed to answering "no such node" or "that name
    #: keys a duplicate group". All three yield ``new_endpoint_uuid=None``, and
    #: the last two are a DELIBERATE collapse (leaf eta's ``ensure_entity_node``
    #: resolves-or-mints and handles them identically); this flag is what stops
    #: the first from being folded in with them.
    #:
    #: THE CONCRETE STAKE. Task 3672's landed repair path reads
    #: :attr:`new_endpoint_uuid` at exactly ONE site —
    #: ``minted=finding.new_endpoint_uuid is None`` in
    #: :meth:`_repair_episode_referents` — so today a transient FalkorDB error
    #: books as ``minted=True`` telemetry: a node reported as newly created
    #: where in fact nobody could look. This flag is the discriminator that
    #: makes that a one-line fix in eta; zeta produces it and does not act on it
    #: (that consequence is filed as its own follow-up).
    #:
    #: Fail-closed like :attr:`resolvable`: a finding whose lookup never ran —
    #: one with no ``intended_referent`` — is not a finding whose lookup was
    #: degraded, so the default is ``False``.
    uuid_lookup_degraded: bool = False
    #: Whether a correct target was determined. Defaults False — fail-closed.
    resolvable: bool = False
    #: Why not, when :attr:`resolvable` is False. Empty on a resolvable
    #: finding.
    reason: str = ''

    @property
    def corroborated(self) -> bool:
        """Does this edge's OWN FACT name the node it is already attached to?

        The READ side of the corroboration veto whose single normative site is
        :func:`_candidate_pool`'s ``if endpoint in cited: return frozenset()``.
        True means the fact asserts the attachment is CORRECT, so the finding is
        a no-observable-defect row: real, recorded, and not actionable. The
        dominant legitimate write shape produces it — ``resolve_referents``
        derives ``source='metadata'`` from an agent's ambient ``task_id``, and
        "an agent working on task 3668 legitimately writes memories about Task
        2500" is that function's own example.

        A ``@property`` OVER THE RECORDED EVIDENCE, deliberately not a stored
        boolean set where the finding is built. A stored flag is a second site
        that must agree byte-for-byte with that guard, which is exactly the
        INV-5 lockstep duplication ``utils/canonical_labels.py`` exists to
        prevent — and exactly the failure mode that produced esc-3671-3's
        blocking bug, where the guard had drifted out of the position its own
        rationale assumed. Derived, the two cannot disagree. It is the same
        property-not-field discipline :class:`ReferentStats` documents for its
        counts and :attr:`Referent.node_name` follows for its rendering.

        Deliberately NOT a :meth:`to_dict` key: that payload's key set is
        contractually the dataclass FIELD names, and nothing is lost, because it
        already carries both inputs — an operator reads the corroboration off
        :attr:`cited` and :attr:`endpoint_referent` on the log line.

        Its two consumers are the ``'corroborated'`` counter bucket (so leaf
        iota can subtract these rows from the membership rate rather than read
        them as scanner defects) and the operator log's level (INFO rather than
        WARNING, for the alert-fatigue reason the pairing arm's
        ``cited_declared`` narrowing already refused to create).

        PRECONDITION — ``node_name`` IS INJECTIVE OVER THE REGISTERED KINDS.
        The veto it derives from compares :class:`Referent` OBJECTS (equality on
        the ``(kind, project_id, number)`` triple); this compares their
        RENDERINGS, because that is what :attr:`cited` carries, and the two
        agree only while distinct referents cannot render alike.
        :attr:`Referent.node_name` drops ``kind`` entirely for a FOREIGN
        referent (``f'{project_id}:{number}'``), so the rendering is injective
        purely because ``canonical_labels._KIND_LABELS`` holds exactly one kind
        today. Register a second, and a foreign citation of the new kind
        corroborates a foreign ``'task'`` endpoint of the same number — a genuine
        misattachment silently demoted to INFO and booked as corroborated, i.e.
        exactly the divergence deriving the property was chosen to make
        unrepresentable. That precondition is PINNED rather than merely noted:
        ``tests/test_referent_verification.py::
        TestCorroboratedIsDerivedFromTheRecordedEvidence::
        test_node_name_is_injective_over_the_registered_kinds`` reds the moment
        a kind is added, and carries the two admissible repairs.
        """
        return self.endpoint_referent.node_name in self.cited

    @property
    def target_cited(self) -> bool:
        """Does this edge's OWN FACT name the target this finding nominates?

        Read it as WHICH EVIDENCE ARM produced that target — exactly, and
        without a stored flag. :func:`_candidate_pool` returns
        ``cited & referents`` whenever that intersection is non-empty and the
        whole declared set otherwise, so a target drawn from the whole-set
        FALLBACK can never be in :attr:`cited` (if it were, the intersection
        would have been non-empty), and a target drawn from the INTERSECTION
        always is. True therefore means "the fact itself names this target";
        False means "this target came from the declared set, on a fact that
        says nothing about it".

        A ``@property`` OVER THE RECORDED EVIDENCE, for the reason its sibling
        :attr:`corroborated` states at length: a stored boolean set where the
        finding is built would be a second site that must agree with
        :func:`_candidate_pool` byte-for-byte, which is the INV-5 lockstep
        duplication that produced esc-3671-3's blocking bug. Derived, the two
        cannot disagree.

        ITS CONSUMER is the repair pass's TARGET-PLAUSIBILITY guard
        (:func:`_implausible_target_reason`, applied in
        :meth:`MemoryService._repair_edge_findings`), which acts ONLY on the
        fallback arm: a fact that NAMES the target is materially stronger
        evidence than the whole declared set, so the intersection arm is out of
        scope by ratified decision.

        Compares rendered ``node_name``s because that is what :attr:`cited`
        carries, under the same injectivity precondition :attr:`corroborated`
        rests on and which ``tests/test_referent_verification.py::
        TestCorroboratedIsDerivedFromTheRecordedEvidence::
        test_node_name_is_injective_over_the_registered_kinds`` already pins —
        that one test reds for BOTH properties, so no second precondition test
        exists.

        ``intended_referent is None`` (an unresolvable finding, which nominates
        no target at all) answers False: the fail-closed direction, since False
        is what makes the guard REFUSE.

        Deliberately NOT a :meth:`to_dict` key, like :attr:`corroborated`: that
        payload's key set is contractually the dataclass FIELD names, and both
        inputs are already on the record.
        """
        return (
            self.intended_referent is not None
            and self.intended_referent.node_name in self.cited
        )

    def __post_init__(self) -> None:
        if self.check not in REFERENT_CHECKS:
            raise ValueError(
                f'unregistered referent check {self.check!r}; registered checks '
                f'are {list(REFERENT_CHECKS)}. Add it to '
                'referent_verification.REFERENT_CHECKS rather than recording '
                'a finding no consumer can key off.'
            )

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-safe dict keyed exactly by this record's field names.

        The payload the operator warning carries, and the payload a durable
        ``referent_findings`` row stores verbatim. Referents render as their
        canonical ``node_name`` rather than as a dataclass repr, so the log
        line and the durable row read as graph names — the same thing an
        operator would type into a query.

        Because the key set IS the field names, :attr:`group_id` and
        :attr:`project_id` travel with every payload automatically: no consumer
        of a rendered finding can be handed one that does not say which project
        it came from.
        """
        return {
            'edge_uuid': self.edge_uuid,
            'which_end': self.which_end,
            'group_id': self.group_id,
            'project_id': self.project_id,
            'check': self.check,
            'old_endpoint_uuid': self.old_endpoint_uuid,
            'old_endpoint_name': self.old_endpoint_name,
            'endpoint_referent': self.endpoint_referent.node_name,
            'referent_set': list(self.referent_set),
            'cited': list(self.cited),
            'intended_referent': (
                self.intended_referent.node_name
                if self.intended_referent is not None
                else None
            ),
            'new_endpoint_uuid': self.new_endpoint_uuid,
            'uuid_lookup_degraded': self.uuid_lookup_degraded,
            'resolvable': self.resolvable,
            'reason': self.reason,
        }

def _endpoint_referent(endpoint_name: str, *, group_id: str) -> Referent | None:
    """The referent an edge ENDPOINT's node name denotes, or ``None``.

    ``None`` for an empty name (the endpoint this episode's result does not
    name), and for a name that is not a canonical task label at all
    ('MergeWorker') or merely MENTIONS one ('Task 42 orchestrator' —
    ``parse_node_name`` is anchored).

    A named function rather than an inline expression so the SOURCE-INVARIANT
    reclassification cannot be dropped by a later edit that only means to
    re-order the tuple this feeds: the bare ``parse_node_name`` it replaces read
    as complete, which is precisely how ζ came to be the one referent path that
    skipped the rule. See
    :func:`~fused_memory.utils.referent_resolution.local_referent`.
    """
    if not endpoint_name:
        return None
    referent = parse_node_name(endpoint_name)
    if referent is None:
        return None
    return local_referent(referent, group_id=group_id)


def _referent_sort_key(referent: Referent) -> tuple[str, str, str]:
    """The total order every rendered or returned referent SEQUENCE is sorted by.

    THE single site for that rule (INV-5), because its callers must agree and
    their correctness is one argument, not two: referents reach both as
    FROZENSETS, whose iteration order is not stable across processes under hash
    randomization, so an unsorted sequence would make :func:`_candidate_targets`'
    output and a finding's ``cited`` payload differ run to run for byte-identical
    inputs — undiffable in leaf eta's audit and unassertable without re-sorting
    at every call site that reads them.

    Keys on the IDENTITY TRIPLE ``(kind, project_id, number)`` — what a
    :class:`~fused_memory.utils.canonical_labels.Referent` IS, and the same
    triple its equality and hash are defined on — rather than on the rendered
    :attr:`~fused_memory.utils.canonical_labels.Referent.node_name`, which drops
    ``kind`` for a foreign referent and would therefore stop being a total order
    the moment a second kind is registered.

    ``number`` sorts as a STRING, matching how it is stored, so ``'10'`` precedes
    ``'9'``. Nothing keys off numeric rank; the property required here is
    STABILITY, not arithmetic order.
    """
    return (referent.kind, referent.project_id, referent.number)


def _candidate_pool(
    *,
    referents: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    ambiguous: frozenset[Referent],
    source: str,
) -> frozenset[Referent]:
    """The evidence rule, before either endpoint is subtracted.

    ``cited & referents``, falling back to the whole of ``referents`` ONLY when
    the fact says nothing about where this edge belongs. The edge's own fact is
    the sharpest evidence available about which node THIS edge belongs on, so a
    citation the declaration corroborates wins; the whole declared set is the
    fallback for when the fact cites nothing the declaration also names.
    INTERSECTING rather than unioning is what keeps a repair target from ever
    originating outside the referent set — an LLM-restated fact naming a task
    the write never declared itself to be about must not become a target.
    Fact-scoping is also what keeps mode (iii) repairable: with referents
    {3074, 3075} the whole-set fallback would see two candidates and abandon a
    repair the fact unambiguously determines.

    THE CORROBORATION GUARD (``endpoint in cited``) is what makes the fallback
    safe on the SET-MEMBERSHIP arm, and it is the membership-arm counterpart of
    the pairing arm's ``cited_declared`` guard — same principle, same
    fail-closed direction. A fact that NAMES the very node its edge landed on is
    the strongest possible evidence the attachment is CORRECT, so it must not be
    read as "the fact is silent, fall back to the declared set". Without the
    guard the dominant legitimate write shape becomes a repair instruction:
    ``resolve_referents`` derives ``source='metadata'`` from the write's ambient
    ``task_id``, and its own docstring names the mismatch as deliberately NOT a
    conflict — "An agent working on task 3668 legitimately writes memories about
    Task 2500". With referents {3668} and an edge whose fact reads "Task 2500
    was completed by the merge worker" hanging off the ``Task 2500`` node, the
    unguarded fallback yields the sole candidate ``Task 3668`` and hands leaf eta
    a ``resolvable=True`` instruction to repoint a CORRECT edge onto the task the
    agent merely happened to be working on — manufacturing the exact
    misattribution this PRD exists to prevent, and polluting the rate leaf iota
    samples with a finding that has no observable defect.

    THE CORROBORATION GUARD IS NOT SUFFICIENT ON ITS OWN, and two further vetoes
    sit beside it because it closes only the SUBSET of that shape where the fact
    happens to name the endpoint. An LLM-paraphrased fact that restates no task
    number at all ("the merge worker completed it") is the routine extraction
    outcome, and on such a fact ``cited`` is empty, so the corroboration guard
    cannot fire and the unguarded fallback re-manufactures exactly the
    ``resolvable=True``-onto-the-ambient-task instruction described above.

    VETO 1 — AN AMBIGUOUS ENDPOINT (PRD boundary row "Ambiguous scan | ref routed
    to ``.ambiguous``; treated as undeclared; recorded, not guessed"). γ routes a
    number claimed by BOTH a bare own-project mention and a foreign-qualified
    reference to ``LabelScan.ambiguous`` and EXCLUDES it from ``.referents``, on
    purpose — so the decoded referent set ALONE cannot distinguish an ambiguous
    endpoint from a genuine conflation: both are simply non-members. That is why
    the ambiguity set is a separate PARAMETER here rather than something this
    function could infer. The pool is suppressed for any endpoint in it: an
    ambiguous reference must be RECORDED and LEFT ALONE, never handed to eta as
    destructive repair surgery. Tested FIRST, ahead of even the corroboration
    guard, because it is the strongest "do not touch this" signal available and
    must hold whatever the fact happens to cite. WHERE the set comes from is ζ's
    concern, documented at :meth:`MemoryService._verify_episode_referents`; this
    function only needs it to be the PRODUCER's set, already through
    :func:`~fused_memory.utils.referent_resolution.local_referent`.

    VETO 2 — A ``source='metadata'`` FALLBACK. ``resolve_referents`` ranks ambient
    ``metadata['task_id']`` ABOVE the content-derived scan, and its own docstring
    names the resulting mismatch as deliberately NOT a conflict: "An agent working
    on task 3668 legitimately writes memories about Task 2500". A referent set
    bridged from the task an agent merely HAPPENS to be dispatched on is not a
    claim about which node any particular edge belongs on, so it must not become a
    repair target by default. The whole-declared-set fallback is therefore
    suppressed for ``source='metadata'``; ``'declared'`` (the caller stated its
    referents) and ``'derived'`` (they were scanned out of this very content) keep
    the fallback, because there the declared set genuinely IS evidence about the
    content. The ``cited & referents`` intersection survives on every source: a
    fact that names a declared referent is per-EDGE evidence regardless of how the
    declaration was sourced.

    THE CORROBORATION GUARD IS TESTED FIRST OF THE TWO CITATION RULES, AND THAT
    ORDER IS LOAD-BEARING — not stylistic.
    Behind the intersection short-circuit the guard is UNREACHABLE for every
    fact that cites the endpoint AND some declared referent, which is not an
    exotic shape but the same legitimate ambient-task write one sentence longer:
    "Task 2500 was completed as part of task 3668 by the merge worker" cites
    {2500, 3668}, so ``cited & referents`` is ``{3668}`` — non-empty — and an
    intersection-first order returns it, re-manufacturing the exact
    ``resolvable=True``-onto-Task-3668 instruction the paragraph above exists to
    prevent, on a fact that literally asserts the edge is about Task 2500. A
    citation of the endpoint therefore suppresses the pool UNCONDITIONALLY:
    corroboration is not merely a fallback the intersection can outrank, it is a
    veto. Nothing this test shadows is lost on the pairing arm, which is only
    reached when ``endpoint_referent not in cited`` — the guard can never fire
    there, so mode (iii) still resolves through the intersection below.

    Corroborated findings are still RECORDED — they are just recorded with an
    empty pool, which becomes ``resolvable=False`` plus a reason at the caller.
    That is this pass's stated postcondition: recorded and left alone, never
    guessed at.

    Extracted so the rule lives at ONE site that both :func:`_candidate_targets`
    and :func:`_unresolvable_reason` read. Without it the reason builder would
    have to RECOMPUTE the pool to explain itself — a second copy that must agree
    with the first byte-for-byte, which is exactly the INV-5 lockstep
    duplication this PRD exists to avoid.

    Args:
        referents: The set the write declared itself to be about.
        cited: The referents this edge's own fact mentions.
        endpoint: The referent the flagged endpoint currently parses as — read
            ONLY to ask whether the fact corroborates it, and whether it was
            ambiguous. The subtraction of the endpoint from the pool stays in
            :func:`_candidate_targets`, so this function remains "which referents
            is there evidence for", not "which targets survive".
        ambiguous: The referents the EPISODE CONTENT was ambiguous about — the
            PRODUCER's set, recovered by
            :meth:`MemoryService._verify_episode_referents`; see there for how.
            Already through
            :func:`~fused_memory.utils.referent_resolution.local_referent`, so it
            compares equal to *endpoint* on a self-qualified spelling.
        source: The ``ReferentSource`` :func:`_decode_referents` read off the
            queue payload — one of :data:`REFERENT_SOURCES`. Read ONLY to decide
            whether the whole-declared-set fallback is licensed (veto 2 above).

    THE WHOLE-SET FALLBACK IS NO LONGER THE LAST WORD. Licensing the fallback
    says the declared set may SUPPLY a target; it never said the target it
    supplies is plausible. The repair pass now tests the nominated target before
    acting on it — :func:`_implausible_target_reason`, applied by
    :meth:`MemoryService._repair_edge_findings` on the fallback arm only — and
    records a refusal instead of repairing when it fails.

    That guard lives THERE, not here, and this function is unchanged by it: no
    veto is added and none is reordered. This pass still detects and records
    (a finding drawn from the fallback is still ``resolvable``, which is what
    leaf iota's rate counts); the refusal belongs at the one site where a write
    can actually happen. Its rule is deliberately not restated here — read it at
    that function (SPOT).
    """
    if endpoint in ambiguous:
        # VETO 1. The episode content itself could not say which project's task
        # this number denotes, so there is nothing here to repair TOWARDS.
        # Ahead of the corroboration guard deliberately: an ambiguous endpoint is
        # unrepairable whatever the fact happens to cite.
        return frozenset()
    if endpoint in cited:
        # The fact names the node this edge end is already on. It is evidence
        # FOR the current attachment, never for repointing it elsewhere, so the
        # pool is suppressed rather than allowed to nominate a target the fact
        # does not support.
        #
        # FIRST, ahead of the intersection: a fact citing BOTH the endpoint and
        # a declared referent ("Task 2500 was completed as part of task 3668")
        # has a non-empty intersection, so testing the intersection first would
        # short-circuit past this guard entirely and return the ambient task as
        # a repair target. See the docstring — this is a veto, not a fallback.
        return frozenset()
    corroborated_citations = cited & referents
    if corroborated_citations:
        return corroborated_citations
    if source == 'metadata':
        # VETO 2. The fact cites no declared referent, and the declaration is
        # only the task this agent happened to be dispatched on — ambient
        # context, not an assertion about where this edge belongs. Falling back
        # to it here is what would manufacture the misattribution this PRD
        # exists to prevent on the dominant legitimate write shape.
        return frozenset()
    return referents


def _candidate_targets(
    *,
    referents: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    other_endpoint: Referent | None,
    ambiguous: frozenset[Referent],
    source: str,
) -> tuple[Referent, ...]:
    """Which referent could this misattached edge end correctly point at?

    A pure module-level function — no ``self``, no I/O — so the rule that
    decides whether leaf eta may perform destructive edge surgery is directly
    unit-testable in isolation from the walk that drives it.

    The rule, in order:

    1. ``pool = _candidate_pool(...)`` — the fact-cited intersection when it is
       non-empty; else the whole declared set, UNLESS one of three vetoes
       empties it: the fact cites the endpoint itself (corroboration), the
       endpoint referent was AMBIGUOUS in the episode content, or the
       declaration came from ambient ``source='metadata'`` and the fact cites no
       declared referent. See that function for why each is load-bearing on the
       dominant legitimate write shape.
    2. Subtract *endpoint*, the referent this finding is ABOUT. A "repair" onto
       the node the edge is already attached to is not a repair — and is not
       even a harmless no-op, because :meth:`_intended_endpoint_uuid` resolves
       the CANONICAL name: with a non-canonical endpoint spelling
       (``'task #3074'``) and a canonical ``'Task 3074'`` node both present it
       yields a DIFFERENT uuid, and eta would perform real edge surgery on an
       endpoint that was already correct.

       On the SET-MEMBERSHIP arm this subtraction is provably a NO-OP: the pool
       is always a subset of ``referents``, and membership fires precisely when
       the endpoint is NOT in ``referents``, so the endpoint can never be in the
       pool. It is therefore a STRUCTURAL GUARANTEE at the single site that
       decides targets rather than a behaviour change on the dominant path —
       which is the point: a future third check cannot silently reintroduce a
       self-targeting repair by forgetting to guard for it.

       The invariant is deliberately NOT additionally enforced by a raising
       ``ReferentFinding.__post_init__`` validator. This pass runs inside an
       already-committed write's identity-lock critical section, where raising
       is strictly worse than recording: the write has landed either way, and an
       exception would destroy the very evidence eta needs.
    3. Subtract *other_endpoint*. Not defensive ceremony: ``reassign_edge``
       (graphiti_client.py) explicitly refuses a move that would fold the edge
       into a self-loop, so a "target" equal to the edge's other end is not a
       repair eta could perform. This subtraction is precisely what turns the
       live Task 2519/2520 case — referents {2519}, endpoints (Task 2519,
       Task 2520), a fact unary about 2519 — into the zero-candidate row the PRD
       names as explicitly unrepairable.
    4. Return in a deterministic order.

    Exactly one survivor means the correct target is DETERMINED. Zero or more
    than one means it is not, and the caller records the finding with
    ``resolvable=False`` and a reason: RECORDED AND LEFT ALONE — never silently
    dropped, and never guessed at. That is why
    :attr:`ReferentFinding.resolvable` defaults to ``False`` rather than
    ``True``: the fail-closed direction is structural rather than a matter of
    every construction site remembering to say so.

    Args:
        referents: The set the write declared itself to be about.
        cited: The referents this edge's own fact mentions
            (``scan_content(...).refs``, which already excludes ambiguity).
        endpoint: The referent the flagged endpoint currently parses as.
            Non-optional: a finding is only ever built for an endpoint that
            PARSED, so the flagged referent is always known — encoded in the
            type rather than accepting a ``None`` no call site can produce.
        other_endpoint: The referent at the edge's OTHER end, or ``None`` when
            that end is not a task node at all.
        ambiguous: The referents the EPISODE CONTENT was ambiguous about.
            Forwarded verbatim to :func:`_candidate_pool` (veto 1).
        source: The ``ReferentSource`` the declaration came from. Forwarded
            verbatim to :func:`_candidate_pool` (veto 2).

    Returns:
        The surviving candidates, sorted by :func:`_referent_sort_key`.
        Sorted rather than kept in the caller's first-seen order because the
        inputs are FROZENSETS, whose iteration order is not stable across
        processes under hash randomization — and a finding must be stable across
        runs and diffable in eta's audit. Through the shared key rather than an
        inline ``lambda`` because the ``cited`` rendering in
        :meth:`MemoryService._verify_episode_referents` sorts for this exact
        reason and must not drift from it.
    """
    # `other_endpoint` may be None; None is simply not a member of a
    # frozenset[Referent], and typeshed types `frozenset.__sub__` as accepting
    # AbstractSet[_T_co | None], so no explicit `- {None}` branch is needed.
    pool = _candidate_pool(
        referents=referents, cited=cited, endpoint=endpoint,
        ambiguous=ambiguous, source=source,
    ) - {endpoint, other_endpoint}
    return tuple(sorted(pool, key=_referent_sort_key))

def _unresolvable_reason(
    candidates: tuple[Referent, ...],
    *,
    pool: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    other_endpoint: Referent | None,
    ambiguous: frozenset[Referent],
    source: str,
) -> str:
    """Why :func:`_candidate_targets` could not determine a correct target.

    Carried on the finding so "recorded and left alone" is legible as a REASON
    rather than as an absence — a reader must be able to tell an unrepairable
    row from a row nobody looked at, and to tell "the check had nothing to point
    at but the node it was already on" from "the only target would form a
    self-loop".

    Args:
        candidates: What :func:`_candidate_targets` returned.
        pool: The PRE-subtraction pool from :func:`_candidate_pool`. Membership
            is tested here rather than inferred from ``other_endpoint is None``
            precisely so the message stays HONEST when BOTH subtractions apply.
        cited: The referents this edge's own fact mentions — the same set the
            pool was computed from, so the corroboration branch reads the SAME
            input :func:`_candidate_pool` decided on rather than re-deriving it
            from the empty pool it produced (which is indistinguishable from an
            empty declared set).
        endpoint: The referent the flagged endpoint currently parses as.
        other_endpoint: The referent at the edge's other end, or ``None``.
        ambiguous: The referents the EPISODE CONTENT was ambiguous about — the
            same set :func:`_candidate_pool` vetoed on, read here for the SAME
            reason ``cited`` is: an emptied pool cannot say WHICH veto emptied
            it, and the three vetoes need three different explanations.
        source: The ``ReferentSource`` the declaration came from, likewise.
    """
    if len(candidates) > 1:
        return (
            'more than one candidate target survives '
            f'({[c.node_name for c in candidates]}) and the edge fact does not '
            'discriminate between them; recorded, not guessed at'
        )
    # Zero candidates. Either `_candidate_pool` returned nothing (the
    # corroboration branch below), or one of the two subtractions emptied it.
    #
    # Ordered FIRST among the zero-candidate branches because corroboration is
    # the most specific thing that can be said about a finding: when the fact
    # names the endpoint, "there was no target" is true but uninformative, and
    # "the fact says this edge belongs where it is" is the reason an operator
    # (and leaf eta) actually needs. It also cannot be inferred from `pool`,
    # which the guard deliberately empties.
    #
    # AMBIGUITY OUTRANKS CORROBORATION here, mirroring the veto order in
    # `_candidate_pool`: when the content could not say which project's task the
    # number denotes, that is the fact about this row an operator (and eta) most
    # needs, and it holds whatever the edge fact happens to cite.
    if endpoint in ambiguous:
        return (
            f'the endpoint referent {endpoint.node_name!r} was AMBIGUOUS in the '
            'episode content — claimed by both a bare own-project mention and a '
            'foreign-qualified reference — so it is treated as undeclared rather '
            'than as a conflation; recorded, not guessed at'
        )
    if endpoint in cited:
        return (
            f"the edge's own fact cites {endpoint.node_name!r}, the endpoint it "
            'landed on, which corroborates the current attachment; the declared '
            'referent set is not evidence for repointing it, so this is '
            'recorded, not repaired'
        )
    if source == 'metadata' and not pool:
        # Veto 2. Reached only when neither veto above fired and the fact cited
        # no declared referent, so the ONLY thing that could have supplied a
        # target was the whole-declared-set fallback the source suppresses.
        return (
            "the write's referent set was bridged from ambient "
            "metadata['task_id'] rather than declared or derived from the "
            'content, and this edge\'s fact cites no declared referent; the task '
            'an agent happens to be dispatched on is not evidence about which '
            'node this edge belongs on, so this is recorded, not repaired'
        )
    if endpoint in pool:
        return (
            f'the only candidate target {endpoint.node_name!r} is the node this '
            'edge end is already attached to, so there is nothing to repoint '
            'to; recorded, not repaired'
        )
    if other_endpoint is not None:
        # The live Task 2519/2520 row.
        return (
            f'the only candidate target {other_endpoint.node_name!r} is this '
            "edge's other endpoint, so repointing would form the self-loop "
            'reassign_edge refuses; there is no correct target'
        )
    # Defensive: unreachable while the caller no-ops on an empty referent set.
    # Kept so a future relaxation records a reason rather than an empty string
    # that reads as "resolvable".
    return 'no candidate target could be determined from the declared referents'


def _implausible_target_reason(
    target: Referent,
    *,
    known_projects: Container[str],
    target_node_exists: bool,
) -> str:
    """Why a nominated repair target must NOT be acted on. ``''`` if it may be.

    WHY THIS RULE EXISTS. :func:`_candidate_pool`'s whole-set fallback nominates
    a target from the DECLARED set with no test of that target's plausibility:
    whatever survives the pool's three vetoes becomes ``resolvable=True``, and
    the repair pass then resolves-or-MINTS a node of that name and repoints a
    real edge onto it. A junk derived referent therefore becomes a PHANTOM NODE
    carrying a real edge. Executed live: ``referents={redis:6379}``, ``cited={}``
    (a paraphrased fact naming no task number — the pool's own docstring calls
    this "the routine extraction outcome"), endpoint ``Task 1251`` yields
    ``candidates=('redis:6379',)`` and a mint. Nothing tested the TARGET; the
    corroboration veto tests the ENDPOINT, and every spurious referent observed
    live was blocked only because the paraphrase happened to cite the endpoint.

    APPLIED ONLY ON THE FALLBACK ARM (``not finding.target_cited``). A fact that
    NAMES the target is materially stronger evidence than the whole declared
    set, and the intersection arm is out of scope by ratified decision — its
    executed repairs were all correct and all fired through fact-cited evidence.

    ``''`` MEANS PLAUSIBLE, so the fail-closed direction is the non-empty
    string: anything this rule cannot positively vouch for is refused.

    DELIBERATELY NOT A FOURTH VETO IN :func:`_candidate_pool`. zeta detects and
    records; eta refuses. Folding it into the pool would EMPTY the pool, turning
    a recorded resolvable finding into an unresolvable one and erasing the
    evidence that a junk referent was declared at all — and it would reorder
    vetoes that must not be reordered. Keeping the refusal at the write point
    also keeps it at the one place a mint can actually happen, so no future
    caller of :func:`_candidate_targets` can route around it.

    Args:
        target: The referent :func:`_candidate_targets` nominated.
        known_projects: The registry of real project keys —
            ``MemoryService._known_projects``, fed by ``build_known_projects_map``
            via ``set_known_projects``. Membership is the ONLY thing that makes a
            foreign qualifier real.
        target_node_exists: Whether a node of this exact name already exists in
            the group. An existing node is its own evidence: repointing onto it
            mints nothing, so there is no phantom to prevent.

    Returns:
        ``''`` when the target may be repaired towards, else the operator-
        readable REASON it was refused, carried verbatim onto the record the
        way :func:`_unresolvable_reason`'s output is.
    """
    if target.project_id and target.project_id in known_projects:
        # (i) A real cross-project reference. The qualifier names a project the
        # registry knows, so the node is a legitimate one to mint or resolve.
        return ''
    if target.kind == 'task' and not target.project_id:
        # (ii) A bare own-project `Task N` referent — the dominant live shape.
        # The `kind` test is redundant TODAY, since `_KIND_LABELS` holds exactly
        # one kind, and is written anyway so that registering a second kind
        # fails CLOSED here rather than silently widening what counts as "a bare
        # own-project Task-N referent".
        return ''
    if target_node_exists:
        # (iii) A node of this exact name is already in the group, so repointing
        # onto it mints nothing — and a phantom node is the whole thing this
        # rule exists to prevent. Its own existence outranks an unrecognized
        # qualifier: something already put it there.
        return ''
    return (
        f'the nominated repair target {target.node_name!r} is not plausible: '
        f'its qualifier {target.project_id!r} names no project this instance '
        'knows, and no node of that name exists in this group, so repairing '
        'towards it would MINT one out of a referent nothing corroborates; '
        'recorded, not repaired'
    )


@dataclass
class ReferentStats:
    """What one ``_verify_episode_referents`` run looked at, and what it found.

    The in-process half of INV-2's structured record: leaf eta reads
    :attr:`findings` off the return value, inside the same identity-lock
    critical section, and acts on it. (The process-lifetime half leaf iota
    reads is ``MemoryService.referent_finding_counts``.)

    The three summary counts are ``@property`` comprehensions over
    :attr:`findings` rather than fields precisely so they CANNOT drift from the
    list they summarize — the same property-not-field discipline
    :attr:`Referent.node_name` follows. A stored count is a second site that
    must be incremented in lockstep with every append.

    :attr:`endpoints_unresolved` exists so this pass's one blind spot — an edge
    endpoint uuid that this episode's ``result.nodes`` does not name, so its
    name is unknown and it cannot be checked at all — is COUNTED rather than
    silently skipped. A skipped endpoint is a check that did not run, and a
    verification pass that cannot say how often it declined to look is not a
    verification pass.

    :attr:`endpoints_unregistered_qualifier` is the MACHINE half of that same
    principle for the OTHER deliberate skip: how many endpoints this pass
    declined to check for set membership because their project qualifier names
    no project the registry knows. :attr:`endpoints_checked` counts a skipped
    endpoint too, so without this field a skip would be indistinguishable from
    an endpoint that AGREED — a silent drop wearing a clean pass's clothes.
    """

    edges_scanned: int = 0
    endpoints_checked: int = 0
    endpoints_unresolved: int = 0
    endpoints_unregistered_qualifier: int = 0
    findings: list[ReferentFinding] = field(default_factory=list)

    @property
    def set_membership_findings(self) -> int:
        """Findings from the SET MEMBERSHIP check."""
        return sum(1 for f in self.findings if f.check == 'set-membership')

    @property
    def pairing_findings(self) -> int:
        """Findings from the PER-EDGE PAIRING check."""
        return sum(1 for f in self.findings if f.check == 'per-edge-pairing')

    @property
    def unresolvable_findings(self) -> int:
        """Findings recorded with no determinable correct target — left alone."""
        return sum(1 for f in self.findings if not f.resolvable)
