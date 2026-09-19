"""Shadow-mode L2 adjudication rulings — MEASUREMENT ONLY, no authority granted.

``docs/escalation-standing-policy.md`` proposes classes of L2 escalation that an
adjudicating session could one day rule without waiting for the human. NONE of
them is adopted. This module is the measurement half of that proposal: a watcher
records what it WOULD have ruled, takes no action, and a weekly count compares
the recorded proposal against what the human actually did. Nothing here decides
anything, closes anything, or widens any caller's authority — the shape of that
authority is owned by ``escalation/src/escalation/authority.py`` and is untouched
by this file.

**Why the payload rides inside ``triage_note``.** The tool that writes it,
``escalation/src/escalation/server.py::stamp_triage``, has exactly one free
parameter — ``triage_note: str``. There is no structured field for a shadow
ruling and this task deliberately adds none: a new ``Escalation`` field plus a
new tool argument would be a schema and API change in the very subsystem whose
authority the task must not touch.

**Why the payload is a LINE rather than the whole note.** ``triage_note``
already carries a freshness contract — a named world-facing predicate plus the
probe used to check it (``skills/escalation-watcher/SKILL.md``, "Reading a
triage-ack annotation"). A shadow stamp must COMPOSE with that contract, not
replace it, so the payload is one ``x_shadow_ruling: {...}`` line among the
note's other lines. Reading it is ``line.startswith(marker)`` plus
``json.loads`` — real structured data behind a fixed token, never an ad-hoc
grammar over prose.

**Promotion path.** If a class is ever adopted, ``x_shadow_ruling`` becomes a
first-class ``Escalation`` field and this codec is DELETED, not extended. The
string-carried form is the price of changing nothing while the measurement runs;
it is not a design to build on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES, L2_AUTO_CLOSE_DENY_ROLES
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan

logger = logging.getLogger(__name__)

#: The fixed token that opens a shadow-ruling line inside a ``triage_note``.
#: A line qualifies only when it BEGINS with this token, so prose that merely
#: mentions it — this module's own guidance, or the skill's — is never mistaken
#: for a stamp.
SHADOW_RULING_MARKER = 'x_shadow_ruling:'

#: The first-tranche candidate classes, SHADOWED AND NOT ADOPTED. Each names a
#: recurring L2 shape the standing-policy skeleton proposes could be ruled by
#: the adjudicating session. Enumerated with their prose gloss in
#: ``docs/escalation-standing-policy.md``; that document and this frozenset are
#: held equal by ``tests/scripts/test_shadow_ruling_doc_contract.py``.
FIRST_TRANCHE_CLASSES: frozenset[str] = frozenset({
    'risk_identified_branch_behind_main',
    'risk_identified_recovery_veto_streak',
    'design_concern_semantic_collision',
    'infra_issue_transient_self_cleared',
})

#: The reversible actions the ratified skeleton permits. A SUPERSET of the C1
#: ``resolution_action`` vocabulary: ``close_only`` and ``resume`` are C1 values
#: a resolved record records, while ``add_dependency``, ``update_task_amendment``
#: and ``file_task`` are task-side operations that leave ``resolution_action``
#: unset — which is why the weekly count needs a third ``not_comparable``
#: outcome rather than folding them into agreement or divergence.
REVERSIBLE_ACTIONS: frozenset[str] = frozenset({
    'close_only',
    'resume',
    'add_dependency',
    'update_task_amendment',
    'file_task',
})

#: The seven classes of decision that stay with the human FOREVER, whatever any
#: shadow measurement shows. Only the first two have a record-level signal; see
#: :func:`mechanically_gated`. The authority for this list is
#: ``docs/escalation-standing-policy.md``, which glosses each one.
HUMAN_FOREVER_GATES: frozenset[str] = frozenset({
    'milestone_gate',
    'deterministic_runner_filing',
    'model_admission',
    'physical_operator_action',
    'irreversible_deletion',
    'spend_or_eval_launch',
    'post_breaker_resume_scheduler',
})

#: The two gate slugs :func:`mechanically_gated` can return. A PROPER subset of
#: :data:`HUMAN_FOREVER_GATES` — the remaining five gates are semantic and have
#: no record-level signal at all.
DETECTABLE_GATES: frozenset[str] = frozenset({'milestone_gate', 'deterministic_runner_filing'})

#: The one auto-close-denied category this detector deliberately does NOT gate
#: on. ``escalation.authority``'s denylist and this detector answer DIFFERENT
#: QUESTIONS, and ``design_concern`` is the single cell where the two answers
#: diverge:
#:
#: * authority.py asks *may the AUTO-WATCHER IDENTITY auto-close this at L2?* —
#:   a table its own docstring scopes to the identified callers in
#:   ``ROLE_LEVEL_ALLOWLIST``, whose sole member is the auto-watcher. Nothing
#:   here changes that answer: the auto-watcher still may not auto-close a
#:   ``design_concern``.
#: * this detector asks *does the ratified skeleton keep this record with the
#:   human FOREVER, whatever the measurement shows?* The skeleton's gate list is
#:   "every ``milestone_gate`` and ``orchestrator-deterministic`` record";
#:   ``design_concern`` is absent from it, absent from
#:   :data:`HUMAN_FOREVER_GATES`, and named by the same ratified text as one of
#:   the four first-tranche classes to shadow
#:   (``design_concern_semantic_collision``).
#:
#: Gating it conflated the two questions and made that class unmeasurable —
#: every stamp landing in the report's ``gated_stamps`` bucket, so the class
#: could never reach its own "95% or better over at least 10 items" threshold
#: (esc-5374-1).
_UNGATED_DENIED_CATEGORY: str = 'design_concern'

#: The categories and roles the detector matches against — DERIVED from the
#: imported ``escalation.authority`` tables, never copies of their literals, so
#: a member added to authority.py propagates here for free and replacing either
#: binding with a literal fails the SPOT guard. The lone subtraction above is
#: the whole of the local policy, stated once rather than by re-listing members.
GATED_CATEGORIES: frozenset[str] = L2_AUTO_CLOSE_DENY_CATEGORIES - {_UNGATED_DENIED_CATEGORY}
GATED_ROLES: frozenset[str] = L2_AUTO_CLOSE_DENY_ROLES

#: Of the gated categories, the ones that name a MILESTONE. The rest report the
#: runner slug instead of being mislabelled as milestones: per
#: ``escalation/src/escalation/authority.py``, every member of
#: :data:`GATED_CATEGORIES` is filed by ``orchestrator.deterministic_runner``
#: under ``agent_role='orchestrator-deterministic'``, and
#: ``curator_adjudication_missing`` — the re-ask raised when a
#: ``human_curator_gate`` task resumes with no adjudication stamp — is not a
#: milestone in any sense.
#:
#: The MILESTONE side is listed rather than the runner side so the derivation
#: property survives: a category added to authority.py still gates for free and
#: falls to ``deterministic_runner_filing``, which is what a new runner-filed
#: category would in fact be.
_MILESTONE_CATEGORIES: frozenset[str] = frozenset({
    'milestone_gate',
    'milestone_check_failed',
})

#: The reversible actions that are ALSO C1 ``resolution_action`` values, so an
#: observed outcome CAN be checked against the proposal. The rest of
#: :data:`REVERSIBLE_ACTIONS` is task-side and leaves ``resolution_action``
#: unset. Membership here makes a proposal comparable-in-principle only: the
#: report also needs the observed side to be present, and counts both absences
#: in the same ``not_comparable`` bucket.
#:
#: Must equal ``REVERSIBLE_ACTIONS & set(escalation.server.RESOLVE_ACTIONS)``,
#: and is pinned in lockstep by a cross-module TEST import rather than derived
#: by a production one — importing the MCP server here would invert the layer
#: direction and pull fastmcp into every reader of the archive. This mirrors the
#: convention ``escalation/src/escalation/authority.py`` already uses for the
#: watcher identity string and ``action_effects.py`` for its target statuses.
COMPARABLE_ACTIONS: frozenset[str] = frozenset({'close_only', 'resume'})

#: Each wire key of the JSON payload against the :class:`ShadowRuling` attribute
#: it carries — the ONE place that vocabulary is written. The encoder renders
#: from it, the decoder validates and reads through it, and
#: ``tests/scripts/test_shadow_ruling_doc_contract.py`` checks the skill's
#: literals against it. Three independent copies is how a key renamed on the
#: encoding side alone would make every freshly-rendered stamp unreadable while
#: a guard that had re-typed the old spelling stayed green (SPOT).
#:
#: ``class`` rather than ``ruling_class`` because that is what a reader of the
#: note sees; the Python attribute cannot be spelled ``class``. The wire
#: spelling is FROZEN by the archive — stamps already written carry it — and is
#: pinned as literals by ``escalation/tests/test_shadow_ruling.py``, the one
#: deliberate copy.
PAYLOAD_WIRE_KEYS: Mapping[str, str] = MappingProxyType({
    'class': 'ruling_class',
    'proposed_action': 'proposed_action',
    'evidence': 'evidence',
    'confidence': 'confidence',
})

#: The key set a payload must carry EXACTLY, read off the vocabulary above.
_PAYLOAD_KEYS: frozenset[str] = frozenset(PAYLOAD_WIRE_KEYS)


@dataclass(frozen=True)
class ShadowRuling:
    """What an adjudicating session WOULD have ruled, had the class been adopted.

    Validated on construction, so a rejected payload renders nothing and
    persists nothing. The four fields are exactly the four the standing policy
    requires of a proposal: which class it falls in, which reversible action it
    proposes, the evidence quoted verbatim, and how sure the adjudicator was.
    """

    ruling_class: str
    proposed_action: str
    evidence: str
    confidence: float

    def __post_init__(self) -> None:
        # TYPE BEFORE MEMBERSHIP, for the same reason `evidence` and
        # `confidence` below are type-checked first: the payload is JSON an LLM
        # session hand-wrote into a `triage_note`, so any of these four values
        # can arrive as a list, dict or number. `x not in frozenset` raises
        # TypeError on an unhashable value rather than returning False, and a
        # TypeError out of this constructor escapes `parse_shadow_ruling`'s
        # rejection path and kills the whole sweep. Every validation failure in
        # this class is a ValueError — uniformly, so one catch covers them all.
        for field, value in (('ruling_class', self.ruling_class),
                             ('proposed_action', self.proposed_action)):
            if not isinstance(value, str):
                raise ValueError(
                    f'shadow {field} must be a string, got {type(value).__name__} {value!r}'
                )
        if self.ruling_class not in FIRST_TRANCHE_CLASSES:
            raise ValueError(
                f'unknown shadow ruling_class {self.ruling_class!r}; '
                f'expected one of {sorted(FIRST_TRANCHE_CLASSES)}'
            )
        if self.proposed_action not in REVERSIBLE_ACTIONS:
            raise ValueError(
                f'unknown shadow proposed_action {self.proposed_action!r}; '
                f'expected one of {sorted(REVERSIBLE_ACTIONS)}'
            )
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError(
                'shadow ruling evidence must be a non-empty string quoting the '
                'deciding evidence verbatim'
            )
        if not isinstance(self.confidence, int | float) or isinstance(self.confidence, bool):
            raise ValueError(f'shadow ruling confidence must be a number, got {self.confidence!r}')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f'shadow ruling confidence {self.confidence!r} outside [0.0, 1.0]'
            )

    def to_note_line(self) -> str:
        """Render the marker line to append to an existing ``triage_note``.

        Always ONE line: JSON escaping folds any newline in *evidence* into
        ``\\n``, so the caller can append this to a note without breaking the
        note's other lines.
        """
        payload = {
            wire: getattr(self, attribute)
            for wire, attribute in PAYLOAD_WIRE_KEYS.items()
        }
        return f'{SHADOW_RULING_MARKER} {json.dumps(payload, sort_keys=True)}'


@dataclass(frozen=True)
class NoteStamp:
    """What a ``triage_note`` carries at the marker.

    THREE states rather than two, because the weekly count has to tell a record
    that was never stamped — the overwhelming majority of the archive, and not a
    sample at all — from one that WAS stamped and whose stamp could not be read.
    The second is a lost sample, and the report has a bucket for it.
    """

    ruling: ShadowRuling | None
    rejected: bool

    def __post_init__(self) -> None:
        if self.rejected and self.ruling is not None:
            raise ValueError('a rejected stamp cannot also carry a ruling')


def _decode_marker_line(line: str) -> ShadowRuling | None:
    """Decode ONE marker line, or ``None`` with a WARNING naming what was wrong."""
    raw = line[len(SHADOW_RULING_MARKER):]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning('unparsable %s payload: %s', SHADOW_RULING_MARKER, exc)
        return None
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        logger.warning(
            '%s payload keys are %r; expected exactly %r',
            SHADOW_RULING_MARKER,
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__,
            sorted(_PAYLOAD_KEYS),
        )
        return None
    try:
        return ShadowRuling(**{
            attribute: payload[wire] for wire, attribute in PAYLOAD_WIRE_KEYS.items()
        })
    except (TypeError, ValueError) as exc:
        # ValueError is what ShadowRuling raises for EVERY rejection, and the
        # type checks in its __post_init__ are what keep that true. TypeError is
        # the redundant backstop on the "It NEVER raises" contract: this decode
        # runs once per record over the whole live queue, so a single escaping
        # exception costs the entire measurement, and that price is too high to
        # pay for a validator invariant enforced only in one place.
        logger.warning('rejected %s payload: %s', SHADOW_RULING_MARKER, exc)
        return None


def scan_triage_note(triage_note: str) -> NoteStamp:
    """Read the shadow stamp *triage_note* carries: usable, unusable, or absent.

    THE LAST MARKER LINE IS THE STAMP. ``queue.py::stamp_triage`` REPLACES the
    note wholesale, so ``skills/escalation-watcher/SKILL.md`` has the session
    re-send the previous note with the new marker appended on its own line; a
    re-stamped record therefore carries several marker lines, of which only the
    newest was ruled. Reading the first scored a SUPERSEDED proposal against the
    observed outcome — a wrong sample, which is strictly worse than a lost one,
    because every other way this module loses a sample lands in a bucket a
    reader can see and a wrong sample lands in the rate itself.

    An earlier line is superseded, not a fallback: when the newest marker cannot
    be decoded the record's current proposal is simply unknown, so the answer is
    ``rejected`` rather than the stale ruling above it. An unusable line does
    not suppress a LATER valid one, which is the same rule read forwards.

    It NEVER raises, INCLUDING on a note that is not a string at all. The count
    sweeps every escalation in the queue and archive, so an exception would turn
    one malformed record into a failed measurement.
    """
    # A note arrives UNVALIDATED: ``models.py::Escalation.from_dict`` passes
    # every JSON value straight into the dataclass, so a record carrying
    # ``"triage_note": null`` reaches here as None despite the declared type,
    # and ``.splitlines()`` raises an AttributeError absent from
    # ``_SWEEP_PARSE_ERRORS`` and raised after the read returned — one
    # hand-written record for the whole measurement. Type before use, as in
    # ``ShadowRuling.__post_init__``.
    if not isinstance(triage_note, str):
        logger.warning(
            'triage_note is %s, not a string; no stamp readable',
            type(triage_note).__name__,
        )
        return NoteStamp(ruling=None, rejected=False)
    markers = [
        line for line in triage_note.splitlines()
        if line.startswith(SHADOW_RULING_MARKER)
    ]
    if not markers:
        return NoteStamp(ruling=None, rejected=False)
    ruling = _decode_marker_line(markers[-1])
    return NoteStamp(ruling=ruling, rejected=ruling is None)


def parse_shadow_ruling(triage_note: str) -> ShadowRuling | None:
    """The :class:`ShadowRuling` *triage_note* carries, or ``None``.

    The ruling half of :func:`scan_triage_note`, for every caller that does not
    need to tell an unreadable stamp from an absent one. ``None`` means "no
    usable shadow ruling here" and never raises, for the reasons stated there.
    """
    return scan_triage_note(triage_note).ruling


def mechanically_gated(record: Escalation) -> str | None:
    """Return the human-forever gate slug *record* trips, or ``None``.

    ``None`` means **no MECHANICAL gate was detected**, NEVER "not gated". Five
    of the seven gates in :data:`HUMAN_FOREVER_GATES` — model admission,
    physical operator actions, irreversible deletions, spend/eval launches and a
    post-breaker ``resume_scheduler`` — are semantic judgements with no signal
    on the record at all. ``docs/escalation-standing-policy.md`` remains the
    authority for the full list; this function covers only its detectable
    subset, and a caller that treats a ``None`` here as clearance is reading it
    wrong.

    Category and role are checked independently, mirroring the defence in depth
    ``escalation/src/escalation/authority.py`` already relies on: neither benign
    half can mask the other.

    The returned slug names the gate actually tripped, which is NOT the same for
    every gated category. Only :data:`_MILESTONE_CATEGORIES` reports
    ``'milestone_gate'``; ``curator_adjudication_missing`` reports
    ``'deterministic_runner_filing'``, the gate it genuinely trips, rather than
    being labelled a milestone it has nothing to do with. The slug is the only
    machine-readable reason a caller receives, so it has to survive being read
    on its own.

    ``design_concern`` IS NOT GATED HERE, and that is the deliberate answer to
    esc-5374-1 rather than an omission. :data:`_UNGATED_DENIED_CATEGORY` owns
    that argument — the two questions this detector and authority.py's denylist
    separately answer — and ``docs/escalation-standing-policy.md`` §"Adoption
    preconditions" owns which arm an adoption would touch. Neither is restated
    here. This file changes no authority.
    """
    if record.category in GATED_CATEGORIES:
        return (
            'milestone_gate' if record.category in _MILESTONE_CATEGORIES
            else 'deterministic_runner_filing'
        )
    if record.agent_role in GATED_ROLES:
        return 'deterministic_runner_filing'
    return None


@dataclass(frozen=True)
class ClassAgreement:
    """How one shadowed class fared over the report window.

    ``agreed`` and ``diverged`` are the COMPARABLE outcomes: the stamped
    proposal was a C1 action AND the record recorded one, so the two could be
    checked against each other. ``not_comparable`` counts the records where one
    of those sides is missing — a task-side proposal, which leaves no C1 trace,
    or an observed ``resolution_action`` of ``None``, which leaves nothing to
    compare against. Either way it is kept as its own number rather than folded
    into either side, so the denominator the adoption threshold is read off is
    one a reader can see, and missing data never reads as disagreement.

    ``non_human_resolver`` counts the in-window stamps whose record was resolved
    by no human at all — a cascade, a reaper sweep, the steward — so the
    proposal was never held against an adjudication. It is PER CLASS because
    ``AgreementReport.resolver_tiers`` says only what took the sample: a reader
    of ``cascade=3`` could not tell which shadowed class paid for it, and a
    class every one of whose records was taken that way had no row in the table
    at all — indistinguishable from a class nobody ever stamped, which is the
    one thing every other bucket here exists to prevent.
    """

    ruling_class: str
    agreed: int
    diverged: int
    not_comparable: int
    non_human_resolver: int

    @property
    def comparable(self) -> int:
        """The denominator of :attr:`agreement_rate`."""
        return self.agreed + self.diverged

    @property
    def total(self) -> int:
        """Every in-window record counted for this class, rate-bearing or not."""
        return (
            self.agreed + self.diverged + self.not_comparable + self.non_human_resolver
        )

    @property
    def agreement_rate(self) -> float | None:
        """Agreement over the comparable subset, or ``None`` when there is none.

        ``None`` rather than 0.0 or 1.0: a class whose proposals are all
        task-side is NOT YET MEASURABLE, and reporting either extreme would make
        it look decided.
        """
        return self.agreed / self.comparable if self.comparable else None


@dataclass(frozen=True)
class AgreementReport:
    """The weekly count over one window. Every field is a decided number.

    The four non-rate buckets are findings, not noise: each counts a stamp that
    must not contribute to any class's rate, and each is reported so a reader
    can tell a small sample from a thrown-away one. ``rejected_stamps`` is the
    same discipline applied to the codec itself — a stamp that was filed and
    could not be read is a sample this measurement lost, and a pasted
    ``comparable=8`` must not be able to hide two of them.

    THREE OF THE FOUR ARE IN-WINDOW AND THE FOURTH CANNOT BE, and the field
    names say which: ``gated_stamps``, ``self_resolved`` and ``rejected_stamps``
    count records RESOLVED inside ``since..until``, exactly like
    ``agreed``/``diverged``, so they are comparable with the denominator printed
    beside them. ``unresolved_lifetime`` counts stamps still pending — a record
    with no resolution instant to window on at all — so it is a standing backlog
    as of the sweep, not a number from this window, and is named for that.
    Windowing it on ``triaged_at`` instead was rejected: it would put a second
    time axis under one window header, and the operator-facing contract
    (``--since``/``--until`` help text) is that the window is read on
    ``resolved_at``.

    ``resolver_tiers`` is NOT the tier breakdown of every in-window stamp its
    bare name suggests: it is tallied after the gated and self-resolved
    exclusions, so it covers only the subset that reached the resolver check.
    Its non-human entries are a fifth way a sample is lost, and which class lost
    it is on :attr:`ClassAgreement.non_human_resolver` — the tier says what took
    the sample, the class row says who paid.
    """

    since: datetime
    until: datetime
    classes: tuple[ClassAgreement, ...]
    gated_stamps: int
    self_resolved: int
    unresolved_lifetime: int
    rejected_stamps: int
    resolver_tiers: Mapping[str, int]

    def for_class(self, ruling_class: str) -> ClassAgreement | None:
        """The row for *ruling_class*, or ``None`` when it had no counted record."""
        return next((c for c in self.classes if c.ruling_class == ruling_class), None)


def _as_aware(parsed: datetime) -> datetime:
    """Read a naive datetime as UTC; leave an offset-bearing one alone.

    THE single rule normalising both sides of the window comparison — the
    record's ``resolved_at`` and the operator's ``--since``/``--until``. Two
    copies that merely happened to agree is how the naive-argument
    ``TypeError`` got in: ``_resolved_at`` coerced and the CLI did not, so a
    perfectly valid ``--since 2026-09-01`` crashed the sweep on the first
    stamped record it reached.

    UTC rather than local time because every instant this module compares is
    written as UTC by ``escalation/src/escalation/queue.py::resolve``, so an
    unqualified operator argument means UTC too.
    """
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _resolved_at(record: Escalation) -> datetime | None:
    """*record*'s resolution instant as an aware datetime, or ``None``.

    Parsed, never string-compared: ``resolved_at`` is written by several call
    sites and an offset-bearing timestamp sorts differently as text than it does
    as an instant.
    """
    if record.resolved_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(record.resolved_at)
    except ValueError:
        logger.warning('unparsable resolved_at %r on %s', record.resolved_at, record.id)
        return None
    return _as_aware(parsed)


#: The CONTENT faults the sweep treats as "skip this record", passed to
#: ``escalation/src/escalation/queue.py::read_escalation_for_scan`` rather than
#: hard-coded there — queue.py and sweep.py deliberately catch different tuples.
#: ``ValueError`` is a deliberate member and wider than queue.py's own tuple:
#: ``UnicodeDecodeError`` on a truncated or binary file subclasses it, and this
#: sweep must survive one, since a single escaping exception anywhere in a
#: 5000-record queue costs the entire weekly measurement.
_SWEEP_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError, KeyError, TypeError, ValueError,
)


def agreement_report(
    escalations_dir: Path | str, *, since: datetime, until: datetime,
) -> AgreementReport:
    """Count how often the shadow proposals matched what actually happened.

    Sweeps the queue root and its archive through
    ``escalation/src/escalation/queue.py::iter_all_escalation_paths``, which
    already handles root-wins-on-collision, archive-only multi-date duplicates
    and a missing directory (yields nothing rather than raising). Records with
    no parsable shadow ruling are skipped — the overwhelming majority carry
    none.

    Each file is read through ``queue.py::read_escalation_for_scan``, the
    audited helper every unlocked glob-then-read scan in this package uses, NOT
    an inline ``read_text``. This is a snapshot-then-read over a LIVE tree: the
    skill points the operator at ``<project_root>/data/escalations``, and a
    concurrent ``resolve()`` or the ``prune_archive`` pass of another
    orchestrator's startup sweep can relocate any listed file before this loop
    reaches it. An inline read raises ``FileNotFoundError`` there and takes the
    whole measurement down. The helper keeps vanished (DEBUG — routine
    archival), unreadable (WARNING — a real I/O fault) and unparsable (WARNING)
    distinguishable, so this stays a skip rather than the blanket
    ``except OSError`` the no-silent-fail-soft invariant forbids.

    THE ORDER OF CHECKS IS PART OF THE CONTRACT: unstamped -> unresolved ->
    out-of-window -> unreadable -> gated -> self_resolved -> non-human resolver
    -> not_comparable -> agreed/diverged. Everything up to ``not_comparable`` is
    "this record must not contribute to a rate at all"; putting any of them
    later would let an in-window matching close fall through to ``agreed``
    first. The non-human-resolver step tallies per class as well as per tier;
    :attr:`ClassAgreement.non_human_resolver` owns why.

    THE WINDOW COMES BEFORE THE TWO EXCLUDED BUCKETS THAT CAN BE WINDOWED, so
    that every number printed under the window header is a number from that
    window. Counting them first made them lifetime totals rendered beside a
    weekly denominator: on the real archive they grow monotonically forever, so
    a weekly report would eventually show ``gated_stamps=140`` next to a
    four-item comparable denominator — the inverse of the "a small sample and a
    discarded one must not look alike" property the buckets exist to give.

    Pendingness is decided FIRST because a pending record has no ``resolved_at``
    to window on: it would otherwise be dropped by the window filter and vanish
    from every bucket. That order also decides where a pending GATED or pending
    UNREADABLE stamp lands — ``unresolved_lifetime``, not ``gated_stamps`` or
    ``rejected_stamps`` — which is the honest reading: nothing has been ruled on
    it yet, and an unwindowable record must not enter a windowed number.

    ``self_resolved`` — ``triaged_by is not None and triaged_by ==
    resolved_by`` — is the bucket task 5361 made necessary.
    ``escalation/src/escalation/classify.py::_HUMAN_RESOLVERS`` contains
    ``escalation-watcher``, so a watcher that stamps a proposal and then closes
    the record itself under that standing rule would otherwise be read back as
    a human agreeing with itself, inflating the very ``design_concern`` class
    this measurement exists to judge, toward its own adoption threshold.

    The limit of that check, stated honestly: attribution is server-enforced
    only for a header-bearing identity —
    ``escalation/src/escalation/server.py::stamp_triage`` overrides
    ``triaged_by`` from ``X-Escalation-Identity`` ONLY when the header is
    present — so for the header-less interactive channel this is a backstop over
    a convention rather than a guarantee. That is why
    ``skills/escalation-watcher/SKILL.md`` forbids the stamp in the first place
    and this bucket only makes a violation visible.
    """
    agreed: Counter[str] = Counter()
    diverged: Counter[str] = Counter()
    not_comparable: Counter[str] = Counter()
    non_human: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    gated_stamps = 0
    self_resolved = 0
    unresolved_lifetime = 0
    rejected_stamps = 0

    for path in iter_all_escalation_paths(Path(escalations_dir)):
        record, _reason = read_escalation_for_scan(
            path, context='shadow_ruling.agreement_report',
            parse_errors=_SWEEP_PARSE_ERRORS,
        )
        if record is None:
            continue

        stamp = scan_triage_note(record.triage_note)
        if stamp.ruling is None and not stamp.rejected:
            continue

        if record.status == 'pending':
            unresolved_lifetime += 1
            continue

        resolved_at = _resolved_at(record)
        if resolved_at is None or not since <= resolved_at <= until:
            continue

        ruling = stamp.ruling
        if ruling is None:
            rejected_stamps += 1
            continue

        if mechanically_gated(record) is not None:
            gated_stamps += 1
            continue
        if record.triaged_by is not None and record.triaged_by == record.resolved_by:
            self_resolved += 1
            continue

        tier = classify_resolver_tier(record.resolved_by)
        tiers[tier] += 1
        if tier != 'human':
            non_human[ruling.ruling_class] += 1
            continue

        # COMPARABILITY IS A PROPERTY OF BOTH SIDES, and the observed side can
        # be missing on a record whose proposal is perfectly comparable:
        # ``server.py::resolve_issue`` documents a live legacy class (D10) that
        # resolves without a ``resolution_action``, and the pre-stamp that would
        # have filled it fires only on the MCP path. Scored against a C1
        # proposal that becomes a silent ``diverged`` for every such record —
        # depressing the single number adoption is read off, in the direction
        # that looks like disagreement rather than like missing data.
        if (ruling.proposed_action not in COMPARABLE_ACTIONS
                or record.resolution_action is None):
            not_comparable[ruling.ruling_class] += 1
        elif record.resolution_action == ruling.proposed_action:
            agreed[ruling.ruling_class] += 1
        else:
            diverged[ruling.ruling_class] += 1

    counted = sorted(set(agreed) | set(diverged) | set(not_comparable) | set(non_human))
    return AgreementReport(
        since=since,
        until=until,
        classes=tuple(
            ClassAgreement(
                ruling_class=slug,
                agreed=agreed[slug],
                diverged=diverged[slug],
                not_comparable=not_comparable[slug],
                non_human_resolver=non_human[slug],
            )
            for slug in counted
        ),
        gated_stamps=gated_stamps,
        self_resolved=self_resolved,
        unresolved_lifetime=unresolved_lifetime,
        rejected_stamps=rejected_stamps,
        resolver_tiers=MappingProxyType(dict(sorted(tiers.items()))),
    )


#: How far back ``main`` looks when no window is given — "weekly", per
#: ``docs/escalation-standing-policy.md``.
_DEFAULT_WINDOW = timedelta(days=7)

_ROW = (
    '{cls:<40} {agreed:>7} {diverged:>9} {not_comparable:>15} {non_human:>10} '
    '{comparable:>11} {rate:>8}'
)


def _as_json(report: AgreementReport) -> str:
    return json.dumps(
        {
            'since': report.since.isoformat(),
            'until': report.until.isoformat(),
            'classes': [
                {
                    'class': c.ruling_class,
                    'agreed': c.agreed,
                    'diverged': c.diverged,
                    'not_comparable': c.not_comparable,
                    'non_human_resolver': c.non_human_resolver,
                    'comparable': c.comparable,
                    'total': c.total,
                    'agreement_rate': c.agreement_rate,
                }
                for c in report.classes
            ],
            'gated_stamps': report.gated_stamps,
            'self_resolved': report.self_resolved,
            'unresolved_lifetime': report.unresolved_lifetime,
            'rejected_stamps': report.rejected_stamps,
            'resolver_tiers': dict(report.resolver_tiers),
        },
        indent=2,
        sort_keys=True,
    )


def _as_table(report: AgreementReport) -> str:
    """Render the report so an operator can paste it and a reader can decide.

    Every number the adoption threshold needs is on the page: the comparable
    DENOMINATOR beside the rate, and every excluded bucket — always, even at
    zero. A class whose stamps were mostly thrown out for self-agreement, or
    thrown out unread, is not a class with a small sample, and the output must
    not let the two look alike.

    The excluded buckets sit on their own line under the window header, and the
    one that is NOT from the window says so in its own name
    (``unresolved_lifetime``) rather than relying on the reader to know which
    numbers on that line share the header's window.
    """
    lines = [
        f'shadow ruling agreement — window {report.since.isoformat()} .. '
        f'{report.until.isoformat()}',
        '',
    ]
    if report.classes:
        lines.append(_ROW.format(
            cls='class', agreed='agreed', diverged='diverged',
            not_comparable='not_comp', non_human='non_human',
            comparable='comparable', rate='rate',
        ))
        for c in report.classes:
            rate = 'n/a' if c.agreement_rate is None else f'{c.agreement_rate * 100:.1f}%'
            lines.append(_ROW.format(
                cls=c.ruling_class, agreed=c.agreed, diverged=c.diverged,
                not_comparable=c.not_comparable, non_human=c.non_human_resolver,
                comparable=c.comparable, rate=rate,
            ))
    else:
        lines.append('no shadow rulings in window')
    lines.extend([
        '',
        f'gated_stamps={report.gated_stamps} self_resolved={report.self_resolved} '
        f'rejected_stamps={report.rejected_stamps} '
        f'unresolved_lifetime={report.unresolved_lifetime}',
        'resolver_tiers: ' + (
            ' '.join(f'{tier}={n}' for tier, n in report.resolver_tiers.items()) or '(none)'
        ),
    ])
    return '\n'.join(lines)


class _BadWindowArg(ValueError):
    """An unparsable ``--since``/``--until``, carrying the flag that carried it.

    Raised rather than exiting so the failure reaches ``main``'s own exit path
    beside the ``--queue-dir`` branch: this module's CLI discipline is to print
    to stderr and RETURN 2, which an argparse ``type=`` callback would break by
    raising ``SystemExit`` from inside ``parse_args``.
    """


def _parse_window_arg(value: str, flag: str) -> datetime:
    """Parse one window bound, naming the flag AND the value when it is junk.

    Naming both matters: an operator who typed ``--since yesterday`` needs to
    see which argument the parser rejected, not a bare isoformat complaint.
    """
    try:
        return _as_aware(datetime.fromisoformat(value))
    except ValueError as exc:
        raise _BadWindowArg(
            f'{flag} is not an ISO-8601 datetime: {value!r} ({exc})'
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m escalation.shadow_ruling``.

    Invoked from the repo root as::

        uv run --directory escalation python -m escalation.shadow_ruling \\
            --queue-dir <project_root>/data/escalations

    Read-only: it opens escalation JSON and prints. It never resolves, stamps or
    moves anything.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Weekly shadow-ruling agreement count: how often a stamped proposal '
            'matched what actually happened. Measurement only — grants no authority.'
        ),
    )
    parser.add_argument(
        '--queue-dir', required=True, type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--since', default=None,
        help='ISO-8601 window start on resolved_at, naive values read as UTC '
             '(default: 7 days ago).',
    )
    parser.add_argument(
        '--until', default=None,
        help='ISO-8601 window end on resolved_at, naive values read as UTC '
             '(default: now).',
    )
    parser.add_argument(
        '--json', action='store_true', default=False,
        help='Emit machine-readable JSON instead of the table.',
    )
    args = parser.parse_args(argv)

    # LOUD, not an all-zero table: an empty report and a misconfigured path must
    # not look identical, or a typo'd path reads back as a clean measurement.
    if not args.queue_dir.is_dir():
        print(f'queue-dir is not a directory: {args.queue_dir}', file=sys.stderr)
        return 2

    now = datetime.now(UTC)
    try:
        until = _parse_window_arg(args.until, '--until') if args.until else now
        since = (
            _parse_window_arg(args.since, '--since') if args.since
            else until - _DEFAULT_WINDOW
        )
    except _BadWindowArg as bad:
        print(bad, file=sys.stderr)
        return 2

    # Same discipline, second failure mode: both bounds parsed, so nothing
    # above catches this, but an inverted window matches no record by
    # construction and so renders as `no shadow rulings in window` with
    # all-zero buckets — byte-for-byte what a genuinely quiet week renders as.
    # Rejected rather than reported, because a reader cannot tell those apart.
    # INVERTED, not empty: `since == until` is a legitimate zero-width query.
    if since > until:
        print(
            f'--since {since.isoformat()} is after --until {until.isoformat()}: '
            'an inverted window matches no record',
            file=sys.stderr,
        )
        return 2

    report = agreement_report(args.queue_dir, since=since, until=until)
    print(_as_json(report) if args.json else _as_table(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
