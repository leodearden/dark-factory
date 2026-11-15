"""orchestrator.decision_registry — the fleet-global DecisionRecord registry at ``<fleet_root>/decisions/<id>.json``.

Schema, store, per-id lock, state transitions, refile merges, queue and
project-token normalisation, reap and migrate. The decision verbs that serve
it from a shell live behind ``session_registry.py <verb>``.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves
(``shared.project_id``) and its sibling ``fleet_files``. It never imports
``session_registry``: in a bare-shell (tier-1) run session_registry is
``__main__``, so an upward import would load a second copy of it. That
footprint also lets task 9778 move this module into ``shared.session``
unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Reach this checkout's stdlib-only shared leaves from a bare shell; rationale in
# test_session_registry.py::TestStdlibOnlySelfContainment.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.project_id import fold_project_token

# Imported by the spelling this module was loaded by: relative as
# orchestrator.decision_registry, bare as a tier-1 sibling or __main__.
# TYPE_CHECKING points the type checker at the package arm.
if TYPE_CHECKING or __package__:
    from .fleet_files import atomic_write_text, exclusive_sidecar_lock, fleet_root
else:
    from fleet_files import atomic_write_text, exclusive_sidecar_lock, fleet_root

logger = logging.getLogger(__name__)


class DecisionState(StrEnum):
    """Wire-format lifecycle state for a DecisionRecord (Fleet Cockpit C1, PRD §6.1).

    Modeled as a StrEnum contract for writers, but ``DecisionRecord.state``
    itself is a plain ``str`` field with NO from_dict coercion through this
    enum -- mirrors session_registry.py::SpawnMode's additive-safe rationale: an unrecognized
    value must still round-trip rather than raise.

    OPEN: filed, awaiting a human decision.
    ANSWERED: a decision has been recorded.
    DROPPED: withdrawn/superseded without an answer.
    """

    OPEN = 'open'
    ANSWERED = 'answered'
    DROPPED = 'dropped'


class AuditVerdict(StrEnum):
    """The sitting's verdict on an audited executor close."""

    OK = 'ok'
    WRONG = 'wrong'


@dataclass(frozen=True)
class AuditOutcome:
    """One audit of a closed DecisionRecord: the verdict, the auditor's note, when, the class judged, and which close.

    ``closed_at`` is the ``closed_at`` of the close the audit judged; ''
    judges a close that recorded no instant.
    """

    verdict: AuditVerdict
    note: str
    audited_at: str
    policy_class: str
    closed_at: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'verdict', AuditVerdict(self.verdict))
        if not isinstance(self.audited_at, str):
            raise ValueError(f'audited_at must be ISO-8601 text, got {self.audited_at!r}')
        datetime.fromisoformat(self.audited_at)
        if not isinstance(self.policy_class, str) or not self.policy_class:
            raise ValueError(f'policy_class must be a non-empty class name, got {self.policy_class!r}')
        if not isinstance(self.closed_at, str):
            raise ValueError(f"closed_at must be ISO-8601 text or '', got {self.closed_at!r}")
        if self.closed_at:
            datetime.fromisoformat(self.closed_at)

    def to_dict(self) -> dict[str, str]:
        return {
            'verdict': str(self.verdict),
            'note': self.note,
            'audited_at': self.audited_at,
            'policy_class': self.policy_class,
            'closed_at': self.closed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditOutcome:
        return cls(
            verdict=data['verdict'],
            note=str(data.get('note') or ''),
            audited_at=data['audited_at'],
            policy_class=data['policy_class'],
            closed_at=data.get('closed_at') or '',
        )


@dataclass
class DecisionRecord:
    """One decision-record — ``<fleet_root>/decisions/<id>.json`` (Fleet Cockpit C1 B2, PRD §6.1).

    Only id/project/text/filed_at are mandatory; every other field has a
    documented default so a minimally-filed decision is still well-formed.
    Mirrors session_registry.py::LeaseHolder's plain to_dict/from_dict/to_json/from_json shape.

    id: this record's identity/filename key; see decision_path_for_id.
    project: the project_id this decision concerns.
    text: the decision/question text as filed.
    filed_at: ISO-8601 timestamp of when this decision was filed.
    session_id: the filer's provenance label, or None -- in practice a
        watcher's lease token (``<lease-name>-<pid>``), which is NOT a
        session-registry record key. Kept as filed; never resolved.
    record_slug: the filer's session-registry record key (a
        ``session_registry.py::record_path_for_slug`` slug), or '' when it was unresolvable or the
        record predates this field. Derived by ``write-decision`` from
        ``$CLAUDE_PID``'s pid pointer, never typed by the filer.
    task_id: the task this decision is scoped to, or None.
    escalation_id: the escalation this decision resolves, or None.
    options: the candidate answers offered, or None.
    manual_boost: an operator-assigned priority nudge; defaults to 0.
    state: current lifecycle state; see DecisionState. Defaults to
        ``DecisionState.OPEN``.
    severity: the parked escalation's severity (info|blocking|critical|
        urgent), used by the cockpit decision queue to weight this ask
        relative to other rows. Defaults to '' (unknown/unset) when the
        filer doesn't supply one or the record predates this field; ''
        falls back to the scoring defaults.severity weight rather than
        raising or coercing to a recognized value.
    escalations_dir: the source escalation QUEUE this record's
        ``escalation_id`` belongs to, stored as a normalized absolute path
        (see normalize_escalations_dir). Needed because DecisionRecords are
        FLEET-GLOBAL while an escalation id (``esc-<taskid>-<n>``) is unique
        only WITHIN one queue, and a single project can run more than one --
        dark_factory runs ``data/escalations`` and
        ``data/reconciliation/escalations`` over the same id namespace, so
        without this second axis the reap-decisions reaper can close one
        watcher's decision against an unrelated same-named escalation in the
        other queue (task 3528). Defaults to '' (unknown/legacy -- a record
        filed before this field existed, or by a caller that didn't supply
        it), which makes the reaper fall back to project-only scoping.
    closing_evidence: the evidence that decided a closed record, quoted
        verbatim. ``docs/escalation-standing-policy.md`` requires the
        DecisionRecord to quote the deciding evidence, which no field could
        hold before; packing it into ``text`` (the cockpit's one-line
        question) would bury a structured fact in prose. Written only by
        close_decision_with_evidence, which the sitting preparer's apply step
        drives through the ``close-decision`` verb, and cleared only by
        reopen_decision. Defaults to '' (not
        closed with evidence, or filed before this field existed). CUSTODY,
        like ``state``: a watcher's re-file never changes it.
    closed_at: the ISO-8601 UTC instant close_decision_with_evidence recorded
        the close -- on the reap-decisions-got-there-first path, when the
        evidence was attached. Written only by close_decision_with_evidence,
        in the same write as ``closing_evidence``, cleared with it by
        reopen_decision, and CUSTODY like it. It
        exists because the return brief windows its autonomous-closes section
        on when a close happened, and ``filed_at`` is the wrong clock for
        that: a carve-out close targets an L2 that has been open a while.
        Defaults to '' (not closed with evidence, or closed before this field
        existed).
    audit_outcomes: the sitting's AuditOutcomes for this record's executor
        closes, oldest first; empty when never audited. APPEND-ONLY: written
        only by record_audit_outcome, which adds one outcome per audited close
        and never replaces one. CUSTODY like ``closing_evidence``, and kept by
        reopen_decision, because every audited close stays history the
        per-class wrong rate counts. Each outcome names the close it judged
        (its ``closed_at``), so a reopened record closed again is unaudited
        (``current_audit`` is None) until that new close is audited in turn,
        and both verdicts then count. Each names the policy class the
        audit judged, copied from the escalation record at audit time:
        escalation archives are pruned
        (escalation/src/escalation/archive.py::DEFAULT_RETENTION_DAYS) and the
        per-class counts must outlive them. The escalation record stays the
        class's home for the close itself.

    Concurrency: unlike session_registry.py::SessionRecord (single-writer-per-slug -- only the
    spawning session ever mutates its own record), a single decision id's
    file may be mutated by SEVERAL different subsystems: a C8 watcher (via
    update_decision_state), the C5 cockpit (via set_manual_boost), -- for
    task 3640's back-fill, running against live records while the watchers
    are up -- scripts/backfill_decision_queue_stamp.py (via
    set_decision_escalations_dir), the ``write-decision`` verb itself
    (task 3559), whose enrichment path folds a SECOND watcher's filing into
    an existing open record, -- for task 5376's sitting preparer -- the
    ``close-decision`` verb (via close_decision_with_evidence), the
    ``reopen-decision`` verb (via reopen_decision, task 4835), and the
    sitting's audit verb (via record_audit_outcome, task 8425). The
    ``write-decision`` verb is the only mutator that may CREATE the record
    rather than merely mutate an existing one, so it races on a path where
    nothing exists on disk yet. All seven serialize their
    read-modify-write span per-decision-id via
    decision_id_lock (a stable ``<id>.json.lock`` sidecar, mirroring task
    1609's escalation_id_lock), so a concurrent state-update, boost-update,
    queue-stamp or cross-queue filing racing on the same id no longer drops
    any of the mutations -- each write remains individually atomic AND the
    read+mutate+write span is serialized against other callers on the same
    id. See update_decision_state/set_manual_boost/
    set_decision_escalations_dir/session_registry.py::_run_write_decision/
    close_decision_with_evidence/reopen_decision/record_audit_outcome for the
    caller-facing note.
    """

    id: str
    project: str
    text: str
    filed_at: str
    session_id: str | None = field(default=None, kw_only=True)
    task_id: str | None = field(default=None, kw_only=True)
    escalation_id: str | None = field(default=None, kw_only=True)
    options: list[str] | None = field(default=None, kw_only=True)
    manual_boost: int = field(default=0, kw_only=True)
    state: str = field(default=DecisionState.OPEN, kw_only=True)
    severity: str = field(default='', kw_only=True)
    escalations_dir: str = field(default='', kw_only=True)
    closing_evidence: str = field(default='', kw_only=True)
    closed_at: str = field(default='', kw_only=True)
    record_slug: str = field(default='', kw_only=True)
    audit_outcomes: tuple[AuditOutcome, ...] = field(default=(), kw_only=True)

    @property
    def linked_session_slug(self) -> str | None:
        """The session-registry slug this decision links to, or None.

        THE one home of the decision -> session link policy: ``record_slug``
        when set, else ``session_id`` (which resolves only when a filer
        happened to write a real record key there). Every consumer that
        resolves a decision against the registry reads this, never either
        field directly.
        """
        return self.record_slug or self.session_id or None

    @property
    def current_audit(self) -> AuditOutcome | None:
        """The outcome that judged the close this record carries now, else None."""
        return next((outcome for outcome in reversed(self.audit_outcomes)
                     if outcome.closed_at == self.closed_at), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'project': self.project,
            'text': self.text,
            'filed_at': self.filed_at,
            'session_id': self.session_id,
            'task_id': self.task_id,
            'escalation_id': self.escalation_id,
            'options': self.options,
            'manual_boost': self.manual_boost,
            'state': str(self.state),
            'severity': self.severity,
            'escalations_dir': self.escalations_dir,
            'closing_evidence': self.closing_evidence,
            'closed_at': self.closed_at,
            'record_slug': self.record_slug,
            'audit_outcomes': [outcome.to_dict() for outcome in self.audit_outcomes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionRecord:
        return cls(
            id=data['id'],
            project=data['project'],
            text=data['text'],
            filed_at=data['filed_at'],
            session_id=data.get('session_id'),
            task_id=data.get('task_id'),
            escalation_id=data.get('escalation_id'),
            options=data.get('options'),
            manual_boost=data.get('manual_boost', 0),
            state=data.get('state', DecisionState.OPEN),
            severity=data.get('severity', ''),
            # `or ''` (not a plain .get default): a missing key AND an
            # explicit null both collapse to the one falsy sentinel, so the
            # `str` annotation stays honest against a hand-edited record and
            # the reaper's queue guard needs no None-vs-''-vs-missing branch.
            escalations_dir=data.get('escalations_dir') or '',
            # Same `or ''` idiom as escalations_dir above, for both custody fields.
            closing_evidence=data.get('closing_evidence') or '',
            closed_at=data.get('closed_at') or '',
            record_slug=data.get('record_slug') or '',
            audit_outcomes=_audit_outcomes_of(data.get('audit_outcomes')),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> DecisionRecord:
        return cls.from_dict(json.loads(raw))


def _audit_outcomes_of(value: Any) -> tuple[AuditOutcome, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f'audit_outcomes must be a list of mappings or null, got {value!r}')
    return tuple(AuditOutcome.from_dict(item) for item in value)


_DECISION_ID_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]')
"""Mirrors session_registry.py::_SLUG_SANITIZE_RE: anything outside [A-Za-z0-9._-] maps to '-', so
a malformed decision id can never escape decisions_dir via a path
separator."""


def decisions_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'decisions'


def decision_path_for_id(decision_id: str, root: Path | str | None = None) -> Path:
    """Resolve *decision_id* (see DecisionRecord.id) to its ``<decisions_dir>/<id>.json`` path.

    *decision_id* is sanitized through _DECISION_ID_SANITIZE_RE first, so it
    always resolves to a single file directly inside decisions_dir (mirrors
    session_registry.py::lease_path_for_name's path-escape guard).
    """
    stem = _DECISION_ID_SANITIZE_RE.sub('-', decision_id)
    return decisions_dir(root) / f'{stem}.json'


DECISION_ID_PROJECT_SEPARATOR = '-'


def qualify_decision_id(project: object, local_id: str) -> str:
    """The fleet-unique decision key: *local_id* prefixed by the canonical *project*.

    A pure join that never inspects *local_id*. A project that folds to ``''``
    leaves *local_id* unqualified. ``-`` is the separator because canonical
    tokens never contain one (so the join is injective), it survives
    _DECISION_ID_SANITIZE_RE (so the file stem equals the id), and it matches
    the sitting's existing ``<project>-<esc>`` ids. See
    plans/4835-decision-plumbing-decisions.md.
    """
    folded = fold_project_token(project)
    if not folded:
        return local_id
    return f'{folded}{DECISION_ID_PROJECT_SEPARATOR}{local_id}'


def write_decision(record: DecisionRecord, root: Path | str | None = None) -> bool:
    """Atomically write *record* to its own ``<decisions_dir>/<id>.json`` file.

    Self-guarding FAIL-SOFT: unlike session_registry.py::write_record (whose sole caller, the CLI
    main(), supplies its own outer try/except), this is called directly by
    C8 watchers and the C5 cockpit, so a write fault must log and return
    rather than raise into them. Returns True on success, False on any
    fault.
    """
    path = decision_path_for_id(record.id, root=root)
    try:
        atomic_write_text(path, record.to_json())
    except Exception:
        logger.error('write_decision: failed to write %s', path, exc_info=True)
        return False
    return True


def list_decisions(root: Path | str | None = None) -> list[DecisionRecord]:
    """Read every decision under decisions_dir. Missing dir -> [] (no raise).

    A single corrupt or foreign ``*.json`` file is skipped (logged at ERROR)
    rather than aborting the whole read -- mirrors session_registry.py::reap_stale_records'
    corrupt-body handling, so one bad file never breaks the cockpit's
    whole-queue read.
    """
    base = decisions_dir(root)
    if not base.is_dir():
        return []
    decisions: list[DecisionRecord] = []
    for path in sorted(base.glob('*.json')):
        try:
            decisions.append(DecisionRecord.from_json(path.read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.error('list_decisions: skipping unreadable %s', path, exc_info=True)
            continue
    return decisions


def _mutate_decision(
    decision_id: str,
    mutate: Callable[[DecisionRecord], None],
    *,
    caller: str,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Lock-serialized, fail-soft read-modify-write of one decision record.

    The single implementation of the field-setter body shared by
    update_decision_state, set_manual_boost, set_decision_escalations_dir
    (task 3640 amendment), close_decision_with_evidence (task 5376),
    reopen_decision (task 4835) and record_audit_outcome (task 8425). Those
    are the public, caller-facing names and keep their own docstrings; this holds the parts that MUST NOT diverge
    between them -- the lock placement, the read, the write, and the
    fail-soft except-tuple.

    That is not merely tidiness. The lock has to sit INSIDE the try/except for
    a lock-acquisition fault to be absorbed rather than raised at a C8/cockpit
    caller, and the except-tuple has to cover exactly the read/parse/write
    fault set. Copied per-setter, a fix to either is a fix that has to be
    remembered three times, and a missed copy fails silently in production
    while every test stays green.

    *mutate* is applied to the freshly-read record in-place; *caller* names the
    public helper in the ERROR log so a fail-soft None is still attributable to
    the API the caller actually invoked.

    Returns the mutated record, or None on ANY fault (missing file, corrupt
    body, lock-acquisition fault, write failure), never raising -- matching
    write_decision's contract.
    """
    path = decision_path_for_id(decision_id, root=root)
    try:
        with decision_id_lock(decision_id, root=root):
            record = DecisionRecord.from_json(path.read_text(encoding='utf-8'))
            mutate(record)
            if not write_decision(record, root=root):
                return None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error('%s: failed to read %s', caller, path, exc_info=True)
        return None
    return record


def update_decision_state(
    decision_id: str,
    state: str,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Read-modify-write *decision_id*'s state field.

    Self-guarding FAIL-SOFT: returns None (logs ERROR) on any fault -- a
    missing file, a corrupt body, a lock-acquisition fault, or a write
    failure -- rather than raising, matching write_decision's contract for
    its direct C8/cockpit callers.

    Concurrency NOTE (see DecisionRecord's docstring): this read-modify-write
    is now serialized per-decision-id via decision_id_lock (a stable
    ``<id>.json.lock`` sidecar, mirroring task 1609's escalation_id_lock), so
    a concurrent set_manual_boost (or a second update_decision_state) racing
    on the SAME decision id no longer drops either call's field mutation.
    """
    def _set(record: DecisionRecord) -> None:
        record.state = state

    return _mutate_decision(
        decision_id, _set, caller='update_decision_state', root=root
    )


def set_manual_boost(
    decision_id: str,
    boost: int,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Read-modify-write *decision_id*'s manual_boost field.

    Self-guarding FAIL-SOFT: returns None (logs ERROR) on any fault -- a
    missing file, a corrupt body, a lock-acquisition fault, or a write
    failure -- rather than raising, matching write_decision's contract for
    its direct C8/cockpit callers.

    Concurrency NOTE (see DecisionRecord's docstring): this read-modify-write
    is now serialized per-decision-id via decision_id_lock (a stable
    ``<id>.json.lock`` sidecar, mirroring task 1609's escalation_id_lock), so
    a concurrent update_decision_state (or a second set_manual_boost) racing
    on the SAME decision id no longer drops either call's field mutation.
    """
    def _set(record: DecisionRecord) -> None:
        record.manual_boost = boost

    return _mutate_decision(decision_id, _set, caller='set_manual_boost', root=root)


def set_decision_escalations_dir(
    decision_id: str,
    escalations_dir: str | Path,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Read-modify-write *decision_id*'s escalations_dir (queue stamp) field.

    Added for task 3640's back-fill of the legacy open population, but written
    as an ordinary field setter: any caller needing to (re)stamp a record's
    owning queue should come through here rather than rewriting the JSON.

    NORMALIZES ON THE WAY IN via normalize_escalations_dir, so every writer of
    this field stores ONE canonical spelling and the reaper's axis-2 compare
    stays honest -- a raw-stored dotted/trailing-slash spelling would compare
    unequal to the very queue it names, and the record would silently never
    close again (fail-open, so invisible). ``UNKNOWN_QUEUE`` passes through
    verbatim by that same normalizer's sentinel case: it is a state, not a
    path, and must not become cwd-dependent.

    Self-guarding FAIL-SOFT: returns None (logs ERROR) on any fault -- a
    missing file, a corrupt body, a lock-acquisition fault, or a write
    failure -- rather than raising, matching write_decision's contract for
    its direct C8/cockpit callers. That matters concretely for the back-fill,
    which lists the whole decision population and then writes each id back: a
    record closed or removed by a live watcher in between is expected, not
    exceptional, and must not abort a migration mid-population.

    Concurrency NOTE (see DecisionRecord's docstring): this read-modify-write
    is serialized per-decision-id via decision_id_lock (a stable
    ``<id>.json.lock`` sidecar, mirroring task 1609's escalation_id_lock).
    Its two siblings already race each other; the back-fill is a THIRD writer,
    running against live records while the C8 watchers and the cockpit are up,
    which is exactly why it goes through this helper instead of rewriting
    decision JSON itself -- doing that in a script would bypass both the lock
    and the atomic tmp+os.replace writer and reintroduce the race 1609/3528
    fixed.
    """
    def _set(record: DecisionRecord) -> None:
        record.escalations_dir = normalize_escalations_dir(escalations_dir)

    return _mutate_decision(
        decision_id, _set, caller='set_decision_escalations_dir', root=root
    )


CLOSING_DECISION_STATES = frozenset({DecisionState.ANSWERED, DecisionState.DROPPED})


def _decision_identity(project: object, escalations_dir: str | Path) -> tuple[str, str]:
    """The folded (project, queue stamp) pair a caller names a fleet-global decision record by."""
    return fold_project_token(project), normalize_escalations_dir(escalations_dir)


def _refuse_unless_named_record(
    decision_id: str,
    record: DecisionRecord,
    wanted: tuple[str, str],
    refused: type[Exception],
    *,
    action: str,
) -> None:
    """Raise *refused* unless *record* is the one *wanted* names: the compare-and-swap of close and reopen."""
    found = _decision_identity(record.project, record.escalations_dir)
    if found != wanted:
        raise refused(
            f'{decision_id} is project {found[0]!r} in queue {found[1]!r}, not the record the '
            f'caller named (project {wanted[0]!r} in queue {wanted[1]!r}); decision ids are '
            f'fleet-global, so another project\'s or queue\'s record is never {action}'
        )


class DecisionCloseRefused(Exception):
    """close_decision_with_evidence refused; the record is untouched and the message names why.

    Not a ValueError on purpose: _mutate_decision's fail-soft except-tuple
    absorbs ValueError, and a refusal must reach its caller rather than read as
    a missing record.
    """


def close_decision_with_evidence(
    decision_id: str,
    state: str,
    evidence: str,
    root: Path | str | None = None,
    *,
    expected_project: str,
    expected_escalations_dir: str | Path,
) -> DecisionRecord | None:
    """Close *decision_id* to *state* and record the deciding *evidence*, in ONE locked read-modify-write.

    Accepted: ``open`` -> ``answered`` | ``dropped``, and attaching evidence to
    a record ALREADY in *state* whose ``closing_evidence`` is empty. The second
    is the reap-decisions-got-there-first path: the sitting preparer's apply
    step resolves the escalation BEFORE recording evidence, so a server-side
    refusal aborts before the registry claims an answer, which leaves the
    reaper free to close the record in between.

    Decision ids are fleet-global while ``esc-<task>-<n>`` numbering restarts
    per project, so the caller names the record it means by the project and
    queue stamp it READ (compared folded, through fold_project_token and
    normalize_escalations_dir), and any other record at that id is refused.
    That is the close-side twin of session_registry.py::_run_write_decision's collision rule.

    Raises DecisionCloseRefused for a target state other than answered or
    dropped, empty evidence, a record whose project or queue stamp is not the
    expected one, a move between two different terminal states, or evidence
    already recorded (never overwritten). The two argument refusals run before
    the lock, because a lock sidecar is never cleaned up (see
    decision_id_lock); the three record refusals run inside the locked span,
    identity first.

    Otherwise FAIL-SOFT like its sibling setters: None (logged at ERROR) on a
    missing file, a corrupt body, a lock fault or a write failure.
    Concurrency: serialized per-decision-id via _mutate_decision, see
    DecisionRecord's docstring.
    """
    if state not in CLOSING_DECISION_STATES:
        raise DecisionCloseRefused(
            f'{decision_id}: close-decision closes to answered or dropped, not {str(state)!r}'
        )
    if not evidence.strip():
        raise DecisionCloseRefused(
            f'{decision_id}: closing evidence is empty; quote the deciding evidence verbatim'
        )

    wanted = _decision_identity(expected_project, expected_escalations_dir)

    def _close(record: DecisionRecord) -> None:
        _refuse_unless_named_record(decision_id, record, wanted, DecisionCloseRefused, action='closed')
        if record.closing_evidence:
            raise DecisionCloseRefused(
                f'{decision_id} already carries closing evidence; refusing to overwrite it'
            )
        if record.state not in (DecisionState.OPEN, state):
            raise DecisionCloseRefused(
                f'{decision_id} is {str(record.state)!r}; a move between terminal states '
                f'({str(record.state)!r} -> {str(state)!r}) is refused'
            )
        record.state = str(state)
        record.closing_evidence = evidence
        record.closed_at = datetime.now(UTC).isoformat()

    return _mutate_decision(
        decision_id, _close, caller='close_decision_with_evidence', root=root
    )


class DecisionReopenRefused(Exception):
    """reopen_decision refused; the record is untouched and the message names why.

    Not a ValueError, for DecisionCloseRefused's reason: _mutate_decision
    absorbs ValueError, and a refusal must reach its caller.
    """


def reopen_decision(
    decision_id: str,
    root: Path | str | None = None,
    *,
    expected_project: str,
    expected_escalations_dir: str | Path,
) -> DecisionRecord | None:
    """Re-open *decision_id* IN PLACE, in ONE locked read-modify-write.

    The caller names the record it means exactly as for
    close_decision_with_evidence, and any other record at that id raises
    DecisionReopenRefused. Clears ``closing_evidence`` and ``closed_at``,
    because a close never overwrites evidence, so a re-opened gate could
    otherwise never be closed with evidence again. Every other field is kept,
    ``filed_at``, ``manual_boost`` and ``audit_outcomes`` included (custody);
    each kept outcome names the close it judged, so the record's next close is
    unaudited until it is audited in turn. It is reopen-only
    rather than a generic state setter, which would bypass the evidence-quoting
    close; see plans/4835-decision-plumbing-decisions.md.

    Otherwise FAIL-SOFT like its sibling setters: None (logged at ERROR) on a
    missing file, a corrupt body, a lock fault or a write failure.
    """
    wanted = _decision_identity(expected_project, expected_escalations_dir)

    def _reopen(record: DecisionRecord) -> None:
        _refuse_unless_named_record(decision_id, record, wanted, DecisionReopenRefused, action='reopened')
        record.state = DecisionState.OPEN
        record.closing_evidence = ''
        record.closed_at = ''

    return _mutate_decision(decision_id, _reopen, caller='reopen_decision', root=root)


class DecisionAuditRefused(Exception):
    """record_audit_outcome refused; the record is untouched and the message names why.

    Not a ValueError, for DecisionCloseRefused's reason: _mutate_decision
    absorbs ValueError, and a refusal must reach its caller.
    """


def record_audit_outcome(
    decision_id: str,
    verdict: AuditVerdict,
    note: str,
    root: Path | str | None = None,
    *,
    expected_project: str,
    expected_escalations_dir: str | Path,
    audited_at: str,
    policy_class: str,
    closed_at: str,
) -> DecisionRecord | None:
    """Record the sitting's *verdict* on *decision_id*'s close, in ONE locked read-modify-write.

    The caller names the record exactly as for close_decision_with_evidence,
    and any other record at that id raises DecisionAuditRefused. *closed_at*
    is the ``closed_at`` of the close the auditor judged; DecisionAuditRefused
    also meets a record with no closing evidence (nothing was closed to
    audit), one whose close is no longer that one (reopened, or closed
    again), and one whose current close already carries an outcome (never
    overwritten). The outcome is APPENDED to ``audit_outcomes``: one that
    judged an earlier close of a reopened record is kept, so both count.
    *policy_class* is the class the audit judged. A malformed
    verdict, stamp or class raises ValueError before the lock.

    Otherwise FAIL-SOFT like its sibling setters: None (logged at ERROR) on a
    missing file, a corrupt body, a lock fault or a write failure.
    """
    outcome = AuditOutcome(
        verdict=verdict, note=note, audited_at=audited_at, policy_class=policy_class, closed_at=closed_at,
    )
    wanted = _decision_identity(expected_project, expected_escalations_dir)

    def _audit(record: DecisionRecord) -> None:
        _refuse_unless_named_record(decision_id, record, wanted, DecisionAuditRefused, action='audited')
        if not record.closing_evidence:
            raise DecisionAuditRefused(
                f'{decision_id} carries no closing evidence; only a closed record is audited'
            )
        if record.closed_at != closed_at:
            raise DecisionAuditRefused(
                f'{decision_id} now carries the close of {record.closed_at!r}, not the judged close of '
                f'{closed_at!r}; it was reopened or closed again since it was drawn'
            )
        current = record.current_audit
        if current is not None:
            raise DecisionAuditRefused(
                f'{decision_id} was already audited {current.verdict!s} at {current.audited_at}; '
                'refusing to overwrite it'
            )
        record.audit_outcomes = (*record.audit_outcomes, outcome)

    return _mutate_decision(decision_id, _audit, caller='record_audit_outcome', root=root)


def _merge_queue_and_escalation_id(
    existing: DecisionRecord,
    incoming: DecisionRecord,
) -> tuple[str, str | None]:
    """Pick ONE ``(escalations_dir, escalation_id)`` PAIR for the merged record.

    THE INVARIANT, and the only reason this is a helper rather than two
    lines inside merge_decision_enrichment: the returned pair is ALWAYS one
    an INPUT record could vouch for. Never one filer's queue joined to the
    other filer's id.

    Merging the two fields by independent rules is what makes that possible
    to get wrong -- adopt the incoming queue while keeping the existing id
    and the survivor carries a pair that existed on neither record.
    ``session_registry.py::_run_reap_decisions._status`` is a JOIN on exactly that pair: axis 2
    compares the stamp, then ``read_escalation_status(escalations_dir,
    escalation_id)`` resolves the id INSIDE that queue. Since
    ``esc-<taskid>-<n>`` ids are unique only WITHIN a queue (task 3528), a
    synthesized pair resolves against an UNRELATED escalation that merely
    shares the id -- and if that one has resolved, the decision closes.

    That is the fail-CLOSED direction, which session_registry.py::_run_reap_decisions' own
    docstring rules out: "an over-held decision is a human-triageable row,
    while a falsely closed one is invisible". It is also not hypothetical --
    it is the incident that docstring records, a RESOLVED orchestrator
    escalation closing a still-PENDING recon blocking gate that then sat
    invisible in the cockpit for ~7 days. Holding a record we cannot place
    is always the cheaper mistake.

    BOTH sides are normalized before comparison, never raw-string-compared:
    that is the fail-open mistake normalize_escalations_dir's own docstring
    exists to prevent, and the reaper's axis-2 guard normalizes both sides
    for exactly this reason.
    """
    existing_queue = normalize_escalations_dir(existing.escalations_dir)
    incoming_queue = normalize_escalations_dir(incoming.escalations_dir)

    # (1) SAME queue -- including both-'' and both-UNKNOWN_QUEUE. One shared
    # id namespace, so the two ids are comparable and no pair can be forged:
    # the ordinary fill-if-empty enrichment applies.
    if existing_queue == incoming_queue:
        return existing_queue, existing.escalation_id or incoming.escalation_id

    # (2) Queues DIFFER and the first filer's is usable -- FIRST-WRITER-WINS,
    # as a PAIR. Note the deliberate consequence: an empty
    # existing.escalation_id is NOT filled from incoming here, because
    # incoming's id belongs to INCOMING's namespace. Filling it would take a
    # record reap_answered_decisions safely skips outright (`if not
    # escalation_id: continue`) and make it resolvable against the wrong
    # queue -- manufacturing the fail-CLOSED join out of nothing.
    if existing_queue and existing_queue != UNKNOWN_QUEUE:
        if incoming_queue and incoming_queue != UNKNOWN_QUEUE:
            logger.warning(
                'decision %s was filed from TWO different escalation queues: keeping the '
                'first filer\'s stamp %s and DISCARDING %s. escalations_dir is a scalar '
                '(task 3640), so a cross-queue collapse can record only one queue -- this '
                'record is reapable only by the first one.',
                existing.id,
                existing_queue,
                incoming_queue,
            )
        return existing_queue, existing.escalation_id

    # (3) The first filer's queue is undetermined ('' or UNKNOWN_QUEUE) but it
    # already holds a DIFFERENT escalation id -- so we cannot tell whose
    # namespace that id lives in, and adopting the incoming queue would pair
    # it with an id from an undetermined one. REFUSE the upgrade, loudly.
    if existing.escalation_id and existing.escalation_id != incoming.escalation_id:
        logger.warning(
            'decision %s already holds escalation id %s under an undetermined queue (%r), '
            'so it will NOT adopt the incoming queue %s: pairing that stamp with an id '
            'from an unknown namespace could close this decision against an unrelated '
            'escalation. The record deliberately stays fail-OPEN (unreapable, a visible '
            'cockpit row); the remedy is the back-fill path '
            '(scripts/backfill_decision_queue_stamp.py / set_decision_escalations_dir, '
            'task 3640), which actually investigates provenance.',
            existing.id,
            existing.escalation_id,
            existing_queue,
            incoming_queue,
        )
        return existing_queue, existing.escalation_id

    # (4) Genuine enrichment -- the post-3640 case this whole merge exists
    # for: a legacy ``escalations_dir=''`` record becoming queue-scoped. The
    # first filer holds no id of its own, or both filers claim the SAME one,
    # so adopting the incoming record's own SELF-CONSISTENT pair clobbers
    # nothing. (The `or existing.escalation_id` fallback is reachable only
    # when BOTH ids are unset -- guard (3) above already returned for every
    # case where existing holds an id incoming does not share.)
    #
    # ADOPTION REQUIRES A *REAL* QUEUE, never UNKNOWN_QUEUE. The sentinel is
    # not an ordinary stamp that happens to be non-empty: '' still closes
    # under the reaper's project-only fallback, while ``<unknown>`` is
    # refused by EVERY reaper by name, so '' -> ``<unknown>`` is a strict
    # DOWNGRADE of reapability that no reaper could ever undo -- exactly the
    # loss the '' -> real direction above exists to repair, run backwards.
    # The sentinel means "investigated, could not determine" and may only be
    # SET by the back-fill path that did the investigating
    # (set_decision_escalations_dir, task 3640); a merge never gets to
    # ACQUIRE it from the other filer. Symmetric with the CLI verb, which
    # refuses the sentinel on input for the same reason.
    if incoming_queue and incoming_queue != UNKNOWN_QUEUE:
        return incoming_queue, incoming.escalation_id or existing.escalation_id
    # Incoming offers no adoptable queue -- none at all, or only the
    # sentinel -- so there is nothing to take and no id to take with it:
    # keep our own pair rather than trading a stamp for something worse.
    return existing_queue, existing.escalation_id


def merge_decision_enrichment(
    existing: DecisionRecord,
    incoming: DecisionRecord,
) -> DecisionRecord:
    """Fold a SECOND watcher's filing into an existing OPEN decision (task 3559).

    Implements the MODE-2 rule verbatim: a second watcher observing the same
    underlying human gate through a DIFFERENT escalation queue must ENRICH
    the existing open record -- never overwrite, clobber, or downgrade what
    the first watcher wrote. That shape is not hypothetical: task 3528
    requirement (b) records ``esc-5914-1`` surfacing the same reify gate on
    both dark_factory queues at once, and the cockpit must show it as ONE
    row, carrying the best information either watcher has.

    Field policy:

    - ``id`` / ``project``   -- from *existing*. The id is the JOIN KEY: it
      is the whole reason these two records are being merged.
    - ``filed_at`` / ``state`` / ``manual_boost`` / ``closing_evidence`` /
      ``closed_at`` / ``audit_outcomes`` -- from *existing* (CUSTODY). A second
      watcher must not restamp queue age, re-open or close the record (that is
      update_decision_state's job), reset an operator's C5 cockpit boost,
      erase the evidence a close recorded or when it recorded it, or erase
      the sitting's audit of that close.
    - ``text`` / ``task_id`` / ``session_id`` / ``record_slug`` /
      ``options`` -- keep *existing* where it is non-empty; take *incoming*
      ONLY to fill a field the first filer left empty/None. That fill is what
      makes this enrichment rather than a no-op.
    - ``severity``           -- ``_max_decision_severity``: never downgrade.
    - ``escalations_dir`` + ``escalation_id`` -- NOT independent fields, and
      ``escalation_id`` is deliberately NOT a plain fill-if-empty one: the
      two move together as a PAIR via _merge_queue_and_escalation_id, whose
      docstring carries the reasoning. In outline: same queue -> one id
      namespace, so ordinary fill-if-empty; different queues with the first
      filer's usable -> keep BOTH of the first filer's (FIRST-writer-wins),
      with a WARNING naming the id and both queues when the discarded one is
      real; first filer's queue undetermined ('' / ``UNKNOWN_QUEUE``) but
      already carrying a different id -> refuse the upgrade, loudly; and
      otherwise adopt the incoming record's own self-consistent pair PROVIDED
      its queue is REAL, which is the genuine post-3640 enrichment (a legacy
      unstamped record becoming queue-scoped, or an ``UNKNOWN_QUEUE`` one
      becoming reapable). A merge never ACQUIRES ``UNKNOWN_QUEUE`` from the
      other filer: reapability only ever moves upwards here.

    WHY THIS FIELD STAYS A SCALAR rather than becoming a list of queues,
    despite the cross-queue case obviously wanting one: (1) it would break
    SILENTLY. normalize_escalations_dir does ``str(value).strip()``, so a
    list stringifies to ``"['/a', '/b']"`` with no exception and no log, and
    the reaper's axis-2 guard would then skip EVERY list-stamped decision on
    EVERY queue -- permanently unreapable, invisibly. from_dict's
    ``data.get(...) or ''`` likewise admits a list past a str-annotated
    field unguarded. (2) Task 3640 (merged) hard-commits the field to a
    scalar, adding UNKNOWN_QUEUE as a THIRD scalar state plus a back-fill
    that stamps the live population as scalars; a list would contradict
    shipped, tested behaviour. (3) DECIDED won't-do by task 4835 (a faithful
    widening would be (queue, escalation_id) pairs, and any-of closure fails
    closed); see plans/4835-decision-plumbing-decisions.md.

    First-writer-wins therefore leaves the known MODE-2 reap gap that
    test_main_reap_decisions_mode2_collapsed_decision_is_reapable_only_by_
    its_stamped_queue already pins and accepts -- unchanged in KIND, but now
    DETERMINISTIC (the first filer's queue) instead of racy (whichever
    watcher wrote last). The WARNING is what surfaces it.

    PURE and side-effect-free apart from the two queue warnings: it returns
    a NEW record via dataclasses.replace and mutates neither argument, so it
    stays trivially testable in isolation from the CLI verb and a caller
    holding the pre-merge record still sees what it read.
    """
    merged_queue, merged_escalation_id = _merge_queue_and_escalation_id(existing, incoming)

    return dataclasses.replace(
        existing,
        text=existing.text or incoming.text,
        task_id=existing.task_id or incoming.task_id,
        escalation_id=merged_escalation_id,
        session_id=existing.session_id or incoming.session_id,
        record_slug=existing.record_slug or incoming.record_slug,
        options=existing.options or incoming.options,
        severity=_max_decision_severity(existing.severity, incoming.severity),
        escalations_dir=merged_queue,
    )


def merge_same_queue_refile(
    existing: DecisionRecord,
    incoming: DecisionRecord,
) -> DecisionRecord:
    """Fold the SAME watcher's re-filing of its OWN id into an existing record.

    Deliberate SIBLING of merge_decision_enrichment above -- read the two
    together. Same custody set, opposite treatment of the watcher-owned half.
    This is the other axis of session_registry.py::_run_write_decision's upsert: not two watchers
    seeing one gate through two queues (that is enrichment), but ONE watcher
    re-filing its own stable id across a restart, which both watcher SKILL.md
    files promise is idempotent.

    Field policy:

    - ``text`` / ``severity`` / ``task_id`` / ``session_id`` /
      ``record_slug`` / ``escalation_id`` / ``options`` / ``escalations_dir``
      -- from *incoming*, VERBATIM, including a severity DOWNGRADE and a field
      going EMPTY. The watcher is the sole authority on its own escalation, and
      freezing the first values (enrichment's fill-if-empty +
      _max_decision_severity) would strand stale prose and a stale severity
      in the cockpit queue forever. This is the whole reason the same-queue
      case is not just routed through merge_decision_enrichment.
    - ``filed_at`` / ``state`` / ``manual_boost`` / ``closing_evidence`` /
      ``closed_at`` / ``audit_outcomes`` -- from *existing* (CUSTODY), and it
      is the SAME set merge_decision_enrichment keeps, because custody does not depend on
      which queue re-filed. ``closing_evidence`` and ``closed_at`` travel
      with ``state``: they are the evidence for the disposition being held
      and when it was recorded. ``filed_at``
      is queue AGE, which drives the cockpit's ordering, and a restart is not
      news about it. ``manual_boost`` is the OPERATOR's C5 field, written by
      set_manual_boost. ``state`` is the operator's / reaper's DISPOSITION,
      written by update_decision_state.
    - ``id`` / ``project`` -- not forced here, unlike enrichment (which
      rebuilds from *existing*): this helper rebuilds from *incoming*, and
      its caller has already established both are equal -- the id is the
      on-disk file key, and session_registry.py::_run_write_decision reaches this arm only after
      ``existing.project == project``. Do not widen those preconditions
      without revisiting this line.

    WHY ``state`` IS SAFE TO HOLD HERE BUT NOT CROSS-QUEUE (task 3872). This
    helper is scoped by its caller to a SAME-project, SAME-queue re-file, and
    within ONE queue an ``esc-<taskid>-<n>`` id is unique -- that is the
    entire premise of task 3528's queue axis. So this is the same gate the
    human already answered or dropped rather than a new ask, and preserving
    their disposition is respecting a VERIFIED human act. Across queues the
    id namespaces genuinely collide (dark_factory runs ``data/escalations``
    and ``data/reconciliation/escalations`` over one namespace), so a
    non-open cross-queue filing may be an unrelated NEW ask;
    session_registry.py::_run_write_decision deliberately keeps today's full overwrite there.

    THAT UNIQUENESS IS STRONG BUT NOT ABSOLUTE, and the hole is named here
    once rather than rounded off, because every other statement of this
    policy (the caller's comment, both watcher SKILLs) leans on it:
    EscalationQueue._recover_seq_from_disk (escalation/queue.py) rebuilds a
    LOST or corrupt per-task seq counter by scanning the queue root and
    archive, and its own docstring concedes both bounds -- the archive half
    is bounded by prune_archive retention, and an id minted but not yet
    submitted when the counter was lost is invisible to the scan. After a
    counter loss an id CAN therefore be re-minted inside one queue, landing a
    genuinely NEW gate on a closed row that this helper then holds closed --
    invisible in C5b, which filters to state=='open', i.e. the fail-CLOSED
    direction. The residual needs a counter loss to reach and is far narrower
    than the unbounded harm below, and session_registry.py::_run_write_decision's divergence
    WARNING is its backstop: the hold is ANNOUNCED for a human or agent to
    adjudicate rather than applied silently, which is exactly the case that
    reads it.

    WHY THIS IS NOT THE FAIL-CLOSED DIRECTION the reaper docstrings in this
    module warn about. "An over-held decision is a human-triageable row,
    while a falsely closed one is invisible" governs the REAPER's join across
    an id namespace it CANNOT verify -- an automatic close on uncertain
    evidence. Here identity is knowable (above, down to the one named
    counter-loss residual) and the closure came from an
    operator's explicit C5b act (cockpit/app.py -> update_decision_state(...,
    DROPPED)) or from the reaper resolving against an escalation in this SAME
    queue. The alternative is not a benign over-surfacing but an UNBOUNDED
    one: a watcher re-files on EVERY restart while an item stays parked, and
    reap_answered_decisions skips a non-open record ("already resolved -- no
    re-close"), so without this the operator's dismissal is undone forever
    and C5b's drop action is inert for exactly the class of row it exists
    for. When the SAME gate genuinely needs a human again, the row is
    re-opened in place with the ``reopen-decision`` verb (reopen_decision),
    which session_registry.py::_run_write_decision's divergence WARNING names; a genuinely
    DIFFERENT ask is filed under a new id.

    ADDITIVE-SAFE: ``state`` is copied as an opaque ``str``, never coerced
    through DecisionState -- mirrors DecisionRecord's own no-coercion note,
    so a disposition a future writer adds round-trips instead of being reset
    to 'open' by a module that has not been taught about it.

    KNOWN RESIDUAL: a LEGACY unstamped (``escalations_dir=''``) non-open
    record re-filed by its own watcher does NOT reach this helper, because
    the caller's queue-equality test cannot resolve ``'' == stamp``. Guessing
    whose namespace an unstamped record belongs to is precisely what
    _merge_queue_and_escalation_id refuses to do, and task 3640's back-fill
    is draining that population; it is left as the full overwrite rather than
    papered over here.

    PURE and side-effect-free -- including of LOGGING, which stays in the CLI
    verb at the policy boundary (mirroring how enrichment's queue warnings
    live in _merge_queue_and_escalation_id, not in the writer). Returns a NEW
    record via dataclasses.replace and mutates neither argument.
    """
    return dataclasses.replace(
        incoming,
        filed_at=existing.filed_at,
        state=existing.state,
        manual_boost=existing.manual_boost,
        closing_evidence=existing.closing_evidence,
        closed_at=existing.closed_at,
        audit_outcomes=existing.audit_outcomes,
    )


@contextlib.contextmanager
def decision_id_lock(decision_id: str, root: Path | str | None = None) -> Iterator[None]:
    """Per-decision-id exclusive advisory lock using a stable sidecar file.

    Mirrors escalation's ``escalation_id_lock`` (escalation/queue.py:24-69,
    task 1609) near-verbatim, retargeted to the decisions dir.

    WHY A SIDECAR (PRD-D3 rationale, same as 1609): write_decision's writer
    is atomic tmp+os.replace (``fleet_files.py::atomic_write_text``). After a replace,
    the data file
    ``<decisions_dir>/<id>.json`` is a NEW inode. A second writer that
    flock()s the (new) data-file path binds to a different inode and races
    anyway -- the lock is defeated. The fix is a STABLE lock target:
    ``<decisions_dir>/<id>.json.lock``, created once via os.open(O_CREAT)
    and NEVER renamed or replaced, so all callers flock the same inode and
    actually serialize.

    Different decision ids resolve to different sidecars (via
    decision_path_for_id's sanitization), so there is no cross-id
    contention -- only two callers racing on the SAME id ever block each
    other.

    ORPHAN SIDECARS: taking this lock for a decision id that has never been
    (or never will be) written -- e.g. update_decision_state/set_manual_boost
    called against an unknown id -- still creates decisions_dir and an empty
    ``<id>.json.lock`` sidecar for that id, even though the subsequent read
    then fails and the caller gets None back. Lock files are, as in 1609,
    intentionally never deleted, so a burst of calls against unknown ids can
    leave harmless orphan sidecars behind at current volumes.

    Usage::

        with decision_id_lock(decision_id, root=root):
            record = DecisionRecord.from_json(path.read_text(encoding='utf-8'))
            record.some_field = new_value
            write_decision(record, root=root)
    """
    with exclusive_sidecar_lock(Path(str(decision_path_for_id(decision_id, root=root)) + '.lock')):
        yield


ESCALATION_ARCHIVE_SUBDIR = 'archive'
"""Mirrors escalation.archive.ARCHIVE_SUBDIR BY CONVENTION, not by import:
this module is deliberately stdlib-only, importing no orchestrator module
but fleet_files (see module docstring), so it cannot import escalation.archive directly.
Kept in sync by hand with that constant.

PUBLIC (task 3640 amendment) because it defines half of what "this queue
holds that escalation id" MEANS: read_escalation_status looks at the queue
root and then under this subdir, and scripts/backfill_decision_queue_stamp.py
must search exactly the same two tiers or it can stamp a record with a queue
the reaper would never match against -- pinning that record OPEN forever.
Out-of-module searchers reference this name rather than re-spelling 'archive',
so the two notions cannot drift."""


UNKNOWN_QUEUE = '<unknown>'
"""The THIRD queue state for DecisionRecord.escalations_dir (task 3640).

Distinct from BOTH a real queue path and the ``''`` unset/legacy sentinel:

- ``''``          -- "nobody told us". A record filed before the field
                     existed, or by a caller that omitted it. The reaper
                     falls back to project-only scoping and MAY close it.
- ``'<unknown>'`` -- "we investigated and could NOT determine the owning
                     queue". The reaper REFUSES to close it; it stays a
                     visible cockpit row for human closure.
- ``'/abs/path'`` -- a normalized queue path. Reaped only by that queue.

The two sentinels are deliberately NOT collapsed. 3528 defined ``''``'s
meaning and an existing test asserts a ``''`` record still closes; redefining
it would change behaviour for every future omitted-flag caller under the
human. Separating them lets task 3640 back-fill the undeterminable records
with an honest value instead of leaving them in the false-closable ``''``
population.

The angle brackets are load-bearing rather than decorative: a resolved queue
path always begins with ``'/'``, so this sentinel can never collide with a
real queue no matter what a project is named or where it is checked out."""


_DECISION_SEVERITY_RANK: dict[str, int] = {'': 0, 'info': 1, 'blocking': 2, 'critical': 3, 'urgent': 4}
"""Total order over DecisionRecord.severity, for the never-downgrade rule (task 3559).

Mirrors escalation.models.KNOWN_SEVERITIES (``frozenset({'info','blocking'})
| BORN_AT_L2_SEVERITIES``) and the cockpit's ordering comment in
cockpit/src/cockpit/priorities.default.yaml ("ordered
urgent>=critical>blocking>info") BY CONVENTION, not by import -- the same
hand-sync idiom ESCALATION_ARCHIVE_SUBDIR documents above. This module is
deliberately stdlib-only, importing no orchestrator module but fleet_files (see the module
docstring) so spawn-claude.sh can invoke it by absolute path with no
venv/PYTHONPATH. ``''`` (severity unset) is the floor.

Deliberately NOT escalation.queue._SEVERITY_RANK, even though that map
exists and looks reusable: it is ``{'info': 0, 'blocking': 1}`` only, so
'critical' and 'urgent' fall to its ``.get(x, 0)`` default and rank
INFO-tier. Copying it here would make _max_decision_severity perform the
exact downgrade it exists to prevent."""


def _max_decision_severity(existing: str, incoming: str) -> str:
    """Return whichever of two severities ranks higher; never downgrade.

    Used when a second watcher enriches an open DecisionRecord it did not
    file (see merge_decision_enrichment): the record must end up carrying
    the MOST urgent view any watcher has of the underlying human gate.

    An unrecognised value (a typo, or a case variant -- KNOWN_SEVERITIES is
    lowercase-only) resolves via ``.get(value, 0)`` to the same rank as the
    unset sentinel, so it can never displace a recognised severity, and this
    never raises: it sits on a watcher's filing path and must not crash the
    watch loop. Ties -- and the ambiguous both-unrecognised case -- return
    ``existing``, so an equal re-file never churns the stored value.
    """
    if _DECISION_SEVERITY_RANK.get(incoming, 0) > _DECISION_SEVERITY_RANK.get(existing, 0):
        return incoming
    return existing


def normalize_escalations_dir(value: str | Path) -> str:
    """The ONE canonical spelling of an escalation-queue path (task 3528).

    Both sides of the (fleet-global decisions <-> per-queue escalations) join
    run their queue path through this helper: the ``write-decision`` verb
    stamps the normalized form onto ``DecisionRecord.escalations_dir``, and
    the ``reap-decisions`` verb normalizes both its own ``--escalations-dir``
    and the decision's stored value before comparing them. Routing both
    through one function is what makes ``data/../data/escalations/``, a
    ``~``-relative spelling and a plain absolute path all compare EQUAL --
    a raw string compare would silently fail open and reintroduce the
    cross-queue false-closure bug this scoping exists to prevent.

    Returns ``''`` -- the "unset/legacy queue" sentinel, never a path -- for
    an empty or whitespace-only *value*, mirroring the sibling
    ``DecisionRecord.severity`` convention; the reaper reads that sentinel as
    "fall back to project-only scoping". Returns ``UNKNOWN_QUEUE`` VERBATIM
    for that sentinel (task 3640) -- the reaper reads it as "refuse to close".
    Otherwise returns ``str(Path(raw).expanduser().resolve())``.
    ``Path.resolve()`` is non-strict on Python 3.11+, so a well-formed queue
    path that does not exist yet still normalizes rather than faulting.

    Stdlib-only and fail-soft: never raises. A value the OS cannot resolve
    (an embedded NUL, an unreadable cwd) degrades to the stripped raw string,
    which simply matches no real queue -- the fail-OPEN direction, leaving a
    decision visible to the human rather than risking a false close. Keeps
    this module's import footprint (see module docstring).
    """
    raw = str(value).strip()
    if not raw:
        return ''
    # BEFORE the resolve(): UNKNOWN_QUEUE is a bare word, so Path.resolve()
    # would treat it as a path RELATIVE TO THE CALLING PROCESS'S CWD and
    # return '<cwd>/<unknown>' -- a different string in the back-fill script
    # than in a watcher's reaper. The stamp would stop being deterministic
    # across processes (the two sides of the join could never compare equal)
    # and the stored value would be an outright lie about where the record's
    # escalation lives. It is a sentinel, not a path: it round-trips verbatim.
    if raw == UNKNOWN_QUEUE:
        return UNKNOWN_QUEUE
    try:
        return str(Path(raw).expanduser().resolve())
    except (OSError, ValueError, RuntimeError):
        return raw


PROJECT_TOKEN_ALIASES_DECLINED: dict[str, tuple[str, str]] = {
    'solar_challenge': (
        'my_solar_challenge',
        'No split left to heal (fold already merged 3+2 into one bucket), and '
        'the identity question is an OPEN human gate in that project (esc-98-1, '
        '"Do NOT auto-act").',
    ),
}
"""Aliases CONSIDERED and DELIBERATELY DECLINED (task 3813), keyed
already-folded-alias -> (already-folded declined canonical, one-line reason).

WHAT THIS IS. The deliberate mirror image of
``shared.project_id.PROJECT_ID_ALIASES``: same folded-to-folded key/value
invariant under ``fold_project_token`` (pinned by a named test), so
PROMOTING a declined entry is a one-line move between the two dicts and
DECLINING a live one is the same move in reverse. It is read by
``declined_project_token_hint`` and by two guard tests; it is deliberately
NOT consulted by ``fold_project_token``, so it costs nothing on the
fold's hot path and every existing test of that fold is untouched. The
in-repo precedent for encoding a deliberate exclusion next to the table it
governs is ``fused-memory/scripts/consolidate_namespace_families.py``
::``GRAPH_FAMILY_ALIASES``; recording it as data rather than a comment is
what makes it checkable.

THE DECISION. ``solar_challenge -> my_solar_challenge`` is DECLINED. Not
deferred, not an oversight, not "pending a decision task" -- 3813 WAS that
decision task, and this is its answer.

THE EVIDENCE (re-measured 2026-09-07 over 748 fleet decision records:
``solar-challenge`` 3, ``solar_challenge`` 2, ``my_solar_challenge`` ZERO,
all 5 ``state=open`` -- identical to the 2026-08-07 measurement a month
earlier):

  - NO SPLIT REMAINS, so there is nothing for an alias to heal. Task 3807's
    case/separator fold already merged the two filed spellings into ONE
    bucket. Adding the alias would not MERGE anything; it would only RENAME
    a populated bucket (5 rows) onto an empty one (0 rows), moving those
    rows out from under whatever reaps them today. That is why the amended
    admission rule's clause 2 exists and why this entry fails it.

  - THE IDENTITY QUESTION IS AN OPEN HUMAN GATE IN THAT PROJECT, and the
    alias would settle it from dark-factory's side. Decision record
    ``esc-98-1`` (``state=open``, filed under ``solar_challenge``, queue
    ``<dark-factory>/data/reconciliation/escalations``) reads: "one-way
    policy decision -- (a) MIGRATE ~1400 orphaned 'my_solar_challenge'
    items (876 graphiti + 524 mem0) into canonical 'solar_challenge' ...
    or (b) ARCHIVE the 'my_solar_challenge' namespace in place. Do NOT
    auto-act. Forward project_id-misconfig fix tracked via a separate
    sibling task (finding 116ceed2)." That gate calls ``solar_challenge``
    the CANONICAL token, ``my_solar_challenge`` the ORPHANED namespace, and
    the config declaration itself a MISCONFIG whose forward fix is already
    tracked elsewhere. Aliasing onto ``my_solar_challenge`` here would
    resolve that gate silently, in the direction it calls "orphaned", with
    no human sign-off.

  - IN-REPO PRECEDENT for keep-separate on this exact family:
    ``consolidate_namespace_families.py::GRAPH_FAMILY_ALIASES`` excludes the
    solar family on the stated ground that keep-separate is the default
    absent an explicit human decision.

  - THE COLLAPSE GUARD IS UNAFFECTED either way:
    ``solar_challenge_platform`` is a distinct project root with a distinct
    declared ``project_id`` and folds to itself. Nothing here touches it.

THE OPERATOR CONSEQUENCE. Reap and file that project with a token that
folds to ``solar_challenge`` -- PERMANENTLY, not "until 3813 lands". The
mismatch with its declared ``memory.project_id`` is now a decided,
standing state, so ``write-decision`` and ``reap-decisions`` WARN when
handed ``my_solar_challenge`` (see ``declined_project_token_hint``): the
trap announces itself at the moment someone types the config-declared
token, instead of returning a silent zero-row no-op that reads as "nothing
to reap".

WHAT WOULD REOPEN IT. Either ``esc-98-1`` resolving in favour of
``my_solar_challenge`` (which would make it the canonical bucket and this
decline wrong), or solar-challenge's config being changed to declare
``solar_challenge`` (which would make the decline moot -- delete the entry
and its hint together)."""


def declined_project_token_hint(value: object, action: str = '') -> str | None:
    """One-line operator warning when *value* names a DECLINED alias target
    (task 3813). Returns None -- the overwhelmingly common case -- otherwise.

    WHY THIS EXISTS. Declining the solar-challenge alias makes that project's
    naming mismatch PERMANENT: an operator who trusts its config-declared
    ``memory.project_id`` gets a silent zero-row no-op forever, which reads
    exactly like "nothing to reap". Under this project's
    loud-over-silent-degradation norm, a decision that manufactures a
    standing silent trap is only defensible if the trap ANNOUNCES ITSELF. So
    the recorded decline gets a live caller instead of staying documentation.

    WHY IT WARNS RATHER THAN REWRITES. Rewriting the passed token to the
    bucket that actually holds the rows would BE the cross-project behaviour
    change task 3813 declined, smuggled in through the CLI boundary instead
    of the alias table. Callers file under, and scope to, EXACTLY the token
    they were given; only a log line is added. Non-blocking by construction,
    so a watcher's filing path can never break on it -- matching the
    fail-soft contract every helper this module hands a watch loop honours.

    WHY IT KEYS ON THE DECLINED VALUE, NOT THE KEY. The trap is typing the
    config-declared id (``my_solar_challenge``), so that is where the warning
    must land. ``solar_challenge`` -- the token both SKILL.md files recommend
    -- must stay SILENT, or a watcher accrues a warning every Main Loop
    cycle and the signal degrades into noise. Because the check is gated on
    a one-entry table it has zero false positives, so it fires on every
    reap. The general "your --project matched zero records" case is
    unmatched_project_token_hint, which is OPT-IN (``reap-decisions
    --expect-matches``) because by count alone a never-filed project cannot
    be told apart from a mismatched one; this hint takes precedence over it.

    WHY IT IS VERB-AWARE. The two callers hit this table for OPPOSITE
    reasons, and one message cannot be true for both. ``reap-decisions`` is
    MATCHING, so its consequence is a zero-row no-op. ``write-decision`` is
    CREATING, so it matches nothing by definition and a "matches no
    decisions" line would be false the moment it is acted on -- one line
    later the verb files a row under exactly that token. Its real
    consequence is also the WORSE of the two and would otherwise go
    unstated: the row lands in a bucket no documented reap scopes to (the
    skills tell watchers to reap ``solar_challenge``), so it can never
    auto-close, whereas a missed reap is merely repeatable. *action* selects
    that consequence clause: ``'reap'`` and ``'file'`` are the two known
    verbs.

    An OMITTED or unrecognised *action* is not an error and is not guessed
    at: it yields the verb-neutral core alone, which states only what the
    decline is and where the rows live. That is fail-soft in the direction
    that matters here -- a future caller that forgets the argument gets a
    message that is less specific but still TRUE, never one that confidently
    describes the wrong verb.

    Folds *value* through ``fold_project_token`` first, so case and
    separator variants of the config-declared token (``My-Solar-Challenge``)
    all hit, and a non-str or ``None`` *value* coerces fail-soft to a
    matchless token or ``''`` rather than raising. Returning None costs one
    scan of a one-entry dict. Stdlib-only, importing no orchestrator module but fleet_files
    (see module docstring).
    """
    folded = fold_project_token(value)
    if not folded:
        return None
    for alias, (declined_canonical, _reason) in PROJECT_TOKEN_ALIASES_DECLINED.items():
        if folded != declined_canonical:
            continue
        # Verb-neutral, and therefore true on EVERY caller's path: it states
        # only what was declined and where the rows live, never what this
        # caller is about to do with them.
        core = (
            f'--project {folded!r}: the alias {alias!r} -> {declined_canonical!r} '
            f'was considered and DECLINED (task 3813), so that project\'s '
            f'decisions live under {alias!r}, not {declined_canonical!r}.'
        )
        # Built only on a hit (rare by construction), so the cost of holding
        # both strings here is never paid on the common None path.
        consequence = {
            'reap': (
                f' This reap therefore matches ZERO of them, and its silent '
                f'no-op reads as "nothing to reap"; re-run scoped to a token '
                f'that folds to {alias!r}.'
            ),
            'file': (
                f' This record is being FILED under {declined_canonical!r}, '
                f'which no documented reap scopes to, so it can never '
                f'auto-close; re-file it under a token that folds to '
                f'{alias!r}.'
            ),
        }.get(action, '')
        return f'{core}{consequence} See PROJECT_TOKEN_ALIASES_DECLINED for the evidence.'
    return None


def project_token_census(decisions: Iterable[DecisionRecord]) -> Counter[str]:
    """How many *decisions*, in any state, carry each folded project token; an unset token is dropped."""
    return Counter(
        token for token in (fold_project_token(d.project) for d in decisions) if token
    )


def unmatched_project_token_hint(value: object, census: Mapping[str, int]) -> str | None:
    """One-line warning when *value* folds to a token no registry record carries, else None.

    *census* is project_token_census over the whole registry. The line names
    the tokens that DO exist, most common first, so a mismatched spelling is
    one glance from its fix.
    """
    folded = fold_project_token(value)
    if not folded or folded in census:
        return None
    present = ', '.join(
        f'{token}={count}'
        for token, count in sorted(census.items(), key=lambda item: (-item[1], item[0]))
    )
    return (
        f'--project {folded!r} matches ZERO of {sum(census.values())} decision records in the '
        f'registry (any state), so this reap can close nothing. Tokens present: '
        f'{present or "none"}. Re-run with the token whose rows you meant.'
    )


def read_escalation_status(escalations_dir: Path | str, escalation_id: str) -> str | None:
    """Best-effort read of *escalation_id*'s ``status`` field (Fleet Cockpit C8 reaper).

    Tries the queue-root file ``<escalations_dir>/<escalation_id>.json`` first
    (a still-pending escalation lives there); if absent, falls back to a
    recursive search under ``<escalations_dir>/archive/`` (a resolved/
    dismissed escalation is moved into a dated ``archive/YYYY-MM-DD/`` subdir
    by ``escalation.queue.EscalationQueue._archive_resolved``) and takes the
    newest match (sorted, last).

    Returns the raw ``status`` string, or None when: the id is unknown (no
    root file, no archive match); the body is missing/unreadable or fails to
    parse as JSON; or its ``status`` field is absent/not a string. Reads the
    on-disk contract directly via stdlib ``json`` rather than importing
    ``escalation.queue``/``escalation.models`` -- this module stays
    stdlib-only, importing nothing from escalation and nothing from the
    orchestrator but fleet_files (see module docstring). Fail-soft: never raises.

    COST NOTE: the archive fallback is a recursive rglob over the *entire*
    ``archive/`` tree, with no memoization across calls. When *escalation_id*
    never resolves there -- purged by archive retention pruning, or never
    existed -- every call still walks the full tree and returns None. A
    caller that re-invokes this once per Main Loop cycle (the
    ``reap-decisions`` CLI verb, see ``reap_answered_decisions``) therefore
    repeats that full scan indefinitely for a decision pinned to such an id.
    This is a known, currently-unbounded cost rather than something this
    function bounds itself (e.g. via a persistent negative-lookup cache) --
    doing so would need to answer staleness/TTL questions (a
    once-dangling id could later appear) that are out of scope for this
    read helper. The affected decision is simply left OPEN forever in that
    case and needs explicit human closure, same as the no-``escalation_id``
    case documented on ``reap_answered_decisions``.
    """
    base = Path(escalations_dir)
    candidate = base / f'{escalation_id}.json'
    if not candidate.is_file():
        matches = sorted((base / ESCALATION_ARCHIVE_SUBDIR).rglob(f'{escalation_id}.json'))
        candidate = matches[-1] if matches else None
    if candidate is None:
        return None
    try:
        data = json.loads(candidate.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    status = data.get('status') if isinstance(data, dict) else None
    return status if isinstance(status, str) else None


DECISION_CLOSE_MAP: dict[str, DecisionState] = {
    'resolved': DecisionState.ANSWERED,
    'dismissed': DecisionState.DROPPED,
}
"""Maps a terminal escalation status to the DecisionState an OPEN decision
resolving it should close to (Fleet Cockpit C8 reaper). Any other status
(including 'pending') or None is absent from this map and leaves the
decision OPEN -- see reap_answered_decisions."""


@dataclass(frozen=True)
class ReapedDecision:
    """One DecisionRecord closed by reap_answered_decisions.

    id: the closed decision's id.
    escalation_id: the escalation whose terminal status triggered the close.
    new_state: the DecisionState it was closed to, as its wire string (see
        DECISION_CLOSE_MAP) -- e.g. 'answered' or 'dropped'.
    """

    id: str
    escalation_id: str
    new_state: str


def reap_answered_decisions(
    root: Path | str | None = None,
    *,
    escalation_status: Callable[[DecisionRecord], str | None],
) -> list[ReapedDecision]:
    """Close every OPEN decision whose escalation has reached a terminal status.

    For each ``list_decisions(root)`` entry: a decision that is not
    ``DecisionState.OPEN`` is skipped outright (already resolved -- no
    re-close); a decision with no ``escalation_id`` is skipped WITHOUT ever
    consulting *escalation_status* (there is nothing to resolve it against).
    Otherwise *escalation_status* is called with the decision, and the
    result is looked up in ``DECISION_CLOSE_MAP`` -- any status not in that
    map (including ``'pending'`` or ``None``) leaves the decision OPEN.

    A close-worthy result is applied via the existing
    ``update_decision_state`` (itself fail-soft: a write fault there returns
    None and is silently skipped here, matching its contract for direct
    C8/cockpit callers) and is only recorded in the returned list when that
    update actually succeeds.

    *escalation_status* is injected rather than read directly here so this
    stdlib-only, import-free core stays unit-testable with a fake callback
    and never needs to know an escalations-dir path or project scoping --
    see the ``reap-decisions`` CLI verb, which builds the production
    closure.

    LIMITATION: a decision whose ``escalation_id`` is well-formed but never
    resolves to a close-worthy status via *escalation_status* (unknown id,
    or one purged by archive retention pruning) is left OPEN indefinitely --
    the same fail-open-on-absent-evidence outcome as the no-``escalation_id``
    case above -- and needs explicit human closure. With the production
    closure (``read_escalation_status``), this also means a full archive
    scan repeats every call for that decision; see its COST NOTE.
    """
    reaped: list[ReapedDecision] = []
    for decision in list_decisions(root):
        if decision.state != DecisionState.OPEN:
            continue
        escalation_id = decision.escalation_id
        if not escalation_id:
            continue
        status = escalation_status(decision)
        new_state = DECISION_CLOSE_MAP.get(status) if status is not None else None
        if new_state is None:
            continue
        updated = update_decision_state(decision.id, new_state, root=root)
        if updated is None:
            continue
        reaped.append(
            ReapedDecision(id=decision.id, escalation_id=escalation_id, new_state=str(new_state))
        )
    return reaped


@dataclass(frozen=True)
class MigratedDecision:
    """One DecisionRecord whose project token was canonicalized (task 3807).

    id: the migrated decision's id -- NEVER rewritten by the migration, so
        this is also its filename stem before and after.
    old_project: the non-canonical token it was filed under.
    new_project: the canonical token it now carries (see
        fold_project_token).
    """

    id: str
    old_project: str
    new_project: str


def migrate_decision_project_tokens(
    root: Path | str | None = None,
    *,
    dry_run: bool = False,
) -> list[MigratedDecision]:
    """One-shot, idempotent backfill of DecisionRecord.project onto the canonical token.

    Repairs the population filed BEFORE ``write-decision`` canonicalized
    ``--project`` (task 3807). For each ``list_decisions(root)`` entry whose
    stored token differs from ``fold_project_token`` of itself, rewrites
    ONLY that field. Returns one MigratedDecision per record actually
    changed; an already-canonical record and an unrelated project's record
    are skipped and absent from the result.

    Load-bearing invariants, in the order they matter:

    (a) SAME LOCK AS EVERY OTHER MUTATOR. The read-modify-write runs under
        ``decision_id_lock``, exactly as ``update_decision_state`` and
        ``set_manual_boost`` do, so a concurrent cockpit ``set_manual_boost``
        or watcher state-update on the SAME id is serialized rather than
        having one side's mutation dropped.

    (b) RE-READ INSIDE THE LOCK, AND RECOMPUTE FROM THE RE-READ. The scan is
        a point-in-time snapshot; a state transition landing between the scan
        and the write would be ROLLED BACK by writing that stale snapshot, so
        re-reading under the lock is what stops the sweep resurrecting a
        decision the human just answered. That argument only covers the
        carried-through fields, so the ONE field this sweep writes is
        recomputed from the re-read record too: both "does this still need
        migrating?" and the new token come from what is actually on disk
        under the lock, never from the snapshot. Otherwise a writer that
        changed ``.project`` in that window would be clobbered by a value
        derived from the old spelling, and the reported ``old_project`` would
        name a value that was never on disk at write time -- an audit trail
        that quietly lies. A record another writer already canonicalized in
        the window is simply skipped. (The snapshot IS still used as a cheap
        pre-filter, to avoid locking the already-canonical majority; a
        pre-filter that lets through a record needing no work costs one
        wasted lock, never a wrong write.)

    (c) ONLY ``.project`` IS TOUCHED. ``state`` and ``filed_at`` (and every
        other field) are carried through from the re-read record untouched,
        so an already-ANSWERED/DROPPED row can never be reopened and the
        cockpit's age/ordering signal is not restamped. (a)+(b)+(c) make this
        structurally identical to ``update_decision_state``, so the guarantee
        is inherited from a shape already pinned by TestDecisionIdLock rather
        than re-argued here.

    (d) IDEMPOTENT. Because ``fold_project_token`` is idempotent, a
        second run finds nothing to change and returns ``[]`` without
        rewriting a single file. That is what makes this safe to keep
        permanently as the repair tool for a hand-edited record, rather than
        a one-shot script that must be deleted after use.

    (e) THE ID IS NEVER REWRITTEN. Only the ``project`` FIELD moves; the
        record keeps its id and therefore its ``<id>.json`` filename, so the
        ``df-`` prefix on ids like ``df-esc-3524-1`` and every cockpit
        cross-link survive intact.

    (f) ``dry_run=True`` PREVIEWS, guaranteed side-effect-free. It returns
        the MigratedDecision list the real run would produce over a quiescent
        tree, reporting off the scan snapshot because taking no lock is the
        whole point -- so a record mutated between the preview and the real
        run is reported as the preview saw it, which is what "preview" means.
        It takes no lock and performs no write, so an operator can see the
        full rewrite of the live population before committing to it. It does
        not even materialize a ``<id>.json.lock`` sidecar, which matters
        because ``decision_id_lock`` documents that merely TAKING a lock
        creates one (its ORPHAN SIDECARS note); a preview that littered one
        per record across the whole fleet would not be side-effect-free.

    Iterates ``list_decisions`` rather than doing its own scan, inheriting
    its skip-a-corrupt-file tolerance for free. FAIL-SOFT PER RECORD: the
    lock/read/write span is guarded, so a record that is unreadable,
    unwritable, or faults mid-write is logged at ERROR and skipped while the
    sweep of the rest completes -- one bad record must never deny the
    operator the repair of the other 390, matching the module-wide
    convention (``write_decision``, ``update_decision_state`` and
    ``list_decisions`` all self-guard). A record whose write fails is simply
    absent from the returned list, mirroring ``reap_answered_decisions``'
    "only record it when the update actually succeeded" contract.
    """
    migrated: list[MigratedDecision] = []
    for decision in list_decisions(root):
        # Cheap pre-filter off the scan snapshot -- avoids taking a lock for
        # the (usually large) already-canonical majority. NOT the decision of
        # record: for a real run that is re-made under the lock below.
        if fold_project_token(decision.project) == decision.project:
            continue
        if dry_run:
            migrated.append(
                MigratedDecision(
                    id=decision.id,
                    old_project=decision.project,
                    new_project=fold_project_token(decision.project),
                )
            )
            continue
        try:
            path = decision_path_for_id(decision.id, root=root)
            with decision_id_lock(decision.id, root=root):
                # Re-read under the lock -- see invariant (b). Both the
                # decision to migrate AND the value written are recomputed
                # from THIS record, never from the snapshot: a writer that
                # changed .project between the scan and the lock must not
                # have its value clobbered by one derived from the old
                # spelling, and old_project must name what was actually on
                # disk at write time or the audit trail lies.
                record = DecisionRecord.from_json(path.read_text())
                old_project = record.project
                new_project = fold_project_token(old_project)
                if new_project == old_project:
                    # A concurrent writer already canonicalized it.
                    continue
                record.project = new_project
                if not write_decision(record, root=root):
                    continue
                migrated.append(
                    MigratedDecision(
                        id=decision.id,
                        old_project=old_project,
                        new_project=new_project,
                    )
                )
        except Exception:
            logger.error(
                'migrate_decision_project_tokens: skipping %s',
                decision.id,
                exc_info=True,
            )
            continue
    return migrated
