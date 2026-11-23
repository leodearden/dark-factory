"""orchestrator.session_leases — role leases: one owner per role (Attention Rail T7).

A lease is a ``<fleet_root>/leases/<name>.lease`` file whose body names its
holder and whose mtime is the holder's heartbeat. A contender claims, beats,
releases or reaps it; a lease whose holder pid is dead past
``LEASE_HEARTBEAT_TTL`` is reclaimable. The ``lease-*`` verbs of
``session_registry.py`` drive it.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves
(``shared.proc_liveness``) and its siblings ``fleet_files``,
``session_identity`` and ``session_records``. It never imports
``session_registry``: in a bare-shell (tier-1) run session_registry is
``__main__``, so an upward import would load a second copy of it. The
bare-shell contract is pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Reach this checkout's stdlib-only shared leaves from a bare shell; rationale in
# test_session_registry.py::TestStdlibOnlySelfContainment.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.proc_liveness import pid_alive

# Siblings are imported relative to orchestrator.session_leases, or else from this
# file's own directory: loaded by path, as a tier-1 sibling or under a foreign dotless
# name, that directory is on sys.path only if Python or the importer put it there, so it
# is borrowed for these imports alone. Each mode is pinned in
# test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import session_identity, session_records
    from .fleet_files import fleet_root
else:
    _OWN_DIR = str(Path(__file__).resolve().parent)
    _OWN_DIR_BORROWED = _OWN_DIR not in sys.path
    if _OWN_DIR_BORROWED:
        sys.path.append(_OWN_DIR)
    try:
        import session_identity
        import session_records
        from fleet_files import fleet_root
    finally:
        if _OWN_DIR_BORROWED:
            sys.path.remove(_OWN_DIR)

logger = logging.getLogger(__name__)


LEASE_NAME_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._#-]')
"""Unlike session_records.sanitize_slug, this PRESERVES '#': task-scoped lease names
(build_lease_name('unblock', project, task_id)) encode their task with a
literal '#', which must survive into the .lease filename. Everything else
outside [A-Za-z0-9._#-] (notably '/') maps to '-', so a malformed name can
never escape leases_dir via a path separator."""


def leases_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'leases'


def lease_path_for_name(name: str, root: Path | str | None = None) -> Path:
    """Resolve *name* (see build_lease_name) to its ``<leases_dir>/<name>.lease`` path.

    *name* is sanitized through LEASE_NAME_SANITIZE_RE first, so it always
    resolves to a single file directly inside leases_dir.
    """
    safe_name = LEASE_NAME_SANITIZE_RE.sub('-', name)
    return leases_dir(root) / f'{safe_name}.lease'


def build_lease_name(role: str, project: str, task_id: str | None = None) -> str:
    """Build the canonical lease name for *role* (PRD T7): ``<role>-<project>[#<task_id>]``.

    task_id is optional; when supplied, the lease is scoped to a single task
    (e.g. ``unblock-df#2085``), so two concurrent /unblock sessions on
    DIFFERENT tasks in the same project never contend the same lease.
    """
    name = f'{role}-{project}'
    if task_id:
        name = f'{name}#{task_id}'
    return name


def resolve_own_record_slug(
    env: Mapping[str, str] | None = None, *, root: Path | str | None = None
) -> str | None:
    """Return THIS claude session's registry record slug, via ``$CLAUDE_PID``'s pointer.

    A lease slug (``default_lease_slug``: ``<lease-name>-$CLAUDE_PID``) and a
    registry record slug (``session_records.build_session_slug``) are different namespaces,
    so neither can be looked up as the other. The pid pointer is the join:
    the session hooks stamp ``sessions-by-pid/<claude pid>`` with the record
    key, and ``session_records.resolve_session_slug_for_pid`` reads it back only while that
    record still names the pid as its owner. Its consumers are
    ``LeaseHolder.record_slug`` and ``decision_registry.py::DecisionRecord.record_slug``.

    None when ``CLAUDE_PID`` is unusable or no vouched-for pointer exists.
    A shared ``claude agents`` host has no session-scoped pid, so the result
    is None there too.
    Quiet by contract: this serves enrichment, so an unresolvable slug is a
    blank field in the caller's result, not a log line.
    """
    pid = session_identity.resolve_session_identity(env).session_scoped_pid
    return None if pid is None else session_records.resolve_session_slug_for_pid(pid, root=root)


def default_lease_slug(
    name: str,
    env: Mapping[str, str] | None = None,
    *,
    identity: session_identity.SessionIdentity | None = None,
) -> str | None:
    """Derive THIS session's lease slug for lease *name*: ``<name>-<session pid>``.

    Under a shared ``claude agents`` host it is
    ``<name>-<session pid>-<CLAUDE_CODE_SESSION_ID>``: every hosted session
    shares that pid, so the pid alone would hand them all one identity.

    WHY THIS EXISTS (task 4248). This is ``session_identity.resolve_session_pid``'s own argument,
    extended from the pid to the slug. 3994 moved the PID into code because "it
    depended on every skill doc getting one shell token right" -- and then, in
    the same change, made the SLUG load-bearing: ``--slug`` became required on
    ``lease-heartbeat``/``lease-release`` and a mismatch is REFUSED. So the
    token that decides whether every heartbeat and the final release ACT or are
    refused was still being assembled in shell, in three SKILL.md files, as
    ``<role>-<project>-${CLAUDE_PID:-$PPID}`` -- and ``$PPID`` is measurably NOT
    stable across Claude Code tool calls (1430433 then 1471645, the first
    already dead), so a session could drift into a slug its own later heartbeat
    would fail to match. Deriving it HERE makes the correct token structural.

    It must be RE-DERIVABLE rather than carried: each Claude Code Bash tool call
    is a fresh ``/bin/bash -c``, so a ``SLUG=$(...)`` captured during the claim
    is gone by the time the heartbeat runs -- the same property that makes ``$$``
    an unusable pid. Hence a pure function of (name, env), evaluated afresh on
    every verb, rather than a value threaded through the skills.

    RETURNS None WHEN THE SESSION PID IS UNRESOLVABLE (or, under a shared host,
    when the session id is) -- deliberately, and specifically NOT
    ``f'{name}-0'`` nor ``f'{name}-<host pid>'``, which every hosted session
    would share. ``session_identity.resolve_session_pid`` may degrade to 0
    because a pid is a LIVENESS probe and 0 is provably dead, which keeps a
    crashed holder's lease reapable. A slug is an IDENTITY, and ``{name}-0`` is
    not unique: two concurrently-degraded sessions contending the same lease
    name would compute an IDENTICAL slug, so each would pass the other's
    ``_is_lease_owner`` check and could heartbeat or release the other's lease
    -- exactly 3994 defect 1 ("any caller may evict/refresh any lease") reopened
    on the degraded path. Refusing to synthesize a token we cannot make unique
    is the only answer that keeps the ownership guard honest. The CLI turns the
    None into a loud ``parser.error`` (exit 2) naming the unusable variable and
    the ``--slug`` override -- the same exit-2 class a slug-less ``lease-claim``
    already produced pre-4248, so only the DERIVABLE case gains new behaviour.

    SILENT ON THE DEGRADED PATH, on purpose: the CLI fails loudly on the None,
    and ``session_identity.lease_liveness_pid`` reports a degraded liveness pid where one is
    written. The *name* is passed through verbatim -- a lease slug is never a
    filesystem path (``lease_path_for_name`` sanitizes only the NAME, and
    ``session_records.sanitize_slug`` applies only to session RECORD slugs), so the ``#`` in a
    task-scoped ``unblock-df#2085`` needs no escaping; see LeaseHolder.session_slug.

    *identity* ACCEPTS AN ALREADY-RESOLVED session_identity.SessionIdentity, so ``main()`` derives
    the lease body's liveness pid and this slug from ONE resolution; *env* is
    then ignored. Passing None resolves from *env* (defaulting to ``os.environ``).
    """
    return (
        identity if identity is not None else session_identity.resolve_session_identity(env)
    ).lease_slug(name)


LEASE_HEARTBEAT_TTL = timedelta(hours=2)
"""How long a lease survives with a dead holder pid and no fresh heartbeat
(mtime touch) before it is stale-reapable. Mirrors NON_TERMINAL_HEARTBEAT_TTL's
role for session records: staleness requires BOTH an aged heartbeat AND a
holder pid that cannot vouch for the lease -- see _lease_is_stale.

VALUE DERIVATION (task 2796 THREAD 1, RE-DERIVED by task 3994). The value is
unchanged at 7200s; its BASIS is not, because task 2796's basis is now false
and a false derivation is exactly what a later rotation reads and
mis-diagnoses itself with.

WITHDRAWN BASIS (2796): "watcher leases are effectively HEARTBEAT-ONLY, not
pid-guarded" -- true only because both watcher SKILLs claimed with
``--pid $$``, the ephemeral Bash-tool wrapper pid, so ``pid_alive`` was
~always False and the staleness AND-guard collapsed to a pure heartbeat-TTL
check. Task 3994's ``session_identity.resolve_session_pid`` restores a real, long-lived
session pid, so that premise no longer holds.

CURRENT BASIS (3994): with the pid guard actually live, a LIVE holder is
never reaped regardless of heartbeat age (a shared ``claude agents`` host pid
excepted, see _lease_is_stale), so this TTL now bounds only ONE
thing -- crash recovery for a genuinely-dead holder. It nonetheless stays at
2h rather than dropping back toward the old 5min, for two reasons:
(a) session_identity.resolve_session_pid still degrades -- to pid 0, which `pid_alive`
reports dead by construction -- whenever CLAUDE_PID is unresolvable on some
launch path or names a shared ``claude agents`` host, and that degradation is
logged but not fatal. On that path the
AND-guard collapses to its heartbeat half exactly as it did pre-3994, so
heartbeat-only behaviour must remain SURVIVABLE:
7200s = 2x the canonical 3600s watcher-rearm.sh ``--timeout`` slice, giving
a full quiet slice plus a full slice of margin, so a live-but-quiet holder
is never reaped-and-reclaimed by a duplicate (the duplicate-spawn incident);
(b) a crashed holder no longer COSTS 2h of silence -- ``lease-claim`` now
prints ``holder_liveness=orphaned`` as soon as the holder's pid is not
running (a SINGLE-signal pid probe; see _run_lease_claim), so the contender
surfaces the crash on the very next claim instead of waiting out the TTL.
Lowering it would
re-introduce live-holder-eviction risk to buy recovery latency that the
orphan signal already provides.

TRADE-OFF: a genuinely-crashed holder's lease still SURVIVES up to 2h before
it is stale-reapable. That is loud and recoverable, not silent: a contender
stands down with an explicit "pid <n> is not running, but its heartbeat is
FRESH ... NOT reclaimable for another <r>s" message plus the orphan signal,
an operator can reclaim immediately via ``lease-release --force``, and any
reap-and-reclaim of a supposedly-live holder emits a structured WARNING
(see claim_lease)."""


class LeasePolicy(StrEnum):
    """How claim_lease should respond when the lease is already held live.

    STAND_DOWN: the caller (e.g. a duplicate escalation-watcher) must exit.
    WARN_AND_PROCEED: the caller (e.g. a second /unblock on the same task)
        surfaces the current holder to the user but continues.
    """

    STAND_DOWN = 'stand-down'
    WARN_AND_PROCEED = 'warn-and-proceed'


class LeaseDecision(StrEnum):
    """The outcome of a claim_lease call."""

    ACQUIRED = 'acquired'
    STAND_DOWN = 'stand-down'
    PROCEED = 'proceed'


class LeaseMutation(StrEnum):
    """The outcome of a heartbeat_lease / release_lease call (task 3994).

    A lease is single-owner-per-role, so a MUTATION of one (refreshing its
    heartbeat, or removing it) is only legitimate from its actual holder.
    Before 3994 both verbs took only ``name`` and mutated unconditionally:
    any caller could evict a live holder or keep a stranger's lease
    structurally unreapable forever. These values report which of those an
    attempted mutation turned out to be.

    APPLIED: the caller IS the holder; the mutation was performed.
    FORCED: the caller is NOT the holder (or the body was unreadable) but
        passed ``force=True``; the mutation was performed anyway and logged
        at WARNING naming BOTH parties (operator recovery, attributable).
    ABSENT: there is no such lease. A no-op, and NOT a failure -- releasing
        an already-released lease is idempotent by design, so this is quiet.
    REFUSED: the caller is not the holder (or the body is unreadable, which
        fails TOWARD held); nothing was touched, logged at ERROR naming both
        parties.
    FAULTED: the mutation itself faulted (an OSError from utime/unlink).
        Logged at ERROR and returned, never raised -- a lease-substrate
        fault must not interrupt a watcher's main loop.
    """

    APPLIED = 'applied'
    FORCED = 'forced'
    ABSENT = 'absent'
    REFUSED = 'refused'
    FAULTED = 'faulted'


class HolderRecordState(StrEnum):
    """What a lease holder's ``record_slug`` says about its registry record.

    An axis INDEPENDENT of pid liveness (``holder_alive``): neither
    overrides the other, and reading them together is the caller's job.

    UNLINKED: the holder carries no ``record_slug`` (a body that predates the
        field, or a claimant whose pointer was unresolvable), so nothing was
        looked up. No evidence either way -- in particular NOT evidence that
        the holder is alive or gone.
    ABSENT: a ``record_slug`` WAS carried and no record exists under it: the
        record has been reaped. Evidence the holder's session is gone.
    UNREADABLE: the ``record_slug`` is not a record key, or its record
        exists but cannot be read or parsed.
    ACTIVE: the record's status is not terminal. For a hand-launched holder
        that tracks its terminal, not the claude process, so ACTIVE never
        overrides a dead pid.
    EXITED: the record's status is terminal (``session_records.TERMINAL_STATUSES``).
    """

    UNLINKED = 'unlinked'
    ABSENT = 'absent'
    UNREADABLE = 'unreadable'
    ACTIVE = 'active'
    EXITED = 'exited'


@dataclass(frozen=True)
class LeaseHolder:
    """Serialized identity of a lease's current holder -- the exact ``.lease`` file body.

    Every field is type-checked at construction, and from_dict also rejects
    a body that is not a JSON object, so a mistyped body is unreadable
    rather than a holder.

    session_slug: a CLAIMANT-CHOSEN ownership token, NOT a session-registry
        record key. The skills prescribe ``<lease-role>-<project>-<pid>``
        (escalation-watcher's and recon-escalation-watcher's "Claiming the
        ... Lease" sections, and unblock's "Claim the unblock lease"). Do NOT
        pass it to ``session_records.read_record``; the holder's record is reached through
        ``record_slug`` below.

        THIS DOCSTRING USED TO CLAIM THE OPPOSITE -- that it was "the
        holder's own session-registry slug (see session_records.build_session_slug), letting
        a contended claim point back at the current owner's
        session_registry record". It cannot be, and task 3994 built (then
        withdrew) a whole orphan-corroboration path on that false premise.
        The two namespaces differ in all three segments: a record slug is
        ``<role>-<project>[-<task>]-<uniqueness>`` where the uniqueness
        token is a session UUID for hand-launched sessions
        (``session_hooks.hook_session_slug``) rather than a pid, and the
        project token is the registry's project id (``dark_factory``) not
        the lease's short name (``df``). Measured on the live fleet root:
        ``watcher-df.lease`` holds ``session_slug=watcher-df-1894895``,
        while 0 of 46,624 session record dirs begin with ``watcher-``. So
        ``session_records.read_record(session_slug)`` ALWAYS raises FileNotFoundError.
        Pinned by tests/test_session_registry.py's
        TestLeaseSlugIsNotASessionRecordKey.

        Its ONLY two roles are (i) the ownership comparison in
        ``_check_lease_ownership`` and (ii) naming the holder in a
        human-readable contention/refusal message.
    pid: the holder process's pid; liveness is checked via pid_alive. Any int, bool
        excluded (0 is session_identity.resolve_session_pid's degraded provably-dead value).
    start_ts: ISO-8601 timestamp of when this holder claimed the lease.
    record_slug: the holder's session-registry record key, resolved at claim
        time from *pid*'s pointer (``session_records.resolve_session_slug_for_pid``). '' when
        it could not be resolved or the body predates the field -- an
        absence of evidence, never evidence of absence.
    """

    session_slug: str
    pid: int
    start_ts: str
    record_slug: str = ''

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise TypeError(f'LeaseHolder.pid must be an int (not a bool), got {self.pid!r}')
        for name in ('session_slug', 'start_ts', 'record_slug'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f'LeaseHolder.{name} must be a str, got {value!r}')

    def to_dict(self) -> dict[str, Any]:
        return {
            'session_slug': self.session_slug,
            'pid': self.pid,
            'start_ts': self.start_ts,
            'record_slug': self.record_slug,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaseHolder:
        if not isinstance(data, dict):
            raise TypeError(f'LeaseHolder body must be a JSON object, got {data!r}')
        record_slug = data.get('record_slug')
        return cls(
            session_slug=data['session_slug'],
            pid=data['pid'],
            start_ts=data['start_ts'],
            record_slug='' if record_slug is None else record_slug,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> LeaseHolder:
        return cls.from_dict(json.loads(raw))


def holder_record_state(
    holder: LeaseHolder | None, *, root: Path | str | None = None
) -> HolderRecordState:
    """Classify *holder*'s registry record -- see ``HolderRecordState``. Never raises."""
    if holder is None or not holder.record_slug:
        return HolderRecordState.UNLINKED
    if not session_records.is_record_key(holder.record_slug):
        return HolderRecordState.UNREADABLE
    try:
        record = session_records.read_record(holder.record_slug, root=root)
    except FileNotFoundError:
        return HolderRecordState.ABSENT
    except (OSError, session_records.CorruptSessionRecord):
        return HolderRecordState.UNREADABLE
    if record.status in session_records.TERMINAL_STATUSES:
        return HolderRecordState.EXITED
    return HolderRecordState.ACTIVE


@dataclass(frozen=True)
class LeaseClaim:
    """The outcome of a claim_lease call.

    name: the lease name claimed (see build_lease_name).
    decision: ACQUIRED / STAND_DOWN / PROCEED.
    acquired: True iff this call became the lease's holder (decision is
        ACQUIRED); False if the lease was already held by someone else.
    holder: the lease's holder after this call -- the caller's own *holder*
        argument when acquired, otherwise the pre-existing holder found on
        disk (or None if that body was missing/unreadable).
    holder_alive: whether `holder`'s pid is currently alive (pid_alive).
    heartbeat_age_secs: seconds since the lease file's mtime; 0.0 for a
        freshly-acquired lease.
    message: fully-formatted, user-observable line -- callers print this
        verbatim rather than re-deriving it from the other fields.
    holder_record_state: `holder`'s registry record, via its ``record_slug``
        (see HolderRecordState) -- a second axis beside ``holder_alive``,
        which alone still drives ``decision``.

    Task 3994 withdrew an earlier ``holder_session_state`` that read
    ``session_records.read_record(holder.session_slug)``: a lease slug is not a record key,
    so that field was a structural constant. This one reads
    ``holder.record_slug``, a real record key, and spells the no-evidence
    case UNLINKED so it cannot be mistaken for a finding.
    """

    name: str
    decision: LeaseDecision
    acquired: bool
    holder: LeaseHolder | None
    holder_alive: bool
    heartbeat_age_secs: float
    message: str
    holder_record_state: HolderRecordState


def _render_contention_message(
    holder: LeaseHolder | None,
    *,
    holder_alive: bool,
    age_secs: float,
    policy: LeasePolicy,
    ttl: timedelta = LEASE_HEARTBEAT_TTL,
) -> str:
    """Build the exact user-observable contention line callers print verbatim.

    Centralizing this here (rather than in each skill) guarantees the signal
    string is identical everywhere it appears and is unit-testable with an
    injected clock. *holder* may be None (an unreadable/corrupt lease body);
    that is rendered as an explicit placeholder rather than raising.

    WHY THE OLD FORM WAS WITHDRAWN (task 3994, defect 3). This used to render
    ``lease held by {slug} ({alive|dead}, heartbeat {n}s ago)``, collapsing
    two INDEPENDENT axes -- whether the holder's PID is running, and how
    fresh its HEARTBEAT is -- into one parenthetical. On the branch that
    matters most that reads as a flat contradiction: "dead, heartbeat 42s
    ago" invites the reader to conclude the holder is gone and the lease is
    therefore reclaimable, when a fresh heartbeat means precisely the
    opposite (staleness requires BOTH a dead pid AND an aged heartbeat, see
    LEASE_HEARTBEAT_TTL). That exact string misled the 2026-08-08 rotation
    into force-releasing a live holder's lease.

    So each axis now names itself and the DECISION explains itself:
    - live holder -> ``(pid {n} alive, heartbeat {n}s ago)``;
    - dead pid, heartbeat within *ttl* -> states the pid is not running AND
      that the heartbeat is fresh AND that the lease is still held and NOT
      reclaimable, with the remaining TTL in seconds, so a contender cannot
      read "dead" and conclude "mine";
    - dead pid past *ttl* -> phrased as reclaim-eligible with the reclaim
      race lost (we are only here because someone else won it), never as a
      healthy holder;
    - unreadable body -> says so, rather than asserting a fake holder.

    *ttl* is a parameter rather than a module-global read so the
    remaining-TTL arithmetic is pinnable in a unit test. Always exactly ONE
    line, so the signal stays greppable in a transcript.
    """
    action = 'standing down' if policy is LeasePolicy.STAND_DOWN else 'proceeding anyway'
    if holder is None:
        return (
            f'lease held by <unknown> (lease body unreadable, '
            f'heartbeat {int(age_secs)}s ago) — {action}'
        )
    if holder_alive:
        state = f'pid {holder.pid} alive, heartbeat {int(age_secs)}s ago'
    elif age_secs <= ttl.total_seconds():
        remaining = int(ttl.total_seconds() - age_secs)
        state = (
            f'pid {holder.pid} is not running, but its heartbeat is FRESH — '
            f'{int(age_secs)}s ago; the lease is still held and is NOT reclaimable '
            f'for another {remaining}s'
        )
    else:
        state = (
            f'pid {holder.pid} is not running and its heartbeat is STALE — '
            f'{int(age_secs)}s ago, past the {int(ttl.total_seconds())}s TTL; the lease was '
            f'reclaim-eligible and someone else won the reclaim race'
        )
    return f'lease held by {holder.session_slug} ({state}) — {action}'


def _create_and_write_lease(path: Path, holder: LeaseHolder) -> None:
    """``os.open(O_CREAT|O_EXCL)`` + write *holder*'s body (mirrors ArtifactStore.lock_plan).

    Raises ``FileExistsError`` if the lease already exists (lost the race).
    On a write failure, cleans up the empty file it just created so a
    subsequent claim is not falsely blocked by a poison-pill empty file.
    """
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, holder.to_json().encode())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def _acquired_claim(name: str, holder: LeaseHolder, root: Path | str | None) -> LeaseClaim:
    return LeaseClaim(
        name=name,
        decision=LeaseDecision.ACQUIRED,
        acquired=True,
        holder=holder,
        holder_alive=True,
        heartbeat_age_secs=0.0,
        message=f'lease {name} acquired by {holder.session_slug}',
        holder_record_state=holder_record_state(holder, root=root),
    )


def _read_lease_holder_state(
    path: Path, *, now: datetime
) -> tuple[LeaseHolder | None, bool, float]:
    """Read an existing lease file's ``(holder, holder_alive, heartbeat_age_secs)``.

    *holder* is None when the body is missing/corrupt/unparseable -- callers
    treat that as "held by an unknown holder" (fail toward held, not free)
    rather than raising; *holder_alive* is then False, so a corrupt body
    older than LEASE_HEARTBEAT_TTL is still stale-reapable.

    If *path* itself vanishes before it can be stat'd (e.g. a concurrent
    release/reap in the narrow window between our failed O_EXCL create and
    this read), reports a synthetic age just past LEASE_HEARTBEAT_TTL
    instead of raising -- mirrors the guarded stat() in reap_stale_leases,
    and makes the caller's existing is_stale check reclaim it exactly as it
    would a dead-and-expired holder rather than propagating an uncaught
    FileNotFoundError.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, False, LEASE_HEARTBEAT_TTL.total_seconds() + 1.0
    age_secs = (now - datetime.fromtimestamp(mtime, tz=UTC)).total_seconds()
    try:
        holder = LeaseHolder.from_json(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False, age_secs
    return holder, pid_alive(holder.pid), age_secs


def _lease_is_stale(holder: LeaseHolder | None, *, holder_alive: bool, age_secs: float) -> bool:
    """Whether a lease is reapable: heartbeat past LEASE_HEARTBEAT_TTL and no pid vouching for it.

    A running shared-host pid cannot vouch for its session (see session_identity.is_shared_session_host).
    """
    if age_secs <= LEASE_HEARTBEAT_TTL.total_seconds():
        return False
    return holder is None or not holder_alive or session_identity.is_shared_session_host(holder.pid)


def claim_lease(
    name: str,
    *,
    holder: LeaseHolder,
    policy: LeasePolicy = LeasePolicy.STAND_DOWN,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> LeaseClaim:
    """Atomically claim the *name* lease for *holder* (PRD T7 single-owner-per-role).

    Uses ``os.open(O_CREAT|O_EXCL|O_WRONLY)`` -- the identical atomic
    exclusive-create idiom as ``ArtifactStore.lock_plan`` (artifacts.py) --
    so the first caller to create the ``.lease`` file wins outright. When the
    lease is already held, the existing holder's liveness (pid_alive) and
    heartbeat age (file mtime) are checked: if _lease_is_stale, the stale
    lease is reaped and the claim retried once; otherwise the existing holder is
    reported under *policy* (STAND_DOWN/WARN_AND_PROCEED) and the on-disk
    body is never touched (no clobbering). *now* is injectable for
    deterministic tests; defaults to the real UTC clock.
    """
    if now is None:
        now = datetime.now(UTC)
    path = lease_path_for_name(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _create_and_write_lease(path, holder)
    except FileExistsError:
        pass
    else:
        return _acquired_claim(name, holder, root)

    existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
    is_stale = _lease_is_stale(existing_holder, holder_alive=holder_alive, age_secs=age_secs)

    if is_stale:
        # Re-verify staleness immediately before unlinking: a competitor
        # could have reaped-and-reclaimed this same stale lease in the gap
        # since the read above (their O_EXCL create winning the race). This
        # narrows -- POSIX has no atomic compare-and-unlink, so it cannot
        # fully eliminate -- that window: if the lease is no longer stale,
        # it must NOT be unlinked, or we would clobber the competitor's
        # brand-new, live lease and silently violate single-owner-per-role.
        existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
        is_stale = _lease_is_stale(existing_holder, holder_alive=holder_alive, age_secs=age_secs)

    if is_stale:
        path.unlink(missing_ok=True)
        try:
            _create_and_write_lease(path, holder)
        except FileExistsError:
            # Lost the retry race to a new competitor: report THEIR holder
            # under the normal held/policy path below (retry is attempted
            # at most once -- we do not loop).
            existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
        else:
            # Reaped-and-reclaimed a stale lease. The displaced holder was
            # SUPPOSED to be gone (_lease_is_stale), but this is exactly the
            # path that -- with a
            # mis-tuned TTL -- once let a duplicate steal a live-but-quiet
            # watcher's lease (task 2796, THREAD 1). Emit a structured, greppable
            # WARNING naming the displaced holder (slug + pid), the observed
            # heartbeat age, and the new holder, so any residual reclaim of a
            # supposedly-live holder is loud rather than silent. Pure logging --
            # no effect on the claim outcome (fail-safe); *existing_holder* may
            # be None (a corrupt/unreadable displaced body), rendered as an
            # explicit placeholder rather than raising.
            displaced_slug = existing_holder.session_slug if existing_holder is not None else '<unknown>'
            displaced_pid = existing_holder.pid if existing_holder is not None else -1
            logger.warning(
                'claim_lease: reaped stale lease %s (displaced holder=%s pid=%s '
                'heartbeat_age=%.0fs); acquired by %s',
                name,
                displaced_slug,
                displaced_pid,
                age_secs,
                holder.session_slug,
            )
            return _acquired_claim(name, holder, root)

    decision = LeaseDecision.STAND_DOWN if policy is LeasePolicy.STAND_DOWN else LeaseDecision.PROCEED
    return LeaseClaim(
        name=name,
        decision=decision,
        acquired=False,
        holder=existing_holder,
        holder_alive=holder_alive,
        heartbeat_age_secs=age_secs,
        message=_render_contention_message(
            existing_holder, holder_alive=holder_alive, age_secs=age_secs, policy=policy
        ),
        holder_record_state=holder_record_state(existing_holder, root=root),
    )


def heartbeat_lease(
    name: str, *, slug: str, force: bool = False, root: Path | str | None = None
) -> LeaseMutation:
    """Bump the *name* lease file's mtime -- but only if *slug* actually holds it.

    The file mtime remains the SINGLE heartbeat clock (the one
    ``claim_lease`` and ``reap_stale_leases`` both read); ``os.utime`` is
    still the only mutation, and no timestamp is written into the lease body
    where it could silently disagree with mtime.

    A REFUSED result is a deliberate no-op, not a fault: before task 3994
    any caller could refresh any lease, and because staleness requires BOTH
    a dead holder pid AND an aged heartbeat, a stranger beating a dead
    holder's lease kept the AND-guard from ever firing -- the lease became
    structurally unreapable and outlived its holder indefinitely.

    Fail-soft is unchanged: an absent lease is ABSENT -- including one a
    concurrent --force release or reap removes after the ownership gate but
    before the touch -- and any other OSError from ``os.utime`` is FAULTED
    after a logger.error -- never a raise, so a lease-substrate fault cannot
    interrupt a watcher's main loop.
    """
    path = lease_path_for_name(name, root=root)
    terminal, holder = _check_lease_ownership(path, slug=slug, force=force, verb='heartbeat_lease')
    if terminal is not None:
        return terminal
    try:
        os.utime(path, None)
    except FileNotFoundError:
        return LeaseMutation.ABSENT
    except OSError:
        logger.error('heartbeat_lease: failed to touch %s', path, exc_info=True)
        return LeaseMutation.FAULTED
    return LeaseMutation.APPLIED if _is_lease_owner(holder, slug) else LeaseMutation.FORCED


def _is_lease_owner(holder: LeaseHolder | None, slug: str) -> bool:
    """Whether *slug* is the lease's holder.

    THE single ownership predicate -- ``_check_lease_ownership`` gates on it
    and the mutating verbs re-use it to tell APPLIED (I am the owner) from
    FORCED (I overrode someone else), so the two can never drift apart.

    Compares ONLY ``session_slug``, never the pid: the slug is a session's
    stable identity key (see session_records.build_session_slug), while the pid is the
    INDEPENDENT liveness probe. Conflating them would make a legitimate
    re-heartbeat fail whenever pid resolution differed between two
    invocations of the same session. An unreadable body (*holder* is None)
    is owned by nobody -- fail toward held.
    """
    return holder is not None and holder.session_slug == slug


def _check_lease_ownership(
    path: Path,
    *,
    slug: str,
    force: bool,
    verb: str,
) -> tuple[LeaseMutation | None, LeaseHolder | None]:
    """Gate a lease mutation on ownership (task 3994).

    Returns ``(None, holder)`` when the caller MAY proceed, and
    ``(<terminal LeaseMutation>, holder)`` when it must not. Shared by
    ``heartbeat_lease`` and ``release_lease`` so there is exactly ONE
    ownership predicate for the two mutating verbs -- a second copy would
    be free to drift, and "who may touch this lease" is precisely the
    question that must have one answer.

    The (holder, alive, age) read goes through ``_read_lease_holder_state``,
    the same reader ``claim_lease`` decides on, so a refusal log carries the
    identical fields a contention message does -- the holder a refused
    caller is told about is the holder a contending claim would report.

    - no readable holder and no file -- absent from the start, or removed
      mid-check by a concurrent --force release or reap -> (ABSENT, None),
      QUIET: an idempotent release is not a failure and must not
      manufacture an ERROR.
    - body still present but unreadable/corrupt -> not owned by anybody, so
      REFUSED (fail TOWARD held, mirroring _read_lease_holder_state's own
      documented rule -- a corrupt lease is not stranded: reap_stale_leases
      still removes it under the ``corrupt`` rule once past
      LEASE_HEARTBEAT_TTL, and --force covers immediate operator recovery).
    - slug matches -> proceed.
    - slug mismatch -> REFUSED + logger.error naming the verb, the lease,
      the REAL holder (slug + pid), the heartbeat age and the requester, so
      the refusal is greppable and both parties are attributable (INV-2).
    - *force* overrides either refusal, at WARNING, naming both parties.
    """
    holder, _holder_alive, age_secs = _read_lease_holder_state(path, now=datetime.now(UTC))
    if holder is None and not path.exists():
        return LeaseMutation.ABSENT, None
    if _is_lease_owner(holder, slug):
        return None, holder

    holder_slug = holder.session_slug if holder is not None else '<unreadable lease body>'
    holder_pid = holder.pid if holder is not None else -1
    if force:
        logger.warning(
            '%s: FORCED on %s -- requester=%s is not the holder '
            '(holder=%s pid=%s heartbeat_age=%.0fs); overriding on explicit --force',
            verb,
            path.stem,
            slug,
            holder_slug,
            holder_pid,
            age_secs,
        )
        return None, holder
    logger.error(
        '%s: REFUSED on %s -- requester=%s is not the holder '
        '(holder=%s pid=%s heartbeat_age=%.0fs); nothing was touched '
        '(inspect with lease-show, override with --force)',
        verb,
        path.stem,
        slug,
        holder_slug,
        holder_pid,
        age_secs,
    )
    return LeaseMutation.REFUSED, holder


def release_lease(
    name: str, *, slug: str, force: bool = False, root: Path | str | None = None
) -> LeaseMutation:
    """Remove the *name* lease -- but only if *slug* actually holds it.

    *slug* is a REQUIRED keyword argument, not an optional check: an
    opt-in ownership test would leave "evict a live holder's lease" reachable
    from Python and would be the path of least resistance for any future
    caller (task 3994). ``force=True`` is the single, loudly-logged bypass.

    Idempotent and fail-soft, exactly as before: an absent lease is ABSENT
    (quiet -- a second release is not a failure), and an OSError from the
    unlink itself is FAULTED after a logger.error, never a raise.

    Returns the LeaseMutation describing what actually happened; see that
    enum for each value's meaning.
    """
    path = lease_path_for_name(name, root=root)
    terminal, holder = _check_lease_ownership(path, slug=slug, force=force, verb='release_lease')
    if terminal is not None:
        return terminal
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error('release_lease: failed to remove %s', path, exc_info=True)
        return LeaseMutation.FAULTED
    return LeaseMutation.APPLIED if _is_lease_owner(holder, slug) else LeaseMutation.FORCED


@dataclass(frozen=True)
class LeaseStatus:
    """A read-only snapshot of one lease -- what ``lease-show`` prints (task 3994).

    name: the lease name inspected (see build_lease_name).
    state: 'held' (a parseable body), 'unreadable' (the file exists but its
        body is missing/corrupt -- reported, never rendered as a fake
        holder) or 'absent' (no such lease). The discriminator: the holder_*
        identity fields are populated only for 'held'.
    holder_slug / holder_pid: the holder's identity from the body, or None.
    holder_pid_alive: whether that pid is currently running (pid_alive) --
        the LIVENESS axis, independent of freshness.
    heartbeat_ts: ISO-8601 timestamp of the last heartbeat, derived from the
        lease file's mtime (None when absent) -- the FRESHNESS axis.
    heartbeat_age_secs: seconds since that heartbeat.
    reclaimable: True iff a contender could legitimately reap this lease --
        i.e. exactly claim_lease/reap_stale_leases' own staleness predicate,
        _lease_is_stale.
    holder_record_slug: the body's ``record_slug``, or None when it is blank
        or the lease is not 'held'.
    holder_record_state: the holder's registry record (see
        HolderRecordState) -- the record axis, independent of liveness.

    Task 3994 withdrew an earlier ``holder_session`` field that printed a
    constant ``holder_session=absent`` (a lease slug is not a record key).
    ``holder_record_state`` reads the body's ``record_slug`` instead, and
    spells the no-evidence case UNLINKED.
    """

    name: str
    state: str
    holder_slug: str | None
    holder_pid: int | None
    holder_pid_alive: bool
    heartbeat_ts: str | None
    heartbeat_age_secs: float
    reclaimable: bool
    holder_record_slug: str | None
    holder_record_state: HolderRecordState


def lease_status(
    name: str, *, root: Path | str | None = None, now: datetime | None = None
) -> LeaseStatus:
    """Inspect the *name* lease without touching it (backs the ``lease-show`` verb).

    WHY THIS EXISTS (task 3994, defect 3). A lease's heartbeat lives ONLY in
    the file's mtime -- the body carries an immutable ``start_ts`` -- so
    ``cat <lease>`` shows you WHO holds it and WHEN THEY STARTED but not
    whether they are still beating. Diagnosing a contended lease therefore
    meant ``cat`` + ``stat -c %y`` + a TTL comparison done by hand, and
    getting that arithmetic wrong is what let the 2026-08-08 rotation
    conclude a live holder's lease was reclaimable. This replaces the
    archaeology with one command.

    Freshness and liveness are computed by ``_read_lease_holder_state`` --
    the SAME reader claim_lease decides on -- and ``heartbeat_ts`` is derived
    from the age that reader returns rather than from a second stat(), so
    there is exactly one clock and the display cannot disagree with the
    decision. Likewise *reclaimable* is claim_lease's own staleness
    predicate, _lease_is_stale.

    READ-ONLY and fail-soft: never mutates the lease (an inspection that
    bumped the mtime would keep a dead holder's lease unreapable), never
    raises -- an unreadable body is a reported *state*, not an exception.
    *now* is injectable for deterministic tests.
    """
    if now is None:
        now = datetime.now(UTC)
    path = lease_path_for_name(name, root=root)
    if not path.is_file():
        return LeaseStatus(
            name=name,
            state='absent',
            holder_slug=None,
            holder_pid=None,
            holder_pid_alive=False,
            heartbeat_ts=None,
            heartbeat_age_secs=0.0,
            reclaimable=False,
            holder_record_slug=None,
            holder_record_state=HolderRecordState.UNLINKED,
        )
    holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
    return LeaseStatus(
        name=name,
        state='held' if holder is not None else 'unreadable',
        holder_slug=holder.session_slug if holder is not None else None,
        holder_pid=holder.pid if holder is not None else None,
        holder_pid_alive=holder_alive,
        heartbeat_ts=(now - timedelta(seconds=age_secs)).isoformat(),
        heartbeat_age_secs=age_secs,
        reclaimable=_lease_is_stale(holder, holder_alive=holder_alive, age_secs=age_secs),
        holder_record_slug=(holder.record_slug or None) if holder is not None else None,
        holder_record_state=holder_record_state(holder, root=root),
    )


@dataclass(frozen=True)
class ReapedLease:
    """One ``.lease`` file removed by reap_stale_leases.

    path: the removed ``<name>.lease`` file.
    lease_name: the lease's identity, derived from the filename itself
        (the stem of *path*) -- never from the body, so a corrupt/missing
        body is still reapable.
    reason: why it was reaped -- ``'stale_pid'`` (a parseable body whose
        pid is not running), ``'shared_host'`` (a parseable body whose pid
        is running but is a shared ``claude agents`` host, which cannot
        vouch for its session) or ``'corrupt'`` (body missing/unparseable),
        each once _lease_is_stale judges the lease stale.
    """

    path: Path
    lease_name: str
    reason: str


def reap_stale_leases(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[ReapedLease]:
    """Sweep ``<root>/leases/*.lease`` and remove stale lease files.

    Structural copy of reap_stale_records (PRD §4.8): identity (which file
    to remove) is derived from the *path* -- the ``<name>.lease`` filename
    stem -- never from the body, so a corrupt or unreadable lease body is
    still reapable. Age is measured from the lease file's mtime, the same
    heartbeat clock claim_lease/heartbeat_lease bump.

    Rules (first match wins):
    - lease body unreadable (missing or corrupt) and age >
      LEASE_HEARTBEAT_TTL -> reaped, reason='corrupt'.
    - a parseable body that _lease_is_stale judges stale -> reaped,
      reason='shared_host' when its pid is running (only a shared host's
      running pid fails to vouch), else reason='stale_pid'.
    - otherwise -> kept -- in particular a LIVE holder pid that can vouch
      for its session is never reaped, regardless of heartbeat age (mirrors
      reap_stale_records' non-terminal live-pid rule).

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock.
    """
    if now is None:
        now = datetime.now(UTC)
    base = leases_dir(root)
    reaped: list[ReapedLease] = []
    if not base.is_dir():
        return reaped

    for lease_path in sorted(base.glob('*.lease')):
        if not lease_path.is_file():
            continue
        lease_name = lease_path.stem
        try:
            mtime = lease_path.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age_secs = (now - datetime.fromtimestamp(mtime, tz=UTC)).total_seconds()

        try:
            holder = LeaseHolder.from_json(lease_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            holder = None
        holder_alive = holder is not None and pid_alive(holder.pid)
        if not _lease_is_stale(holder, holder_alive=holder_alive, age_secs=age_secs):
            reason = None
        elif holder is None:
            reason = 'corrupt'
        elif holder_alive:
            reason = 'shared_host'
        else:
            reason = 'stale_pid'

        if reason is not None:
            try:
                lease_path.unlink()
            except OSError:
                # A single unreapable lease (permission error, a concurrent
                # reap/release race) must not abort the sweep -- log and
                # move on to the next candidate.
                logger.error('reap_stale_leases: failed to remove %s', lease_path, exc_info=True)
                continue
            reaped.append(ReapedLease(path=lease_path, lease_name=lease_name, reason=reason))

    return reaped
