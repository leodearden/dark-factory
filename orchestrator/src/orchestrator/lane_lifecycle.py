"""LaneLifecycle — the single durable-record writer for warm-lane state.

PRD W11 (``plans/worktree-lane-lifecycle-prd.md``), task alpha (mechanism 1):
gives the warm-lane pool one authoritative durable state record per lane,
written through one writer. ``WarmLanePool``'s in-memory map becomes a cache
of these records (consumed by task gamma, the GitOps acquire/release writer,
and task delta, the Harness crash-recovery reader).

Deliberately a SEPARATE module from ``warm_lane_pool.py``: that module is a
pure, git/escalation-free in-memory state machine (its own docstring: "No git
I/O"). This module owns file I/O, escalation filing, and async quarantine —
mixing those in would break the pool's purity and its existing 2-value
``LaneState{FREE,ASSIGNED}`` (PRD Open Q3; resolved: new module).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

# Escalation sentinel role for illegal-transition escalations. Matches the
# 'harness-' prefix in escalation.server._HARNESS_SENTINEL_ROLE_PREFIXES so
# the born-at-L2 record is exempt from the agent-role downgrade gate and
# stays L2 (routes straight to a human). PRD Open Q4.
ESCALATION_SENTINEL_ROLE = 'harness-lane-lifecycle'

# Directory (a direct child of worktree_base) holding every lane's durable
# LaneRecord — see LaneLifecycle.state_dir.
LANE_STATE_DIRNAME = '.lane-state'

# Sentinel filename marking worktree_base as backed by live pool storage
# (task 2099; folded into this module by W11 gamma).  Lives ON the pool
# storage itself (plain dir or real mount alike — substrate-independent, no
# config knob), so it disappears along with an unmounted mountpoint even
# though the mountpoint DIR still exists.  See
# LaneLifecycle.pool_storage_present() / mark_pool_storage_present().
# GitOps re-exports this constant and delegates both methods to a shared
# LaneLifecycle instance so the literal '.pool-root' lives only here.
POOL_ROOT_SENTINEL = '.pool-root'


class LaneState(Enum):
    """Lifecycle states for a single warm lane (PRD W11 Contract)."""

    SEED = 'seed'
    REGISTERED = 'registered'
    ASSIGNED = 'assigned'
    IN_USE = 'in_use'
    RELEASED = 'released'
    QUARANTINED = 'quarantined'


# Legal (from, to) edges. ``from`` is ``None`` for the pre-record "—" origin
# (a lane with no durable record yet). Built as the explicit table from the
# PRD's "Lane state transition table" plus a comprehension adding
# (state, QUARANTINED) for every state INCLUDING the None origin (recovery
# divergence can quarantine a lane at any point, even before a record exists).
LEGAL_TRANSITIONS: frozenset[tuple[LaneState | None, LaneState]] = frozenset(
    {
        (None, LaneState.SEED),
        (LaneState.SEED, LaneState.REGISTERED),
        (LaneState.REGISTERED, LaneState.ASSIGNED),
        (LaneState.RELEASED, LaneState.ASSIGNED),
        (LaneState.ASSIGNED, LaneState.IN_USE),
        (LaneState.IN_USE, LaneState.RELEASED),
        (LaneState.ASSIGNED, LaneState.RELEASED),
    }
    | {(origin, LaneState.QUARANTINED) for origin in [*list(LaneState), None]}
)


class AcquireRoute(Enum):
    """Named routes through ``GitOps._acquire_warm_lane_impl`` (PRD W11 eta,
    mechanism 3). Each member names one of the 7 acquire-time decision
    branches in ``git_ops.py``; see ``ACQUIRE_ROUTE_TRANSITIONS`` for the
    canonical durable edge each one takes.
    """

    #: git_ops.py ~2894-2906 — in-memory map hit, identity OK.
    REUSE = 'reuse'
    #: git_ops.py ~2882-2892,2906 — reused lane whose worktree registration
    #: was restored in place by ``_repair_orphaned_reuse_lane`` (task 2097)
    #: before reuse.
    REUSE_REPAIR = 'reuse_repair'
    #: git_ops.py ~3070-3122 — ``git worktree add -b`` + seed on a
    #: non-registered lane.
    CREATE_ONCE_FRESH = 'create_once_fresh'
    #: git_ops.py ~3020-3068 — orphan ``task/<id>`` branch has commits beyond
    #: main; ``git worktree add`` (no ``-b``) + seed, then the reuse tail.
    CREATE_ONCE_REATTACH = 'create_once_reattach'
    #: git_ops.py ~3132-3161 — already-registered lane whose
    #: ``.task-meta/plan.json`` matches this task (post-restart backstop).
    DISK_BACKSTOP_REUSE = 'disk_backstop_reuse'
    #: git_ops.py ~3163-3213 — already-registered lane whose orphan branch
    #: has commits beyond main; ``git checkout -f`` + the reuse tail.
    RESET_IN_PLACE_REATTACH = 'reset_in_place_reattach'
    #: git_ops.py ~2909-2945 (in-memory reuse identity MISMATCH variant) and
    #: ~3215-3240 (registered FREE lane) — fresh reset+reseed via
    #: ``_reset_and_seed_recycled_lane`` or the inline mismatch reset.
    RECYCLE = 'recycle'


# Canonical (from, to) edge for each AcquireRoute — the durable-state
# contract every acquire_warm_lane route takes en route to ASSIGNED. This is
# a documented/asserted contract, not a static replacement for the runtime
# transition: GitOps._note_assigned_via_route still resolves the actual
# predecessor state at runtime (the pool's 2-value cache does not map 1:1
# onto this 6-state durable model) and uses this table for the terminal
# ASSIGNED target + the route name.
ACQUIRE_ROUTE_TRANSITIONS: dict[AcquireRoute, tuple[LaneState | None, LaneState]] = {
    AcquireRoute.REUSE: (LaneState.RELEASED, LaneState.ASSIGNED),
    AcquireRoute.REUSE_REPAIR: (LaneState.RELEASED, LaneState.ASSIGNED),
    AcquireRoute.CREATE_ONCE_FRESH: (LaneState.REGISTERED, LaneState.ASSIGNED),
    AcquireRoute.CREATE_ONCE_REATTACH: (LaneState.REGISTERED, LaneState.ASSIGNED),
    AcquireRoute.DISK_BACKSTOP_REUSE: (LaneState.RELEASED, LaneState.ASSIGNED),
    AcquireRoute.RESET_IN_PLACE_REATTACH: (LaneState.RELEASED, LaneState.ASSIGNED),
    AcquireRoute.RECYCLE: (LaneState.RELEASED, LaneState.ASSIGNED),
}


def _validate_acquire_route_transitions() -> None:
    """Module-import guard: the route table must stay in sync with both
    ``AcquireRoute`` and ``LEGAL_TRANSITIONS`` as either vocabulary evolves.
    """
    keys = set(ACQUIRE_ROUTE_TRANSITIONS)
    routes = set(AcquireRoute)
    if keys != routes:
        raise AssertionError(
            f'ACQUIRE_ROUTE_TRANSITIONS keys {keys!r} do not match '
            f'AcquireRoute members {routes!r}'
        )
    for route, edge in ACQUIRE_ROUTE_TRANSITIONS.items():
        if edge not in LEGAL_TRANSITIONS:
            raise AssertionError(
                f'ACQUIRE_ROUTE_TRANSITIONS[{route!r}] = {edge!r} is not a '
                f'member of LEGAL_TRANSITIONS'
            )
        if edge[1] is not LaneState.ASSIGNED:
            raise AssertionError(
                f'ACQUIRE_ROUTE_TRANSITIONS[{route!r}] must target '
                f'LaneState.ASSIGNED, got {edge[1]!r}'
            )


_validate_acquire_route_transitions()


class IllegalLaneTransition(Exception):
    """Raised when a caller attempts a (from, to) edge not in LEGAL_TRANSITIONS.

    Never silent-heal (PRD I2): the durable record is left unchanged when this
    is raised.
    """


class CorruptLaneRecord(Exception):
    """Raised internally when a lane's record file exists but fails to parse.

    Distinct from "no record yet" (``None``): a corrupt record must never be
    silently treated as a fresh, pre-record lane (PRD I2, never silent-heal).
    ``LaneLifecycle.read()`` still maps this to ``None`` for callers that only
    care about "current state or absent", but ``transition()`` catches it
    separately so it can refuse (and escalate) rather than fall through to
    the ``None``-origin legal-transition logic.
    """


@dataclass
class LaneRecord:
    """Durable per-lane record (PRD W11 Contract): ``.lane-state/<lane>.json``.

    ``state`` is a ``LaneState`` member in memory; ``to_dict``/``to_json``
    persist it as ``.value`` (a lowercase string) and ``from_dict``/
    ``from_json`` parse it back, mirroring ``escalation.models.Escalation``'s
    round-trip shape.
    """

    state: LaneState
    task_id: str | None = None
    title: str | None = None
    branch: str | None = None
    seeded_from_sha: str | None = None
    updated_at: str = ''

    def to_dict(self) -> dict:
        data = asdict(self)
        data['state'] = self.state.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> LaneRecord:
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields['state'] = LaneState(fields['state'])
        return cls(**fields)

    @classmethod
    def from_json(cls, text: str) -> LaneRecord:
        return cls.from_dict(json.loads(text))


class LaneLifecycle:
    """Single-writer, durable per-lane state record for the warm-lane pool.

    Args:
        worktree_base: Same directory ``WarmLanePool``/``GitOps`` root lanes
            under. Records live at ``<worktree_base>/.lane-state/<lane>.json``.
        escalation_queue: Injected ``EscalationQueue`` used to file a
            born-at-L2 escalation on an illegal transition. ``None`` keeps
            unwired/bare-unit contexts green (no filing, no crash) — mirrors
            ``Harness._file_pool_storage_absent_escalation``.
        quarantine_worktree: Injected async ``(worktree, branch, reason) ->
            Path | None`` callable (``GitOps.quarantine_worktree`` in
            production). Injected rather than importing ``GitOps`` directly
            to avoid the GitOps<->LaneLifecycle import cycle (gamma makes
            GitOps own a LaneLifecycle).
    """

    def __init__(
        self,
        worktree_base: Path,
        *,
        escalation_queue: EscalationQueue | None = None,
        quarantine_worktree=None,
    ) -> None:
        self._worktree_base = Path(worktree_base)
        self._escalation_queue = escalation_queue
        self._quarantine_worktree = quarantine_worktree

    @property
    def state_dir(self) -> Path:
        """Directory holding every lane's durable record."""
        return self._worktree_base / LANE_STATE_DIRNAME

    def _record_path(self, lane: Path | str) -> Path:
        return self.state_dir / f'{Path(lane).name}.json'

    def read(self, lane: Path | str) -> LaneRecord | None:
        """Return the durable record for *lane*, or ``None`` if absent/corrupt."""
        try:
            return self._read_or_raise(lane)
        except CorruptLaneRecord:
            logger.warning(
                'lane_lifecycle: failed to parse record for lane %s',
                Path(lane).name, exc_info=True,
            )
            return None

    def _read_or_raise(self, lane: Path | str) -> LaneRecord | None:
        """Return the durable record for *lane*, or ``None`` if no file exists.

        Raises ``CorruptLaneRecord`` (instead of returning ``None``) when a
        record file exists but fails to parse, so ``transition()`` can tell
        "no record yet" apart from "record present but unreadable".
        """
        path = self._record_path(lane)
        if not path.is_file():
            return None
        try:
            return LaneRecord.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CorruptLaneRecord(f'unparseable lane record at {path}') from exc

    def _write(self, lane: Path | str, record: LaneRecord) -> None:
        """Atomically write *record* for *lane* (tmp file + os.replace).

        Mirrors ``escalation.queue.EscalationQueue._atomic_write_path``: the
        tmp file is created in the target's own parent dir so the replace
        stays within one filesystem, and is cleaned up on failure.
        """
        path = self._record_path(lane)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            suffix='.tmp', prefix=path.stem, dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(record.to_json())
            os.replace(tmp_path_str, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            raise

    def transition(self, lane: Path | str, to: LaneState, **fields: object) -> LaneRecord:
        """The one mutator: validate (from, to), persist, return the new record.

        Rejects unknown ``**fields`` keys with ``TypeError`` before doing any
        I/O, so a caller typo (e.g. ``seeded_sha=`` instead of
        ``seeded_from_sha=``) fails loudly instead of silently vanishing on
        persist (``to_dict``/``asdict`` only ever see declared fields).

        On an illegal edge, or a *corrupt* on-disk record (see
        ``CorruptLaneRecord``) when ``to`` is anything other than
        ``QUARANTINED``: files a best-effort born-at-L2 escalation (if an
        escalation queue is wired) and raises ``IllegalLaneTransition``
        WITHOUT touching the on-disk record (PRD I2, never silent-heal). A
        corrupt record is never silently treated as a fresh, ``None``-origin
        lane.

        On a legal edge: merges **fields onto the current record (or an
        all-``None`` base if this is the first transition, or if the prior
        record was corrupt and ``to`` is ``QUARANTINED``), stamps ``state``
        and ``updated_at``, clears ``task_id``/``title`` on the ``RELEASED``
        edge, atomically writes, and returns the new record.
        """
        unknown_fields = set(fields) - set(LaneRecord.__dataclass_fields__)
        if unknown_fields:
            raise TypeError(
                f'LaneLifecycle.transition: unknown LaneRecord field(s) '
                f'{sorted(unknown_fields)!r}'
            )

        try:
            current = self._read_or_raise(lane)
            corrupt = False
        except CorruptLaneRecord:
            current = None
            corrupt = True

        current_state = current.state if current is not None else None

        if corrupt and to is not LaneState.QUARANTINED:
            self._file_corrupt_record_escalation(lane, to)
            raise IllegalLaneTransition(
                f'lane {Path(lane).name!r} has a corrupt durable record; '
                f'refusing transition to {to} (repair or quarantine required)'
            )

        if not corrupt and (current_state, to) not in LEGAL_TRANSITIONS:
            self._file_illegal_transition_escalation(lane, current_state, to)
            raise IllegalLaneTransition(
                f'illegal lane transition for lane {Path(lane).name!r}: '
                f'{current_state} -> {to}'
            )

        record = LaneRecord(
            state=to,
            task_id=current.task_id if current is not None else None,
            title=current.title if current is not None else None,
            branch=current.branch if current is not None else None,
            seeded_from_sha=current.seeded_from_sha if current is not None else None,
            updated_at=datetime.now(UTC).isoformat(),
        )
        for key, value in fields.items():
            setattr(record, key, value)
        if to is LaneState.RELEASED:
            record.task_id = None
            record.title = None

        self._write(lane, record)
        return record

    def _file_escalation(self, lane: Path | str, *, summary: str, detail: str) -> None:
        """Best-effort born-at-L2 filer shared by the illegal-transition and
        corrupt-record paths.

        No-op when no escalation queue is wired (bare unit-test / unwired
        contexts stay green, mirrors
        ``Harness._file_pool_storage_absent_escalation``). Any submit failure
        is swallowed + logged so escalation filing never masks the caller's
        ``IllegalLaneTransition`` raise.
        """
        if self._escalation_queue is None:
            return
        try:
            from escalation.models import Escalation  # noqa: PLC0415

            sentinel_task_id = f'lane-lifecycle-{Path(lane).name}'
            esc = Escalation(
                id=self._escalation_queue.make_id(sentinel_task_id),
                task_id=sentinel_task_id,
                agent_role=ESCALATION_SENTINEL_ROLE,
                severity='critical',
                category='risk_identified',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self._escalation_queue.submit(esc)
        except Exception:
            logger.warning(
                'lane_lifecycle: failed to file escalation for lane %s',
                Path(lane).name, exc_info=True,
            )

    def _file_illegal_transition_escalation(
        self, lane: Path | str, current_state: LaneState | None, to: LaneState,
    ) -> None:
        """Best-effort born-at-L2 filer for an illegal transition attempt."""
        lane_name = Path(lane).name
        self._file_escalation(
            lane,
            summary=f'Illegal lane transition on {lane_name!r}: {current_state} -> {to}',
            detail=(
                f'LaneLifecycle.transition rejected an illegal edge for lane '
                f'{lane_name!r}: current state={current_state}, requested '
                f'transition to={to}. The durable record was left unchanged '
                '(never silent-heal, PRD W11 I2).'
            ),
        )

    def _file_corrupt_record_escalation(self, lane: Path | str, to: LaneState) -> None:
        """Best-effort born-at-L2 filer for a corrupt on-disk record."""
        lane_name = Path(lane).name
        self._file_escalation(
            lane,
            summary=f'Corrupt lane record for {lane_name!r}; refused transition to {to}',
            detail=(
                f'LaneLifecycle.transition found an unparseable durable record '
                f'for lane {lane_name!r} and refused the requested transition '
                f'to {to} rather than silently treating it as a fresh '
                '(None-origin) lane. Manual repair or quarantine (any -> '
                'QUARANTINED is always legal) is required.'
            ),
        )

    async def quarantine(self, lane: Path | str, branch: str, reason: str) -> Path | None:
        """Delegate the git-side relocation, then record the QUARANTINED edge.

        Requires an injected ``quarantine_worktree`` callable (raises
        ``RuntimeError`` if unwired). ``any -> QUARANTINED`` is always legal
        (see ``LEGAL_TRANSITIONS``), so the durable record transition below
        never raises ``IllegalLaneTransition``.

        The durable record's on-disk location never moves: it always stays
        at ``<worktree_base>/.lane-state/<lane>.json`` (keyed by lane name).
        It is NOT written beside the relocated worktree, even though
        ``quarantine_worktree`` moves the git worktree elsewhere (typically
        under ``GitOps.quarantine_base``).

        Ordering caveat: the git-side relocation happens first and the
        ``QUARANTINED`` record write second. If relocation succeeds but the
        subsequent ``transition()`` write then fails, there is a transient
        window where the worktree has already moved while the durable record
        still reflects its pre-quarantine state.
        """
        if self._quarantine_worktree is None:
            raise RuntimeError(
                'LaneLifecycle.quarantine: no quarantine_worktree callable wired'
            )
        dest = await self._quarantine_worktree(lane, branch, reason)
        self.transition(lane, LaneState.QUARANTINED)
        return dest

    def pool_storage_present(self) -> bool:
        """True iff worktree_base is backed by live pool storage (task 2099).

        Checks for the ``POOL_ROOT_SENTINEL`` (``.pool-root``) sentinel FILE
        directly inside ``worktree_base``.  The sentinel lives ON the pool
        storage itself, so it disappears along with an unmounted mountpoint
        even though the mountpoint directory still exists — unlike
        ``worktree_base.exists()``, which is True for an empty, unmounted
        mountpoint dir (the root cause of the Jul-3 incident this guards
        against).

        Fail-safe: any ``OSError`` (e.g. a torn read racing a concurrent
        mount transition) is treated as absent. Never raises. Ported
        verbatim (W11 gamma sentinel fold) from the original
        ``GitOps.pool_storage_present`` — that method is now a thin
        delegator to this one.
        """
        try:
            return (self._worktree_base / POOL_ROOT_SENTINEL).is_file()
        except OSError:
            logger.debug(
                'pool_storage_present: stat error checking %s — treating as '
                'absent (fail-safe)',
                self._worktree_base,
                exc_info=True,
            )
            return False

    def mark_pool_storage_present(self) -> None:
        """Write the ``.pool-root`` sentinel marking storage as present.

        Idempotent best-effort: creates ``worktree_base`` if missing, then
        writes the sentinel file (a no-op if it already exists). Any
        ``OSError`` is logged and swallowed — never raises, since a failed
        mark must not block whatever seed/warmup operation triggered it.
        Ported verbatim (W11 gamma sentinel fold) from the original
        ``GitOps.mark_pool_storage_present`` — that method is now a thin
        delegator to this one.
        """
        try:
            self._worktree_base.mkdir(parents=True, exist_ok=True)
            (self._worktree_base / POOL_ROOT_SENTINEL).touch(exist_ok=True)
        except OSError:
            logger.warning(
                'mark_pool_storage_present: failed to write sentinel under %s',
                self._worktree_base,
                exc_info=True,
            )
