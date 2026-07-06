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


class IllegalLaneTransition(Exception):
    """Raised when a caller attempts a (from, to) edge not in LEGAL_TRANSITIONS.

    Never silent-heal (PRD I2): the durable record is left unchanged when this
    is raised.
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
        return self._worktree_base / '.lane-state'

    def _record_path(self, lane: Path | str) -> Path:
        return self.state_dir / f'{Path(lane).name}.json'

    def read(self, lane: Path | str) -> LaneRecord | None:
        """Return the durable record for *lane*, or ``None`` if absent/corrupt."""
        path = self._record_path(lane)
        if not path.is_file():
            return None
        try:
            return LaneRecord.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                'lane_lifecycle: failed to parse record at %s', path, exc_info=True,
            )
            return None

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

        On an illegal edge: files a best-effort born-at-L2 escalation (if an
        escalation queue is wired) and raises ``IllegalLaneTransition``
        WITHOUT touching the on-disk record (PRD I2, never silent-heal).

        On a legal edge: merges **fields onto the current record (or an
        all-``None`` base if this is the first transition), stamps ``state``
        and ``updated_at``, clears ``task_id``/``title`` on the ``RELEASED``
        edge, atomically writes, and returns the new record.
        """
        current = self.read(lane)
        current_state = current.state if current is not None else None

        if (current_state, to) not in LEGAL_TRANSITIONS:
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

    def _file_illegal_transition_escalation(
        self, lane: Path | str, current_state: LaneState | None, to: LaneState,
    ) -> None:
        """Best-effort born-at-L2 filer for an illegal transition attempt.

        No-op when no escalation queue is wired (bare unit-test / unwired
        contexts stay green, mirrors
        ``Harness._file_pool_storage_absent_escalation``). Any submit failure
        is swallowed + logged so escalation filing never masks the
        ``IllegalLaneTransition`` raise.
        """
        if self._escalation_queue is None:
            return
        try:
            from escalation.models import Escalation  # noqa: PLC0415

            lane_name = Path(lane).name
            sentinel_task_id = f'lane-lifecycle-{lane_name}'
            esc = Escalation(
                id=self._escalation_queue.make_id(sentinel_task_id),
                task_id=sentinel_task_id,
                agent_role=ESCALATION_SENTINEL_ROLE,
                severity='critical',
                category='risk_identified',
                summary=(
                    f'Illegal lane transition on {lane_name!r}: '
                    f'{current_state} -> {to}'
                )[:200],
                detail=(
                    f'LaneLifecycle.transition rejected an illegal edge for lane '
                    f'{lane_name!r}: current state={current_state}, requested '
                    f'transition to={to}. The durable record was left unchanged '
                    '(never silent-heal, PRD W11 I2).'
                ),
                level=2,
            )
            self._escalation_queue.submit(esc)
        except Exception:
            logger.warning(
                'lane_lifecycle: failed to file illegal-transition escalation '
                'for lane %s', Path(lane).name, exc_info=True,
            )

    async def quarantine(self, lane: Path | str, branch: str, reason: str) -> Path | None:
        """Delegate the git-side relocation, then record the QUARANTINED edge.

        Requires an injected ``quarantine_worktree`` callable (raises
        ``RuntimeError`` if unwired). ``any -> QUARANTINED`` is always legal
        (see ``LEGAL_TRANSITIONS``), so the durable record transition below
        never raises ``IllegalLaneTransition``; it is preserved beside the
        relocated worktree.
        """
        if self._quarantine_worktree is None:
            raise RuntimeError(
                'LaneLifecycle.quarantine: no quarantine_worktree callable wired'
            )
        dest = await self._quarantine_worktree(lane, branch, reason)
        self.transition(lane, LaneState.QUARANTINED)
        return dest
