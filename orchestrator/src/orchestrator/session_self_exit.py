"""The Stop hook's opt-in self-exit: decide from the record dir, then SIGINT the owning ``claude``.

A spawned session that has finished touches ``exit-requested`` in its record
dir. On a proven-owner Stop, ``SelfExit.evaluate`` checks every precondition
and returns the ``ExitRequest`` the hook stamps in its one write;
``SelfExit.enact`` then signals the owning ``claude`` and deletes the marker.
Stdlib-only: ``session_hooks.py`` imports this at module scope and runs from a
bare shell with only ``PYTHONPATH=orchestrator/src``, pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
Contract: plans/review-all-program-tooling-prd.md §4 H5 and §5 decision 10.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from orchestrator import session_registry

logger = logging.getLogger(__name__)

_MARKER_READ_LIMIT = 4096
"""Bytes of the marker read as provenance; the trailer writes one session id."""


class SelfExitRefusal(StrEnum):
    """Why an evaluation with the marker present did not attempt the exit."""

    SELF_EXIT_OFF = 'self-exit-off'
    RESULT_UNREADABLE = 'result-unreadable'
    OUTCOME_NOT_TERMINAL = 'outcome-not-terminal'
    OWNER_PID_UNRESOLVED = 'owner-pid-unresolved'
    OWNER_PID_MISMATCH = 'owner-pid-mismatch'
    SHARED_SESSION_HOST = 'shared-session-host'
    BACKGROUND_TASKS_UNREPORTED = 'background-tasks-unreported'
    BACKGROUND_TASKS_RUNNING = 'background-tasks-running'
    SESSION_CRONS_UNREPORTED = 'session-crons-unreported'
    SESSION_CRONS_SCHEDULED = 'session-crons-scheduled'


@dataclass(frozen=True)
class SelfExitEvaluation:
    """One Stop event's decision, and the marker it was made from."""

    request: session_registry.ExitRequest
    marker: Path

    def stamp(self, record: session_registry.SessionRecord) -> None:
        record.exit_request = self.request
        if self.request.state is session_registry.ExitRequestState.ATTEMPTED:
            record.exit_requested_at = self.request.at


@dataclass(frozen=True)
class SelfExit:
    """The decision and the action; *kill* is the one seam, the live ``os.kill`` by default."""

    kill: Callable[[int, int], None] = os.kill

    def evaluate(
        self,
        record: session_registry.SessionRecord,
        hook_input: Mapping[str, Any],
        *,
        owner_pid: int | None,
        root: Path | str | None,
    ) -> SelfExitEvaluation | None:
        """The evaluation to stamp, or None when the record dir holds no marker."""
        record_dir = session_registry.record_path_for_slug(record.session_slug, root=root).parent
        marker = record_dir / session_registry.EXIT_REQUESTED_FILENAME
        present, marker_session_id = _read_marker(marker)
        if not present:
            return None
        refusal = _first_refusal(record, hook_input, owner_pid=owner_pid, record_dir=record_dir)
        if refusal is None:
            state, reason = session_registry.ExitRequestState.ATTEMPTED, None
        else:
            state, reason = session_registry.ExitRequestState.REFUSED, refusal.value
        request = session_registry.ExitRequest(
            state=state,
            reason=reason,
            pid=owner_pid,
            at=datetime.now(UTC).isoformat(),
            marker_session_id=marker_session_id,
        )
        return SelfExitEvaluation(request, marker)

    def enact(
        self,
        evaluation: SelfExitEvaluation | None,
        record: session_registry.SessionRecord,
        root: Path | str | None,
    ) -> None:
        """Signal the owner of an ATTEMPTED evaluation, then delete its marker; anything else is a no-op.

        A failed signal is stamped ``kill_failed`` in a second, fail-soft
        write and the marker stays, so the next Stop retries.
        """
        if evaluation is None:
            return
        request = evaluation.request
        if request.state is not session_registry.ExitRequestState.ATTEMPTED or request.pid is None:
            return
        try:
            self.kill(request.pid, signal.SIGINT)
        except (OSError, OverflowError) as exc:
            record.exit_request = replace(
                request, state=session_registry.ExitRequestState.KILL_FAILED, reason=_kill_failure_reason(exc),
            )
            _write_kill_failed(record, root)
            return
        _delete_marker(evaluation.marker, record.session_slug)


def _read_marker(marker: Path) -> tuple[bool, str | None]:
    """Whether *marker* exists, and its content as provenance; an unreadable marker is present, unattributed."""
    try:
        with marker.open('rb') as handle:
            raw = handle.read(_MARKER_READ_LIMIT)
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None
    return True, raw.decode('utf-8', errors='replace').strip() or None


def _first_refusal(
    record: session_registry.SessionRecord,
    hook_input: Mapping[str, Any],
    *,
    owner_pid: int | None,
    record_dir: Path,
) -> SelfExitRefusal | None:
    if not record.self_exit:
        return SelfExitRefusal.SELF_EXIT_OFF
    return (
        _outcome_refusal(record_dir)
        or _owner_refusal(record.claude_owner_pid, owner_pid)
        or _hook_list_refusal(
            hook_input,
            'background_tasks',
            unreported=SelfExitRefusal.BACKGROUND_TASKS_UNREPORTED,
            busy=SelfExitRefusal.BACKGROUND_TASKS_RUNNING,
        )
        or _hook_list_refusal(
            hook_input,
            'session_crons',
            unreported=SelfExitRefusal.SESSION_CRONS_UNREPORTED,
            busy=SelfExitRefusal.SESSION_CRONS_SCHEDULED,
        )
    )


def _outcome_refusal(record_dir: Path) -> SelfExitRefusal | None:
    try:
        text = (record_dir / session_registry.RESULT_FILENAME).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return SelfExitRefusal.RESULT_UNREADABLE
    if session_registry.parse_result_outcome(text) in session_registry.TERMINAL_RESULT_OUTCOMES:
        return None
    return SelfExitRefusal.OUTCOME_NOT_TERMINAL


def _owner_refusal(bound_pid: int | None, owner_pid: int | None) -> SelfExitRefusal | None:
    if bound_pid is None or owner_pid is None:
        return SelfExitRefusal.OWNER_PID_UNRESOLVED
    if bound_pid != owner_pid:
        return SelfExitRefusal.OWNER_PID_MISMATCH
    if session_registry.is_shared_session_host(owner_pid):
        return SelfExitRefusal.SHARED_SESSION_HOST
    return None


def _hook_list_refusal(
    hook_input: Mapping[str, Any],
    key: str,
    *,
    unreported: SelfExitRefusal,
    busy: SelfExitRefusal,
) -> SelfExitRefusal | None:
    """An absent or non-list *key* cannot be read as empty, so it refuses too."""
    value = hook_input.get(key)
    if not isinstance(value, list):
        return unreported
    return busy if value else None


def _kill_failure_reason(exc: OSError | OverflowError) -> str:
    """The errno's symbolic name (``'ESRCH'``), else the exception's class name."""
    code = getattr(exc, 'errno', None)
    if isinstance(code, int) and code in errno.errorcode:
        return errno.errorcode[code]
    return type(exc).__name__


def _write_kill_failed(record: session_registry.SessionRecord, root: Path | str | None) -> None:
    try:
        session_registry.write_record(record, root=root)
    except OSError:
        logger.warning('kill_failed stamp for session %s could not be written', record.session_slug, exc_info=True)


def _delete_marker(marker: Path, slug: str) -> None:
    try:
        marker.unlink()
    except OSError:
        logger.warning('self-exit marker for session %s could not be deleted', slug, exc_info=True)
