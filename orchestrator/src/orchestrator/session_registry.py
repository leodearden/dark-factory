"""orchestrator.session_registry — durable session-registry substrate (Attention Rail T3).

PRD: plans/session-attention-rail-prd.md T3 (spine root; §4.1-4.2, §4.8, §6 G5).

This module is the shared, versioned contract for the global, cross-project
session registry at ``~/.claude/fleet/sessions/<slug>/record.json``: session
records and their pointers, the reaps, the T7 leases, and the fleet registry
CLI, whose decision verbs live in ``decision_cli``.

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves and
its stdlib-only siblings, layered so each imports only downward:
``fleet_files`` <- ``decision_registry`` <- ``decision_cli`` <-
``session_registry``, which also imports ``fleet_files`` directly. Nothing
else from the orchestrator, so it can be:

- invoked directly by ``skills/spawn/spawn-claude.sh`` and the SKILL.md verb
  lines via an absolute path (no venv/PYTHONPATH/install required: the
  bootstrap below reaches ``shared/src`` from ``__file__``), and
- imported as ``orchestrator.session_registry`` by downstream Python
  consumers (T4 verify, T5 result, T6 hooks, T7 leases).

Consumers import the schema/contract defined here; they never re-derive the
record shape (PRD §6 G5).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sized
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

# NOTE: this module is executed by absolute path from
# skills/spawn/spawn-claude.sh and the SKILL.md verb lines with no venv,
# install or PYTHONPATH, so the only non-stdlib imports allowed at module
# scope are this checkout's stdlib-only `shared` leaves, reached through the
# bootstrap below. It APPENDS, so an installed or PYTHONPATH `shared` wins.
# No third-party import, and no unguarded `from orchestrator import <sibling>`:
# running a script by path puts only the script's OWN directory on sys.path,
# so intra-orchestrator imports fail there even though they work for
# consumers that import this module with PYTHONPATH set. Stdlib-only
# siblings are imported through the load-mode block below the shared imports.
# Pinned by the tier-1 (bare shell, NO PYTHONPATH) row of
# _BARE_SHELL_ENTRYPOINTS in
# test_session_registry.py::TestStdlibOnlySelfContainment, and mutation-tested
# there via _MUST_BE_REJECTED.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.proc_liveness import pid_alive
from shared.process_identity import Liveness, ProcessIdentity
from shared.project_layout import CheckoutRootError, main_checkout_of
from shared.session_liveness import capture_for_record, judge

# Siblings are imported relative to orchestrator.session_registry, or else from this
# file's own directory: loaded by path, as tier-1 __main__ or under a foreign dotless
# name, that directory is on sys.path only if Python put it there (not under -P, -I or
# PYTHONSAFEPATH, nor for a foreign-name load), so it is borrowed for these imports
# alone. Each mode is pinned in test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import decision_cli
    from .fleet_files import (
        ATOMIC_WRITE_TEMP_SUFFIX,
        atomic_write_text,
        exclusive_sidecar_lock,
        fleet_root,
    )
else:
    _OWN_DIR = str(Path(__file__).resolve().parent)
    _OWN_DIR_BORROWED = _OWN_DIR not in sys.path
    if _OWN_DIR_BORROWED:
        sys.path.append(_OWN_DIR)
    try:
        import decision_cli
        from fleet_files import (
            ATOMIC_WRITE_TEMP_SUFFIX,
            atomic_write_text,
            exclusive_sidecar_lock,
            fleet_root,
        )
    finally:
        if _OWN_DIR_BORROWED:
            sys.path.remove(_OWN_DIR)

# ---------------------------------------------------------------------------
# Schema / contract (PRD §6 G5): consumers import this; they never re-derive
# the record shape. Bump SCHEMA_VERSION on any breaking field change.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# 3: LeaseHolder.record_slug and decision_registry.py::DecisionRecord.record_slug (task 4237).
# 4: SessionRecord.self_exit (task 7179).
# 5: SessionRecord.exit_request and exit_requested_at (task 7193).
# 6: SessionRecord.claude_launch_session_id (task 6069).
# 7: SessionRecord.launcher_identity and claude_owner_identity (task 8125).
# 8: SessionRecord.t3_end (task 8126).
SCHEMA_MINOR = 8
"""Additive-extension counter for this module's contract (Fleet Cockpit C1,
plans/fleet-cockpit-prd.md §6.1). A CODE-LEVEL signal only -- never persisted
per-record. Bump this when a new backward-compatible (optional/defaulted)
field is added; SCHEMA_VERSION (the PERSISTED major) stays unchanged so
rail-vintage and post-extension records remain version-indistinguishable on
disk, and any consumer gating on ``record.schema_version == 1`` keeps
working unmodified. Not read by any consumer as of C1 -- it is intentional
forward scaffolding (not dead code): a code-level seam later Fleet Cockpit
steps (plans/fleet-cockpit-prd.md §2's C1-C10 decomposition) can consult
when reasoning about additive schema drift."""

RESULT_FILENAME = 'result.md'
"""Basename of the result-handback file inside each record dir (Attention
Rail T5): ``<record_dir>/result.md``. Kept in sync BY CONVENTION (not
cross-checked at runtime) with skills/spawn/spawn-claude.sh's own
``CLAUDE_SPAWN_RESULT_FILE`` literal -- ``_run_launching`` returns
``str(record_dir)`` (== spawn-claude.sh's ``SESSION_RECORD_DIR``), so bash's
``$SESSION_RECORD_DIR/result.md`` and this module's
``str(record_dir / RESULT_FILENAME)`` are byte-identical paths."""

EXIT_REQUESTED_FILENAME = 'exit-requested'
"""The marker a self-exit session writes into its record dir; spawn-claude.sh's trailer spells it literally."""


class ResultOutcome(StrEnum):
    """The ``outcome:`` header of RESULT_FILENAME; IN_PROGRESS marks a checkpoint, not an end."""

    DONE = 'done'
    BLOCKED = 'blocked'
    ABANDONED = 'abandoned'
    HANDED_OFF = 'handed-off'
    IN_PROGRESS = 'in-progress'


RESULT_OUTCOMES: tuple[ResultOutcome, ...] = tuple(ResultOutcome)
TERMINAL_RESULT_OUTCOMES: frozenset[ResultOutcome] = frozenset(RESULT_OUTCOMES) - {ResultOutcome.IN_PROGRESS}
OWNING_OUTCOMES: frozenset[ResultOutcome] = frozenset({ResultOutcome.DONE, ResultOutcome.HANDED_OFF})


def parse_result_outcome(text: str) -> ResultOutcome | None:
    """The spawn contract's ``outcome:`` header value, or None when absent or not a contract outcome."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].strip() == '---':
        lines = lines[1:]
    for line in lines:
        if not line.strip() or line.strip() == '---':
            return None
        key, sep, value = line.partition(':')
        if sep and key.strip() == 'outcome':
            try:
                return ResultOutcome(value.strip())
            except ValueError:
                return None
    return None


class ExitRequestState(StrEnum):
    """Where the Stop hook's self-exit evaluation ended (plans/review-all-program-tooling-prd.md §4 H5)."""

    REFUSED = 'refused'
    ATTEMPTED = 'attempted'
    KILL_FAILED = 'kill_failed'


@dataclass(frozen=True)
class ExitRequest:
    """One self-exit evaluation as the Stop hook stamped it; ``reason`` names a refusal or a failed signal."""

    state: ExitRequestState
    reason: str | None
    pid: int | None
    at: str
    marker_session_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            'state': ExitRequestState(self.state).value,
            'reason': self.reason,
            'pid': self.pid,
            'at': self.at,
            'marker_session_id': self.marker_session_id,
        }


class Status(StrEnum):
    """Wire-format session-lifecycle status (PRD §4 decision 4).

    All six members are defined here as the shared contract even though this
    module (T3) only ever writes LAUNCHING/EXITED itself:
      - LAUNCHING / EXITED: written by spawn-claude.sh (this task).
      - RUNNING / FAILED_TO_START: written by the T4 verify-spawn-started step.
      - AWAITING_INPUT / IDLE: written by the T6 Notification/Stop hooks.

    Subclassing ``str`` means a record's status round-trips through
    ``json.dumps``/``json.loads`` as its plain wire string with no custom
    JSON encoder needed.
    """

    LAUNCHING = 'launching'
    RUNNING = 'running'
    AWAITING_INPUT = 'awaiting-input'
    IDLE = 'idle'
    EXITED = 'exited'
    FAILED_TO_START = 'failed-to-start'


TERMINAL_STATUSES: frozenset[Status] = frozenset({Status.EXITED, Status.FAILED_TO_START})
"""Statuses past which a record is eligible for the terminal-TTL reap rule."""


class SpawnMode(StrEnum):
    """How a session relates to its parent (Fleet Cockpit C1, PRD §6.1).

    Modeled as a StrEnum contract for writers, but ``SessionRecord.spawn_mode``
    itself is a plain ``str`` field with NO from_dict coercion through this
    enum -- an unrecognized/foreign value must still round-trip rather than
    raise ``CorruptSessionRecord`` (additive-safe, unlike the closed-set
    ``Status`` contract).

    CHILD: spawned directly by a parent session; parent_session_id names the
        direct spawner. The default -- see spawn-claude.sh's env-export
        baseline.
    SIBLING: spawned to run alongside a parent under a shared ancestor (C7);
        parent_session_id names that shared ancestor, not the direct spawner.
    DETACHED: spawned with no tracked parent linkage.
    """

    CHILD = 'child'
    SIBLING = 'sibling'
    DETACHED = 'detached'


class DisplayKind(StrEnum):
    """Where a session's terminal/pane lives, for focus-arrange (C4).

    Not consumed within C1 itself -- ``Display.kind`` stores the wire value
    as a plain ``str`` with no coercion through this enum (see Display's
    docstring). This is intentional forward scaffolding, not dead code: it
    is the named-constant contract that C4's wm/X11 and tmux focus-arrange
    backends and C6's tmux lane (both PRD-declared as depending on C1;
    plans/fleet-cockpit-prd.md §2) will match against once they land.
    """

    WM = 'wm'
    TMUX = 'tmux'
    T3 = 't3'


class T3End(StrEnum):
    """The end-of-work act a T3-lane spawn asked for, None meaning none; see skills/spawn/SKILL.md "Ending a T3 session"."""

    SETTLE = 'settle'
    ARCHIVE = 'archive'


def parse_t3_end_input(raw: str) -> T3End | None:
    """Parse the ``CLAUDE_SPAWN_T3_END`` input strictly: '' or 'none' is None, a member's value is that member."""
    if raw in ('', 'none'):
        return None
    try:
        return T3End(raw)
    except ValueError:
        raise ValueError(f'CLAUDE_SPAWN_T3_END={raw!r} is not one of none|settle|archive') from None


def _parse_end_of_work_inputs(env: Mapping[str, str]) -> tuple[bool, T3End | None]:
    """Parse (self_exit, t3_end) strictly; an act is set only on the T3 lane, where self-exit means exactly archive."""
    self_exit_raw = env.get('CLAUDE_SPAWN_SELF_EXIT', '') or ''
    t3_end_raw = env.get('CLAUDE_SPAWN_T3_END', '') or ''
    self_exit = self_exit_raw == '1'
    t3_end = parse_t3_end_input(t3_end_raw)
    if t3_end is not None and self_exit != (t3_end is T3End.ARCHIVE):
        raise ValueError(
            f'CLAUDE_SPAWN_SELF_EXIT={self_exit_raw!r} contradicts CLAUDE_SPAWN_T3_END={t3_end_raw!r}: '
            'a T3 spawn self-exits exactly when its act is archive'
        )
    return self_exit, t3_end


@dataclass
class Display:
    """Where to find/focus this session's terminal (Fleet Cockpit C1, PRD §6.1).

    kind: DisplayKind's wire value ('wm', 'tmux' or 't3'); stored as a plain
        ``str`` (no from_dict coercion -- see SpawnMode's docstring for why).
    wm_title: the spawned terminal's window title, for a 'wm' kind's
        window-manager lookup.
    wm_window_id: the window-manager's own window id, when known.
    tmux_target: a tmux target spec (e.g. ``'session:window.pane'``), for a
        'tmux' kind.
    t3_thread_id: the T3 Code thread hosting the session, for a 't3' kind.
    t3_environment_id: the T3 environment that thread lives in, for a 't3' kind.
    """

    kind: str
    wm_title: str = ''
    wm_window_id: str | None = None
    tmux_target: str | None = None
    t3_thread_id: str | None = None
    t3_environment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'wm_title': self.wm_title,
            'wm_window_id': self.wm_window_id,
            'tmux_target': self.tmux_target,
            't3_thread_id': self.t3_thread_id,
            't3_environment_id': self.t3_environment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Display:
        return cls(
            kind=data['kind'],
            wm_title=data.get('wm_title', ''),
            wm_window_id=data.get('wm_window_id'),
            tmux_target=data.get('tmux_target'),
            t3_thread_id=data.get('t3_thread_id'),
            t3_environment_id=data.get('t3_environment_id'),
        )


@dataclass
class Question:
    """A pending question queued against this session (Fleet Cockpit C1, PRD §6.1).

    text: the question text.
    asked_at: ISO-8601 timestamp of when it was queued.
    """

    text: str
    asked_at: str

    def to_dict(self) -> dict[str, Any]:
        return {'text': self.text, 'asked_at': self.asked_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        return cls(text=data['text'], asked_at=data['asked_at'])


def _is_strict_int(value: object) -> TypeGuard[int]:
    """An int that is not a bool: Python's ``True`` would otherwise pass as int 1."""
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_owner_pid(value: Any) -> int | None:
    """Read ``claude_owner_pid`` from a record body, tolerating junk.

    Anything that is not a positive ``_is_strict_int`` -- absent, null, a
    bool, a string, a float, 0, a negative -- reads as None, i.e. "no owner
    pid bound". Never raises: this is on ``from_dict``'s path, and a
    hand-edited or older record body must not be what breaks a session hook.
    """
    if not _is_strict_int(value):
        return None
    return value if value > 0 else None


def _coerce_identity(value: Any, pid: Any) -> ProcessIdentity | None:
    """Read a stored ProcessIdentity naming *pid*; any other shape reads None. Never raises."""
    if not isinstance(value, dict) or not _is_strict_int(pid):
        return None
    identity_pid = value.get('pid')
    start_ticks = value.get('start_ticks')
    boot_id = value.get('boot_id')
    host = value.get('host')
    if not (
        _is_strict_int(identity_pid)
        and identity_pid == pid
        and _is_strict_int(start_ticks)
        and isinstance(boot_id, str)
        and isinstance(host, str)
    ):
        return None
    try:
        return ProcessIdentity(identity_pid, start_ticks, boot_id, host)
    except ValueError:
        return None


def _coerce_session_id(value: Any) -> str | None:
    """Read a stored Claude Code session id from a record body, tolerating junk.

    Mirrors ``_coerce_owner_pid``: anything that is not a ``str`` reads as
    None ("no session id bound") instead of surviving to the hook trio's
    ``(record.claude_session_id or '').strip()`` call, where a non-str would
    raise AttributeError outside the ownership probe's try/except and lose
    the whole hook event. A str value is stripped, and a whitespace-only
    string also reads as None. Never raises: this is on ``from_dict``'s
    path, and a hand-edited or older record body must not be what breaks a
    session hook.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_exit_request(value: Any) -> ExitRequest | None:
    """Read ``exit_request`` from a record body; a non-dict or an unknown state reads None. Never raises."""
    if not isinstance(value, dict):
        return None
    try:
        state = ExitRequestState(value.get('state'))
    except ValueError:
        return None
    reason = value.get('reason')
    at = value.get('at')
    return ExitRequest(
        state=state,
        reason=reason if isinstance(reason, str) else None,
        pid=_coerce_owner_pid(value.get('pid')),
        at=at if isinstance(at, str) else '',
        marker_session_id=_coerce_session_id(value.get('marker_session_id')),
    )


def _coerce_t3_end(value: Any) -> T3End | None:
    """Read a stored ``t3_end``; anything but a member's exact wire string reads None. Never raises."""
    if not isinstance(value, str):
        return None
    try:
        return T3End(value)
    except ValueError:
        return None


def _identity_wire(identity: ProcessIdentity | None) -> dict[str, Any] | None:
    return None if identity is None else dataclasses.asdict(identity)


@dataclass
class SessionRecord:
    """One session-registry record — ``<fleet_root>/sessions/<slug>/record.json``.

    Only ``session_slug`` and ``status`` are mandatory; every other field has
    a documented default so a partial record (e.g. the T6 hand-launched-
    capture upsert path, which knows little beyond the slug) is still
    well-formed. Fields:

    schema_version: contract version this record was written under.
    session_slug: this record's identity/directory key; see build_session_slug.
    title: the spawned terminal's title, e.g. ``'unblock:df#2085 <slug>'``.
    role: the spawned skill/command, e.g. ``'unblock'``.
    project: the project_id the session operates on.
    task_id: the task this session is scoped to, or None if project-level.
    escalation_id: the escalation this session was spawned to resolve, if any.
    prompt: the literal prompt passed to `claude`.
    cwd: the working directory `claude` was launched in.
    launcher_pid: spawn-claude.sh's own $$ — the reaper's liveness proxy.
    launcher_identity: the ProcessIdentity of ``launcher_pid`` captured when
        the record was first written, or None for a record written before
        task 8125 or when /proc could not answer. Reads None unless it names
        ``launcher_pid`` (see ``_coerce_identity``).
    start_ts: ISO-8601 timestamp of when this record was first written.
    status: current lifecycle status; see Status.
    exit_code: claude's exit code, populated once status is terminal.
    result_file: path to this session's result.md (T5); allocated by
        `_run_launching` at launch time (``<record_dir>/RESULT_FILENAME``)
        and preserved unchanged by every later read-modify-write.
    transcript_path: best-effort ``~/.claude/projects/<encoded-cwd>`` path.
    parent_session_id: the spawning session's own session_slug, or None for
        a human-launched root (Fleet Cockpit C1, PRD §6.1).
    spawn_mode: how this session relates to its parent; see SpawnMode.
        Defaults to ``SpawnMode.CHILD``.
    display: where to find/focus this session's terminal, or None if
        unknown; see Display.
    question: a pending question queued against this session, or None; see
        Question.
    claude_session_id: the Claude Code ``session_id`` (hook stdin) of the
        session that OWNS this record, bound once by the first hook event to
        adopt it, or None for a record no hook has claimed yet (e.g.
        spawn-claude.sh's ``launching`` write, or a pre-task-4193 record).
        ``session_hooks.hook_session_slug`` compares it against the current
        hook's stdin session_id to tell the session spawn-claude.sh launched
        from a nested claude that merely inherited CLAUDE_SPAWN_SESSION_ID.
        A non-str or whitespace-only value in a record body reads back as
        None instead of surviving unchanged (see ``_coerce_session_id``).
    claude_launch_session_id: the Claude Code ``session_id`` this record was
        FIRST bound to, stamped once by ``session_hooks._bind_claude_session_id``
        and never re-bound, so it still names the launch transcript after a
        ``/clear`` or compaction re-mint moves ``claude_session_id``. None for
        an unbound record or one bound before this field existed; junk reads
        None (see ``_coerce_session_id``). Consumer:
        ``scripts/legibility/check_transcript_persistence.py::find_matching_transcript``.
    claude_owner_pid: pid of the ``claude`` PROCESS that bound
        ``claude_session_id``, stamped at the same moment, or None for a
        record bound before this field existed (or where the pid could not
        be resolved). Claude Code RE-MINTS ``session_id`` in place on
        ``/clear`` and on automatic compaction, so a session_id mismatch
        alone cannot tell "the owner re-minted" from "a nested claude
        inherited the env var". The owning process keeps its pid across a
        re-mint; a nested ``claude`` never shares it. See
        ``session_hooks._env_slug_is_owned``. A non-int, bool, or
        non-positive value in a record body reads back as None instead of
        surviving unchanged (see ``_coerce_owner_pid``).
    claude_owner_identity: the ProcessIdentity of ``claude_owner_pid``,
        stamped with it, or None for a record bound before task 8125 or when
        /proc could not answer. Reads None unless it names
        ``claude_owner_pid`` (see ``_coerce_identity``).
    self_exit: true when the spawn launched with self-exit on, as resolved
        per lane by ``skills/spawn/spawn-claude.sh`` (rule: the
        ``CLAUDE_SPAWN_SELF_EXIT`` entry in ``skills/spawn/SKILL.md``), which
        hands the registry 1 or empty; stamped at ``launching`` and preserved
        by every later read-modify-write. Anything but a JSON true reads back
        as False.
    t3_end: the end-of-work act the spawn asked for, or None for none;
        stamped at ``launching`` from the input spawn-claude.sh resolved and
        preserved by every later read-modify-write. Anything but a member's
        wire string reads back as None (see ``_coerce_t3_end``).
    exit_request: the Stop hook's last self-exit evaluation, or None; written
        only when the ``exit-requested`` marker is present on a proven-owner
        event. See ExitRequest and ``session_self_exit.py``.
    exit_requested_at: when the hook last stamped an ATTEMPTED evaluation, or
        None if it never has; a later refusal or kill_failed stamp leaves it
        in place.
    """

    session_slug: str
    status: Status
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)
    title: str = field(default='', kw_only=True)
    role: str = field(default='', kw_only=True)
    project: str = field(default='', kw_only=True)
    task_id: str | None = field(default=None, kw_only=True)
    escalation_id: str | None = field(default=None, kw_only=True)
    prompt: str = field(default='', kw_only=True)
    cwd: str = field(default='', kw_only=True)
    launcher_pid: int = field(default=0, kw_only=True)
    launcher_identity: ProcessIdentity | None = field(default=None, kw_only=True)
    start_ts: str = field(default='', kw_only=True)
    exit_code: int | None = field(default=None, kw_only=True)
    result_file: str | None = field(default=None, kw_only=True)
    transcript_path: str | None = field(default=None, kw_only=True)
    parent_session_id: str | None = field(default=None, kw_only=True)
    spawn_mode: str = field(default=SpawnMode.CHILD, kw_only=True)
    display: Display | None = field(default=None, kw_only=True)
    question: Question | None = field(default=None, kw_only=True)
    claude_session_id: str | None = field(default=None, kw_only=True)
    claude_launch_session_id: str | None = field(default=None, kw_only=True)
    claude_owner_pid: int | None = field(default=None, kw_only=True)
    claude_owner_identity: ProcessIdentity | None = field(default=None, kw_only=True)
    self_exit: bool = field(default=False, kw_only=True)
    t3_end: T3End | None = field(default=None, kw_only=True)
    exit_request: ExitRequest | None = field(default=None, kw_only=True)
    exit_requested_at: str | None = field(default=None, kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-scalar-only dict (status as its wire string)."""
        return {
            'schema_version': self.schema_version,
            'session_slug': self.session_slug,
            'title': self.title,
            'role': self.role,
            'project': self.project,
            'task_id': self.task_id,
            'escalation_id': self.escalation_id,
            'prompt': self.prompt,
            'cwd': self.cwd,
            'launcher_pid': self.launcher_pid,
            'launcher_identity': _identity_wire(self.launcher_identity),
            'start_ts': self.start_ts,
            'status': Status(self.status).value,
            'exit_code': self.exit_code,
            'result_file': self.result_file,
            'transcript_path': self.transcript_path,
            'parent_session_id': self.parent_session_id,
            'spawn_mode': str(self.spawn_mode),
            'display': self.display.to_dict() if self.display is not None else None,
            'question': self.question.to_dict() if self.question is not None else None,
            'claude_session_id': self.claude_session_id,
            'claude_launch_session_id': self.claude_launch_session_id,
            'claude_owner_pid': self.claude_owner_pid,
            'claude_owner_identity': _identity_wire(self.claude_owner_identity),
            'self_exit': self.self_exit,
            't3_end': None if self.t3_end is None else self.t3_end.value,
            'exit_request': self.exit_request.to_dict() if self.exit_request is not None else None,
            'exit_requested_at': self.exit_requested_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
        display_data = data.get('display')
        question_data = data.get('question')
        exit_requested_at = data.get('exit_requested_at')
        launcher_pid = data.get('launcher_pid', 0)
        claude_owner_pid = _coerce_owner_pid(data.get('claude_owner_pid'))
        return cls(
            session_slug=data['session_slug'],
            status=Status(data['status']),
            schema_version=data.get('schema_version', SCHEMA_VERSION),
            title=data.get('title', ''),
            role=data.get('role', ''),
            project=data.get('project', ''),
            task_id=data.get('task_id'),
            escalation_id=data.get('escalation_id'),
            prompt=data.get('prompt', ''),
            cwd=data.get('cwd', ''),
            launcher_pid=launcher_pid,
            launcher_identity=_coerce_identity(data.get('launcher_identity'), launcher_pid),
            start_ts=data.get('start_ts', ''),
            exit_code=data.get('exit_code'),
            result_file=data.get('result_file'),
            transcript_path=data.get('transcript_path'),
            parent_session_id=data.get('parent_session_id'),
            spawn_mode=data.get('spawn_mode', SpawnMode.CHILD),
            display=Display.from_dict(display_data) if isinstance(display_data, dict) else None,
            question=Question.from_dict(question_data) if isinstance(question_data, dict) else None,
            claude_session_id=_coerce_session_id(data.get('claude_session_id')),
            claude_launch_session_id=_coerce_session_id(data.get('claude_launch_session_id')),
            claude_owner_pid=claude_owner_pid,
            claude_owner_identity=_coerce_identity(
                data.get('claude_owner_identity'), claude_owner_pid
            ),
            self_exit=data.get('self_exit') is True,
            t3_end=_coerce_t3_end(data.get('t3_end')),
            exit_request=_coerce_exit_request(data.get('exit_request')),
            exit_requested_at=exit_requested_at if isinstance(exit_requested_at, str) else None,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> SessionRecord:
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Identity, paths, transcript encoding
# ---------------------------------------------------------------------------

_SLUG_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]')
_ALL_DOTS_RE = re.compile(r'^\.+$')


def sanitize_slug(raw: str) -> str:
    """Map any character outside ``[A-Za-z0-9._-]`` in *raw* to ``'-'``.

    The shared filesystem-safe-slug normalizer: ``build_session_slug`` calls
    this on its joined segments, and ``session_hooks.hook_session_slug``
    applies it to an externally-supplied slug token (``CLAUDE_SPAWN_SESSION_ID``)
    so that value is always safe to use as a ``record_path_for_slug`` directory
    name -- e.g. a stray ``/`` in an adversarial/malformed token can never
    resolve outside ``sessions_dir`` as a path separator.

    ``'.'`` is itself in the allowed set (kept for ordinary slugs), so the
    character-class substitution alone does not stop a result that is
    *entirely* dots (``'.'`` or ``'..'``): unlike any other sanitized value,
    that would be handed to ``record_path_for_slug`` as a filesystem
    traversal segment ("this directory" / "the parent directory") rather
    than a literal directory name. Such an all-dots result is therefore
    additionally collapsed (every ``'.'`` mapped to ``'-'``) so a ``'..'``
    (or ``'.'``) token can never escape ``sessions_dir`` either -- closing
    the one path-escape the regex substitution by itself cannot block. A
    string that merely *contains* dots alongside other characters (e.g.
    ``'..foo'``) is a literal, non-traversing directory name and is left
    untouched.
    """
    cleaned = _SLUG_SANITIZE_RE.sub('-', raw)
    # Deleting this branch is now caught: test_sanitize_slug_collapses_all_dots_slug
    # (orchestrator/tests/test_session_registry.py) pins it, mutation-verified under
    # task 4112 -- before that, removing it shipped green.
    if _ALL_DOTS_RE.match(cleaned):
        cleaned = cleaned.replace('.', '-')
    return cleaned


def build_session_slug(
    role: str,
    project: str,
    task_id: str | None,
    launcher_pid: int,
) -> str:
    """Build the record-identity/directory key ``<role>-<project>[-<taskid>]-<pid>``.

    The pid guarantees per-record uniqueness across concurrent spawns that
    share role+project+task (PRD §4 decision 2); single-ownership is a
    separate concern (T7 leases), not encoded in the key. Any character
    outside ``[A-Za-z0-9._-]`` in any segment (or the joined whole) is
    sanitized to ``'-'`` (see ``sanitize_slug``) so the slug is always
    filesystem-safe.
    """
    segments = [role, project]
    if task_id:
        segments.append(task_id)
    segments.append(str(launcher_pid))
    raw = '-'.join(segments)
    return sanitize_slug(raw)


def sessions_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'sessions'


def record_path_for_slug(slug: str, root: Path | str | None = None) -> Path:
    return sessions_dir(root) / slug / 'record.json'


SESSION_POINTERS_DIRNAME = 'sessions-by-pid'


def session_pointers_dir(root: Path | str | None = None) -> Path:
    """The pid-pointer dir, ``<fleet_root>/sessions-by-pid/``.

    A SIBLING of ``sessions_dir``, never a child of it: every sweep over
    ``sessions_dir`` treats each of its children as a record dir, so a
    pointer dir inside it would be read as a corrupt record.
    """
    return fleet_root(root) / SESSION_POINTERS_DIRNAME


def session_pointer_path_for_pid(pid: int, root: Path | str | None = None) -> Path:
    return session_pointers_dir(root) / str(pid)


def is_record_key(slug: str) -> bool:
    """True iff *slug* can name a record dir directly inside ``sessions_dir``.

    The one well-formedness check for every pointer and ``record_slug`` read
    or write: non-empty, already sanitized, and not an all-dots traversal
    segment.
    """
    return bool(slug) and sanitize_slug(slug) == slug and not _ALL_DOTS_RE.match(slug)


def _read_session_pointer(path: Path) -> str | None:
    """The slug a pointer file holds, stripped; None when it cannot be read."""
    try:
        return path.read_text(encoding='utf-8').strip()
    except (OSError, ValueError):
        return None


def _write_pointer_file(path: Path, slug: str) -> bool:
    """Make the pointer file *path* hold *slug*: the shared body of every pointer writer.

    True without rewriting when it already holds *slug*; False after a
    WARNING when the write fails. Never raises.
    """
    if _read_session_pointer(path) == slug:
        return True
    try:
        atomic_write_text(path, slug)
    except OSError as exc:
        logger.warning('session pointer write failed: slug=%s path=%s: %s', slug, path, exc)
        return False
    return True


def write_session_pointer(pid: int, slug: str, root: Path | str | None = None) -> bool:
    """Record *slug* as the record key of claude process *pid*'s current record.

    The file ``sessions-by-pid/<pid>`` holds exactly that slug, last writer
    wins. *pid* is the OWNING ``claude`` process's pid (the value
    ``$CLAUDE_PID`` carries), never the record's ``launcher_pid``.

    Returns True without rewriting when the pointer already holds *slug*.
    Returns False without writing for a non-positive *pid* or a *slug* that
    is not a record key, and False (after a WARNING) when the write itself
    fails. Never raises: its callers are session hooks.
    """
    if pid <= 0 or not is_record_key(slug):
        return False
    return _write_pointer_file(session_pointer_path_for_pid(pid, root=root), slug)


def resolve_session_slug_for_pid(pid: int, root: Path | str | None = None) -> str | None:
    """Return the record slug of claude process *pid*'s current record, or None.

    The pointer's slug is returned only when it is a record key AND the
    record it names still carries ``claude_owner_pid == pid``: the record
    must vouch for the pointer, so an unowned record, another pid's record,
    a reaped one and an unreadable one all read as None. None is the
    caller-visible degradation signal; this function is read-only and never
    raises.

    Contract limit: a new process that reuses a dead claude's pid can read
    that session's pointer until its own SessionStart hook overwrites it or
    ``reap_stale_session_pointers`` removes it.
    """
    if pid <= 0:
        return None
    slug = _read_session_pointer(session_pointer_path_for_pid(pid, root=root))
    if slug is None or not is_record_key(slug):
        return None
    try:
        record = read_record(slug, root=root)
    except (OSError, CorruptSessionRecord):
        return None
    return slug if record.claude_owner_pid == pid else None


CLAUDE_SESSION_POINTERS_DIRNAME = 'sessions-by-claude-id'


def claude_session_pointers_dir(root: Path | str | None = None) -> Path:
    """The session-id pointer dir, ``<fleet_root>/sessions-by-claude-id/``; a sibling of ``sessions_dir`` like ``session_pointers_dir``."""
    return fleet_root(root) / CLAUDE_SESSION_POINTERS_DIRNAME


def claude_session_pointer_path(session_id: str, root: Path | str | None = None) -> Path:
    return claude_session_pointers_dir(root) / session_id


def _is_claude_session_pointer_name(name: str) -> bool:
    """True iff *name* can name a ``sessions-by-claude-id/`` entry: the ``CLAUDE_CODE_SESSION_ID`` token shape.

    The ``sessions-by-t3-thread/`` twin is ``is_t3_thread_pointer_name``.
    """
    return _SESSION_ID_RE.fullmatch(name) is not None


_T3_THREAD_ID_RE = re.compile(r'[A-Za-z0-9:-]{1,128}')


def is_t3_thread_pointer_name(thread_id: str) -> bool:
    """True iff *thread_id* can name a ``sessions-by-t3-thread/`` entry.

    T3 thread ids are a UI thread's bare UUID or an MCP-started thread's
    ``mcp:<uuid>`` / ``mcp:launch:<hex>``; the class admits ``:`` and still
    excludes ``/``, ``.`` and NUL.
    """
    return _T3_THREAD_ID_RE.fullmatch(thread_id) is not None


def write_claude_session_pointer(
    session_id: str, slug: str, root: Path | str | None = None
) -> bool:
    """Record *slug* as the record key of the Claude Code session *session_id*.

    The session-id twin of ``write_session_pointer``, with the same contract:
    last writer wins, True without rewriting when the pointer already holds
    *slug*, False without writing for an unsafe *session_id* or a *slug* that
    is not a record key, False after a WARNING when the write fails, and
    never raises.
    """
    if not _is_claude_session_pointer_name(session_id) or not is_record_key(slug):
        return False
    return _write_pointer_file(claude_session_pointer_path(session_id, root=root), slug)


def resolve_session_slug_for_claude_session_id(
    session_id: str, root: Path | str | None = None
) -> str | None:
    """Return the record slug the Claude Code session *session_id* is bound to, or None.

    Vouched like ``resolve_session_slug_for_pid``: the pointer's slug is
    returned only when the record it names carries
    ``claude_session_id == session_id``. Read-only; never raises.
    """
    if not _is_claude_session_pointer_name(session_id):
        return None
    slug = _read_session_pointer(claude_session_pointer_path(session_id, root=root))
    if slug is None or not is_record_key(slug):
        return None
    try:
        record = read_record(slug, root=root)
    except (OSError, CorruptSessionRecord):
        return None
    return slug if record.claude_session_id == session_id else None


T3_THREAD_POINTERS_DIRNAME = 'sessions-by-t3-thread'


def t3_thread_pointers_dir(root: Path | str | None = None) -> Path:
    """The T3 thread pointer dir, ``<fleet_root>/sessions-by-t3-thread/``; a sibling of ``sessions_dir`` like ``session_pointers_dir``.

    It is the index of live T3-hosted records that ``mark_t3_sessions_exited`` iterates.
    """
    return fleet_root(root) / T3_THREAD_POINTERS_DIRNAME


def t3_thread_pointer_path(thread_id: str, root: Path | str | None = None) -> Path:
    return t3_thread_pointers_dir(root) / thread_id


def write_t3_thread_pointer(thread_id: str, slug: str, root: Path | str | None = None) -> bool:
    """Record *slug* as the record key of the session T3 thread *thread_id* hosts.

    The thread twin of ``write_claude_session_pointer``, with the same contract.
    """
    if not is_t3_thread_pointer_name(thread_id) or not is_record_key(slug):
        return False
    return _write_pointer_file(t3_thread_pointer_path(thread_id, root=root), slug)


def resolve_session_slug_for_t3_thread(
    thread_id: str, root: Path | str | None = None
) -> str | None:
    """Return the slug of the live record T3 thread *thread_id* hosts, or None.

    Vouched more strictly than the other pointers: the named record must be
    non-terminal and ``t3_thread_of`` it must be *thread_id*. That is the one
    definition of the index's invariant -- a terminal record has no pointer
    -- which the reap and every terminal write enforce through
    ``_is_stale_t3_thread_pointer``. Read-only; never raises.
    """
    if not is_t3_thread_pointer_name(thread_id):
        return None
    slug = _read_session_pointer(t3_thread_pointer_path(thread_id, root=root))
    if slug is None or not is_record_key(slug):
        return None
    try:
        record = read_record(slug, root=root)
    except (OSError, CorruptSessionRecord):
        return None
    vouches = t3_thread_of(record) == thread_id and record.status not in TERMINAL_STATUSES
    return slug if vouches else None


def encode_cwd(cwd: str) -> str:
    """Encode *cwd* to Claude Code's own ``~/.claude/projects/<enc>`` dir name.

    THE authoritative statement of the rule; every other copy in this repo
    is a mirror of this one and is pinned to it by
    ``scripts/tests/test_legibility_inventory.py``'s ``TestEncoderLockstep``.

    ``/``, ``.`` and ``_`` all map to ``-``. ``-`` passes through unchanged.
    CASE IS PRESERVED — there is no case-folding step (a live dir named
    ``-opt-Auto-Claude-resources-backend`` proves it).

    Validated against 738 real (encoded-dir, decoded-cwd) pairs sampled from
    a live ``~/.claude/projects`` tree, which the rule reproduces exactly.
    Over that sample the only non-alphanumeric characters appearing in ANY
    cwd were ``- . / _``, so the rule is complete over the observed domain
    and UNVERIFIED outside it — a cwd containing some other punctuation
    character has never been measured, and this function's output for one is
    a guess. Extend the sample before extending the rule (task 3272).

    Returns the BARE dir name, no ``~/.claude/projects/`` prefix — see
    :func:`transcript_path_for_cwd` for the display-path form.
    """
    return cwd.replace('/', '-').replace('.', '-').replace('_', '-')


def transcript_path_for_cwd(cwd: str) -> str:
    """Best-effort mirror of Claude Code's own ``~/.claude/projects/<enc>`` path.

    Delegates the encoding to :func:`encode_cwd` (``/``, ``.`` and ``_`` all
    map to ``-``; case preserved), which carries the authoritative statement
    of the rule and the record of what it was measured against — 738 real
    (encoded-dir, decoded-cwd) pairs from a live ``~/.claude/projects`` tree.

    This is read-only enrichment metadata — never used as a lookup key by
    this module. Note that OTHER modules do use the encoding as a lookup key
    (``scripts/legibility/check_transcript_persistence.py``), so its exactness
    is load-bearing beyond this file.
    """
    return f'~/.claude/projects/{encode_cwd(cwd)}'


# ---------------------------------------------------------------------------
# Single-writer atomic write / read / update
# ---------------------------------------------------------------------------


class CorruptSessionRecord(Exception):
    """Raised by read_record when a record.json exists but fails to parse."""


def write_record(record: SessionRecord, root: Path | str | None = None) -> None:
    """Atomically write *record* (tmp file in the same dir, then os.replace).

    Every successful write bumps record.json's mtime, which
    reap_stale_records() reads as this record's heartbeat.
    """
    path = record_path_for_slug(record.session_slug, root=root)
    atomic_write_text(path, record.to_json())


def read_record(slug: str, root: Path | str | None = None) -> SessionRecord:
    """Read the record for *slug*.

    Raises ``FileNotFoundError`` if no record.json exists at this key, or
    ``CorruptSessionRecord`` if one exists but fails to parse -- so callers
    can tell "no record yet" apart from "record present but unreadable".
    """
    path = record_path_for_slug(slug, root=root)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        return SessionRecord.from_json(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CorruptSessionRecord(f'unparseable session record at {path}') from exc


def update_status(
    slug: str,
    root: Path | str | None = None,
    *,
    status: Status,
    exit_code: int | None = None,
) -> SessionRecord:
    """Strict read-modify-write of *status* (and optionally *exit_code*).

    Raises if no record exists for *slug* -- used by the ``exit`` CLI
    subcommand (spawn-claude.sh's ``finish()``), where a missing record
    signals a genuine bug: the ``launching`` write should already have
    created it moments earlier.
    """
    record = read_record(slug, root=root)
    record.status = status
    if exit_code is not None:
        record.exit_code = exit_code
    write_record(record, root=root)
    return record


def update_display(
    slug: str,
    root: Path | str | None = None,
    *,
    kind: str,
    wm_title: str = '',
    wm_window_id: str | None = None,
    tmux_target: str | None = None,
    t3_thread_id: str | None = None,
) -> SessionRecord:
    """Strict read-modify-write of *display* (mirrors update_status).

    Used by the ``set-display`` CLI subcommand (spawn-claude.sh's tmux lane,
    Fleet Cockpit C6), which only learns the tmux target after `tmux
    new-window` runs -- well after ``launching`` allocated the record with
    display=None. Raises if no record exists for *slug*, matching
    update_status's contract; preserves every other field (incl.
    status/exit_code/result_file), so a later `exit`/`refresh`
    read-modify-write sees the stamped display unchanged.
    """
    record = read_record(slug, root=root)
    record.display = Display(
        kind=kind,
        wm_title=wm_title,
        wm_window_id=wm_window_id,
        tmux_target=tmux_target,
        t3_thread_id=t3_thread_id,
    )
    write_record(record, root=root)
    return record


def refresh_record(
    slug: str,
    root: Path | str | None = None,
    *,
    status: Status | None = None,
) -> SessionRecord:
    """Upsert used by the T6 hook seam; bumps the mtime heartbeat (PRD G5).

    Updates the existing record for *slug* in place when one exists (a
    read-modify-write, like update_status). When none exists -- the T6
    hand-launched-capture path, where a session was started with no prior
    spawn-claude.sh write -- creates a new, well-formed record under the
    SAME key: schema_version/session_slug/start_ts/status are populated;
    every other field is left at its documented ``SessionRecord`` default.
    This is the write<->refresh boundary contract every downstream
    Attention Rail task (T4/T5/T6/T7) relies on resolving to one identical
    ``<root>/sessions/<slug>/record.json`` path.

    A *corrupt* existing body is NOT treated as absent -- it continues to
    raise ``CorruptSessionRecord`` rather than silently overwriting data the
    reaper's own 'corrupt' rule already accounts for.

    The upsert body itself lives in ``apply_refresh`` (the pure half); this
    function is read -> apply_refresh -> write, so there is exactly ONE
    definition of what a freshly-upserted record looks like.
    ``session_hooks.py::_run_status_refresh_and_retitle`` is the caller that
    needs those semantics WITHOUT a second read -- it already holds the
    record, from the ownership probe's own read -- and composes
    ``apply_refresh`` with a single ``write_record`` instead of calling this.
    It still calls this on its unreadable-body fault lane, precisely for the
    corrupt-propagation guarantee above.

    NON-GOAL, deliberate: this is not a compare-and-swap. ``write_record``
    is an atomic whole-body replace with no CAS, so registry writes are
    last-writer-wins here and everywhere else.
    """
    try:
        prior = read_record(slug, root=root)
    except FileNotFoundError:
        prior = None
    record = apply_refresh(slug, prior, status=status)
    write_record(record, root=root)
    return record


def apply_refresh(
    slug: str,
    prior: SessionRecord | None,
    *,
    status: Status | None = None,
) -> SessionRecord:
    """PURE (no-I/O) half of ``refresh_record``: the upsert body alone.

    Performs NO filesystem access whatsoever -- it neither reads nor writes.
    *prior* is the already-read record for *slug*, or None when no record
    exists at that key yet.

    When *prior* is not None it is MUTATED AND RETURNED IN PLACE (the same
    read-modify contract ``refresh_record``/``update_status`` have always
    had), so a caller holding the object sees the applied status. When
    *prior* is None a fresh, well-formed ``SessionRecord`` is synthesized
    under the same key: schema_version/session_slug/start_ts/status are
    populated and every other field is left at its documented default.

    Why this exists as its own function:
    ``session_hooks.py::_run_status_refresh_and_retitle`` already holds the
    record (the ownership probe read it) and must settle one hook event on
    ONE snapshot and ONE write -- calling ``refresh_record`` would force a
    redundant re-read and a second write. It cannot instead re-derive the
    upsert inline: the module docstring's PRD Section 6 G5 rule is that
    consumers import the shared record contract and never re-derive it, so
    there must stay exactly one definition of what a freshly-upserted
    record looks like. This is that definition.

    Note *prior* being None is the ABSENT state only. A record that exists
    but is unreadable must NOT be routed here as None -- see
    ``refresh_record``'s corrupt-body guarantee, which callers preserve by
    letting ``CorruptSessionRecord`` propagate rather than synthesizing over
    the body.
    """
    if prior is None:
        # No prior write for this slug: synthesize a fresh record. LAUNCHING
        # is the sensible default identity for "a record just came into
        # being" when the caller upserts without an explicit status.
        return SessionRecord(
            session_slug=slug,
            status=status if status is not None else Status.LAUNCHING,
            start_ts=datetime.now(UTC).isoformat(),
        )
    if status is not None:
        prior.status = status
    return prior


# ---------------------------------------------------------------------------
# TTL / pid stale-record reaper (PRD §4.8)
# ---------------------------------------------------------------------------

TERMINAL_TTL = timedelta(hours=24)
"""How long a terminal (EXITED/FAILED_TO_START) record survives after its
last write before the reaper reclaims it, regardless of launcher_pid."""

NON_TERMINAL_HEARTBEAT_TTL = timedelta(hours=1)
"""How long a non-terminal record survives with its launcher judged DEAD and
no fresh write (heartbeat) before the reaper reclaims it. A launcher not
judged DEAD retains the record regardless of age."""


def _swept_as_dead(
    identity: ProcessIdentity | None, pid: int, *, sweep: str, subject: str
) -> bool:
    """Whether a liveness sweep may treat *pid* as dead: ``judge``'s DEAD, with an OTHER_HOST skip logged."""
    verdict = judge(identity, pid, pid_alive=pid_alive)
    if verdict is Liveness.OTHER_HOST and identity is not None:
        logger.info('%s: skipped %s, reason=other_host host=%s', sweep, subject, identity.host)
    return verdict is Liveness.DEAD


@dataclass(frozen=True)
class ReapedSessionRecord:
    """One session-registry record directory removed by reap_stale_records.

    path: the removed ``<slug>/`` directory (not the record.json file).
    session_slug: the session's identity/directory key, derived from the
        directory name itself (never from the record body).
    reason: why it was reaped -- one of ``'terminal_ttl'`` (a terminal
        record older than TERMINAL_TTL), ``'stale_pid'`` (a non-terminal
        record whose launcher is judged DEAD and older than
        NON_TERMINAL_HEARTBEAT_TTL), or ``'corrupt'`` (record.json missing
        or unparseable and older than NON_TERMINAL_HEARTBEAT_TTL).
    """

    path: Path
    session_slug: str
    reason: str


def reap_stale_records(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[ReapedSessionRecord]:
    """Sweep ``<root>/sessions/*/`` and remove stale session-record directories.

    Mirrors the ``reap_interactive_worktrees``/``interactive.json`` idiom
    (git_ops.py): identity (which directory to remove) is derived from the
    *path* -- the slug directory name -- never from the record body, so a
    missing or corrupt record.json is still reapable. Age is measured from
    record.json's mtime (bumped by every write_record/refresh_record call,
    i.e. its heartbeat); when record.json itself is absent, the slug
    directory's own mtime is used instead so identity-only reap still works.

    Rules (first match wins):
    - record body unreadable (missing or corrupt) and age >
      NON_TERMINAL_HEARTBEAT_TTL -> reaped, reason='corrupt'.
    - status is terminal (EXITED/FAILED_TO_START) and age > TERMINAL_TTL ->
      reaped, reason='terminal_ttl' (independent of launcher_pid liveness).
    - status is non-terminal, age > NON_TERMINAL_HEARTBEAT_TTL, and the
      launcher is judged DEAD by
      ``shared/src/shared/session_liveness.py::judge`` -> reaped,
      reason='stale_pid'. A T3-hosted
      record (``display.kind == 't3'``) is exempt while
      ``_t3_thread_sweep_owns`` holds, since its liveness is its thread.
    - otherwise -> kept.

    Directories are visited in ``sorted(iterdir())`` order (directory-name
    order, not age order).

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock.
    """
    if now is None:
        now = datetime.now(UTC)
    base = sessions_dir(root)
    reaped: list[ReapedSessionRecord] = []
    if not base.is_dir():
        return reaped

    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        record_path = slug_dir / 'record.json'
        age_source = record_path if record_path.is_file() else slug_dir
        try:
            mtime = age_source.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age = now - datetime.fromtimestamp(mtime, tz=UTC)

        try:
            record = read_record(slug, root=root)
        except (FileNotFoundError, CorruptSessionRecord):
            reason = 'corrupt' if age > NON_TERMINAL_HEARTBEAT_TTL else None
        else:
            if record.status in TERMINAL_STATUSES:
                reason = 'terminal_ttl' if age > TERMINAL_TTL else None
            else:
                stale = (
                    not _t3_thread_sweep_owns(record, slug, root)
                    and age > NON_TERMINAL_HEARTBEAT_TTL
                    and _swept_as_dead(
                        record.launcher_identity,
                        record.launcher_pid,
                        sweep='reap_stale_records',
                        subject=slug,
                    )
                )
                reason = 'stale_pid' if stale else None

        if reason is not None:
            try:
                shutil.rmtree(slug_dir)
            except OSError:
                # A single unreapable directory (permission error, a file held
                # open, an ENOTEMPTY race with a concurrent writer) must not
                # abort the sweep -- log and move on to the next candidate.
                logger.error('reap_stale_records: failed to remove %s', slug_dir, exc_info=True)
                continue
            reaped.append(ReapedSessionRecord(path=slug_dir, session_slug=slug, reason=reason))

    return reaped


_POINTER_TEMP_FILE_GRACE = timedelta(minutes=5)
"""How long a ``sessions-by-pid/`` temp file is presumed to belong to a writer
still between ``mkstemp`` and ``os.replace``. Past it, the writer was killed."""


def _is_abandoned_temp_file(path: Path) -> bool:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    age = datetime.now(UTC) - datetime.fromtimestamp(mtime, tz=UTC)
    return age > _POINTER_TEMP_FILE_GRACE


def _is_stale_session_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    try:
        pid = int(path.name)
    except ValueError:
        return True
    if pid <= 0 or path.name != str(pid):
        return True
    slug = _read_session_pointer(path)
    if (
        slug is None
        or not is_record_key(slug)
        or not record_path_for_slug(slug, root=root).is_file()
    ):
        return True
    return _swept_as_dead(
        _pointed_owner_identity(slug, pid, root),
        pid,
        sweep='reap_stale_session_pointers',
        subject=path.name,
    )


def _pointed_owner_identity(
    slug: str, pid: int, root: Path | str | None
) -> ProcessIdentity | None:
    """The owner identity *slug*'s record holds for *pid*; None when unreadable or naming another pid."""
    try:
        identity = read_record(slug, root=root).claude_owner_identity
    except (CorruptSessionRecord, OSError):
        return None
    return identity if identity is not None and identity.pid == pid else None


def _is_stale_claude_session_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    if not _is_claude_session_pointer_name(path.name):
        return True
    return resolve_session_slug_for_claude_session_id(path.name, root=root) is None


def _is_stale_t3_thread_pointer(path: Path, root: Path | str | None) -> bool:
    if path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX):
        return _is_abandoned_temp_file(path)
    if not is_t3_thread_pointer_name(path.name):
        return True
    return resolve_session_slug_for_t3_thread(path.name, root=root) is None


def _drop_t3_thread_pointer_if_stale(record: SessionRecord, root: Path | str | None) -> None:
    """After a terminal write of *record*, unlink its thread's pointer unless a live record holds it.

    A failed unlink is logged, never raised: the reap retries it.
    """
    thread_id = t3_thread_of(record)
    if thread_id is None:
        return
    path = t3_thread_pointer_path(thread_id, root=root)
    if not _is_stale_t3_thread_pointer(path, root):
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('t3 thread pointer drop failed: path=%s: %s', path, exc)


def _reap_pointer_dir(
    base: Path, is_stale: Callable[[Path, Path | str | None], bool], root: Path | str | None
) -> list[Path]:
    if not base.is_dir():
        return []
    entries = sorted(base.iterdir())
    removed: list[Path] = []
    for path in entries:
        if not is_stale(path, root):
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning('reap_stale_session_pointers: failed to remove %s: %s', path, exc)
            continue
        removed.append(path)
    logger.info(
        'reap_stale_session_pointers: removed %d of %d pointer(s) under %s',
        len(removed),
        len(entries),
        base,
    )
    return removed


def reap_stale_session_pointers(root: Path | str | None = None) -> list[Path]:
    """Remove every pointer entry that can no longer serve a reader.

    A ``sessions-by-pid/`` entry is stale when its name is not a canonical
    positive pid, its content is not a record key, the record it names has no
    ``record.json``, or its pid is judged DEAD by
    ``shared/src/shared/session_liveness.py::judge`` against that record's
    ``claude_owner_identity`` when it names the same pid, so each such entry
    costs one record read.
    A dead pid's pointer goes even while its record still exists: its owner
    can never query it again, and a process that reuses the pid must not
    inherit it.

    A ``sessions-by-claude-id/`` entry is stale when its name is not a
    session-id token or the record it names no longer vouches for it (gone,
    unreadable, or bound to another session id), so each such entry costs one
    record read.

    A ``sessions-by-t3-thread/`` entry is stale when its name is not a token
    or ``resolve_session_slug_for_t3_thread`` finds no live t3 record of that
    thread behind it, so each such entry costs one record read.

    In every dir a temp file of ``fleet_files.py::atomic_write_text`` is left to its writer
    until ``_POINTER_TEMP_FILE_GRACE`` has passed, so a sweep never unlinks it
    out from under a concurrent ``os.replace``.

    This is a separate sweep from ``reap_stale_records`` because that one
    derives identity from each record's PATH and never reads the body, so it
    cannot know which pointers name the records it removes. Returns the
    removed paths; an entry that cannot be removed is logged and skipped.
    """
    return [
        *_reap_pointer_dir(session_pointers_dir(root), _is_stale_session_pointer, root),
        *_reap_pointer_dir(
            claude_session_pointers_dir(root), _is_stale_claude_session_pointer, root
        ),
        *_reap_pointer_dir(t3_thread_pointers_dir(root), _is_stale_t3_thread_pointer, root),
    ]


ORPHAN_EXIT_CODE: int = 200
"""Synthetic exit code stamped by ``mark_orphaned_sessions_exited`` when a
non-terminal session's ``launcher_pid`` is provably dead but no real exit
code was ever recorded -- ``finish()`` never ran (SIGKILL, force-closed
terminal, crash, reboot). Distinct from every real/sentinel code
spawn-claude.sh can record (0-125 claude exit codes; 126/127/129/143/144
spawn-claude.sh sentinels; 145-199 called out as reserved-free by
spawn-claude.sh's own exit-codes comment) and from ``None`` (no exit_code
recorded yet), so a downstream consumer can tell an orphan-reaped record
from a cleanly-exited one. Deliberately a POSITIVE value at/past 200, not
-1: by POSIX/subprocess convention a NEGATIVE returncode means "killed by
signal N" (``returncode == -N``), so -1 could be misread by a consumer
applying that convention as "killed by SIGHUP". A positive sentinel clear
of every reserved range avoids that ambiguity."""


def _mark_exited_if_still_non_terminal(
    slug: str,
    root: Path | str | None,
    *,
    exit_code: int,
    still_applies: Callable[[SessionRecord], bool] | None = None,
) -> SessionRecord | None:
    """TOCTOU-safe terminal stamp used by the liveness sweeps and ``mark_session_folded``.

    Re-reads *slug* itself -- a second, fresher read than the sweep's own
    caller-side liveness check -- and, ONLY if status is still non-terminal
    on THIS read and *still_applies* (when given) accepts THIS read, stamps
    status=EXITED/exit_code=*exit_code* and writes. Returns the marked
    record, or ``None`` if a concurrent writer (most plausibly
    spawn-claude.sh's own ``finish()``, or a hook re-binding the record)
    already changed it in the window between the sweep's own liveness check
    and this call -- in which case nothing is written, so that writer's
    real state is never clobbered by the orphan sentinel.

    Deliberately a narrow variant rather than a change to ``update_status``
    itself: ``update_status``'s own ``exit`` CLI subcommand caller is an
    authoritative, unconditional final report from the exiting process and
    must stay unconditional; only this sweep's best-effort guess needs to
    defer to a fresher real write.
    """
    record = read_record(slug, root=root)
    if record.status in TERMINAL_STATUSES:
        return None
    if still_applies is not None and not still_applies(record):
        return None
    record.status = Status.EXITED
    record.exit_code = exit_code
    write_record(record, root=root)
    _drop_t3_thread_pointer_if_stale(record, root)
    return record


def mark_orphaned_sessions_exited(
    root: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> list[SessionRecord]:
    """Sweep ``<root>/sessions/*/`` and mark provably-orphaned records EXITED.

    This is the liveness sweep that fixes the F2 staleness backlog: record
    CONVERGENCE (session_hooks' CLAUDE_SPAWN_SESSION_ID adoption) only
    advances CLEANLY-exiting spawned sessions through their hook-driven
    launching -> running -> idle/awaiting-input -> exited lifecycle. The
    backlog is overwhelmingly UNCLEAN deaths (SIGKILL, a force-closed
    terminal, a crash, a reboot) where spawn-claude.sh's own ``finish()``
    never runs, so ``exited`` is never written -- only this sweep ever marks
    those records terminal.

    Mirrors ``reap_stale_records``'s ``stale_pid`` predicate EXACTLY --
    status is non-terminal (not in TERMINAL_STATUSES), the launcher is judged
    DEAD (``shared/src/shared/session_liveness.py::judge``), and age
    (record.json's mtime vs *now*) exceeds
    ``NON_TERMINAL_HEARTBEAT_TTL`` -- the same false-positive guard that
    rule already trusts: a genuinely-live session (including a
    detached/tmux one) keeps bumping its record's mtime via its own hooks,
    so it is never swept regardless of how long it has been non-terminal.

    Known trade-off for DETACHED spawns (reviewer-flagged): spawn-claude.sh
    itself exits once it hands off (its ``launcher_pid`` dies) while claude
    keeps running standalone, so for the entire remaining life of a
    detached session the launcher is already judged DEAD -- the heartbeat
    (record.json's mtime, bumped only by that session's own
    Notification/Stop hooks after convergence) is the SOLE thing keeping it
    unswept. A detached session that goes quieter than
    ``NON_TERMINAL_HEARTBEAT_TTL`` (1h) between hook events -- e.g. one long
    tool-heavy turn with no intervening Notification/Stop -- can therefore
    be marked EXITED while genuinely still running, by the reap every other
    spawn's ``_run_launching`` starts, not just by an explicit CLI ``reap``.
    This is self-correcting
    -- its next hook event calls ``refresh_record`` and flips status back --
    but there is a real window of an incorrectly-EXITED cockpit row. Reusing
    the pre-existing ``NON_TERMINAL_HEARTBEAT_TTL`` (rather than introducing
    a second, sweep-specific threshold) keeps this sweep's semantics
    identical to the reaper's already-trusted ``stale_pid`` rule; widening
    the TTL or tracking a detached session's actual claude pid instead of
    its exited launcher would be a separate, broader design change (touching
    session_hooks.py/spawn-claude.sh's pid bookkeeping) and is left as a
    follow-up rather than folded into this sweep.

    Unlike ``reap_stale_records``, this does NOT delete anything -- it marks
    a matching record EXITED (exit_code=``ORPHAN_EXIT_CODE``), preserving
    every other field, and leaves its directory in place.
    ``reap_stale_records``'s existing ``terminal_ttl`` rule reclaims it
    later, unchanged. An already-terminal record is left fully untouched
    (idempotent -- never re-stamped with the sentinel), and a record whose
    body is missing or corrupt is skipped here -- that is
    ``reap_stale_records``'s own ``'corrupt'`` rule's concern, not this
    sweep's.

    The final stamp itself goes through ``_mark_exited_if_still_non_terminal``,
    which re-reads the record ONE more time immediately before writing: if a
    concurrent writer (most plausibly ``finish()``) already made it terminal
    in the window since this sweep's own liveness check above, the sweep
    backs off instead of clobbering that writer's real exit_code with the
    orphan sentinel.

    A per-record try/except means one bad record (a corrupt body, a
    concurrent-reap race vanishing the file mid-sweep, an unwritable
    record) never aborts the sweep -- it is logged and skipped, exactly
    like ``reap_stale_records``'s own fault handling.

    A T3-hosted record (``display.kind == 't3'``) is not swept here while
    ``_t3_thread_sweep_owns`` holds: its ``launcher_pid`` is the spawn
    script, which exits in sibling mode, and its hooks fall silent whenever
    the thread waits on a human. ``mark_t3_sessions_exited`` sweeps it by
    thread instead.

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock. Returns the list of records this call marked (their post-mark,
    now-EXITED state).
    """
    if now is None:
        now = datetime.now(UTC)
    base = sessions_dir(root)
    marked: list[SessionRecord] = []
    if not base.is_dir():
        return marked

    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name

        try:
            record = read_record(slug, root=root)
        except (FileNotFoundError, CorruptSessionRecord):
            continue  # missing/corrupt body -- reap_stale_records' 'corrupt' rule's job
        if record.status in TERMINAL_STATUSES:
            continue  # already terminal -- idempotent, never re-stamped
        if _t3_thread_sweep_owns(record, slug, root):
            continue
        if not _swept_as_dead(
            record.launcher_identity,
            record.launcher_pid,
            sweep='mark_orphaned_sessions_exited',
            subject=slug,
        ):
            continue  # a launcher not judged dead is never swept, regardless of age

        record_path = slug_dir / 'record.json'
        try:
            mtime = record_path.stat().st_mtime
        except OSError:
            continue  # vanished mid-sweep (e.g. concurrent reap) -- skip it
        age = now - datetime.fromtimestamp(mtime, tz=UTC)
        if age <= NON_TERMINAL_HEARTBEAT_TTL:
            continue  # heartbeat grace -- not stale enough yet

        try:
            marked_record = _mark_exited_if_still_non_terminal(
                slug, root, exit_code=ORPHAN_EXIT_CODE
            )
        except (FileNotFoundError, CorruptSessionRecord, OSError):
            # A single unmarkable record (a concurrent-reap race, an
            # unwritable record) must not abort the sweep -- log and move on
            # to the next candidate.
            logger.error(
                'mark_orphaned_sessions_exited: failed to mark %s', slug, exc_info=True
            )
            continue
        if marked_record is None:
            continue  # lost the race: a concurrent writer already made this terminal
        marked.append(marked_record)

    return marked


def is_t3_record(record: SessionRecord) -> bool:
    """Whether *record* is hosted by a T3 Code thread (``display.kind == 't3'``)."""
    return record.display is not None and record.display.kind == DisplayKind.T3


def t3_thread_of(record: SessionRecord) -> str | None:
    """The T3 thread *record* is a t3 record of, when that id can name a thread pointer; else None.

    The one definition shared by the thread pointer's writers and by its
    vouching check, ``resolve_session_slug_for_t3_thread``.
    """
    if not is_t3_record(record) or record.display is None:
        return None
    thread_id = record.display.t3_thread_id
    return thread_id if thread_id and is_t3_thread_pointer_name(thread_id) else None


def index_t3_record_if_unheld(
    record: SessionRecord, slug: str, root: Path | str | None = None
) -> bool:
    """Write *slug* as the thread pointer of live t3 *record* when no live record holds its thread; whether it wrote.

    A held pointer is never moved: the T3 lane's bind
    (``orchestrator/src/orchestrator/spawn_t3_lane.py::_bind_to``) is the one
    writer that re-points one. A failed write is logged by the writer, and
    ``index_unpointed_t3_records`` retries it.
    """
    thread_id = t3_thread_of(record)
    if (
        thread_id is None
        or record.status in TERMINAL_STATUSES
        or resolve_session_slug_for_t3_thread(thread_id, root=root) is not None
    ):
        return False
    return write_t3_thread_pointer(thread_id, slug, root=root)


def index_unpointed_t3_records(root: Path | str | None = None) -> list[str]:
    """Pass every record under ``<root>/sessions/`` to ``index_t3_record_if_unheld``; returns the slugs indexed.

    ``run_reap``'s first phase. A live t3 record whose thread no pointer
    holds -- one written before the index existed, or whose pointer write
    failed -- joins the index ``mark_t3_sessions_exited`` iterates only here.
    Until then ``_t3_thread_sweep_owns`` keeps it out of the pid sweeps. An
    unreadable record is skipped, never fatal.
    """
    base = sessions_dir(root)
    if not base.is_dir():
        return []
    indexed: list[str] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        try:
            record = read_record(slug_dir.name, root=root)
        except (OSError, CorruptSessionRecord):
            continue
        if index_t3_record_if_unheld(record, slug_dir.name, root=root):
            indexed.append(slug_dir.name)
    return indexed


def _t3_thread_sweep_owns(record: SessionRecord, slug: str, root: Path | str | None) -> bool:
    """Whether ``mark_t3_sessions_exited``, not a pid sweep, owns non-terminal *record*. Read-only.

    True for a t3 record whose thread pointer names *slug* or no live record.
    An unheld thread stays owned, so a record ``index_unpointed_t3_records``
    has not indexed yet is never handed to the pid sweeps. Only displacement
    by another live record, or a thread id that cannot name a pointer, does.
    """
    thread_id = t3_thread_of(record)
    if thread_id is None:
        return False
    held_by = resolve_session_slug_for_t3_thread(thread_id, root=root)
    return held_by is None or held_by == slug


def mark_t3_sessions_exited(
    environment_id: str,
    listed_threads: AbstractSet[str],
    root: Path | str | None = None,
    *,
    listed_at: datetime | None = None,
    thread_ended: Callable[[str], bool],
    routine: bool = False,
) -> list[str]:
    """Mark EXITED every indexed T3-hosted record of *environment_id* whose thread has gone.

    *listed_threads* is one complete listing of the environment's active
    threads. Only ``sessions-by-t3-thread/`` is read: a record behind a
    vouched entry whose thread is unlisted and whose ``t3_environment_id`` is
    *environment_id* or unset is marked (exit_code ``ORPHAN_EXIT_CODE``),
    which drops its pointer, once *thread_ended* confirms that thread ended:
    a listing omits live side chats. With *listed_at*, a record written after
    that instant is skipped: the listing predates it and cannot vouch for its
    thread's absence.

    Returns the marked slugs. One unreadable or unwritable record is logged
    and skipped, never fatal. One line reports the index size, at INFO
    unless *routine* -- one pass of a wait loop -- marked nothing, then at
    DEBUG.
    """
    base = t3_thread_pointers_dir(root)
    entries = sorted(base.iterdir()) if base.is_dir() else []
    thread_ids = [path.name for path in entries if not path.name.endswith(ATOMIC_WRITE_TEMP_SUFFIX)]
    marked: list[str] = []
    for thread_id in thread_ids:
        if thread_id in listed_threads:
            continue
        slug = _mark_t3_session_gone(thread_id, environment_id, root, listed_at, thread_ended)
        if slug is not None:
            marked.append(slug)
    logger.log(
        logging.DEBUG if routine and not marked else logging.INFO,
        'mark_t3_sessions_exited: marked %d of %d indexed session(s) under %s',
        len(marked),
        len(thread_ids),
        base,
    )
    return marked


def _mark_t3_session_gone(
    thread_id: str,
    environment_id: str,
    root: Path | str | None,
    listed_at: datetime | None,
    thread_ended: Callable[[str], bool],
) -> str | None:
    """Mark the live record of unlisted thread *thread_id* EXITED once *thread_ended* confirms it; its slug, or None."""
    slug = resolve_session_slug_for_t3_thread(thread_id, root=root)
    if slug is None:
        return None
    if listed_at is not None and _written_after(record_path_for_slug(slug, root=root), listed_at):
        return None

    def still_this_environments(record: SessionRecord) -> bool:
        return (
            t3_thread_of(record) == thread_id
            and record.display is not None
            and record.display.t3_environment_id in (None, environment_id)
        )

    try:
        record = read_record(slug, root=root)
    except (FileNotFoundError, CorruptSessionRecord, OSError):
        logger.error('mark_t3_sessions_exited: failed to read %s', slug, exc_info=True)
        return None
    if record.status in TERMINAL_STATUSES or not still_this_environments(record):
        return None
    if not thread_ended(thread_id):
        return None
    try:
        marked = _mark_exited_if_still_non_terminal(
            slug, root, exit_code=ORPHAN_EXIT_CODE, still_applies=still_this_environments
        )
    except (FileNotFoundError, CorruptSessionRecord, OSError):
        logger.error('mark_t3_sessions_exited: failed to mark %s', slug, exc_info=True)
        return None
    return slug if marked is not None else None


def _written_after(path: Path, instant: datetime) -> bool:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) > instant
    except OSError:
        return True


FOLDED_EXIT_CODE: int = 201
"""Synthetic exit code ``mark_session_folded`` stamps on a hand-launched
record whose session turned out to be a spawn's: the spawn record now carries
that session. Clear of every code ``ORPHAN_EXIT_CODE``'s docstring lists, and
of that sentinel itself."""


def mark_session_folded(slug: str, root: Path | str | None = None) -> SessionRecord | None:
    """Mark the record *slug* EXITED with ``FOLDED_EXIT_CODE`` unless already terminal.

    Returns the marked record, or None when it was already terminal. Raises
    like ``read_record`` when it is missing or unreadable.
    """
    return _mark_exited_if_still_non_terminal(slug, root, exit_code=FOLDED_EXIT_CODE)


def _normalize_wm_window_id(raw: str | None) -> str | None:
    """Canonicalize a window id to unpadded ``0x<hex>``, or ``None`` if unprovable.

    ``_resolve_display``'s WINDOWID branch (session_hooks.py) can persist a
    DECIMAL window id (e.g. ``'26'``), while ``wmctrl -l`` always prints
    zero-padded hex (e.g. ``'0x0000001a'``). Comparing the two raw strings
    would treat a live decimal-captured window as "gone" and falsely reap
    it -- avoiding exactly that is ``mark_windowless_wm_sessions_exited``'s
    whole reason for calling this. This parses *raw* as base-16 when it
    starts with ``'0x'``/``'0X'``, else base-10, and re-renders the parsed
    value as unpadded ``f'0x{value:x}'`` -- so ``'26'`` and ``'0x0000001a'``
    both normalize to ``'0x1a'`` and compare equal; an already-canonical
    ``'0x1a'`` round-trips unchanged.

    Fails toward "keep, don't reap": returns ``None`` for ``None``/empty/
    whitespace-only input, a value that doesn't parse as an int in the
    selected base, or a negative value. Callers must treat ``None`` as
    "cannot prove this window is gone" -- never as a wildcard match against
    another ``None``.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text, 16) if text.startswith(('0x', '0X')) else int(text, 10)
    except ValueError:
        return None
    if value < 0:
        return None
    return f'0x{value:x}'


_WMCTRL_TIMEOUT_SECS = 2
"""Per-probe ``wmctrl -l`` timeout for ``_wmctrl_list``, seconds. Mirrors
``session_hooks._WMCTRL_TIMEOUT_SECS`` -- kept as a separate module-level
constant (not imported) per this module's stdlib-only/import-free
constraint; see ``_wmctrl_list``'s docstring."""


def _wmctrl_list(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Fail-soft ``wmctrl -l`` runner -- default for ``_wmctrl_live_window_ids``.

    DUPLICATED (not imported) from ``session_hooks._wmctrl_list``: this
    module is deliberately stdlib-only and import-free of the rest of the
    orchestrator (see module docstring), and ``session_hooks`` already
    imports ``session_registry``, so importing back would be circular. Keep
    this in sync with ``session_hooks._wmctrl_list`` if either changes.

    Never raises: a genuinely-missing ``wmctrl`` binary (``FileNotFoundError``)
    yields a permanent-failure rc=127 sentinel; any other ``OSError``/
    ``subprocess.SubprocessError`` (notably a timeout) yields a distinct,
    transient rc=124 sentinel. Both sentinels -- and every other nonzero
    return code -- are treated identically (fail-soft: no window evidence)
    by ``_wmctrl_live_window_ids``; they are only distinguished here in case
    a future caller ever needs to react to them differently.
    """
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_WMCTRL_TIMEOUT_SECS)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, returncode=127, stdout='')
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(argv, returncode=124, stdout='')


def _wmctrl_live_window_ids(
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> set[str] | None:
    """Return the set of normalized window ids ``wmctrl -l`` currently reports.

    HARD fail-soft: returns ``None`` -- never an empty set -- for a missing
    ``wmctrl`` binary, ANY nonzero return code, or any exception raised by
    *run* itself. Callers MUST treat ``None`` ("no window evidence") as
    distinct from an empty set (a successful probe that legitimately found
    zero windows) -- only a successful run (rc=0, however many windows)
    authorizes ``mark_windowless_wm_sessions_exited`` to reap anything.

    Each ``wmctrl -l`` line is parsed via ``split(None, 3)`` -- the same
    idiom as ``WmBackend.is_alive`` (cockpit/src/cockpit/backends/wm.py) and
    ``session_hooks._resolve_wm_window_id`` -- but, unlike both of those,
    this takes column 0 (the window id) of ANY non-empty line, even a
    titleless line with fewer than 4 columns. A reaper must fail toward
    keeping live sessions: the stricter >=4-column gate those two use would
    silently drop a live-but-titleless window from the live set, risking a
    false reap. ``wmctrl -l`` has no header and every line begins with a
    window id, so this can't admit a phantom id. Each id is normalized via
    ``_normalize_wm_window_id`` before being added to the set (a column 0
    that fails to parse is simply skipped, not an error).

    NOTE (reviewer-flagged): this parse is now duplicated across THREE call
    sites -- ``WmBackend.is_alive``, ``session_hooks._resolve_wm_window_id``,
    and this function -- keep all three in sync if the ``wmctrl -l`` column
    layout ever changes. The three already diverge in one respect (this
    function's deliberate any-non-empty-line/column-0 leniency vs. the other
    two's stricter >=4-column gate, documented above), which raises the odds
    of further drift the next time any one of them changes. No change is
    needed now given this module's stdlib-only/import-free constraint (see
    module docstring); if/when that constraint is ever relaxed, extracting
    this parse and ``_wmctrl_list`` into one shared stdlib-only helper all
    three call sites import would remove the drift risk entirely.

    *run* defaults to ``None``, which resolves to the real ``_wmctrl_list``
    at CALL time -- a plain name lookup in this function's body, not a bound
    default-parameter value -- so a test's
    ``monkeypatch.setattr(session_registry, '_wmctrl_list', fake)`` takes
    effect even when reached indirectly via
    ``mark_windowless_wm_sessions_exited()``'s own default ``run``
    passthrough.
    """
    if run is None:
        run = _wmctrl_list
    try:
        result = run(['wmctrl', '-l'])
    except Exception:
        logger.warning('_wmctrl_live_window_ids: wmctrl -l invocation raised', exc_info=True)
        return None
    if result.returncode != 0:
        return None
    live_ids: set[str] = set()
    for line in result.stdout.splitlines():
        columns = line.split(None, 3)
        if not columns:
            continue
        normalized = _normalize_wm_window_id(columns[0])
        if normalized is not None:
            live_ids.add(normalized)
    return live_ids


def _wm_window_id_of(record: SessionRecord) -> str | None:
    """Normalized window id of a wm-display *record*; ``None`` for other displays or a bad id."""
    if record.display is None or record.display.kind != DisplayKind.WM.value:
        return None
    return _normalize_wm_window_id(record.display.wm_window_id)


def _names_window(window_id: str) -> Callable[[SessionRecord], bool]:
    return lambda record: _wm_window_id_of(record) == window_id


def mark_windowless_wm_sessions_exited(
    root: Path | str | None = None,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> list[SessionRecord]:
    """Sweep ``<root>/sessions/*/`` and mark EXITED any wm-display session whose window is gone.

    A SEPARATE liveness sweep from ``mark_orphaned_sessions_exited`` above,
    not a folded-in extension of it: a closed terminal window is DEFINITIVE
    evidence the session is dead/un-interactable, so this sweep is
    deliberately PID- and AGE-INDEPENDENT -- it reaps exactly the zombies
    the pid/TTL sweep is forced to keep (a live ``launcher_pid``, or a
    record younger than ``NON_TERMINAL_HEARTBEAT_TTL``). Folding the two
    predicates together would muddy ``mark_orphaned_sessions_exited``'s
    "mirrors reap_stale_records' stale_pid predicate EXACTLY" contract, and
    would run ``wmctrl -l`` on every liveness check instead of at most twice
    per sweep.

    Scope: only records with ``display is not None and display.kind ==
    DisplayKind.WM.value`` are ever considered. A ``display=None`` (headless
    fleet-agent) or ``display.kind == 'tmux'`` record is untouched here --
    both stay on the existing pid/TTL path (a closed tmux pane is not
    "window gone" in the X11 sense this sweep checks). A wm record whose
    captured ``wm_window_id`` is missing or fails to parse
    (``_normalize_wm_window_id`` returns ``None``) is ALSO left untouched --
    we cannot prove the window is gone, so we fail toward keep and defer to
    the pid/TTL sweep.

    HARD fail-soft: if the first ``_wmctrl_live_window_ids(run)`` probe is
    ``None`` (missing ``wmctrl`` binary, a nonzero return code, or any
    exception -- see its docstring), this returns ``[]`` immediately and
    marks NOTHING. "No window evidence" (a headless host, an X11 hiccup)
    must never be treated as "every window is gone" -- that would mass-mark
    every wm session on the fleet EXITED.

    Two-probe rule: a record is marked only on a probe taken AFTER it was
    read. ``_run_launching``'s detached reap runs concurrently with the
    spawned session's SessionStart hook, which writes its ``wm_window_id``
    (``orchestrator/src/orchestrator/session_hooks.py::_resolve_display``),
    so a window id can land after the first probe; the pass therefore only
    collects candidates absent from the first probe, and one confirming
    probe after the pass decides (``None`` again marks nothing). The mark
    itself re-reads the record and goes ahead only if it still names the
    candidate's window, so a hook that re-bound it to a new window since
    the pass read it is left alone.

    Self-correction trade-off (mirrors ``mark_orphaned_sessions_exited``): a
    window that both probes miss (e.g. racing a window remap) can mark a
    still-live session EXITED. This is deliberately accepted -- the
    session's next hook event calls ``refresh_record`` and flips status back
    -- rather than adding retries beyond the confirming probe.

    Known trade-off for a multi-DISPLAY fleet (reviewer-flagged): ``wmctrl
    -l`` only reports windows on the single X ``DISPLAY`` the invoking
    process is connected to -- whatever ``_run_launching``/``run_reap`` (or
    a directly-invoked CLI) happens to inherit as ``$DISPLAY`` -- never
    every display a fleet's sessions might be spread across. A session
    whose window genuinely lives on a DIFFERENT display than the one this
    sweep's probe ran under is indistinguishable here from one that is
    actually gone -- its id is simply absent from ``live_ids`` -- so a
    multi-display fleet risks a false reap of a still-live session on
    another display. This is accepted for the same reason as the
    self-correction trade-off above (a live session's next hook event flips
    it back), and single-DISPLAY is this sweep's assumed common case. To
    keep a suspected cross-display false reap diagnosable rather than
    silent, any sweep that marks at least one record logs the ``DISPLAY``
    value it probed under.

    Unlike ``reap_stale_records``, this does NOT delete anything -- it marks
    a matching record EXITED (exit_code=``ORPHAN_EXIT_CODE``), preserving
    every other field, and leaves its directory in place;
    ``reap_stale_records``'s existing ``terminal_ttl`` rule reclaims it
    later, unchanged. An already-terminal record is left fully untouched
    (idempotent -- never re-stamped with the sentinel).

    A per-record try/except means one bad record (a corrupt body, a
    concurrent-reap race vanishing the file mid-sweep, an unwritable record)
    never aborts the sweep -- it is logged and skipped, exactly like
    ``mark_orphaned_sessions_exited``'s own fault handling.

    *run* is the injectable ``wmctrl -l`` runner passed through to
    ``_wmctrl_live_window_ids`` (``None``, the default, resolves to the real
    ``_wmctrl_list`` at call time).
    """
    live_ids = _wmctrl_live_window_ids(run)
    if live_ids is None:
        return []  # HARD fail-soft: no window evidence -- mark nothing.

    base = sessions_dir(root)
    marked: list[SessionRecord] = []
    if not base.is_dir():
        return marked

    candidates: list[tuple[str, str]] = []
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name

        try:
            record = read_record(slug, root=root)
        except (FileNotFoundError, CorruptSessionRecord):
            continue  # missing/corrupt body -- reap_stale_records' 'corrupt' rule's job
        if record.status in TERMINAL_STATUSES:
            continue  # already terminal -- idempotent, never re-stamped
        window_id = _wm_window_id_of(record)
        if window_id is None or window_id in live_ids:
            continue  # tmux / headless / unprovable id, or the window is still live
        candidates.append((slug, window_id))

    if not candidates:
        return marked
    confirmed_live_ids = _wmctrl_live_window_ids(run)
    if confirmed_live_ids is None:
        return marked  # the confirming probe yielded no evidence -- mark nothing.

    for slug, window_id in candidates:
        if window_id in confirmed_live_ids:
            continue  # the window appeared after the first probe

        try:
            marked_record = _mark_exited_if_still_non_terminal(
                slug, root, exit_code=ORPHAN_EXIT_CODE, still_applies=_names_window(window_id)
            )
        except (FileNotFoundError, CorruptSessionRecord, OSError):
            # A single unmarkable record (a concurrent-reap race, an
            # unwritable record) must not abort the sweep -- log and move on
            # to the next candidate.
            logger.error(
                'mark_windowless_wm_sessions_exited: failed to mark %s', slug, exc_info=True
            )
            continue
        if marked_record is None:
            continue  # lost the race: a concurrent writer already made this terminal
        marked.append(marked_record)

    if marked:
        # Diagnostic breadcrumb for the multi-DISPLAY trade-off documented
        # above: ties every mark to the DISPLAY the probe actually ran
        # under, so a suspected cross-display false reap is diagnosable
        # after the fact instead of silent. Only logged when this sweep
        # actually marked something, to stay quiet on the common no-op call.
        logger.info(
            'mark_windowless_wm_sessions_exited: marked %d session(s) EXITED (probed under DISPLAY=%r)',
            len(marked),
            os.environ.get('DISPLAY'),
        )
    return marked


# ---------------------------------------------------------------------------
# Reap pipeline: one serialised, timed, fail-soft drain (task 7944)
# ---------------------------------------------------------------------------

REAP_LOCK_FILENAME = 'reap.lock'
REAP_REPORT_FILENAME = 'last-reap.json'


def reap_lock_path(root: Path | str | None = None) -> Path:
    return fleet_root(root) / REAP_LOCK_FILENAME


def reap_report_path(root: Path | str | None = None) -> Path:
    return fleet_root(root) / REAP_REPORT_FILENAME


@contextlib.contextmanager
def reap_lock(root: Path | str | None = None) -> Iterator[None]:
    """Blocking exclusive flock on ``<fleet_root>/reap.lock``, serialising reaps.

    The sidecar is a stable lock target, created once and never renamed or
    deleted, for the same reason as ``decision_registry.py::decision_id_lock``'s.
    """
    with exclusive_sidecar_lock(reap_lock_path(root)):
        yield


@dataclass(frozen=True)
class ReapPhase:
    """One phase of ``run_reap``: its wall time, and either its count or its error."""

    name: str
    seconds: float
    count: int | None
    error: str | None


@dataclass(frozen=True)
class ReapReport:
    """What one ``run_reap`` did, persisted at ``reap_report_path``.

    ``requested_at`` to ``started_at`` is the time it queued on ``reap_lock``;
    ``remaining_record_dirs`` is the registry's size after the drain, or
    ``None`` if it could not be counted.
    """

    requested_at: str
    started_at: str
    finished_at: str
    phases: tuple[ReapPhase, ...]
    remaining_record_dirs: int | None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2) + '\n'


def _run_reap_phase(
    name: str, phase: Callable[[Path | str | None], Sized], root: Path | str | None
) -> ReapPhase:
    started = time.monotonic()
    try:
        count = len(phase(root))
    except Exception as exc:
        logger.error('run_reap: phase %s failed', name, exc_info=True)
        return ReapPhase(
            name=name,
            seconds=time.monotonic() - started,
            count=None,
            error=f'{type(exc).__name__}: {exc}',
        )
    return ReapPhase(name=name, seconds=time.monotonic() - started, count=count, error=None)


def _count_record_dirs(root: Path | str | None) -> int | None:
    try:
        with os.scandir(sessions_dir(root)) as entries:
            return sum(1 for entry in entries if entry.is_dir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning('run_reap: could not count record dirs', exc_info=True)
        return None


def _a_reap_started_since(requested_at: datetime, root: Path | str | None) -> bool:
    try:
        last = json.loads(reap_report_path(root).read_text(encoding='utf-8'))
        return datetime.fromisoformat(last['started_at']) > requested_at
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_reap(root: Path | str | None = None) -> ReapReport | None:
    """Drain the registry: index unpointed t3 records, mark orphans/windowless-wm exited, THEN delete stale dirs and pointers.

    The ``reap`` verb, for the operator and for ``_run_launching``'s detached
    child alike. Concurrent calls queue on ``reap_lock``. A call whose turn
    comes after a reap that STARTED after this call was requested returns
    ``None`` without draining: that reap already covered everything this
    call could see, so a burst of spawns costs one drain, not one each.

    Mark-then-delete keeps a pass complete: a record either liveness sweep
    marks exited this pass must not simultaneously be eligible for stale_pid
    deletion (marking bumps its mtime, so it lands back within
    reap_stale_records' NON_TERMINAL_HEARTBEAT_TTL heartbeat grace) -- it
    only becomes reapable later, once it ages past TERMINAL_TTL as any other
    terminal record does. ``mark_windowless_wm_sessions_exited`` runs here
    alongside ``mark_orphaned_sessions_exited`` because it marks exactly the
    live-pid / age<1h wm-display zombies the pid/TTL sweep is forced to keep.
    ``index_unpointed_t3_records`` runs first; the sweeps after it only read
    the thread index. ``reap_stale_session_pointers`` runs last so pointers
    to records removed in this same pass go too.

    Each phase is timed and fail-soft on its own, so one phase's fault never
    skips the rest. The report is written atomically to
    ``reap_report_path``: it is the only window into a detached child whose
    stdio is DEVNULL. A failed report write is logged, not raised.
    """
    requested_at = datetime.now(UTC)
    with reap_lock(root):
        if _a_reap_started_since(requested_at, root):
            logger.info('run_reap: a reap started since this one was requested; skipping')
            return None
        started_at = datetime.now(UTC)
        phases = (
            _run_reap_phase('index_unpointed_t3_records', index_unpointed_t3_records, root),
            _run_reap_phase('mark_orphaned_sessions_exited', mark_orphaned_sessions_exited, root),
            _run_reap_phase(
                'mark_windowless_wm_sessions_exited', mark_windowless_wm_sessions_exited, root
            ),
            _run_reap_phase('reap_stale_records', reap_stale_records, root),
            _run_reap_phase('reap_stale_session_pointers', reap_stale_session_pointers, root),
        )
        report = ReapReport(
            requested_at=requested_at.isoformat(),
            started_at=started_at.isoformat(),
            finished_at=datetime.now(UTC).isoformat(),
            phases=phases,
            remaining_record_dirs=_count_record_dirs(root),
        )
        try:
            atomic_write_text(reap_report_path(root), report.to_json())
        except OSError:
            logger.error('run_reap: failed to write %s', reap_report_path(root), exc_info=True)
    return report


# ---------------------------------------------------------------------------
# Role leases: single-owner-per-role (Attention Rail T7; PRD T7 §4.1-4.2, §6 G5)
# ---------------------------------------------------------------------------

LEASE_NAME_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._#-]')
"""Unlike _SLUG_SANITIZE_RE, this PRESERVES '#': task-scoped lease names
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


SESSION_PID_ENV = 'CLAUDE_PID'
"""Env var naming the long-lived ``claude`` process's pid (see resolve_session_pid)."""

SESSION_ID_ENV = 'CLAUDE_CODE_SESSION_ID'
"""Env var naming THIS Claude Code session's id; the CLI sets it beside ``CLAUDE_PID``."""

_SESSION_ID_RE = re.compile(r'[A-Za-z0-9-]{1,128}')

_SHARED_SESSION_HOST_SUBCOMMAND = b'agents'
"""``argv[1]`` of a ``claude agents`` process, which runs the sessions it hosts in-process."""


def _parse_session_pid(env: Mapping[str, str]) -> int | None:
    """Parse *env*'s ``CLAUDE_PID`` as a positive int, else None. Never raises, never logs."""
    try:
        pid = int(env.get(SESSION_PID_ENV, '').strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _parse_session_id(env: Mapping[str, str]) -> str | None:
    """Parse *env*'s ``CLAUDE_CODE_SESSION_ID`` as a session-id token, else None. Never raises, never logs."""
    try:
        session_id = env.get(SESSION_ID_ENV, '').strip()
    except (AttributeError, TypeError):
        return None
    return session_id if _SESSION_ID_RE.fullmatch(session_id) else None


def is_shared_session_host(pid: int) -> bool:
    """Whether *pid* is a ``claude agents`` process hosting several sessions.

    The one home of the rule that a pid hosting several sessions cannot vouch
    for any one of them: it outlives each. An unreadable
    ``/proc/<pid>/cmdline`` (no /proc, a dead pid) reads as not a host.
    """
    if pid <= 0:
        return False
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError):
        return False
    return len(argv) >= 2 and argv[1] == _SHARED_SESSION_HOST_SUBCOMMAND


@dataclass(frozen=True)
class SessionIdentity:
    """THIS session's lease identity, resolved once from the env (resolve_session_identity).

    claude_pid: ``$CLAUDE_PID`` parsed, or None when unusable.
    session_id: ``$CLAUDE_CODE_SESSION_ID`` parsed, or None when unusable.
    shared_host: whether *claude_pid* is a shared ``claude agents`` host
        (is_shared_session_host), so that pid does not die with this session.
    """

    claude_pid: int | None
    session_id: str | None
    shared_host: bool

    @property
    def session_scoped_pid(self) -> int | None:
        """A pid that dies with THIS session, or None."""
        return None if self.shared_host else self.claude_pid

    def lease_slug(self, name: str) -> str | None:
        """``<name>-<claude pid>``; under a shared host ``<name>-<claude pid>-<session id>``.

        None when that token is underivable -- see default_lease_slug.
        """
        if self.claude_pid is None:
            return None
        if not self.shared_host:
            return f'{name}-{self.claude_pid}'
        if self.session_id is None:
            return None
        return f'{name}-{self.claude_pid}-{self.session_id}'


def resolve_session_identity(env: Mapping[str, str] | None = None) -> SessionIdentity:
    """Resolve THIS session's SessionIdentity from *env* (default ``os.environ``). Never logs."""
    if env is None:
        env = os.environ
    claude_pid = _parse_session_pid(env)
    return SessionIdentity(
        claude_pid=claude_pid,
        session_id=_parse_session_id(env),
        shared_host=claude_pid is not None and is_shared_session_host(claude_pid),
    )


def lease_liveness_pid(identity: SessionIdentity) -> int:
    """The pid a lease body records for *identity*: its session-scoped pid, else 0, LOUDLY.

    The single home of the lease liveness degradation WARNING; why the
    fallback is 0 is resolve_session_pid's docstring.
    """
    if identity.session_scoped_pid is not None:
        return identity.session_scoped_pid
    if identity.shared_host:
        logger.warning(
            'lease_liveness_pid: $%s=%s names a shared `claude agents` host, whose pid '
            'outlives every session it hosts, so it cannot vouch for this one. The lease '
            'pid/liveness guard is DEGRADED to heartbeat-only staleness: recording pid 0, '
            'which is never alive, so this lease ages out and is reaped once its heartbeat '
            'stops.',
            SESSION_PID_ENV,
            identity.claude_pid,
        )
    else:
        logger.warning(
            'lease_liveness_pid: $%s is unset or unusable; the lease pid/liveness '
            'guard is DEGRADED to heartbeat-only staleness. Recording pid 0, which is '
            'never alive, so a crashed holder still ages out and is reaped normally. '
            'Note: neither `$$` nor `$PPID` substitutes for it -- `$$` in a Claude Code '
            'Bash tool call is the transient /bin/bash -c wrapper, and `$PPID` is not '
            'stable across tool calls either. Set $CLAUDE_PID; if you cannot, the lease '
            'slug is underivable too, so pass an explicit --slug <stable-token>.',
            SESSION_PID_ENV,
        )
    return 0


def resolve_session_pid(env: Mapping[str, str] | None = None) -> int:
    """Resolve THIS Claude Code session's long-lived pid -- the lease's liveness anchor.

    WHY THIS EXISTS (task 3994, defect 2). ``$$`` inside a Claude Code Bash
    tool call is NOT the session: each tool call is a fresh ``/bin/bash -c``
    wrapper whose pid is dead the instant the call returns. Measured:
    ``$$``=2487079 (transient wrapper) vs ``$PPID``==``$CLAUDE_PID``=2345789,
    with ``ps -o comm=`` confirming the latter is the long-lived ``claude``
    process. Both watcher SKILLs nonetheless prescribed ``--pid $$`` for a
    year, so ``pid_alive`` was ~always False for a watcher lease and the
    (dead pid AND aged heartbeat) staleness AND-guard collapsed to a pure
    heartbeat check.

    Resolving it HERE rather than only documenting it in each SKILL.md is
    deliberate: documentation alone re-creates the failure mode it fixes,
    because the pid was wrong for that whole year precisely because it
    depended on every skill doc getting one shell token right.

    Chain, in order: ``CLAUDE_PID`` (parsed, must be > 0, and not a shared
    ``claude agents`` host -- see SessionIdentity) -> ``0``.

    THE FALLBACK IS 0 DELIBERATELY, and specifically NOT ``os.getsid(0)`` ->
    ``os.getppid()`` (which this returned before review). Those are the
    durable-pid idiom ``session_ownership_probes.py::EventProbes.hand_launched_liveness_pid``
    uses for T6's hand-launched RECORDS, and they are wrong HERE, because a
    lease pid is consumed by a REAPER rather than merely displayed: the POSIX
    session leader is ordinarily the interactive shell, which lives for the
    terminal's whole lifetime -- so it stays alive after the ``claude``
    process it stood in for has crashed.
    ``pid_alive`` would then answer True forever, and because BOTH staleness
    rules (claim_lease's ``is_stale`` and reap_stale_leases' ``stale_pid``)
    require a DEAD pid AND an aged heartbeat, a crashed holder's lease would
    become PERMANENTLY unreapable -- recoverable only by a human
    ``lease-release --force``. That is a strictly worse failure than the
    ``--pid $$`` defect this function exists to fix, and the exact opposite of
    the heartbeat-only degradation documented here and in LEASE_HEARTBEAT_TTL.

    0 makes that documented degradation literally true: ``pid_alive`` returns
    False for every pid <= 0, so an unresolvable session pid collapses the
    AND-guard to its heartbeat half -- precisely the pre-3994 behaviour, which
    LEASE_HEARTBEAT_TTL's 7200s is still sized to survive -- instead of to a
    never-reapable lease. The cost is bounded and non-destructive: a contender
    reads ``holder_liveness=orphaned`` for a degraded holder that may well be
    alive, but a FRESH heartbeat still forces that contender to stand down,
    and the prescribed response to ``orphaned`` is to file a DecisionRecord
    and exit -- never to steal.

    Degradation is LOUD, never silent: lease_liveness_pid emits the WARNING.
    Never raises -- a pid we cannot resolve must not break a lease claim.
    """
    return lease_liveness_pid(resolve_session_identity(env))


def resolve_own_record_slug(
    env: Mapping[str, str] | None = None, *, root: Path | str | None = None
) -> str | None:
    """Return THIS claude session's registry record slug, via ``$CLAUDE_PID``'s pointer.

    A lease slug (``default_lease_slug``: ``<lease-name>-$CLAUDE_PID``) and a
    registry record slug (``build_session_slug``) are different namespaces,
    so neither can be looked up as the other. The pid pointer is the join:
    the session hooks stamp ``sessions-by-pid/<claude pid>`` with the record
    key, and ``resolve_session_slug_for_pid`` reads it back only while that
    record still names the pid as its owner. Its consumers are
    ``LeaseHolder.record_slug`` and ``decision_registry.py::DecisionRecord.record_slug``.

    None when ``CLAUDE_PID`` is unusable or no vouched-for pointer exists.
    A shared ``claude agents`` host has no session-scoped pid, so the result
    is None there too.
    Quiet by contract: this serves enrichment, so an unresolvable slug is a
    blank field in the caller's result, not a log line.
    """
    pid = resolve_session_identity(env).session_scoped_pid
    return None if pid is None else resolve_session_slug_for_pid(pid, root=root)


def default_lease_slug(
    name: str, env: Mapping[str, str] | None = None, *, identity: SessionIdentity | None = None
) -> str | None:
    """Derive THIS session's lease slug for lease *name*: ``<name>-<session pid>``.

    Under a shared ``claude agents`` host it is
    ``<name>-<session pid>-<CLAUDE_CODE_SESSION_ID>``: every hosted session
    shares that pid, so the pid alone would hand them all one identity.

    WHY THIS EXISTS (task 4248). This is ``resolve_session_pid``'s own argument,
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
    would share. ``resolve_session_pid`` may degrade to 0
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
    and ``lease_liveness_pid`` reports a degraded liveness pid where one is
    written. The *name* is passed through verbatim -- a lease slug is never a
    filesystem path (``lease_path_for_name`` sanitizes only the NAME, and
    ``sanitize_slug`` applies only to session RECORD slugs), so the ``#`` in a
    task-scoped ``unblock-df#2085`` needs no escaping; see LeaseHolder.session_slug.

    *identity* ACCEPTS AN ALREADY-RESOLVED SessionIdentity, so ``main()`` derives
    the lease body's liveness pid and this slug from ONE resolution; *env* is
    then ignored. Passing None resolves from *env* (defaulting to ``os.environ``).
    """
    return (identity if identity is not None else resolve_session_identity(env)).lease_slug(name)


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
check. Task 3994's ``resolve_session_pid`` restores a real, long-lived
session pid, so that premise no longer holds.

CURRENT BASIS (3994): with the pid guard actually live, a LIVE holder is
never reaped regardless of heartbeat age (a shared ``claude agents`` host pid
excepted, see _lease_is_stale), so this TTL now bounds only ONE
thing -- crash recovery for a genuinely-dead holder. It nonetheless stays at
2h rather than dropping back toward the old 5min, for two reasons:
(a) resolve_session_pid still degrades -- to pid 0, which `pid_alive`
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
    EXITED: the record's status is terminal (``TERMINAL_STATUSES``).
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
        pass it to ``read_record``; the holder's record is reached through
        ``record_slug`` below.

        THIS DOCSTRING USED TO CLAIM THE OPPOSITE -- that it was "the
        holder's own session-registry slug (see build_session_slug), letting
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
        ``read_record(session_slug)`` ALWAYS raises FileNotFoundError.
        Pinned by tests/test_session_registry.py's
        TestLeaseSlugIsNotASessionRecordKey.

        Its ONLY two roles are (i) the ownership comparison in
        ``_check_lease_ownership`` and (ii) naming the holder in a
        human-readable contention/refusal message.
    pid: the holder process's pid; liveness is checked via pid_alive. Any int, bool
        excluded (0 is resolve_session_pid's degraded provably-dead value).
    start_ts: ISO-8601 timestamp of when this holder claimed the lease.
    record_slug: the holder's session-registry record key, resolved at claim
        time from *pid*'s pointer (``resolve_session_slug_for_pid``). '' when
        it could not be resolved or the body predates the field -- an
        absence of evidence, never evidence of absence.
    """

    session_slug: str
    pid: int
    start_ts: str
    record_slug: str = ''

    def __post_init__(self) -> None:
        if not _is_strict_int(self.pid):
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
    if not is_record_key(holder.record_slug):
        return HolderRecordState.UNREADABLE
    try:
        record = read_record(holder.record_slug, root=root)
    except FileNotFoundError:
        return HolderRecordState.ABSENT
    except (OSError, CorruptSessionRecord):
        return HolderRecordState.UNREADABLE
    if record.status in TERMINAL_STATUSES:
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
    ``read_record(holder.session_slug)``: a lease slug is not a record key,
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

    A running shared-host pid cannot vouch for its session (see is_shared_session_host).
    """
    if age_secs <= LEASE_HEARTBEAT_TTL.total_seconds():
        return False
    return holder is None or not holder_alive or is_shared_session_host(holder.pid)


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
    stable identity key (see build_session_slug), while the pid is the
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


# ---------------------------------------------------------------------------
# Project token from cwd: shared.project_layout.main_checkout_of; never raises
# ---------------------------------------------------------------------------


def _project_checkout(cwd: str) -> Path | None:
    """The main checkout enclosing absolute *cwd*, or the enclosing checkout itself when git's layout names no main one."""
    if not os.path.isabs(cwd):
        return None
    try:
        return main_checkout_of(cwd)
    except CheckoutRootError as refused:
        return refused.checkout


def _cwd_project_token(cwd: str) -> str:
    """``parse_spawn_identity``'s default project -- see its step 3."""
    checkout = _project_checkout(cwd)
    if checkout is not None and checkout.name:
        return checkout.name
    return os.path.basename(cwd.rstrip('/')) or 'unknown'


# ---------------------------------------------------------------------------
# CLI + fail-soft (PRD: a registry fault must never change the spawn's exit code)
# ---------------------------------------------------------------------------

_TITLE_RE = re.compile(r'^(?P<role>[^:]+):(?P<project>[^#\s]+)(?:#(?P<task_id>\S+))?')
"""Matches the documented spawn terminal-title convention (skills/spawn/SKILL.md):
``'<role>:<project>#<task-id> <short-slug>'``, task-id and short-slug optional."""


@dataclass(frozen=True)
class SpawnIdentity:
    """Resolved role/project/task_id/escalation_id for a `launching` write."""

    role: str
    project: str
    task_id: str | None
    escalation_id: str | None


def parse_spawn_identity(
    env: Mapping[str, str],
    title: str,
    prompt: str,
    cwd: str,
) -> SpawnIdentity:
    """Resolve spawn identity: env > title-parse > defaults.

    1. ``CLAUDE_SPAWN_ROLE``/``PROJECT``/``TASK_ID``/``ESCALATION_ID`` in *env*,
       when set to a non-empty value.
    2. For whichever of role/project/task_id env did not supply, a best-effort
       parse of the documented terminal-title convention
       ``'<role>:<project>#<task-id> <short-slug>'`` (skills/spawn/SKILL.md).
       The trailing short-slug is always discarded; a project-level title
       (no ``#``) yields ``task_id=None``.
    3. Defaults: ``role='session'``; ``project`` is the basename of the
       enclosing git checkout's MAIN working tree, with *cwd* resolved
       (following symlinks) via ``shared.project_layout.main_checkout_of``.
       A checkout whose main checkout cannot be derived (bare-backed
       worktree, submodule, broken gitfile) names itself; with no enclosing
       checkout it is ``basename(cwd.rstrip('/'))`` or ``'unknown'``. A
       relative *cwd* is not resolved against the process cwd. The token is
       written raw (unfolded); the cockpit folds at its read boundary.

    *prompt* takes no part in this resolution; it is accepted so callers have
    one function for the whole spawn-identity cascade.
    """
    role = env.get('CLAUDE_SPAWN_ROLE') or None
    project = env.get('CLAUDE_SPAWN_PROJECT') or None
    task_id = env.get('CLAUDE_SPAWN_TASK_ID') or None
    escalation_id = env.get('CLAUDE_SPAWN_ESCALATION_ID') or None

    if (role is None or project is None) and title:
        m = _TITLE_RE.match(title.strip())
        if m:
            role = role or m.group('role')
            project = project or m.group('project')
            if task_id is None:
                task_id = m.group('task_id')

    if role is None:
        role = 'session'
    if project is None:
        project = _cwd_project_token(cwd)

    return SpawnIdentity(role=role, project=project, task_id=task_id, escalation_id=escalation_id)


def start_detached_reap(root: Path | str | None = None) -> bool:
    """Start ``run_reap`` on *root*'s fleet in a detached ``reap`` child and return at once.

    The child gets its own session, no inherited fds, DEVNULL stdio and ``/``
    as its cwd, so it can never hold a caller's pipe open nor keep the
    caller's worktree in use while it runs or queues. It inherits
    ``os.environ`` (so ``DISPLAY`` carries over) with ``CLAUDE_FLEET_ROOT``
    pinned to the absolute fleet root. A start failure is logged to stderr
    only and returns False.
    """
    try:
        env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root(root).absolute())}
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'reap'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd='/',
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.warning('start_detached_reap: could not start the reap child', exc_info=True)
        return False
    return True


def _run_launching(env: Mapping[str, str]) -> str:
    """Build + write a LAUNCHING record from CLAUDE_SPAWN_* env; return its dir.

    Every spawn still drives the full mark-then-delete drain (``run_reap``),
    now in a detached ``reap`` child (``start_detached_reap``), so the
    launcher never waits on an O(N) registry pass. That child's stdio is
    DEVNULL and a start failure only logs to stderr, so the printed record dir
    stays the only stdout -- spawn-claude.sh captures it into
    ``SESSION_RECORD_DIR``. Concurrent spawns' reaps serialise on
    ``reap_lock``.
    """
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    cwd = env.get('CLAUDE_SPAWN_CWD') or os.getcwd()
    launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
    launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else os.getpid()
    self_exit, t3_end = _parse_end_of_work_inputs(env)

    identity = parse_spawn_identity(env, title, prompt, cwd)
    slug = build_session_slug(identity.role, identity.project, identity.task_id, launcher_pid)
    record_dir = record_path_for_slug(slug).parent

    record = SessionRecord(
        session_slug=slug,
        status=Status.LAUNCHING,
        title=title,
        role=identity.role,
        project=identity.project,
        task_id=identity.task_id,
        escalation_id=identity.escalation_id,
        prompt=prompt,
        cwd=cwd,
        launcher_pid=launcher_pid,
        launcher_identity=capture_for_record(launcher_pid),
        start_ts=datetime.now(UTC).isoformat(),
        result_file=str(record_dir / RESULT_FILENAME),
        transcript_path=transcript_path_for_cwd(cwd),
        self_exit=self_exit,
        t3_end=t3_end,
    )
    write_record(record)
    start_detached_reap()
    return str(record_dir)


def slug_root_from_record_dir(record_dir: str) -> tuple[str, Path]:
    """Derive (slug, root) from a record dir printed by `launching`.

    The dir is ``<root>/sessions/<slug>``, so its name is the slug and its
    grandparent is the root -- this is how the `exit`/`refresh` CLI
    subcommands (which only ever receive this printed dir, not a root) find
    their way back to the same write_record/read_record key.
    """
    path = Path(record_dir)
    return path.name, path.parent.parent


def _run_exit(record_dir: str, code: int) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = update_status(slug, root=root, status=Status.EXITED, exit_code=code)
    _drop_t3_thread_pointer_if_stale(record, root)


def _run_refresh(record_dir: str, status_value: str) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    record = refresh_record(slug, root=root, status=Status(status_value))
    if record.status in TERMINAL_STATUSES:
        _drop_t3_thread_pointer_if_stale(record, root)


def _run_set_display(
    record_dir: str,
    kind: str,
    wm_title: str,
    wm_window_id: str | None,
    tmux_target: str | None,
    t3_thread_id: str | None = None,
) -> None:
    slug, root = slug_root_from_record_dir(record_dir)
    update_display(
        slug,
        root=root,
        kind=kind,
        wm_title=wm_title,
        wm_window_id=wm_window_id,
        tmux_target=tmux_target,
        t3_thread_id=t3_thread_id,
    )


def _run_lease_claim(
    name: str, slug: str, pid: int | None, policy_value: str, *, record_slug: str = ''
) -> None:
    """Run the ``lease-claim`` verb: ALWAYS prints a ``decision=<value>`` line + message.

    *pid* is the claimant's long-lived session pid. Resolving it in code
    rather than in the SKILL.md makes the correct pid STRUCTURAL: the
    ``--pid $$`` defect (task 3994) existed because it depended on every skill
    doc getting one shell token right.

    ON THE CLI PATH IT ARRIVES ALREADY RESOLVED (task 4248): ``main()`` resolves
    the SessionIdentity ONCE and derives both the slug and this pid from it, so
    the identity token and the pid written into the lease body cannot
    desynchronise. The ``pid is None`` fallback below is
    therefore a defensive path for a DIRECT caller (a test, a future in-process
    caller), not the CLI's route -- and it stays, so this function remains
    correct standalone rather than depending on a caller it cannot see.
    *record_slug* likewise arrives resolved from ``main()``, from that same
    pid; it is never resolved here, so the fallback path records ''.

    This carries its OWN fail-open guard, independent of main()'s outer
    try/except: a fault raised by claim_lease itself (a corrupt lease body,
    an unwritable leases_dir, ...) must never surface as a silent failure or
    -- worse -- a false stand-down. On any exception here, the caller is
    still told to PROCEED (fail-open), exactly as if the lease were free,
    because a lease-substrate fault must never block a watcher cycle or a
    /unblock session.
    """
    try:
        if pid is None:
            pid = resolve_session_pid()
        holder = LeaseHolder(
            session_slug=slug,
            pid=pid,
            start_ts=datetime.now(UTC).isoformat(),
            record_slug=record_slug,
        )
        claim = claim_lease(name, holder=holder, policy=LeasePolicy(policy_value))
    except Exception:
        logger.error('lease-claim %s failed', name, exc_info=True)
        print(f'decision={LeaseDecision.PROCEED.value}')
        print(f'lease-claim {name} faulted; proceeding (fail-open)')
        # `slug=` is emitted here too, while `holder_liveness=` deliberately is
        # not: the fault made the HOLDER unknown, not US. This is also where
        # the line is most useful -- the claim's outcome is least certain.
        print(f'slug={slug}')
        return
    print(f'decision={claim.decision.value}')
    print(claim.message)
    # ADDITIVE and LAST (task 3994 defect 4; extended by task 4248): `decision=`
    # stays line 1 and the message line 2, so a skill parser that reads only the
    # first two lines is unaffected -- which is why these are appended lines
    # rather than a fourth LeaseDecision token a not-yet-reloaded skill would
    # face as an unrecognised value. Lease SEMANTICS are unchanged: we never
    # auto-steal. This only lets a stand-down name a machine-readable owner, so
    # a watcher can file a DecisionRecord for a provably-gone holder instead of
    # exiting silently (INV-6/INV-7).
    #
    # TWO appended lines now: `holder_liveness=` (about the CONTENDING HOLDER)
    # and, last, `slug=` (about THIS CALLER). The second exists because the
    # skills no longer pass `--slug` at all: it makes the CLI-derived identity
    # LEGIBLE, so an agent whose later heartbeat returns `result=refused` can
    # cross-check what it claimed as against `lease-show`'s holder_slug. It is a
    # DIAGNOSTIC, explicitly NOT a value for the skills to thread into the next
    # verb -- a shell variable cannot survive to the next Claude Code Bash tool
    # call (each is a fresh `/bin/bash -c`), which is precisely why the
    # derivation had to move into the CLI rather than be printed for quoting.
    #
    # The MUTATING verbs deliberately gain no `slug=` line: `_run_lease_mutation`
    # already names the presented slug alongside the real holder in its refusal
    # message, so a second line would restate what the caller was just told.
    #
    # THREE values, not two: an ACQUIRED claim has no contending holder at all
    # -- the holder is this caller -- so it prints `none`. This first shipped
    # emitting `held` on a successful claim, which reads as "someone else has
    # it": an ambiguous machine-readable token in a change whose entire purpose
    # is removing ambiguous lease signals. The skills only branch on this line
    # under `decision=stand-down`, so `none` narrows the vocabulary rather than
    # breaking a reader. The line is absent only on the fail-open path above,
    # where a substrate fault means we genuinely know nothing about a holder
    # and must not assert one either way.
    #
    # `holder_record=` (task 4237), last of all, is a SECOND, INDEPENDENT orphan
    # axis: the holder's registry record, per HolderRecordState -- the
    # claimant's own on an acquired claim, the existing holder's on a contended
    # one. It deliberately does not change the `orphaned` predicate below: every
    # lease claimed before LeaseHolder.record_slug existed reads `unlinked`, so
    # requiring corroboration would silence today's orphan signal for that whole
    # population. Reading the two lines together is the skills' job; collapsing
    # them into one verdict is what task 3994 defect 3 forbade.
    if claim.acquired:
        print('holder_liveness=none')
        print(f'slug={slug}')
        print(f'holder_record={claim.holder_record_state.value}')
        return
    # A SINGLE signal, deliberately: `orphaned` means exactly "the pid
    # recorded in the lease body is not running". This predicate used to
    # AND in `claim.holder_session_state in ('absent', 'exited')`, which
    # WHENEVER THERE WAS A HOLDER measured as a structural constant 'absent'
    # -- a lease slug is not a session-registry record key (see
    # LeaseHolder.session_slug) -- so dropping it is an honesty fix, not a
    # behaviour change. The one case where that conjunct was NOT constant is
    # an unreadable body, whose holder is None and which therefore resolved
    # 'unknown' -> held; `claim.holder is not None` reproduces that
    # explicitly, on its real reason (no holder means no pid to probe, so
    # there is nothing to call orphaned) rather than as a side effect of a
    # withdrawn lookup. The emitted values are byte-identical before and
    # after, in every branch that HAS a contending holder.
    #
    # It is sound only because resolve_session_pid now records the
    # long-lived `claude` pid; the old `--pid $$` was the transient Bash-tool
    # wrapper, always dead, which would have made this a tautology. Both
    # error directions are safe: pid REUSE yields a false `held` (silence,
    # the status quo, never a steal), and a false `orphaned` still triggers
    # only a NON-destructive response (file a DecisionRecord, then exit --
    # never auto-steal, which is the duplicate-spawn incident path). A holder
    # claimed on resolve_session_pid's DEGRADED path (pid 0: an unresolvable
    # CLAUDE_PID, or a shared `claude agents` host) therefore reads `orphaned`
    # while possibly alive -- the
    # deliberate, bounded cost of keeping such a lease reapable at all.
    orphaned = claim.holder is not None and not claim.holder_alive
    print(f'holder_liveness={"orphaned" if orphaned else "held"}')
    # THIS CALLER's slug, never the holder's -- see the block comment above.
    print(f'slug={slug}')
    print(f'holder_record={claim.holder_record_state.value}')


def _run_lease_mutation(
    verb: str,
    name: str,
    slug: str,
    mutate: Callable[[], LeaseMutation],
) -> None:
    """Drive a mutating lease verb and print ``result=<value>`` + a human line.

    The machine-readable token is ALWAYS the first line, mirroring
    ``lease-claim``'s ``decision=`` so an agent parses all four lease verbs
    with one rule. On REFUSED/FORCED a second line names the lease's actual
    holder, so a caller told "no" also learns WHO owns it and can act
    (inspect with ``lease-show``) rather than guess.

    The holder is read BEFORE *mutate* runs: a FORCED ``lease-release``
    deletes the very file the holder's identity lives in, so reading after
    the fact would report ``<unreadable lease body>`` for exactly the case
    where naming the displaced holder matters most.

    ``--force`` reaches the mutation through *mutate*'s closure and nowhere
    else -- this helper takes no ``force`` parameter, deliberately. It had one
    briefly, unread by any line of the body, which invites the next reader to
    assume the wording below branches on it; the RESULT (FORCED vs REFUSED) is
    the only thing that decides what is printed.
    """
    holder, holder_alive, age_secs = _read_lease_holder_state(
        lease_path_for_name(name), now=datetime.now(UTC)
    )
    result = mutate()
    print(f'result={result.value}')
    if result not in (LeaseMutation.REFUSED, LeaseMutation.FORCED):
        return
    holder_slug = holder.session_slug if holder is not None else '<unreadable lease body>'
    holder_pid = holder.pid if holder is not None else -1
    alive_word = 'alive' if holder_alive else 'not running'
    holder_facts = f'{holder_slug} (pid {holder_pid} {alive_word}, heartbeat {int(age_secs)}s ago)'
    if result is LeaseMutation.REFUSED:
        print(
            f'{verb} {name} refused: held by {holder_facts}, not by {slug} '
            f'— inspect with `lease-show --name {name}`'
        )
    else:
        print(f'{verb} {name} FORCED by {slug} over holder {holder_facts}')


def _run_lease_heartbeat(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-heartbeat',
        name,
        slug,
        lambda: heartbeat_lease(name, slug=slug, force=force),
    )


def _run_lease_release(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-release',
        name,
        slug,
        lambda: release_lease(name, slug=slug, force=force),
    )


def _run_lease_show(name: str) -> None:
    """Run the ``lease-show`` verb: print one lease's full state as key=value lines.

    The replacement for ``cat <lease>`` + ``stat -c %y`` archaeology (task
    3994 defect 3), in the same machine-readable ``<key>=<value>`` form as
    ``decision=``/``result=`` so an agent parses every lease verb with one
    rule. ``state=`` is the discriminator: the holder_* identity lines are
    printed only when the body actually parsed, so an unreadable body reads
    as unreadable rather than as a holder named ``<unknown>``. Booleans are
    rendered lowercase (``true``/``false``) so a shell test is trivial.
    ``holder_record=`` and ``holder_record_slug=`` are ADDITIVE and LAST,
    the convention ``_run_lease_claim`` states.

    Read-only and fail-soft, like lease_status itself: it never mutates the
    lease and never raises.
    """
    status = lease_status(name)
    print(f'name={status.name}')
    print(f'state={status.state}')
    if status.state == 'absent':
        return
    if status.holder_slug is not None:
        print(f'holder_slug={status.holder_slug}')
    if status.holder_pid is not None:
        print(f'holder_pid={status.holder_pid}')
    print(f'holder_pid_alive={str(status.holder_pid_alive).lower()}')
    print(f'heartbeat_ts={status.heartbeat_ts}')
    print(f'heartbeat_age_secs={int(status.heartbeat_age_secs)}')
    print(f'reclaimable={str(status.reclaimable).lower()}')
    print(f'holder_record={status.holder_record_state.value}')
    if status.holder_record_slug is not None:
        print(f'holder_record_slug={status.holder_record_slug}')


def _run_lease_reap() -> list[ReapedLease]:
    return reap_stale_leases()


def _resolve_filing_record_slug(local_id: str, project: str) -> str:
    """Return the record_slug a write-decision filing of *local_id* stamps.

    That is resolve_own_record_slug's slug, else '' after a WARNING, because a
    blank one leaves that cockpit row unable to focus its filer's session for
    the record's life (task 7527).
    """
    slug = resolve_own_record_slug()
    if slug is not None:
        return slug
    logger.warning(
        'write-decision --id %s --project %s: $%s=%r resolves to no vouched session record '
        '(unset, unusable or a shared `claude agents` host pid; or no sessions-by-pid/<pid> '
        'pointer whose record still names that pid as its owner), so this decision gets a '
        'BLANK record_slug and its cockpit row cannot focus the filing session (Enter-to-focus) '
        'for the life of the record. This alone does not refuse the filing; an id on stdout is '
        'what says it landed.',
        local_id,
        project,
        SESSION_PID_ENV,
        os.environ.get(SESSION_PID_ENV),
    )
    return ''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='session_registry')
    sub = parser.add_subparsers(dest='verb', required=True)

    sub.add_parser('launching', help='write a LAUNCHING record from CLAUDE_SPAWN_* env')

    exit_p = sub.add_parser('exit', help='set status=exited + exit_code on a record')
    exit_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    exit_p.add_argument('--code', required=True, type=int)

    refresh_p = sub.add_parser('refresh', help='upsert a record with a new status (hook seam)')
    refresh_p.add_argument('--record', required=True, help='record dir')
    refresh_p.add_argument('--status', required=True, choices=[s.value for s in Status])

    set_display_p = sub.add_parser(
        'set-display', help='stamp display (kind/title/target) on a record'
    )
    set_display_p.add_argument('--record', required=True, help='record dir, as printed by `launching`')
    set_display_p.add_argument('--kind', required=True, choices=[k.value for k in DisplayKind])
    set_display_p.add_argument('--tmux-target', default=None)
    set_display_p.add_argument('--wm-title', default='')
    set_display_p.add_argument('--wm-window-id', default=None)
    set_display_p.add_argument('--t3-thread-id', default=None)

    sub.add_parser('reap', help='sweep and remove stale session records')

    lease_claim_p = sub.add_parser('lease-claim', help='claim a single-owner-per-role lease (T7)')
    lease_claim_p.add_argument('--name', required=True, help='lease name, see build_lease_name')
    lease_claim_p.add_argument(
        '--slug',
        default=None,
        help="this claimant's own session_slug; defaults to "
        'default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
        '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
        'host). Only pass this as '
        'an explicit operator override -- the derived value is what every '
        'later lease-heartbeat/lease-release re-derives and matches against',
    )
    lease_claim_p.add_argument(
        '--pid',
        type=int,
        default=None,
        help="this claimant's own long-lived session pid; defaults to "
        'resolve_session_pid() ($CLAUDE_PID, or 0 under a shared `claude agents` host, '
        'whose pid outlives this session). Only pass this as an explicit '
        'operator override -- never `$$`, which is the transient Bash-tool shell. '
        'This is the lease body\'s LIVENESS pid only: it does not supply the '
        "slug's identity (which always derives from $CLAUDE_PID), so --pid alone "
        'does not satisfy an underivable --slug -- lease-heartbeat/lease-release '
        'have no --pid, and a slug only one verb can derive is not an identity',
    )
    lease_claim_p.add_argument(
        '--policy',
        choices=[p.value for p in LeasePolicy],
        default=LeasePolicy.STAND_DOWN.value,
        help='how to respond when the lease is already held live',
    )

    lease_heartbeat_p = sub.add_parser('lease-heartbeat', help='touch a held lease (heartbeat)')
    lease_heartbeat_p.add_argument('--name', required=True)

    lease_release_p = sub.add_parser('lease-release', help='release a held lease')
    lease_release_p.add_argument('--name', required=True)

    # Ownership (task 3994): both MUTATING verbs act only for the lease's
    # actual holder, identified by slug. AMENDED BY TASK 4248 -- the slug is
    # now DERIVED (default_lease_slug, in main()) rather than required from the
    # caller. That does NOT retract 3994's guard. The defect 3994 fixed was that
    # both verbs mutated UNCONDITIONALLY, with no ownership check at all; the
    # check still runs, and it now compares against a slug derived from THIS
    # caller's own $CLAUDE_PID, so a stranger session derives a different slug
    # and is still `result=refused` with the lease body untouched. What is
    # retired is only the requirement that the token be SPELLED IN SHELL, which
    # is exactly what let it drift: 3994 left three SKILL.md files assembling
    # `<role>-<project>-${CLAUDE_PID:-$PPID}` by hand, and $PPID is measurably
    # not stable across Claude Code tool calls. `required=True` was a forcing
    # function to make the call sites pass a slug at all; the call sites have
    # now been updated, and the derivation is the stronger guarantee -- a caller
    # can no longer MIS-spell a token it never spells. An unresolvable
    # $CLAUDE_PID still exits 2 (see main()); it never falls back silently.
    for _lease_mutation_p in (lease_heartbeat_p, lease_release_p):
        _lease_mutation_p.add_argument(
            '--slug',
            default=None,
            help='the slug you claimed this lease with; a mismatch is refused. '
            'Defaults to default_lease_slug(--name) (`<name>-$CLAUDE_PID`, or '
            '`<name>-$CLAUDE_PID-$CLAUDE_CODE_SESSION_ID` under a shared `claude agents` '
            'host), the same derivation lease-claim uses -- pass it only as an operator '
            'override',
        )
        _lease_mutation_p.add_argument(
            '--force',
            action='store_true',
            help='operator recovery: act even when you are not the holder; logged loudly',
        )

    lease_show_p = sub.add_parser(
        'lease-show',
        help="inspect a lease (holder, pid liveness, heartbeat freshness); "
        'read-only. Use this instead of `cat`, which cannot show freshness',
    )
    lease_show_p.add_argument('--name', required=True, help='lease name, see build_lease_name')

    sub.add_parser('lease-reap', help='sweep and remove stale leases')

    decision_cli.add_verb_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ALWAYS returns 0 for a session or lease verb: a fault in its core logic
    is logged loudly (stderr, via the standard logging machinery) and
    swallowed here rather than raised, so a registry fault can never change
    the exit code of the bash caller (spawn-claude.sh) that invokes this
    script. That covers runtime FAULTS only: a malformed invocation exits 2
    from argparse's parse_args() below, before the try/except.

    A decision verb goes to decision_cli.run_verb, which owns the decision
    half of this policy, including why close-decision and reopen-decision
    exit non-zero on a refusal. The filer's record slug it needs is session
    identity, so this module hands it _resolve_filing_record_slug.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ONE derivation point for the lease slug, shared by all three lease verbs
    # (task 4248). It lives HERE, in main(), and not inside _run_lease_claim /
    # _run_lease_mutation, because all three must derive a BYTE-IDENTICAL slug
    # or the whole scheme fails -- a claim the holder's own later heartbeat
    # cannot match is precisely the drift this change removes. One shared point
    # makes that structural instead of a three-way convention nobody enforces.
    # An explicit --slug always wins: it stays the operator's override.
    #
    # ONE IDENTITY RESOLUTION TOO. `lease-claim` both WRITES a liveness pid
    # (into the lease body) and EMBEDS an identity (the derived slug), so both
    # derive from the one SessionIdentity resolved here: to default_lease_slug
    # via identity=, and to _run_lease_claim via args.pid
    # (lease_liveness_pid). Pinned by
    # test_main_lease_claim_derives_the_body_pid_and_the_slug_pid_from_one_resolution.
    # The two legitimately differ under a shared `claude agents` host (body pid
    # 0, slug carrying the host pid and session id) and under an explicit --pid,
    # which overrides the BODY's liveness pid only, never the slug -- the
    # identity has to stay derivable by the mutating verbs, which have no --pid.
    # The body pid also keys the holder's `record_slug` lookup below.
    #
    # Resolution is DEMAND-DRIVEN, and only lease-claim's liveness pid can log:
    # a degradation WARNING is useful only when it is TRUE of the work being
    # done, and a heartbeat writes no pid.
    if args.verb in ('lease-claim', 'lease-heartbeat', 'lease-release'):
        derives_body_pid = args.verb == 'lease-claim' and args.pid is None
        identity = None
        if args.slug is None or derives_body_pid:
            identity = resolve_session_identity()
            if derives_body_pid:
                args.pid = lease_liveness_pid(identity)
            if args.slug is None:
                args.slug = default_lease_slug(args.name, identity=identity)
        # FAIL LOUDLY, never silently. default_lease_slug returns None only
        # when this session's identity is underivable, and it refuses to
        # synthesize `<name>-0` (or `<name>-<shared host pid>`) because that
        # token is IDENTICAL for two such sessions -- each would then pass the
        # other's _is_lease_owner check, reopening 3994 defect 1. So we stop here.
        #
        # This is NOT a breach of _run_lease_claim's fail-open contract.
        # Fail-open covers lease-SUBSTRATE faults (a corrupt lease body, an
        # unwritable leases_dir) -- conditions the caller cannot fix and must
        # not be blocked by. An underivable slug is a CALLER/ENVIRONMENT INPUT
        # error, exactly the class argparse already exits 2 for, and it is
        # fixable at the call site (`--slug`). Note also that pre-4248 a
        # slug-less lease verb exited 2 as well: the undeterminable case KEEPS
        # its current exit code, and only the derivable case gained a default.
        #
        # parser.error raises SystemExit -- a BaseException -- so it
        # deliberately escapes main()'s `except Exception`, which swallows
        # faults and returns 0. That escape is the point: this must reach the
        # shell as a nonzero exit, not as a silently-successful no-op.
        if not args.slug:
            cause = (
                f'${SESSION_PID_ENV}={identity.claude_pid} names a shared `claude agents` '
                f'host, so this session\'s identity needs ${SESSION_ID_ENV}, which is unset '
                'or unusable'
                if identity is not None and identity.shared_host
                else f'${SESSION_PID_ENV} is unset or unusable'
            )
            parser.error(
                f'cannot derive the lease slug for --name {args.name}: {cause}, '
                'so this session has no stable identity to claim, '
                'heartbeat or release a lease with. Refusing to invent one -- a synthesized '
                'token would collide with every other degraded session and let each act on '
                "the other's lease. Fix the environment, or pass an explicit "
                '--slug <stable-token> that this session will re-use on every later lease verb. '
                'Note --pid does NOT substitute for --slug: it sets the lease body\'s liveness '
                'pid, not this session\'s identity, and lease-heartbeat/lease-release have no '
                '--pid at all -- only --slug is honoured by all three verbs.'
            )
        if args.verb == 'lease-claim':
            args.record_slug = ''
            if args.pid is not None:
                args.record_slug = resolve_session_slug_for_pid(args.pid) or ''

    if args.verb in decision_cli.DECISION_VERBS:
        return decision_cli.run_verb(args, filing_record_slug=_resolve_filing_record_slug)

    try:
        if args.verb == 'launching':
            print(_run_launching(os.environ))
        elif args.verb == 'exit':
            _run_exit(args.record, args.code)
        elif args.verb == 'refresh':
            _run_refresh(args.record, args.status)
        elif args.verb == 'set-display':
            _run_set_display(
                args.record,
                args.kind,
                args.wm_title,
                args.wm_window_id,
                args.tmux_target,
                args.t3_thread_id,
            )
        elif args.verb == 'reap':
            run_reap()
        elif args.verb == 'lease-claim':
            _run_lease_claim(
                args.name, args.slug, args.pid, args.policy, record_slug=args.record_slug
            )
        elif args.verb == 'lease-heartbeat':
            _run_lease_heartbeat(args.name, args.slug, args.force)
        elif args.verb == 'lease-release':
            _run_lease_release(args.name, args.slug, args.force)
        elif args.verb == 'lease-show':
            _run_lease_show(args.name)
        elif args.verb == 'lease-reap':
            _run_lease_reap()
        else:
            logger.error('session_registry %s: the parser accepts this verb but no runner handles it', args.verb)
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
