"""orchestrator.session_records — the session-record schema and its store under ``<fleet_root>/sessions/``.

The record types, the slugs and record paths, the pid, claude-session and
T3-thread pointers, and read/write/update/refresh. Consumers import the schema
from here and never re-derive the record shape (PRD §6 G5).

It imports only the stdlib, this checkout's stdlib-only ``shared`` leaves
(``shared.process_identity``) and its siblings ``fleet_files`` and
``session_identity``. It never imports ``session_registry``: in a bare-shell
(tier-1) run session_registry is ``__main__``, so an upward import would load a
second copy of it. The bare-shell contract is pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

# Reach this checkout's stdlib-only shared leaves from a bare shell; rationale in
# test_session_registry.py::TestStdlibOnlySelfContainment.
if (
    _CHECKOUT_SHARED_SRC := str(Path(__file__).resolve().parents[3] / 'shared' / 'src')
) not in sys.path and Path(_CHECKOUT_SHARED_SRC).is_dir():
    sys.path.append(_CHECKOUT_SHARED_SRC)

from shared.process_identity import ProcessIdentity

# Siblings are imported relative to orchestrator.session_records, or else from this
# file's own directory: loaded by path, as a tier-1 sibling or under a foreign dotless
# name, that directory is on sys.path only if Python or the importer put it there, so it
# is borrowed for these imports alone. Each mode is pinned in
# test_session_registry.py::TestStdlibOnlySelfContainment.
# TYPE_CHECKING points the type checker at the package arm, the only one it can resolve.
if TYPE_CHECKING or __package__:
    from . import session_identity
    from .fleet_files import atomic_write_text, fleet_root
else:
    _OWN_DIR = str(Path(__file__).resolve().parent)
    _OWN_DIR_BORROWED = _OWN_DIR not in sys.path
    if _OWN_DIR_BORROWED:
        sys.path.append(_OWN_DIR)
    try:
        import session_identity
        from fleet_files import atomic_write_text, fleet_root
    finally:
        if _OWN_DIR_BORROWED:
            sys.path.remove(_OWN_DIR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema / contract (PRD §6 G5): consumers import this; they never re-derive
# the record shape. Bump SCHEMA_VERSION on any breaking field change.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# 3: session_registry.py::LeaseHolder.record_slug and decision_registry.py::DecisionRecord.record_slug (task 4237).
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
``CLAUDE_SPAWN_RESULT_FILE`` literal -- ``session_registry.py::_run_launching`` returns
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
        `session_registry.py::_run_launching` at launch time (``<record_dir>/RESULT_FILENAME``)
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
        ``session_hooks.py::_env_slug_ownership``. A non-int, bool, or
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
    uniqueness_token: int | str,
) -> str:
    """Build the record-identity/directory key ``<role>-<project>[-<taskid>]-<token>``.

    The token is the launcher pid for a spawned session (session_registry's
    ``launching`` verb) or the Claude session_id for a hook-keyed session
    (``session_hooks.hook_session_slug``). Either way it keeps records unique
    across concurrent sessions that share role+project+task (PRD §4
    decision 2); single-ownership is a separate concern (T7 leases), not
    encoded in the key. Any character outside ``[A-Za-z0-9._-]`` in any
    segment (or the joined whole) is sanitized to ``'-'`` (see
    ``sanitize_slug``) so the slug is always filesystem-safe.
    """
    segments = [role, project]
    if task_id:
        segments.append(task_id)
    segments.append(str(uniqueness_token))
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


def read_session_pointer(path: Path) -> str | None:
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
    if read_session_pointer(path) == slug:
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
    ``session_registry.py::reap_stale_session_pointers`` removes it.
    """
    if pid <= 0:
        return None
    slug = read_session_pointer(session_pointer_path_for_pid(pid, root=root))
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


def is_claude_session_pointer_name(name: str) -> bool:
    """True iff *name* can name a ``sessions-by-claude-id/`` entry: the ``CLAUDE_CODE_SESSION_ID`` token shape.

    The ``sessions-by-t3-thread/`` twin is ``is_t3_thread_pointer_name``.
    """
    return session_identity.is_session_id(name)


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
    if not is_claude_session_pointer_name(session_id) or not is_record_key(slug):
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
    if not is_claude_session_pointer_name(session_id):
        return None
    slug = read_session_pointer(claude_session_pointer_path(session_id, root=root))
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

    It is the index of live T3-hosted records that ``session_registry.py::mark_t3_sessions_exited`` iterates.
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
    ``session_registry.py::_is_stale_t3_thread_pointer``. Read-only; never raises.
    """
    if not is_t3_thread_pointer_name(thread_id):
        return None
    slug = read_session_pointer(t3_thread_pointer_path(thread_id, root=root))
    if slug is None or not is_record_key(slug):
        return None
    try:
        record = read_record(slug, root=root)
    except (OSError, CorruptSessionRecord):
        return None
    vouches = t3_thread_of(record) == thread_id and record.status not in TERMINAL_STATUSES
    return slug if vouches else None


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
    ``session_registry.py::index_unpointed_t3_records`` retries it.
    """
    thread_id = t3_thread_of(record)
    if (
        thread_id is None
        or record.status in TERMINAL_STATUSES
        or resolve_session_slug_for_t3_thread(thread_id, root=root) is not None
    ):
        return False
    return write_t3_thread_pointer(thread_id, slug, root=root)


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
    session_registry.py::reap_stale_records() reads as this record's heartbeat.
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
