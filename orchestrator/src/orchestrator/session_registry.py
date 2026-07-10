"""orchestrator.session_registry — durable session-registry substrate (Attention Rail T3).

PRD: plans/session-attention-rail-prd.md T3 (spine root; §4.1-4.2, §4.8, §6 G5).

This module is the shared, versioned contract for the global, cross-project
session registry at ``~/.claude/fleet/sessions/<slug>/record.json``. It is
deliberately stdlib-only and self-contained (no intra-orchestrator imports)
so it can be:

- invoked directly by ``skills/spawn/spawn-claude.sh`` via an absolute path
  (no venv/PYTHONPATH/install required), and
- imported as ``orchestrator.session_registry`` by downstream Python
  consumers (T4 verify, T5 result, T6 hooks, T7 leases).

Consumers import the schema/contract defined here; they never re-derive the
record shape (PRD §6 G5).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema / contract (PRD §6 G5): consumers import this; they never re-derive
# the record shape. Bump SCHEMA_VERSION on any breaking field change.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_MINOR = 1
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


@dataclass
class Display:
    """Where to find/focus this session's terminal (Fleet Cockpit C1, PRD §6.1).

    kind: DisplayKind's wire value ('wm' or 'tmux'); stored as a plain
        ``str`` (no from_dict coercion -- see SpawnMode's docstring for why).
    wm_title: the spawned terminal's window title, for a 'wm' kind's
        window-manager lookup.
    wm_window_id: the window-manager's own window id, when known.
    tmux_target: a tmux target spec (e.g. ``'session:window.pane'``), for a
        'tmux' kind.
    """

    kind: str
    wm_title: str = ''
    wm_window_id: str | None = None
    tmux_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'wm_title': self.wm_title,
            'wm_window_id': self.wm_window_id,
            'tmux_target': self.tmux_target,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Display:
        return cls(
            kind=data['kind'],
            wm_title=data.get('wm_title', ''),
            wm_window_id=data.get('wm_window_id'),
            tmux_target=data.get('tmux_target'),
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
    start_ts: str = field(default='', kw_only=True)
    exit_code: int | None = field(default=None, kw_only=True)
    result_file: str | None = field(default=None, kw_only=True)
    transcript_path: str | None = field(default=None, kw_only=True)
    parent_session_id: str | None = field(default=None, kw_only=True)
    spawn_mode: str = field(default=SpawnMode.CHILD, kw_only=True)
    display: Display | None = field(default=None, kw_only=True)
    question: Question | None = field(default=None, kw_only=True)

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
            'start_ts': self.start_ts,
            'status': Status(self.status).value,
            'exit_code': self.exit_code,
            'result_file': self.result_file,
            'transcript_path': self.transcript_path,
            'parent_session_id': self.parent_session_id,
            'spawn_mode': str(self.spawn_mode),
            'display': self.display.to_dict() if self.display is not None else None,
            'question': self.question.to_dict() if self.question is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
        display_data = data.get('display')
        question_data = data.get('question')
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
            launcher_pid=data.get('launcher_pid', 0),
            start_ts=data.get('start_ts', ''),
            exit_code=data.get('exit_code'),
            result_file=data.get('result_file'),
            transcript_path=data.get('transcript_path'),
            parent_session_id=data.get('parent_session_id'),
            spawn_mode=data.get('spawn_mode', SpawnMode.CHILD),
            display=Display.from_dict(display_data) if isinstance(display_data, dict) else None,
            question=Question.from_dict(question_data) if isinstance(question_data, dict) else None,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> SessionRecord:
        return cls.from_dict(json.loads(raw))


class DecisionState(StrEnum):
    """Wire-format lifecycle state for a DecisionRecord (Fleet Cockpit C1, PRD §6.1).

    Modeled as a StrEnum contract for writers, but ``DecisionRecord.state``
    itself is a plain ``str`` field with NO from_dict coercion through this
    enum -- mirrors SpawnMode's additive-safe rationale: an unrecognized
    value must still round-trip rather than raise.

    OPEN: filed, awaiting a human decision.
    ANSWERED: a decision has been recorded.
    DROPPED: withdrawn/superseded without an answer.
    """

    OPEN = 'open'
    ANSWERED = 'answered'
    DROPPED = 'dropped'


@dataclass
class DecisionRecord:
    """One decision-record — ``<fleet_root>/decisions/<id>.json`` (Fleet Cockpit C1 B2, PRD §6.1).

    Only id/project/text/filed_at are mandatory; every other field has a
    documented default so a minimally-filed decision is still well-formed.
    Mirrors LeaseHolder's plain to_dict/from_dict/to_json/from_json shape.

    id: this record's identity/filename key; see decision_path_for_id.
    project: the project_id this decision concerns.
    text: the decision/question text as filed.
    filed_at: ISO-8601 timestamp of when this decision was filed.
    session_id: the session that filed this decision, or None.
    task_id: the task this decision is scoped to, or None.
    escalation_id: the escalation this decision resolves, or None.
    options: the candidate answers offered, or None.
    manual_boost: an operator-assigned priority nudge; defaults to 0.
    state: current lifecycle state; see DecisionState. Defaults to
        ``DecisionState.OPEN``.

    Concurrency: unlike SessionRecord (single-writer-per-slug -- only the
    spawning session ever mutates its own record), a single decision id's
    file may be mutated by TWO different subsystems: a C8 watcher (via
    update_decision_state) and the C5 cockpit (via set_manual_boost). Both
    helpers are unsynchronized read-modify-write cycles with no cross-process
    locking or compare-and-swap, so a concurrent state-update and
    boost-update racing on the same id can silently drop one side's mutation
    (last os.replace() wins). This is a known, accepted limitation --
    consistent with the module's existing lock-free convention -- not a
    per-record corruption risk (each write is still individually atomic).
    See update_decision_state/set_manual_boost for the caller-facing note.
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
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> DecisionRecord:
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Identity, paths, transcript encoding
# ---------------------------------------------------------------------------

_SLUG_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]')


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
    sanitized to ``'-'`` so the slug is always filesystem-safe.
    """
    segments = [role, project]
    if task_id:
        segments.append(task_id)
    segments.append(str(launcher_pid))
    raw = '-'.join(segments)
    return _SLUG_SANITIZE_RE.sub('-', raw)


def fleet_root(root: Path | str | None = None) -> Path:
    """Resolve the fleet root: explicit *root* > $CLAUDE_FLEET_ROOT > ~/.claude/fleet.

    Every registry function takes this same *root* parameter (or derives it
    via this function) so tests never touch the real ``~/.claude`` tree.
    """
    if root is not None:
        return Path(root)
    env_root = os.environ.get('CLAUDE_FLEET_ROOT')
    if env_root:
        return Path(env_root)
    return Path.home() / '.claude' / 'fleet'


def sessions_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'sessions'


def record_path_for_slug(slug: str, root: Path | str | None = None) -> Path:
    return sessions_dir(root) / slug / 'record.json'


def transcript_path_for_cwd(cwd: str) -> str:
    """Best-effort mirror of Claude Code's own ``~/.claude/projects/<enc>`` encoding.

    Both ``/`` and ``.`` map to ``-`` (confirmed against a real
    ``~/.claude/projects/`` directory; PRD §3). This is read-only enrichment
    metadata — never used as a lookup key by this module.
    """
    encoded = cwd.replace('/', '-').replace('.', '-')
    return f'~/.claude/projects/{encoded}'


_DECISION_ID_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]')
"""Mirrors _SLUG_SANITIZE_RE: anything outside [A-Za-z0-9._-] maps to '-', so
a malformed decision id can never escape decisions_dir via a path
separator."""


def decisions_dir(root: Path | str | None = None) -> Path:
    return fleet_root(root) / 'decisions'


def decision_path_for_id(decision_id: str, root: Path | str | None = None) -> Path:
    """Resolve *decision_id* (see DecisionRecord.id) to its ``<decisions_dir>/<id>.json`` path.

    *decision_id* is sanitized through _DECISION_ID_SANITIZE_RE first, so it
    always resolves to a single file directly inside decisions_dir (mirrors
    lease_path_for_name's path-escape guard).
    """
    stem = _DECISION_ID_SANITIZE_RE.sub('-', decision_id)
    return decisions_dir(root) / f'{stem}.json'


# ---------------------------------------------------------------------------
# Single-writer atomic write / read / update
# ---------------------------------------------------------------------------


class CorruptSessionRecord(Exception):
    """Raised by read_record when a record.json exists but fails to parse."""


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write *text* to *path* (tmp file in the same dir, then os.replace).

    Mirrors ``LaneStore._write`` (lane_lifecycle.py:279-298): the tmp file is
    created in the target's own parent dir so the replace stays within one
    filesystem, and is cleaned up on any failure. Shared atomic-write core:
    write_record calls this and lets a failure propagate (its sole caller,
    the CLI main(), provides the outer fail-soft boundary); write_decision
    calls this too but swallows a failure itself (it is called directly by
    watchers/cockpit code with no such boundary).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix='.tmp',
        prefix=path.stem,
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(tmp_path_str, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


def write_record(record: SessionRecord, root: Path | str | None = None) -> None:
    """Atomically write *record* (tmp file in the same dir, then os.replace).

    Every successful write bumps record.json's mtime, which
    reap_stale_records() reads as this record's heartbeat.
    """
    path = record_path_for_slug(record.session_slug, root=root)
    _atomic_write_text(path, record.to_json())


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
        return SessionRecord.from_json(path.read_text())
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
        kind=kind, wm_title=wm_title, wm_window_id=wm_window_id, tmux_target=tmux_target
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
    """
    try:
        record = read_record(slug, root=root)
    except FileNotFoundError:
        # No prior write for this slug: synthesize a fresh record. LAUNCHING
        # is the sensible default identity for "a record just came into
        # being" when the caller upserts without an explicit status.
        record = SessionRecord(
            session_slug=slug,
            status=status if status is not None else Status.LAUNCHING,
            start_ts=datetime.now(UTC).isoformat(),
        )
    if status is not None:
        record.status = status
    write_record(record, root=root)
    return record


def write_decision(record: DecisionRecord, root: Path | str | None = None) -> bool:
    """Atomically write *record* to its own ``<decisions_dir>/<id>.json`` file.

    Self-guarding FAIL-SOFT: unlike write_record (whose sole caller, the CLI
    main(), supplies its own outer try/except), this is called directly by
    C8 watchers and the C5 cockpit, so a write fault must log and return
    rather than raise into them. Returns True on success, False on any
    fault.
    """
    path = decision_path_for_id(record.id, root=root)
    try:
        _atomic_write_text(path, record.to_json())
    except Exception:
        logger.error('write_decision: failed to write %s', path, exc_info=True)
        return False
    return True


def list_decisions(root: Path | str | None = None) -> list[DecisionRecord]:
    """Read every decision under decisions_dir. Missing dir -> [] (no raise).

    A single corrupt or foreign ``*.json`` file is skipped (logged at ERROR)
    rather than aborting the whole read -- mirrors reap_stale_records'
    corrupt-body handling, so one bad file never breaks the cockpit's
    whole-queue read.
    """
    base = decisions_dir(root)
    if not base.is_dir():
        return []
    decisions: list[DecisionRecord] = []
    for path in sorted(base.glob('*.json')):
        try:
            decisions.append(DecisionRecord.from_json(path.read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.error('list_decisions: skipping unreadable %s', path, exc_info=True)
            continue
    return decisions


def update_decision_state(
    decision_id: str,
    state: str,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Read-modify-write *decision_id*'s state field.

    Self-guarding FAIL-SOFT: returns None (logs ERROR) on any fault -- a
    missing file, a corrupt body, or a write failure -- rather than raising,
    matching write_decision's contract for its direct C8/cockpit callers.

    Concurrency NOTE (see DecisionRecord's docstring): this read-modify-write
    is unsynchronized against a concurrent set_manual_boost (or a second
    update_decision_state) racing on the SAME decision id -- the later
    os.replace() wins and silently drops the earlier call's field mutation.
    Each individual write remains atomic; only the read+mutate+write SPAN is
    unsynchronized. Accepted for now, consistent with the module's existing
    lock-free convention.
    """
    path = decision_path_for_id(decision_id, root=root)
    try:
        record = DecisionRecord.from_json(path.read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error('update_decision_state: failed to read %s', path, exc_info=True)
        return None
    record.state = state
    if not write_decision(record, root=root):
        return None
    return record


def set_manual_boost(
    decision_id: str,
    boost: int,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Read-modify-write *decision_id*'s manual_boost field.

    Self-guarding FAIL-SOFT: returns None (logs ERROR) on any fault -- a
    missing file, a corrupt body, or a write failure -- rather than raising,
    matching write_decision's contract for its direct C8/cockpit callers.

    Concurrency NOTE (see DecisionRecord's docstring): this read-modify-write
    is unsynchronized against a concurrent update_decision_state (or a
    second set_manual_boost) racing on the SAME decision id -- the later
    os.replace() wins and silently drops the earlier call's field mutation.
    Each individual write remains atomic; only the read+mutate+write SPAN is
    unsynchronized. Accepted for now, consistent with the module's existing
    lock-free convention.
    """
    path = decision_path_for_id(decision_id, root=root)
    try:
        record = DecisionRecord.from_json(path.read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error('set_manual_boost: failed to read %s', path, exc_info=True)
        return None
    record.manual_boost = boost
    if not write_decision(record, root=root):
        return None
    return record


# ---------------------------------------------------------------------------
# TTL / pid stale-record reaper (PRD §4.8)
# ---------------------------------------------------------------------------

TERMINAL_TTL = timedelta(hours=24)
"""How long a terminal (EXITED/FAILED_TO_START) record survives after its
last write before the reaper reclaims it, regardless of launcher_pid."""

NON_TERMINAL_HEARTBEAT_TTL = timedelta(hours=1)
"""How long a non-terminal record survives with a dead launcher_pid and no
fresh write (heartbeat) before the reaper reclaims it. A live launcher_pid
retains the record regardless of age."""


def _pid_alive(pid: int) -> bool:
    """Return True if the process identified by *pid* is alive.

    Copied (not imported) from harness.py:295-317 to keep this module
    stdlib-only and self-contained (invocable as a standalone script from
    bash with no orchestrator package import).

    - Returns False for pid <= 0 (invalid).
    - Uses os.kill(pid, 0): success -> alive; ProcessLookupError -> dead;
      PermissionError -> alive (visible but unsignalable); other OSError ->
      treated as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class ReapedSessionRecord:
    """One session-registry record directory removed by reap_stale_records.

    path: the removed ``<slug>/`` directory (not the record.json file).
    session_slug: the session's identity/directory key, derived from the
        directory name itself (never from the record body).
    reason: why it was reaped -- one of ``'terminal_ttl'`` (a terminal
        record older than TERMINAL_TTL), ``'stale_pid'`` (a non-terminal
        record whose launcher_pid is dead and older than
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
    - status is non-terminal, launcher_pid is dead, and age >
      NON_TERMINAL_HEARTBEAT_TTL -> reaped, reason='stale_pid'.
    - otherwise -> kept.

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
            elif not _pid_alive(record.launcher_pid) and age > NON_TERMINAL_HEARTBEAT_TTL:
                reason = 'stale_pid'
            else:
                reason = None

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


LEASE_HEARTBEAT_TTL = timedelta(minutes=5)
"""How long a lease survives with a dead holder pid and no fresh heartbeat
(mtime touch) before it is stale-reapable. Mirrors NON_TERMINAL_HEARTBEAT_TTL's
role for session records: staleness requires BOTH a dead holder pid AND an
aged heartbeat (see claim_lease/reap_stale_leases) -- a live holder is never
reaped regardless of heartbeat age."""


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


@dataclass(frozen=True)
class LeaseHolder:
    """Serialized identity of a lease's current holder -- the exact ``.lease`` file body.

    session_slug: the holder's own session-registry slug (see
        build_session_slug), letting a contended claim point back at the
        current owner's session_registry record.
    pid: the holder process's pid; liveness is checked via _pid_alive.
    start_ts: ISO-8601 timestamp of when this holder claimed the lease.
    """

    session_slug: str
    pid: int
    start_ts: str

    def to_dict(self) -> dict[str, Any]:
        return {'session_slug': self.session_slug, 'pid': self.pid, 'start_ts': self.start_ts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaseHolder:
        return cls(session_slug=data['session_slug'], pid=data['pid'], start_ts=data['start_ts'])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> LeaseHolder:
        return cls.from_dict(json.loads(raw))


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
    holder_alive: whether `holder`'s pid is currently alive (_pid_alive).
    heartbeat_age_secs: seconds since the lease file's mtime; 0.0 for a
        freshly-acquired lease.
    message: fully-formatted, user-observable line -- callers print this
        verbatim rather than re-deriving it from the other fields.
    """

    name: str
    decision: LeaseDecision
    acquired: bool
    holder: LeaseHolder | None
    holder_alive: bool
    heartbeat_age_secs: float
    message: str


def _render_contention_message(
    holder: LeaseHolder | None,
    *,
    holder_alive: bool,
    age_secs: float,
    policy: LeasePolicy,
) -> str:
    """Build the exact user-observable contention line callers print verbatim.

    Centralizing this here (rather than in each skill) guarantees the signal
    string is identical everywhere it appears and is unit-testable with an
    injected clock. *holder* may be None (an unreadable/corrupt lease body);
    that is rendered as an explicit placeholder rather than raising.
    """
    slug = holder.session_slug if holder is not None else '<unknown>'
    alive_word = 'alive' if holder_alive else 'dead'
    action = 'standing down' if policy is LeasePolicy.STAND_DOWN else 'proceeding anyway'
    return f'lease held by {slug} ({alive_word}, heartbeat {int(age_secs)}s ago) — {action}'


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


def _acquired_claim(name: str, holder: LeaseHolder) -> LeaseClaim:
    return LeaseClaim(
        name=name,
        decision=LeaseDecision.ACQUIRED,
        acquired=True,
        holder=holder,
        holder_alive=True,
        heartbeat_age_secs=0.0,
        message=f'lease {name} acquired by {holder.session_slug}',
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
        holder = LeaseHolder.from_json(path.read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False, age_secs
    return holder, _pid_alive(holder.pid), age_secs


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
    lease is already held, the existing holder's liveness (_pid_alive) and
    heartbeat age (file mtime) are checked: if BOTH the holder is dead AND
    the heartbeat is older than LEASE_HEARTBEAT_TTL, the stale lease is
    reaped and the claim retried once; otherwise the existing holder is
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
        return _acquired_claim(name, holder)

    existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
    is_stale = (not holder_alive) and age_secs > LEASE_HEARTBEAT_TTL.total_seconds()

    if is_stale:
        # Re-verify staleness immediately before unlinking: a competitor
        # could have reaped-and-reclaimed this same stale lease in the gap
        # since the read above (their O_EXCL create winning the race). This
        # narrows -- POSIX has no atomic compare-and-unlink, so it cannot
        # fully eliminate -- that window: if the lease is no longer stale,
        # it must NOT be unlinked, or we would clobber the competitor's
        # brand-new, live lease and silently violate single-owner-per-role.
        existing_holder, holder_alive, age_secs = _read_lease_holder_state(path, now=now)
        is_stale = (not holder_alive) and age_secs > LEASE_HEARTBEAT_TTL.total_seconds()

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
            return _acquired_claim(name, holder)

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
    )


def heartbeat_lease(name: str, *, root: Path | str | None = None) -> bool:
    """Bump the *name* lease file's mtime to now -- the reaper's heartbeat clock.

    Returns False (fail-soft, no raise) when the lease is absent or the
    ``os.utime`` call itself faults; a lease-substrate error must never
    interrupt a watcher's main loop.
    """
    path = lease_path_for_name(name, root=root)
    if not path.exists():
        return False
    try:
        os.utime(path, None)
    except OSError:
        logger.error('heartbeat_lease: failed to touch %s', path, exc_info=True)
        return False
    return True


def release_lease(name: str, *, root: Path | str | None = None) -> bool:
    """Remove the *name* lease file. Idempotent: a second call returns False.

    Returns whether the lease existed before this call removed it; fail-soft
    (logs loudly at ERROR, returns False) on an OSError from the removal
    itself.
    """
    path = lease_path_for_name(name, root=root)
    existed = path.exists()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error('release_lease: failed to remove %s', path, exc_info=True)
        return False
    return existed


@dataclass(frozen=True)
class ReapedLease:
    """One ``.lease`` file removed by reap_stale_leases.

    path: the removed ``<name>.lease`` file.
    lease_name: the lease's identity, derived from the filename itself
        (the stem of *path*) -- never from the body, so a corrupt/missing
        body is still reapable.
    reason: why it was reaped -- ``'stale_pid'`` (holder pid dead and the
        heartbeat is older than LEASE_HEARTBEAT_TTL) or ``'corrupt'`` (body
        missing/unparseable and older than LEASE_HEARTBEAT_TTL).
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
    - holder pid is dead and age > LEASE_HEARTBEAT_TTL -> reaped,
      reason='stale_pid'.
    - otherwise -> kept -- in particular a LIVE holder is never reaped,
      regardless of heartbeat age (mirrors reap_stale_records' non-terminal
      live-pid rule).

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
        stale = age_secs > LEASE_HEARTBEAT_TTL.total_seconds()

        try:
            holder = LeaseHolder.from_json(lease_path.read_text())
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            reason = 'corrupt' if stale else None
        else:
            reason = 'stale_pid' if (not _pid_alive(holder.pid)) and stale else None

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
    3. Defaults: ``role='session'``, ``project=basename(cwd.rstrip('/'))`` or
       ``'unknown'`` (a trailing ``'/'`` is stripped first so it doesn't
       degrade the basename to empty).

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
        project = os.path.basename(cwd.rstrip('/')) or 'unknown'

    return SpawnIdentity(role=role, project=project, task_id=task_id, escalation_id=escalation_id)


def _run_launching(env: Mapping[str, str]) -> str:
    """Build + write a LAUNCHING record from CLAUDE_SPAWN_* env; return its dir."""
    title = env.get('CLAUDE_SPAWN_TITLE', '') or ''
    prompt = env.get('CLAUDE_SPAWN_PROMPT', '') or ''
    cwd = env.get('CLAUDE_SPAWN_CWD') or os.getcwd()
    launcher_pid_raw = env.get('CLAUDE_SPAWN_LAUNCHER_PID')
    launcher_pid = int(launcher_pid_raw) if launcher_pid_raw else os.getpid()

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
        start_ts=datetime.now(UTC).isoformat(),
        result_file=str(record_dir / RESULT_FILENAME),
        transcript_path=transcript_path_for_cwd(cwd),
    )
    write_record(record)
    return str(record_dir)


def _slug_root_from_record_dir(record_dir: str) -> tuple[str, Path]:
    """Derive (slug, root) from a record dir printed by `launching`.

    The dir is ``<root>/sessions/<slug>``, so its name is the slug and its
    grandparent is the root -- this is how the `exit`/`refresh` CLI
    subcommands (which only ever receive this printed dir, not a root) find
    their way back to the same write_record/read_record key.
    """
    path = Path(record_dir)
    return path.name, path.parent.parent


def _run_exit(record_dir: str, code: int) -> None:
    slug, root = _slug_root_from_record_dir(record_dir)
    update_status(slug, root=root, status=Status.EXITED, exit_code=code)


def _run_refresh(record_dir: str, status_value: str) -> None:
    slug, root = _slug_root_from_record_dir(record_dir)
    refresh_record(slug, root=root, status=Status(status_value))


def _run_set_display(
    record_dir: str,
    kind: str,
    wm_title: str,
    wm_window_id: str | None,
    tmux_target: str | None,
) -> None:
    slug, root = _slug_root_from_record_dir(record_dir)
    update_display(
        slug,
        root=root,
        kind=kind,
        wm_title=wm_title,
        wm_window_id=wm_window_id,
        tmux_target=tmux_target,
    )


def _run_reap() -> list[ReapedSessionRecord]:
    return reap_stale_records()


def _run_lease_claim(name: str, slug: str, pid: int, policy_value: str) -> None:
    """Run the ``lease-claim`` verb: ALWAYS prints a ``decision=<value>`` line + message.

    This carries its OWN fail-open guard, independent of main()'s outer
    try/except: a fault raised by claim_lease itself (a corrupt lease body,
    an unwritable leases_dir, ...) must never surface as a silent failure or
    -- worse -- a false stand-down. On any exception here, the caller is
    still told to PROCEED (fail-open), exactly as if the lease were free,
    because a lease-substrate fault must never block a watcher cycle or a
    /unblock session.
    """
    try:
        holder = LeaseHolder(session_slug=slug, pid=pid, start_ts=datetime.now(UTC).isoformat())
        claim = claim_lease(name, holder=holder, policy=LeasePolicy(policy_value))
    except Exception:
        logger.error('lease-claim %s failed', name, exc_info=True)
        print(f'decision={LeaseDecision.PROCEED.value}')
        print(f'lease-claim {name} faulted; proceeding (fail-open)')
        return
    print(f'decision={claim.decision.value}')
    print(claim.message)


def _run_lease_heartbeat(name: str) -> None:
    heartbeat_lease(name)


def _run_lease_release(name: str) -> None:
    release_lease(name)


def _run_lease_reap() -> list[ReapedLease]:
    return reap_stale_leases()


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

    sub.add_parser('reap', help='sweep and remove stale session records')

    lease_claim_p = sub.add_parser('lease-claim', help='claim a single-owner-per-role lease (T7)')
    lease_claim_p.add_argument('--name', required=True, help='lease name, see build_lease_name')
    lease_claim_p.add_argument('--slug', required=True, help="this claimant's own session_slug")
    lease_claim_p.add_argument(
        '--pid', type=int, default=os.getpid(), help="this claimant's own pid"
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

    sub.add_parser('lease-reap', help='sweep and remove stale leases')

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ALWAYS returns 0: a fault in any subcommand's core logic is logged
    loudly (stderr, via the standard logging machinery) and swallowed here
    rather than raised, so a registry fault can never change the exit code
    of the bash caller (spawn-claude.sh) that invokes this script.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.verb == 'launching':
            print(_run_launching(os.environ))
        elif args.verb == 'exit':
            _run_exit(args.record, args.code)
        elif args.verb == 'refresh':
            _run_refresh(args.record, args.status)
        elif args.verb == 'set-display':
            _run_set_display(args.record, args.kind, args.wm_title, args.wm_window_id, args.tmux_target)
        elif args.verb == 'reap':
            _run_reap()
        elif args.verb == 'lease-claim':
            _run_lease_claim(args.name, args.slug, args.pid, args.policy)
        elif args.verb == 'lease-heartbeat':
            _run_lease_heartbeat(args.name)
        elif args.verb == 'lease-release':
            _run_lease_release(args.name)
        elif args.verb == 'lease-reap':
            _run_lease_reap()
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
