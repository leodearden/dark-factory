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
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema / contract (PRD §6 G5): consumers import this; they never re-derive
# the record shape. Bump SCHEMA_VERSION on any breaking field change.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


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
    result_file: path to this session's result.md (T5), once allocated.
    transcript_path: best-effort ``~/.claude/projects/<encoded-cwd>`` path.
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
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


def build_session_slug(
    role: str, project: str, task_id: str | None, launcher_pid: int,
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


# ---------------------------------------------------------------------------
# Single-writer atomic write / read / update
# ---------------------------------------------------------------------------


class CorruptSessionRecord(Exception):
    """Raised by read_record when a record.json exists but fails to parse."""


def write_record(record: SessionRecord, root: Path | str | None = None) -> None:
    """Atomically write *record* (tmp file in the same dir, then os.replace).

    Mirrors ``LaneStore._write`` (lane_lifecycle.py:279-298): the tmp file is
    created in the target's own parent dir so the replace stays within one
    filesystem, and is cleaned up on any failure. Every successful write
    bumps record.json's mtime, which reap_stale_records() reads as this
    record's heartbeat.
    """
    path = record_path_for_slug(record.session_slug, root=root)
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


def refresh_record(
    slug: str,
    root: Path | str | None = None,
    *,
    status: Status | None = None,
) -> SessionRecord:
    """Read-modify-write used by the T6 hook seam; bumps the mtime heartbeat.

    Strict as of this task (raises if no record exists, like update_status).
    T6's hand-launched-capture path additionally needs upsert-on-absent
    semantics for a session with no prior spawn-claude.sh write; that
    extension lands in step-16 alongside the ``refresh`` CLI subcommand.
    """
    record = read_record(slug, root=root)
    if status is not None:
        record.status = status
    write_record(record, root=root)
    return record
