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
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

# NOTE: no non-stdlib imports at module scope, deliberately — see the module
# docstring's stdlib-only clause and _atomic_write_text below. This module is
# executed by absolute path from skills/spawn/spawn-claude.sh with no venv,
# install or workspace packages available, so a `from shared import ...` here
# makes it unimportable there. Pinned by
# test_session_registry.py::TestStdlibOnlySelfContainment.

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

    Concurrency: unlike SessionRecord (single-writer-per-slug -- only the
    spawning session ever mutates its own record), a single decision id's
    file may be mutated by SEVERAL different subsystems: a C8 watcher (via
    update_decision_state), the C5 cockpit (via set_manual_boost), and -- for
    task 3640's back-fill, running against live records while the watchers
    are up -- scripts/backfill_decision_queue_stamp.py (via
    set_decision_escalations_dir). All three helpers serialize their
    read-modify-write span per-decision-id via
    decision_id_lock (a stable ``<id>.json.lock`` sidecar, mirroring task
    1609's escalation_id_lock), so a concurrent state-update, boost-update
    or queue-stamp racing on the same id no longer drops any of the
    mutations -- each write remains individually atomic AND the
    read+mutate+write span is serialized against other callers on the same
    id. See update_decision_state/set_manual_boost/
    set_decision_escalations_dir for the caller-facing note.
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

    THIS IS THE DELIBERATE EXCEPTION to task 3223's consolidation, which moved
    this repo's copies of the tmp+rename writer onto
    :func:`shared.safe_io.atomic_write_text`. This module does NOT delegate,
    and must not be "finished off" by a later cleanup.

    Why: this module is invoked by absolute path from
    ``skills/spawn/spawn-claude.sh`` under an interpreter with no venv, no
    install and no workspace packages on ``sys.path`` (see the module
    docstring's stdlib-only clause, which is a hard constraint, not a stylistic
    preference). A module-scope ``from shared import safe_io`` makes the whole
    module unimportable there — measured as ``ModuleNotFoundError: No module
    named 'shared'`` — and the only visible symptom is a hook subprocess that
    silently never writes ``record.json``. The contract is pinned directly by
    ``test_session_registry.py::TestStdlibOnlySelfContainment``, and this
    function is recorded in ``_ALLOWED_RENAMERS`` in
    ``shared/tests/test_safe_io.py`` so the anti-regrowth guard reads it as the
    documented exception it is rather than a fresh copy.

    The cost is conscious: the repo keeps two hand-rolled copies of this
    pattern instead of one. A documented, allowlisted, test-pinned second copy
    is strictly better than a module that cannot be imported by its own
    documented entrypoint.

    ``os.fdopen(fd, 'w')`` is locale-dependent rather than utf-8. That is a
    latent bug — JSON written under a non-UTF-8 locale — but it is the
    behaviour this module has always had, and changing it is out of scope here.

    Error policy stays with the callers: write_record lets a failure propagate
    (its sole caller, the CLI main(), provides the outer fail-soft boundary);
    write_decision swallows a failure itself (it is called directly by
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


def _mutate_decision(
    decision_id: str,
    mutate: Callable[[DecisionRecord], None],
    *,
    caller: str,
    root: Path | str | None = None,
) -> DecisionRecord | None:
    """Lock-serialized, fail-soft read-modify-write of one decision record.

    The single implementation of the field-setter body shared by
    update_decision_state, set_manual_boost and set_decision_escalations_dir
    (task 3640 amendment). Those three are the public, caller-facing names and
    keep their own docstrings; this holds the parts that MUST NOT diverge
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
            record = DecisionRecord.from_json(path.read_text())
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


@contextlib.contextmanager
def decision_id_lock(decision_id: str, root: Path | str | None = None) -> Iterator[None]:
    """Per-decision-id exclusive advisory lock using a stable sidecar file.

    Mirrors escalation's ``escalation_id_lock`` (escalation/queue.py:24-69,
    task 1609) near-verbatim, retargeted to the decisions dir.

    WHY A SIDECAR (PRD-D3 rationale, same as 1609): write_decision's writer
    is atomic tmp+os.replace (``_atomic_write_text`` above). After a replace,
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
            record = DecisionRecord.from_json(path.read_text())
            record.some_field = new_value
            write_decision(record, root=root)
    """
    lock_path = Path(str(decision_path_for_id(decision_id, root=root)) + '.lock')
    # Defensively create the parent dir so a caller can take this lock
    # without a decision having been written for this id yet -- note this
    # means the sidecar itself is created even for an unknown id (see
    # ORPHAN SIDECARS above).
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


ESCALATION_ARCHIVE_SUBDIR = 'archive'
"""Mirrors escalation.archive.ARCHIVE_SUBDIR BY CONVENTION, not by import:
this module is deliberately stdlib-only with no intra-orchestrator imports
(see module docstring), so it cannot import escalation.archive directly.
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
    this module's no-intra-orchestrator-imports rule (see module docstring).
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
    stdlib-only and import-free of the rest of the orchestrator/escalation
    packages (see module docstring). Fail-soft: never raises.

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
        data = json.loads(candidate.read_text())
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


# ---------------------------------------------------------------------------
# TTL / pid stale-record reaper (PRD §4.8)
# ---------------------------------------------------------------------------

TERMINAL_TTL = timedelta(hours=24)
"""How long a terminal (EXITED/FAILED_TO_START) record survives after its
last write before the reaper reclaims it, regardless of launcher_pid."""

REAP_BATCH_LIMIT = 100
"""Per-call cap on how many stale record dirs a single OPPORTUNISTIC
(spawn-path) reap removes -- see reap_stale_records' `limit` param. Caps a
single spawn's synchronous prune cost regardless of how large the on-disk
backlog has grown, at the cost of draining a large backlog over several
spawns rather than in one call. Drain order follows reap_stale_records'
sorted(iterdir()) (directory-name) order, not oldest-first, so if the
on-disk backlog ever grows faster than spawns can drain it,
alphabetically-later stale dirs can persist longer than
alphabetically-earlier ones (TERMINAL_TTL eligibility itself is unaffected
either way). The CLI `reap` verb (_run_reap) stays unbounded (limit=None)
so an operator wanting a full, immediate drain can still clear the entire
backlog in a single invocation."""

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
    limit: int | None = None,
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

    Directories are visited in ``sorted(iterdir())`` order (directory-name
    order, not age order).

    *now* is injectable for deterministic tests; defaults to the real UTC
    clock.

    *limit* is None (the default) for an unbounded full sweep -- today's
    behavior, unchanged. A positive int stops the sweep once *limit*
    directories have actually been removed, bounding the rmtree work (the
    dominant reclamation cost) per call -- this is what lets an
    opportunistic per-spawn caller stay cheap regardless of backlog size.
    The directory *scan* itself is NOT bounded by *limit* in the worst
    case: directories are visited in sorted(iterdir()) (name) order and the
    sweep only breaks after a removal, so if enough kept (non-stale)
    directories sort ahead of the *limit*-th stale one, every one of them
    is still stat'd and read before the sweep stops -- the scan remains
    O(N) in the total directory count, short-circuiting only once *limit*
    removals have occurred. A directory whose removal is attempted and
    fails (logged, see below) does NOT count against *limit*.
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
            if limit is not None and len(reaped) >= limit:
                break

    return reaped


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
    slug: str, root: Path | str | None, *, exit_code: int
) -> SessionRecord | None:
    """TOCTOU-safe terminal stamp used only by the liveness sweep.

    Re-reads *slug* itself -- a second, fresher read than the sweep's own
    caller-side liveness check -- and, ONLY if status is still non-terminal
    on THIS read, stamps status=EXITED/exit_code=*exit_code* and writes.
    Returns the marked record, or ``None`` if a concurrent writer (most
    plausibly spawn-claude.sh's own ``finish()``) already made the record
    terminal in the narrow window between the sweep's own liveness check
    and this call -- in which case nothing is written, so that writer's
    real exit_code is never clobbered by the orphan sentinel.

    Deliberately a narrow variant rather than a change to ``update_status``
    itself: ``update_status``'s own ``exit`` CLI subcommand caller is an
    authoritative, unconditional final report from the exiting process and
    must stay unconditional; only this sweep's best-effort guess needs to
    defer to a fresher real write.
    """
    record = read_record(slug, root=root)
    if record.status in TERMINAL_STATUSES:
        return None
    record.status = Status.EXITED
    record.exit_code = exit_code
    write_record(record, root=root)
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
    status is non-terminal (not in TERMINAL_STATUSES), ``launcher_pid`` is
    dead (``_pid_alive``), and age (record.json's mtime vs *now*) exceeds
    ``NON_TERMINAL_HEARTBEAT_TTL`` -- the same false-positive guard that
    rule already trusts: a genuinely-live session (including a
    detached/tmux one) keeps bumping its record's mtime via its own hooks,
    so it is never swept regardless of how long it has been non-terminal.

    Known trade-off for DETACHED spawns (reviewer-flagged): spawn-claude.sh
    itself exits once it hands off (its ``launcher_pid`` dies) while claude
    keeps running standalone, so for the entire remaining life of a
    detached session ``_pid_alive`` is already False -- the heartbeat
    (record.json's mtime, bumped only by that session's own
    Notification/Stop hooks after convergence) is the SOLE thing keeping it
    unswept. A detached session that goes quieter than
    ``NON_TERMINAL_HEARTBEAT_TTL`` (1h) between hook events -- e.g. one long
    tool-heavy turn with no intervening Notification/Stop -- can therefore
    be marked EXITED while genuinely still running, driven opportunistically
    from every other spawn's ``_run_launching``/from ``reap`` (not just an
    explicit CLI ``reap``, unlike before this task). This is self-correcting
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
        if _pid_alive(record.launcher_pid):
            continue  # a live launcher_pid is never swept, regardless of age

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
    would run ``wmctrl -l`` on every liveness check instead of once per
    sweep.

    Scope: only records with ``display is not None and display.kind ==
    DisplayKind.WM.value`` are ever considered. A ``display=None`` (headless
    fleet-agent) or ``display.kind == 'tmux'`` record is untouched here --
    both stay on the existing pid/TTL path (a closed tmux pane is not
    "window gone" in the X11 sense this sweep checks). A wm record whose
    captured ``wm_window_id`` is missing or fails to parse
    (``_normalize_wm_window_id`` returns ``None``) is ALSO left untouched --
    we cannot prove the window is gone, so we fail toward keep and defer to
    the pid/TTL sweep.

    HARD fail-soft: resolves ``live_ids = _wmctrl_live_window_ids(run)``
    exactly once per sweep; if that is ``None`` (missing ``wmctrl`` binary, a
    nonzero return code, or any exception -- see its docstring), this
    returns ``[]`` immediately and marks NOTHING. "No window evidence" (a
    headless host, an X11 hiccup) must never be treated as "every window is
    gone" -- that would mass-mark every wm session on the fleet EXITED.

    Self-correction trade-off (mirrors ``mark_orphaned_sessions_exited``): a
    single missed/late ``wmctrl`` probe (e.g. racing a window remap) can mark
    a still-live session EXITED. This is deliberately accepted -- the
    session's next hook event calls ``refresh_record`` and flips status back
    -- rather than adding retries here, keeping this sweep a single
    fail-soft decision point per call.

    Known trade-off for a multi-DISPLAY fleet (reviewer-flagged): ``wmctrl
    -l`` only reports windows on the single X ``DISPLAY`` the invoking
    process is connected to -- whatever ``_run_launching``/``_run_reap`` (or
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
        if record.display is None or record.display.kind != DisplayKind.WM.value:
            continue  # tmux / headless -- stays on the pid/TTL path
        normalized = _normalize_wm_window_id(record.display.wm_window_id)
        if normalized is None or normalized in live_ids:
            continue  # unprovable, or the window is still live

        try:
            marked_record = _mark_exited_if_still_non_terminal(
                slug, root, exit_code=ORPHAN_EXIT_CODE
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


LEASE_HEARTBEAT_TTL = timedelta(hours=2)
"""How long a lease survives with a dead holder pid and no fresh heartbeat
(mtime touch) before it is stale-reapable. Mirrors NON_TERMINAL_HEARTBEAT_TTL's
role for session records: staleness requires BOTH a dead holder pid AND an
aged heartbeat (see claim_lease/reap_stale_leases) -- a live holder is never
reaped regardless of heartbeat age.

VALUE DERIVATION (task 2796, THREAD 1): watcher leases are effectively
HEARTBEAT-ONLY, not pid-guarded. The interactive escalation-watcher SKILL
claims with ``--pid $$`` -- the EPHEMERAL Bash-tool shell pid, dead within
milliseconds of the lease-claim -- so ``_pid_alive`` is ~always False for a
watcher lease and the (dead-pid AND aged-heartbeat) staleness AND-guard
collapses to a pure heartbeat-TTL check. That watcher heartbeats only ONCE
per Main Loop cycle, and each cycle's blocking wait is one watcher-rearm.sh
``--timeout`` slice (canonical 3600s). So the staleness TTL MUST exceed that
heartbeat cadence, or a live-but-quiet holder's lease goes
stale->reap-eligible during any single quiet slice and a duplicate
reaps-and-reclaims it (the duplicate-spawn incident). 7200s = 2x the 3600s
slice gives a full quiet slice plus a full slice of margin -- a
cadence-derived bound, not an arbitrary bump.

TRADE-OFF: a GENUINELY-crashed holder's lease now survives up to 2h before
it is stale-reapable, versus 5min before. This is loud and recoverable, not
silent: a contender STANDS DOWN with an explicit "dead, heartbeat Ns ago"
message, an operator can force-reclaim immediately via ``lease-release``, and
any reap-and-reclaim of a supposedly-live holder now emits a structured
WARNING (see claim_lease)."""


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
            # Reaped-and-reclaimed a stale lease. The displaced holder was
            # SUPPOSED to be dead (dead pid AND heartbeat aged past
            # LEASE_HEARTBEAT_TTL), but this is exactly the path that -- with a
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

    Fail-soft is unchanged: an absent lease is ABSENT and an OSError from
    ``os.utime`` is FAULTED after a logger.error -- never a raise, so a
    lease-substrate fault cannot interrupt a watcher's main loop.
    """
    path = lease_path_for_name(name, root=root)
    terminal, holder = _check_lease_ownership(path, slug=slug, force=force, verb='heartbeat_lease')
    if terminal is not None:
        return terminal
    try:
        os.utime(path, None)
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
    now: datetime | None = None,
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

    - lease absent -> (ABSENT, None), and QUIET: an idempotent release is
      not a failure and must not manufacture an ERROR.
    - body unreadable/corrupt -> not owned by anybody, so REFUSED (fail
      TOWARD held, mirroring _read_lease_holder_state's own documented rule
      -- a corrupt lease is not stranded: reap_stale_leases still removes it
      under the ``corrupt`` rule once past LEASE_HEARTBEAT_TTL, and --force
      covers immediate operator recovery).
    - slug matches -> proceed.
    - slug mismatch -> REFUSED + logger.error naming the verb, the lease,
      the REAL holder (slug + pid), the heartbeat age and the requester, so
      the refusal is greppable and both parties are attributable (INV-2).
    - *force* overrides either refusal, at WARNING, naming both parties.
    """
    if now is None:
        now = datetime.now(UTC)
    if not path.exists():
        return LeaseMutation.ABSENT, None

    holder, _holder_alive, age_secs = _read_lease_holder_state(path, now=now)
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
    """Build + write a LAUNCHING record from CLAUDE_SPAWN_* env; return its dir.

    Also opportunistically drives the liveness sweeps
    (``mark_orphaned_sessions_exited`` and, task 2934,
    ``mark_windowless_wm_sessions_exited``) AND a bounded prune
    (``reap_stale_records(limit=REAP_BATCH_LIMIT)``): none of these sweeps has
    a periodic production driver of its own (CLI-only), so every spawn is
    what drains prior orphaned (unclean-death) and windowless-wm records to
    ``exited`` AND reclaims terminal/stale record dirs from disk -- the
    backlog is bounded by spawn rate and self-limits as it drains, over
    successive spawns for the bounded prune. Each sweep is wrapped in its OWN
    fail-soft guard: a sweep fault must never raise here and must never write
    to stdout, or it would corrupt the printed record dir spawn-claude.sh
    captures into ``SESSION_RECORD_DIR`` -- and one sweep's fault must not
    skip the others.

    Cost model (reviewer-flagged): the pid/TTL sweep is O(N) in the number of
    session directories -- one ``iterdir`` + one ``record.json`` parse per
    peer, plus a second read+write for each record it actually marks -- and
    runs synchronously in this call, so every spawn pays for scanning all
    of its peers, not just its own write. The wm-window sweep adds ONE
    ``wmctrl -l`` subprocess per spawn on top of that same O(N) scan (fail-
    soft: a missing binary or nonzero rc just short-circuits it to a no-op,
    per ``_wmctrl_live_window_ids``). This is a deliberate trade-off given
    neither sweep has a periodic/out-of-band driver to run on instead (see
    above); it is bounded and self-limiting (the backlog this drains only
    shrinks the cost over time), acceptable at today's fleet sizes, and the
    right place to revisit if spawn latency ever becomes a problem is a
    dedicated periodic timer (systemd/cron) rather than this synchronous
    call -- out of this task's module scope (would touch harness/systemd).
    The bounded prune (``limit=REAP_BATCH_LIMIT``) caps its own scan/rmtree
    cost per call regardless of backlog size; the CLI ``reap`` verb
    (``_run_reap``) remains the operator's unbounded full-drain path.
    """
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
    with contextlib.suppress(Exception):
        mark_orphaned_sessions_exited()
    with contextlib.suppress(Exception):
        mark_windowless_wm_sessions_exited()
    with contextlib.suppress(Exception):
        reap_stale_records(limit=REAP_BATCH_LIMIT)
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
    """Run the canonical ``reap`` verb: mark orphans/windowless-wm exited, THEN delete stale dirs.

    Mark-then-delete keeps `reap` complete: a record either liveness sweep
    would mark exited this same pass must not simultaneously be eligible
    for stale_pid deletion (marking bumps its mtime, so it lands back
    within reap_stale_records' NON_TERMINAL_HEARTBEAT_TTL heartbeat grace) --
    it only becomes reapable later, once it ages past TERMINAL_TTL as any
    other terminal record does. This is why ``mark_windowless_wm_sessions_exited``
    (task 2934) runs here too, alongside ``mark_orphaned_sessions_exited`` --
    it reaps exactly the live-pid / age<1h wm-display zombies (a closed
    terminal window with a still-alive launcher_pid) the pid/TTL sweep is
    forced to keep, so `reap` must drive both before deleting.
    """
    mark_orphaned_sessions_exited()
    mark_windowless_wm_sessions_exited()
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


def _run_lease_mutation(
    verb: str,
    name: str,
    slug: str,
    force: bool,
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
        force,
        lambda: heartbeat_lease(name, slug=slug, force=force),
    )


def _run_lease_release(name: str, slug: str, force: bool = False) -> None:
    _run_lease_mutation(
        'lease-release',
        name,
        slug,
        force,
        lambda: release_lease(name, slug=slug, force=force),
    )


def _run_lease_reap() -> list[ReapedLease]:
    return reap_stale_leases()


def _run_write_decision(
    decision_id: str,
    project: str,
    text: str,
    task_id: str | None,
    escalation_id: str | None,
    session_id: str | None,
    severity: str = '',
    escalations_dir: str = '',
) -> None:
    """Run the ``write-decision`` verb (Fleet Cockpit C8: park-to-registry).

    Files an OPEN DecisionRecord from a watcher's park / tell-the-human
    moment, IN ADDITION to its in-session note / afk-digest line, so the
    cockpit decision queue (C5b) becomes the primary return-triage surface.
    ``state`` is left at its DecisionState.OPEN default (a watcher only ever
    files open decisions; state transitions are the cockpit's job via
    update_decision_state). Root resolves via $CLAUDE_FLEET_ROOT, same as
    every other verb.

    ``severity`` threads the parked escalation's severity (info|blocking|
    critical|urgent) onto the record so the cockpit decision queue can
    weight this ask (Fleet Cockpit F7 fix 1); defaults to '' when the
    caller doesn't supply one.

    ``escalations_dir`` names the escalation QUEUE *escalation_id* belongs
    to, and is stored NORMALIZED (see normalize_escalations_dir). A watcher
    passes the SAME queue dir it later reaps with, so this fleet-global
    decision can be joined back to the right per-queue escalation-id
    namespace: ids are unique only within one queue, and a project may run
    several (task 3528). Omitting it stores '' and keeps today's
    project-only-scoped reaper behaviour, so no existing caller breaks.

    On success, prints the filed record's id (mirrors `launching` printing
    the record dir and `lease-claim` printing `decision=`) so the caller can
    cross-link it into its in-session note or afk-digest line. write_decision
    is itself self-guarding fail-soft (never raises), so a fault there is
    silently skipped here rather than printed as a false confirmation.

    Intentionally omits DecisionRecord's ``options`` field: C8 watchers only
    ever file plain open/text decisions, and the peer `update_decision_state`
    / `set_manual_boost` helpers already own post-filing mutation (state,
    boost). If a watcher ever needs to offer candidate answers up front, add
    an optional repeatable ``--option`` flag here rather than widening this
    verb's other args.
    """
    record = DecisionRecord(
        id=decision_id,
        project=project,
        text=text,
        filed_at=datetime.now(UTC).isoformat(),
        task_id=task_id,
        escalation_id=escalation_id,
        session_id=session_id,
        severity=severity,
        escalations_dir=normalize_escalations_dir(escalations_dir),
    )
    if write_decision(record):
        print(record.id)


def _run_reap_decisions(project: str, escalations_dir: str) -> None:
    """Run the ``reap-decisions`` verb (Fleet Cockpit C8: close-on-resolve driver).

    Builds the production ``escalation_status`` closure for
    reap_answered_decisions. The (fleet-global decisions <-> per-queue
    escalations) join is scoped on TWO axes, and a decision failing either
    one is left alone (returns None -- unresolved):

    1. PROJECT. DecisionRecords are fleet-global
       (``~/.claude/fleet/decisions/``) but escalations are per-project
       (``<project_root>/data/escalations``), so a decision belonging to a
       DIFFERENT project is skipped. Scoping this way keeps the join correct
       without fragile project_id -> path auto-discovery.
    2. QUEUE (task 3528). An escalation id (``esc-<taskid>-<n>``) is unique
       only WITHIN one queue, and a project can run several: dark_factory
       runs ``data/escalations`` (orchestrator) and
       ``data/reconciliation/escalations`` (recon watcher) over the SAME id
       namespace. Project scoping alone therefore let either watcher's
       reaper resolve the OTHER watcher's decisions against its own queue --
       observed with ``esc-3036-1``, where an unrelated resolved
       orchestrator escalation silently closed a still-pending recon
       blocking gate, which then sat invisible in the cockpit for ~7 days
       (15 ids were resolved in both queues at time of measurement). So a
       decision stamped with a queue OTHER than *escalations_dir* is
       skipped. Both sides go through normalize_escalations_dir, including
       the decision's own stored value, so equivalent spellings match and a
       hand-written record is compared honestly rather than fail-open.

    A decision carrying NO queue (``escalations_dir == ''`` -- every record
    filed before that field existed) falls back to axis 1 alone, keeping
    today's exact behaviour for the existing population rather than changing
    it under the human; the two watchers stamp the queue on newly-filed
    decisions, so the unprotected set only shrinks. Task 3640 then BACK-FILLED
    the pre-existing open population, so that fallback set is drained rather
    than merely shrinking. Otherwise the closure defers to
    read_escalation_status against *escalations_dir*. Root resolves via
    $CLAUDE_FLEET_ROOT, same as every other verb.

    UNKNOWN QUEUE (task 3640). A decision stamped ``UNKNOWN_QUEUE`` is
    refused by name: its owning queue was investigated and could not be
    determined, so NO reaper may close it and it stays a visible cockpit row
    until a human closes it. Honestly: the axis-2 compare above would already
    refuse it, since it refuses ANY truthy stamp that is not this reaper's
    queue -- so the by-name guard changes no outcome today. It is here so the
    policy is INTENTIONAL and named rather than an accident of string
    inequality that a future refactor of that compare could silently remove,
    and because it closes the one degenerate case the compare genuinely does
    not cover: a reaper invoked with the sentinel as its OWN
    ``--escalations-dir`` makes both sides compare EQUAL, and the record then
    survives only because read_escalation_status happens to find no queue at
    that bogus relative path.

    Both guards are fail-OPEN: on any doubt the decision stays OPEN and
    visible in the cockpit queue, matching reap_answered_decisions' contract
    that any status not in DECISION_CLOSE_MAP leaves a decision OPEN. The
    asymmetry of harm is the reason -- an over-held decision is a
    human-triageable row, while a falsely closed one is invisible.

    A watcher runs this once per Main Loop cycle (SKILL.md "Closing parked
    decisions on resolve") so a decision it (or anyone -- /unblock, an L2
    cascade, the human) files closes once its escalation is actually
    resolved/dismissed. Prints one ``f'{id} {new_state}'`` line per closed
    decision, mirroring `write-decision` printing the filed id.
    """
    reaper_dir = normalize_escalations_dir(escalations_dir)

    def _status(decision: DecisionRecord) -> str | None:
        if decision.project != project:
            return None
        if decision.escalation_id is None:
            # Belt-and-suspenders, not reachable via reap_answered_decisions
            # today: it already skips any decision with a falsy
            # escalation_id before ever calling this closure. Kept so this
            # narrows `str | None` -> `str` for the read_escalation_status
            # call below (typed to take a `str`) and stays correct if this
            # closure is ever invoked from elsewhere.
            return None
        # Axis 2: normalize the decision's OWN stored value at compare time
        # too, not just at write time -- a raw compare would fail open for
        # any record write-decision did not produce (hand-repaired, migrated
        # between checkouts, or written by a future caller), and a false
        # NON-match is how silent divergence creeps back in.
        decision_dir = normalize_escalations_dir(decision.escalations_dir)
        # Task 3640: refuse an undeterminable-queue record BY NAME, before the
        # inequality compare below -- see the UNKNOWN QUEUE paragraph above
        # for why this deliberately-redundant-today guard exists.
        if decision_dir == UNKNOWN_QUEUE:
            return None
        if decision_dir and decision_dir != reaper_dir:
            return None
        # RAW escalations_dir here, deliberately: only the comparison needs a
        # canonical spelling; the read keeps using exactly what the caller
        # passed.
        return read_escalation_status(escalations_dir, decision.escalation_id)

    for reaped in reap_answered_decisions(escalation_status=_status):
        print(f'{reaped.id} {reaped.new_state}')


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

    # Ownership (task 3994): both MUTATING verbs require the claiming slug.
    # required=True is deliberate -- a silent no-slug fallback would preserve
    # the "any caller may evict/refresh any lease" defect indefinitely, since
    # nothing would ever force the call sites to change. A stale invocation
    # now fails to ACT rather than succeeding at evicting a live holder.
    for _lease_mutation_p in (lease_heartbeat_p, lease_release_p):
        _lease_mutation_p.add_argument(
            '--slug',
            required=True,
            help='the slug you claimed this lease with; a mismatch is refused',
        )
        _lease_mutation_p.add_argument(
            '--force',
            action='store_true',
            help='operator recovery: act even when you are not the holder; logged loudly',
        )

    sub.add_parser('lease-reap', help='sweep and remove stale leases')

    write_decision_p = sub.add_parser(
        'write-decision',
        help='file an OPEN DecisionRecord (Fleet Cockpit C8: park-to-registry)',
    )
    write_decision_p.add_argument('--id', required=True, help="this decision's id")
    write_decision_p.add_argument('--project', required=True)
    write_decision_p.add_argument('--text', required=True, help='the decision/question text')
    write_decision_p.add_argument('--task-id', default=None)
    write_decision_p.add_argument('--escalation-id', default=None)
    write_decision_p.add_argument('--session-id', default=None)
    write_decision_p.add_argument(
        '--severity',
        default='',
        help='escalation severity to weight this ask (info|blocking|critical|urgent)',
    )
    write_decision_p.add_argument(
        '--escalations-dir',
        default='',
        help=(
            "the escalation queue dir this decision's --escalation-id belongs to; "
            'scopes the reaper (see reap-decisions)'
        ),
    )

    reap_decisions_p = sub.add_parser(
        'reap-decisions',
        help='close OPEN decisions whose escalation has resolved/dismissed (Fleet Cockpit C8)',
    )
    reap_decisions_p.add_argument('--project', required=True)
    reap_decisions_p.add_argument('--escalations-dir', required=True)

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
            _run_lease_heartbeat(args.name, args.slug, args.force)
        elif args.verb == 'lease-release':
            _run_lease_release(args.name, args.slug, args.force)
        elif args.verb == 'lease-reap':
            _run_lease_reap()
        elif args.verb == 'write-decision':
            _run_write_decision(
                args.id,
                args.project,
                args.text,
                args.task_id,
                args.escalation_id,
                args.session_id,
                args.severity,
                args.escalations_dir,
            )
        elif args.verb == 'reap-decisions':
            _run_reap_decisions(args.project, args.escalations_dir)
    except Exception:
        logger.error('session_registry %s failed', args.verb, exc_info=True)
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
