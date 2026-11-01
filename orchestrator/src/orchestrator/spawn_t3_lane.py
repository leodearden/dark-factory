"""orchestrator.spawn_t3_lane — the /spawn lane that hosts a session in a T3 Code thread.

Design: plans/spawn-t3-lane-design-2026-10-25.md. ``skills/spawn/spawn-claude.sh``
runs this file by absolute path with ``PYTHONPATH=<repo>/orchestrator/src``,
the import contract ``session_hooks.py`` documents, and maps each verb's
verdict onto its own exit codes.

Every verb prints one line on stdout, a status word and then ``key=value``
tokens (``%``-escaped, ``none`` for an absent value), and exits 0. An internal
fault prints nothing on stdout, a traceback on stderr, and exits 3.

Two facts of ``t3`` v0.1.2 shape this module:

* No ``t3`` command prints a project's ``workspaceRoot``: ``thread start``
  resolves a path to the deepest registered project containing it through the
  MCP tool ``t3_project_list`` and runs the thread at that project's root
  (``apps/server/src/cli/thread.ts::resolveProject``). ``launch`` therefore
  reads the registered roots from the server's state store,
  ``$T3CODE_HOME/userdata/statev2.sqlite`` table ``projection_projects``, the
  table ``t3_project_list`` is served from
  (``apps/server/src/orchestration-v2/ProjectStore.ts``), read-only, and
  refuses when it cannot.
* ``thread list --json`` carries active threads only
  (``ThreadManagementService.listProjectThreads`` reads location ``active``),
  so a thread missing from it is archived or deleted; ``wait`` tells which
  from ``thread show --json``'s ``thread.archived``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from orchestrator import session_registry
from orchestrator.session_ownership_probes import ProcessTree

MESSAGE_CHAR_LIMIT = 120_000
CLAUDE_PROVIDER_INSTANCE = 'claudeAgent'
PENDING_RUN_STATUSES = frozenset({'preparing', 'queued', 'starting'})
ENDED_RUN_STATUSES = frozenset({'failed', 'cancelled', 'interrupted', 'rolled_back'})
STARTUP_FAILURE_STATUSES = frozenset({'failed', 'cancelled', 'rolled_back'})
SERVER_START_TOLERANCE_SECS = 120
STARTED_POLL_SECS = 5.0
DEFAULT_WAIT_POLL_SECS = 30.0
ENDED_RUN_GRACE_SECS = 120.0
UNREACHABLE_GRACE_SECS = 600.0
INTERNAL_FAULT_EXIT = 3
_T3_TIMEOUT_SECS = 300
_PROCESS_TREE_MAX_HOPS = 64
_TRANSCRIPT_HEAD_LINES = 200


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _token(value: object) -> str:
    text = '' if value is None else str(value)
    return quote(text, safe='/-_.~:@+,') if text else 'none'


def _verdict(word: str, **fields: object) -> str:
    return ' '.join([word, *(f'{key}={_token(value)}' for key, value in fields.items())])


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The t3 CLI seam: every call to t3 goes through run_t3
# ---------------------------------------------------------------------------


class T3CommandFailed(Exception):
    """``t3`` is missing, exited non-zero, timed out, or printed no JSON."""


def run_t3(args: Sequence[str], stdin: str | None = None) -> Any:
    """Run ``t3 <args>`` and return its stdout parsed as JSON."""
    try:
        completed = subprocess.run(
            ['t3', *args],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=_T3_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise T3CommandFailed(f't3 {args[:2]}: {exc}') from exc
    if completed.returncode != 0:
        raise T3CommandFailed(
            f't3 {args[:2]} exited {completed.returncode}: {completed.stderr.strip()}'
        )
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise T3CommandFailed(f't3 {args[:2]} printed no JSON') from exc


@dataclass(frozen=True)
class ThreadSummary:
    """One entry of ``t3 thread list --json`` (``OrchestratorMcpThreadListItem``)."""

    thread_id: str
    title: str
    status: str
    latest_run_id: str | None
    created_at: datetime | None

    @classmethod
    def from_json(cls, item: Mapping[str, Any]) -> ThreadSummary:
        return cls(
            thread_id=str(item['threadId']),
            title=str(item.get('title') or ''),
            status=str(item.get('status') or ''),
            latest_run_id=item.get('latestRunId'),
            created_at=_parse_instant(item.get('createdAt')),
        )


def _parse_instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def list_threads(cwd: str) -> dict[str, ThreadSummary]:
    """The active threads of the project *cwd* resolves to, keyed by thread id."""
    items = run_t3(['thread', 'list', '--project', cwd, '--json'])
    if not isinstance(items, list):
        raise T3CommandFailed('t3 thread list printed no array')
    try:
        summaries = [ThreadSummary.from_json(item) for item in items]
    except (KeyError, TypeError, AttributeError) as exc:
        raise T3CommandFailed(f't3 thread list printed a malformed item: {exc!r}') from exc
    return {summary.thread_id: summary for summary in summaries}


def thread_detail(thread_id: str) -> Mapping[str, Any] | None:
    """``thread show --json``'s ``thread`` object, or None when a successful show carries none.

    Raises ``T3CommandFailed`` when the show itself fails.
    """
    shown = run_t3(['thread', 'show', thread_id, '--messages', '1', '--json'])
    thread = shown.get('thread') if isinstance(shown, dict) else None
    return thread if isinstance(thread, dict) else None


# ---------------------------------------------------------------------------
# T3 server facts on disk
# ---------------------------------------------------------------------------


def t3_userdata_dir(env: Mapping[str, str]) -> Path:
    home = env.get('T3CODE_HOME') or str(Path.home() / '.t3')
    return Path(home) / 'userdata'


def read_server_runtime(env: Mapping[str, str]) -> Mapping[str, Any] | None:
    try:
        runtime = json.loads((t3_userdata_dir(env) / 'server-runtime.json').read_text())
    except (OSError, ValueError):
        return None
    return runtime if isinstance(runtime, dict) else None


@dataclass(frozen=True)
class T3Project:
    project_id: str
    workspace_root: str


class ProjectStoreUnreadable(Exception):
    """The T3 state store or its ``projection_projects`` table cannot be read."""


def registered_projects(env: Mapping[str, str]) -> list[T3Project]:
    store = t3_userdata_dir(env) / 'statev2.sqlite'
    if not store.is_file():
        raise ProjectStoreUnreadable(f'no T3 state store at {store}')
    try:
        with contextlib.closing(
            sqlite3.connect(f'{store.as_uri()}?mode=ro', uri=True, timeout=10)
        ) as connection:
            rows = connection.execute(
                'SELECT project_id, workspace_root FROM projection_projects '
                'WHERE deleted_at IS NULL'
            ).fetchall()
    except sqlite3.Error as exc:
        raise ProjectStoreUnreadable(f'{store}: {exc}') from exc
    return [T3Project(project_id=str(pid), workspace_root=str(root)) for pid, root in rows]


def containing_project(cwd: str, projects: Sequence[T3Project]) -> T3Project | None:
    """The project ``t3`` resolves *cwd* into: the deepest real root containing it."""
    target = os.path.realpath(cwd)
    containing = [
        project
        for project in projects
        if _is_within(target, os.path.realpath(project.workspace_root))
    ]
    return max(containing, key=lambda p: len(os.path.realpath(p.workspace_root)), default=None)


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip('/') + '/')


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcStat:
    """The ``/proc/<pid>/stat`` fields detect reads: ppid (4), tty_nr (7), starttime (22)."""

    ppid: int
    tty_nr: int
    start_ticks: int


def read_proc_stat(pid: int, proc_root: Path) -> ProcStat | None:
    try:
        raw = (proc_root / str(pid) / 'stat').read_text()
        fields = raw[raw.rindex(')') + 2 :].split()
        return ProcStat(ppid=int(fields[1]), tty_nr=int(fields[4]), start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def _boot_epoch(proc_root: Path) -> int | None:
    try:
        for line in (proc_root / 'stat').read_text().splitlines():
            key, _, value = line.partition(' ')
            if key == 'btime':
                return int(value)
    except (OSError, ValueError):
        return None
    return None


def _ancestors(pid: int, proc_root: Path) -> Iterator[tuple[int, ProcStat]]:
    stat = read_proc_stat(pid, proc_root)
    for _ in range(_PROCESS_TREE_MAX_HOPS):
        if stat is None or stat.ppid <= 1:
            return
        pid = stat.ppid
        stat = read_proc_stat(pid, proc_root)
        if stat is not None:
            yield pid, stat


def is_t3_hosted(env: Mapping[str, str], proc_root: Path = Path('/proc')) -> bool:
    """Whether this process runs under the T3 server named by ``server-runtime.json``, with no tty."""
    own = read_proc_stat(os.getpid(), proc_root)
    runtime = read_server_runtime(env)
    boot = _boot_epoch(proc_root)
    if own is None or runtime is None or boot is None or own.tty_nr != 0:
        return False
    server_pid = runtime.get('pid')
    started_at = _parse_instant(runtime.get('startedAt'))
    if not isinstance(server_pid, int) or started_at is None:
        return False
    for pid, stat in _ancestors(os.getpid(), proc_root):
        if pid == server_pid:
            started = boot + stat.start_ticks / os.sysconf('SC_CLK_TCK')
            return abs(started - started_at.timestamp()) <= SERVER_START_TOLERANCE_SECS
    return False


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def _record_key(record_dir: str) -> tuple[str, Path]:
    return session_registry.slug_root_from_record_dir(record_dir)


def _message_for(prompt: str, record_dir: str) -> str:
    if len(prompt) <= MESSAGE_CHAR_LIMIT:
        return prompt
    if record_dir:
        spill = Path(record_dir) / 'prompt.md'
        spill.write_text(prompt, encoding='utf-8')
    else:
        fd, name = tempfile.mkstemp(prefix='spawn-t3-prompt-', suffix='.md')
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(prompt)
        spill = Path(name)
    return _spill_pointer(spill)


def _spill_pointer(spill: Path) -> str:
    return f'Read {spill} and follow it as your brief; that file holds your whole task.'


def _start_argv(cwd: str, title: str, model: str | None) -> list[str]:
    model_flags = ['--provider', CLAUDE_PROVIDER_INSTANCE, '--model', model] if model else []
    return ['thread', 'start', '--json', '--title', title, *model_flags, cwd, '-']


@dataclass(frozen=True)
class LaunchedThread:
    thread_id: str
    project_id: str | None
    run_id: str | None


def _start_thread(cwd: str, title: str, model: str | None, message: str) -> LaunchedThread:
    launched = run_t3(_start_argv(cwd, title, model), stdin=message)
    if not isinstance(launched, dict) or not launched.get('threadId'):
        raise T3CommandFailed('t3 thread start printed no threadId')
    return LaunchedThread(
        thread_id=str(launched['threadId']),
        project_id=launched.get('projectId'),
        run_id=launched.get('runId'),
    )


def _find_launched_after_failure(
    cwd: str, title: str, launched_at: datetime, project: T3Project
) -> LaunchedThread | None:
    """A thread titled *title* created since *launched_at*: a launch the CLI failed to report."""
    try:
        threads = list_threads(cwd)
    except T3CommandFailed as exc:
        logger.warning(f'cannot list threads after a failed launch: {exc}')
        return None
    matches = [
        summary
        for summary in threads.values()
        if summary.title == title and summary.created_at and summary.created_at >= launched_at
    ]
    if len(matches) > 1:
        logger.warning(f'{len(matches)} threads titled {title!r} since the launch; not guessing')
    if len(matches) != 1:
        return None
    [summary] = matches
    return LaunchedThread(summary.thread_id, project.project_id, summary.latest_run_id)


def _stamp_launched_record(
    record_dir: str, title: str, thread_id: str, parent: str | None
) -> None:
    """One read-modify-write: the t3 display, status running, and the parent when given."""
    slug, root = _record_key(record_dir)
    try:
        record = session_registry.read_record(slug, root=root)
        record.display = session_registry.Display(
            kind=session_registry.DisplayKind.T3.value, wm_title=title, t3_thread_id=thread_id
        )
        if parent is not None:
            record.parent_session_id = parent
        session_registry.write_record(
            session_registry.apply_refresh(slug, record, status=session_registry.Status.RUNNING),
            root=root,
        )
    except Exception as exc:
        logger.warning(f'registry stamp failed for {record_dir}: {exc}')


def _usable_parent(parent: str | None) -> str | None:
    if parent and not session_registry.is_record_key(parent):
        logger.warning(f'ignoring --parent {parent!r}: not a session record key')
        return None
    return parent or None


def launch(
    cwd: str,
    title: str,
    prompt_file: str,
    record_dir: str,
    model: str | None,
    env: Mapping[str, str],
    *,
    parent: str | None = None,
) -> str:
    if shutil.which('t3') is None:
        return _verdict('no-launcher', reason='t3-missing')
    if read_server_runtime(env) is None:
        return _verdict('no-launcher', reason='server-runtime-missing')
    try:
        project = containing_project(cwd, registered_projects(env))
    except ProjectStoreUnreadable as exc:
        logger.warning(f'cannot verify that {cwd} is a T3 project root: {exc}')
        return _verdict('refused', reason='cwd-not-project-root', detail='unverifiable')
    if project is None or os.path.realpath(project.workspace_root) != os.path.realpath(cwd):
        detail = os.path.realpath(project.workspace_root) if project else None
        return _verdict('refused', reason='cwd-not-project-root', detail=detail)

    _sweep_project_listing(cwd)
    message = _message_for(Path(prompt_file).read_text(encoding='utf-8'), record_dir)
    launched_at = datetime.now(UTC)
    try:
        launched = _start_thread(cwd, title, model, message)
    except T3CommandFailed as exc:
        logger.warning(str(exc))
        recovered = _find_launched_after_failure(cwd, title, launched_at, project)
        if recovered is None:
            return _verdict('launch-failed')
        launched = recovered
    if record_dir:
        _stamp_launched_record(record_dir, title, launched.thread_id, _usable_parent(parent))
    return _verdict(
        'started',
        thread_id=launched.thread_id,
        project_id=launched.project_id,
        run_id=launched.run_id,
    )


# ---------------------------------------------------------------------------
# Transcript discovery and binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transcript:
    session_id: str
    path: Path
    mtime: float


def _transcript_dirs(cwd: str) -> list[Path]:
    roots = [Path(os.environ.get('CLAUDE_PROJECTS_DIR') or Path.home() / '.claude' / 'projects')]
    if os.environ.get('CLAUDE_CONFIG_DIR'):
        roots.append(Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects')
    encoded = [session_registry.encode_cwd(path) for path in (cwd, os.path.realpath(cwd))]
    return list(dict.fromkeys(root / name for root in roots for name in encoded))


def _first_user_text(path: Path) -> str:
    """The text of the transcript's first non-meta user message, or ''."""
    try:
        with path.open(encoding='utf-8', errors='replace') as handle:
            for _, line in zip(range(_TRANSCRIPT_HEAD_LINES), handle, strict=False):
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get('type') == 'user' and not entry.get('isMeta'):
                    return _message_text(entry.get('message'))
    except OSError:
        return ''
    return ''


def _message_text(message: object) -> str:
    content = message.get('content') if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(
            str(block.get('text') or '')
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return ''


def _transcripts_since(cwd: str, since: float) -> Iterator[Transcript]:
    for directory in _transcript_dirs(cwd):
        with contextlib.suppress(OSError):
            for path in directory.glob('*.jsonl'):
                with contextlib.suppress(OSError):
                    mtime = path.stat().st_mtime
                    if mtime >= since:
                        yield Transcript(session_id=path.stem, path=path, mtime=mtime)


def _launch_markers(record_dir: str) -> tuple[str, str]:
    """What only this spawn's own first message says: spawn-claude.sh's result trailer, or the spill pointer."""
    directory = Path(record_dir)
    return (
        f'write your outcome to: {directory / session_registry.RESULT_FILENAME}',
        _spill_pointer(directory / 'prompt.md'),
    )


def discover_transcript(cwd: str, record_dir: str, since: float) -> Transcript | None:
    """The newest transcript since *since* whose first user message is this spawn's launch message."""
    markers = _launch_markers(record_dir)
    matches = [
        t
        for t in _transcripts_since(cwd, since)
        if any(marker in _first_user_text(t.path) for marker in markers)
    ]
    return max(matches, key=lambda t: t.mtime, default=None)


def _transcript_mtime(cwd: str, session_id: str) -> float | None:
    for directory in _transcript_dirs(cwd):
        with contextlib.suppress(OSError):
            return (directory / f'{session_id}.jsonl').stat().st_mtime
    return None


@dataclass(frozen=True)
class Binding:
    session_id: str
    hand_slug: str | None
    bound: bool


def _is_hand_launched(record: session_registry.SessionRecord) -> bool:
    """A spawn's record always has a result file; a hooks-made hand-launched one never does."""
    return record.result_file is None and not session_registry.is_t3_record(record)


def _record_vouching_for(
    session_id: str, spawn_slug: str, root: Path
) -> session_registry.SessionRecord | None:
    """The other record *session_id*'s pointer is vouched for by, if any."""
    slug = session_registry.resolve_session_slug_for_claude_session_id(session_id, root=root)
    if slug is None or slug == spawn_slug:
        return None
    try:
        return session_registry.read_record(slug, root=root)
    except (OSError, session_registry.CorruptSessionRecord) as exc:
        logger.warning('cannot read hand-launched record %s: %s', slug, exc)
        return None


def _fold(hand: session_registry.SessionRecord, root: Path) -> None:
    try:
        session_registry.mark_session_folded(hand.session_slug, root=root)
    except (OSError, session_registry.CorruptSessionRecord) as exc:
        logger.warning('cannot fold hand-launched record %s: %s', hand.session_slug, exc)


def bind_session(record_dir: str, transcript: Transcript) -> Binding:
    """Bind *transcript*'s session to the spawn record, then fold its hand-launched record.

    A session another spawn's record already owns is left alone, unbound.

    The spawn record and both pointers are written before the fold, so a hook
    event landing in between already resolves to the spawn record.
    """
    slug, root = _record_key(record_dir)
    session_id = transcript.session_id
    hand = _record_vouching_for(session_id, slug, root)
    if hand is not None and not _is_hand_launched(hand):
        logger.warning(
            'session %s belongs to spawn record %s; not binding it to %s',
            session_id,
            hand.session_slug,
            slug,
        )
        return Binding(session_id, None, bound=False)
    owner_pid = hand.claude_owner_pid if hand else None
    hand_slug = hand.session_slug if hand else None
    try:
        record = session_registry.read_record(slug, root=root)
        if record.claude_launch_session_id is None:
            record.claude_launch_session_id = session_id
        record.claude_session_id = session_id
        if owner_pid is not None:
            record.claude_owner_pid = owner_pid
        session_registry.write_record(record, root=root)
    except Exception as exc:
        logger.warning(f'cannot bind session {session_id} to {record_dir}: {exc}')
        return Binding(session_id, hand_slug, bound=False)
    session_registry.write_claude_session_pointer(session_id, slug, root=root)
    if owner_pid is not None:
        session_registry.write_session_pointer(owner_pid, slug, root=root)
    if hand is not None:
        _fold(hand, root)
    return Binding(session_id, hand_slug, bound=True)


def _needs_binding(record: session_registry.SessionRecord, cwd: str, found: Transcript) -> bool:
    """Unbound, or bound to a session whose transcript the newer *found* has replaced.

    A bound transcript that does not exist yet (a fresh ``/clear``) is no
    evidence of replacement.
    """
    bound = record.claude_session_id
    if bound is None:
        return True
    if bound == found.session_id:
        return False
    bound_mtime = _transcript_mtime(cwd, bound)
    return bound_mtime is not None and found.mtime > bound_mtime


def maintain_binding(record_dir: str, cwd: str) -> None:
    """Bind an unbound record, or re-bind one whose transcript a newer one naming it replaced."""
    slug, root = _record_key(record_dir)
    record = session_registry.read_record(slug, root=root)
    since = _parse_instant(record.start_ts)
    found = discover_transcript(cwd, record_dir, since.timestamp() if since else 0.0)
    if found is not None and _needs_binding(record, cwd, found):
        bind_session(record_dir, found)


# ---------------------------------------------------------------------------
# await-started
# ---------------------------------------------------------------------------


def await_started(thread_id: str, cwd: str, record_dir: str, timeout: float, since: float) -> str:
    deadline = time.monotonic() + timeout
    while True:
        verdict = _started_verdict(thread_id, cwd, record_dir, since)
        if verdict is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _verdict('timeout')
        time.sleep(min(STARTED_POLL_SECS, remaining))
    if verdict != 'started':
        return _verdict(verdict)
    binding = _bind_within(record_dir, cwd, since, deadline) if record_dir else None
    return _verdict(
        'started',
        sid=binding.session_id if binding else None,
        hand_slug=binding.hand_slug if binding else None,
        bound='yes' if binding and binding.bound else 'no',
    )


def _started_verdict(thread_id: str, cwd: str, record_dir: str, since: float) -> str | None:
    """None while preparing; a startup-failure status is failed-to-start until a transcript names the record."""
    try:
        summary = list_threads(cwd).get(thread_id)
    except T3CommandFailed as exc:
        logger.warning(str(exc))
        return None
    if summary is None or summary.status in PENDING_RUN_STATUSES:
        return None
    if summary.status in STARTUP_FAILURE_STATUSES and not (
        record_dir and discover_transcript(cwd, record_dir, since)
    ):
        return 'failed-to-start'
    return 'started'


def _bind_within(record_dir: str, cwd: str, since: float, deadline: float) -> Binding | None:
    while True:
        found = discover_transcript(cwd, record_dir, since)
        if found is not None:
            return bind_session(record_dir, found)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(STARTED_POLL_SECS, remaining))


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


class ThreadState(StrEnum):
    LISTED = 'listed'
    ARCHIVED = 'archived'
    GONE = 'gone'
    UNREACHABLE = 'unreachable'


@dataclass(frozen=True)
class ThreadObservation:
    state: ThreadState
    status: str | None = None
    latest_run_id: str | None = None


class JoinTracker:
    """Child-mode join over successive observations (design §4.3); ``verdict`` is the exit code or None."""

    def __init__(self) -> None:
        self._ended_run: tuple[str | None, float] | None = None
        self._unreachable_since: float | None = None

    def verdict(self, observation: ThreadObservation, *, result_final: bool, now: float) -> int | None:
        if result_final or observation.state is ThreadState.ARCHIVED:
            return 0
        if observation.state is ThreadState.GONE:
            return 127
        if observation.state is ThreadState.UNREACHABLE:
            if self._unreachable_since is None:
                self._unreachable_since = now
            return 126 if now - self._unreachable_since > UNREACHABLE_GRACE_SECS else None
        self._unreachable_since = None
        if observation.status not in ENDED_RUN_STATUSES:
            self._ended_run = None
            return None
        if self._ended_run is None or self._ended_run[0] != observation.latest_run_id:
            self._ended_run = (observation.latest_run_id, now)
        return 1 if now - self._ended_run[1] >= ENDED_RUN_GRACE_SECS else None


def observe_thread(thread_id: str, cwd: str) -> tuple[ThreadObservation, dict[str, bool] | None]:
    """The thread's state, and the project's listing as thread id -> archived for the sweep."""
    try:
        threads = list_threads(cwd)
    except T3CommandFailed:
        return ThreadObservation(ThreadState.UNREACHABLE), None
    known = dict.fromkeys(threads, False)
    if thread_id in threads:
        return _listed(threads[thread_id]), known
    try:
        detail = thread_detail(thread_id)
    except T3CommandFailed:
        return ThreadObservation(ThreadState.UNREACHABLE), known
    if detail is None:
        return ThreadObservation(ThreadState.GONE), known
    known[thread_id] = detail.get('archived') is True
    if known[thread_id]:
        return ThreadObservation(ThreadState.ARCHIVED), known
    return ThreadObservation(
        ThreadState.LISTED, str(detail.get('status') or ''), detail.get('latestRunId')
    ), known


def _listed(summary: ThreadSummary) -> ThreadObservation:
    return ThreadObservation(ThreadState.LISTED, summary.status, summary.latest_run_id)


def result_is_final(record_dir: str) -> bool:
    if not record_dir:
        return False
    try:
        text = (Path(record_dir) / session_registry.RESULT_FILENAME).read_text(encoding='utf-8')
    except (OSError, ValueError):
        return False
    return session_registry.parse_result_outcome(text) in session_registry.TERMINAL_RESULT_OUTCOMES


def _wait_poll_secs(env: Mapping[str, str]) -> float:
    try:
        return max(0.0, float(env.get('SPAWN_T3_POLL_SECS') or DEFAULT_WAIT_POLL_SECS))
    except ValueError:
        return DEFAULT_WAIT_POLL_SECS


class CallerGone(Exception):
    """The process that started ``wait`` has exited; nobody reads its verdict."""


def wait(
    thread_id: str,
    cwd: str,
    record_dir: str,
    env: Mapping[str, str],
    parent_pid: Callable[[], int] = os.getppid,
) -> str:
    caller = parent_pid()
    tracker = JoinTracker()
    while True:
        if parent_pid() != caller:
            raise CallerGone(f'caller {caller} is gone; stopped waiting on thread {thread_id}')
        listed_at = datetime.now(UTC)
        observation, known = observe_thread(thread_id, cwd)
        if known is not None:
            _sweep_project(cwd, known, listed_at)
        if record_dir:
            _maintain_binding_softly(record_dir, cwd)
        code = tracker.verdict(
            observation, result_final=result_is_final(record_dir), now=time.monotonic()
        )
        if code is not None:
            return _verdict('finished', code=code)
        time.sleep(_wait_poll_secs(env))


def _sweep_project_listing(cwd: str) -> None:
    listed_at = datetime.now(UTC)
    try:
        threads = list_threads(cwd)
    except T3CommandFailed as exc:
        logger.warning(f't3 session sweep skipped, no thread listing: {exc}')
        return
    _sweep_project(cwd, dict.fromkeys(threads, False), listed_at)


def _sweep_project(cwd: str, known: Mapping[str, bool], listed_at: datetime) -> None:
    try:
        session_registry.mark_t3_sessions_exited(cwd, known, listed_at=listed_at)
    except Exception as exc:
        logger.warning(f't3 session sweep failed: {exc}')


def _maintain_binding_softly(record_dir: str, cwd: str) -> None:
    try:
        maintain_binding(record_dir, cwd)
    except Exception as exc:
        logger.warning(f'binding upkeep failed for {record_dir}: {exc}')


# ---------------------------------------------------------------------------
# self-record
# ---------------------------------------------------------------------------


def nearest_claude_ancestor(tree: ProcessTree) -> int | None:
    pid: int | None = tree.hook_parent_pid()
    for _ in range(_PROCESS_TREE_MAX_HOPS):
        if pid is None or pid <= 1:
            return None
        if tree.comm_of(pid) == 'claude':
            return pid
        pid = tree.parent_pid_of(pid)
    return None


def self_record(tree: ProcessTree) -> str:
    claude_pid = nearest_claude_ancestor(tree)
    slug = session_registry.resolve_session_slug_for_pid(claude_pid) if claude_pid else None
    parent = None
    if slug is not None:
        try:
            parent = session_registry.read_record(slug).parent_session_id
        except (OSError, session_registry.CorruptSessionRecord) as exc:
            logger.warning('cannot read own record %s: %s', slug, exc)
    return f'slug={_token(slug)} parent={_token(parent)}'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='spawn_t3_lane')
    verbs = parser.add_subparsers(dest='verb', required=True)
    verbs.add_parser('detect', help='print t3 when this process is hosted by a T3 thread')

    launch_p = verbs.add_parser('launch', help='start a T3 thread for a spawn')
    launch_p.add_argument('--cwd', required=True)
    launch_p.add_argument('--title', required=True)
    launch_p.add_argument('--prompt-file', required=True)
    launch_p.add_argument('--record', required=True, help="record dir, '' when none")
    launch_p.add_argument('--model', default=None)
    launch_p.add_argument('--parent', default=None, help="the spawner's record slug")

    started_p = verbs.add_parser('await-started', help='wait for the thread to start, then bind')
    started_p.add_argument('--thread-id', required=True)
    started_p.add_argument('--cwd', required=True)
    started_p.add_argument('--record', required=True)
    started_p.add_argument('--timeout', required=True, type=float)
    started_p.add_argument('--since', required=True, type=float)

    wait_p = verbs.add_parser('wait', help='block until the thread is finished (child mode)')
    wait_p.add_argument('--thread-id', required=True)
    wait_p.add_argument('--cwd', required=True)
    wait_p.add_argument('--record', required=True)

    verbs.add_parser('self-record', help="print this session's own record slug and parent")
    return parser


def _dispatch(args: argparse.Namespace) -> str:
    if getattr(args, 'cwd', None):
        args.cwd = os.path.abspath(args.cwd)
    if args.verb == 'detect':
        return 't3' if is_t3_hosted(os.environ) else 'not-t3'
    if args.verb == 'launch':
        return launch(
            args.cwd,
            args.title,
            args.prompt_file,
            args.record,
            args.model,
            os.environ,
            parent=args.parent,
        )
    if args.verb == 'await-started':
        return await_started(args.thread_id, args.cwd, args.record, args.timeout, args.since)
    if args.verb == 'wait':
        return wait(args.thread_id, args.cwd, args.record, os.environ)
    return self_record(ProcessTree())


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        line = _dispatch(args)
    except CallerGone as gone:
        print(f'spawn_t3_lane: {gone}', file=sys.stderr)
        return INTERNAL_FAULT_EXIT
    except Exception:
        traceback.print_exc()
        return INTERNAL_FAULT_EXIT
    print(line)
    return 0


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, format='spawn_t3_lane: %(message)s')
    sys.exit(main(sys.argv[1:]))
