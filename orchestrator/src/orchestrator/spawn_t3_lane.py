"""orchestrator.spawn_t3_lane — the /spawn lane that hosts a session in a T3 Code thread.

Design: plans/spawn-t3-lane-design-2026-10-25.md. ``skills/spawn/spawn-claude.sh``
runs this file by absolute path with ``PYTHONPATH=<repo>/orchestrator/src``,
the import contract ``session_hooks.py`` documents, and maps each verb's
verdict onto its own exit codes.

Every verb prints one line on stdout, a status word and then ``key=value``
tokens (``%``-escaped, ``none`` for an absent value), and exits 0. An internal
fault prints nothing on stdout, a traceback on stderr, and exits 3. ``pair``
and ``status`` manage this host's credential for T3's ``/mcp``
(``t3_mcp_client``): ``pair`` prints ``paired`` or ``pair-failed``, and
``status`` prints ``paired``, ``expired``, ``unpaired`` or ``status-failed``.

``launch``, ``provider``, ``await-started`` and ``wait`` reach T3 only through
``t3_mcp_client``'s ``/mcp`` client; ``detect`` and ``self-record`` read the
process tree and ``server-runtime.json``. The tool contract the lane relies on
is plans/spawn-t3-lane-mcp-and-host-registry-prd.md's.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import quote

from orchestrator import session_registry, spawn_t3_provider, t3_mcp_client
from orchestrator.session_ownership_probes import ProcessTree

MESSAGE_CHAR_LIMIT = 120_000
PAGE_LIMIT = 100
LAUNCH_ATTEMPTS = 2
PENDING_RUN_STATUSES = frozenset({'preparing', 'queued', 'starting'})
ENDED_RUN_STATUSES = frozenset({'failed', 'cancelled', 'interrupted', 'rolled_back'})
STARTUP_FAILURE_STATUSES = frozenset({'failed', 'cancelled', 'rolled_back'})
SERVER_START_TOLERANCE_SECS = 120
STARTED_POLL_SECS = 5.0
WAIT_SLICE_SECS = min(600, t3_mcp_client.T3_MAX_WAIT_SLICE_SECS)
CALL_TIMEOUT_SECS = 60.0
"""The reply bound on every lane MCP call but ``t3_thread_wait``."""
SETTLED_RECHECK_SECS = 60.0
"""The pause after a wait slice T3 did not hold: it answers at once for a settled run."""
ENDED_RUN_GRACE_SECS = 120.0
UNREACHABLE_GRACE_SECS = 600.0
INTERNAL_FAULT_EXIT = 3
_PROCESS_TREE_MAX_HOPS = 64
_READ_TIMEOUT_FLOOR_SECS = 0.1
"""A deadline-shortened reply bound never drops to a socket timeout of 0, which does not block at all."""
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


def _parse_instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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


# ---------------------------------------------------------------------------
# T3 over MCP: the connection, paged listings, projects
# ---------------------------------------------------------------------------


def _detail_field(detail: str | None) -> dict[str, str]:
    return {} if detail is None else {'detail': detail}


@dataclass(frozen=True)
class NoLauncher:
    """No MCP session with T3: server-runtime.json names no usable origin, or connecting failed."""

    reason: RuntimeOriginFailure | t3_mcp_client.ConnectFailure | t3_mcp_client.DescriptorFailure
    detail: str | None = None

    def verdict(self) -> str:
        credential_fault = isinstance(self.reason, t3_mcp_client.ConnectFailure)
        word = f't3-mcp-{self.reason.value}' if credential_fault else self.reason.value
        return _verdict('no-launcher', reason=word, **_detail_field(self.detail))


class PlacementRefusal(StrEnum):
    PROJECT_ARCHIVED = 'project-archived'
    CWD_OUTSIDE_PROJECTS = 'cwd-outside-projects'
    CWD_NOT_PROJECT_WORKTREE = 'cwd-not-project-worktree'


@dataclass(frozen=True)
class Refused:
    """No live T3 project hosts a thread in this cwd."""

    reason: PlacementRefusal
    detail: str | None = None

    def verdict(self) -> str:
        return _verdict('refused', reason=self.reason.value, **_detail_field(self.detail))


@dataclass(frozen=True)
class LaunchFailed:
    detail: str

    def verdict(self) -> str:
        return _verdict('launch-failed', detail=self.detail)


NotLaunched = NoLauncher | Refused | LaunchFailed


def _connect(
    env: Mapping[str, str], repair: t3_mcp_client.CodeSource | None
) -> t3_mcp_client.T3McpClient | NoLauncher:
    """A client on this host's credential, self-pairing through *repair* when it has none."""
    origin = _runtime_origin(env)
    if isinstance(origin, RuntimeOriginFailed):
        return NoLauncher(origin.reason, origin.detail)
    connected = t3_mcp_client.connect(origin, repair=repair)
    if isinstance(connected, t3_mcp_client.ConnectFailed):
        return NoLauncher(connected.reason, connected.detail)
    return connected


class _Session:
    """One verb's MCP client across its polls, connected again on the poll after a failed connect.

    A self-pair that failed is not tried again: each try mints a pairing code.
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env
        self._client: t3_mcp_client.T3McpClient | None = None
        self._repair: t3_mcp_client.CodeSource | None = t3_mcp_client.CliCodeSource()

    def client(self) -> t3_mcp_client.T3McpClient | None:
        if self._client is None:
            connected = _connect(self._env, self._repair)
            if isinstance(connected, NoLauncher):
                logger.warning(f'cannot reach T3 over MCP: {connected.verdict()}')
                if connected.reason is t3_mcp_client.ConnectFailure.PAIR_FAILED:
                    self._repair = None
                return None
            self._client = connected
        return self._client


@dataclass(frozen=True)
class ThreadUnread:
    """No thread detail; ``gone`` only when T3 answered that it has no such thread."""

    gone: bool


def _read_thread(
    client: t3_mcp_client.T3McpClient | None, thread_id: str, *, read_timeout: float
) -> Mapping[str, Any] | ThreadUnread:
    """``t3_thread_read``'s ``thread`` detail."""
    if client is None:
        return ThreadUnread(gone=False)
    reply = client.call('t3_thread_read', {'threadId': thread_id, 'limit': 1}, read_timeout=read_timeout)
    if isinstance(reply, t3_mcp_client.ToolFailed) and reply.code == 'thread_not_found':
        return ThreadUnread(gone=True)
    thread = reply.value.get('thread') if isinstance(reply, t3_mcp_client.ToolOk) else None
    return thread if isinstance(thread, Mapping) else ThreadUnread(gone=False)


@dataclass(frozen=True)
class Incomplete:
    """A listing not read whole; ``transport`` is the failed call's, when the transport failed."""

    detail: str
    transport: t3_mcp_client.TransportFailed | None = None


def _incomplete(tool: str, reply: t3_mcp_client.ToolFailed | t3_mcp_client.TransportFailed) -> Incomplete:
    if isinstance(reply, t3_mcp_client.TransportFailed):
        return Incomplete(f'{tool}: {reply.detail}', reply)
    return Incomplete(f'{tool}: {reply.code}: {reply.message}')


def _is_new_cursor(cursor: object, seen: Sequence[int]) -> TypeGuard[int | None]:
    return cursor is None or (type(cursor) is int and cursor not in seen)


def _paged(
    client: t3_mcp_client.T3McpClient, tool: str, arguments: Mapping[str, Any], key: str
) -> list[Mapping[str, Any]] | Incomplete:
    """Every item of a cursor-paged listing, or Incomplete when any page fails or is malformed.

    T3 pages by offset, so a listing whose reported ``total`` moved between
    pages, or that collected other than ``total`` items, may have skipped one.
    """
    items: list[Mapping[str, Any]] = []
    cursors: list[int] = []
    totals: set[object] = set()
    while True:
        page_arguments = {**arguments, 'limit': PAGE_LIMIT, **({'cursor': cursors[-1]} if cursors else {})}
        reply = client.call(tool, page_arguments, read_timeout=CALL_TIMEOUT_SECS)
        if not isinstance(reply, t3_mcp_client.ToolOk):
            return _incomplete(tool, reply)
        page, cursor = reply.value.get(key), reply.value.get('nextCursor')
        if not isinstance(page, list) or not _is_new_cursor(cursor, cursors):
            return Incomplete(f'{tool}: a malformed page')
        items.extend(item for item in page if isinstance(item, Mapping))
        totals.add(reply.value.get('total'))
        if cursor is None:
            return items if totals in ({None}, {len(items)}) else Incomplete(f'{tool}: the listing moved while paged')
        cursors.append(cursor)


def _failed_call(failed: Incomplete) -> NoLauncher | LaunchFailed:
    transport = failed.transport
    if transport is not None and transport.fault is t3_mcp_client.TransportFault.CREDENTIAL_EXPIRED:
        return NoLauncher(t3_mcp_client.ConnectFailure.CREDENTIAL_EXPIRED, transport.detail)
    return LaunchFailed(failed.detail)


@dataclass(frozen=True)
class T3Project:
    project_id: str
    workspace_root: str
    archived: bool

    @classmethod
    def from_json(cls, item: Mapping[str, Any]) -> T3Project | None:
        """None for a deleted project, or one without an id and a root."""
        project_id, root = item.get('id'), item.get('workspaceRoot')
        if item.get('deletedAt') is not None or not isinstance(project_id, str) or not isinstance(root, str):
            return None
        return cls(project_id, root, archived=item.get('archivedAt') is not None)


def _target_project(client: t3_mcp_client.T3McpClient, cwd: str) -> T3Project | NotLaunched:
    """The live project of T3's whole project list that a thread for *cwd* runs in."""
    listed = _paged(client, 't3_project_list', {}, 'projects')
    if isinstance(listed, Incomplete):
        return _failed_call(listed)
    projects = [project for project in map(T3Project.from_json, listed) if project is not None]
    live = containing_project(cwd, [project for project in projects if not project.archived])
    if live is not None:
        return live
    archived = containing_project(cwd, [project for project in projects if project.archived])
    if archived is not None:
        return Refused(PlacementRefusal.PROJECT_ARCHIVED, os.path.realpath(archived.workspace_root))
    return Refused(PlacementRefusal.CWD_OUTSIDE_PROJECTS)


def containing_project(cwd: str, projects: Sequence[T3Project]) -> T3Project | None:
    """The deepest project whose real workspace root contains *cwd*."""
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


def _runtime_mode(skip_permissions: bool) -> str:
    return 'full-access' if skip_permissions else 'approval-required'


@dataclass(frozen=True)
class WorkspaceStrategy:
    worktree_path: str | None
    """The existing worktree the thread is bound to; None for the project root."""

    @classmethod
    def for_cwd(cls, project: T3Project, cwd: str) -> WorkspaceStrategy:
        at_root = os.path.realpath(cwd) == os.path.realpath(project.workspace_root)
        return cls(None if at_root else cwd)

    @property
    def binds_worktree(self) -> bool:
        return self.worktree_path is not None

    def to_json(self) -> dict[str, str]:
        if self.worktree_path is None:
            return {'type': 'root'}
        return {'type': 'existing_worktree', 'worktreePath': self.worktree_path}


def _launch_arguments(
    project: T3Project,
    title: str,
    message: str,
    model: str | None,
    instance_id: str,
    *,
    skip_permissions: bool,
    strategy: WorkspaceStrategy,
    request_id: str,
) -> dict[str, Any]:
    model_selection = {'modelSelection': {'instanceId': instance_id, 'model': model}} if model else {}
    return {
        'projectId': project.project_id,
        'title': title,
        'message': message,
        **model_selection,
        'runtimeMode': _runtime_mode(skip_permissions),
        'workspaceStrategy': strategy.to_json(),
        'clientRequestId': request_id,
    }


@dataclass(frozen=True)
class LaunchedThread:
    thread_id: str
    project_id: str | None
    run_id: str | None


_LOST_REPLY = frozenset({t3_mcp_client.TransportFault.UNREACHABLE, t3_mcp_client.TransportFault.PROTOCOL})


def _send_launch(client: t3_mcp_client.T3McpClient, arguments: Mapping[str, Any]) -> t3_mcp_client.CallResult:
    """t3_thread_launch, resent with the same clientRequestId while its reply is lost: T3 replays that thread."""
    reply = client.call('t3_thread_launch', arguments, read_timeout=CALL_TIMEOUT_SECS)
    for _ in range(LAUNCH_ATTEMPTS - 1):
        if not (isinstance(reply, t3_mcp_client.TransportFailed) and reply.fault in _LOST_REPLY):
            break
        reply = client.call('t3_thread_launch', arguments, read_timeout=CALL_TIMEOUT_SECS)
    return reply


def _launch_outcome(reply: t3_mcp_client.CallResult, strategy: WorkspaceStrategy) -> LaunchedThread | NotLaunched:
    """The launched thread, else why none is; T3, not the lane, judges whether a worktree is the project's."""
    if isinstance(reply, t3_mcp_client.ToolFailed):
        if reply.code == 'invalid_request' and strategy.binds_worktree:
            return Refused(PlacementRefusal.CWD_NOT_PROJECT_WORKTREE, reply.message)
        return LaunchFailed(f'{reply.code}: {reply.message}')
    if isinstance(reply, t3_mcp_client.TransportFailed):
        return _failed_call(_incomplete('t3_thread_launch', reply))
    thread_id = reply.value.get('threadId')
    if not isinstance(thread_id, str) or not thread_id:
        return LaunchFailed('t3_thread_launch answered no threadId')
    return LaunchedThread(thread_id, reply.value.get('projectId'), reply.value.get('runId'))


def _stamp_launched_record(
    record_dir: str, title: str, launched: LaunchedThread, environment_id: str, parent: str | None
) -> None:
    """One read-modify-write: the t3 display, status running, and the parent when given; then the thread pointer's reconcile."""
    slug, root = _record_key(record_dir)
    try:
        record = session_registry.read_record(slug, root=root)
        record.display = session_registry.Display(
            kind=session_registry.DisplayKind.T3.value,
            wm_title=title,
            t3_thread_id=launched.thread_id,
            t3_environment_id=environment_id,
        )
        if parent is not None:
            record.parent_session_id = parent
        session_registry.write_record(
            session_registry.apply_refresh(slug, record, status=session_registry.Status.RUNNING),
            root=root,
        )
        bind_thread_pointer(record_dir, launched.thread_id)
    except Exception as exc:
        logger.warning(f'registry stamp failed for {record_dir}: {exc}')


def _usable_parent(parent: str | None) -> str | None:
    if parent and not session_registry.is_record_key(parent):
        logger.warning(f'ignoring --parent {parent!r}: not a session record key')
        return None
    return parent or None


def provider_instance(
    cwd: str, explicit: str | None, env: Mapping[str, str]
) -> spawn_t3_provider.ProviderChoice:
    """The instance a thread for *cwd* runs on; see ``spawn_t3_provider`` for the order."""
    return _provider_for(_live_project_softly(cwd, env), explicit, env)


def _live_project_softly(cwd: str, env: Mapping[str, str]) -> T3Project | None:
    client = _connect(env, t3_mcp_client.CliCodeSource())
    project = client if isinstance(client, NoLauncher) else _target_project(client, cwd)
    if isinstance(project, T3Project):
        return project
    logger.warning(f'provider enablement is checked environment-wide: {project.verdict()}')
    return None


def _provider_for(
    project: T3Project | None, explicit: str | None, env: Mapping[str, str]
) -> spawn_t3_provider.ProviderChoice:
    project_id = project.project_id if project else None
    return spawn_t3_provider.resolve_instance(t3_userdata_dir(env), project_id, explicit)


def _provider_verdict(choice: spawn_t3_provider.ProviderChoice) -> str:
    named = {'instance_id': choice.instance_id, 'source': choice.source.value}
    if choice.enabled is False:
        return _verdict('refused', reason='provider-disabled', **named)
    return _verdict('instance', **named)


def launch(
    cwd: str,
    title: str,
    prompt_file: str,
    record_dir: str,
    model: str | None,
    env: Mapping[str, str],
    *,
    provider: str | None = None,
    parent: str | None = None,
    skip_permissions: bool,
) -> str:
    client = _connect(env, t3_mcp_client.CliCodeSource())
    if isinstance(client, NoLauncher):
        return client.verdict()
    project = _target_project(client, cwd)
    if not isinstance(project, T3Project):
        return project.verdict()
    choice = _provider_for(project, provider, env)
    if model and choice.enabled is False:
        return _provider_verdict(choice)

    _sweep_listed_threads(client, routine=False)
    message = _message_for(Path(prompt_file).read_text(encoding='utf-8'), record_dir)
    strategy = WorkspaceStrategy.for_cwd(project, cwd)
    arguments = _launch_arguments(
        project,
        title,
        message,
        model,
        choice.instance_id,
        skip_permissions=skip_permissions,
        strategy=strategy,
        request_id=str(uuid.uuid4()),
    )
    launched = _launch_outcome(_send_launch(client, arguments), strategy)
    if not isinstance(launched, LaunchedThread):
        return launched.verdict()
    if record_dir:
        _stamp_launched_record(record_dir, title, launched, client.environment_id, _usable_parent(parent))
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


class BindingSource(StrEnum):
    """How the spawn record reached its session; the ``started`` verdict's ``via=``."""

    THREAD_POINTER = 'thread-pointer'
    TRANSCRIPT = 'transcript'
    NONE = 'none'


@dataclass(frozen=True)
class Binding:
    session_id: str | None
    hand_slug: str | None
    bound: bool
    via: BindingSource = BindingSource.NONE


_UNBOUND = Binding(None, None, bound=False)


def _is_hand_launched(record: session_registry.SessionRecord) -> bool:
    """A spawn's record always has a result file; a hooks-made hand-launched one never does."""
    return record.result_file is None


def _read_record_softly(slug: str, root: Path, role: str) -> session_registry.SessionRecord | None:
    try:
        return session_registry.read_record(slug, root=root)
    except (OSError, session_registry.CorruptSessionRecord) as exc:
        logger.warning('cannot read %s record %s: %s', role, slug, exc)
        return None


def _record_vouching_for(
    session_id: str, spawn_slug: str, root: Path
) -> session_registry.SessionRecord | None:
    """The other record *session_id*'s pointer is vouched for by, if any."""
    slug = session_registry.resolve_session_slug_for_claude_session_id(session_id, root=root)
    if slug is None or slug == spawn_slug:
        return None
    return _read_record_softly(slug, root, 'hand-launched')


def _fold(hand: session_registry.SessionRecord, root: Path) -> None:
    try:
        session_registry.mark_session_folded(hand.session_slug, root=root)
    except (OSError, session_registry.CorruptSessionRecord) as exc:
        logger.warning('cannot fold hand-launched record %s: %s', hand.session_slug, exc)


def _bind_to(
    record_dir: str, session_id: str, hand: session_registry.SessionRecord | None, via: BindingSource
) -> Binding:
    """Bind *session_id* to the spawn record, then fold *hand*, the hand-launched record that held it.

    A session another spawn's record already owns is left alone, unbound.

    The spawn record and every pointer naming it (session id, owner pid, and
    its T3 thread when it is t3) are written before the fold, so a hook event
    landing in between already resolves to the spawn record.
    """
    slug, root = _record_key(record_dir)
    if hand is not None and not _is_hand_launched(hand):
        logger.warning(
            'session %s belongs to spawn record %s; not binding it to %s',
            session_id,
            hand.session_slug,
            slug,
        )
        return Binding(session_id, None, bound=False)
    owner_pid = hand.claude_owner_pid if hand else None
    owner_identity = hand.claude_owner_identity if hand else None
    hand_slug = hand.session_slug if hand else None
    try:
        record = session_registry.read_record(slug, root=root)
        if record.claude_launch_session_id is None:
            record.claude_launch_session_id = session_id
        record.claude_session_id = session_id
        if owner_pid is not None:
            record.claude_owner_pid = owner_pid
            record.claude_owner_identity = owner_identity
        session_registry.write_record(record, root=root)
    except Exception as exc:
        logger.warning(f'cannot bind session {session_id} to {record_dir}: {exc}')
        return Binding(session_id, hand_slug, bound=False)
    session_registry.write_claude_session_pointer(session_id, slug, root=root)
    if owner_pid is not None:
        session_registry.write_session_pointer(owner_pid, slug, root=root)
    thread_id = session_registry.t3_thread_of(record)
    if thread_id is not None:
        session_registry.write_t3_thread_pointer(thread_id, slug, root=root)
    if hand is not None:
        _fold(hand, root)
    return Binding(session_id, hand_slug, bound=True, via=via)


def bind_session(record_dir: str, transcript: Transcript) -> Binding:
    """Bind *transcript*'s session to the spawn record, folding the record its session-id pointer names."""
    slug, root = _record_key(record_dir)
    hand = _record_vouching_for(transcript.session_id, slug, root)
    return _bind_to(record_dir, transcript.session_id, hand, BindingSource.TRANSCRIPT)


def bind_thread_pointer(record_dir: str, thread_id: str) -> Binding:
    """Bind the spawn record through *thread_id*'s pointer, which ends naming it.

    A hand-launched record holding the pointer won the race to the thread:
    its session is bound to the spawn record and it is folded. Another
    spawn's record is left alone. With no other holder the pointer is
    written for a t3 spawn record of *thread_id*, bound once a hook bound it.
    """
    slug, root = _record_key(record_dir)
    holder_slug = session_registry.resolve_session_slug_for_t3_thread(thread_id, root=root)
    if holder_slug is not None and holder_slug != slug:
        holder = _read_record_softly(holder_slug, root, 'thread-pointer holder')
        if holder is None or not holder.claude_session_id:
            return _UNBOUND
        return _bind_to(record_dir, holder.claude_session_id, holder, BindingSource.THREAD_POINTER)
    record = _read_record_softly(slug, root, 'spawn')
    if record is None or session_registry.t3_thread_of(record) != thread_id:
        return _UNBOUND
    session_registry.write_t3_thread_pointer(thread_id, slug, root=root)
    if not record.claude_session_id:
        return _UNBOUND
    return Binding(record.claude_session_id, None, bound=True, via=BindingSource.THREAD_POINTER)


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


def await_started(
    thread_id: str, cwd: str, record_dir: str, timeout: float, since: float, env: Mapping[str, str]
) -> str:
    deadline = time.monotonic() + timeout
    session = _Session(env)
    while True:
        client = session.client()
        read_timeout = max(_READ_TIMEOUT_FLOOR_SECS, min(CALL_TIMEOUT_SECS, deadline - time.monotonic()))
        thread = _read_thread(client, thread_id, read_timeout=read_timeout)
        verdict = _started_verdict(thread, cwd, record_dir, since)
        if verdict is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _verdict('timeout')
        time.sleep(min(STARTED_POLL_SECS, remaining))
    if verdict != 'started':
        return _verdict(verdict)
    binding = _bind_within(record_dir, thread_id, cwd, since, deadline) if record_dir else None
    return _verdict(
        'started',
        sid=binding.session_id if binding else None,
        hand_slug=binding.hand_slug if binding else None,
        bound='yes' if binding and binding.bound else 'no',
        via=binding.via if binding else BindingSource.NONE,
    )


def _started_verdict(
    thread: Mapping[str, Any] | ThreadUnread, cwd: str, record_dir: str, since: float
) -> str | None:
    """None while unread or preparing; a startup-failure status is failed-to-start until a transcript names the record."""
    status = None if isinstance(thread, ThreadUnread) else thread.get('status')
    if not isinstance(status, str) or status in PENDING_RUN_STATUSES:
        return None
    if status in STARTUP_FAILURE_STATUSES and not (
        record_dir and discover_transcript(cwd, record_dir, since)
    ):
        return 'failed-to-start'
    return 'started'


def _bind_within(record_dir: str, thread_id: str, cwd: str, since: float, deadline: float) -> Binding | None:
    """Per poll, the thread pointer first and transcript discovery second, until one binds or *deadline* passes."""
    while True:
        by_pointer = bind_thread_pointer(record_dir, thread_id)
        if by_pointer.bound:
            return by_pointer
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
    PRESENT = 'present'
    ARCHIVED = 'archived'
    GONE = 'gone'
    UNREACHABLE = 'unreachable'


ENDED_THREAD_STATES = frozenset({ThreadState.GONE, ThreadState.ARCHIVED})


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


def result_is_final(record_dir: str) -> bool:
    if not record_dir:
        return False
    try:
        text = (Path(record_dir) / session_registry.RESULT_FILENAME).read_text(encoding='utf-8')
    except (OSError, ValueError):
        return False
    return session_registry.parse_result_outcome(text) in session_registry.TERMINAL_RESULT_OUTCOMES


class CallerGone(Exception):
    """The process that started ``wait`` has exited; nobody reads its verdict."""


def wait(
    thread_id: str,
    cwd: str,
    record_dir: str,
    env: Mapping[str, str],
    *,
    parent_pid: Callable[[], int] = os.getppid,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Observe, then wait one slice in T3, until the join has its exit code."""
    caller = parent_pid()
    tracker = JoinTracker()
    session = _Session(env)
    while True:
        if parent_pid() != caller:
            raise CallerGone(f'caller {caller} is gone; stopped waiting on thread {thread_id}')
        client = session.client()
        observation = _observe(client, thread_id, cwd, record_dir)
        code = tracker.verdict(observation, result_final=result_is_final(record_dir), now=clock())
        if code is not None:
            return _verdict('finished', code=code)
        if _needs_pause(_wait_slice(client, thread_id), record_dir):
            sleep(SETTLED_RECHECK_SECS)


def _observe(
    client: t3_mcp_client.T3McpClient | None, thread_id: str, cwd: str, record_dir: str
) -> ThreadObservation:
    """The thread's state, then one sweep of its environment and the binding's upkeep."""
    observation = _observe_thread(client, thread_id)
    if client is not None:
        _sweep_listed_threads(client, routine=True)
    if record_dir:
        _maintain_binding_softly(record_dir, cwd)
    return observation


def _observe_thread(client: t3_mcp_client.T3McpClient | None, thread_id: str) -> ThreadObservation:
    thread = _read_thread(client, thread_id, read_timeout=CALL_TIMEOUT_SECS)
    if isinstance(thread, ThreadUnread):
        return ThreadObservation(ThreadState.GONE if thread.gone else ThreadState.UNREACHABLE)
    if thread.get('archived') is True:
        return ThreadObservation(ThreadState.ARCHIVED)
    return ThreadObservation(ThreadState.PRESENT, str(thread.get('status') or ''), thread.get('latestRunId'))


def _wait_slice(client: t3_mcp_client.T3McpClient | None, thread_id: str) -> bool:
    """Hold ``t3_thread_wait`` for up to one slice; whether T3 held it until the slice ran out."""
    if client is None:
        return False
    arguments = {'threadId': thread_id, 'timeoutMs': WAIT_SLICE_SECS * 1000}
    reply = client.call('t3_thread_wait', arguments, read_timeout=WAIT_SLICE_SECS + t3_mcp_client.HTTP_TIMEOUT_SECS)
    return isinstance(reply, t3_mcp_client.ToolOk) and reply.value.get('timedOut') is True


def _needs_pause(slice_held: bool, record_dir: str) -> bool:
    """A slice T3 answered at once is followed by a pause, unless a final result ends the join on the next pass."""
    return not slice_held and not result_is_final(record_dir)


def _sweep_listed_threads(client: t3_mcp_client.T3McpClient, *, routine: bool) -> None:
    """Sweep this environment's indexed t3 records against its active threads, only from a complete listing.

    An unlisted thread's record is marked only once a read finds the thread
    gone or archived. A *routine* sweep, one pass of ``wait``, logs its index
    line at DEBUG unless it marked a record.
    """
    listed_at = datetime.now(UTC)
    threads = _paged(client, 't3_thread_list', {'allProjects': True}, 'threads')
    thread_ids = None if isinstance(threads, Incomplete) else [thread.get('threadId') for thread in threads]
    if thread_ids is None or not all(isinstance(thread_id, str) for thread_id in thread_ids):
        logger.warning('t3 session sweep skipped: no complete thread listing of this environment')
        return
    try:
        session_registry.mark_t3_sessions_exited(
            client.environment_id,
            frozenset(map(str, thread_ids)),
            listed_at=listed_at,
            thread_ended=_ThreadEndConfirmer(client),
            routine=routine,
        )
    except Exception as exc:
        logger.warning(f't3 session sweep failed: {exc}')


class _ThreadEndConfirmer:
    """One sweep's ``thread_ended``: whether a read finds the thread gone or archived.

    After one unreachable read it reads no further thread and answers "not
    confirmed", as an unreachable read itself does: each such read can cost a
    full ``CALL_TIMEOUT_SECS``, and ``launch`` waits on its sweep.
    """

    def __init__(self, client: t3_mcp_client.T3McpClient) -> None:
        self._client = client
        self._unreachable = False

    def __call__(self, thread_id: str) -> bool:
        if self._unreachable:
            return False
        state = _observe_thread(self._client, thread_id).state
        self._unreachable = state is ThreadState.UNREACHABLE
        return state in ENDED_THREAD_STATES


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
# MCP credential: pair and status
# ---------------------------------------------------------------------------


class RuntimeOriginFailure(StrEnum):
    MISSING = 'server-runtime-missing'
    UNUSABLE = 'server-runtime-origin-unusable'


@dataclass(frozen=True)
class RuntimeOriginFailed:
    """server-runtime.json names no origin a credential may be sent to."""

    reason: RuntimeOriginFailure
    origin: str | None = None
    """The ``origin`` string as found, for an UNUSABLE one."""
    detail: str | None = None

    def fields(self) -> dict[str, object]:
        """The verdict tokens: the reason, then for an UNUSABLE origin what was found and why."""
        if self.reason is RuntimeOriginFailure.MISSING:
            return {'reason': self.reason.value}
        return {'reason': self.reason.value, 'origin': self.origin, 'detail': self.detail}


def _runtime_origin(env: Mapping[str, str]) -> str | RuntimeOriginFailed:
    runtime = read_server_runtime(env)
    if runtime is None:
        return RuntimeOriginFailed(RuntimeOriginFailure.MISSING)
    origin = runtime.get('origin')
    if not isinstance(origin, str):
        return RuntimeOriginFailed(RuntimeOriginFailure.UNUSABLE, detail='server-runtime.json has no origin string')
    try:
        return t3_mcp_client.checked_origin(origin)
    except ValueError as error:
        return RuntimeOriginFailed(RuntimeOriginFailure.UNUSABLE, origin, str(error))


def _pair_failure_fields(failed: t3_mcp_client.PairFailed) -> dict[str, object]:
    fields: dict[str, object] = {'reason': failed.reason.value}
    if failed.reason is t3_mcp_client.PairFailure.CLI_VERSION_MISMATCH:
        fields |= {'cli': failed.cli_version, 'server': failed.server_version}
    if failed.detail is not None:
        fields['detail'] = failed.detail
    return fields


def pair_verb(env: Mapping[str, str], code: str | None, origin: str | None) -> str:
    resolved = origin or _runtime_origin(env)
    if isinstance(resolved, RuntimeOriginFailed):
        return _verdict('pair-failed', **resolved.fields())
    source = t3_mcp_client.OperatorCode(code) if code else t3_mcp_client.CliCodeSource()
    result = t3_mcp_client.pair(resolved, source)
    if isinstance(result, t3_mcp_client.PairFailed):
        return _verdict('pair-failed', **_pair_failure_fields(result))
    credential = result.credential
    return _verdict(
        'paired',
        environment_id=credential.environment_id,
        expires_at=credential.expires_at.isoformat(),
        access=credential.access,
    )


def status_verb(env: Mapping[str, str], origin: str | None) -> str:
    resolved = origin or _runtime_origin(env)
    if isinstance(resolved, RuntimeOriginFailed):
        return _verdict('status-failed', **({'reason': resolved.reason.value, 'origin': None} | resolved.fields()))
    status = t3_mcp_client.credential_status_at(resolved)
    if isinstance(status, t3_mcp_client.DescriptorFailed):
        return _verdict('status-failed', reason=status.reason.value, origin=resolved, detail=status.detail)
    expires_at = status.credential.expires_at.isoformat() if status.credential else None
    return _verdict(status.state.value, environment_id=status.environment_id, expires_at=expires_at, origin=resolved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _origin_argument(url: str) -> str:
    """``checked_origin`` for argparse, which prints an ArgumentTypeError's message and drops a ValueError's."""
    try:
        return t3_mcp_client.checked_origin(url)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='spawn_t3_lane')
    verbs = parser.add_subparsers(dest='verb', required=True)
    verbs.add_parser('detect', help='print t3 when this process is hosted by a T3 thread')

    provider_p = verbs.add_parser('provider', help='resolve the provider instance for a launch')
    provider_p.add_argument('--cwd', required=True)
    provider_p.add_argument('--provider', default=None, help='explicit provider instance id')

    launch_p = verbs.add_parser('launch', help='start a T3 thread for a spawn')
    launch_p.add_argument('--cwd', required=True)
    launch_p.add_argument('--title', required=True)
    launch_p.add_argument('--prompt-file', required=True)
    launch_p.add_argument('--record', required=True, help="record dir, '' when none")
    launch_p.add_argument('--model', default=None)
    launch_p.add_argument('--provider', default=None, help='explicit provider instance id')
    launch_p.add_argument('--parent', default=None, help="the spawner's record slug")
    launch_p.add_argument('--skip-permissions', action='store_true', help='full-access; approval-required without it')

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

    pair_p = verbs.add_parser(
        'pair', help="pair this host with the T3 server's /mcp (self-pair, or --code from an operator)"
    )
    pair_p.add_argument('--code', default=None)
    pair_p.add_argument('--origin', type=_origin_argument, default=None)

    status_p = verbs.add_parser('status', help="print the lane's MCP credential state")
    status_p.add_argument('--origin', type=_origin_argument, default=None)
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
            provider=args.provider,
            parent=args.parent,
            skip_permissions=args.skip_permissions,
        )
    if args.verb == 'provider':
        return _provider_verdict(provider_instance(args.cwd, args.provider, os.environ))
    if args.verb == 'await-started':
        return await_started(args.thread_id, args.cwd, args.record, args.timeout, args.since, os.environ)
    if args.verb == 'wait':
        return wait(args.thread_id, args.cwd, args.record, os.environ)
    if args.verb == 'pair':
        return pair_verb(os.environ, args.code, args.origin)
    if args.verb == 'status':
        return status_verb(os.environ, args.origin)
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
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='spawn_t3_lane: %(message)s')
    sys.exit(main(sys.argv[1:]))
