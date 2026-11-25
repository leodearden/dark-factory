"""The defect filer's wire to fused-memory (plans/client-feed-prd.md §14.2 decision 7, §14.4): the closed tool set, the one failure table, typed replies and the session."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from shared.defects import DefectStatus
from shared.mcp_session import McpCallError, SyncMcpSession

from client_feed.contract import TaskStatus, Token
from client_feed.policy import Policy

FILER_IDENTITY: Final = 'client-defect-filer'
SOURCE_KEY: Final = 'source'
CLIENT_DEFECT_KEY: Final = 'x_client_defect'
GATE_FOR_KEY: Final = 'x_client_defect_gate_for'
DEFECTS_KEY: Final = 'defects'

READ_TOOLS: Final = frozenset({'lookup_defect_ref', 'find_tasks_by_metadata', 'get_task'})
WRITE_TOOLS: Final = frozenset({
    'submit_task',
    'add_dependency',
    'commit_planning',
    'declare_defect',
    'record_defect_outcome',
    'set_task_status',
})
CLOSED_TOOLS: Final = READ_TOOLS | WRITE_TOOLS

LIST_TOOLS: Final = 'tools/list'
IS_ERROR_CODE: Final = 'is_error'
UNRECOGNISED_CODE: Final = 'unrecognised'
_PAGE_SIZE: Final = 100
_SESSION_TIMEOUT_SECONDS: Final = 30.0

_T = TypeVar('_T')


class ToolCaller(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]: ...


class ListingCaller(ToolCaller, Protocol):
    """A session that also names the tools its server lists."""

    def list_tools(self) -> tuple[str, ...]: ...


_Caller = TypeVar('_Caller', bound=ToolCaller)


class RefusalKind(StrEnum):
    """What refused a call: one of shared.mcp_session.McpFailureKind, or a reply naming its own error code."""

    TRANSPORT = 'TRANSPORT'
    HTTP_STATUS = 'HTTP_STATUS'
    UNDECODABLE = 'UNDECODABLE'
    JSON_RPC_ERROR = 'JSON_RPC_ERROR'
    TOOL_ERROR = 'TOOL_ERROR'
    REPLY_SHAPE = 'REPLY_SHAPE'
    REPLY_CODE = 'REPLY_CODE'


class Disposition(StrEnum):
    RETRY = 'retry'
    FALLBACK = 'fallback'
    RETRY_THEN_FALLBACK = 'retry_then_fallback'
    ALREADY_DONE = 'already_done'


_BY_FAILURE_KIND: Final[Mapping[RefusalKind, Disposition]] = MappingProxyType({
    RefusalKind.TRANSPORT: Disposition.RETRY,
    RefusalKind.HTTP_STATUS: Disposition.RETRY,
    RefusalKind.TOOL_ERROR: Disposition.FALLBACK,
    RefusalKind.UNDECODABLE: Disposition.RETRY_THEN_FALLBACK,
    RefusalKind.JSON_RPC_ERROR: Disposition.RETRY_THEN_FALLBACK,
    RefusalKind.REPLY_SHAPE: Disposition.RETRY_THEN_FALLBACK,
})
_BY_REPLY_CODE: Final[Mapping[str, Disposition]] = MappingProxyType({
    'ReconciliationBacklogExceeded': Disposition.RETRY,
    'ValidationError': Disposition.FALLBACK,
    'defect_creation_refused': Disposition.FALLBACK,
    'defect_write_to_terminal_task': Disposition.ALREADY_DONE,
})


def classify(kind: RefusalKind, code: str) -> Disposition:
    """THE failure table: a session failure by its kind, a refusing reply by its code; an unknown code retries, then falls back."""
    if kind is RefusalKind.REPLY_CODE:
        return _BY_REPLY_CODE.get(code, Disposition.RETRY_THEN_FALLBACK)
    return _BY_FAILURE_KIND[kind]


class ToolRefused(Exception):
    """A closed-tool call fused-memory did not carry out; only its kind and code are kept, never the server's text, which can echo client text."""

    def __init__(self, tool: str, kind: RefusalKind, code: str) -> None:
        self.tool = tool
        self.kind = kind
        self.code = code
        self.disposition = classify(kind, code)
        super().__init__(f'{tool} refused ({kind}, {code}): {self.disposition}')


_TOKEN: Final = TypeAdapter(Token)


def _token(code: object) -> str:
    try:
        return _TOKEN.validate_python(code)
    except ValidationError:
        return UNRECOGNISED_CODE


def _refusal_code(reply: Mapping[str, Any]) -> str | None:
    """The code a refusing reply names (its error_type, else its error); None for a reply that is no refusal."""
    if reply.get('error') is None and reply.get('success') is not False:
        return None
    return _token(reply.get('error_type') or reply.get('error'))


def _session_refusal(tool: str, exc: McpCallError) -> ToolRefused:
    kind = RefusalKind(exc.kind.value)
    return ToolRefused(tool, kind, IS_ERROR_CODE if kind is RefusalKind.TOOL_ERROR else kind.lower())


def _reply_shape(tool: str) -> ToolRefused:
    return ToolRefused(tool, RefusalKind.REPLY_SHAPE, RefusalKind.REPLY_SHAPE.lower())


def _parsed(tool: str, parse: Callable[[], _T]) -> _T:
    try:
        return parse()
    except (KeyError, TypeError, ValueError):
        raise _reply_shape(tool) from None


@dataclass(frozen=True)
class TaskSubmission:
    """submit_task's fields as the filer sends them."""

    title: str
    description: str
    priority: str
    task_kind: str
    metadata: Mapping[str, Any]


class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class Carrier(_View):
    """A task lookup_defect_ref found declaring the ref, and where the ref stands on it."""

    task_id: str
    task_status: TaskStatus
    defect_status: DefectStatus


class TaskView(_View):
    """What the filer reads of a task; metadata another writer shaped differently reads as absent."""

    task_id: str
    status: TaskStatus
    dependencies: frozenset[str]
    source: str | None
    x_client_defect: dict[str, Any] | None
    defects: tuple[dict[str, Any], ...]


def _task_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f'a task id is an int or a str, got {value!r}')
    return str(value)


def _carrier(entry: Mapping[str, Any]) -> Carrier:
    return Carrier(
        task_id=_task_id(entry['task_id']),
        task_status=entry['task_status'],
        defect_status=entry['defect_status'],
    )


def _task_view(reply: Mapping[str, Any]) -> TaskView:
    metadata = reply.get('metadata') or {}
    if not isinstance(metadata, dict):
        metadata = {}
    source = metadata.get(SOURCE_KEY)
    facts = metadata.get(CLIENT_DEFECT_KEY)
    defects = metadata.get(DEFECTS_KEY)
    return TaskView(
        task_id=_task_id(reply['id']),
        status=reply['status'],
        dependencies=frozenset(_task_id(task_id) for task_id in reply.get('dependencies') or ()),
        source=source if isinstance(source, str) else None,
        x_client_defect=facts if isinstance(facts, dict) else None,
        defects=tuple(entry for entry in defects if isinstance(entry, dict)) if isinstance(defects, list) else (),
    )


def _gate_page(reply: Mapping[str, Any]) -> tuple[tuple[str, ...], bool]:
    page = tuple(_task_id(row['task_id']) for row in reply['results'])
    has_more = reply['pagination']['has_more']
    if not isinstance(has_more, bool) or (has_more and not page):
        raise ValueError(f'has_more={has_more!r} after a page of {len(page)}')
    return page, has_more


def _commit_refusal(reply: Mapping[str, Any]) -> str | None:
    """A multi-id status reply refuses by its first refused id; a single-id reply refuses as any reply does."""
    results = reply.get('results')
    if reply.get('error') is not None or not isinstance(results, list):
        return _refusal_code(reply)
    codes = (_refusal_code(item['result']) for item in results)
    return next((code for code in codes if code is not None), None)


class ClosedTools:
    """The filer's one way to fused-memory: a typed method per permitted call, every call naming the policy's project root."""

    def __init__(self, caller: ToolCaller, project_root: Path) -> None:
        self._caller = caller
        self._project_root = str(project_root)

    def _call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        return self._reply(tool, self._send(tool, arguments), _refusal_code)

    def _send(self, tool: str, arguments: Mapping[str, Any]) -> object:
        try:
            return self._caller.call_tool(tool, {**arguments, 'project_root': self._project_root})
        except McpCallError as exc:
            raise _session_refusal(tool, exc) from None

    @staticmethod
    def _reply(
        tool: str, reply: object, refusal: Callable[[Mapping[str, Any]], str | None]
    ) -> dict[str, Any]:
        if not isinstance(reply, dict):
            raise _reply_shape(tool)
        code = _parsed(tool, lambda: refusal(reply))
        if code is not None:
            raise ToolRefused(tool, RefusalKind.REPLY_CODE, code)
        return reply

    def lookup_defect_ref(self, ref: str) -> tuple[Carrier, ...]:
        reply = self._call('lookup_defect_ref', ref=ref)
        return _parsed('lookup_defect_ref', lambda: tuple(map(_carrier, reply['carriers'])))

    def find_gates(self, ref: str) -> tuple[str, ...]:
        """Every task, of any status, whose metadata names it the human gate of `ref`."""
        found: list[str] = []
        while True:
            reply = self._call(
                'find_tasks_by_metadata', key=GATE_FOR_KEY, value=ref, limit=_PAGE_SIZE, offset=len(found)
            )
            page, has_more = _parsed('find_tasks_by_metadata', lambda reply=reply: _gate_page(reply))
            found.extend(page)
            if not has_more:
                return tuple(found)

    def get_task(self, task_id: str) -> TaskView:
        reply = self._call('get_task', id=task_id)
        return _parsed('get_task', lambda: _task_view(reply))

    def submit_planned(self, submission: TaskSubmission) -> str:
        """Submit under a planning hold, so the task is born deferred and nothing schedules it before commit_planning."""
        reply = self._call(
            'submit_task',
            title=submission.title,
            description=submission.description,
            priority=submission.priority,
            metadata=dict(submission.metadata),
            planning_mode=True,
            task_kind=submission.task_kind,
            agent_id=FILER_IDENTITY,
        )
        return _parsed('submit_task', lambda: _task_id(reply['task_id']))

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        self._call('add_dependency', id=task_id, depends_on=depends_on)

    def commit_planning(self, task_ids: Iterable[str]) -> None:
        arguments = {'task_ids': ','.join(task_ids), 'agent_id': FILER_IDENTITY}
        self._reply('commit_planning', self._send('commit_planning', arguments), _commit_refusal)

    def declare_withdrawal(self, task_id: str, event: Mapping[str, Any]) -> None:
        self._call('declare_defect', task_id=task_id, event=dict(event), agent_id=FILER_IDENTITY)

    def record_acceptance(self, task_id: str, ref: str, event: Mapping[str, Any]) -> None:
        self._call(
            'record_defect_outcome', task_id=task_id, ref=ref, event=dict(event), agent_id=FILER_IDENTITY
        )

    def cancel(self, task_id: str) -> None:
        self._call('set_task_status', id=task_id, status=TaskStatus.CANCELLED.value, agent_id=FILER_IDENTITY)


@dataclass(frozen=True)
class PlannedCall:
    tool: str
    arguments: Mapping[str, Any]
    withheld: bool


class WithheldWrites:
    """A ToolCaller for --check: reads reach fused-memory, writes are recorded and answered here and never sent."""

    def __init__(self, caller: ToolCaller) -> None:
        self._caller = caller
        self._calls: list[PlannedCall] = []
        self._submitted = 0

    @property
    def calls(self) -> tuple[PlannedCall, ...]:
        return tuple(self._calls)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        withheld = name not in READ_TOOLS
        self._calls.append(PlannedCall(name, dict(arguments), withheld))
        if not withheld:
            return self._caller.call_tool(name, arguments)
        if name == 'submit_task':
            self._submitted += 1
            return {'task_id': f'planned-{self._submitted}', 'status': 'deferred', 'planning_mode': True}
        return {'success': True}


def missing_tools(session: ListingCaller) -> tuple[str, ...]:
    """The closed tools the session's server does not list, sorted."""
    try:
        listed = frozenset(session.list_tools())
    except McpCallError as exc:
        raise _session_refusal(LIST_TOOLS, exc) from None
    return tuple(sorted(CLOSED_TOOLS - listed))


OpenTools = Callable[[Policy], AbstractContextManager[ToolCaller]]
OpenSession = Callable[[Policy], AbstractContextManager[ListingCaller]]


class SessionUnavailable(Exception):
    """A session to fused-memory that could not be opened, named by its failure kind only."""

    def __init__(self, kind: RefusalKind) -> None:
        self.kind = kind
        super().__init__(f'the fused-memory session did not open ({kind})')


@contextmanager
def opened(policy: Policy, open_caller: Callable[[Policy], AbstractContextManager[_Caller]]) -> Iterator[_Caller]:
    """One session from `open_caller`; a failed handshake is SessionUnavailable."""
    with ExitStack() as stack:
        try:
            caller = stack.enter_context(open_caller(policy))
        except McpCallError as exc:
            raise SessionUnavailable(RefusalKind(exc.kind.value)) from None
        yield caller


@contextmanager
def closed_tools(policy: Policy, open_tools: OpenTools) -> Iterator[ClosedTools]:
    """The policy's closed tools over one session from `open_tools`."""
    with opened(policy, open_tools) as caller:
        yield ClosedTools(caller, policy.project_root)


@contextmanager
def open_session(policy: Policy) -> Iterator[SyncMcpSession]:
    """A session to the policy's fused-memory over a client that reads no proxy variable and no .netrc (trust_env=False)."""
    url = str(policy.require_filer().fused_memory_url)
    with (
        httpx.Client(trust_env=False, follow_redirects=True, timeout=_SESSION_TIMEOUT_SECONDS) as client,
        SyncMcpSession(url, client_name=FILER_IDENTITY, http_client=client) as session,
    ):
        yield session
