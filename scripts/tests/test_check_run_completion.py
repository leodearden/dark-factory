"""scripts/check_run_completion.py: the weighted completion measure, the run
selection over the real find_tasks_by_metadata tool, the predicate CLI and the
--pending report."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import check_run_completion as crc
import pytest
from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import RESULT_BYTE_BUDGET, create_mcp_server
from shared.mcp_session import McpCallError, McpFailureKind
from shared.testing_mcp_stub import StubAnswer, StubMcpServer
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

RUN = 'review-dark_factory-20261109-1'
OTHER_RUN = 'hotspot-survey-dark_factory-20261101-1'
LOOKUP = 'find_tasks_by_metadata'
_LOOP_TIMEOUT_SECS = 60

_T = TypeVar('_T')
CallTool = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _row(task_id: int, priority: str | None, status: str) -> dict[str, object]:
    return {'task_id': task_id, 'priority': priority, 'status': status}


def _rows(*spec: tuple[str | None, str]) -> list[dict[str, object]]:
    return [_row(task_id, priority, status) for task_id, (priority, status) in enumerate(spec, start=1)]


BOUNDARY_ROW_1 = _rows(('high', 'pending'), ('medium', 'done'), ('low', 'done'))
BOUNDARY_ROW_1_JSON = (
    '{"run":"' + RUN + '","verdict":"not_yet","share":0.5,"threshold":0.7,"landed":3,"pending":3,'
    '"total":3,"vacuous":false,"by_priority":{"high":[0,1,0],"medium":[1,0,0],"low":[1,0,0]}}'
)


def _naive_float_pass(landed: int, pending: int, threshold: float) -> bool:
    return landed >= threshold * (landed + pending)


def test_priority_weights_are_the_contract_numbers():
    assert crc.PRIORITY_WEIGHTS == {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'polish': 1}


def test_boundary_row_1_is_not_yet_at_half():
    verdict = crc.weighted_completion(BOUNDARY_ROW_1, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (3, 3, 3)
    assert verdict.share == 0.5
    assert verdict.passed is False
    assert verdict.vacuous is False
    assert verdict.verdict == 'not_yet'
    assert verdict.error is None
    assert list(verdict.by_priority.items()) == [('high', (0, 1, 0)), ('medium', (1, 0, 0)), ('low', (1, 0, 0))]
    assert verdict.to_json() == BOUNDARY_ROW_1_JSON


def test_boundary_row_11_passes_at_exactly_the_threshold():
    rows = _rows(('critical', 'done'), ('high', 'done'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (7, 3)
    assert verdict.passed is True
    assert verdict.verdict == 'pass'


def test_three_tenths_passes_at_point_three():
    rows = _rows(('high', 'done'), ('critical', 'pending'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.3, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (3, 7)
    assert verdict.passed is True


def test_exact_threshold_passes_where_the_float_product_overshoots():
    rows = _rows(
        ('critical', 'done'), ('high', 'done'),
        ('critical', 'pending'), ('critical', 'pending'), ('critical', 'pending'), ('critical', 'pending'),
        ('medium', 'pending'),
    )
    assert not _naive_float_pass(7, 18, 0.28)

    verdict = crc.weighted_completion(rows, 0.28, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (7, 18)
    assert verdict.passed is True
    assert verdict.verdict == 'pass'


def test_six_tenths_is_not_yet_at_point_seven():
    rows = _rows(('critical', 'done'), ('medium', 'done'), ('critical', 'pending'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (6, 4)
    assert verdict.passed is False
    assert verdict.verdict == 'not_yet'


def test_a_missing_priority_weighs_as_medium():
    rows = [_row(1, None, 'done'), {'task_id': 2, 'status': 'pending'}]

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending) == (2, 2)
    assert dict(verdict.by_priority) == {'medium': (1, 1, 0)}


@pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked', 'deferred', 'review'])
def test_every_non_terminal_status_is_pending_weight(status):
    verdict = crc.weighted_completion([_row(1, 'high', status)], 0.7, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (0, 3, 1)
    assert dict(verdict.by_priority) == {'high': (0, 1, 0)}


def test_cancelled_weighs_nothing_but_counts_in_total():
    rows = _rows(('critical', 'cancelled'), ('high', 'done'), ('high', 'pending'))

    verdict = crc.weighted_completion(rows, 0.5, run_id=RUN)

    assert (verdict.landed, verdict.pending, verdict.total) == (3, 3, 3)
    assert dict(verdict.by_priority) == {'critical': (0, 0, 1), 'high': (1, 1, 0)}


def test_an_all_cancelled_run_passes_vacuously():
    rows = _rows(('high', 'cancelled'), ('low', 'cancelled'))

    verdict = crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert verdict.vacuous is True
    assert verdict.share is None
    assert verdict.passed is True
    assert verdict.verdict == 'pass'
    assert verdict.total == 2


def test_no_rows_is_an_empty_run_error():
    verdict = crc.weighted_completion([], 0.7, run_id=RUN)

    assert verdict.total == 0
    assert verdict.share is None
    assert verdict.passed is False
    assert verdict.verdict == 'error'
    assert verdict.error == 'empty_run'
    assert json.loads(verdict.to_json())['error'] == 'empty_run'
    assert '"error":"empty_run"' in verdict.to_json()


def test_an_unknown_priority_is_refused_naming_task_and_value():
    rows = [_row(4, 'high', 'done'), _row(5, 'urgent', 'pending')]

    with pytest.raises(crc.UnknownPriority) as raised:
        crc.weighted_completion(rows, 0.7, run_id=RUN)

    assert raised.value.task_id == 5
    assert raised.value.value == 'urgent'


@pytest.mark.parametrize('threshold', [0, -0.1, 1.5, math.nan])
def test_a_threshold_outside_zero_one_is_refused(threshold):
    with pytest.raises(ValueError, match=r'\(0, 1\]'):
        crc.weighted_completion(BOUNDARY_ROW_1, threshold, run_id=RUN)


def test_a_threshold_of_one_is_accepted():
    verdict = crc.weighted_completion(_rows(('high', 'done')), 1.0, run_id=RUN)

    assert verdict.passed is True


def test_the_worst_case_verdict_json_fits_the_note_cap():
    tier = crc.TierCounts(999999, 909091, 999999)
    verdict = crc.CompletionVerdict(
        run_id='hotspot-survey-dark_factory-20261004-99',
        threshold=0.6666666666666666,
        by_priority={name: tier for name in crc.PRIORITY_WEIGHTS},
    )
    rendered = verdict.to_dict()

    assert all(len(str(rendered[key])) == 8 for key in ('landed', 'pending', 'total'))
    assert len(str(rendered['share'])) == len('0.xxxx')
    assert len(verdict.to_json()) <= 400


def test_a_share_just_short_of_the_threshold_never_prints_as_reaching_it():
    verdict = crc.CompletionVerdict(run_id=RUN, threshold=0.7, by_priority={'low': crc.TierCounts(17499, 7501, 0)})
    rendered = json.loads(verdict.to_json())

    assert verdict.verdict == 'not_yet'
    assert rendered['share'] == 0.6999
    assert rendered['share'] < rendered['threshold']


def test_an_exact_threshold_share_prints_as_the_threshold():
    verdict = crc.CompletionVerdict(run_id=RUN, threshold=0.7, by_priority={'high': crc.TierCounts(7, 3, 0)})

    assert verdict.to_dict()['share'] == 0.7


@pytest.mark.parametrize(('threshold', 'by_priority'), [
    (1.5, {'high': crc.TierCounts(1, 0, 0)}),
    (0.7, {'urgent': crc.TierCounts(1, 0, 0)}),
    (0.7, {'high': crc.TierCounts(1, -1, 0)}),
], ids=['threshold_above_one', 'unknown_tier', 'negative_count'])
def test_a_verdict_cannot_be_built_from_impossible_counts(threshold, by_priority):
    with pytest.raises(ValueError):
        crc.CompletionVerdict(run_id=RUN, threshold=threshold, by_priority=by_priority)


def test_an_empty_verdict_can_never_pass():
    verdict = crc.CompletionVerdict(run_id=RUN, threshold=0.7, by_priority={})

    assert (verdict.passed, verdict.share, verdict.vacuous, verdict.total) == (False, None, False, 0)
    assert verdict.to_dict()['verdict'] == 'error'


# ── The real store: SqliteTaskBackend behind the real fused-memory tools ──


def _git_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


class FixtureStore:
    """A real task store served by the real MCP tool functions, driven synchronously.

    SqliteTaskBackend is aiosqlite-backed, so every coroutine runs on one loop,
    owned by a daemon thread; ``call_tool`` is safe from any thread.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.root = _git_root(tmp_path / 'project')
        self._seeded = 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._backend, self._interceptor, self._server = self._run(self._build())

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=_LOOP_TIMEOUT_SECS)

    async def _build(self):
        backend = SqliteTaskBackend(TaskmasterConfig(project_root=self.root))
        await backend.start()
        interceptor = TaskInterceptor(backend, None, EventBuffer(db_path=None))
        return backend, interceptor, create_mcp_server(AsyncMock(), task_interceptor=interceptor)

    def seed(
        self,
        *,
        priority: str | None = None,
        status: str = 'pending',
        metadata: Mapping[str, Any] | None = None,
        root: str | None = None,
    ) -> int:
        self._seeded += 1
        added = self._run(self._backend.add_task(
            project_root=root or self.root,
            title=f'fixture task {self._seeded}',
            description=f'seeded completion fixture task {self._seeded}',
            priority=priority,
            status=status,
            metadata=json.dumps(dict(metadata)) if metadata is not None else None,
        ))
        return int(added['id'])

    def seed_run(self, run_id: str, *spec: tuple[str | None, str], root: str | None = None) -> list[int]:
        return [
            self.seed(priority=priority, status=status, metadata={crc.FINDING_RUN_KEY: run_id}, root=root)
            for priority, status in spec
        ]

    def another_root(self, name: str) -> str:
        """A second project served by the same backend and tools, as one fused-memory serves many."""
        return _git_root(self._tmp_path / name)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        answer = self._run(self._server.call_tool(name, dict(arguments)))
        assert isinstance(answer, tuple) and len(answer) == 2
        structured = answer[1]
        assert isinstance(structured, dict)
        return structured

    async def _shutdown(self) -> None:
        await self._interceptor.close()
        await self._backend.close()

    def close(self) -> None:
        try:
            self._run(self._shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=_LOOP_TIMEOUT_SECS)
            self._loop.close()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[FixtureStore]:
    fixture = FixtureStore(tmp_path)
    try:
        yield fixture
    finally:
        fixture.close()


class Recording:
    """Forwards each call to *call_tool* and records ``(name, arguments)``."""

    def __init__(self, call_tool: CallTool) -> None:
        self._call_tool = call_tool
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.replies: list[Mapping[str, Any]] = []

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((name, dict(arguments)))
        reply = self._call_tool(name, arguments)
        self.replies.append(reply)
        return reply


def recording(call_tool: CallTool) -> Recording:
    return Recording(call_tool)


def _answering(reply: Mapping[str, Any]) -> CallTool:
    return lambda name, arguments: reply


# ── The selection: run_tasks over the real lookup tool ──


def test_run_tasks_selects_every_status_of_the_run_and_nothing_else(store):
    ours = [
        store.seed(priority='high', status=status, metadata={'x_finding_run': RUN})
        for status in ('pending', 'done', 'cancelled', 'deferred')
    ]
    store.seed(priority='high', metadata={'x_finding_run': OTHER_RUN})
    store.seed(priority='high', metadata={'x_other': RUN})
    store.seed(priority='high')

    selection = crc.run_tasks(store.root, RUN, call_tool=store.call_tool)

    assert sorted(row['task_id'] for row in selection.rows) == sorted(ours)
    assert {row['status'] for row in selection.rows} == {'pending', 'done', 'cancelled', 'deferred'}


def test_run_tasks_sends_the_presence_query_for_the_run(store):
    store.seed_run(RUN, ('high', 'pending'))
    calls = recording(store.call_tool)

    crc.run_tasks(store.root, RUN, call_tool=calls)

    assert calls.calls[0] == (
        LOOKUP, {'project_root': store.root, 'key': 'x_finding_run', 'value': RUN, 'limit': 100, 'offset': 0},
    )


def test_run_tasks_pages_by_the_returned_count(store):
    seeded = store.seed_run(RUN, *[('low', 'pending')] * 150)
    calls = recording(store.call_tool)

    selection = crc.run_tasks(store.root, RUN, call_tool=calls)

    assert sorted(row['task_id'] for row in selection.rows) == sorted(seeded)
    assert [arguments['offset'] for _, arguments in calls.calls] == [0, 100]


def test_run_tasks_sends_a_relative_root_absolute(store, monkeypatch):
    store.seed_run(RUN, ('high', 'pending'))
    monkeypatch.chdir(Path(store.root).parent)
    calls = recording(store.call_tool)

    selection = crc.run_tasks('project', RUN, call_tool=calls)

    assert calls.calls[0][1]['project_root'] == store.root
    assert len(selection.rows) == 1


def test_tasks_with_key_is_zero_when_no_task_carries_the_key(store):
    store.seed(priority='high')

    selection = crc.run_tasks(store.root, RUN, call_tool=store.call_tool)

    assert selection.tasks_with_key == 0
    assert selection.rows == ()


def test_tasks_with_key_counts_a_live_key_the_run_does_not_match(store):
    store.seed_run(OTHER_RUN, ('high', 'pending'), ('low', 'done'))

    selection = crc.run_tasks(store.root, RUN, call_tool=store.call_tool)

    assert selection.tasks_with_key > 0
    assert selection.rows == ()


def test_a_tool_error_payload_is_raised_with_its_text(store, tmp_path):
    plain = tmp_path / 'plain'
    plain.mkdir()
    answered = store.call_tool(LOOKUP, {'project_root': str(plain), 'key': 'x_finding_run', 'value': RUN})
    assert 'error' in answered

    with pytest.raises(crc.ToolErrorReply) as raised:
        crc.run_tasks(str(plain), RUN, call_tool=store.call_tool)

    assert answered['error'] in str(raised.value)
    assert raised.value.code is crc.FailureCode.TOOL_ERROR


def test_an_error_key_beside_results_is_still_refused():
    reply = {'error': 'boom', 'error_type': 'Odd', 'results': [], 'tasks_with_key': 0, 'pagination': {'has_more': False}}

    with pytest.raises(crc.ToolErrorReply, match='boom'):
        crc.run_tasks('/nowhere', RUN, call_tool=_answering(reply))


def test_a_raising_transport_is_store_unreachable_never_no_tasks():
    cause = RuntimeError('connection refused')

    def raising(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        raise cause

    with pytest.raises(crc.StoreUnreachable) as raised:
        crc.run_tasks('/nowhere', RUN, call_tool=raising)

    assert raised.value.__cause__ is cause
    assert raised.value.code is crc.FailureCode.STORE_UNREACHABLE


_LAST_PAGE = {'has_more': False}
_ONE_ROW = [{'task_id': 1, 'status': 'pending'}]


@pytest.mark.parametrize('reply', [
    {'results': [], 'tasks_with_key': 1, 'pagination': {'has_more': True, 'returned': 0}},
    {'tasks_with_key': 1, 'pagination': _LAST_PAGE},
    {'results': [], 'tasks_with_key': 1, 'pagination': {}},
    {'results': [], 'tasks_with_key': '1', 'pagination': _LAST_PAGE},
    {'results': [], 'tasks_with_key': True, 'pagination': _LAST_PAGE},
    {'results': [{'task_id': '1', 'status': 'pending'}], 'tasks_with_key': 1, 'pagination': _LAST_PAGE},
    {'results': [{'task_id': 1, 'status': None}], 'tasks_with_key': 1, 'pagination': _LAST_PAGE},
    {'results': ['row'], 'tasks_with_key': 1, 'pagination': _LAST_PAGE},
    {'results': _ONE_ROW, 'tasks_with_key': 1, 'pagination': {'has_more': True}},
    ['not', 'an', 'object'],
], ids=[
    'has_more_on_an_empty_page', 'no_results', 'no_has_more', 'tasks_with_key_not_an_int',
    'tasks_with_key_a_bool', 'task_id_not_an_int', 'status_not_a_str', 'row_not_an_object',
    'has_more_past_the_page_cap', 'reply_not_an_object',
])
def test_a_malformed_reply_is_refused(reply):
    with pytest.raises(crc.MalformedReply) as raised:
        crc.run_tasks('/nowhere', RUN, call_tool=_answering(reply))

    assert raised.value.code is crc.FailureCode.MALFORMED_REPLY


def test_every_lookup_failure_is_a_lookup_failed():
    for failure in (crc.StoreUnreachable, crc.ToolErrorReply, crc.MalformedReply):
        assert issubclass(failure, crc.LookupFailed)


def test_the_default_binding_posts_to_the_configured_url(store, monkeypatch):
    seeded = store.seed_run(RUN, ('high', 'pending'), ('low', 'done'))

    with StubMcpServer(stateful=False, tools=store.call_tool) as stub:
        monkeypatch.setenv('FUSED_MEMORY_MCP_URL', stub.url)
        selection = crc.run_tasks(store.root, RUN)

    assert sorted(row['task_id'] for row in selection.rows) == sorted(seeded)
    tool_calls = [
        request for request in stub.requests
        if request.http_method == 'POST' and request.path == '/mcp'
        and isinstance(request.body, dict) and request.body.get('method') == 'tools/call'
    ]
    assert [request.body['params']['name'] for request in tool_calls] == [LOOKUP]


@pytest.mark.parametrize(('answer', 'failure'), [
    (StubAnswer.IS_ERROR, crc.ToolErrorReply),
    (StubAnswer.JSON_RPC_ERROR, crc.ToolErrorReply),
    (StubAnswer.NON_JSON_TEXT, crc.MalformedReply),
    (StubAnswer.NON_JSON_BODY, crc.MalformedReply),
], ids=['is_error', 'json_rpc_error', 'non_json_text', 'non_json_body'])
def test_the_default_binding_names_what_the_store_answered(monkeypatch, answer, failure):
    with StubMcpServer(stateful=False, answer=answer) as stub:
        monkeypatch.setenv('FUSED_MEMORY_MCP_URL', stub.url)
        with pytest.raises(crc.LookupFailed) as raised:
            crc.run_tasks('/nowhere', RUN)

    assert type(raised.value) is failure
    assert isinstance(raised.value.__cause__, McpCallError)
    assert LOOKUP in str(raised.value)


def test_the_default_binding_maps_every_mcp_failure_kind():
    assert set(crc.FAILURE_BY_MCP_KIND) == set(McpFailureKind)
    assert crc.FAILURE_BY_MCP_KIND[McpFailureKind.TRANSPORT] is crc.StoreUnreachable
    assert crc.FAILURE_BY_MCP_KIND[McpFailureKind.HTTP_STATUS] is crc.StoreUnreachable


def test_the_fused_memory_url_comes_from_the_environment_or_the_dashboard_default():
    assert crc.fused_memory_url({}) == DEFAULT_FUSED_MEMORY_URLS[0]
    assert crc.fused_memory_url({'FUSED_MEMORY_MCP_URL': 'http://x:1'}) == 'http://x:1'


# ── The CLI check mode, run as the predicate runner runs it ──

SCRIPT = Path(__file__).resolve().parents[1] / 'check_run_completion.py'
VERDICT_KEYS = ['run', 'verdict', 'share', 'threshold', 'landed', 'pending', 'total', 'vacuous', 'by_priority', 'error']
NEVER_MEASURED = ('share', 'landed', 'pending', 'total', 'vacuous', 'by_priority')
COULD_NOT_CHECK = 2
_SCRIPT_TIMEOUT_SECS = 120


@dataclass(frozen=True)
class ScriptRun:
    rc: int
    lines: list[str]
    stderr: str

    @property
    def verdict(self) -> dict[str, Any]:
        return json.loads(self.lines[-1])

    @property
    def human(self) -> list[str]:
        return self.lines[:-1]


def run_script(*args: str, url: str, env: Mapping[str, str] | None = None) -> ScriptRun:
    done = subprocess.run(
        [sys.executable, '-s', str(SCRIPT), *args],
        env={**os.environ, **(env or {}), 'FUSED_MEMORY_MCP_URL': url},
        capture_output=True,
        text=True,
        timeout=_SCRIPT_TIMEOUT_SECS,
    )
    return ScriptRun(done.returncode, done.stdout.splitlines(), done.stderr)


@contextlib.contextmanager
def served(store: FixtureStore) -> Iterator[str]:
    with StubMcpServer(stateful=False, tools=store.call_tool) as stub:
        yield stub.url


def check(store: FixtureStore, run_id: str = RUN, *, root: str | None = None) -> ScriptRun:
    with served(store) as url:
        return run_script('--run', run_id, '--threshold', '0.7', '--project-root', root or store.root, url=url)


def _closed_port_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    return f'http://127.0.0.1:{port}'


def _assert_could_not_check(run: ScriptRun, code: str) -> None:
    assert run.rc == COULD_NOT_CHECK
    assert list(run.verdict) == VERDICT_KEYS
    assert run.verdict['verdict'] == 'error'
    assert run.verdict['error'] == code
    assert run.verdict['run'] == RUN
    assert run.verdict['threshold'] == 0.7


def test_boundary_row_1_exits_tempfail_with_the_verdict_as_the_last_line(store):
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))

    run = check(store)

    assert run.rc == os.EX_TEMPFAIL
    assert run.lines[-1] == BOUNDARY_ROW_1_JSON
    assert run.human
    assert run.stderr == ''


def test_boundary_row_11_exits_zero(store):
    store.seed_run(RUN, ('critical', 'done'), ('high', 'done'), ('high', 'pending'))

    run = check(store)

    assert run.rc == 0
    assert run.verdict['verdict'] == 'pass'
    assert run.verdict['share'] == 0.7


def test_an_unknown_run_on_a_live_key_is_an_empty_run(store):
    store.seed_run(OTHER_RUN, ('high', 'pending'), ('low', 'done'))

    run = check(store)

    assert run.rc == COULD_NOT_CHECK
    assert run.verdict['verdict'] == 'error'
    assert run.verdict['error'] == 'empty_run'
    assert run.verdict['total'] == 0
    assert list(run.verdict) == VERDICT_KEYS
    assert any(
        f'x_finding_run is live in {store.root}' in line and f'no task carries x_finding_run={RUN}' in line
        for line in run.human
    )


def test_a_project_where_no_task_carries_the_key_says_so(store):
    store.seed(priority='high')

    run = check(store)

    assert run.rc == COULD_NOT_CHECK
    assert run.verdict['error'] == 'empty_run'
    assert any(f'no task in {store.root} carries x_finding_run at all' in line for line in run.human)
    assert not any('is live in' in line for line in run.human)


def test_an_unknown_priority_could_not_be_checked(store):
    store.seed_run(RUN, ('high', 'done'))
    urgent = store.seed(priority='urgent', metadata={'x_finding_run': RUN})

    run = check(store)

    _assert_could_not_check(run, 'unknown_priority')
    assert any(str(urgent) in line and 'urgent' in line for line in run.human)
    for key in NEVER_MEASURED:
        assert run.verdict[key] is None


def test_a_tool_error_payload_could_not_be_checked(store, tmp_path):
    plain = tmp_path / 'plain'
    plain.mkdir()
    answered = store.call_tool(LOOKUP, {'project_root': str(plain), 'key': 'x_finding_run', 'value': RUN})

    run = check(store, root=str(plain))

    _assert_could_not_check(run, 'tool_error')
    assert any(answered['error'] in line for line in run.human)
    for key in NEVER_MEASURED:
        assert run.verdict[key] is None


def test_an_unreachable_store_could_not_be_checked(tmp_path):
    run = run_script('--run', RUN, '--threshold', '0.7', '--project-root', str(tmp_path), url=_closed_port_url())

    _assert_could_not_check(run, 'store_unreachable')
    for key in NEVER_MEASURED:
        assert run.verdict[key] is None


def test_a_tool_refused_over_the_wire_is_a_tool_error_not_an_unreachable_store(tmp_path):
    with StubMcpServer(stateful=False, answer=StubAnswer.IS_ERROR) as stub:
        run = run_script('--run', RUN, '--threshold', '0.7', '--project-root', str(tmp_path), url=stub.url)

    _assert_could_not_check(run, 'tool_error')
    assert any('scripted stub failure' in line for line in run.human)


def test_an_all_cancelled_run_passes_vacuously_on_the_cli(store):
    store.seed_run(RUN, ('high', 'cancelled'), ('low', 'cancelled'))

    run = check(store)

    assert run.rc == 0
    assert run.verdict['verdict'] == 'pass'
    assert run.verdict['vacuous'] is True


@pytest.mark.parametrize('args', [
    ['--run', RUN, '--project-root', '/tmp'],
    ['--run', RUN, '--threshold', '0', '--project-root', '/tmp'],
    ['--run', RUN, '--threshold', '1.5', '--project-root', '/tmp'],
    ['--run', RUN, '--threshold', 'abc', '--project-root', '/tmp'],
    ['--run', RUN, '--threshold', '0.7', '--project-root', '/tmp', '--project-root', '/var'],
    ['--run', RUN, '--pending'],
    ['--pending', '--threshold', '0.7'],
    ['--run', RUN, '--threshold', '0.7', '--project-root', '/tmp', '--json'],
], ids=[
    'no_threshold', 'threshold_zero', 'threshold_above_one', 'threshold_not_a_number', 'two_roots',
    'run_with_pending', 'threshold_with_pending', 'json_in_check_mode',
])
def test_a_usage_error_exits_two_with_no_verdict(args):
    run = run_script(*args, url=_closed_port_url())

    assert run.rc == COULD_NOT_CHECK
    assert run.lines == []
    assert 'usage:' in run.stderr


PENDING_OPTION_HELP = 'report the open trigger chains instead'


@pytest.mark.parametrize('args', [
    ['--help'],
    ['-h'],
    ['--run', RUN, '--threshold', '0.7', '--project-root', '/tmp', '--help'],
    ['--pending', '--help'],
], ids=['help', 'h', 'check_with_help', 'pending_with_help'])
def test_help_runs_no_check_and_exits_two(args):
    run = run_script(*args, url=_closed_port_url())

    assert run.rc == COULD_NOT_CHECK
    assert run.lines == []
    assert 'usage:' in run.stderr
    assert PENDING_OPTION_HELP in run.stderr


def test_the_script_is_an_executable_python3_predicate():
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().splitlines()[0] == '#!/usr/bin/env python3'


# ── The --pending report over open trigger chains ──

NEXT_CHECK = '2026-11-10T09:00:00+00:00'
FIRST_REARMED = '2026-11-08T09:00:00+00:00'
LAST_CHECKED = '2026-11-09T09:00:00+00:00'
NEXT_RUN = 'review-dark_factory-20261110-1'


@dataclass(frozen=True)
class Chain:
    gate: int | None
    human: int


def _chain_link(role: str, skill: str, run_id: str, superseded_by: str | None) -> dict[str, Any]:
    return {'role': role, 'skill': skill, 'run_id': run_id, 'superseded_by': superseded_by}


def predicate(args: list[str] | None) -> dict[str, Any]:
    """A completion gate's before_done as §3 files it; None leaves the args out."""
    before_done: dict[str, Any] = {
        'kind': 'predicate',
        'script': 'scripts/check_run_completion.py',
        'script_root': 'factory',
        'timeout_secs': 120,
        'recheck_secs': 86400,
    }
    if args is not None:
        before_done['args'] = args
    return before_done


def seed_chain(
    store: FixtureStore,
    *,
    skill: str = 'review',
    run_id: str = RUN,
    threshold: str = '0.7',
    streak: int | None = None,
    superseded_by: str | None = None,
    gate_status: str = 'pending',
    human_status: str = 'pending',
    root: str | None = None,
    pad: str | None = None,
    before_done: Mapping[str, Any] | None = None,
) -> Chain:
    """A final-form chain as skills/_shared/filing-the-trigger-chain.md §3 files it."""
    project = root or store.root
    link = _chain_link('completion_gate', skill, run_id, superseded_by)
    if pad is not None:
        link['pad'] = pad
    gate_metadata: dict[str, Any] = {
        'trigger_chain': link,
        'milestone': {'mode': 'dated', 'at': NEXT_CHECK},
        'before_done': before_done or predicate(
            ['--run', run_id, '--threshold', threshold, '--project-root', project],
        ),
    }
    if streak is not None:
        gate_metadata['predicate_recheck'] = {
            'consecutive': streak,
            'first_rearmed_at': FIRST_REARMED,
            'last_checked_at': LAST_CHECKED,
            'next_check_at': NEXT_CHECK,
        }
    gate = store.seed(priority='high', status=gate_status, metadata=gate_metadata, root=project)
    human_link = _chain_link('human_gate', skill, run_id, superseded_by)
    if pad is not None:
        human_link['pad'] = pad
    human = store.seed(
        priority='high', status=human_status, root=project,
        metadata={'trigger_chain': human_link, 'always_escalates': True},
    )
    return Chain(gate, human)


def seed_interim_chain(store: FixtureStore, *, run_id: str = RUN) -> Chain:
    """The interim form (§4): one human gate on a delayed milestone, no completion gate."""
    human = store.seed(priority='high', metadata={
        'trigger_chain': _chain_link('human_gate', 'review', run_id, None),
        'milestone': {'mode': 'delayed', 'after_secs': 259200},
        'always_escalates': True,
    })
    return Chain(None, human)


def pending(store: FixtureStore, *roots: str, as_json: bool = False, env: Mapping[str, str] | None = None) -> ScriptRun:
    args = ['--pending', *[arg for root in roots for arg in ('--project-root', root)]]
    with served(store) as url:
        return run_script(*args, *(['--json'] if as_json else []), url=url, env=env)


def pending_json(store: FixtureStore, *roots: str) -> dict[str, Any]:
    run = pending(store, *roots, as_json=True)
    assert run.rc == 0, run
    return json.loads('\n'.join(run.lines))


def _only_chain(document: Mapping[str, Any]) -> dict[str, Any]:
    [project] = document['projects']
    [chain] = project['chains']
    return chain


def _line_with(run: ScriptRun, fragment: str) -> str:
    matching = [line for line in run.lines if fragment in line]
    assert matching, f'no line carries {fragment!r} in {run.lines}'
    return matching[0]


def test_a_project_with_no_chain_says_none_was_ever_filed(store):
    store.seed(priority='high')

    run = pending(store, store.root)

    assert run.rc == 0
    _line_with(run, f'no trigger chain has been filed in {store.root}')


def test_a_project_whose_chains_all_closed_says_none_is_open(store):
    seed_chain(store, gate_status='done', human_status='done')

    run = pending(store, store.root)

    assert run.rc == 0
    _line_with(run, f'no open trigger chain in {store.root}')
    assert not any('no trigger chain has been filed' in line for line in run.lines)


def test_an_open_chain_reports_its_gates_and_the_live_share(store):
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    chain = seed_chain(store, streak=2)

    checked = check(store)
    run = pending(store, store.root)

    assert checked.rc == os.EX_TEMPFAIL
    assert run.rc == 0
    _line_with(run, RUN)
    _line_with(run, 'review')
    gate_line = _line_with(run, f'completion gate {chain.gate}')
    for fragment in ('pending', NEXT_CHECK, 'streak 2', 'threshold 0.7'):
        assert fragment in gate_line
    assert 'pending' in _line_with(run, f'human gate {chain.human}')
    _line_with(run, f'share {checked.verdict["share"]}')


def test_the_json_report_carries_the_check_mode_verdict(store):
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    chain = seed_chain(store, streak=2)

    checked = check(store)
    document = pending_json(store, store.root)

    [project] = document['projects']
    assert project['project_root'] == store.root
    assert project['state'] == 'open_chains'
    assert project['unreadable_task_ids'] == []
    reported = _only_chain(document)
    assert (reported['skill'], reported['run_id']) == ('review', RUN)
    assert reported['verdict'] == checked.verdict
    assert reported['completion_gate'] == {
        'id': chain.gate, 'status': 'pending', 'next_check_at': NEXT_CHECK, 'streak': 2, 'threshold': 0.7,
    }
    assert reported['human_gate'] == {'id': chain.human, 'status': 'pending'}


def test_a_passed_gate_under_a_blocked_human_gate_still_reports_its_share(store):
    store.seed_run(RUN, ('high', 'done'), ('medium', 'done'), ('low', 'pending'))
    chain = seed_chain(store, gate_status='done', human_status='blocked')

    checked = check(store)
    run = pending(store, store.root)
    reported = _only_chain(pending_json(store, store.root))

    assert checked.rc == 0
    assert run.rc == 0
    assert not any('interim form' in line for line in run.lines)
    assert 'done' in _line_with(run, f'completion gate {chain.gate}')
    assert 'blocked' in _line_with(run, f'human gate {chain.human}')
    _line_with(run, f'share {checked.verdict["share"]}')
    assert reported['completion_gate']['id'] == chain.gate
    assert reported['completion_gate']['status'] == 'done'
    assert reported['verdict'] == checked.verdict


def test_the_threshold_is_read_back_from_the_gates_own_args(store):
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    seed_chain(store, threshold='0.55')

    reported = _only_chain(pending_json(store, store.root))

    assert reported['completion_gate']['threshold'] == 0.55
    assert reported['verdict']['threshold'] == 0.55
    assert reported['completion_gate']['streak'] == 0


def test_a_superseded_chain_is_not_open(store):
    store.seed_run(RUN, ('high', 'done'))
    seed_chain(store, superseded_by=NEXT_RUN)

    [project] = pending_json(store, store.root)['projects']

    assert project['chains'] == []
    assert project['state'] == 'no_open_chains'


def test_each_named_project_is_reported_under_its_own_root(store):
    second = store.another_root('second')
    store.seed_run(RUN, ('high', 'done'))
    store.seed_run(OTHER_RUN, ('high', 'pending'), root=second)
    seed_chain(store)
    seed_chain(store, skill='hotspot-survey', run_id=OTHER_RUN, root=second)

    document = pending_json(store, store.root, second)

    reported = {project['project_root']: project['chains'] for project in document['projects']}
    assert list(reported) == [store.root, second]
    assert [chain['run_id'] for chain in reported[store.root]] == [RUN]
    assert [chain['run_id'] for chain in reported[second]] == [OTHER_RUN]


def test_with_no_root_named_the_known_roots_are_reported(store):
    store.seed_run(RUN, ('high', 'done'))
    seed_chain(store)

    run = pending(store, env={'DASHBOARD_KNOWN_PROJECT_ROOTS': store.root})

    assert run.rc == 0
    _line_with(run, store.root)
    _line_with(run, RUN)


def test_a_bad_root_is_reported_beside_the_good_one(store, tmp_path):
    store.seed_run(RUN, ('high', 'done'))
    chain = seed_chain(store)
    plain = tmp_path / 'plain'
    plain.mkdir()
    answered = store.call_tool(LOOKUP, {'project_root': str(plain), 'key': 'trigger_chain'})

    run = pending(store, store.root, str(plain))

    assert run.rc == COULD_NOT_CHECK
    _line_with(run, f'completion gate {chain.gate}')
    bad_line = _line_with(run, answered['error'])
    assert str(plain) in bad_line


def test_an_unreadable_trigger_chain_is_named_not_skipped(store):
    bogus = store.seed(priority='high', metadata={
        'trigger_chain': {'role': 'bogus', 'skill': 'review', 'run_id': RUN, 'superseded_by': None},
    })

    run = pending(store, store.root)

    _line_with(run, f'unreadable trigger_chain on task {bogus}')


def test_an_interim_chain_is_listed_with_its_human_gate(store):
    store.seed_run(RUN, ('high', 'pending'))
    chain = seed_interim_chain(store)

    run = pending(store, store.root)
    reported = _only_chain(pending_json(store, store.root))

    assert run.rc == 0
    _line_with(run, 'none (interim form)')
    _line_with(run, f'human gate {chain.human}')
    assert not any(line.strip().startswith('share') for line in run.lines)
    assert reported['completion_gate'] is None
    assert reported['verdict'] is None
    assert reported['human_gate'] == {'id': chain.human, 'status': 'pending'}


def test_a_chain_refiled_for_the_same_run_reports_its_live_gates(store):
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    seed_chain(store, gate_status='cancelled', human_status='cancelled')
    refiled = seed_chain(store, threshold='0.55')

    run = pending(store, store.root)
    [project] = pending_json(store, store.root)['projects']

    assert project['state'] == 'open_chains'
    [reported] = project['chains']
    assert reported['completion_gate']['id'] == refiled.gate
    assert reported['completion_gate']['status'] == 'pending'
    assert reported['completion_gate']['threshold'] == 0.55
    assert reported['human_gate'] == {'id': refiled.human, 'status': 'pending'}
    _line_with(run, f'completion gate {refiled.gate}')


def test_with_no_project_root_the_report_says_so():
    assert crc.render_text([]) == ['no project root to report: none named and none discovered']


ASKING_FOR_HELP = [['--help'], ['-h'], ['--he']]
ASKING_FOR_HELP_IDS = ['help', 'h', 'abbreviated']


def _seed_beside_a_gate(store: FixtureStore, before_done: Mapping[str, Any]) -> Chain:
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    seed_chain(store)
    store.seed_run(OTHER_RUN, ('low', 'done'))
    return seed_chain(store, skill='hotspot-survey', run_id=OTHER_RUN, before_done=before_done)


@pytest.mark.parametrize('help_args', ASKING_FOR_HELP, ids=ASKING_FOR_HELP_IDS)
def test_a_gate_asking_for_help_is_unreadable_not_the_whole_report(store, help_args):
    helped = _seed_beside_a_gate(store, predicate(help_args))

    checked = check(store)
    run = pending(store, store.root)

    assert run.rc == 0, run
    _line_with(run, f'share {checked.verdict["share"]}')
    assert 'unreadable' in _line_with(run, f'completion gate {helped.gate}:')
    assert not any('usage:' in line or PENDING_OPTION_HELP in line for line in run.lines)


@pytest.mark.parametrize(
    'gate_args', [*ASKING_FOR_HELP, None, ['--pending']], ids=[*ASKING_FOR_HELP_IDS, 'no_args', 'pending'],
)
def test_a_gate_whose_args_run_no_check_is_unreadable_in_the_json_report(store, gate_args):
    _seed_beside_a_gate(store, predicate(gate_args))

    checked = check(store)
    [project] = pending_json(store, store.root)['projects']

    chains = {chain['run_id']: chain for chain in project['chains']}
    assert chains[OTHER_RUN]['completion_gate']['threshold'] is None
    assert chains[OTHER_RUN]['completion_gate']['threshold_unreadable']
    assert chains[OTHER_RUN]['verdict'] is None
    assert chains[RUN]['verdict'] == checked.verdict


# ── --pending under the lookup's byte budget ──

SKILLS = ('review', 'hotspot-survey', 'review-all')


def _seed_padded_chains(store: FixtureStore) -> list[str]:
    run_ids = [f'{skill}-proj-20261109' for skill in SKILLS]
    for skill, run_id in zip(SKILLS, run_ids, strict=True):
        seed_chain(store, skill=skill, run_id=run_id, pad='p' * 20000)
    return run_ids


def test_a_row_over_the_byte_budget_is_re_read_not_dropped(store):
    pad = 'p' * 60000
    assert len(json.dumps(pad)) > RESULT_BYTE_BUDGET
    store.seed_run(RUN, ('high', 'pending'), ('medium', 'done'), ('low', 'done'))
    chain = seed_chain(store, pad=pad)
    elided = store.call_tool(LOOKUP, {
        'project_root': store.root, 'key': 'trigger_chain', 'include_value': True, 'limit': 100,
    })
    assert elided['results'][0]['truncated_value'] is True

    checked = check(store)
    document = pending_json(store, store.root)

    [project] = document['projects']
    assert project['unreadable_task_ids'] == []
    reported = _only_chain(document)
    assert (reported['skill'], reported['run_id']) == ('review', RUN)
    assert reported['completion_gate']['id'] == chain.gate
    assert reported['human_gate'] == {'id': chain.human, 'status': 'pending'}
    assert reported['verdict'] == checked.verdict


def test_chains_spread_over_budget_limited_pages_are_all_listed(store):
    run_ids = _seed_padded_chains(store)

    [project] = pending_json(store, store.root)['projects']

    assert sorted(chain['run_id'] for chain in project['chains']) == sorted(run_ids)
    assert project['unreadable_task_ids'] == []


def test_the_chain_lookup_pages_by_each_pages_returned_count(store):
    run_ids = _seed_padded_chains(store)
    calls = recording(store.call_tool)

    [report] = crc.read_pending([store.root], call_tool=calls)

    assert sorted(chain.run_id for chain in report.chains) == sorted(run_ids)
    pages = [
        (arguments['offset'], reply['pagination']['returned'])
        for (name, arguments), reply in zip(calls.calls, calls.replies, strict=True)
        if name == LOOKUP and arguments['key'] == 'trigger_chain' and arguments.get('include_value')
    ]
    assert len(pages) > 1
    assert pages[0][0] == 0
    for (offset, returned), (next_offset, _) in itertools.pairwise(pages):
        assert next_offset == offset + returned


def test_a_failed_re_read_marks_the_project_an_error(store):
    seed_chain(store, pad='p' * 60000)

    def refusing_get_task(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name == 'get_task':
            return {'error': 'gone', 'error_type': 'TaskNotFoundError'}
        return store.call_tool(name, arguments)

    [report] = crc.read_pending([store.root], call_tool=refusing_get_task)

    assert report.state is crc.ProjectState.ERROR
    assert report.error is not None and 'get_task' in report.error and 'gone' in report.error
