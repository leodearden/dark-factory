"""B30 (plans/client-feed-prd.md §14.5): a filer run reaches only its closed tools on its own project, its policy, its inbox and its state dir, whatever proxy the environment names."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml
from _audit_hook import AuditRecord, recording
from _defects import quarantined, reported, ruled
from _fake_fused_memory import FakeFusedMemory
from _in_dir import append_inbox
from shared.testing_mcp_stub import StubMcpServer
from shared.toolcall_markup import MARKUP_OVERRIDE_KEY

from client_feed.cli import main
from client_feed.contract import INBOX_FILENAME, QuarantineReason
from client_feed.filer_tools import CLOSED_TOOLS

PROJECT = 'acme-widgets'
AT = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
PROXY_VARIABLES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')

Snapshot = frozenset[tuple[str, int, int]]


@dataclass(frozen=True)
class Scope:
    """What a filer run may touch: its policy P, inbox I and state dir S; O and R it must leave as they were."""

    policy: Path
    in_dir: Path
    out_dir: Path
    root: Path
    state_dir: Path


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]


def _snapshot(directory: Path) -> Snapshot:
    entries = (directory, *directory.rglob('*'))
    return frozenset(
        (str(path.relative_to(directory)), path.lstat().st_size, path.lstat().st_mtime_ns) for path in entries
    )


def _tool_calls(server: StubMcpServer, since: int) -> list[ToolCall]:
    bodies = (request.body for request in server.requests[since:])
    return [
        ToolCall(body['params']['name'], body['params']['arguments'])
        for body in bodies
        if isinstance(body, dict) and body.get('method') == 'tools/call'
    ]


def _assert_calls_closed(calls: list[ToolCall], scope: Scope) -> None:
    assert calls
    assert {call.name for call in calls} <= CLOSED_TOOLS
    assert all(call.arguments['project_root'] == str(scope.root) for call in calls)
    assert all(call.arguments['planning_mode'] is True for call in calls if call.name == 'submit_task')
    assert all(MARKUP_OVERRIDE_KEY not in call.arguments for call in calls)
    assert all(MARKUP_OVERRIDE_KEY not in call.arguments.get('metadata', {}) for call in calls)
    assert all(call.arguments['status'] == 'cancelled' for call in calls if call.name == 'set_task_status')


def _assert_effects_scoped(record: AuditRecord, scope: Scope, port: int) -> None:
    allowed = {scope.policy, scope.in_dir / INBOX_FILENAME}
    strays = [
        opened.path
        for opened in record.opens_outside_runtime()
        if opened.path not in allowed and not opened.path.is_relative_to(scope.state_dir)
    ]
    assert strays == []
    assert record.connects
    assert {connected.address for connected in record.connects} == {('127.0.0.1', port)}
    assert record.spawns == ()
    assert all(path.is_relative_to(scope.state_dir) for mutated in record.mutations for path in mutated.paths)


def _scoped_once(scope: Scope, server: StubMcpServer) -> list[ToolCall]:
    """One `file-defects --once`, recorded; it must leave O, I and R as they were and touch nothing outside its scope."""
    before = {directory: _snapshot(directory) for directory in (scope.out_dir, scope.in_dir, scope.root)}
    since = len(server.requests)

    with recording() as record:
        code = main(['file-defects', '--config', str(scope.policy), '--once'])

    assert code == 0
    assert {directory: _snapshot(directory) for directory in before} == before
    _assert_effects_scoped(record, scope, urlsplit(server.url).port or 0)
    calls = _tool_calls(server, since)
    _assert_calls_closed(calls, scope)
    return calls


@pytest.fixture
def scope(tmp_path: Path) -> Scope:
    root, out_dir = tmp_path / 'root', tmp_path / 'out'
    root.mkdir()
    (root / 'README.md').write_text('hello\n')
    out_dir.mkdir()
    (out_dir / 'feed.jsonl').write_text('')
    return Scope(
        policy=tmp_path / 'policy.yaml',
        in_dir=tmp_path / 'in',
        out_dir=out_dir,
        root=root,
        state_dir=tmp_path / 'state',
    )


@pytest.fixture
def dead_proxies(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with StubMcpServer(stateful=False) as dead:
        dead_url = dead.url
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, dead_url)
    yield


@pytest.mark.usefixtures('dead_proxies')
def test_b30_a_filer_run_stays_inside_its_closed_tools_its_project_and_its_state_dir(scope: Scope) -> None:
    fake = FakeFusedMemory(scope.root)
    append_inbox(
        scope.in_dir,
        reported(2, 3, at=AT, seq=1),
        reported(2, 4, at=AT, seq=2, outcome=quarantined(QuarantineReason.SUSPECTED_INJECTION)),
    )

    with StubMcpServer(stateful=False, tools=fake.handle, listed_tools=sorted(CLOSED_TOOLS)) as server:
        scope.policy.write_text(
            yaml.safe_dump({
                'project_id': PROJECT,
                'project_root': str(scope.root),
                'out_dir': str(scope.out_dir),
                'in_dir': str(scope.in_dir),
                'visible_globs': ['src/**'],
                'filer': {'fused_memory_url': server.url, 'state_dir': str(scope.state_dir)},
            })
        )
        filing = _scoped_once(scope, server)
        append_inbox(scope.in_dir, ruled(2, 3, at=AT, seq=3))
        ruling = _scoped_once(scope, server)

    assert [call.name for call in filing].count('submit_task') == 4
    assert 'set_task_status' not in [call.name for call in filing]
    assert [call.arguments['status'] for call in ruling if call.name == 'set_task_status'] == ['cancelled', 'cancelled']
    assert {task['status'] for task in fake.tasks()} == {'pending', 'cancelled'}
