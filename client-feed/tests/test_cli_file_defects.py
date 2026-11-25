"""`client-feed file-defects`: the defect filer run from its command line, --once and --check (plans/client-feed-prd.md §14.4)."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from _defects import reported
from _fake_fused_memory import FakeFusedMemory
from _in_dir import append_inbox
from shared.mcp_session import McpCallError, McpFailureKind
from shared.testing_mcp_stub import StubMcpServer, ToolHandler

from client_feed.cli import EXIT_LOCK_HELD, main
from client_feed.contract import INBOX_FILENAME
from client_feed.defect_ref import check_ref, portal_ref
from client_feed.filer_state import (
    CURSOR_NAME,
    LOCK_NAME,
    FilerCursor,
    FilerLockHeld,
    held_state_dir,
)
from client_feed.filer_tools import CLOSED_TOOLS, READ_TOOLS, WRITE_TOOLS
from client_feed.policy import PolicyProblem, load_filer_policy

PROJECT = 'acme-widgets'
REF = portal_ref(PROJECT, 2, 3)
AT = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Client:
    """One client project's dirs and policy file, and the fake fused-memory behind the policy's URL."""

    tmp_path: Path
    root: Path
    in_dir: Path
    out_dir: Path
    state_dir: Path
    fake: FakeFusedMemory

    def config(self, url: str, **filer: Any) -> Path:
        path = self.tmp_path / 'policy.yaml'
        path.write_text(
            yaml.safe_dump({
                'project_id': PROJECT,
                'project_root': str(self.root),
                'out_dir': str(self.out_dir),
                'in_dir': str(self.in_dir),
                'visible_globs': ['src/**'],
                'filer': {'fused_memory_url': url, 'state_dir': str(self.state_dir), **filer},
            })
        )
        return path

    def tool_names(self) -> list[str]:
        return [name for name, _ in self.fake.calls]


@pytest.fixture
def client(tmp_path: Path) -> Client:
    root = tmp_path / 'root'
    root.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    return Client(
        tmp_path=tmp_path,
        root=root,
        in_dir=tmp_path / 'in',
        out_dir=out_dir,
        state_dir=tmp_path / 'state',
        fake=FakeFusedMemory(root),
    )


@pytest.fixture
def server(client: Client) -> Iterator[StubMcpServer]:
    with StubMcpServer(stateful=False, tools=client.fake.handle, listed_tools=sorted(CLOSED_TOOLS)) as stub:
        yield stub


def _closed_url() -> str:
    with StubMcpServer(stateful=False) as stub:
        return stub.url


def test_once_files_a_restated_report_prints_its_pass_and_stores_the_cursor(
    client: Client, server: StubMcpServer, capsys: pytest.CaptureFixture[str]
) -> None:
    append_inbox(client.in_dir, reported(2, 3, at=AT))

    code = main(['file-defects', '--config', str(client.config(server.url)), '--once'])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['converged'] == [REF]
    assert printed['unconverged'] == []
    fix, gate = client.fake.tasks()
    assert fix['metadata']['defects'][0]['ref'] == REF
    assert gate['metadata']['x_client_defect_gate_for'] == REF
    assert fix['dependencies'] == [gate['id']]
    cursor = FilerCursor.model_validate_json((client.state_dir / CURSOR_NAME).read_bytes())
    assert cursor.inbox_offset == (client.in_dir / INBOX_FILENAME).stat().st_size == printed['offset_to']


@pytest.mark.parametrize('mode', [('--once',), ('--check',), ()], ids=['once', 'check', 'loop'])
def test_a_held_lock_exits_lock_held_naming_it_before_any_request(
    client: Client, server: StubMcpServer, capsys: pytest.CaptureFixture[str], mode: tuple[str, ...]
) -> None:
    append_inbox(client.in_dir, reported(2, 3, at=AT))
    config = client.config(server.url)

    with held_state_dir(load_filer_policy(config)):
        code = main(['file-defects', '--config', str(config), *mode])

    assert code == EXIT_LOCK_HELD == 3
    assert capsys.readouterr().err.strip() == str(FilerLockHeld(client.state_dir / LOCK_NAME))
    assert server.requests == ()


@pytest.mark.parametrize('mode', ['--once', '--check'])
def test_a_policy_without_a_filer_block_exits_1_naming_the_refusal(
    client: Client, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    config = client.config(_closed_url())
    mapping = yaml.safe_load(config.read_text())
    del mapping['filer']
    config.write_text(yaml.safe_dump(mapping))

    code = main(['file-defects', '--config', str(config), mode])

    assert code == 1
    assert PolicyProblem.NO_FILER_BLOCK.value in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['--once', '--check'])
def test_a_state_dir_reaching_out_dir_through_a_symlink_exits_1_untouched(
    client: Client, server: StubMcpServer, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    link = client.tmp_path / 'link'
    link.symlink_to(client.out_dir, target_is_directory=True)
    config = client.config(server.url)
    mapping = yaml.safe_load(config.read_text())
    mapping['filer']['state_dir'] = str(link / 'filer')
    config.write_text(yaml.safe_dump(mapping))

    code = main(['file-defects', '--config', str(config), mode])

    assert code == 1
    err = capsys.readouterr().err
    assert 'refused' in err
    assert str(client.out_dir) in err
    assert list(client.out_dir.iterdir()) == []
    assert server.requests == ()


def test_once_and_check_are_mutually_exclusive(client: Client) -> None:
    config = str(client.config(_closed_url()))

    with pytest.raises(SystemExit) as caught:
        main(['file-defects', '--config', config, '--once', '--check'])

    assert caught.value.code == 2


def test_check_prints_the_reads_performed_and_the_writes_withheld_and_stores_nothing(
    client: Client, server: StubMcpServer, capsys: pytest.CaptureFixture[str]
) -> None:
    append_inbox(client.in_dir, reported(2, 3, at=AT))

    code = main(['file-defects', '--config', str(client.config(server.url)), '--check'])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is True
    assert printed['problem'] is None
    assert printed['missing_tools'] == []
    assert printed['check_ref'] == check_ref(PROJECT) == f'portal:{PROJECT}:check:1'
    assert printed['check_ref_live'] is False
    calls = printed['calls']
    assert all(call['withheld'] is (call['tool'] not in READ_TOOLS) for call in calls)
    assert [call['tool'] for call in calls if call['withheld']] == [
        'submit_task',
        'submit_task',
        'add_dependency',
        'commit_planning',
    ]
    assert {'tool': 'lookup_defect_ref', 'ref': REF, 'withheld': False} in [
        {'tool': call['tool'], 'ref': call['arguments'].get('ref'), 'withheld': call['withheld']} for call in calls
    ]
    assert all(call['arguments']['project_root'] == str(client.root) for call in calls)
    assert client.fake.tasks() == ()
    assert not set(client.tool_names()) & WRITE_TOOLS
    assert ('lookup_defect_ref', check_ref(PROJECT)) in [
        (name, arguments.get('ref')) for name, arguments in client.fake.calls
    ]
    assert sorted(path.name for path in client.state_dir.iterdir()) == [LOCK_NAME]


def test_check_names_a_closed_tool_the_server_does_not_list(
    client: Client, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = sorted(CLOSED_TOOLS - {'set_task_status'})
    with StubMcpServer(stateful=False, tools=client.fake.handle, listed_tools=listed) as stub:
        code = main(['file-defects', '--config', str(client.config(stub.url)), '--check'])

    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is False
    assert printed['missing_tools'] == ['set_task_status']
    assert not set(client.tool_names()) & WRITE_TOOLS


def _serving_only(listed: list[str], fake: FakeFusedMemory) -> ToolHandler:
    """A server that answers only the tools it lists, as fused-memory refuses a tool it does not have."""

    def handle(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name not in listed:
            raise McpCallError(McpFailureKind.TOOL_ERROR, f'Unknown tool: {name}')
        return fake.handle(name, arguments)

    return handle


def test_check_names_a_missing_lookup_tool_beside_the_refused_lookup(
    client: Client, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = sorted(CLOSED_TOOLS - {'lookup_defect_ref'})
    with StubMcpServer(stateful=False, tools=_serving_only(listed, client.fake), listed_tools=listed) as stub:
        code = main(['file-defects', '--config', str(client.config(stub.url)), '--check'])

    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is False
    assert printed['missing_tools'] == ['lookup_defect_ref']
    assert 'lookup_defect_ref refused' in printed['problem']
    assert printed['check_ref_live'] is None
    assert client.tool_names() == []


def test_check_against_a_closed_port_names_the_transport_failure(
    client: Client, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(['file-defects', '--config', str(client.config(_closed_url())), '--check'])

    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is False
    assert 'TRANSPORT' in printed['problem']
    assert not (client.state_dir / CURSOR_NAME).exists()
