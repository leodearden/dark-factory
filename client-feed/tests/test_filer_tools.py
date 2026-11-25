"""The filer's wire (plans/client-feed-prd.md §14.2 decision 7, §14.4): the closed tool set, the one failure table and the typed replies."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from _defects import declared
from _fake_fused_memory import FakeFusedMemory
from pydantic import HttpUrl
from shared.defects import DefectDeclaration, DefectStatus
from shared.mcp_session import McpCallError, McpFailureKind
from shared.testing_mcp_stub import StubMcpServer

from client_feed.contract import TaskStatus
from client_feed.filer_tools import (
    CLOSED_TOOLS,
    FILER_IDENTITY,
    READ_TOOLS,
    WRITE_TOOLS,
    Carrier,
    ClosedTools,
    Disposition,
    RefusalKind,
    TaskSubmission,
    ToolRefused,
    WithheldWrites,
    classify,
    open_session,
)
from client_feed.policy import FilerPolicy, Policy

REF = 'portal:acme-widgets:2:3'
SECRET = 'SECRET-SERVER-TEXT'


class Scripted:
    """A ToolCaller answering each tool from a queue of replies or raised errors, recording every call."""

    def __init__(self, **replies: list[dict[str, Any] | McpCallError]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        reply = self.replies[name].pop(0)
        if isinstance(reply, McpCallError):
            raise reply
        return reply


DECLARATION = {
    'ref': REF,
    'origin': 'client',
    'summary': 's',
    'declared_at': '2026-11-01T00:00:00+00:00',
    'declared_by': FILER_IDENTITY,
    'status': 'open',
    'sightings': 1,
}
WITHDRAWAL = {'event': 'declare_closure', 'ref': REF, 'closure': {'kind': 'withdrawal', 'reason': {'reason': 'not_a_defect'}}}
ACCEPTANCE = {'event': 'rule', 'ref': REF, 'ruling': {'subject': 'closure', 'accepted': True, 'ruled_by': 'operator'}}


def _submission(**metadata: Any) -> TaskSubmission:
    return TaskSubmission(
        title='t',
        description='d',
        priority='high',
        task_kind='normal',
        metadata={'source': FILER_IDENTITY, **metadata},
    )


@pytest.mark.parametrize(
    ('kind', 'code', 'disposition'),
    [
        (RefusalKind.TRANSPORT, 'transport', Disposition.RETRY),
        (RefusalKind.HTTP_STATUS, 'http_status', Disposition.RETRY),
        (RefusalKind.TOOL_ERROR, 'is_error', Disposition.FALLBACK),
        (RefusalKind.UNDECODABLE, 'undecodable', Disposition.RETRY_THEN_FALLBACK),
        (RefusalKind.JSON_RPC_ERROR, 'json_rpc_error', Disposition.RETRY_THEN_FALLBACK),
        (RefusalKind.REPLY_SHAPE, 'reply_shape', Disposition.RETRY_THEN_FALLBACK),
        (RefusalKind.REPLY_CODE, 'ReconciliationBacklogExceeded', Disposition.RETRY),
        (RefusalKind.REPLY_CODE, 'ValidationError', Disposition.FALLBACK),
        (RefusalKind.REPLY_CODE, 'defect_creation_refused', Disposition.FALLBACK),
        (RefusalKind.REPLY_CODE, 'defect_write_to_terminal_task', Disposition.ALREADY_DONE),
        (RefusalKind.REPLY_CODE, 'TaskNotFound', Disposition.RETRY_THEN_FALLBACK),
        (RefusalKind.REPLY_CODE, 'unrecognised', Disposition.RETRY_THEN_FALLBACK),
    ],
)
def test_the_one_table_classifies_every_refusal(kind: RefusalKind, code: str, disposition: Disposition) -> None:
    assert classify(kind, code) is disposition


def test_a_refusal_kind_is_a_session_failure_kind_or_a_reply_code() -> None:
    assert {kind.value for kind in RefusalKind} == {kind.value for kind in McpFailureKind} | {'REPLY_CODE'}


def test_every_call_names_the_policy_root_and_only_the_closed_tools(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)
    tools = ClosedTools(fake, tmp_path)

    fix = tools.submit_planned(_submission(defects=[DECLARATION]))
    gate = tools.submit_planned(_submission(x_client_defect_gate_for=REF))
    tools.add_dependency(fix, gate)
    tools.commit_planning((fix, gate))
    assert [carrier.task_id for carrier in tools.lookup_defect_ref(REF)] == [fix]
    assert tools.find_gates(REF) == (gate,)
    assert tools.get_task(fix).dependencies == {gate}
    tools.declare_withdrawal(fix, WITHDRAWAL)
    tools.record_acceptance(fix, REF, ACCEPTANCE)
    tools.cancel(gate)

    assert fake.task(fix)['metadata']['defects'][0]['status'] == 'withdrawn'
    assert fake.task(gate)['status'] == 'cancelled'
    assert {name for name, _ in fake.calls} == CLOSED_TOOLS
    assert all(arguments['project_root'] == str(tmp_path) for _, arguments in fake.calls)


def test_the_closed_set_is_its_reads_and_its_writes() -> None:
    assert {'lookup_defect_ref', 'find_tasks_by_metadata', 'get_task'} == READ_TOOLS
    assert {
        'submit_task',
        'add_dependency',
        'commit_planning',
        'declare_defect',
        'record_defect_outcome',
        'set_task_status',
    } == WRITE_TOOLS
    assert CLOSED_TOOLS == READ_TOOLS | WRITE_TOOLS


def test_a_submission_is_always_planned_and_names_the_filer(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)

    task_id = ClosedTools(fake, tmp_path).submit_planned(_submission())

    [(name, arguments)] = fake.calls
    assert name == 'submit_task'
    assert arguments['planning_mode'] is True
    assert arguments['agent_id'] == FILER_IDENTITY
    assert fake.task(task_id)['status'] == 'deferred'


def test_cancel_sends_cancelled_and_nothing_else(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)
    task_id = str(fake.seed_task())

    ClosedTools(fake, tmp_path).cancel(task_id)

    [(name, arguments)] = fake.calls
    assert name == 'set_task_status'
    assert arguments == {
        'id': task_id,
        'status': 'cancelled',
        'project_root': str(tmp_path),
        'agent_id': FILER_IDENTITY,
    }


@pytest.mark.parametrize(
    ('reply', 'code', 'disposition'),
    [
        ({'error': SECRET, 'error_type': 'ValidationError'}, 'ValidationError', Disposition.FALLBACK),
        ({'success': False, 'error': 'defect_creation_refused', 'detail': SECRET}, 'defect_creation_refused', Disposition.FALLBACK),
        ({'error': SECRET, 'error_type': 'ReconciliationBacklogExceeded'}, 'ReconciliationBacklogExceeded', Disposition.RETRY),
        ({'error': f'{SECRET} with spaces'}, 'unrecognised', Disposition.RETRY_THEN_FALLBACK),
    ],
    ids=['error-type', 'success-false', 'backlog', 'not-a-token'],
)
def test_an_error_reply_is_a_named_refusal_keeping_only_its_code(
    tmp_path: Path, reply: dict[str, Any], code: str, disposition: Disposition
) -> None:
    caller = Scripted(submit_task=[reply])

    with pytest.raises(ToolRefused) as caught:
        ClosedTools(caller, tmp_path).submit_planned(_submission())

    refused = caught.value
    assert (refused.tool, refused.kind, refused.code, refused.disposition) == (
        'submit_task',
        RefusalKind.REPLY_CODE,
        code,
        disposition,
    )
    assert all(str(part) in str(refused) for part in ('submit_task', RefusalKind.REPLY_CODE, code, disposition))
    assert SECRET not in str(refused)


@pytest.mark.parametrize(
    ('kind', 'code'),
    [
        (McpFailureKind.TOOL_ERROR, 'is_error'),
        (McpFailureKind.TRANSPORT, 'transport'),
        (McpFailureKind.REPLY_SHAPE, 'reply_shape'),
    ],
)
def test_a_session_failure_is_a_refusal_of_its_kind_without_the_servers_text(
    tmp_path: Path, kind: McpFailureKind, code: str
) -> None:
    caller = Scripted(submit_task=[McpCallError(kind, f'Error calling tool: {SECRET}')])

    with pytest.raises(ToolRefused) as caught:
        ClosedTools(caller, tmp_path).submit_planned(_submission())

    assert caught.value.kind is RefusalKind(kind.value)
    assert caught.value.code == code
    assert SECRET not in str(caught.value)


def test_lookup_is_a_tuple_of_typed_carriers(tmp_path: Path) -> None:
    caller = Scripted(
        lookup_defect_ref=[
            {
                'ref': REF,
                'live': True,
                'carriers': [{'task_id': 7, 'task_status': 'pending', 'defect_status': 'open'}],
                'unreadable_task_ids': [],
            }
        ]
    )

    carriers = ClosedTools(caller, tmp_path).lookup_defect_ref(REF)

    assert carriers == (Carrier(task_id='7', task_status=TaskStatus.PENDING, defect_status=DefectStatus.OPEN),)


@pytest.mark.parametrize('ids', [(4, 5), ('4', '5')], ids=['int', 'str'])
def test_get_task_is_a_typed_view_with_normalised_ids(tmp_path: Path, ids: tuple[object, object]) -> None:
    entry = declared(REF)
    caller = Scripted(
        get_task=[
            {
                'id': ids[0],
                'status': 'deferred',
                'dependencies': [ids[1]],
                'metadata': {'source': FILER_IDENTITY, 'x_client_defect': {'block': 2}, 'defects': [entry]},
            }
        ]
    )

    view = ClosedTools(caller, tmp_path).get_task('4')

    assert view.task_id == '4'
    assert view.status is TaskStatus.DEFERRED
    assert view.dependencies == frozenset({'5'})
    assert view.source == FILER_IDENTITY
    assert view.x_client_defect == {'block': 2}
    assert view.defects == (DefectDeclaration.model_validate(entry),)


def test_a_defects_entry_that_is_no_declaration_reads_as_absent(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)
    malformed = {'ref': REF, 'status': 'open'}
    task_id = str(fake.seed_task(metadata={'defects': [malformed, declared(REF), 'not-a-mapping']}))

    view = ClosedTools(fake, tmp_path).get_task(task_id)

    assert view.defects == (DefectDeclaration.model_validate(declared(REF)),)


def test_a_task_without_filer_metadata_has_no_source_or_facts(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)
    task_id = str(fake.seed_task())

    view = ClosedTools(fake, tmp_path).get_task(task_id)

    assert (view.source, view.x_client_defect, view.defects) == (None, None, ())


def test_an_unusable_task_reply_is_a_reply_shape_refusal(tmp_path: Path) -> None:
    caller = Scripted(get_task=[{'id': 4, 'status': 'no-such-status', 'dependencies': [], 'metadata': {}}])

    with pytest.raises(ToolRefused) as caught:
        ClosedTools(caller, tmp_path).get_task('4')

    assert caught.value.kind is RefusalKind.REPLY_SHAPE


def test_find_gates_pages_until_nothing_more(tmp_path: Path) -> None:
    def page(ids: list[int], *, has_more: bool) -> dict[str, Any]:
        return {
            'results': [{'task_id': task_id} for task_id in ids],
            'pagination': {'returned': len(ids), 'has_more': has_more},
        }

    caller = Scripted(find_tasks_by_metadata=[page([3, 4], has_more=True), page([9], has_more=False)])

    gates = ClosedTools(caller, tmp_path).find_gates(REF)

    assert gates == ('3', '4', '9')
    assert [arguments['offset'] for _, arguments in caller.calls] == [0, 2]
    assert all(
        (arguments['key'], arguments['value']) == ('x_client_defect_gate_for', REF) for _, arguments in caller.calls
    )


def test_a_per_id_commit_failure_is_a_refusal(tmp_path: Path) -> None:
    caller = Scripted(
        commit_planning=[
            {
                'success': False,
                'results': [
                    {'task_id': '4', 'result': {'success': True}},
                    {'task_id': '5', 'result': {'error': SECRET, 'error_type': 'DeferralInvalid'}},
                ],
            }
        ]
    )

    with pytest.raises(ToolRefused) as caught:
        ClosedTools(caller, tmp_path).commit_planning(('4', '5'))

    assert (caught.value.tool, caught.value.code) == ('commit_planning', 'DeferralInvalid')
    assert caller.calls[0][1]['task_ids'] == '4,5'


def test_withheld_writes_forward_reads_and_answer_writes_themselves(tmp_path: Path) -> None:
    fake = FakeFusedMemory(tmp_path)
    withheld = WithheldWrites(fake)
    tools = ClosedTools(withheld, tmp_path)

    fix = tools.submit_planned(_submission())
    gate = tools.submit_planned(_submission())
    tools.add_dependency(fix, gate)
    tools.commit_planning((fix, gate))
    carriers = tools.lookup_defect_ref(REF)

    assert (fix, gate, carriers) == ('planned-1', 'planned-2', ())
    assert [(call.tool, call.withheld) for call in withheld.calls] == [
        ('submit_task', True),
        ('submit_task', True),
        ('add_dependency', True),
        ('commit_planning', True),
        ('lookup_defect_ref', False),
    ]
    assert [name for name, _ in fake.calls] == ['lookup_defect_ref']
    assert fake.tasks() == ()


def _policy(tmp_path: Path, url: str) -> Policy:
    return Policy.model_validate({
        'project_id': 'acme-widgets',
        'project_root': tmp_path / 'root',
        'out_dir': tmp_path / 'out',
        'in_dir': tmp_path / 'in',
        'visible_globs': ['src/**'],
        'filer': FilerPolicy(fused_memory_url=HttpUrl(url), state_dir=tmp_path / 'state'),
    })


@pytest.mark.parametrize('variable', ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'])
def test_the_session_ignores_proxy_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    with StubMcpServer(stateful=False) as dead:
        dead_url = dead.url
    monkeypatch.setenv(variable, dead_url)
    fake = FakeFusedMemory(tmp_path / 'root')

    with StubMcpServer(stateful=False, tools=fake.handle) as server, open_session(_policy(tmp_path, server.url)) as session:
        carriers = ClosedTools(session, tmp_path / 'root').lookup_defect_ref(REF)

    assert carriers == ()
    assert [name for name, _ in fake.calls] == ['lookup_defect_ref']
