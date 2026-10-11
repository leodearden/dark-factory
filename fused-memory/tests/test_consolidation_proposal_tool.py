"""The `propose_consolidation` MCP tool (task 5239, PRD C1/D3/D12).

Stage 1 PROPOSES a near-duplicate Mem0 cluster; code decides and executes.
The tool checks the proposal at the emit boundary — one ValidationError naming
every offender, with a hint, so the stage fixes its shape in-turn — and
persists it as one recon-ledger row that always carries an expiry.

Harness per `tests/test_consolidate_memories_tool.py`: `create_mcp_server(svc)`
+ `_tool_manager.call_tool` + `_parse_result`. The config and the ledger are
REAL (`ConsolidationAutoConfig`, an initialized `ReconLedgerStore`): on a bare
AsyncMock every config leaf and every ledger read is a Mock, and a case could
pass for the wrong reason.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fused_memory_tool_surface import registered_fused_memory_tools

from fused_memory.config.schema import ConsolidationAutoConfig
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_CONSOLIDATION_PROPOSAL,
    ConsolidationProposal,
    from_ledger_record,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.server.tools import create_mcp_server

PROJECT_ID = 'dark_factory'
AGENT = 'recon-stage-memory_consolidator'
RUN_ID = 'run-abc'
TOPIC = 'memory-consolidation'
CATEGORY = 'procedural_knowledge'
CLAIM = 'Stage one keeps reproposing the identical unaddressed duplicate cluster'
OTHER_CLAIM = 'Every reconciliation pass rewrote the same rule under a fresh wording'
RATIONALE = 'Five records state the same rule in different words.'
SEEDED_FROM = 'near-duplicate soft-block naming 11111111-1111-4111-8111-111111111111'
MEMBERS = [
    '11111111-1111-4111-8111-111111111111',
    '22222222-2222-4222-8222-222222222222',
    '33333333-3333-4333-8333-333333333333',
    '44444444-4444-4444-8444-444444444444',
    '55555555-5555-4555-8555-555555555555',
]


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def svc(ledger):
    service = AsyncMock()
    service.durable_queue = None
    service.config.consolidation_auto = ConsolidationAutoConfig()
    service.recon_ledger = ledger
    return service


async def _call(server, **overrides):
    args = {
        'project_id': PROJECT_ID,
        'run_id': RUN_ID,
        'topic': TOPIC,
        'member_ids': list(MEMBERS),
        'category': CATEGORY,
        'claim': CLAIM,
        'rationale': RATIONALE,
        'seeded_from': SEEDED_FROM,
        'agent_id': AGENT,
    }
    args.update(overrides)
    return _parse_result(await server._tool_manager.call_tool('propose_consolidation', args))


async def _proposals(ledger) -> list:
    return await ledger.list_by_kind(PROJECT_ID, RECORD_KIND_CONSOLIDATION_PROPOSAL, 'proposed')


@pytest.mark.asyncio
async def test_an_index_banner_claim_is_refused_with_a_hint_and_writes_nothing(svc, ledger):
    """PRD B6: the banner shape that made the hand-folded canonicals unreadable."""
    result = await _call(create_mcp_server(svc), claim='INDEX — foo')

    assert result['error_type'] == 'ValidationError'
    assert result['hint']
    assert await _proposals(ledger) == []


@pytest.mark.asyncio
async def test_every_offender_is_named_in_one_error(svc, ledger):
    result = await _call(
        create_mcp_server(svc),
        claim='INDEX — foo',
        member_ids=['873889a1', *MEMBERS[1:]],
        category='facts',
        topic='memory_consolidation',
    )

    assert result['error_type'] == 'ValidationError'
    error = result['error']
    assert 'INDEX' in error
    assert "member_ids[0] is not a full 36-character UUID: '873889a1'" in error
    assert "'facts'" in error
    assert "'memory_consolidation'" in error
    assert await _proposals(ledger) == []


@pytest.mark.asyncio
async def test_unbounded_reviewer_text_is_refused_and_writes_nothing(svc, ledger):
    """Both fields are stored verbatim in the shared ledger and read back in full."""
    result = await _call(create_mcp_server(svc), rationale=' ', seeded_from='s' * 501)

    assert result['error_type'] == 'ValidationError'
    assert '`rationale`' in result['error']
    assert '`seeded_from` is 501 characters' in result['error']
    assert await _proposals(ledger) == []


@pytest.mark.asyncio
async def test_a_valid_proposal_lands_as_one_ledger_row(svc, ledger):
    result = await _call(create_mcp_server(svc))

    assert result == {
        'ok': True,
        'ledger_key': {
            'project_id': PROJECT_ID,
            'record_kind': RECORD_KIND_CONSOLIDATION_PROPOSAL,
            'task_id': '',
            'flag_type': TOPIC,
            'run_id': RUN_ID,
        },
    }
    row = await ledger.get_by_identity(**result['ledger_key'])
    assert row is not None
    assert await _proposals(ledger) == [row]
    assert (row.flag_type, row.run_id, row.state) == (TOPIC, RUN_ID, 'proposed')
    assert from_ledger_record(row) == ConsolidationProposal(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        topic=TOPIC,
        member_ids=tuple(MEMBERS),
        category=CATEGORY,
        claim=CLAIM,
        rationale=RATIONALE,
        seeded_from=SEEDED_FROM,
    )


@pytest.mark.asyncio
async def test_a_proposal_expires_after_the_configured_ttl(svc, ledger):
    await _call(create_mcp_server(svc))

    (row,) = await _proposals(ledger)
    assert row.expires_at is not None
    created = datetime.fromisoformat(row.created_at)
    expires = datetime.fromisoformat(row.expires_at)
    assert created.utcoffset() == timedelta(0)
    assert expires.utcoffset() == timedelta(0)
    assert expires - created == timedelta(hours=168)


@pytest.mark.asyncio
async def test_seeded_from_is_optional(svc, ledger):
    result = await _call(create_mcp_server(svc), seeded_from=None)

    assert result['ok'] is True
    (row,) = await _proposals(ledger)
    assert from_ledger_record(row).seeded_from is None


@pytest.mark.asyncio
async def test_the_ttl_is_read_live_on_every_call(svc, ledger):
    """A captured TTL would cost the leaf its green tier: a reload would not apply."""
    server = create_mcp_server(svc)

    await _call(server, topic='first-cluster')
    svc.config.consolidation_auto.proposal_ttl_hours = 1
    await _call(server, topic='second-cluster')

    ttls = {
        row.flag_type: datetime.fromisoformat(row.expires_at)
        - datetime.fromisoformat(row.created_at)
        for row in await _proposals(ledger)
        if row.expires_at is not None
    }
    assert ttls == {'first-cluster': timedelta(hours=168), 'second-cluster': timedelta(hours=1)}


@pytest.mark.asyncio
async def test_the_last_write_wins_per_project_topic_and_run(svc, ledger):
    server = create_mcp_server(svc)

    await _call(server)
    await _call(server, claim=OTHER_CLAIM)
    (row,) = await _proposals(ledger)
    assert from_ledger_record(row).claim == OTHER_CLAIM

    await _call(server, run_id='run-def')
    assert {row.run_id for row in await _proposals(ledger)} == {RUN_ID, 'run-def'}


@pytest.mark.asyncio
async def test_a_proposal_already_acted_on_is_never_reset(svc, ledger):
    """A reset row would hand the executor the same cluster a second time."""
    server = create_mcp_server(svc)
    await _call(server)
    (proposed,) = await _proposals(ledger)
    acted_on = dataclasses.replace(proposed, state='executed')
    await ledger.upsert(acted_on)

    result = await _call(server, claim=OTHER_CLAIM)

    assert 'ok' not in result
    assert result['error_type'] == 'ConsolidationProposalAlreadyActed'
    assert result['ledger_key']['flag_type'] == TOPIC
    assert await ledger.get_by_identity(**result['ledger_key']) == acted_on


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_id',
    [
        None,
        'curator-x',
        'orchestrator',
        'recon-stage-task_knowledge_sync',
        'recon-stage-integrity_check',
        'recon-stage-audit_trail_rotation',
    ],
)
async def test_only_recon_stage_1_may_propose(svc, ledger, agent_id):
    """PRD D12's one proposer, enforced here as well as by the stage deny lists."""
    result = await _call(create_mcp_server(svc), agent_id=agent_id)

    assert result['error_type'] == 'ConsolidationProposalNotPermitted'
    assert result['agent_id'] == agent_id
    assert await _proposals(ledger) == []


@pytest.mark.asyncio
async def test_an_unwired_ledger_fails_closed(svc):
    """An ok for a proposal that was never persisted is one nobody ever executes."""
    svc.recon_ledger = None

    result = await _call(create_mcp_server(svc))

    assert 'ok' not in result
    assert result['error_type'] == 'ReconLedgerUnavailable'


@pytest.mark.asyncio
async def test_an_unknown_project_is_refused(svc, ledger):
    server = create_mcp_server(svc, known_projects={'reify': '/tmp/reify'})

    result = await _call(server)

    assert result['error_type'] == 'ValidationError'
    assert 'not a known project' in result['error']
    assert await _proposals(ledger) == []


def test_the_tool_takes_exactly_the_proposal_fields():
    """No `metadata` parameter: the markup guard forwards `allow_mcp_markup` to
    any tool declaring one, and a proposal has nothing to carry there."""
    tool = registered_fused_memory_tools()['mcp__fused-memory__propose_consolidation']

    assert set(tool.inputSchema['properties']) == {
        'project_id',
        'run_id',
        'topic',
        'member_ids',
        'category',
        'claim',
        'rationale',
        'seeded_from',
        'agent_id',
    }


_PROPOSAL = ConsolidationProposal(
    project_id=PROJECT_ID,
    run_id=RUN_ID,
    topic=TOPIC,
    member_ids=tuple(MEMBERS),
    category=CATEGORY,
    claim=CLAIM,
    rationale=RATIONALE,
)
_NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ('now', 'ttl_hours'),
    [(datetime(2026, 10, 10, 12, 0), 168), (_NOW, 0), (_NOW, -1), (_NOW, True)],
    ids=['naive-now', 'zero-ttl', 'negative-ttl', 'bool-ttl'],
)
def test_a_row_without_a_real_expiry_is_never_built(now, ttl_hours):
    """gc() reaps only rows that carry expires_at, so this is the TTL's one guard."""
    with pytest.raises(ValueError, match='now|ttl_hours'):
        _PROPOSAL.to_ledger_record(now=now, ttl_hours=ttl_hours)


def test_a_non_utc_now_is_stored_as_utc():
    """gc() compares expires_at as text, which holds only for one offset notation."""
    now = datetime(2026, 10, 10, 14, 0, tzinfo=timezone(timedelta(hours=2)))

    record = _PROPOSAL.to_ledger_record(now=now, ttl_hours=1)

    assert record.created_at == '2026-10-10T12:00:00+00:00'
    assert record.expires_at == '2026-10-10T13:00:00+00:00'


@pytest.mark.parametrize(
    ('field', 'value', 'named'),
    [
        ('record_kind', 'mem0_tombstone', 'record_kind'),
        ('payload_json', 'not json', 'payload_json'),
        ('payload_json', '[1, 2]', 'payload_json'),
        ('payload_json', '{"member_ids": "abc", "category": "c", "claim": "c", "rationale": "r"}', 'member_ids'),
        ('payload_json', '{"member_ids": [], "category": 3, "claim": "c", "rationale": "r"}', 'category'),
        ('payload_json', '{"member_ids": [], "category": "c", "rationale": "r"}', 'claim'),
    ],
)
def test_a_malformed_row_is_refused_naming_the_field(field, value, named):
    record = dataclasses.replace(_PROPOSAL.to_ledger_record(now=_NOW, ttl_hours=1), **{field: value})

    with pytest.raises(ValueError, match=named):
        from_ledger_record(record)
