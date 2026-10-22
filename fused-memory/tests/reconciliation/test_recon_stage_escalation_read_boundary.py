"""Row B8 of ``plans/escalation-store-ambiguity-prd.md`` §6.5: a recon stage cannot obtain an escalation read.

Each real stage's ``run()`` spawns a fake ``claude``, reached through
``CLAUDE_BINARY``, that records the argv and ``--mcp-config`` it was handed;
those are checked against the tool surface of a real reconciliation-identity
escalation server. PRD §11 open question 4 was resolved during α (task 3163) as
OMISSION — ``--disallowed-tools`` drops a tool from the agent's surface — so no
rejection is asserted.
"""

from __future__ import annotations

import itertools
import json
import stat
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _fm_helpers import complete_paged_read
from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity
from fastmcp import Client, FastMCP

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)

ESCALATION_URL = 'http://127.0.0.1:8103/mcp'
PROJECT_ID = 'dark_factory'
PER_TASK_ESCALATION_READS = (
    'get_pending_escalations',
    'get_escalation',
    'get_task_escalations',
    'get_task_escalation_history',
)

_FAKE_CLAUDE_BODY = '''\
import json
import os
import sys

sys.stdin.read()
argv = sys.argv[1:]
with open(argv[argv.index('--mcp-config') + 1]) as config:
    mcp_config = json.load(config)
with open(os.environ['FAKE_CLAUDE_DUMP'], 'w') as dump:
    json.dump({'argv': argv, 'mcp_config': mcp_config}, dump)
print(json.dumps({
    'result': '',
    'is_error': False,
    'subtype': 'success',
    'cost_usd': 0.0,
    'duration_ms': 1,
    'num_turns': 1,
    'session_id': 'fake-claude',
    'structured_output': {'summary': '', 'stats': {}, 'flagged_items': []},
}))
'''

STAGES = pytest.mark.parametrize(
    ('stage_cls', 'stage_id'),
    [
        (MemoryConsolidator, StageId.memory_consolidator),
        (TaskKnowledgeSync, StageId.task_knowledge_sync),
        (IntegrityCheck, StageId.integrity_check),
    ],
    ids=['stage1', 'stage2', 'stage3'],
)


def denied_by(argv: list[str]) -> set[str]:
    if '--disallowed-tools' not in argv:
        return set()
    after_flag = argv[argv.index('--disallowed-tools') + 1:]
    return set(itertools.takewhile(lambda token: not token.startswith('--'), after_flag))


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``CLAUDE_BINARY`` at a recorder; return the file it dumps what it received to."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'claude'
    script.write_text(f'#!{sys.executable}\n{_FAKE_CLAUDE_BODY}')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    dump = tmp_path / 'fake-claude-dump.json'
    monkeypatch.setenv('CLAUDE_BINARY', str(script))
    monkeypatch.setenv('FAKE_CLAUDE_DUMP', str(dump))
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)
    return dump


@pytest.fixture
def recon_escalation_server(tmp_path: Path) -> FastMCP:
    queue = EscalationQueue(tmp_path / 'recon-esc')
    identity = StoreIdentity(
        kind='reconciliation', queue_dir=queue.queue_dir, project_id=None, project_root=None,
    )
    return create_server(queue, startup_sweep=False, store_identity=identity)


async def spawn(
    stage_cls: type[BaseStage], stage_id: StageId, tmp_path: Path, dump: Path,
) -> dict[str, Any]:
    """Run the real stage to its CLI spawn and return what the child received."""
    journal = AsyncMock()
    journal.data_dir = tmp_path
    journal.record_run_session = AsyncMock()
    journal.clear_run_session = AsyncMock()
    memory = AsyncMock()
    memory.graphiti.enumerate_all_valid_edges = AsyncMock(
        return_value=({}, complete_paged_read()),
    )
    taskmaster = AsyncMock()
    taskmaster.get_statuses = AsyncMock(return_value={})
    stage = stage_cls(
        stage_id,
        memory,
        taskmaster,
        journal,
        ReconciliationConfig(
            enabled=True, sandbox_recon_agents=False, explore_codebase_root=str(tmp_path),
        ),
        scope=ProjectScope(ProjectId(PROJECT_ID), ProjectRoot(str(tmp_path))),
    )
    stage._escalation_url = ESCALATION_URL

    with patch.object(stage, 'assemble_payload', new=AsyncMock(return_value='payload')):
        await stage.run([], Watermark(project_id=PROJECT_ID), [], run_id='recon-b8')

    return json.loads(dump.read_text())


@STAGES
@pytest.mark.asyncio
async def test_b8_recon_stage_spawn_denies_both_escalation_reads(
    stage_cls: type[BaseStage], stage_id: StageId, tmp_path: Path, fake_claude: Path,
) -> None:
    received = await spawn(stage_cls, stage_id, tmp_path, fake_claude)

    denied = denied_by(received['argv'])
    assert 'mcp__escalation__get_pending_escalations' in denied
    assert 'mcp__escalation__get_task_escalation_history' in denied


@STAGES
@pytest.mark.asyncio
async def test_b8_recon_stage_cannot_obtain_an_escalation_read_result(
    stage_cls: type[BaseStage],
    stage_id: StageId,
    tmp_path: Path,
    fake_claude: Path,
    recon_escalation_server: FastMCP,
) -> None:
    received = await spawn(stage_cls, stage_id, tmp_path, fake_claude)
    async with Client(recon_escalation_server) as client:
        registered = {tool.name for tool in await client.list_tools()}
        answer = await client.call_tool('get_pending_escalations', {'task_id': '5534'})

    servers = received['mcp_config']['mcpServers']
    keys = [key for key, entry in servers.items() if entry.get('url') == ESCALATION_URL]
    assert len(keys) == 1, servers
    key = keys[0]
    visible = {f'mcp__{key}__{name}' for name in registered} - denied_by(received['argv'])

    assert set(PER_TASK_ESCALATION_READS) <= registered
    assert answer.structured_content == {'result': []}
    for read in PER_TASK_ESCALATION_READS:
        assert f'mcp__{key}__{read}' not in visible, read
