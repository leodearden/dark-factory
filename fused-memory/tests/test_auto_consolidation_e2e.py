"""Auto-consolidation end to end: the integration gate (task 5243, PRD task eta).

One test per row of the boundary-test sketch in
``plans/memory-auto-consolidation-prd.md`` §5 (B1-B13); the table itself lives
there and is not restated here.

The rows drive the REAL path rather than the executor alone. A
``MemoryConsolidator`` runs its full-cycle ``run()``; the stubbed Stage-1 LLM
turn calls the real ``propose_consolidation`` tool under the cycle's run id, so
the post-step reads what the tool wrote. The Mem0 corpus is the stateful fake
from ``tests/_auto_consolidation_fakes.py`` grafted onto the stage's memory
service, and the tool server is built over that same service, so the tool and
the executor share one config object. The recon ledger and the escalation
queue are real, on ``tmp_path``.

INV-10: every row that depends on the executor runs twice. Its ``reverted``
twin replaces the stage's ``run_auto_consolidation`` with a no-op and must fail
on an assertion (strict xfail), so a row that would pass without the executor
fails the suite.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from _auto_consolidation_fakes import (
    CATEGORY,
    PROJECT_ID,
    VALID_CLAIM,
    FakeMemoryService,
    FakeTaskBackend,
    enabled_config,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from mcp.server.fastmcp import FastMCP
from reconciliation.consolidator_fixtures import make_consolidator

from fused_memory.config.schema import ConsolidationAutoConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.consolidation_auto import AutoOutcome, build_auto_canonical
from fused_memory.reconciliation.consolidation_auto_executor import revert_auto_consolidation
from fused_memory.reconciliation.consolidation_proposal import (
    PROPOSER_AGENT_ID,
    RECORD_KIND_CONSOLIDATION_PROPOSAL,
    ProposalOutcome,
    outcome_from_record,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.stages import memory_consolidator as memory_consolidator_module
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import filter_task_tree
from fused_memory.server.tools import create_mcp_server

TOPIC = 'e2e-topic-alpha'
RATIONALE = 'These records state the same rule in different words.'

#: The Mem0 methods the executor, the closure scroll and the revert call.
_MEM0_SURFACE = (
    'get_memory_by_id',
    'update_memory',
    'add_memory',
    'delete_memory',
    'count_memories_by_metadata',
    'get_memories_by_metadata',
    'add_system_record',
    'search',
)


# ---- harness ----------------------------------------------------------------


@dataclass(frozen=True)
class Emission:
    """One proposal the Stage-1 LLM turn emits through the tool."""

    topic: str
    member_ids: Sequence[str]
    claim: str = VALID_CLAIM
    category: str = CATEGORY


@dataclass(frozen=True)
class CycleResult:
    report: StageReport
    emissions: tuple[dict[str, Any], ...]


@dataclass
class World:
    """One Stage-1 instance over the fake corpus, the real ledger and the real queue.

    ``fake`` is for seeding and inspection; the stage, the tool and a revert all
    reach the corpus through ``stage.memory``.
    """

    stage: MemoryConsolidator
    fake: FakeMemoryService
    ledger: ReconLedgerStore
    queue: EscalationQueue
    tasks: FakeTaskBackend
    server: FastMCP
    project_id: str
    project_root: str

    async def cycle(
        self,
        run_id: str,
        *,
        emit: Sequence[Emission] = (),
        active_tasks: Sequence[dict[str, Any]] = (),
    ) -> CycleResult:
        """One full Stage-1 cycle whose LLM turn emits *emit*, over *active_tasks*."""
        self.stage.filtered_task_tree = filter_task_tree({'tasks': list(active_tasks)})
        emissions: list[dict[str, Any]] = []

        async def llm_turn(*_args: Any, **_kwargs: Any) -> StageReport:
            for emission in emit:
                emissions.append(await self._propose(run_id, emission))
            now = datetime.now(UTC)
            return StageReport(
                stage=StageId.memory_consolidator, started_at=now, completed_at=now, stats={},
            )

        with patch.object(BaseStage, 'run', new=llm_turn):
            report = await self.stage.run(
                events=[],
                watermark=Watermark(project_id=self.project_id),
                prior_reports=[],
                run_id=run_id,
            )
        return CycleResult(report=report, emissions=tuple(emissions))

    async def _propose(self, run_id: str, emission: Emission) -> dict[str, Any]:
        result = await self.server._tool_manager.call_tool(
            'propose_consolidation',
            {
                'project_id': self.project_id,
                'run_id': run_id,
                'topic': emission.topic,
                'member_ids': list(emission.member_ids),
                'category': emission.category,
                'claim': emission.claim,
                'rationale': RATIONALE,
                'agent_id': PROPOSER_AGENT_ID,
            },
        )
        return _parse_tool_result(result)

    async def row(self, topic: str, run_id: str) -> ReconLedgerRecord:
        """The proposal row for (*topic*, *run_id*), which must exist."""
        row = await self.ledger.get_by_identity(
            project_id=self.project_id,
            record_kind=RECORD_KIND_CONSOLIDATION_PROPOSAL,
            task_id='',
            flag_type=topic,
            run_id=run_id,
        )
        assert row is not None, f'no proposal row for {topic!r} in {run_id!r}'
        return row

    def system_records(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.fake.system_records if r['metadata'].get('record_kind') == kind]

    def pending(self, category: str | None = None) -> list[Escalation]:
        return [
            esc for esc in self.queue.get_pending() if category is None or esc.category == category
        ]

    async def cycle_summary_stats(self, run_id: str) -> dict[str, Any]:
        """The stats Stage 1's cycle-summary ledger row recorded for *run_id*."""
        rows = await self.ledger.list_by_kind(self.project_id, 'cycle_summary', 'active')
        row = _only([r for r in rows if r.run_id == run_id], 'cycle summary row')
        return json.loads(row.payload_json)['stats']


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """A FastMCP ``call_tool`` result (a list of TextContent) as the tool's dict."""
    if isinstance(result, list):
        return json.loads(result[0].text)
    return result


def build_world(
    tmp_path: Path,
    ledger: ReconLedgerStore,
    *,
    project_id: str = PROJECT_ID,
    config: ConsolidationAutoConfig | None = None,
) -> World:
    stage = make_consolidator(project_root=str(tmp_path / 'project'), project_id=project_id)
    fake = FakeMemoryService(
        ledger, consolidation_auto=config if config is not None else enabled_config(),
    )
    memory = stage.memory
    for name in _MEM0_SURFACE:
        setattr(memory, name, getattr(fake, name))
    memory.recon_ledger = ledger
    memory.config.consolidation_auto = fake.config.consolidation_auto
    memory.config.mem0_update = fake.config.mem0_update
    memory.durable_queue = None
    queue = EscalationQueue(tmp_path / 'esc')
    stage._escalation_queue = queue
    tasks = FakeTaskBackend()
    assert stage.taskmaster is not None
    stage.taskmaster.add_task = tasks.add_task
    return World(
        stage=stage,
        fake=fake,
        ledger=ledger,
        queue=queue,
        tasks=tasks,
        server=create_mcp_server(memory),
        project_id=project_id,
        project_root=stage.project_root,
    )


def _only(items: Sequence[Any], what: str = 'item') -> Any:
    assert len(items) == 1, f'expected exactly one {what}, got {items!r}'
    return items[0]


def outcome(row: ReconLedgerRecord) -> ProposalOutcome:
    recorded = outcome_from_record(row)
    assert recorded is not None, f'no outcome was recorded on the {row.state!r} row'
    return recorded


async def _no_executor(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'ledger.db')
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def world(tmp_path, ledger) -> World:
    return build_world(tmp_path, ledger)


@pytest.fixture(
    params=[
        'real',
        pytest.param(
            'reverted',
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    'INV-10 (PRD eta): every executor-dependent row must be RED when '
                    'run_auto_consolidation is a no-op'
                ),
            ),
        ),
    ],
)
def executor(request, monkeypatch) -> str:
    if request.param == 'reverted':
        monkeypatch.setattr(memory_consolidator_module, 'run_auto_consolidation', _no_executor)
    return request.param


REQUIRES_EXECUTOR = pytest.mark.usefixtures('executor')


# ---- rows -------------------------------------------------------------------


@REQUIRES_EXECUTOR
@pytest.mark.asyncio
async def test_b1_fresh_cluster_passes(world):
    members = world.fake.seed_many(5)

    result = await world.cycle('run-b1', emit=[Emission(TOPIC, members)])

    assert _only(result.emissions, 'emission')['ok'] is True
    for member in members:
        assert world.fake.metadata_of(member).get('topic') == TOPIC
        assert world.fake.metadata_of(member).get('x_auto_consolidated_run') == 'run-b1'
    canonical = _only(world.fake.canonical_ids(), 'canonical')
    record = world.fake.records[canonical]
    assert record['content'] == build_auto_canonical(VALID_CLAIM, TOPIC, 5, 'run-b1')
    assert record['metadata']['topic'] == TOPIC
    assert record['metadata']['canonical'] is True
    assert record['metadata']['category'] == CATEGORY
    stamp = record['metadata']['x_auto_consolidated']
    assert (stamp['run_id'], stamp['predicate_version'], stamp['members_count']) == (
        'run-b1', '1', 5,
    )

    row = await world.row(TOPIC, 'run-b1')
    assert row.state == 'addressed'
    landed = outcome(row)
    assert landed.canonical_id == canonical
    assert landed.canonical_minted is True
    assert landed.tagged_ids == tuple(members)
    facts = _only(world.system_records('auto_consolidation'), 'system record')['metadata'][
        'x_auto_consolidation'
    ]
    assert (facts['topic'], facts['canonical_id'], facts['run_id'], facts['member_ids']) == (
        TOPIC, canonical, 'run-b1', members,
    )

    assert result.report.stats['auto_consolidations'] == 1
    assert (await world.cycle_summary_stats('run-b1'))['auto_consolidations'] == 1
    assert world.pending('milestone_gate') == []
    assert world.queue.get_pending() == []
    assert world.tasks.calls == []


async def _consolidated(world: World, run_id: str) -> tuple[list[str], str]:
    """Five members consolidated by one B1 cycle: the members and their canonical."""
    members = world.fake.seed_many(5)
    await world.cycle(run_id, emit=[Emission(TOPIC, members)])
    assert (await world.row(TOPIC, run_id)).state == 'addressed'
    return members, _only(world.fake.canonical_ids(), 'canonical')


@REQUIRES_EXECUTOR
@pytest.mark.asyncio
async def test_b2_re_emission_with_regrowth_tags_only_the_new_members(world):
    members, incumbent = await _consolidated(world, 'run-b2-1')
    fresh = world.fake.seed_many(3)
    before = world.fake.snapshot(incumbent)

    result = await world.cycle('run-b2-2', emit=[Emission(TOPIC, [*members, *fresh, incumbent])])

    landed = outcome(await world.row(TOPIC, 'run-b2-2'))
    assert landed.verdict is AutoOutcome.PASS_TAG_ONLY
    assert landed.stripped_ids == (incumbent,)
    assert landed.canonical_id == incumbent
    assert landed.canonical_minted is False
    assert landed.tagged_ids == tuple(fresh)
    for member in fresh:
        assert world.fake.metadata_of(member)['topic'] == TOPIC
        assert world.fake.metadata_of(member)['x_auto_consolidated_run'] == 'run-b2-2'
    for member in members:
        assert world.fake.metadata_of(member)['x_auto_consolidated_run'] == 'run-b2-1'
    assert world.fake.snapshot(incumbent) == before
    assert world.fake.canonical_ids() == [incumbent]
    assert result.report.stats['auto_consolidations_tag_only'] == 1
    assert result.report.stats['auto_consolidations'] == 0


@REQUIRES_EXECUTOR
@pytest.mark.asyncio
async def test_b3_fully_consolidated_re_emission_is_a_noop(world):
    members, _ = await _consolidated(world, 'run-b3-1')
    corpus = world.corpus_snapshot()
    writes = world.mem0_writes()

    result = await world.cycle('run-b3-2', emit=[Emission(TOPIC, members)])

    row = await world.row(TOPIC, 'run-b3-2')
    assert row.state == 'addressed'
    assert outcome(row).verdict is AutoOutcome.NOOP
    assert outcome(row).reason_codes == ('already_consolidated',)
    assert world.mem0_writes() == writes
    assert world.corpus_snapshot() == corpus
    assert len(world.system_records('auto_consolidation')) == 1
    assert result.report.stats['auto_consolidations_noop'] == 1


@REQUIRES_EXECUTOR
@pytest.mark.asyncio
async def test_b10_reversal_round_trips_both_runs(world):
    members, incumbent = await _consolidated(world, 'run-b10-1')
    fresh = world.fake.seed_many(3)
    await world.cycle('run-b10-2', emit=[Emission(TOPIC, [*members, *fresh, incumbent])])
    assert (await world.row(TOPIC, 'run-b10-2')).state == 'addressed'
    before = world.fake.snapshot(incumbent)

    regrowth = await revert_auto_consolidation(
        world.stage.memory, await world.row(TOPIC, 'run-b10-2'), agent_id=REVERT_AGENT,
    )

    assert regrowth.reverted is True
    assert regrowth.canonical_deleted is None
    assert regrowth.untagged == tuple(fresh)
    for member in fresh:
        assert 'topic' not in world.fake.metadata_of(member)
        assert 'x_auto_consolidated_run' not in world.fake.metadata_of(member)
    for member in members:
        assert world.fake.metadata_of(member)['topic'] == TOPIC
    assert world.fake.snapshot(incumbent) == before
    assert (await world.row(TOPIC, 'run-b10-2')).state == 'reverted'

    minting = await revert_auto_consolidation(
        world.stage.memory, await world.row(TOPIC, 'run-b10-1'), agent_id=REVERT_AGENT,
    )

    assert minting.reverted is True
    assert minting.canonical_deleted == incumbent
    assert minting.untagged == tuple(members)
    assert incumbent not in world.fake.records
    for member in [*members, *fresh]:
        assert 'topic' not in world.fake.metadata_of(member)
    assert await world.stage.memory.get_memories_by_metadata(
        project_id=PROJECT_ID, filters={'topic': TOPIC},
    ) == []
    assert world.fake.canonical_ids() == []
    assert (await world.row(TOPIC, 'run-b10-1')).state == 'reverted'
