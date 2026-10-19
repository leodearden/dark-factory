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

import pytest
from _auto_consolidation_fakes import CATEGORY, VALID_CLAIM

from fused_memory.reconciliation.consolidation_auto import build_auto_canonical

TOPIC = 'e2e-topic-alpha'

REQUIRES_EXECUTOR = pytest.mark.usefixtures('executor')


@REQUIRES_EXECUTOR
@pytest.mark.asyncio
async def test_b1_fresh_cluster_passes(world):
    members = world.fake.seed_many(5)

    result = await world.cycle('run-b1', emit=[Emission(TOPIC, members)])

    assert _only(result.emissions, 'emission')['ok'] is True
    for member in members:
        assert world.fake.metadata_of(member)['topic'] == TOPIC
        assert world.fake.metadata_of(member)['x_auto_consolidated_run'] == 'run-b1'
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
