"""Empirical probe (task 2221, W5-γ): does a recon infer=False system write
ever get silently dedup-dropped by Mem0 (§2.1's '~0.92 cosine' premise)?

Binds the premise for downstream tasks δ/λ. See module-level FINDING once
the harness below is implemented and green.
"""
from __future__ import annotations

import pytest

FIXED_RECON_SUMMARY = (
    'Stage 2 cycle summary for run task-2221-premise-probe — '
    'byte-identical fixture content issued N times to test whether '
    'Mem0.add(infer=False) ever dedup-drops a repeated recon system write.'
)


@pytest.mark.asyncio
async def test_identical_infer_false_writes_all_land_distinct(
    mock_config, recon_scope, clean_collection, monkeypatch,
):
    """N byte-identical recon-stage infer=False writes must all land distinct.

    Issues N=8 identical Mem0Backend.add(...) calls (content + metadata both
    fixed — no nonce) through the real production write path (infer=False
    pinned in Mem0Backend.add) against a real, isolated Qdrant collection.
    If the §2.1 premise ('~0.92 cosine dedup can drop a repeat recon write')
    were true, at least one of these 8 writes would return zero results
    and/or the collection would end up with fewer than 8 points.
    """
    n = 8
    backend = await _build_recon_backend(mock_config, recon_scope, monkeypatch)
    try:
        ids = []
        metadata = {
            'kind': 'cycle_summary',
            'stage': 'task_knowledge_sync',
            'run_id': 'task-2221-premise-probe-hermetic',
        }
        for _ in range(n):
            response = await backend.add(
                content=FIXED_RECON_SUMMARY,
                scope=recon_scope,
                metadata=metadata,
            )
            results = response.get('results') or []
            assert len(results) == 1, (
                f'expected exactly one result under infer=False, got {results!r}'
            )
            assert 'id' in results[0]
            ids.append(results[0]['id'])

        assert len(set(ids)) == n, (
            f'expected {n} distinct ids (no dedup drop), got {len(set(ids))} distinct: {ids!r}'
        )
        assert await backend.count(recon_scope) == n
    finally:
        await backend.close()
