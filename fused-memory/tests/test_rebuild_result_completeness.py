"""RebuildResult carries the tri-state read completeness (task 6498).

MemoryService.rebuild_entity_summaries reports entities_complete /
entities_incomplete_kind / edges_complete / edges_incomplete_kind; these tests
pin that RebuildResult parses them, that a missing key reads as UNKNOWN, and
that only two True verdicts prove the corpus complete.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import pydantic_spec
from _rebuild_service_results import rebuild_service_result

from fused_memory.backends.graphiti_client import INCOMPLETE_SHORT_READ
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.maintenance.rebuild_summaries import (
    RebuildResult,
    RebuildSummariesManager,
    run_rebuild_summaries,
)


def _kind_for(verdict: bool | None) -> str | None:
    return INCOMPLETE_SHORT_READ if verdict is False else None


@pytest.mark.asyncio
async def test_manager_carries_the_four_completeness_keys():
    service = MagicMock()
    service.rebuild_entity_summaries = AsyncMock(return_value=rebuild_service_result(
        edges_complete=False, edges_incomplete_kind=INCOMPLETE_SHORT_READ,
    ))

    result = await RebuildSummariesManager(service=service, group_id='proj').run()

    assert result.entities_complete is True
    assert result.entities_incomplete_kind is None
    assert result.edges_complete is False
    assert result.edges_incomplete_kind == INCOMPLETE_SHORT_READ


def test_missing_completeness_keys_read_as_unknown_never_complete():
    result = RebuildResult.from_service_result({
        'total_entities': 1,
        'stale_entities': 0,
        'rebuilt': 0,
        'skipped': 0,
        'errors': 0,
        'details': [],
    })

    assert result.entities_complete is None
    assert result.entities_incomplete_kind is None
    assert result.edges_complete is None
    assert result.edges_incomplete_kind is None
    assert result.corpus_proven_complete is False


@pytest.mark.parametrize(
    ('entities_complete', 'edges_complete', 'expected'),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (True, None, False),
        (None, True, False),
        (None, None, False),
    ],
)
def test_corpus_proven_complete_only_when_both_reads_are_true(
    entities_complete, edges_complete, expected,
):
    result = RebuildResult(
        entities_complete=entities_complete,
        entities_incomplete_kind=_kind_for(entities_complete),
        edges_complete=edges_complete,
        edges_incomplete_kind=_kind_for(edges_complete),
    )

    assert result.corpus_proven_complete is expected


def test_completeness_summary_renders_all_four_verdicts_in_contract_order():
    result = RebuildResult(
        entities_complete=False,
        entities_incomplete_kind=INCOMPLETE_SHORT_READ,
        edges_complete=None,
        edges_incomplete_kind=None,
    )

    assert result.completeness_summary == (
        f'entities_complete=False entities_incomplete_kind={INCOMPLETE_SHORT_READ} '
        'edges_complete=None edges_incomplete_kind=None'
    )


async def _cli_summary_records(
    make_fake_maintenance_service, caplog, service_result: dict[str, Any],
) -> list[logging.LogRecord]:
    mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    service = AsyncMock()
    service.rebuild_entity_summaries = AsyncMock(return_value=service_result)
    fake_ctx = make_fake_maintenance_service(mock_cfg, service)
    with caplog.at_level(logging.INFO, logger='fused_memory.maintenance.rebuild_summaries'), patch(
        'fused_memory.maintenance.rebuild_summaries.maintenance_service',
        side_effect=fake_ctx,
    ):
        await run_rebuild_summaries(config_path='/fake/config.yaml', group_id='proj')
    return [r for r in caplog.records if 'run_rebuild_summaries complete' in r.message]


@pytest.mark.asyncio
async def test_cli_summary_reports_a_proven_complete_corpus_at_info(
    make_fake_maintenance_service, caplog,
):
    records = await _cli_summary_records(
        make_fake_maintenance_service, caplog, rebuild_service_result(),
    )

    assert len(records) == 1
    (record,) = records
    assert record.levelno == logging.INFO
    assert 'entities_complete=True' in record.getMessage()
    assert 'edges_complete=True' in record.getMessage()


@pytest.mark.asyncio
async def test_cli_summary_escalates_a_partial_corpus_to_warning(
    make_fake_maintenance_service, caplog,
):
    records = await _cli_summary_records(
        make_fake_maintenance_service, caplog,
        rebuild_service_result(edges_complete=False, edges_incomplete_kind=INCOMPLETE_SHORT_READ),
    )

    assert len(records) == 1
    (record,) = records
    assert record.levelno == logging.WARNING
    assert 'edges_complete=False' in record.getMessage()
    assert INCOMPLETE_SHORT_READ in record.getMessage()


@pytest.mark.asyncio
async def test_cli_summary_escalates_an_unknown_verdict_to_warning(
    make_fake_maintenance_service, caplog,
):
    records = await _cli_summary_records(
        make_fake_maintenance_service, caplog,
        rebuild_service_result(
            entities_complete=None, entities_incomplete_kind=None,
            edges_complete=None, edges_incomplete_kind=None,
        ),
    )

    assert len(records) == 1
    (record,) = records
    assert record.levelno == logging.WARNING
    assert 'entities_complete=None' in record.getMessage()
