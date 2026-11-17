"""Contract suite for ``server/write_triage_harness.py``.

The harness is the one copy of the ``add_memory`` write-path config stand-in,
post-RRF search hit and pass-through result that the triage, near-duplicate,
entities and completion-claim tool suites share; these contracts hold for
every importer. The config and search-hit contracts are read back through
production's own resolvers and band decision, so a misspelled leaf fails here.
The pass-through contracts drive the configured mock alone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.near_duplicate_guard import (
    resolve_near_dup_guard_enabled,
    resolve_near_dup_threshold,
    resolve_topic_guard_clusters,
)
from fused_memory.server.write_triage import (
    OUTCOME_RESTATED,
    decide_band,
    resolve_bands,
    resolve_write_triage_enabled,
)
from fused_memory.server.write_triage_judge import resolve_judge_enabled
from server.write_triage_harness import (
    T_HIGH,
    T_LOW,
    configure_pass_through_add_memory,
    configure_reject_guard_config,
    configure_triage_config,
    search_candidate,
    topic_cluster,
)


def test_the_defaults_read_as_triage_on_with_the_harness_bands() -> None:
    svc = AsyncMock()
    configure_triage_config(svc)

    assert resolve_write_triage_enabled(svc) is True
    assert resolve_bands(svc) == (T_HIGH, T_LOW)
    assert resolve_near_dup_guard_enabled(svc) is True
    assert resolve_near_dup_threshold(svc) == 0.90


def test_the_llm_judge_is_off_unless_a_test_turns_it_on() -> None:
    svc = AsyncMock()
    configure_triage_config(svc)
    assert resolve_judge_enabled(svc) is False, (
        'production defaults the judge ON when the leaf is missing or not a '
        'bool, so a dropped or misspelled leaf would let every consumer\'s '
        'middle-band write place a real, billed LLM call while still passing'
    )

    configure_triage_config(svc, judge_enabled=True)
    assert resolve_judge_enabled(svc) is True


def test_the_disabled_flags_read_as_disabled() -> None:
    svc = AsyncMock()
    configure_triage_config(svc, enabled=False, near_dup_guard_enabled=False)

    assert resolve_write_triage_enabled(svc) is False
    assert resolve_near_dup_guard_enabled(svc) is False


def test_the_reject_guard_config_reads_as_configured_with_triage_off() -> None:
    svc = AsyncMock()
    cluster = topic_cluster()
    configure_reject_guard_config(
        svc,
        near_dup_guard_enabled=False,
        near_dup_threshold=0.85,
        topic_clusters=[cluster],
    )

    assert resolve_write_triage_enabled(svc) is False
    assert resolve_near_dup_guard_enabled(svc) is False
    assert resolve_near_dup_threshold(svc) == 0.85
    assert resolve_topic_guard_clusters(svc) == [cluster]


def test_the_default_candidate_lands_in_the_deterministic_band() -> None:
    decision = decide_band([search_candidate()], t_high=T_HIGH, t_low=T_LOW)

    assert decision.outcome == OUTCOME_RESTATED
    assert decision.canonical_id == 'm1'
    assert decision.similarity == pytest.approx(0.97)


@pytest.mark.asyncio
async def test_the_pass_through_result_is_returned_and_dumps_to_a_plain_dict() -> None:
    svc = AsyncMock()
    returned = configure_pass_through_add_memory(svc)

    result = await svc.add_memory(content='x')

    assert result is returned
    assert result.model_dump() == {'id': 'new-id'}


@pytest.mark.asyncio
async def test_an_explicit_dump_is_what_the_result_dumps_to() -> None:
    svc = AsyncMock()
    dump = {'id': 'child-id', 'category': 'procedural_knowledge'}
    configure_pass_through_add_memory(svc, dump=dump)

    ack = (await svc.add_memory()).model_dump()
    assert ack == dump

    ack['routed'] = 'restated'
    assert dump == {'id': 'child-id', 'category': 'procedural_knowledge'}, (
        'tools.py::add_memory mutates the ack model_dump() hands it, so the '
        'result must dump to a copy rather than to the caller\'s own dict'
    )


@pytest.mark.asyncio
async def test_each_write_through_one_service_gets_its_own_dump() -> None:
    svc = AsyncMock()
    configure_pass_through_add_memory(svc)

    first_dump = (await svc.add_memory()).model_dump()
    first_dump['routed'] = 'restated'
    second_dump = (await svc.add_memory()).model_dump()

    assert second_dump == {'id': 'new-id'}, (
        'tools.py::add_memory mutates the ack model_dump() hands it, so a dict '
        'shared across writes would leak the first write\'s ack keys into the second'
    )


@pytest.mark.asyncio
async def test_each_configured_service_gets_its_own_default_dump() -> None:
    first, second = AsyncMock(), AsyncMock()
    configure_pass_through_add_memory(first)
    configure_pass_through_add_memory(second)

    first_dump = (await first.add_memory()).model_dump()
    first_dump['routed'] = 'restated'
    second_dump = (await second.add_memory()).model_dump()

    assert second_dump == {'id': 'new-id'}, (
        'tools.py::add_memory mutates the ack model_dump() hands it, so a '
        'shared default dict would leak one test\'s ack keys into the next'
    )
    assert first_dump is not second_dump
