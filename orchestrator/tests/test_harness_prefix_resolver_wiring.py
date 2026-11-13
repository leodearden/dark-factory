"""The orchestrator harness hands its escalation server an HTTP prefix resolver.

Task 5324 (leaf δ of ``plans/uuid-prefix-resolution-prd.md``): the 8100/8102
escalation servers are built in-process by ``Harness._start_escalation_server``
and cannot import fused_memory (D7), so their uuid-prefix guard resolves over
fused-memory's ``POST /resolve-prefix`` at the configured URL. Under the D8
null sentinel fused-memory is absent by design, so the server gets no resolver
and ``create_server`` registers no guard.
"""

from __future__ import annotations

import pytest
from _escalation_server_capture import create_server_kwargs
from _workflow_helpers import _build_harness
from escalation.prefix_resolver_client import PrefixResolverClient
from shared.uuid_prefix_guard import ResolverUnavailable

from orchestrator.fm_retry import FM_NULL_SENTINEL_URL

CONFIGURED_BUT_DOWN_URL = 'http://localhost:1'


async def _server_kwargs(mock_orch_config, fused_memory_url: str) -> dict:
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test-project'
    mock_orch_config.fused_memory.url = fused_memory_url
    mock_orch_config.escalation.host = '127.0.0.1'
    mock_orch_config.escalation.port = 0
    h = _build_harness(mock_orch_config)
    h.review_checkpoint = None
    return await create_server_kwargs(h)


@pytest.mark.asyncio
async def test_the_resolver_is_a_client_of_the_configured_fused_memory(mock_orch_config) -> None:
    server_kwargs = await _server_kwargs(mock_orch_config, CONFIGURED_BUT_DOWN_URL)
    assert 'prefix_resolver' in server_kwargs, (
        f'prefix_resolver not passed to create_server; got {sorted(server_kwargs)}'
    )
    assert server_kwargs['prefix_resolver'].__self__ == PrefixResolverClient(
        base_url=CONFIGURED_BUT_DOWN_URL,
    )


@pytest.mark.asyncio
async def test_an_unreachable_fused_memory_is_reported_unavailable(mock_orch_config) -> None:
    server_kwargs = await _server_kwargs(mock_orch_config, CONFIGURED_BUT_DOWN_URL)
    with pytest.raises(ResolverUnavailable) as raised:
        await server_kwargs['prefix_resolver']('test-project', 'bff81530')

    assert raised.value.store == 'fused-memory'


@pytest.mark.asyncio
async def test_the_null_sentinel_gets_no_resolver(mock_orch_config) -> None:
    server_kwargs = await _server_kwargs(mock_orch_config, FM_NULL_SENTINEL_URL)

    assert 'prefix_resolver' in server_kwargs
    assert server_kwargs['prefix_resolver'] is None
