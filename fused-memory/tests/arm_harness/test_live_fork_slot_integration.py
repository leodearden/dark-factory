"""The live FalkorDB server reports every INFO persistence flag ForkHolder names.

Read-only: it never starts a BGSAVE or an AOF rewrite on the shared server.
Deselected by the default addopts; run it with ``uv run pytest -m integration
tests/arm_harness/test_live_fork_slot_integration.py``.
"""

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif
from falkordb.asyncio import FalkorDB

from fused_memory.arm_harness.fork_slot import ForkHolder, read_fork_holders

pytestmark = [falkor_skipif(), pytest.mark.timeout(300)]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_live_server_reports_every_fork_holder_flag() -> None:
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    try:
        raw = await client.connection.info('persistence')
        await read_fork_holders(client.connection)
    finally:
        await client.aclose()

    assert {holder.value for holder in ForkHolder} <= set(raw)
    assert all(
        isinstance(raw[holder.value], int) and raw[holder.value] in (0, 1) for holder in ForkHolder
    )
