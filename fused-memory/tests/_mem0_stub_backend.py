"""A Mem0Backend wired to ``_stub_qdrant_server.py`` through real sockets.

The scaffolding the stub-backed Mem0Backend suites share, in one copy: a
config whose mem0 sends no telemetry and embeds through a local mock, a backend
that is always closed, and httpx's library-default read timeout, which is the
hidden deadline mem0's own Qdrant client carried.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from _mock_openai_server import mock_openai_server

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig

#: A well-formed point id that no stub collection holds.
MEMORY_ID = '00000000-0000-4000-8000-000000000001'


@contextmanager
def stub_ready_config(
    mock_config: FusedMemoryConfig, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[FusedMemoryConfig]:
    """A copy of *mock_config* whose mem0 sends no telemetry and embeds through a local mock.

    No telemetry means no posthog call and no mem0migrations collection on the stub.
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    monkeypatch.setattr('mem0.memory.main.MEM0_TELEMETRY', False)
    config = mock_config.model_copy(deep=True)
    embedder = config.embedder.providers.openai
    assert embedder is not None
    with mock_openai_server() as server:
        embedder.api_url = server.base_url
        yield config


@asynccontextmanager
async def backend_against(qdrant_url: str, config: FusedMemoryConfig) -> AsyncIterator[Mem0Backend]:
    """A Mem0Backend on *config*, pointed at *qdrant_url* and closed on exit."""
    config.mem0.qdrant_url = qdrant_url
    backend = Mem0Backend(config)
    try:
        yield backend
    finally:
        await backend.close()


def httpx_default_read_timeout() -> float:
    """httpx's library-default read timeout: the hidden deadline mem0's own Qdrant client carried."""
    with httpx.Client() as client:
        read = client.timeout.read
    assert read is not None
    return read
