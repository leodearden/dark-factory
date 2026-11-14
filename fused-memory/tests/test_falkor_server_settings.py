"""The read-only check of the FalkorDB server settings fused-memory depends on (tasks 6330, 6848).

No FalkorDB I/O: the FalkorDB client is a fake that answers the two reads and
records every command it is sent. It has no ``config_set``, so a SET would
raise rather than pass unnoticed.
"""

import logging
from unittest.mock import MagicMock

import pytest

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.falkor_server_settings import (
    FalkorServerSettings,
    read_falkor_server_settings,
)
from fused_memory.backends.graphiti_client import GraphitiBackend

_LOGGERS = 'fused_memory.backends'


class _FakeRedisConnection:
    def __init__(self, appendonly: str, error: Exception | None, calls: list[tuple[str, str]]):
        self._appendonly = appendonly
        self._error = error
        self._calls = calls

    async def config_get(self, name: str) -> dict[str, str]:
        self._calls.append(('CONFIG GET', name))
        if self._error is not None:
            raise self._error
        return {name: self._appendonly}


class _FakeFalkorClient:
    def __init__(
        self,
        *,
        timeout: int = 30000,
        appendonly: str = 'yes',
        timeout_error: Exception | None = None,
        appendonly_error: Exception | None = None,
    ):
        self.calls: list[tuple[str, str]] = []
        self.timeout = timeout
        self._timeout_error = timeout_error
        self.connection = _FakeRedisConnection(appendonly, appendonly_error, self.calls)

    async def config_get(self, name: str) -> int:
        self.calls.append(('GRAPH.CONFIG GET', name))
        if self._timeout_error is not None:
            raise self._timeout_error
        return self.timeout


async def _read(client: _FakeFalkorClient, expected: int = 30000) -> FalkorServerSettings:
    return await read_falkor_server_settings(client, expected_graph_timeout_ms=expected)


@pytest.mark.asyncio
async def test_settings_matching_expectations_are_ok():
    settings = await _read(_FakeFalkorClient())

    assert settings.graph_timeout_ms == 30000
    assert settings.appendonly == 'yes'
    assert settings.shortfalls == ()
    assert settings.ok


@pytest.mark.asyncio
async def test_a_timeout_below_expected_is_a_shortfall_naming_both_values():
    settings = await _read(_FakeFalkorClient(timeout=1000))

    assert not settings.ok
    assert len(settings.shortfalls) == 1
    assert '1000' in settings.shortfalls[0] and '30000' in settings.shortfalls[0]


@pytest.mark.asyncio
async def test_a_timeout_above_expected_is_not_a_shortfall():
    assert (await _read(_FakeFalkorClient(timeout=60000))).ok


@pytest.mark.asyncio
async def test_timeout_zero_means_no_timeout_and_is_not_a_shortfall():
    assert (await _read(_FakeFalkorClient(timeout=0))).ok


@pytest.mark.asyncio
async def test_appendonly_off_is_a_shortfall():
    settings = await _read(_FakeFalkorClient(appendonly='no'))

    assert not settings.ok
    assert any('appendonly' in shortfall for shortfall in settings.shortfalls)


@pytest.mark.asyncio
async def test_a_failed_read_is_warned_and_recorded_never_raised(caplog):
    client = _FakeFalkorClient(timeout_error=ConnectionError('refused'))

    with caplog.at_level(logging.WARNING, logger=_LOGGERS):
        settings = await _read(client)

    assert settings.graph_timeout_ms is None
    assert settings.appendonly == 'yes'
    assert not settings.ok
    assert any('refused' in error for error in settings.read_errors)
    assert any('TIMEOUT' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_the_check_only_reads():
    client = _FakeFalkorClient(timeout=1000, appendonly='no')

    await _read(client)

    assert client.calls == [('GRAPH.CONFIG GET', 'TIMEOUT'), ('CONFIG GET', 'appendonly')]


@pytest.mark.asyncio
async def test_status_exposes_effective_and_expected_values():
    status = (await _read(_FakeFalkorClient(timeout=1000))).to_status()

    assert status['graph_timeout_ms'] == 1000
    assert status['expected_graph_timeout_ms'] == 30000
    assert status['appendonly'] == 'yes'
    assert status['ok'] is False
    assert status['shortfalls'] and status['read_errors'] == []


@pytest.fixture
def backend_over(mock_config, monkeypatch):
    """A GraphitiBackend whose driver class is stubbed so ``.client`` is the given fake."""

    def _build(fake: _FakeFalkorClient) -> GraphitiBackend:
        real_driver_cls = graphiti_client_module._MultiTenantFalkorDriver
        driver = MagicMock(spec=real_driver_cls)
        driver.client = fake
        monkeypatch.setattr(
            graphiti_client_module, '_MultiTenantFalkorDriver', MagicMock(return_value=driver),
        )
        return GraphitiBackend(mock_config, registered_graph_ids=())

    return _build


@pytest.mark.asyncio
async def test_initialize_warns_loudly_when_the_live_timeout_is_below_expected(backend_over, caplog):
    backend = backend_over(_FakeFalkorClient(timeout=1000))

    with caplog.at_level(logging.WARNING, logger=_LOGGERS):
        await backend.initialize(skip_maintenance=True)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('1000' in message and '30000' in message for message in warnings), warnings


@pytest.mark.asyncio
async def test_initialize_warns_when_appendonly_is_off(backend_over, caplog):
    backend = backend_over(_FakeFalkorClient(appendonly='no'))

    with caplog.at_level(logging.WARNING, logger=_LOGGERS):
        await backend.initialize(skip_maintenance=True)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('appendonly' in message for message in warnings), warnings


@pytest.mark.asyncio
async def test_initialize_survives_an_unreadable_server(backend_over, caplog):
    backend = backend_over(
        _FakeFalkorClient(timeout_error=OSError('down'), appendonly_error=OSError('down')),
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGERS):
        await backend.initialize(skip_maintenance=True)

    assert backend.client is not None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_initialize_does_not_warn_when_settings_match(backend_over, caplog):
    backend = backend_over(_FakeFalkorClient())

    with caplog.at_level(logging.WARNING, logger=_LOGGERS):
        await backend.initialize(skip_maintenance=True)

    assert not [r for r in caplog.records if 'FalkorDB server setting' in r.getMessage()]


@pytest.mark.asyncio
async def test_read_server_settings_reads_live_values_on_every_call(backend_over):
    fake = _FakeFalkorClient()
    backend = backend_over(fake)
    await backend.initialize(skip_maintenance=True)
    fake.timeout = 1000

    settings = await backend.read_server_settings()

    assert settings.graph_timeout_ms == 1000
    assert settings.expected_graph_timeout_ms == 30000


@pytest.mark.asyncio
async def test_read_server_settings_before_initialize_is_unreadable_not_raised(mock_config):
    settings = await GraphitiBackend(mock_config, registered_graph_ids=()).read_server_settings()

    assert not settings.ok
    assert settings.read_errors
