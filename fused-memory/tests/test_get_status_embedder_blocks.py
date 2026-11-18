"""get_status reports the embedder identity and the config overlay it came from.

The ``embedder`` block is the startup endpoint probe's cached result (B5 at the
tool surface); the ``config_overlay`` block is the LIVE config's overlay
provenance (B19c's status half), so a hot reload is reflected at once. The
service is an AsyncMock with a fixed backend status, holding a real
``FusedMemoryConfig`` built from temp base and overlay YAML.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from _config_overlay_files import write_overlay
from _mock_openai_server import MockOpenAIServer, mock_openai_server

from fused_memory.config.overlay import OVERLAY_ENV_VAR
from fused_memory.config.reload import apply_reload
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.embedder.endpoint_probe import EmbedderEndpointProbe, ProbeFaultKind
from fused_memory.server.tools import create_mcp_server

MODEL = 'granite-x'
DIMS = 768


@pytest.fixture
def endpoint() -> Iterator[MockOpenAIServer]:
    with mock_openai_server() as server:
        server.served_models = [MODEL]
        server.embedding_len = DIMS
        yield server


@pytest.fixture
def base(tmp_path, monkeypatch, endpoint) -> Path:
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({
        'embedder': {
            'model': MODEL,
            'dimensions': DIMS,
            'providers': {'openai': {'api_key': 'test-key', 'api_url': endpoint.base_url}},
        },
    }))
    monkeypatch.setenv('CONFIG_PATH', str(path))
    monkeypatch.delenv(OVERLAY_ENV_VAR, raising=False)
    return path


def _write_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: dict[str, Any], mtime: int,
) -> Path:
    path = write_overlay(tmp_path / 'overlay.yaml', data, mtime=mtime)
    monkeypatch.setenv(OVERLAY_ENV_VAR, str(path))
    return path


def _service(config: FusedMemoryConfig) -> AsyncMock:
    svc = AsyncMock()
    svc.get_status = AsyncMock(return_value={
        'graphiti': {'connected': True},
        'mem0': {'connected': True},
        'projects': {},
        'queue': {'counts': {}, 'oldest_pending_age_seconds': None},
    })
    svc.durable_queue = MagicMock()
    svc.config = config
    return svc


async def _finished_probe(config: FusedMemoryConfig) -> EmbedderEndpointProbe:
    probe = EmbedderEndpointProbe(config.embedder, timeout_seconds=2.0)
    probe.start()
    await probe.wait()
    return probe


async def _get_status(svc: AsyncMock, **server_kwargs: Any) -> dict[str, Any]:
    server = create_mcp_server(svc, **server_kwargs)
    return await server._tool_manager.call_tool('get_status', {})


class TestEmbedderBlock:
    @pytest.mark.asyncio
    async def test_a_width_mismatch_reports_degraded_with_the_typed_fault(
        self, base, endpoint,
    ):
        endpoint.embedding_len = 1024
        config = FusedMemoryConfig()
        probe = await _finished_probe(config)

        result = await _get_status(_service(config), embedder_probe=probe)

        block = result['embedder']
        assert block['status'] == 'degraded'
        assert block['model'] == MODEL
        assert block['endpoint'] == endpoint.base_url
        assert block['dimensions'] == DIMS
        assert block['reachable'] is True
        [error] = block['errors']
        assert error['kind'] == ProbeFaultKind.WIDTH_MISMATCH.value
        assert error['expected'] == DIMS
        assert error['observed'] == 1024

    @pytest.mark.asyncio
    async def test_a_healthy_probe_reports_ok(self, base):
        config = FusedMemoryConfig()
        probe = await _finished_probe(config)

        result = await _get_status(_service(config), embedder_probe=probe)

        assert result['embedder']['status'] == 'ok'
        assert result['embedder']['errors'] == []


class TestConfigOverlayBlock:
    @pytest.mark.asyncio
    async def test_an_active_overlay_reports_its_provenance(self, base, tmp_path, monkeypatch):
        overlay = _write_overlay(
            tmp_path, monkeypatch, {'write_triage': {'t_high': 0.8}}, mtime=1_700_000_000,
        )
        config = FusedMemoryConfig()
        probe = await _finished_probe(config)

        result = await _get_status(_service(config), embedder_probe=probe)

        assert result['config_overlay'] == {
            'path': str(overlay),
            'merged_keys': ['write_triage.t_high'],
            'mtime': datetime.fromtimestamp(1_700_000_000, UTC).isoformat(),
        }

    @pytest.mark.asyncio
    async def test_no_overlay_means_no_overlay_block(self, base):
        config = FusedMemoryConfig()
        probe = await _finished_probe(config)

        result = await _get_status(_service(config), embedder_probe=probe)

        assert 'config_overlay' not in result
        assert 'embedder' in result

    @pytest.mark.asyncio
    async def test_a_hot_reloaded_overlay_is_reported_from_the_live_config(
        self, base, tmp_path, monkeypatch,
    ):
        _write_overlay(
            tmp_path, monkeypatch, {'write_triage': {'t_high': 0.8}}, mtime=1_700_000_000,
        )
        svc = _service(FusedMemoryConfig())
        probe = await _finished_probe(svc.config)
        _write_overlay(
            tmp_path, monkeypatch, {'write_triage': {'t_high': 0.7, 't_low': 0.5}},
            mtime=1_700_000_100,
        )
        report = apply_reload(svc.config, FusedMemoryConfig())
        assert 'config_overlay' in report['applied']

        result = await _get_status(svc, embedder_probe=probe)

        assert result['config_overlay']['merged_keys'] == [
            'write_triage.t_high', 'write_triage.t_low',
        ]
        assert result['config_overlay']['mtime'] == (
            datetime.fromtimestamp(1_700_000_100, UTC).isoformat()
        )


@pytest.mark.asyncio
async def test_a_server_built_without_a_probe_adds_neither_block(base, tmp_path, monkeypatch):
    _write_overlay(tmp_path, monkeypatch, {'write_triage': {'t_high': 0.8}}, mtime=1_700_000_000)

    result = await _get_status(_service(FusedMemoryConfig()))

    assert 'embedder' not in result
    assert 'config_overlay' not in result
