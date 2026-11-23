"""Tests for base._find_fused_memory_server injectable-candidates + corrupt-file handling,
and for base.recon_core_mcp_servers, recon's core server set built from it.

step-09 (RED) / step-10 (GREEN):
  _find_fused_memory_server gains an injectable ``candidates`` param and routes
  each candidate through load_json_or_warn so corrupt files emit a WARNING
  instead of being silently swallowed.
"""

from __future__ import annotations

import json
import logging

import pytest

from fused_memory.reconciliation.stages.base import (
    _find_fused_memory_server,
    recon_core_mcp_servers,
)

_DEFAULT_UV_STDIO = {
    'command': 'uv',
    'args': [
        'run', '--project', 'fused-memory',
        'python', '-m', 'fused_memory.server.main',
    ],
}


class TestFindFusedMemoryServerCandidates:
    """_find_fused_memory_server must support injectable candidates and loud corrupt handling."""

    def test_corrupt_mcp_json_emits_warning_and_returns_default(
        self, tmp_path, caplog
    ):
        """(a) corrupt: non-JSON bytes → WARNING emitted AND default uv-stdio config returned.

        RED: current impl swallows corrupt via `except (JSONDecodeError, KeyError): pass`
        with no log, and has no candidates param.
        """
        corrupt_path = tmp_path / '.mcp.json'
        corrupt_path.write_bytes(b'NOT VALID JSON !!!')

        with caplog.at_level(logging.WARNING):
            result = _find_fused_memory_server(candidates=[corrupt_path])

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, (
            'Expected a WARNING when .mcp.json is corrupt, got none. '
            'RED: bare except swallows the error silently; no candidates param exists.'
        )
        assert result == _DEFAULT_UV_STDIO, (
            f'Expected default uv-stdio config on corrupt file, got {result!r}. '
            'RED: candidates param not yet supported.'
        )

    def test_absent_mcp_json_is_silent_and_returns_default(self, tmp_path, caplog):
        """(b) absent: missing file → NO warning, returns default uv-stdio config."""
        absent_path = tmp_path / 'nope.mcp.json'

        with caplog.at_level(logging.WARNING):
            result = _find_fused_memory_server(candidates=[absent_path])

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f'Expected no WARNING for absent file, got: {[r.message for r in warnings]!r}'
        )
        assert result == _DEFAULT_UV_STDIO, (
            f'Expected default uv-stdio config for absent file, got {result!r}.'
        )

    def test_valid_http_mcp_json_is_returned(self, tmp_path, caplog):
        """(c) valid HTTP config → returns {'type':'http','url':'http://x'}, no warning."""
        valid_path = tmp_path / '.mcp.json'
        valid_path.write_text(json.dumps({
            'mcpServers': {
                'fused-memory': {'type': 'http', 'url': 'http://x'},
            }
        }))

        with caplog.at_level(logging.WARNING):
            result = _find_fused_memory_server(candidates=[valid_path])

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f'Expected no WARNING for valid file, got: {[r.message for r in warnings]!r}'
        )
        assert result == {'type': 'http', 'url': 'http://x'}, (
            f'Expected HTTP config dict, got {result!r}.'
        )

    def test_valid_stdio_mcp_json_is_returned(self, tmp_path):
        """(d) valid stdio config → returns command/args dict."""
        valid_path = tmp_path / '.mcp.json'
        valid_path.write_text(json.dumps({
            'mcpServers': {
                'fused-memory': {
                    'command': 'python',
                    'args': ['-m', 'fused_memory.server.main'],
                    'env': {'FOO': 'bar'},
                }
            }
        }))

        result = _find_fused_memory_server(candidates=[valid_path])
        assert result == {
            'command': 'python',
            'args': ['-m', 'fused_memory.server.main'],
            'env': {'FOO': 'bar'},
        }, f'Expected stdio config dict, got {result!r}.'


def _mcp_json(tmp_path, entry: dict):
    path = tmp_path / '.mcp.json'
    path.write_text(json.dumps({'mcpServers': {'fused-memory': entry}}))
    return [path]


class TestReconCoreMcpServers:
    """recon_core_mcp_servers is the one home of recon's fused-memory entry."""

    def test_an_http_entry_yields_the_url(self, tmp_path):
        candidates = _mcp_json(tmp_path, {'type': 'http', 'url': 'http://127.0.0.1:8002/mcp'})
        assert recon_core_mcp_servers(candidates=candidates) == {
            'fused-memory': {'type': 'http', 'url': 'http://127.0.0.1:8002/mcp'},
        }

    def test_a_stdio_entry_takes_openai_api_key_from_the_process_env(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-from-env')
        candidates = _mcp_json(tmp_path, {
            'command': 'python', 'args': ['-m', 'fused_memory.server.main'], 'env': {'FOO': 'bar'},
        })
        assert recon_core_mcp_servers(candidates=candidates)['fused-memory'] == {
            'command': 'python',
            'args': ['-m', 'fused_memory.server.main'],
            'env': {'FOO': 'bar', 'OPENAI_API_KEY': 'sk-from-env'},
        }

    def test_a_stdio_entry_keeps_its_own_openai_api_key(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-from-env')
        candidates = _mcp_json(tmp_path, {
            'command': 'python', 'args': [], 'env': {'OPENAI_API_KEY': 'sk-from-file'},
        })
        assert recon_core_mcp_servers(candidates=candidates)['fused-memory']['env'] == {
            'OPENAI_API_KEY': 'sk-from-file',
        }

    def test_a_stdio_entry_without_the_key_anywhere_gets_an_empty_one(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        candidates = _mcp_json(tmp_path, {'command': 'python', 'args': []})
        assert recon_core_mcp_servers(candidates=candidates)['fused-memory'] == {
            'command': 'python', 'args': [], 'env': {'OPENAI_API_KEY': ''},
        }

    def test_it_names_exactly_the_core_servers(self, tmp_path):
        candidates = _mcp_json(tmp_path, {'type': 'http', 'url': 'http://x'})
        assert sorted(recon_core_mcp_servers(candidates=candidates)) == ['fused-memory']
