"""Tests for base._find_fused_memory_server injectable-candidates + corrupt-file handling.

step-09 (RED) / step-10 (GREEN):
  _find_fused_memory_server gains an injectable ``candidates`` param and routes
  each candidate through load_json_or_warn so corrupt files emit a WARNING
  instead of being silently swallowed.
"""

from __future__ import annotations

import json
import logging

from fused_memory.reconciliation.stages.base import _find_fused_memory_server

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
