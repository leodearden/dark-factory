"""Tests for vllm_bridge: protocol translation functions and VllmBridge server."""

from __future__ import annotations

import pytest

from shared.vllm_bridge import _normalize_tool_use_block


class TestNormalizeToolUseBlockJsonStringInput:

    def test_parses_jsonstring_input(self):
        """When `input` is a JSON string, it is parsed into a dict."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['input'] == {'city': 'SF'}

    def test_type_preserved(self):
        """The `type` field stays 'tool_use' after normalization."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['type'] == 'tool_use'

    def test_name_preserved(self):
        """The `name` field is preserved after normalization."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['name'] == 'get_weather'
