"""Tests for vllm_bridge: protocol translation functions and VllmBridge server."""

from __future__ import annotations

import pytest

from shared.vllm_bridge import _normalize_tool_use_block


class TestNormalizeToolUseBlockIdHandling:

    def test_generates_toolu_id_when_missing(self):
        """When block has no `id`, result gets an id starting with 'toolu_' of length > 7."""
        block = {'type': 'tool_use', 'name': 'get_weather', 'input': {}}
        result = _normalize_tool_use_block(block)
        assert result['id'].startswith('toolu_')
        assert len(result['id']) > 7

    def test_rewrites_openai_call_id(self):
        """When `id` is 'call_abc123', result id starts with 'toolu_'."""
        block = {'type': 'tool_use', 'id': 'call_abc123', 'name': 'Read', 'input': {}}
        result = _normalize_tool_use_block(block)
        assert result['id'].startswith('toolu_')


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
