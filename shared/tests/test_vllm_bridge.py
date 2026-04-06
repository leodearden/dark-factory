"""Tests for vllm_bridge: protocol translation functions and VllmBridge server."""

from __future__ import annotations

import pytest

from shared.vllm_bridge import _normalize_tool_use_block, _translate_messages_response


class TestTranslateMessagesResponseConvertsOpenAIToolCalls:

    def test_converts_openai_tool_calls_to_anthropic_content_list(self):
        """OpenAI-style top-level tool_calls are converted to Anthropic content blocks."""
        body = {
            'role': 'assistant',
            'content': 'I will look.',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'Read', 'arguments': '{"path": "/x"}'},
                }
            ],
            'stop_reason': 'tool_calls',
        }
        result = _translate_messages_response(body)
        # content is a list
        assert isinstance(result['content'], list)
        # contains a text block
        text_blocks = [b for b in result['content'] if b.get('type') == 'text']
        assert len(text_blocks) == 1
        assert text_blocks[0]['text'] == 'I will look.'
        # contains a tool_use block with normalised fields
        tool_blocks = [b for b in result['content'] if b.get('type') == 'tool_use']
        assert len(tool_blocks) == 1
        tb = tool_blocks[0]
        assert tb['name'] == 'Read'
        assert tb['id'].startswith('toolu_')
        assert isinstance(tb['input'], dict)
        assert tb['input'] == {'path': '/x'}
        # top-level tool_calls is gone
        assert 'tool_calls' not in result


class TestNormalizeToolUseBlockIdempotent:

    def test_idempotent_for_correct_block(self):
        """A well-formed Anthropic tool_use block is returned value-equal (no-op)."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'Read',
            'input': {'path': '/tmp/x'},
        }
        result = _normalize_tool_use_block(block)
        assert result == block


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
