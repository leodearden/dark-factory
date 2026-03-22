"""Tests for the extract_json() utility in json_extract.py."""

import pytest

from fused_memory.routing.json_extract import extract_json


class TestExtractJson:
    """Tests for the extract_json() function."""

    def test_simple_flat_json(self):
        """Simple JSON with no nesting is returned correctly."""
        text = '{"primary": "entities_and_relations", "confidence": 0.9}'
        result = extract_json(text)
        assert result is not None
        assert result == '{"primary": "entities_and_relations", "confidence": 0.9}'

    def test_nested_braces_in_value(self):
        """JSON with nested braces in a string value is extracted whole."""
        text = '{"primary": "decisions_and_rationale", "reasoning": "chose X because {rationale here}", "confidence": 0.8}'
        result = extract_json(text)
        assert result is not None
        # Should parse successfully as JSON
        import json
        data = json.loads(result)
        assert data["primary"] == "decisions_and_rationale"
        assert "{rationale here}" in data["reasoning"]
        assert data["confidence"] == 0.8

    def test_json_in_markdown_code_fence(self):
        """JSON wrapped in ```json...``` fences is extracted."""
        text = '```json\n{"primary": "temporal_facts", "confidence": 0.7}\n```'
        result = extract_json(text)
        assert result is not None
        import json
        data = json.loads(result)
        assert data["primary"] == "temporal_facts"

    def test_json_in_plain_code_fence(self):
        """JSON wrapped in plain ``` fences is extracted."""
        text = '```\n{"primary": "preferences_and_norms"}\n```'
        result = extract_json(text)
        assert result is not None
        import json
        data = json.loads(result)
        assert data["primary"] == "preferences_and_norms"

    def test_multiple_json_objects_returns_first(self):
        """When multiple JSON objects appear, the first complete one is returned."""
        text = 'first: {"a": 1} second: {"b": 2}'
        result = extract_json(text)
        assert result is not None
        import json
        data = json.loads(result)
        assert data == {"a": 1}

    def test_no_json_returns_none(self):
        """Text with no JSON returns None."""
        result = extract_json("This is just plain text with no braces")
        assert result is None

    def test_text_before_and_after_ignored(self):
        """Surrounding non-JSON text is stripped."""
        text = 'Sure! Here is the result: {"key": "value"} Hope that helps.'
        result = extract_json(text)
        assert result is not None
        import json
        data = json.loads(result)
        assert data == {"key": "value"}

    def test_deeply_nested_json(self):
        """Deeply nested JSON is handled correctly."""
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        result = extract_json(text)
        assert result is not None
        import json
        data = json.loads(result)
        assert data["a"]["b"]["c"]["d"] == 1

    def test_empty_string_returns_none(self):
        """Empty input returns None."""
        result = extract_json("")
        assert result is None

    def test_unclosed_brace_returns_none(self):
        """Unbalanced/unclosed JSON returns None."""
        result = extract_json('{"key": "value"')
        assert result is None
