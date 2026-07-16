"""Tests for scripts/clear_malformed_empty_memory.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors the pattern in test_consolidate_namespace_families.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'clear_malformed_empty_memory.py'


def _load_module() -> types.ModuleType:
    """Load clear_malformed_empty_memory.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'clear_malformed_empty_memory'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Tests: extract_content
# ===========================================================================

class TestExtractContent:
    """Tests for the pure function extract_content(payload) -> str."""

    def test_returns_data_key_when_present(self):
        """'data' is the first-tried key and wins when present."""
        payload = {'data': 'hello from data', 'memory': 'hello from memory', 'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from data'

    def test_falls_back_to_memory_when_data_absent(self):
        """'memory' is used when 'data' is absent."""
        payload = {'memory': 'hello from memory', 'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from memory'

    def test_falls_back_to_content_when_data_and_memory_absent(self):
        """'content' is used when both 'data' and 'memory' are absent."""
        payload = {'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from content'

    def test_returns_empty_string_when_no_keys_present(self):
        """No content key at all -> ''."""
        payload = {'category': 'observations_and_summaries'}

        assert _mod.extract_content(payload) == ''

    def test_returns_empty_string_when_all_values_empty(self):
        """Every content key present but empty -> ''."""
        payload = {'data': '', 'memory': '', 'content': ''}

        assert _mod.extract_content(payload) == ''

    def test_returns_empty_string_for_empty_payload(self):
        """An entirely empty payload dict -> ''."""
        assert _mod.extract_content({}) == ''

    def test_skips_empty_data_and_falls_back_to_memory(self):
        """An empty 'data' value is treated as absent -- falls through to 'memory'."""
        payload = {'data': '', 'memory': 'hello from memory'}

        assert _mod.extract_content(payload) == 'hello from memory'

    def test_skips_non_string_values(self):
        """A non-string value (e.g. None) at a higher-priority key is
        skipped, falling through to the next key."""
        payload = {'data': None, 'memory': 'hello from memory'}

        assert _mod.extract_content(payload) == 'hello from memory'


# ===========================================================================
# Tests: is_malformed_empty_payload
# ===========================================================================

class TestIsMalformedEmptyPayload:
    """Tests for the pure function is_malformed_empty_payload(payload) -> bool."""

    def test_true_when_content_category_and_agent_id_all_absent(self):
        """The exact malformed fingerprint: empty content, category=None,
        agent_id=None."""
        payload = {'data': '', 'category': None, 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is True

    def test_true_when_keys_entirely_missing(self):
        """Missing keys are treated the same as explicit None -- an empty
        dict is still malformed."""
        payload = {}

        assert _mod.is_malformed_empty_payload(payload) is True

    def test_false_when_content_present(self):
        """Non-empty content alone rules out malformed, even with
        category/agent_id both None."""
        payload = {'data': 'some real content', 'category': None, 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_when_category_present(self):
        """A set category alone rules out malformed, even with empty
        content and agent_id=None."""
        payload = {'data': '', 'category': 'observations_and_summaries', 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_when_agent_id_present(self):
        """A set agent_id alone rules out malformed, even with empty
        content and category=None."""
        payload = {'data': '', 'category': None, 'agent_id': 'claude-task-2691-implementer'}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_for_healthy_observations_and_summaries_payload(self):
        """A realistic healthy record -- non-empty content, category set,
        agent_id set -- must never be classified malformed."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }

        assert _mod.is_malformed_empty_payload(payload) is False


# ===========================================================================
# Tests: classify_payload
# ===========================================================================

class TestClassifyPayload:
    """Tests for the pure function classify_payload(payload_or_none) -> str."""

    def test_none_classifies_absent(self):
        """A None payload (record not found) classifies as 'absent'."""
        assert _mod.classify_payload(None) == 'absent'

    def test_malformed_fingerprint_classifies_malformed(self):
        """A payload matching the malformed fingerprint classifies as
        'malformed'."""
        payload = {'data': '', 'category': None, 'agent_id': None}

        assert _mod.classify_payload(payload) == 'malformed'

    def test_healthy_payload_classifies_healthy(self):
        """A payload NOT matching the malformed fingerprint classifies as
        'healthy'."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }

        assert _mod.classify_payload(payload) == 'healthy'

    def test_empty_dict_classifies_malformed_not_absent(self):
        """An empty (but present) payload dict is 'malformed', distinct
        from a None (absent) record."""
        assert _mod.classify_payload({}) == 'malformed'
