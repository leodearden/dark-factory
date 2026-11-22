"""Tests for fused_memory/utils/mem0_content_keys.py: what counts as a Mem0 memory's text."""

from __future__ import annotations

import types

from fused_memory.utils.mem0_content_keys import (
    MEM0_CONTENT_KEYS,
    MEM0_TEXT_KEY,
    mem0_payload_text,
)


class TestContentKeys:
    def test_the_canonical_key_order(self):
        assert MEM0_CONTENT_KEYS == ('data', 'memory', 'content')

    def test_the_text_key_mem0_writes_is_the_first_content_key(self):
        assert MEM0_TEXT_KEY == 'data'
        assert MEM0_CONTENT_KEYS[0] == MEM0_TEXT_KEY


class TestMem0PayloadText:
    def test_data_wins_when_present(self):
        payload = {'data': 'hello from data', 'memory': 'hello from memory', 'content': 'hello from content'}

        assert mem0_payload_text(payload) == 'hello from data'

    def test_falls_back_to_memory_when_data_absent(self):
        payload = {'memory': 'hello from memory', 'content': 'hello from content'}

        assert mem0_payload_text(payload) == 'hello from memory'

    def test_falls_back_to_content_when_data_and_memory_absent(self):
        assert mem0_payload_text({'content': 'hello from content'}) == 'hello from content'

    def test_empty_string_when_no_content_key_is_present(self):
        assert mem0_payload_text({'category': 'observations_and_summaries'}) == ''

    def test_empty_string_when_every_content_value_is_empty(self):
        assert mem0_payload_text({'data': '', 'memory': '', 'content': ''}) == ''

    def test_empty_string_for_an_empty_payload(self):
        assert mem0_payload_text({}) == ''

    def test_an_empty_data_value_falls_through_to_memory(self):
        assert mem0_payload_text({'data': '', 'memory': 'hello from memory'}) == 'hello from memory'

    def test_a_non_string_value_falls_through_to_the_next_key(self):
        assert mem0_payload_text({'data': None, 'memory': 'hello from memory'}) == 'hello from memory'

    def test_any_mapping_is_accepted(self):
        payload = types.MappingProxyType({'memory': 'hello from memory'})

        assert mem0_payload_text(payload) == 'hello from memory'
