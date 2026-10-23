"""The supersession mark: its key's standing in the metadata vocabulary."""

from __future__ import annotations

from fused_memory.memory_metadata import (
    SUPERSEDED_BY_METADATA_KEY,
    classify_unknown_keys,
    validate_memory_metadata,
)

NEW_ID = '0f6c2a4e-8b1d-4c3e-9a7f-5d2e1b0c9a8f'


class TestTheMarkKeyIsQuietVocabulary:
    def test_the_mark_raises_no_violation_even_under_the_strict_registry(self):
        assert validate_memory_metadata(
            {SUPERSEDED_BY_METADATA_KEY: NEW_ID}, enforce_kind_registry=True
        ) == []

    def test_the_mark_is_not_an_unknown_key(self):
        assert classify_unknown_keys({SUPERSEDED_BY_METADATA_KEY: NEW_ID}) == []
