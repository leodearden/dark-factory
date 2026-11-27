"""The typed ``metadata.early_start`` mark and its write-boundary registration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from shared.early_start_mark import (
    EARLY_START_METADATA_KEY,
    EarlyStartMark,
    early_start_predecessor,
)
from shared.task_metadata import parse_metadata


def test_the_key_is_the_one_parse_metadata_recognises() -> None:
    assert EARLY_START_METADATA_KEY == 'early_start'

    metadata, warnings = parse_metadata(
        {EARLY_START_METADATA_KEY: {'predecessor': '10476'}}, direction='write',
    )

    assert warnings == []
    assert metadata.model_extra is not None
    mark = metadata.model_extra[EARLY_START_METADATA_KEY]
    assert isinstance(mark, EarlyStartMark)
    assert mark.predecessor == '10476'


@pytest.mark.parametrize(
    'malformed',
    [{}, {'predecessor': ''}, ['x']],
    ids=['no-predecessor', 'empty-predecessor', 'a-list'],
)
def test_a_malformed_mark_is_warned_as_a_malformed_slice(malformed: object) -> None:
    _, warnings = parse_metadata({'early_start': malformed}, direction='write')

    slice_codes = [w.code for w in warnings if w.field == 'early_start']
    assert slice_codes
    assert 'unknown_key' not in slice_codes
    assert set(slice_codes) <= {'invalid_submodel', 'wrong_cardinality'}


def test_the_reader_names_the_predecessor_of_a_well_formed_mark() -> None:
    assert early_start_predecessor({'early_start': {'predecessor': '10476'}}) == '10476'


def test_the_reader_accepts_the_typed_slice_parse_metadata_hands_back() -> None:
    metadata, _ = parse_metadata({'early_start': {'predecessor': '10476'}}, direction='read')

    assert early_start_predecessor(metadata.model_extra) == '10476'


@pytest.mark.parametrize(
    'metadata',
    [
        None,
        {},
        {'other': 1},
        {'early_start': {}},
        {'early_start': {'predecessor': ''}},
        {'early_start': ['x']},
        {'early_start': 'x'},
        {'early_start': {'predecessor': 7}},
    ],
    ids=[
        'none', 'empty', 'no-mark', 'no-predecessor', 'empty-predecessor',
        'a-list', 'a-string', 'a-non-string-predecessor',
    ],
)
def test_the_reader_answers_none_for_an_absent_or_malformed_mark(
    metadata: Mapping[str, Any] | None,
) -> None:
    assert early_start_predecessor(metadata) is None
