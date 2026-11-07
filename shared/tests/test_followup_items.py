"""``metadata.followup_items``: the item record, its key, uniqueness, the one merge and the legacy mapping."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    MAX_FOLLOWUP_ITEMS,
    MAX_SUMMARY_CHARS,
    UNPARENTED_ORIGIN,
    BundleViolation,
    DuplicateFollowupItemKey,
    FollowupBundleInvalid,
    FollowupItem,
    item_key,
    legacy_item,
    merge_items,
    observed_item,
    require_unique_item_keys,
)
from shared.task_metadata import parse_metadata

_HEX16 = re.compile(r'[0-9a-f]{16}')


def _fields(**overrides: object) -> dict[str, object]:
    return {
        'key': 'k1',
        'summary': 'the retry loop swallows the timeout',
        'files': ['pkg/a.py'],
        'origin_task': '8414',
        **overrides,
    }


def _item(key: str, summary: str = 'an observation') -> FollowupItem:
    return FollowupItem.model_validate(_fields(key=key, summary=summary))


class TestFollowupItem:
    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FollowupItem.model_validate(_fields(title='not a field'))

    def test_the_record_is_frozen(self) -> None:
        item = FollowupItem.model_validate(_fields())
        with pytest.raises(ValidationError):
            item.summary = 'changed'  # type: ignore[misc]

    def test_the_summary_is_stored_as_one_line(self) -> None:
        item = FollowupItem.model_validate(_fields(summary='  line one\n\n  line\ttwo  '))
        assert item.summary == 'line one line two'

    @pytest.mark.parametrize('summary', ['', '   ', '\n\t\n'])
    def test_a_blank_summary_is_refused(self, summary: str) -> None:
        with pytest.raises(ValidationError):
            FollowupItem.model_validate(_fields(summary=summary))

    def test_the_summary_bound_applies_after_collapsing(self) -> None:
        at_bound = 'x' * (MAX_SUMMARY_CHARS - 2) + '\n\n\n a'
        assert len(FollowupItem.model_validate(_fields(summary=at_bound)).summary) == MAX_SUMMARY_CHARS
        with pytest.raises(ValidationError):
            FollowupItem.model_validate(_fields(summary='x' * (MAX_SUMMARY_CHARS + 1)))

    def test_files_are_stored_sorted_and_unique(self) -> None:
        item = FollowupItem.model_validate(_fields(files=['pkg/b.py', 'pkg/a.py', 'pkg/b.py']))
        assert item.files == ('pkg/a.py', 'pkg/b.py')

    @pytest.mark.parametrize('files', [[], [''], ['pkg/a.py', '   ']])
    def test_empty_files_or_a_blank_entry_are_refused(self, files: list[str]) -> None:
        with pytest.raises(ValidationError):
            FollowupItem.model_validate(_fields(files=files))

    @pytest.mark.parametrize('field', ['key', 'origin_task'])
    def test_a_blank_identity_field_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError):
            FollowupItem.model_validate(_fields(**{field: '  '}))

    def test_origin_escalation_id_defaults_to_none(self) -> None:
        assert FollowupItem.model_validate(_fields()).origin_escalation_id is None


class TestItemKey:
    _BASE = {'origin_task': '8414', 'files': ['pkg/a.py', 'pkg/b.py'], 'summary': 'The loop hides it'}

    def test_it_is_sixteen_lowercase_hex_chars_and_deterministic(self) -> None:
        key = item_key(**self._BASE)
        assert _HEX16.fullmatch(key)
        assert item_key(**self._BASE) == key

    @pytest.mark.parametrize(
        'variant',
        [
            pytest.param({'files': ['pkg/b.py', 'pkg/a.py']}, id='file-order'),
            pytest.param({'files': ['pkg/a.py', 'pkg/b.py', 'pkg/a.py']}, id='duplicate-file'),
            pytest.param({'summary': 'the LOOP hides it'}, id='summary-case'),
            pytest.param({'summary': '  The loop\n  hides\tit '}, id='summary-whitespace'),
        ],
    )
    def test_presentation_differences_share_one_key(self, variant: dict[str, object]) -> None:
        assert item_key(**{**self._BASE, **variant}) == item_key(**self._BASE)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        'variant',
        [
            pytest.param({'origin_task': '8415'}, id='origin-task'),
            pytest.param({'files': ['pkg/a.py', 'pkg/c.py']}, id='a-file'),
            pytest.param({'files': ['pkg/a.py']}, id='fewer-files'),
            pytest.param({'summary': 'The loop shows it'}, id='summary-words'),
        ],
    )
    def test_a_substantive_difference_changes_the_key(self, variant: dict[str, object]) -> None:
        assert item_key(**{**self._BASE, **variant}) != item_key(**self._BASE)  # type: ignore[arg-type]


class TestObservedItem:
    def test_the_key_is_derived_from_the_stored_fields(self) -> None:
        item = observed_item(
            summary=' The loop\nhides it ', files=['pkg/b.py', 'pkg/a.py'], origin_task='8414',
        )
        assert item.summary == 'The loop hides it'
        assert item.files == ('pkg/a.py', 'pkg/b.py')
        assert item.key == item_key(origin_task='8414', files=item.files, summary=item.summary)
        assert item.origin_escalation_id is None

    def test_presentation_differences_share_one_key(self) -> None:
        first = observed_item(summary='The loop hides it', files=['pkg/a.py', 'pkg/b.py'], origin_task='1')
        second = observed_item(summary='The  loop\nhides it', files=['pkg/b.py', 'pkg/a.py'], origin_task='1')
        assert first.key == second.key

    def test_origin_escalation_id_is_carried(self) -> None:
        item = observed_item(
            summary='s', files=['pkg/a.py'], origin_task='1', origin_escalation_id='agent-followup-1',
        )
        assert item.origin_escalation_id == 'agent-followup-1'

    @pytest.mark.parametrize(
        ('summary', 'files'), [('  ', ['pkg/a.py']), ('s', []), ('s', ['  '])],
    )
    def test_an_invalid_observation_is_refused(self, summary: str, files: list[str]) -> None:
        with pytest.raises(ValidationError):
            observed_item(summary=summary, files=files, origin_task='1')


class TestRequireUniqueItemKeys:
    def test_unique_keys_pass(self) -> None:
        require_unique_item_keys([_item('a'), _item('b')])

    def test_repeated_keys_are_named_once_each(self) -> None:
        with pytest.raises(DuplicateFollowupItemKey) as caught:
            require_unique_item_keys([_item('a'), _item('b'), _item('a'), _item('c'), _item('b'), _item('a')])

        assert isinstance(caught.value, FollowupBundleInvalid)
        assert caught.value.reason is BundleViolation.DUPLICATE_KEY
        assert caught.value.keys == ('a', 'b')
        message = str(caught.value)
        assert message.count("'a'") == 1
        assert message.count("'b'") == 1


class TestMergeItems:
    def test_base_order_then_new_extra_items(self) -> None:
        merged = merge_items([_item('a'), _item('b')], [_item('c'), _item('b'), _item('d')])
        assert [item.key for item in merged] == ['a', 'b', 'c', 'd']

    def test_a_key_in_both_keeps_the_base_item(self) -> None:
        base = _item('a', summary='from base')
        merged = merge_items([base], [_item('a', summary='from extra')])
        assert merged == (base,)

    def test_merging_nothing_is_the_base(self) -> None:
        base = [_item('a'), _item('b')]
        assert merge_items(base, ()) == tuple(base)

    def test_merging_the_same_extra_twice_adds_nothing(self) -> None:
        once = merge_items([_item('a')], [_item('b'), _item('c')])
        assert merge_items(once, [_item('b'), _item('c')]) == once

    @pytest.mark.parametrize(
        ('base', 'extra'),
        [
            pytest.param(['a', 'a'], ['b'], id='base-repeats'),
            pytest.param(['a'], ['b', 'b'], id='extra-repeats'),
        ],
    )
    def test_a_repeating_input_is_refused(self, base: list[str], extra: list[str]) -> None:
        with pytest.raises(DuplicateFollowupItemKey):
            merge_items([_item(key) for key in base], [_item(key) for key in extra])


class TestLegacyItem:
    _META = {
        'source': 'agent-followup',
        'spawned_from': 8000,
        'escalation_id': 'agent-followup-8000',
        'suggestion_hash': 'abc123',
    }

    def test_a_pre_bundle_single_maps_to_one_item(self) -> None:
        item = legacy_item(title='Fix  the\nretry loop', files=['pkg/b.py', 'pkg/a.py'], metadata=self._META)
        assert item == FollowupItem(
            key='abc123',
            summary='Fix the retry loop',
            files=('pkg/a.py', 'pkg/b.py'),
            origin_task='8000',
            origin_escalation_id='agent-followup-8000',
        )

    def test_a_long_title_is_clipped_to_the_summary_bound(self) -> None:
        item = legacy_item(title='word ' * 200, files=['pkg/a.py'], metadata=self._META)
        assert item is not None
        assert len(item.summary) == MAX_SUMMARY_CHARS

    @pytest.mark.parametrize('spawned_from', [None, '', '  '])
    def test_an_unparented_single_has_the_unparented_origin(self, spawned_from: object) -> None:
        metadata = {**self._META, 'spawned_from': spawned_from}
        metadata.pop('escalation_id')
        item = legacy_item(title='t', files=['pkg/a.py'], metadata=metadata)
        assert item is not None
        assert (item.origin_task, item.origin_escalation_id) == (UNPARENTED_ORIGIN, None)

    @pytest.mark.parametrize(
        ('files', 'overrides'),
        [
            pytest.param(['pkg/a.py'], {'suggestion_hash': None}, id='hash-absent'),
            pytest.param(['pkg/a.py'], {'suggestion_hash': 42}, id='hash-not-str'),
            pytest.param(['pkg/a.py'], {'suggestion_hash': '  '}, id='hash-blank'),
            pytest.param([], {}, id='no-files'),
            pytest.param(['pkg/a.py'], {FOLLOWUP_ITEMS_METADATA_KEY: []}, id='already-a-bundle'),
        ],
    )
    def test_it_is_none_when_the_row_is_not_a_legacy_single(
        self, files: list[str], overrides: dict[str, object],
    ) -> None:
        metadata = {key: value for key, value in {**self._META, **overrides}.items() if value is not None}
        assert legacy_item(title='t', files=files, metadata=metadata) is None


class TestMetadataRegistration:
    def test_a_list_of_items_parses_to_typed_items(self) -> None:
        first = observed_item(summary='one', files=['pkg/a.py'], origin_task='1')
        second = observed_item(summary='two', files=['pkg/b.py'], origin_task='1')
        model, warnings = parse_metadata(
            {FOLLOWUP_ITEMS_METADATA_KEY: [first.model_dump(mode='json'), second.model_dump(mode='json')]},
            direction='write',
            enforce=True,
        )
        assert warnings == []
        assert model.model_extra is not None
        assert model.model_extra[FOLLOWUP_ITEMS_METADATA_KEY] == [first, second]

    def test_a_single_mapping_is_the_wrong_cardinality(self) -> None:
        blob = {FOLLOWUP_ITEMS_METADATA_KEY: _fields()}
        _model, warnings = parse_metadata(blob, direction='read')
        assert [(w.field, w.code) for w in warnings] == [(FOLLOWUP_ITEMS_METADATA_KEY, 'wrong_cardinality')]
        with pytest.raises(TypeError):
            parse_metadata(blob, direction='write', enforce=True)

    def test_an_element_without_files_is_an_invalid_submodel(self) -> None:
        element = _fields()
        element.pop('files')
        blob = {FOLLOWUP_ITEMS_METADATA_KEY: [element]}
        _model, warnings = parse_metadata(blob, direction='read')
        assert [(w.field, w.code) for w in warnings] == [(FOLLOWUP_ITEMS_METADATA_KEY, 'invalid_submodel')]
        with pytest.raises(ValidationError):
            parse_metadata(blob, direction='write', enforce=True)


def test_the_contract_constants() -> None:
    assert FOLLOWUP_ITEMS_METADATA_KEY == 'followup_items'
    assert MAX_FOLLOWUP_ITEMS == 5
