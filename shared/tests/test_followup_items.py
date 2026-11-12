"""``metadata.followup_items``: the item record, its key, uniqueness, the one merge and the legacy mapping."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    FOLLOWUP_ITEMS_REFUSED_ERROR,
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
    parse_item,
    read_items,
    require_unique_item_keys,
    task_items,
    with_derived_bundle_keys,
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
            item.summary = 'changed'

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


@dataclass(frozen=True)
class _KeyArgs:
    origin_task: str = '8414'
    files: tuple[str, ...] = ('pkg/a.py', 'pkg/b.py')
    summary: str = 'The loop hides it'

    def key(self) -> str:
        return item_key(origin_task=self.origin_task, files=self.files, summary=self.summary)


class TestItemKey:
    def test_it_is_sixteen_lowercase_hex_chars_and_deterministic(self) -> None:
        key = _KeyArgs().key()
        assert _HEX16.fullmatch(key)
        assert _KeyArgs().key() == key

    @pytest.mark.parametrize(
        'variant',
        [
            pytest.param(_KeyArgs(files=('pkg/b.py', 'pkg/a.py')), id='file-order'),
            pytest.param(_KeyArgs(files=('pkg/a.py', 'pkg/b.py', 'pkg/a.py')), id='duplicate-file'),
            pytest.param(_KeyArgs(summary='the LOOP hides it'), id='summary-case'),
            pytest.param(_KeyArgs(summary='  The loop\n  hides\tit '), id='summary-whitespace'),
        ],
    )
    def test_presentation_differences_share_one_key(self, variant: _KeyArgs) -> None:
        assert variant.key() == _KeyArgs().key()

    @pytest.mark.parametrize(
        'variant',
        [
            pytest.param(_KeyArgs(origin_task='8415'), id='origin-task'),
            pytest.param(_KeyArgs(files=('pkg/a.py', 'pkg/c.py')), id='a-file'),
            pytest.param(_KeyArgs(files=('pkg/a.py',)), id='fewer-files'),
            pytest.param(_KeyArgs(summary='The loop shows it'), id='summary-words'),
        ],
    )
    def test_a_substantive_difference_changes_the_key(self, variant: _KeyArgs) -> None:
        assert variant.key() != _KeyArgs().key()

    def test_one_str_of_files_is_refused_rather_than_keyed_by_character(self) -> None:
        with pytest.raises(TypeError, match='files'):
            item_key(origin_task='8414', files='pkg/a.py', summary='The loop hides it')


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
        item = legacy_item(title='x' * (MAX_SUMMARY_CHARS + 50), files=['pkg/a.py'], metadata=self._META)
        assert item is not None
        assert item.summary == 'x' * MAX_SUMMARY_CHARS

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

    @pytest.mark.parametrize(
        ('title', 'files'),
        [
            pytest.param(' \n ', ['pkg/a.py'], id='blank-title'),
            pytest.param('t', ['pkg/a.py', '  '], id='blank-file'),
            pytest.param('t', 'pkg/a.py', id='one-str-of-files'),
        ],
    )
    def test_it_is_none_when_no_item_can_hold_the_row(self, title: str, files: list[str]) -> None:
        assert legacy_item(title=title, files=files, metadata=self._META) is None


class TestParseItem:
    def test_a_valid_element_is_its_item(self) -> None:
        item = observed_item(summary='one', files=['pkg/a.py'], origin_task='1')
        assert parse_item(item.model_dump(mode='json')) == item

    @pytest.mark.parametrize(
        'element',
        [
            pytest.param('one', id='not-a-mapping'),
            pytest.param(None, id='none'),
            pytest.param(_fields(title='t'), id='unknown-key'),
            pytest.param({name: value for name, value in _fields().items() if name != 'key'}, id='no-key'),
        ],
    )
    def test_an_invalid_element_is_none(self, element: object) -> None:
        assert parse_item(element) is None

    def test_it_reads_any_mapping_as_the_strict_reader_does(self) -> None:
        element = MappingProxyType(_fields())
        assert parse_item(element) == read_items({FOLLOWUP_ITEMS_METADATA_KEY: [element]})[0]


class TestTaskItems:
    _LEGACY = {
        'source': 'agent-followup',
        'spawned_from': '8000',
        'escalation_id': 'agent-followup-8000',
        'suggestion_hash': 'abc123',
    }

    def test_a_bundle_is_its_stored_items_in_order(self) -> None:
        first = observed_item(summary='second filed first', files=['pkg/b.py'], origin_task='1')
        second = observed_item(summary='first filed second', files=['pkg/a.py'], origin_task='1')
        metadata = {
            **self._LEGACY,
            FOLLOWUP_ITEMS_METADATA_KEY: [first.model_dump(mode='json'), second.model_dump(mode='json')],
        }
        items = task_items(title='t', files=['pkg/a.py', 'pkg/b.py'], metadata=metadata)
        assert items == (first, second)
        assert items == read_items(metadata)

    def test_a_pre_bundle_single_is_its_one_legacy_item(self) -> None:
        items = task_items(title='Fix the retry loop', files=['pkg/a.py'], metadata=self._LEGACY)
        legacy = legacy_item(title='Fix the retry loop', files=['pkg/a.py'], metadata=self._LEGACY)
        assert legacy is not None
        assert items == (legacy,)
        assert items[0].key == 'abc123'

    def test_a_row_that_is_neither_has_no_items(self) -> None:
        assert task_items(title='t', files=['pkg/a.py'], metadata={'source': 'steward'}) == ()

    def test_a_legacy_single_without_files_has_no_items(self) -> None:
        assert task_items(title='t', files=[], metadata=self._LEGACY) == ()

    @pytest.mark.parametrize(
        ('value', 'reason'),
        [
            pytest.param('not-a-list', BundleViolation.NOT_A_LIST, id='not-a-list'),
            pytest.param([{'summary': 'no key'}], BundleViolation.INVALID_ITEM, id='invalid-element'),
        ],
    )
    def test_a_malformed_stored_list_is_refused_as_read_items_refuses_it(
        self, value: object, reason: BundleViolation,
    ) -> None:
        metadata = {**self._LEGACY, FOLLOWUP_ITEMS_METADATA_KEY: value}
        with pytest.raises(FollowupBundleInvalid) as raised:
            task_items(title='t', files=['pkg/a.py'], metadata=metadata)
        assert raised.value.reason is reason


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
    assert FOLLOWUP_ITEMS_REFUSED_ERROR == 'followup_items_refused'
    assert MAX_FOLLOWUP_ITEMS == 5


# --- The submit-boundary derivation ------------------------------------------------


def _raw(summary: str, *files: str, **extra: object) -> dict[str, object]:
    return {'summary': summary, 'files': list(files), **extra}


def _bundle(*items: object, **top: object) -> dict[str, object]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': '8414',
        'files': ['pkg/a.py', 'pkg/b.py', 'pkg/c.py'],
        'x_note': 'kept',
        FOLLOWUP_ITEMS_METADATA_KEY: list(items) if items else [
            _raw('one', 'pkg/a.py'), _raw('two', 'pkg/b.py', 'pkg/c.py'),
        ],
        **top,
    }


def _stored(output: dict[str, object]) -> tuple[FollowupItem, ...]:
    return read_items(output)


class TestWithDerivedBundleKeys:
    def test_metadata_without_items_comes_back_as_an_equal_new_dict(self) -> None:
        metadata = {'source': 'steward', 'escalation_id': 'esc-1', 'suggestion_hash': 'h'}
        output = with_derived_bundle_keys(metadata)
        assert output == metadata
        assert output is not metadata

    @pytest.mark.parametrize('spawned_from', [8414, '8414'])
    def test_a_parented_bundle_derives_every_item_identity(self, spawned_from: object) -> None:
        output = with_derived_bundle_keys(_bundle(spawned_from=spawned_from))

        assert output['escalation_id'] == 'agent-followup-8414'
        for stored in _stored(output):
            assert stored.origin_task == '8414'
            assert stored.origin_escalation_id == 'agent-followup-8414'
            assert stored.key == item_key(origin_task='8414', files=stored.files, summary=stored.summary)

    def test_caller_supplied_identities_are_overwritten(self) -> None:
        forged = {'key': 'forged', 'origin_task': '1', 'origin_escalation_id': 'esc-forged'}
        honest = with_derived_bundle_keys(_bundle())
        output = with_derived_bundle_keys(_bundle(
            _raw('one', 'pkg/a.py', **forged),
            _raw('two', 'pkg/b.py', 'pkg/c.py', **forged),
            escalation_id='esc-forged',
            suggestion_hash='forged',
        ))
        assert output == honest

    def test_the_bundle_suggestion_hash_is_derived_over_the_item_keys(self) -> None:
        first, second = _raw('one', 'pkg/a.py'), _raw('two', 'pkg/b.py')
        output = with_derived_bundle_keys(_bundle(first, second))
        suggestion_hash = output['suggestion_hash']

        assert isinstance(suggestion_hash, str)
        assert _HEX16.fullmatch(suggestion_hash)
        assert with_derived_bundle_keys(_bundle(second, first))['suggestion_hash'] == suggestion_hash
        changed = with_derived_bundle_keys(_bundle(first, _raw('three', 'pkg/b.py')))
        assert changed['suggestion_hash'] != suggestion_hash
        assert suggestion_hash not in {stored.key for stored in _stored(output)}

    @pytest.mark.parametrize('spawned_from', [None, '', '   '])
    def test_an_unparented_bundle_has_no_top_level_escalation_id(self, spawned_from: object) -> None:
        metadata = _bundle(escalation_id='esc-forged')
        if spawned_from is None:
            metadata.pop('spawned_from')
        else:
            metadata['spawned_from'] = spawned_from

        output = with_derived_bundle_keys(metadata)

        assert 'escalation_id' not in output
        assert _HEX16.fullmatch(str(output['suggestion_hash']))
        for stored in _stored(output):
            assert (stored.origin_task, stored.origin_escalation_id) == (UNPARENTED_ORIGIN, None)

    def test_every_other_caller_key_is_preserved(self) -> None:
        metadata = _bundle()
        output = with_derived_bundle_keys(metadata)
        for key in ('source', 'spawn_context', 'spawned_from', 'files', 'x_note'):
            assert output[key] == metadata[key]

    def test_the_output_is_plain_json(self) -> None:
        output = with_derived_bundle_keys(_bundle())
        raw = output[FOLLOWUP_ITEMS_METADATA_KEY]
        assert isinstance(raw, list)
        assert all(isinstance(element, dict) and isinstance(element['files'], list) for element in raw)
        assert json.loads(json.dumps(output)) == output


def _refusal(metadata: dict[str, object]) -> FollowupBundleInvalid:
    with pytest.raises(FollowupBundleInvalid) as caught:
        with_derived_bundle_keys(metadata)
    return caught.value


class TestWithDerivedBundleKeysRefusals:
    @pytest.mark.parametrize('value', [_raw('one', 'pkg/a.py'), 'one', None])
    def test_a_non_list_value_is_not_a_list(self, value: object) -> None:
        metadata = _bundle()
        metadata[FOLLOWUP_ITEMS_METADATA_KEY] = value
        assert _refusal(metadata).reason is BundleViolation.NOT_A_LIST

    @pytest.mark.parametrize('count', [0, MAX_FOLLOWUP_ITEMS + 1])
    def test_a_count_outside_the_bound_is_refused(self, count: int) -> None:
        metadata = _bundle(files=[f'pkg/m{index}.py' for index in range(count)])
        metadata[FOLLOWUP_ITEMS_METADATA_KEY] = [
            _raw(f'observation {index}', f'pkg/m{index}.py') for index in range(count)
        ]
        refusal = _refusal(metadata)
        assert refusal.reason is BundleViolation.ITEM_COUNT
        assert str(count) in refusal.detail
        assert f'1..{MAX_FOLLOWUP_ITEMS}' in refusal.detail

    @pytest.mark.parametrize(
        'element',
        [
            pytest.param('one', id='not-a-mapping'),
            pytest.param(_raw('one', 'pkg/b.py', title='t'), id='unknown-key'),
            pytest.param(_raw('   ', 'pkg/b.py'), id='blank-summary'),
            pytest.param(_raw('one'), id='empty-files'),
        ],
    )
    def test_an_invalid_element_is_named_by_index(self, element: object) -> None:
        refusal = _refusal(_bundle(_raw('fine', 'pkg/a.py'), element))
        assert refusal.reason is BundleViolation.INVALID_ITEM
        assert refusal.index == 1

    def test_two_items_equal_after_normalisation_are_duplicates(self) -> None:
        refusal = _refusal(_bundle(_raw('One  thing', 'pkg/a.py'), _raw(' one thing', 'pkg/a.py', 'pkg/a.py')))
        assert isinstance(refusal, DuplicateFollowupItemKey)

    def test_an_item_file_outside_metadata_files_is_undeclared(self) -> None:
        refusal = _refusal(_bundle(
            _raw('one', 'pkg/a.py', 'pkg/z.py'), _raw('two', 'pkg/y.py'), files=['pkg/a.py'],
        ))
        assert refusal.reason is BundleViolation.UNDECLARED_FILES
        assert 'pkg/y.py' in refusal.detail
        assert 'pkg/z.py' in refusal.detail
        assert 'pkg/a.py' not in refusal.detail

    @pytest.mark.parametrize('files', [None, 'pkg/a.py'])
    def test_absent_or_non_list_metadata_files_declare_nothing(self, files: object) -> None:
        metadata = _bundle(_raw('one', 'pkg/a.py'))
        if files is None:
            metadata.pop('files')
        else:
            metadata['files'] = files
        assert _refusal(metadata).reason is BundleViolation.UNDECLARED_FILES

    def test_metadata_files_may_declare_more_than_the_items_name(self) -> None:
        output = with_derived_bundle_keys(_bundle(_raw('one', 'pkg/a.py'), files=['pkg/a.py', 'pkg/extra.py']))
        assert len(_stored(output)) == 1
