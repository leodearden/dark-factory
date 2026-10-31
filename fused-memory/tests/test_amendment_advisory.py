"""The write-boundary amendment advisory decision (task 4034, PRD §9 C7)."""

from __future__ import annotations

import json

import pytest
from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.middleware.amendment_advisory import amendment_advisory, text_fields_written

ORIG = 'Add a retry loop around the flush.'
NEW = 'Add a bounded retry loop around the flush, capped at three attempts.'
DETAILS = 'Touch only writer.py.'
AUTHORED_AT = '2026-10-01T00:00:00+00:00'
STAMP = task_text_fingerprint(ORIG, DETAILS)


def written(
    *,
    description: str = NEW,
    details: str = DETAILS,
    status: str = 'pending',
    metadata: object = None,
) -> dict[str, object]:
    if metadata is None:
        metadata = {'plan_text_fp': STAMP, 'plan_authored_at': AUTHORED_AT}
    return {
        'status': status,
        'description': description,
        'details': details,
        'metadata': metadata,
    }


class TestTextFieldsWritten:
    def test_returns_supplied_text_fields_in_field_order(self) -> None:
        write = {'details': 'd', 'title': 't', 'prompt': 'p', 'description': 'x'}
        assert text_fields_written(write) == ('prompt', 'title', 'description', 'details')

    def test_a_key_present_with_none_is_not_written(self) -> None:
        write = {'prompt': None, 'title': None, 'description': None, 'details': None}
        assert text_fields_written(write) == ()

    def test_an_empty_string_is_written(self) -> None:
        assert text_fields_written({'description': '', 'details': None}) == ('description',)

    def test_non_text_keys_are_ignored(self) -> None:
        write = {'metadata': {'plan_text_fp': STAMP}, 'priority': 'high'}
        assert text_fields_written(write) == ()

    def test_a_rotation_rewrite_adds_description(self) -> None:
        assert text_fields_written({'title': 't'}, description_rewritten=True) == (
            'title',
            'description',
        )

    def test_a_rotation_rewrite_adds_description_only_once(self) -> None:
        write = {'description': 'x'}
        assert text_fields_written(write, description_rewritten=True) == ('description',)


class TestAmendmentAdvisory:
    def test_a_drifted_description_write_is_advised(self) -> None:
        assert amendment_advisory('42', written(), ('description',)) == {
            'task_id': '42',
            'plan_authored_at': AUTHORED_AT,
            'fields_changed': ['description'],
            'will_replan': True,
            'plan_text_fp_before': STAMP,
            'task_text_fp_after': task_text_fingerprint(NEW, DETAILS),
        }

    def test_text_that_still_matches_the_stamp_is_not_advised(self) -> None:
        assert amendment_advisory('42', written(description=ORIG), ('description',)) is None

    def test_a_whitespace_only_reflow_is_not_advised(self) -> None:
        reflowed = '  Add a retry\n loop around   the flush.  '
        assert amendment_advisory('42', written(description=reflowed), ('description',)) is None

    @pytest.mark.parametrize(
        'metadata',
        [
            {'plan_authored_at': AUTHORED_AT},
            {'plan_text_fp': '', 'plan_authored_at': AUTHORED_AT},
            {'plan_text_fp': 7, 'plan_authored_at': AUTHORED_AT},
        ],
        ids=['absent', 'empty', 'non-string'],
    )
    def test_an_unstamped_task_is_not_advised(self, metadata: dict[str, object]) -> None:
        assert amendment_advisory('42', written(metadata=metadata), ('description',)) is None

    def test_a_write_that_touched_no_text_does_not_re_report_drift(self) -> None:
        assert amendment_advisory('42', written(), ()) is None

    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_a_terminal_task_will_not_replan(self, status: str) -> None:
        advisory = amendment_advisory('42', written(status=status), ('description',))
        assert advisory is not None
        assert advisory['will_replan'] is False

    @pytest.mark.parametrize('status', ['in-progress', 'blocked', 'deferred'])
    def test_a_dispatchable_task_will_replan(self, status: str) -> None:
        advisory = amendment_advisory('42', written(status=status), ('description',))
        assert advisory is not None
        assert advisory['will_replan'] is True

    def test_metadata_as_a_json_object_string_is_read(self) -> None:
        metadata = json.dumps({'plan_text_fp': STAMP, 'plan_authored_at': AUTHORED_AT})
        advisory = amendment_advisory('42', written(metadata=metadata), ('description',))
        assert advisory is not None
        assert advisory['plan_text_fp_before'] == STAMP

    @pytest.mark.parametrize('metadata', ['not json', [STAMP]], ids=['not-json', 'list'])
    def test_unreadable_metadata_is_not_advised(self, metadata: object) -> None:
        assert amendment_advisory('42', written(metadata=metadata), ('description',)) is None

    @pytest.mark.parametrize('written_task', [None, 'a row', ['a', 'row']])
    def test_a_missing_or_non_mapping_row_is_not_advised(self, written_task: object) -> None:
        assert amendment_advisory('42', written_task, ('description',)) is None

    def test_a_non_string_authored_at_surfaces_as_none(self) -> None:
        metadata = {'plan_text_fp': STAMP, 'plan_authored_at': 20261001}
        advisory = amendment_advisory('42', written(metadata=metadata), ('description',))
        assert advisory is not None
        assert advisory['plan_authored_at'] is None

    def test_a_title_only_write_on_a_drifted_task_is_advised(self) -> None:
        advisory = amendment_advisory('42', written(), ('title',))
        assert advisory is not None
        assert advisory['fields_changed'] == ['title']
