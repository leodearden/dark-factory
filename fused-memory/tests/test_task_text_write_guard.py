"""Tests for the reconciliation task-text write guard (task 4650).

Two symptom instances motivate it. Task 3708: a Stage-2 cycle counted a
``details`` amendment as applied while the stored text still held the old
paragraph, because nothing read the field back. Task 4644: two writers
composed full-field ``details`` rewrites against the same read, and the later
one silently clobbered the earlier one, because nothing compared the base the
writer composed against with the text it replaced.
"""

from __future__ import annotations

from typing import Any

from shared.mcp_envelope import write_failure_reason
from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.middleware.task_text_write_guard import (
    check_text_base,
    plan_text_write,
    text_fp_of,
)

EMPTY_TEXT_FP = task_text_fingerprint(None, None)


def _forwarded_kwargs(**overrides: Any) -> dict[str, Any]:
    """The kwargs shape ``server/tools.py::update_task`` forwards to the
    interceptor: every text key present, None when unset."""
    kwargs: dict[str, Any] = {
        'prompt': None,
        'metadata': None,
        'append': None,
        'metadata_mode': None,
        'tag': None,
        'title': None,
        'description': None,
        'details': None,
        'priority': None,
        'status': None,
        'dependencies': None,
    }
    kwargs.update(overrides)
    return kwargs


class TestTextFpOf:
    def test_delegates_to_the_shared_fingerprint(self):
        task = {'description': 'A short summary.', 'details': 'Long body\n\nwith paragraphs.'}

        assert text_fp_of(task) == task_text_fingerprint(task['description'], task['details'])

    def test_absent_or_none_text_is_the_empty_text_fingerprint(self):
        assert text_fp_of({'id': '1'}) == EMPTY_TEXT_FP
        assert text_fp_of({'description': None, 'details': None}) == EMPTY_TEXT_FP

    def test_a_non_mapping_is_the_empty_text_fingerprint(self):
        assert text_fp_of(None) == EMPTY_TEXT_FP
        assert text_fp_of('not a task') == EMPTY_TEXT_FP


class TestPlanTextWrite:
    def test_no_text_kwargs_writes_no_text(self):
        write = plan_text_write(_forwarded_kwargs(metadata='{"a": 1}'))

        assert write.writes_text is False
        assert write.replaces_text is False
        assert write.fields == ()

    def test_details_replace(self):
        write = plan_text_write(_forwarded_kwargs(details='x'))

        assert write.details == 'x'
        assert write.writes_text is True
        assert write.replaces_text is True
        assert write.fields == ('details',)

    def test_details_append_writes_but_does_not_replace(self):
        write = plan_text_write(_forwarded_kwargs(details='x', append=True))

        assert write.writes_text is True
        assert write.replaces_text is False
        assert write.details_appends is True

    def test_prompt_is_the_details_payload_when_details_is_unset(self):
        write = plan_text_write(_forwarded_kwargs(prompt='x'))

        assert write.details == 'x'
        assert write.replaces_text is True
        assert write.fields == ('details',)

    def test_details_wins_over_prompt(self):
        write = plan_text_write(_forwarded_kwargs(details='from-details', prompt='from-prompt'))

        assert write.details == 'from-details'

    def test_description_always_replaces(self):
        assert plan_text_write(_forwarded_kwargs(description='x')).replaces_text is True
        assert plan_text_write(
            _forwarded_kwargs(description='x', append=True),
        ).replaces_text is True

    def test_fields_are_listed_description_first(self):
        write = plan_text_write(_forwarded_kwargs(details='d', description='s'))

        assert write.fields == ('description', 'details')

    def test_absent_keys_read_as_unset(self):
        assert plan_text_write({}).writes_text is False


BEFORE = {'id': '4644', 'description': 'Current summary.', 'details': 'Current body.'}
RECON_AGENT = 'recon-stage-task_knowledge_sync'


def _check(*, expected_text_fp: str | None, fp_required: bool = True, before=BEFORE, **kwargs):
    return check_text_base(
        before=before,
        write=plan_text_write(_forwarded_kwargs(**kwargs)),
        expected_text_fp=expected_text_fp,
        fp_required=fp_required,
        task_id='4644',
        agent_id=RECON_AGENT,
    )


class TestCheckTextBase:
    def test_a_metadata_only_write_needs_no_token(self):
        assert _check(expected_text_fp=None, metadata='{"a": 1}') is None

    def test_a_matching_token_admits_a_replace(self):
        assert _check(expected_text_fp=text_fp_of(BEFORE), details='New body.') is None

    def test_a_stale_token_is_rejected_without_echoing_the_current_fingerprint(self):
        rejection = _check(expected_text_fp='tfp:stale', details='New body.')

        assert rejection is not None
        error = rejection.to_error_dict()
        assert error['error_type'] == 'TaskTextBaseStale'
        assert error['fields'] == ['details']
        assert error['expected_text_fp'] == 'tfp:stale'
        assert 'get_task' in error['hint']
        assert 'task_text_fp' in error['hint']
        assert text_fp_of(BEFORE) not in repr(error)

    def test_a_recon_replace_without_a_token_is_rejected(self):
        rejection = _check(expected_text_fp=None, details='New body.')

        assert rejection is not None
        assert rejection.to_error_dict()['error_type'] == 'ReconTaskTextBaseRequired'

    def test_a_token_is_optional_when_not_required(self):
        assert _check(expected_text_fp=None, fp_required=False, details='New body.') is None

    def test_a_details_append_needs_no_token(self):
        assert _check(expected_text_fp=None, details='More.', append=True) is None

    def test_a_description_write_needs_a_token_even_under_append(self):
        assert _check(expected_text_fp=None, description='New summary.') is not None
        assert _check(expected_text_fp=None, description='New summary.', append=True) is not None

    def test_a_prompt_replace_needs_a_token(self):
        assert _check(expected_text_fp=None, prompt='New body.') is not None

    def test_a_supplied_token_is_honoured_on_an_append(self):
        rejection = _check(expected_text_fp='tfp:stale', details='More.', append=True)

        assert rejection is not None
        assert rejection.to_error_dict()['error_type'] == 'TaskTextBaseStale'

    def test_a_task_with_no_text_accepts_the_empty_text_token(self):
        assert _check(
            expected_text_fp=text_fp_of({}), before={'id': '4644'}, details='First body.',
        ) is None

    def test_every_rejection_reads_as_a_failed_write(self):
        rejections = [
            _check(expected_text_fp='tfp:stale', details='New body.'),
            _check(expected_text_fp=None, details='New body.'),
        ]

        for rejection in rejections:
            assert rejection is not None
            assert write_failure_reason(rejection.to_error_dict()) is not None
