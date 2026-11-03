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

from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.middleware.task_text_write_guard import plan_text_write, text_fp_of

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
