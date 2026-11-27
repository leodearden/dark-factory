"""Tests for ``orchestrator.auto_eval_redo.redo_submit_text``: the task text an auto-eval redo sibling is filed with."""

from __future__ import annotations

from os.path import commonprefix

import pytest
from shared.elision import elision_marker

from orchestrator.auto_eval_redo import redo_submit_text


def _submit_text(original: dict) -> dict[str, str]:
    return redo_submit_text(
        original,
        original_id='orig-task',
        block_phase='verify',
        optimistic_path='simple_task',
        new_branch='orig-task-skip-attempt',
    )


def test_a_long_original_title_is_clipped_at_a_word_with_a_counted_marker():
    long = ('Fix the widget renderer so it ' * 12).strip()

    submit_text = _submit_text({'title': long, 'description': 'd'})

    title = submit_text['title']
    full = f'[auto-eval redo] {long}'
    head = commonprefix([title, full])
    assert len(title) <= 200
    assert title == head + elision_marker(len(full) - len(head))
    assert full[len(head)].isspace()
    assert long in submit_text['description']


def test_the_description_names_the_original_title_and_the_blocked_run():
    submit_text = _submit_text({'title': 'Document foo', 'description': 'd'})

    assert submit_text['title'] == '[auto-eval redo] Document foo'
    description = submit_text['description']
    assert 'Original title: Document foo' in description
    assert 'orig-task' in description
    assert 'verify' in description
    assert 'simple_task' in description
    assert 'orig-task-skip-attempt' in description


def test_the_original_description_is_carried_verbatim_and_uncapped():
    original_description = '\n'.join(f'spec line {i}: do the thing' for i in range(400))

    submit_text = _submit_text({'title': 'Document foo', 'description': original_description})

    assert original_description in submit_text['description']


def test_the_original_details_are_carried_verbatim():
    details = '  Step 1: read the spec.\n\nStep 2: implement it.\n'

    submit_text = _submit_text({'title': 'Document foo', 'description': 'd', 'details': details})

    assert submit_text['details'] == details


@pytest.mark.parametrize('original', [
    {'title': 'Document foo', 'description': 'd'},
    {'title': 'Document foo', 'description': 'd', 'details': None},
    {'title': 'Document foo', 'description': 'd', 'details': ' \n\t '},
])
def test_an_original_without_details_sends_no_details_argument(original: dict):
    assert 'details' not in _submit_text(original)


@pytest.mark.parametrize('original', [
    {'title': 'Document foo', 'description': ''},
    {'title': 'Document foo'},
])
def test_an_original_without_a_description_still_names_its_title(original: dict):
    description = _submit_text(original)['description']

    assert 'Original title: Document foo' in description
    assert 'Original description' not in description
