"""Behaviour of shared.task_text_fingerprint (task 4031, PRD C1)."""

import inspect

import pytest

import shared.task_text_fingerprint as module
from shared.task_text_fingerprint import (
    TaskTextDrift,
    task_text_drift,
    task_text_fingerprint,
)


def test_whitespace_only_change_keeps_the_fingerprint():
    assert task_text_fingerprint('fix the  bug\nnow', 'a\t b') == task_text_fingerprint(
        '  fix the bug now ', 'a b\n',
    )


def test_unicode_whitespace_collapses_too():
    assert task_text_fingerprint('a b', None) == task_text_fingerprint('a b', None)


def test_case_change_is_an_amendment():
    assert task_text_fingerprint('Foo', None) != task_text_fingerprint('foo', None)


def test_none_and_empty_are_the_same_text():
    assert task_text_fingerprint(None, None) == task_text_fingerprint('', '')


def test_fields_are_separated():
    assert task_text_fingerprint('a b', '') != task_text_fingerprint('a', 'b')


def test_shape_is_prefixed_and_fixed_length():
    fp = task_text_fingerprint('d', 'x')
    assert fp.startswith('tfp:')
    assert len(fp) == 36


def test_deterministic():
    assert task_text_fingerprint('d', 'x') == task_text_fingerprint('d', 'x')


def test_any_text_change_is_an_amendment():
    assert task_text_fingerprint('d', 'x') != task_text_fingerprint('d', 'x y')


@pytest.mark.parametrize('stamped', [None, '', 0, 42, ['tfp:x'], {'fp': 'tfp:x'}])
def test_missing_or_malformed_stamp_is_unstamped(stamped):
    assert task_text_drift(stamped, task_text_fingerprint('d', None)) is TaskTextDrift.UNSTAMPED


def test_equal_stamp_is_current():
    fp = task_text_fingerprint('d', None)
    assert task_text_drift(fp, fp) is TaskTextDrift.CURRENT


def test_unequal_stamp_is_amended():
    assert task_text_drift(
        task_text_fingerprint('items 1-3', None), task_text_fingerprint('items 1-4', None),
    ) is TaskTextDrift.AMENDED


def test_is_not_the_casefolding_recon_normalizer():
    assert '_normalize_content_description' in inspect.getsource(module)
    assert task_text_drift(
        task_text_fingerprint('Foo', None), task_text_fingerprint('foo', None),
    ) is TaskTextDrift.AMENDED
