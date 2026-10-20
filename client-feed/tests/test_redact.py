"""Removing absolute paths and home paths from free text a client will see."""

import pytest

from client_feed.redact import PATH_PLACEHOLDER, redact_paths

REDACTED = [
    ('/home/leo/x', '[path]'),
    ('Fix (see /etc/passwd).', 'Fix (see [path]).'),
    ('read ~/src/scoping-intake/notes, then', 'read [path], then'),
    ('file:///home/leo/x', '[path]'),
    ("root='/home/leo/src/reify' x", "root='[path]' x"),
    ('/tmp/x', '[path]'),
    ('POST /api/v2/x proxy', 'POST [path] proxy'),
]

KEPT = [
    'search/delete',
    'ports 8000/8001',
    'dashboard/reconciliation.py',
    'a / b',
    'and/or',
    'https://example.com/a',
    'Run /review-briefing now',
    '/home',
    '/opt/',
    '/',
    '~/',
]


def test_the_placeholder_is_spelled_once() -> None:
    assert PATH_PLACEHOLDER == '[path]'


@pytest.mark.parametrize(('text', 'expected'), REDACTED)
def test_a_path_token_becomes_the_placeholder(text: str, expected: str) -> None:
    assert redact_paths(text) == expected


@pytest.mark.parametrize('text', KEPT)
def test_a_token_that_is_not_a_path_is_kept(text: str) -> None:
    assert redact_paths(text) == text


@pytest.mark.parametrize('text', [text for text, _ in REDACTED] + KEPT)
def test_redaction_is_idempotent(text: str) -> None:
    once = redact_paths(text)
    assert redact_paths(once) == once


def test_text_with_no_path_comes_back_identical() -> None:
    text = 'Add a retry to the merge lane (task 42): keep 3 attempts, then stop.'
    assert redact_paths(text) == text
