"""Removing absolute paths and home paths from free text a client will see."""

import pytest

from client_feed.redact import redact_paths

REDACTED = [
    ('/home/leo/x', '[path]'),
    ('Fix (see /etc/passwd).', 'Fix (see [path]).'),
    ('read ~/src/scoping-intake/notes, then', 'read [path], then'),
    ('file:///home/leo/x', '[path]'),
    ("root='/home/leo/src/reify' x", "root='[path]' x"),
    ('/tmp/x', '[path]'),
    ('POST /api/v2/x proxy', 'POST [path] proxy'),
    ('root=/opt/acme/src x', 'root=[path] x'),
    ('--config=/srv/a/b', '--config=[path]'),
    ('PATH=/usr/local/bin:/usr/bin', 'PATH=[path]'),
    ('dir:/etc/passwd', 'dir:[path]'),
    ('a,/srv/secret/x', 'a,[path]'),
    ('HOME=~/notes/x', 'HOME=[path]'),
    ('https://example.com/home/x', '[path]'),
]

KEPT = [
    'search/delete',
    'ports 8000/8001',
    'dashboard/reconciliation.py',
    'a / b',
    'and/or',
    'https://example.com/a',
    'https://example.com/a/b/c',
    'http://localhost:8000/api/v2',
    'k=v/w',
    'ratio 3:2,1/2',
    'Run /review-briefing now',
    '/home',
    '/opt/',
    '/',
    '~/',
]


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
