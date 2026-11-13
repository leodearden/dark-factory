"""Guard tests: an ambient CLAUDE_BINARY never reaches a test body.

``shared.cli_invoke::claude_binary_spec`` reads CLAUDE_BINARY ahead of PATH, so
a test that hides or fakes ``claude`` by editing PATH would still spawn the
binary an operator's environment names. ``df_pytest_isolation._df_claude_binary_unset``
scrubs it per test; these tests drive that fixture end to end in a nested
session, pin its structure, and pin its variable name to the one production reads.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from nested_pytest_session import binding_conftest, run_nested_pytest
from shared.cli_invoke import resolve_claude_binary

import df_pytest_isolation
from df_pytest_isolation import fixture_marker

_AMBIENT = '/nonexistent/ambient/claude'

_SCRUBBED_TESTS = '''\
import os


def test_the_ambient_value_is_gone():
    assert 'CLAUDE_BINARY' not in os.environ


def test_a_deliberate_value_still_reaches_the_test(monkeypatch):
    monkeypatch.setenv('CLAUDE_BINARY', '/deliberate')
    assert os.environ['CLAUDE_BINARY'] == '/deliberate'
'''

_UNSCRUBBED_TEST = f'''\
import os


def test_the_ambient_value_is_visible():
    assert os.environ['CLAUDE_BINARY'] == {_AMBIENT!r}
'''


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def test_an_ambient_claude_binary_never_reaches_a_test_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_BINARY', _AMBIENT)

    result = run_nested_pytest(
        tmp_path / 'bound',
        {
            'conftest.py': binding_conftest('_df_claude_binary_unset'),
            'test_probe.py': _SCRUBBED_TESTS,
        },
    )

    assert result.returncode == 0, _output(result)
    assert '2 passed' in _output(result), _output(result)


def test_without_the_binding_the_ambient_value_does_reach_a_test_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CLAUDE_BINARY', _AMBIENT)

    result = run_nested_pytest(tmp_path / 'unbound', {'test_probe.py': _UNSCRUBBED_TEST})

    assert result.returncode == 0, _output(result)
    assert '1 passed' in _output(result), _output(result)


def test_the_scrub_is_autouse_and_function_scoped() -> None:
    marker = fixture_marker(df_pytest_isolation._df_claude_binary_unset)

    assert marker.autouse is True, 'the scrub must be autouse — nothing requests it'
    assert marker.scope == 'function', (
        f'scope is {marker.scope!r}: a leak can arrive mid-session, so the scrub '
        'must run before every test'
    )


def test_the_scrubbed_name_is_the_one_the_resolver_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exe = tmp_path / 'claude-override'
    exe.write_text('#!/bin/sh\nexit 0\n')
    exe.chmod(0o755)
    monkeypatch.setenv(df_pytest_isolation.CLAUDE_BINARY_ENV, str(exe))

    assert resolve_claude_binary() == str(exe)
