"""The probe's header names the ``claude`` binary its runs spawn, and that binary's version.

Driven through the public ``CLAUDE_BINARY`` / ``PATH`` seams with the fake CLI
from ``shared.testing_fake_claude``; nothing in the script is patched.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.testing_fake_claude import install_fake_claude_cli

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'probe_schema_max_turns.py'

FAKE_VERSION = '9.9.9 (Claude Code)'


def _probe() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='probe_schema_max_turns')


def test_claude_binary_override_is_the_binary_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake_claude_cli(tmp_path, [FAKE_VERSION + '\n'])
    monkeypatch.setenv('CLAUDE_BINARY', str(fake.executable))

    cli = _probe()._probed_cli()

    assert (cli.binary, cli.version) == (str(fake.executable), FAKE_VERSION)
    assert fake.recorded_argv() == [['--version']]


def test_bare_claude_is_reported_as_the_path_it_resolves_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake_claude_cli(tmp_path, [FAKE_VERSION])
    monkeypatch.delenv('CLAUDE_BINARY', raising=False)
    monkeypatch.setenv('PATH', str(fake.executable.parent))

    cli = _probe()._probed_cli()

    assert (cli.binary, cli.version) == (str(fake.executable), FAKE_VERSION)


def test_unrunnable_binary_is_still_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / 'no-such-claude'
    monkeypatch.setenv('CLAUDE_BINARY', str(missing))

    cli = _probe()._probed_cli()

    assert (cli.binary, cli.version) == (str(missing), '(claude CLI not runnable)')
