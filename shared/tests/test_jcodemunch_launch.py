"""Tests for shared.jcodemunch_launch — the jcodemunch-mcp stdio launch contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from shared.jcodemunch_launch import (
    JCODEMUNCH_COMMAND,
    JCODEMUNCH_ENV,
    JCODEMUNCH_PYTHON,
    JCODEMUNCH_REQUIREMENT,
    jcodemunch_install_argv,
    jcodemunch_server_config,
)

_SHARED_SRC = Path(__file__).resolve().parents[1] / 'src'


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the contract module's CLI the way a shell consumer does: bare stdlib, `-S`."""
    return subprocess.run(
        [sys.executable, '-S', '-m', 'shared.jcodemunch_launch', *args],
        env={**os.environ, 'PYTHONPATH': str(_SHARED_SRC)},
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestJcodemunchLaunchContract:
    """The shared constants are the single source of truth for the launch contract."""

    def test_command_is_prebuilt_launcher_not_uvx(self):
        """The prebuilt launcher on PATH, not ``uvx`` — see the rationale at the constant."""
        assert JCODEMUNCH_COMMAND == 'jcodemunch-mcp'

    def test_env_contains_no_version_hint(self):
        """JCODEMUNCH_ENV silences the stderr version-drift note."""
        assert JCODEMUNCH_ENV['JCODEMUNCH_NO_VERSION_HINT'] == '1'

    def test_env_sets_git_root_identity_lever(self):
        """JCODEMUNCH_ENV pins the identity lever to the exact literal '0'.

        jcodemunch's bool env parser treats '0' as False (lever ON, selecting
        per-worktree ``local/<basename>-<sha1[:8]>`` identity) but '1'/'true'
        as True (lever OFF, silently reverting to the worktree-collapsing
        git-root default) — so this must assert the literal value, not just
        key presence.
        """
        assert JCODEMUNCH_ENV['JCODEMUNCH_GIT_ROOT_IDENTITY'] == '0'

    def test_env_is_exactly_the_two_keys(self):
        """No third key may be added at one site without this contract noticing.

        The env block is small and load-bearing; an extra key silently added
        by one consumer would diverge the contract it exists to unify.
        """
        assert set(JCODEMUNCH_ENV) == {
            'JCODEMUNCH_NO_VERSION_HINT',
            'JCODEMUNCH_GIT_ROOT_IDENTITY',
        }


class TestJcodemunchServerConfig:
    """The contract rendered as one MCP server config, for Python and shell consumers alike."""

    def test_server_config_is_the_launch_contract(self):
        assert jcodemunch_server_config() == {
            'command': JCODEMUNCH_COMMAND,
            'env': JCODEMUNCH_ENV,
        }

    def test_server_config_env_is_a_fresh_copy(self):
        """A consumer mutating its copy must not corrupt the process-wide contract."""
        cfg = jcodemunch_server_config()
        assert cfg['env'] is not JCODEMUNCH_ENV

        cfg['env']['X'] = 'y'

        assert 'X' not in JCODEMUNCH_ENV
        assert 'X' not in jcodemunch_server_config()['env']

    def test_module_prints_the_server_config_as_json_on_the_bare_stdlib(self):
        """`-S` drops site-packages: a plain `python3` outside the venv can render it.

        That is how scripts/setup-host.sh consumes the contract.
        """
        result = _run_module()

        assert result.returncode == 0, result.stderr
        assert len(result.stdout.splitlines()) == 1, result.stdout
        assert json.loads(result.stdout) == jcodemunch_server_config()


class TestJcodemunchLauncherInstall:
    """The prebuilt launcher JCODEMUNCH_COMMAND names, installed at an exact pin."""

    def test_install_argv_installs_the_launcher_at_an_exact_pin(self):
        """An exact `==` pin and interpreter: the contract rejects a floating uvx resolve."""
        assert jcodemunch_install_argv() == [
            'uv',
            'tool',
            'install',
            '--python',
            JCODEMUNCH_PYTHON,
            JCODEMUNCH_REQUIREMENT,
        ]
        assert re.fullmatch(r'jcodemunch-mcp==\d+(\.\d+)+', JCODEMUNCH_REQUIREMENT)
        assert re.fullmatch(r'\d+\.\d+', JCODEMUNCH_PYTHON)

    def test_install_argv_is_a_fresh_list(self):
        jcodemunch_install_argv().append('x')

        assert 'x' not in jcodemunch_install_argv()

    def test_pin_stays_below_the_release_that_drops_the_identity_lever(self):
        version = JCODEMUNCH_REQUIREMENT.split('==', 1)[1]
        major = int(version.split('.', 1)[0])

        assert major < 2, (
            f'{JCODEMUNCH_REQUIREMENT}: the JCODEMUNCH_GIT_ROOT_IDENTITY env '
            'fallback is deprecated for removal in v2.0 (see the comment at '
            'JCODEMUNCH_ENV); re-establish the lever in config.jsonc before '
            'bumping the pin past it'
        )

    def test_module_prints_the_install_argv_one_per_line_on_the_bare_stdlib(self):
        result = _run_module('install-argv')

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == jcodemunch_install_argv()

    def test_module_rejects_an_unknown_rendering(self):
        """A typo'd rendering in a shell consumer must fail, not receive the server config."""
        result = _run_module('bogus')

        assert result.returncode == 2
        assert result.stdout == ''
