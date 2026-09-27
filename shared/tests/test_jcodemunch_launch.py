"""Tests for shared.jcodemunch_launch — the jcodemunch-mcp stdio launch contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from shared.jcodemunch_launch import (
    JCODEMUNCH_COMMAND,
    JCODEMUNCH_ENV,
    jcodemunch_server_config,
)

_SHARED_SRC = Path(__file__).resolve().parents[1] / 'src'


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
        result = subprocess.run(
            [sys.executable, '-S', '-m', 'shared.jcodemunch_launch'],
            env={**os.environ, 'PYTHONPATH': str(_SHARED_SRC)},
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert len(result.stdout.splitlines()) == 1, result.stdout
        assert json.loads(result.stdout) == jcodemunch_server_config()
