"""Tests for cockpit.panes.spawn_bar — the `n` spawn-bar's argv construction.

build_spawn_argv is pure: it renders the verified spawn-claude.sh positional
contract (`spawn-claude.sh <cwd> <skip_perms:true|false> <title> <prompt>`)
as a plain, subprocess-ready list[str], with no process launched here. The
SpawnScreen picker widget itself is covered by test_app.py's pilot tests (a
later C5b step).
"""

from __future__ import annotations


class TestBuildSpawnArgv:
    def test_returns_positional_argv_matching_spawn_claude_contract(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
        )

        assert argv == [
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'false',
            'unblock:df#2085',
            'Please look at this',
        ]

    def test_skip_perms_true_maps_to_true_literal(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
            skip_perms=True,
        )

        assert argv[2] == 'true'

    def test_skip_perms_false_maps_to_false_literal(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
            skip_perms=False,
        )

        assert argv[2] == 'false'

    def test_skip_perms_defaults_to_false(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
        )

        assert argv[2] == 'false'

    def test_empty_title_passed_through_as_empty_string(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            '',
            'Please look at this',
        )

        assert argv[3] == ''

    def test_script_path_object_rendered_as_str(self):
        from pathlib import Path

        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            Path('/repo/skills/spawn/spawn-claude.sh'),
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
        )

        assert argv[0] == '/repo/skills/spawn/spawn-claude.sh'
        assert isinstance(argv[0], str)

    def test_argv_is_plain_list_of_str_subprocess_ready(self):
        from cockpit.panes.spawn_bar import build_spawn_argv

        argv = build_spawn_argv(
            '/repo/skills/spawn/spawn-claude.sh',
            '/home/leo/src/dark-factory',
            'unblock:df#2085',
            'Please look at this',
            skip_perms=True,
        )

        assert isinstance(argv, list)
        assert all(isinstance(part, str) for part in argv)
        assert len(argv) == 5


class TestDefaultSkipPerms:
    def test_unset_env_defaults_to_false(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.delenv('CLAUDE_SPAWN_SKIP_PERMS', raising=False)

        assert default_skip_perms() is False

    def test_env_true_resolves_to_true(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', 'true')

        assert default_skip_perms() is True

    def test_env_false_resolves_to_false(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', 'false')

        assert default_skip_perms() is False

    def test_env_1_resolves_to_true(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', '1')

        assert default_skip_perms() is True

    def test_env_0_resolves_to_false(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', '0')

        assert default_skip_perms() is False

    def test_env_value_is_case_and_whitespace_insensitive(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', '  TRUE  ')

        assert default_skip_perms() is True

    def test_env_yes_resolves_to_true(self, monkeypatch):
        from cockpit.panes.spawn_bar import default_skip_perms

        monkeypatch.setenv('CLAUDE_SPAWN_SKIP_PERMS', 'yes')

        assert default_skip_perms() is True
