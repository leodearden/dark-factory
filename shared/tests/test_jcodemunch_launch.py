"""Tests for shared.jcodemunch_launch — the jcodemunch-mcp stdio launch contract."""

from __future__ import annotations

from shared.jcodemunch_launch import JCODEMUNCH_COMMAND, JCODEMUNCH_ENV


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
