"""Tests for shared.jcodemunch_launch — the jcodemunch-mcp stdio launch contract.

Locks the two constants that every project's MCP config injection must
reference, so the launch contract has exactly one definition rather than
being re-spelled with divergent fidelity at each site.  Mirrors
``orchestrator/tests/test_mcp_lifecycle.py::TestJcodemunchLaunchPinned``
(task 4562), which pins the same contract as observed through the
orchestrator's ``mcp_config_json()``.

Consumers guarded by this contract:
``orchestrator.mcp_lifecycle.McpLifecycle.mcp_config_json`` and
``fused_memory.reconciliation.stages.base.BaseStage._build_mcp_config``.
"""

from __future__ import annotations

from shared.jcodemunch_launch import JCODEMUNCH_COMMAND, JCODEMUNCH_ENV


class TestJcodemunchLaunchContract:
    """The shared constants are the single source of truth for the launch contract."""

    def test_command_is_prebuilt_launcher_not_uvx(self):
        """JCODEMUNCH_COMMAND is the prebuilt launcher 'jcodemunch-mcp', not 'uvx'.

        ``uvx`` re-resolves the package and builds tree-sitter C-extension
        sdists from source on every launch, which under host load stalled
        agent startup past the 1200s wall (the 0-turn MCP-startup wedge,
        reify esc-4415-232).  The installed launcher on PATH starts in <1s
        and fails fast when absent.
        """
        assert JCODEMUNCH_COMMAND == 'jcodemunch-mcp'
        assert JCODEMUNCH_COMMAND != 'uvx'

    def test_env_contains_no_version_hint(self):
        """JCODEMUNCH_ENV silences the stderr version-drift note."""
        assert 'JCODEMUNCH_NO_VERSION_HINT' in JCODEMUNCH_ENV
        assert JCODEMUNCH_ENV['JCODEMUNCH_NO_VERSION_HINT'] == '1'

    def test_env_sets_git_root_identity_lever(self):
        """JCODEMUNCH_ENV pins the identity lever to the exact literal '0'.

        jcodemunch's bool env parser treats '0' as False (lever ON, selecting
        per-worktree ``local/<basename>-<sha1[:8]>`` identity) but '1'/'true'
        as True (lever OFF, silently reverting to the worktree-collapsing
        git-root default) — so this must assert the literal value, not just
        key presence.
        """
        assert 'JCODEMUNCH_GIT_ROOT_IDENTITY' in JCODEMUNCH_ENV
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
