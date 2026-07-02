"""Tests for shared.neutral_cwd.neutral_cli_cwd.

A neutral CLI cwd is an empty scratch directory used for pure prompt-contained
LLM classifiers (disallowed_tools=['*']) so the `claude` CLI does not
auto-load the filing project's CLAUDE.md or cwd-keyed MEMORY.md into every
call's cache-creation — see task 1989.
"""

from __future__ import annotations


class TestNeutralCliCwd:
    def test_returns_existing_directory(self):
        from shared.neutral_cwd import neutral_cli_cwd

        path = neutral_cli_cwd()
        assert path.exists()
        assert path.is_dir()

    def test_directory_has_no_claude_md(self):
        """Neutral — no filing-project context (CLAUDE.md) auto-loads."""
        from shared.neutral_cwd import neutral_cli_cwd

        path = neutral_cli_cwd()
        assert not (path / 'CLAUDE.md').exists()

    def test_stable_across_calls(self):
        """Process-wide cache — no per-call temp-dir leak."""
        from shared.neutral_cwd import neutral_cli_cwd

        first = neutral_cli_cwd()
        second = neutral_cli_cwd()
        assert first == second

    def test_differs_from_project_root(self, tmp_path):
        """Neutral cwd sits outside the filing project's tree entirely.

        Not merely unequal (a fresh mkdtemp path is trivially unequal to any
        other path), but genuinely neither an ancestor nor a descendant of
        the project root.
        """
        from shared.neutral_cwd import neutral_cli_cwd

        project_root = tmp_path / 'proj'
        project_root.mkdir()
        neutral = neutral_cli_cwd()
        assert neutral != project_root
        assert project_root not in neutral.parents
        assert neutral not in project_root.parents
