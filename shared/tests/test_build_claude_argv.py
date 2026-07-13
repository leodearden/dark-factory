"""Tests for build_claude_argv() — the single source of truth for Claude CLI
argv assembly, shared by the non-sandbox (_invoke_claude) and sandbox
(_invoke_claude_with_sandbox) invocation paths (task 2465 dedup).
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.cli_invoke import _REAL_BUILTIN_TOOLS_DENYLIST, build_claude_argv

_TMP_PLACEHOLDER = '<TMP>'


def _normalize(cmd: list[str], temp_files: list[str]) -> list[str]:
    """Replace any temp-file-valued token in *cmd* with a fixed placeholder.

    Temp file paths are nondeterministic (tempfile.mkstemp), so structural
    comparisons must normalize them before asserting equality.
    """
    temp_set = set(temp_files)
    return [_TMP_PLACEHOLDER if tok in temp_set else tok for tok in cmd]


def _cleanup(temp_files: list[str]) -> None:
    for path in temp_files:
        Path(path).unlink(missing_ok=True)


def test_build_claude_argv_fresh_full_options() -> None:
    """FRESH (non-resume) case: full flag set in the documented order.

    Also confirms the sysprompt/mcp-config temp files are actually created
    on disk with the expected contents, and both paths are returned in
    temp_files for the caller to clean up.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='sys prompt text',
        max_turns=50,
        permission_mode='bypassPermissions',
        allowed_tools=['Read', 'Grep'],
        disallowed_tools=['Bash'],
        mcp_config={'mcpServers': {'foo': {'command': 'bar'}}},
        output_schema={'type': 'object'},
        effort='high',
        resume_session_id=None,
        session_id='sess-123',
    )
    try:
        assert len(temp_files) == 2, f'expected sysprompt + mcp temp files; got {temp_files!r}'
        sysprompt_path, mcp_path = temp_files
        assert Path(sysprompt_path).read_text() == 'sys prompt text'
        assert json.loads(Path(mcp_path).read_text()) == {'mcpServers': {'foo': {'command': 'bar'}}}

        expected = _normalize([
            'claude', '--print', '--output-format', 'json',
            '--model', 'opus',
            '--max-budget-usd', '5.0',
            '--system-prompt-file', sysprompt_path,
            '--session-id', 'sess-123',
            '--permission-mode', 'bypassPermissions',
            '--max-turns', '50',
            '--effort', 'high',
            '--allowed-tools', 'Read', 'Grep',
            '--disallowed-tools', 'Bash',
            '--mcp-config', mcp_path,
            '--json-schema', json.dumps({'type': 'object'}),
        ], temp_files)

        assert _normalize(cmd, temp_files) == expected, f'got {cmd!r}'
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_resume_skips_system_prompt_file() -> None:
    """RESUME case: --resume replaces --system-prompt-file/--session-id entirely.

    No sysprompt temp file is created (temp_files is empty) since the system
    prompt was already set on the session being resumed.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='unused system prompt',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=None,
        mcp_config=None,
        output_schema=None,
        effort=None,
        resume_session_id='resume-abc',
        session_id='sess-should-be-ignored',
    )
    try:
        assert temp_files == [], f'resume path must create no sysprompt temp file; got {temp_files!r}'
        assert '--system-prompt-file' not in cmd
        assert '--session-id' not in cmd
        assert cmd == [
            'claude', '--print', '--output-format', 'json',
            '--model', 'opus',
            '--max-budget-usd', '5.0',
            '--resume', 'resume-abc',
            '--permission-mode', 'bypassPermissions',
            '--max-turns', '10',
        ], f'got {cmd!r}'
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_expands_deny_list_when_schema_and_wildcard() -> None:
    """CLI-2.1.168 fix: output_schema + disallowed_tools=['*'] expands the
    wildcard into the real-builtins deny-list (which omits StructuredOutput)
    instead of emitting the literal '*'.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='sys',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=['*'],
        mcp_config=None,
        output_schema={'type': 'object'},
        effort=None,
        resume_session_id=None,
        session_id=None,
    )
    try:
        idx = cmd.index('--disallowed-tools')
        next_flag_idx = cmd.index('--json-schema', idx)
        values = cmd[idx + 1:next_flag_idx]
        assert values == _REAL_BUILTIN_TOOLS_DENYLIST, f'got {values!r}'
        assert '*' not in values
        assert 'StructuredOutput' not in values
    finally:
        _cleanup(temp_files)
