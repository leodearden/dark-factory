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


def test_build_claude_argv_resume_keeps_json_schema_and_denylist() -> None:
    """RESUME + output_schema: the schema survives, the system prompt does not.

    This asymmetry is load-bearing for every caller that carries an output
    contract across a cap-retry resume (cli_invoke.py:1270-1272 re-invokes with
    resume_session_id set and all other kwargs intact):

      --system-prompt-file  DROPPED on resume  (:1501-1503, inside the branch)
      --json-schema         ALWAYS emitted     (:1552-1553, outside the branch)

    So a prose-carried contract silently evaporates on resume while a
    schema-carried one is undroppable — which is why the reconciliation judge was
    migrated onto output_schema (task 3067).  The existing resume test above
    cannot pin this: it passes output_schema=None.  Also confirms the wildcard
    deny-list expansion applies on the resume path too, so the synthetic
    StructuredOutput tool the schema rides on is not blocked.

    If this test fails, the premise of the judge migration is broken — escalate
    rather than editing the assertions.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='dropped on resume',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=['*'],
        mcp_config=None,
        output_schema={'type': 'object'},
        effort=None,
        resume_session_id='resume-abc',
        session_id='sess-ignored',
    )
    try:
        # The system prompt (and its pre-allocated session id) are gone...
        assert temp_files == [], f'resume path must create no sysprompt file; got {temp_files!r}'
        assert '--system-prompt-file' not in cmd
        assert '--session-id' not in cmd
        assert '--resume' in cmd
        assert cmd[cmd.index('--resume') + 1] == 'resume-abc'

        # ...but the schema is still there, serialized immediately after the flag.
        assert '--json-schema' in cmd
        assert cmd[cmd.index('--json-schema') + 1] == json.dumps({'type': 'object'})

        # ...and the wildcard was expanded, not emitted literally, so the
        # synthetic StructuredOutput tool is never denied.
        idx = cmd.index('--disallowed-tools')
        values = cmd[idx + 1:cmd.index('--json-schema', idx)]
        assert values == _REAL_BUILTIN_TOOLS_DENYLIST, f'got {values!r}'
        assert '*' not in values
        assert 'StructuredOutput' not in values
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_strict_mcp_config_emits_flag_after_mcp_config() -> None:
    """strict_mcp_config=True (with an mcp_config) appends --strict-mcp-config
    immediately after the --mcp-config <path> pair — the recon-watch isolation
    pattern that scopes the invocation to ONLY the --mcp-config servers,
    ignoring the ambient .mcp.json (task 2796, THREAD 2).
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
        strict_mcp_config=True,
    )
    try:
        assert '--strict-mcp-config' in cmd
        sysprompt_path, mcp_path = temp_files
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
            '--strict-mcp-config',
            '--json-schema', json.dumps({'type': 'object'}),
        ], temp_files)
        assert _normalize(cmd, temp_files) == expected, f'got {cmd!r}'
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_default_omits_strict_mcp_config() -> None:
    """Default (strict_mcp_config omitted) WITH an mcp_config emits NO
    --strict-mcp-config — byte-compatible with every existing caller.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='sys',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=None,
        mcp_config={'mcpServers': {'foo': {'command': 'bar'}}},
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
    )
    try:
        assert '--mcp-config' in cmd
        assert '--strict-mcp-config' not in cmd
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_strict_mcp_config_noop_without_mcp_config() -> None:
    """strict_mcp_config=True with mcp_config=None is a no-op: --strict-mcp-config
    is meaningless with no --mcp-config (it is emitted only inside the
    `if mcp_config:` block), so it is never added.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='sys',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=None,
        mcp_config=None,
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
    )
    try:
        assert '--mcp-config' not in cmd
        assert '--strict-mcp-config' not in cmd
    finally:
        _cleanup(temp_files)
