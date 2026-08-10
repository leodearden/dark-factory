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


def test_build_claude_argv_resume_still_passes_system_prompt_file() -> None:
    """RESUME case: --resume displaces ONLY --session-id, never the system prompt.

    The system prompt is a process-invocation parameter that the session does
    not carry, so a resumed invocation that omits --system-prompt-file runs
    under the stock Claude Code prompt with no role charter (task 3983).  The
    sysprompt temp file is therefore created on the resume path too, and holds
    the CURRENT ``system_prompt`` argument.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='role charter text',
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
        assert len(temp_files) == 1, f'resume path must create a sysprompt temp file; got {temp_files!r}'
        assert Path(temp_files[0]).read_text() == 'role charter text'
        assert '--system-prompt-file' in cmd
        assert cmd[cmd.index('--system-prompt-file') + 1] == temp_files[0]
        # --session-id stays displaced by --resume: unlike --system-prompt-file,
        # it IS genuinely exclusive with it.  Probed on CLI 2.1.226:
        #   "Error: --session-id can only be used with --continue or --resume
        #    if --fork-session is also specified."
        # Do not "helpfully" hoist --session-id out of the branch too.
        assert '--session-id' not in cmd

        expected = _normalize([
            'claude', '--print', '--output-format', 'json',
            '--model', 'opus',
            '--max-budget-usd', '5.0',
            '--system-prompt-file', temp_files[0],
            '--resume', 'resume-abc',
            '--permission-mode', 'bypassPermissions',
            '--max-turns', '10',
        ], temp_files)
        assert _normalize(cmd, temp_files) == expected, f'got {cmd!r}'
    finally:
        _cleanup(temp_files)


def test_build_claude_argv_fresh_and_resume_carry_identical_system_prompt() -> None:
    """A resumed invocation is byte-identical to a fresh one but for --resume.

    Task 3983's central property, which no single-case test can express: the
    ONE hoisted sysprompt write serves both arms, so replacing the prompt (not
    appending it) makes the resumed argv the fresh argv with a single
    ``['--resume', <id>]`` pair spliced in.  Both arms carry the same prompt
    text to disk.
    """
    kwargs: dict = dict(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='shared role charter',
        max_turns=10,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=None,
        mcp_config=None,
        output_schema=None,
        effort=None,
        session_id=None,
    )
    fresh_cmd, fresh_temps = build_claude_argv(resume_session_id=None, **kwargs)
    resume_cmd, resume_temps = build_claude_argv(resume_session_id='r1', **kwargs)
    try:
        assert '--system-prompt-file' in fresh_cmd
        assert '--system-prompt-file' in resume_cmd
        assert len(fresh_temps) == 1 and len(resume_temps) == 1
        assert Path(fresh_temps[0]).read_text() == Path(resume_temps[0]).read_text() == 'shared role charter'

        # The resumed argv is the fresh argv with ['--resume', 'r1'] spliced in
        # immediately after the --system-prompt-file <path> pair — nothing else
        # differs once the nondeterministic temp paths are normalized.
        fresh_norm = _normalize(fresh_cmd, fresh_temps)
        resume_norm = _normalize(resume_cmd, resume_temps)
        splice_at = fresh_norm.index('--system-prompt-file') + 2
        assert fresh_norm[:splice_at] + ['--resume', 'r1'] + fresh_norm[splice_at:] == resume_norm, (
            f'fresh={fresh_cmd!r} resume={resume_cmd!r}'
        )
    finally:
        _cleanup(fresh_temps)
        _cleanup(resume_temps)


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


def test_build_claude_argv_resume_keeps_system_prompt_schema_and_denylist() -> None:
    """RESUME + output_schema: BOTH the system prompt and the schema survive.

    HISTORY.  This test used to pin the OPPOSITE — an asymmetry where a
    prose-carried contract silently evaporated on resume while a schema-carried
    one was undroppable:

      --system-prompt-file  DROPPED on resume  (inside the resume branch)
      --json-schema         ALWAYS emitted     (outside the branch)

    That asymmetry is GONE as of task 3983, which hoisted the sysprompt write
    above the resume branch: --system-prompt-file is now emitted unconditionally
    alongside --json-schema, so both survive a resume.  This test's old
    "if this test fails, escalate rather than editing the assertions" tripwire
    was retired deliberately by that task — it fired for the one reason it was
    meant to allow, namely the underlying defect being FIXED rather than an
    assertion being weakened to hide a regression.

    The reconciliation judge's migration onto output_schema (task 3067) remains
    correct on its own merits and is NOT invalidated by this: the
    anthropic/openai provider branches never see --json-schema at all, and a
    machine-validated contract beats a prose one regardless.  It simply is no
    longer needed as a *resume* workaround.

    Also confirms the wildcard deny-list expansion applies on the resume path,
    so the synthetic StructuredOutput tool the schema rides on is not blocked.
    """
    cmd, temp_files = build_claude_argv(
        model='opus',
        max_budget_usd=5.0,
        system_prompt='carried across resume',
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
        # The system prompt survives; only its pre-allocated session id is gone...
        assert len(temp_files) == 1, f'resume path must create a sysprompt file; got {temp_files!r}'
        assert '--system-prompt-file' in cmd
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
