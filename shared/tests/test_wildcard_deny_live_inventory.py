"""Live check of what a schema + ``disallowed_tools=['*']`` argv leaves callable.

``test_build_claude_argv.py`` pins the argv TOKENS.  This test pins their
EFFECT: it runs the installed CLI on exactly the argv ``build_claude_argv``
builds for a pure classifier, reads the tool registry the CLI reports in its
stream-json ``system/init`` event, and asserts that only the synthetic
StructuredOutput tool is left.  It is the check that catches a future CLI
release changing ``--tools ''`` semantics so that tools leak back in.  The
old enumerated deny list failed this way, silently: ToolSearch loaded and ran
CronList under it.

Integration-only, following the ``shared/pyproject.toml`` marker convention for
tests that spawn the real CLI.  It costs nothing: the config dir is empty and
every auth env var is stripped, so the CLI cannot authenticate and makes no
API call.  It still emits ``system/init`` (then 'Not logged in'), and the
process group is killed as soon as that event arrives.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

import pytest

from shared.cli_invoke import build_claude_argv, no_mcp_servers_config

_AUTH_ENV_VARS = ('CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')
_INIT_DEADLINE_SECS = 60.0

_need_claude_cli = pytest.mark.skipif(
    shutil.which('claude') is None,
    reason='Requires the real claude CLI on PATH',
)


def _stream_json_argv(cmd: list[str]) -> list[str]:
    fmt_idx = cmd.index('--output-format') + 1
    return [*cmd[:fmt_idx], 'stream-json', *cmd[fmt_idx + 1:], '--verbose']


def _read_init_event(cmd: list[str], *, cwd: Path, config_dir: Path) -> dict | None:
    env = {k: v for k, v in os.environ.items() if k not in _AUTH_ENV_VARS}
    env['CLAUDE_CONFIG_DIR'] = str(config_dir)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )

    def kill_group() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)

    deadline = threading.Timer(_INIT_DEADLINE_SECS, kill_group)
    deadline.start()
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(b'x')
        proc.stdin.close()
        for line in proc.stdout:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('type') == 'system' and record.get('subtype') == 'init':
                return record
        return None
    finally:
        deadline.cancel()
        kill_group()
        proc.wait()


@pytest.mark.integration
@_need_claude_cli
def test_schema_wildcard_argv_leaves_only_structured_output(tmp_path: Path) -> None:
    cmd, temp_files = build_claude_argv(
        model='haiku',
        max_budget_usd=0.01,
        system_prompt='classifier',
        max_turns=1,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        disallowed_tools=['*'],
        mcp_config=no_mcp_servers_config(),
        output_schema={'type': 'object', 'properties': {'action': {'type': 'string'}}},
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
    )
    cwd = tmp_path / 'cwd'
    config_dir = tmp_path / 'config'
    cwd.mkdir()
    config_dir.mkdir()
    try:
        init = _read_init_event(_stream_json_argv(cmd), cwd=cwd, config_dir=config_dir)
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)

    assert init is not None, 'the CLI never emitted its system/init event'
    assert sorted(init['tools']) == ['StructuredOutput']
    assert init['mcp_servers'] == []
