"""Live check that a TaskConfigDir's ``.credentials.json`` never authenticates the CLI.

``invoke_with_cap_retry`` rewrites that file, non-atomically, on every gated
call, and the task curator shares ONE config dir across concurrent calls on
different accounts. That is safe only because each call authenticates from
the ``CLAUDE_CODE_OAUTH_TOKEN`` it sets, never from the file. Measured on CLI
2.1.284 (task 3995): a valid env token authenticated over a garbage, empty or
half-written file; a garbage env token failed with a 401 over a valid file;
and a file written by ``write_credentials`` with no env token reported "Not
logged in" even when it held a valid token.

This test pins the last fact at zero cost. A CLI that read the file would try
its garbage token and fail with a 401; one that ignores the file makes no API
call. If it fails, the file has become a credential source, and the curator's
shared config dir must be re-examined.

Integration-only, like the other tests that spawn the real CLI. User settings
are not loaded (``--setting-sources project`` at an empty cwd), so this
machine's hooks do not run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from shared.cli_invoke import no_mcp_servers_config
from shared.config_dir import TaskConfigDir

_AUTH_ENV_VARS = ('CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')
_NOT_A_TOKEN = 'sk-ant-oat01-not-a-real-token'


@pytest.mark.integration
@pytest.mark.skipif(shutil.which('claude') is None, reason='Requires the real claude CLI on PATH')
def test_written_credentials_file_does_not_authenticate(tmp_path: Path) -> None:
    config_dir = TaskConfigDir('credentials-probe', base_dir=tmp_path)
    config_dir.write_credentials(_NOT_A_TOKEN)
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _AUTH_ENV_VARS}
    env['CLAUDE_CONFIG_DIR'] = str(config_dir.path)

    proc = subprocess.run(
        [
            'claude', '-p', '--output-format', 'json', '--model', 'haiku', '--max-turns', '1',
            '--tools', '', '--strict-mcp-config',
            '--mcp-config', json.dumps(no_mcp_servers_config()),
            '--setting-sources', 'project',
        ],
        input=b'x',
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=120,
    )

    result = json.loads(proc.stdout)
    assert result['is_error'] is True
    assert 'Not logged in' in result['result']
