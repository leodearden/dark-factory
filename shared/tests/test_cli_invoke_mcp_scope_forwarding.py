"""invoke_with_cap_retry carries zero-server strict MCP scoping into the CLI argv.

The wildcard-MCP scoping gate (``shared/tests/wildcard_mcp_scoping_scan.py``)
counts a call site as closed when it spells
``mcp_config=no_mcp_servers_config()`` and ``strict_mcp_config=True``, and each
site's own regression test checks only the kwargs handed to a patched
``invoke_with_cap_retry``.  Both rest on the default dispatch path
(``invoke_with_cap_retry`` -> ``invoke_claude_agent`` -> ``build_claude_argv``)
forwarding the pair; this is the one test of that, covering every site at once.

The argv is read through the public ``sandbox_wrap`` hook, which receives the
final command just before spawn.  The hook raises, so no subprocess starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.cli_invoke import invoke_with_cap_retry, no_mcp_servers_config


class _SpawnIntercepted(Exception):
    def __init__(self, argv: list[str], mcp_config_file: object) -> None:
        super().__init__('sandbox_wrap stopped the run before spawn')
        self.argv = argv
        self.mcp_config_file = mcp_config_file


def _stop_before_spawn(cmd: list[str]) -> list[str]:
    mcp_config_file = None
    if '--mcp-config' in cmd:
        mcp_config_file = json.loads(Path(cmd[cmd.index('--mcp-config') + 1]).read_text())
    raise _SpawnIntercepted(list(cmd), mcp_config_file)


async def test_zero_server_strict_scoping_reaches_the_cli_argv(tmp_path: Path) -> None:
    with pytest.raises(_SpawnIntercepted) as intercepted:
        await invoke_with_cap_retry(
            None,
            'mcp-scope-forwarding',
            prompt='classify this',
            system_prompt='sys',
            cwd=tmp_path,
            disallowed_tools=['*'],
            output_schema={'type': 'object'},
            mcp_config=no_mcp_servers_config(),
            strict_mcp_config=True,
            sandbox_wrap=_stop_before_spawn,
        )

    argv = intercepted.value.argv
    assert '--mcp-config' in argv, argv
    assert argv[argv.index('--mcp-config') + 2] == '--strict-mcp-config', argv
    assert intercepted.value.mcp_config_file == {'mcpServers': {}}
