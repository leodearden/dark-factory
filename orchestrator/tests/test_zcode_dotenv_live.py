"""The real ZCode CLI, sandboxed like a production dispatch, never hands its agent's tools a secret from the checkout's .env.

ZCode loads the first ``.env`` on a lexical walk up from its ``--cwd`` before
the agent acts; plans/zcode-harness-probe-2026-11.md §"Dotenv walk" holds the
measurement. A stdio MCP child stands in for every agent tool: ZCode spawns it
at startup with the process env its Bash tool also gets. A local endpoint that
refuses every request ends the run within seconds. A control run on a worktree
holding its own ``.env``, which the walk still loads, shows the channel can see
a loaded value at all; the probed worktree holds none, so the walk would go on
up to the checkout if nothing stopped it.
"""

from __future__ import annotations

import http.server
import json
import shutil
import sys
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _landlock_required import LANDLOCK_SKIP_REASON, skip_without_landlock

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import invoke_agent

CANARY_VAR = 'ZCODE_DOTENV_CANARY'
CONTROL_VAR = 'ZCODE_DOTENV_CONTROL'
REFUSAL = json.dumps(
    {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'probe'}},
).encode()
ENV_DUMP_SERVER = '''import json
import os
import sys

with open(sys.argv[1], 'w') as handle:
    json.dump({'environ': dict(os.environ), 'cwd': os.getcwd()}, handle)
'''

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which('zcode') is None,
        reason='the ZCode CLI is not installed; plans/zcode-harness-probe-2026-11.md §Distribution',
    ),
    pytest.mark.skipif(skip_without_landlock(), reason=LANDLOCK_SKIP_REASON),
]


class _RefusingHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        self.send_response(401)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(REFUSAL)))
        self.end_headers()
        self.wfile.write(REFUSAL)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def refusing_endpoint() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _RefusingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


@dataclass(frozen=True)
class ToolView:
    environ: dict[str, str]
    cwd: Path


@pytest.fixture
def env_dump_server(tmp_path) -> Path:
    script = tmp_path / 'env_dump_server.py'
    script.write_text(ENV_DUMP_SERVER)
    return script


def _worktree_in(checkout: Path, name: str) -> Path:
    wt = checkout / '.worktrees' / name
    wt.mkdir(parents=True)
    (wt / 'README.md').write_text('fixture\n')
    return wt


async def _what_the_tools_see(
    wt: Path, endpoint: str, server_script: Path, dump_dir: Path,
) -> ToolView:
    dump_dir.mkdir()
    dump_path = dump_dir / 'env-dump.json'
    server = {'command': sys.executable, 'args': [str(server_script), str(dump_path)]}
    result = await invoke_agent(
        prompt='reply with ok', system_prompt='SYS', cwd=wt, model='glm-5.3', backend='zcode',
        env_overrides={'ANTHROPIC_BASE_URL': endpoint, 'ANTHROPIC_AUTH_TOKEN': 'probe-key'},
        mcp_config={'mcpServers': {'env-dump': server}},
        sandbox_modules=[], sandbox_extras=[str(dump_dir)], timeout_seconds=90,
    )
    assert dump_path.exists(), (
        'ZCode never started its MCP child, so the probe observed nothing: '
        f'output={result.output!r} stderr={result.stderr!r}'
    )
    dump = json.loads(dump_path.read_text())
    return ToolView(environ=dump['environ'], cwd=Path(dump['cwd']))


@pytest.mark.asyncio
async def test_the_checkout_env_file_never_reaches_a_sandboxed_zcode_agent_s_tools(
    tmp_path, refusing_endpoint, env_dump_server,
):
    canary = uuid.uuid4().hex
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / '.env').write_text(f'{CANARY_VAR}={canary}\n')
    control_wt = _worktree_in(checkout, 'control')
    control = uuid.uuid4().hex
    (control_wt / '.env').write_text(f'{CONTROL_VAR}={control}\n')
    wt = _worktree_in(checkout, 'wt')
    sandbox_dispatch.set_backend('landlock')

    seen = await _what_the_tools_see(
        control_wt, refusing_endpoint, env_dump_server, tmp_path / 'control-dump',
    )
    assert seen.environ.get(CONTROL_VAR) == control, (
        "the observation channel went blind: the worktree's own .env, which ZCode's walk "
        'loads, never reached the MCP child, so an absent checkout canary proves nothing'
    )

    seen = await _what_the_tools_see(
        wt, refusing_endpoint, env_dump_server, tmp_path / 'probe-dump',
    )
    assert CANARY_VAR not in seen.environ
    assert not [name for name, value in seen.environ.items() if canary in value]
    assert seen.cwd.resolve() == wt.resolve()
