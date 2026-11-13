"""The real ZCode CLI, sandboxed like a production dispatch, never hands its agent's tools a secret from the checkout's .env.

ZCode loads the first ``.env`` on a lexical walk up from its ``--cwd`` before
the agent acts; plans/zcode-harness-probe-2026-11.md §"Dotenv walk" holds the
measurement. A stdio MCP child stands in for every agent tool: ZCode spawns it
at startup with the process env its Bash tool also gets. A local endpoint that
refuses every request ends the run within seconds.
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

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.landlock import is_landlock_available

CANARY_VAR = 'ZCODE_DOTENV_CANARY'
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
    pytest.mark.skipif(not is_landlock_available(), reason='landlock is unavailable on this host'),
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
class EnvDump:
    mcp_config: dict[str, Any]
    path: Path

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())


@pytest.fixture
def env_dump(tmp_path) -> EnvDump:
    script = tmp_path / 'env_dump_server.py'
    script.write_text(ENV_DUMP_SERVER)
    dump_path = tmp_path / 'env-dump.json'
    server = {'command': sys.executable, 'args': [str(script), str(dump_path)]}
    return EnvDump(mcp_config={'mcpServers': {'env-dump': server}}, path=dump_path)


@pytest.mark.asyncio
async def test_the_checkout_env_file_never_reaches_a_sandboxed_zcode_agent_s_tools(
    tmp_path, refusing_endpoint, env_dump,
):
    canary = uuid.uuid4().hex
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / '.env').write_text(f'{CANARY_VAR}={canary}\n')
    wt = checkout / '.worktrees' / 'wt'
    wt.mkdir(parents=True)
    (wt / 'README.md').write_text('fixture\n')
    sandbox_dispatch.set_backend('landlock')

    result = await invoke_agent(
        prompt='reply with ok', system_prompt='SYS', cwd=wt, model='glm-5.3', backend='zcode',
        env_overrides={
            'ANTHROPIC_BASE_URL': refusing_endpoint, 'ANTHROPIC_AUTH_TOKEN': 'probe-key',
        },
        mcp_config=env_dump.mcp_config, sandbox_modules=[], timeout_seconds=90,
    )

    assert env_dump.path.exists(), (
        'ZCode never started its MCP child, so the probe observed nothing: '
        f'output={result.output!r} stderr={result.stderr!r}'
    )
    environ = env_dump.read()['environ']
    assert CANARY_VAR not in environ
    assert not [name for name, value in environ.items() if canary in value]
    assert Path(env_dump.read()['cwd']).resolve() == wt.resolve()
