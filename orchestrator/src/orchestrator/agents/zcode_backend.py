"""The ZCode CLI as an agent backend: the documents, argv and result parsing of one headless ``zcode --prompt`` run.

The measured evidence behind every shape here is plans/zcode-harness-probe-2026-11.md.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.cli_invoke import AgentResult, _SubprocessResult
from shared.process_cost import ProcessCostBasis

from orchestrator.agents.agent_run_home import (
    private_run_home,
    redirected_home_env,
    write_owner_only_json,
)

logger = logging.getLogger(__name__)

BASE_URL_VAR = 'ANTHROPIC_BASE_URL'
AUTH_TOKEN_VAR = 'ANTHROPIC_AUTH_TOKEN'
PROVIDER_ID = 'dark-factory'
INTERACTIVE_TOOLS = ('AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode')
MAX_ARG_STRLEN = 131072
PROMPT_FLAG = '--prompt'

_PLAIN_TOOL_NAME = re.compile(r'[A-Za-z0-9_-]+')

CostOf = Callable[[int, int, int], float]


class ZCodeEndpointMissing(ValueError):
    """The env overrides lack a half of the endpoint pair ZCode reaches its model through."""


@dataclass(frozen=True)
class ZCodeEndpoint:
    base_url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env_overrides(cls, env_overrides: Mapping[str, str] | None) -> ZCodeEndpoint:
        overrides = env_overrides or {}
        missing = [
            name for name in (BASE_URL_VAR, AUTH_TOKEN_VAR)
            if not (overrides.get(name) or '').strip()
        ]
        if missing:
            raise ZCodeEndpointMissing(
                f'the zcode backend needs {" and ".join(missing)} in env_overrides: '
                'ZCode reaches its model only through that endpoint pair',
            )
        return cls(base_url=overrides[BASE_URL_VAR], api_key=overrides[AUTH_TOKEN_VAR])


def zcode_provider_config(model: str, endpoint: ZCodeEndpoint) -> dict[str, Any]:
    rule = {
        'providerId': PROVIDER_ID,
        'providerName': 'Dark Factory',
        'config': {
            'group': 'standard-personal',
            'access': {'type': 'api-key', 'apiKey': endpoint.api_key},
            'api': {'type': 'anthropic-messages', 'baseUrl': endpoint.base_url},
            'personalModelIds': [model],
        },
    }
    return {
        'schemaVersion': 1,
        'config': {
            'providerConfigRules': {'providerRules': [rule]},
            'modelConfigRules': {'providerModelRules': [], 'manualProviderModelRules': []},
            'defaultModelSelection': {'providerId': PROVIDER_ID, 'modelId': model},
        },
    }


def zcode_cli_config(mcp_config: Mapping[str, Any] | None) -> dict[str, Any]:
    servers = {
        name: _zcode_mcp_server(name, server)
        for name, server in (mcp_config or {}).get('mcpServers', {}).items()
    }
    return {'features': {'mcp': bool(servers)}, 'mcp': {'servers': servers}}


def _zcode_mcp_server(name: str, server: Mapping[str, Any]) -> dict[str, Any]:
    transport = server.get('type')
    if server.get('command') and transport in (None, 'stdio'):
        return {
            'type': 'stdio',
            'command': server['command'],
            'args': [str(arg) for arg in server.get('args') or []],
            'env': {str(key): str(value) for key, value in (server.get('env') or {}).items()},
            'enabled': True,
        }
    if server.get('url') and transport in (None, 'http', 'sse'):
        headers = {'headers': dict(server['headers'])} if 'headers' in server else {}
        return {'type': transport or 'http', 'url': server['url'], **headers, 'enabled': True}
    raise ValueError(
        f'MCP server {name!r} has neither a stdio command nor an http/sse url for zcode to carry',
    )


@dataclass(frozen=True)
class ZCodeRunHome:
    root: Path
    provider_config_path: Path
    operator_home: Path
    workspace: Path


@contextmanager
def zcode_run_home(
    *,
    system_prompt: str,
    model: str,
    endpoint: ZCodeEndpoint,
    mcp_config: Mapping[str, Any] | None,
    operator_home: Path,
    worktree: Path,
) -> Iterator[ZCodeRunHome]:
    """A private home for one invocation, holding the only copy of the endpoint key and the link to *worktree* ZCode runs in; removed on exit."""
    with private_run_home(backend='zcode', operator_home=operator_home) as root:
        zcode_dir = root / '.zcode'
        provider_config_path = zcode_dir / 'v2' / 'provider_config.json'
        write_owner_only_json(provider_config_path, zcode_provider_config(model, endpoint))
        cli_config_path = zcode_dir / 'cli' / 'config.json'
        cli_config_path.parent.mkdir(parents=True)
        cli_config_path.write_text(json.dumps(zcode_cli_config(mcp_config)))
        (zcode_dir / 'AGENTS.md').write_text(system_prompt)
        workspace = _fenced_workspace(root, worktree)
        yield ZCodeRunHome(
            root=root, provider_config_path=provider_config_path, operator_home=operator_home,
            workspace=workspace,
        )


def _fenced_workspace(root: Path, worktree: Path) -> Path:
    """See plans/zcode-harness-probe-2026-11.md §"Dotenv walk"."""
    (root / '.env').touch(exist_ok=False)
    workspace = root / 'workspace'
    workspace.symlink_to(worktree.resolve(), target_is_directory=True)
    return workspace


def zcode_spawn_env(base_env: Mapping[str, str], home: ZCodeRunHome) -> dict[str, str]:
    return {
        **redirected_home_env(base_env, root=home.root, operator_home=home.operator_home),
        'ZCODE_DATA_BASE_DIR': str(home.root),
        'ZCODE_PERSONAL_PROVIDER_CONFIG_FILE': str(home.provider_config_path),
    }


def zcode_argv(
    prompt: str, home: ZCodeRunHome, disallowed_tools: Iterable[str] | None,
) -> list[str]:
    return [
        'zcode', _prompt_argument(prompt), '--cwd', str(home.workspace), '--mode', 'yolo',
        '--output-format', 'json', '--no-color',
        '--disallowed-tools', ' '.join(_disallowed_tool_names(disallowed_tools)),
    ]


def _prompt_argument(prompt: str) -> str:
    """Inline, because strict parseArgs refuses a separate value that starts with '-'."""
    argument = f'{PROMPT_FLAG}={prompt}'
    size = len(argument.encode())
    if size >= MAX_ARG_STRLEN:
        raise ValueError(
            f'the zcode prompt is {len(prompt.encode())} bytes, so its {PROMPT_FLAG}= argument '
            f'is {size} bytes, but zcode takes the prompt only as that one argument and Linux '
            f'execve refuses an argument of {MAX_ARG_STRLEN} bytes or more, counting its '
            'terminating NUL (MAX_ARG_STRLEN)',
        )
    return argument


def _disallowed_tool_names(specs: Iterable[str] | None) -> list[str]:
    names: list[str] = list(INTERACTIVE_TOOLS)
    dropped: list[str] = []
    for spec in specs or ():
        if not _PLAIN_TOOL_NAME.fullmatch(spec):
            dropped.append(spec)
        elif spec not in names:
            names.append(spec)
    if dropped:
        logger.warning(
            'zcode backend: dropped disallowed tool spec(s) %s: --disallowed-tools removes '
            'whole tools, so a pattern or wildcard would remove more than it names',
            ', '.join(dropped),
        )
    return names


def parse_zcode_output(result: _SubprocessResult, *, cost_of: CostOf) -> AgentResult:
    """*cost_of* prices (uncached input, output, cached input) tokens in dollars."""
    if result.returncode != 0 or result.timed_out:
        return _failure(result, '\n'.join(filter(None, [_exit_line(result), result.stderr.strip()])))
    if not result.stdout.strip():
        return _failure(result, 'Agent produced no output', subtype='error_empty_output')
    payload = _json_object(result.stdout)
    if payload is None:
        return _failure(result, f'{_exit_line(result)} without a JSON result object')
    return _success(result, payload, cost_of)


def _exit_line(result: _SubprocessResult) -> str:
    return f'zcode exited rc={result.returncode}' + (' (timed out)' if result.timed_out else '')


def _json_object(stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _failure(result: _SubprocessResult, output: str, subtype: str = 'error') -> AgentResult:
    return AgentResult(
        success=False,
        output=output,
        subtype=subtype,
        cost_basis=ProcessCostBasis.USAGE_ESTIMATE,
        duration_ms=result.duration_ms,
        stderr=result.stderr,
        timed_out=result.timed_out,
    )


def _success(result: _SubprocessResult, payload: dict[str, Any], cost_of: CostOf) -> AgentResult:
    usage = payload.get('usage') or {}
    cache_read = int(usage.get('cacheReadTokens') or 0)
    cache_write = int(usage.get('cacheWriteTokens') or 0)
    input_tokens = int(usage.get('inputTokens') or 0)
    output_tokens = int(usage.get('outputTokens') or 0)
    cached = cache_read + cache_write
    uncached = input_tokens - cached
    if uncached < 0:
        logger.warning(
            'zcode reported cache read+write tokens %d above inputTokens %d, which should '
            'include them; counting the non-cached input as 0',
            cached, input_tokens,
        )
        uncached = 0
    return AgentResult(
        success=True,
        output=str(payload.get('response') or ''),
        cost_usd=cost_of(uncached, output_tokens, cached),
        cost_basis=ProcessCostBasis.USAGE_ESTIMATE,
        duration_ms=result.duration_ms,
        turns=int(usage.get('modelRequestCount') or 0),
        session_id=str(payload.get('sessionId') or ''),
        subtype='success',
        stderr=result.stderr,
        timed_out=result.timed_out,
        input_tokens=uncached,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_create_tokens=cache_write,
    )
