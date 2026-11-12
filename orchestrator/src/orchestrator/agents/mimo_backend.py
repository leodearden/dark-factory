"""The MiMo Code CLI as an agent backend: the config document, argv, session-store read and result parsing of one headless ``mimo run``.

The measured evidence behind every shape here is plans/mimo-code-harness-probe-2026-11.md.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.cli_invoke import AgentResult, _SubprocessResult
from shared.process_cost import ProcessCostBasis

logger = logging.getLogger(__name__)

BASE_URL_VAR = 'ANTHROPIC_BASE_URL'
AUTH_TOKEN_VAR = 'ANTHROPIC_AUTH_TOKEN'
CONFIG_SCHEMA = 'https://mimo.xiaomi.com/mimocode/config.json'
PROVIDER_ID = 'mimo-token-plan'
MODEL_LIMITS = {'context': 1048576, 'output': 131072}
TITLE = 'dark-factory'
MIMO_VARIANTS = ('low', 'medium', 'high')
FALLBACK_VARIANT = 'high'
# MiMo Code's catalog providers xiaomi-token-plan-* / xiaomi; see plans/mimo-code-harness-probe-2026-11.md (e).
OPENAI_BASE_BY_ANTHROPIC_BASE = {
    'https://token-plan-cn.xiaomimimo.com/anthropic': 'https://token-plan-cn.xiaomimimo.com/v1',
    'https://token-plan-sgp.xiaomimimo.com/anthropic': 'https://token-plan-sgp.xiaomimimo.com/v1',
    'https://token-plan-ams.xiaomimimo.com/anthropic': 'https://token-plan-ams.xiaomimimo.com/v1',
    'https://api.xiaomimimo.com/anthropic': 'https://api.xiaomimimo.com/v1',
}

_PLAIN_TOOL_NAME = re.compile(r'[A-Za-z0-9_-]+')
_MCP_TOOL = re.compile(r'mcp__(?P<server>[A-Za-z0-9_-]+?)__(?P<tool>[A-Za-z0-9_-]+)')
_BASH_PATTERN = re.compile(r'Bash\((?P<pattern>.+)\)')

CostOf = Callable[[int, int, int], float]


class MimoEndpointMissing(ValueError):
    """The env overrides lack, or misname, the endpoint MiMo Code reaches its model through."""


@dataclass(frozen=True)
class MimoEndpoint:
    openai_base_url: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env_overrides(cls, env_overrides: Mapping[str, str] | None) -> MimoEndpoint:
        overrides = env_overrides or {}
        missing = [
            name for name in (BASE_URL_VAR, AUTH_TOKEN_VAR)
            if not (overrides.get(name) or '').strip()
        ]
        if missing:
            raise MimoEndpointMissing(
                f'the mimo backend needs {" and ".join(missing)} in env_overrides: '
                'MiMo Code reaches its model only through that endpoint pair',
            )
        anthropic_base = overrides[BASE_URL_VAR].strip().rstrip('/')
        openai_base = OPENAI_BASE_BY_ANTHROPIC_BASE.get(anthropic_base)
        if openai_base is None:
            raise MimoEndpointMissing(
                f'the mimo backend has no OpenAI-protocol base for {BASE_URL_VAR} '
                f'{anthropic_base!r}; known bases: {", ".join(OPENAI_BASE_BY_ANTHROPIC_BASE)}',
            )
        return cls(openai_base_url=openai_base, api_key=overrides[AUTH_TOKEN_VAR])


def mimo_permission_rules(disallowed_tools: Iterable[str] | None) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    dropped: list[str] = []
    for spec in disallowed_tools or ():
        if mcp_tool := _MCP_TOOL.fullmatch(spec):
            rules[f'{mcp_tool["server"]}_{mcp_tool["tool"]}'] = 'deny'
        elif spec.startswith('mcp__'):
            dropped.append(spec)
        elif bash := _BASH_PATTERN.fullmatch(spec):
            _deny_bash_pattern(rules, bash['pattern'])
        elif _PLAIN_TOOL_NAME.fullmatch(spec):
            rules[spec.lower()] = 'deny'
        else:
            dropped.append(spec)
    if dropped:
        logger.warning(
            'mimo backend: dropped disallowed tool spec(s) %s: MiMo permissions name one '
            'tool or one bash glob, so a wildcard has no faithful translation',
            ', '.join(dropped),
        )
    return rules


def _deny_bash_pattern(rules: dict[str, Any], pattern: str) -> None:
    bash = rules.setdefault('bash', {'*': 'allow'})
    if isinstance(bash, dict):
        glob = pattern[:-2] + '*' if pattern.endswith(':*') else pattern
        bash[glob] = 'deny'


def mimo_config_document(
    *,
    model: str,
    endpoint: MimoEndpoint,
    mcp_config: Mapping[str, Any] | None,
    permission: Mapping[str, Any],
    max_turns: int,
) -> dict[str, Any]:
    return {
        '$schema': CONFIG_SCHEMA,
        'model': f'{PROVIDER_ID}/{model}',
        'autoupdate': False,
        'share': 'disabled',
        'provider': {PROVIDER_ID: {
            'name': 'Dark Factory MiMo',
            'only_configured_models': True,
            'models': {model: {'name': model, 'reasoning': True, 'limit': dict(MODEL_LIMITS)}},
            'options': {'baseURL': endpoint.openai_base_url, 'apiKey': endpoint.api_key},
        }},
        'mcp': {
            name: _mimo_mcp_server(name, server)
            for name, server in (mcp_config or {}).get('mcpServers', {}).items()
        },
        'permission': dict(permission),
        'agent': {'build': {'steps': max_turns}},
        'dream': {'auto': False},
        'distill': {'auto': False},
    }


def _mimo_mcp_server(name: str, server: Mapping[str, Any]) -> dict[str, Any]:
    transport = server.get('type')
    if server.get('command') and transport in (None, 'stdio'):
        return {
            'type': 'local',
            'command': [str(server['command']), *(str(arg) for arg in server.get('args') or [])],
            'environment': {str(key): str(value) for key, value in (server.get('env') or {}).items()},
            'enabled': True,
        }
    if server.get('url') and transport in (None, 'http', 'sse'):
        headers = {'headers': dict(server['headers'])} if 'headers' in server else {}
        return {'type': 'remote', 'url': server['url'], **headers, 'enabled': True}
    raise ValueError(
        f'MCP server {name!r} has neither a stdio command nor an http/sse url for mimo to carry',
    )


def mimo_variant(effort: str | None) -> str | None:
    if not effort:
        return None
    if effort in MIMO_VARIANTS:
        return effort
    logger.warning(
        'mimo backend: effort %r is not one of the variants mimo forwards (%s), and mimo '
        'silently drops any other; sending --variant %s',
        effort, ', '.join(MIMO_VARIANTS), FALLBACK_VARIANT,
    )
    return FALLBACK_VARIANT


def mimo_run_argv(model: str, cwd: Path, effort: str | None) -> list[str]:
    variant = mimo_variant(effort)
    return [
        'mimo', 'run', '--title', TITLE, '--format', 'json', '--dir', str(cwd),
        '--dangerously-skip-permissions', '-m', f'{PROVIDER_ID}/{model}',
        *(['--variant', variant] if variant is not None else []),
    ]


def mimo_session_list_argv() -> list[str]:
    return ['mimo', 'session', 'list', '--format', 'json']


def mimo_export_argv(session_id: str) -> list[str]:
    return ['mimo', 'export', session_id]


@dataclass(frozen=True)
class _Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @classmethod
    def of_step(cls, info: Mapping[str, Any]) -> _Usage:
        tokens = info.get('tokens') or {}
        cache = tokens.get('cache') or {}
        return cls(
            input=int(tokens.get('input') or 0),
            output=int(tokens.get('output') or 0) + int(tokens.get('reasoning') or 0),
            cache_read=int(cache.get('read') or 0),
            cache_write=int(cache.get('write') or 0),
        )

    def __add__(self, other: _Usage) -> _Usage:
        return _Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


def parse_mimo_sessions(
    exports: Sequence[Mapping[str, Any]], run: _SubprocessResult, *, cost_of: CostOf,
) -> AgentResult:
    """Every ``mimo export`` document of one run home; *cost_of* prices (uncached input, output, cached input) tokens in dollars."""
    if not exports:
        return AgentResult(
            success=False,
            output=f'{_exit_line(run)} and left no session in its store',
            subtype='error_no_session',
            cost_basis=ProcessCostBasis.USAGE_ESTIMATE,
            duration_ms=run.duration_ms,
            stderr=run.stderr,
            timed_out=run.timed_out,
        )
    steps = [message['info'] for export in exports for message in _assistant_messages(export)]
    usage = sum((_Usage.of_step(step) for step in steps), _Usage())
    root = next((export for export in exports if not export['info'].get('parentID')), exports[0])
    terminal = next(reversed(_assistant_messages(root)), None)
    terminal_info = terminal['info'] if terminal else {}
    error = terminal_info.get('error')
    text = _text_of(terminal) if terminal else ''
    success = (
        run.returncode == 0 and not run.timed_out
        and error is None and terminal_info.get('finish') == 'stop'
    )
    return AgentResult(
        success=success,
        output=text if success else (_error_message(error) or text or _unfinished(run, terminal_info)),
        cost_usd=cost_of(usage.input + usage.cache_write, usage.output, usage.cache_read),
        cost_basis=ProcessCostBasis.USAGE_ESTIMATE,
        duration_ms=run.duration_ms,
        turns=len(steps),
        session_id=str(root['info'].get('id') or ''),
        subtype='success' if success else 'error',
        stderr=run.stderr,
        timed_out=run.timed_out,
        api_error_status=_api_error_status(error),
        input_tokens=usage.input,
        output_tokens=usage.output,
        cache_read_tokens=usage.cache_read,
        cache_create_tokens=usage.cache_write,
    )


def _assistant_messages(export: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message for message in export.get('messages') or []
        if (message.get('info') or {}).get('role') == 'assistant'
    ]


def _text_of(message: Mapping[str, Any]) -> str:
    return '\n'.join(
        part['text'] for part in message.get('parts') or []
        if part.get('type') == 'text' and part.get('text')
    )


def _error_message(error: Mapping[str, Any] | None) -> str:
    if not error:
        return ''
    data = error.get('data') or {}
    return f'{error.get("name") or "error"}: {data.get("message") or ""}'.strip()


def _api_error_status(error: Mapping[str, Any] | None) -> int | None:
    status = ((error or {}).get('data') or {}).get('statusCode')
    return status if isinstance(status, int) else None


def _exit_line(run: _SubprocessResult) -> str:
    return f'mimo exited rc={run.returncode}' + (' (timed out)' if run.timed_out else '')


def _unfinished(run: _SubprocessResult, terminal_info: Mapping[str, Any]) -> str:
    return f'{_exit_line(run)} with its last step finished {terminal_info.get("finish")!r}'
