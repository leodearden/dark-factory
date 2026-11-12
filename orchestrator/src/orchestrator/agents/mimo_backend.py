"""The MiMo Code CLI as an agent backend: the config document, argv, session-store read and result parsing of one headless ``mimo run``.

The measured evidence behind every shape here is plans/mimo-code-harness-probe-2026-11.md.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
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
CONFIG_SCHEMA = 'https://mimo.xiaomi.com/mimocode/config.json'
PROVIDER_ID = 'mimo-token-plan'
MODEL_LIMITS = {'context': 1048576, 'output': 131072}
TITLE = 'dark-factory'
MIMO_VARIANTS = ('low', 'medium', 'high')
FALLBACK_VARIANT = 'high'
SESSION_READ_TIMEOUT_SECS = 120
MIMOCODE_ENV_PREFIX = 'MIMOCODE_'
MIMOCODE_FLAGS = {
    'MIMOCODE_ENABLE_ANALYSIS': 'false',
    'MIMOCODE_DISABLE_CLAUDE_CODE_MCP': '1',
    'MIMOCODE_DISABLE_MODELS_FETCH': '1',
    'MIMOCODE_DISABLE_AUTOUPDATE': '1',
    'MIMOCODE_DISABLE_CRON': '1',
    'MIMOCODE_DISABLE_SHARE': '1',
}
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
SessionSpawn = Callable[[list[str], float], Awaitable[_SubprocessResult]]


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


@dataclass(frozen=True)
class MimoRunHome:
    root: Path
    config_path: Path
    operator_home: Path


@contextmanager
def mimo_run_home(
    *, system_prompt: str, config_document: Mapping[str, Any], operator_home: Path,
) -> Iterator[MimoRunHome]:
    """A private MIMOCODE_HOME for one invocation, holding the only copy of the endpoint key; removed on exit."""
    with private_run_home(backend='mimo', operator_home=operator_home) as root:
        config_path = root / 'config' / 'mimocode.json'
        write_owner_only_json(config_path, config_document)
        (root / 'config' / 'AGENTS.md').write_text(system_prompt)
        yield MimoRunHome(root=root, config_path=config_path, operator_home=operator_home)


def mimo_spawn_env(base_env: Mapping[str, str], home: MimoRunHome) -> dict[str, str]:
    """The child env; no inherited MIMOCODE_* variable reaches mimo, so none can add MCP servers or relax a deny."""
    redirected = redirected_home_env(base_env, root=home.root, operator_home=home.operator_home)
    return {
        **{name: value for name, value in redirected.items() if not name.startswith(MIMOCODE_ENV_PREFIX)},
        'MIMOCODE_HOME': str(home.root),
        **MIMOCODE_FLAGS,
    }


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


async def read_mimo_sessions(spawn: SessionSpawn) -> list[dict[str, Any]]:
    """Every session the run home's store holds, as ``mimo export`` documents, in list order.

    *spawn* runs one argv under a timeout; each read gets SESSION_READ_TIMEOUT_SECS of its own, outside the run's wall clock.
    """
    listing = await spawn(mimo_session_list_argv(), SESSION_READ_TIMEOUT_SECS)
    sessions = _decoded(listing)
    if not isinstance(sessions, list):
        logger.warning(
            'mimo backend: the session list is unusable (rc=%d), so no usage is read: %.200r',
            listing.returncode, listing.stdout,
        )
        return []
    exports: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session.get('id') if isinstance(session, dict) else None
        if not session_id:
            continue
        exported = await spawn(mimo_export_argv(str(session_id)), SESSION_READ_TIMEOUT_SECS)
        document = _decoded(exported)
        if isinstance(document, dict):
            exports.append(document)
        else:
            logger.warning(
                'mimo backend: skipped session %s, whose export is unusable (rc=%d)',
                session_id, exported.returncode,
            )
    return exports


def _decoded(result: _SubprocessResult) -> Any:
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


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
    exports: Sequence[Mapping[str, Any]], run: _SubprocessResult, *, max_turns: int, cost_of: CostOf,
) -> AgentResult:
    """Every ``mimo export`` document of one run home; *cost_of* prices (uncached input, output, cached input) tokens in dollars.

    The root session is the one the run's stdout names or, when stdout names none, the one session without a parent.
    """
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
    named_id = _stdout_session_id(run.stdout)
    root = _root_session(exports, named_id)
    root_steps = _assistant_messages(root) if root is not None else []
    terminal = root_steps[-1] if root_steps else {}
    terminal_info = terminal.get('info') or {}
    error = terminal_info.get('error')
    text = _text_of(terminal)
    subtype = _subtype(run, root, len(root_steps), terminal_info, max_turns)
    success = subtype == 'success'
    return AgentResult(
        success=success,
        output=text if success else _failure_output(run, subtype, text, terminal_info, max_turns),
        cost_usd=cost_of(usage.input + usage.cache_write, usage.output, usage.cache_read),
        cost_basis=ProcessCostBasis.USAGE_ESTIMATE,
        duration_ms=run.duration_ms,
        turns=len(steps),
        session_id=named_id or str(_session_info(root or {}).get('id') or ''),
        subtype=subtype,
        stderr=run.stderr,
        timed_out=run.timed_out,
        api_error_status=_api_error_status(error),
        input_tokens=usage.input,
        output_tokens=usage.output,
        cache_read_tokens=usage.cache_read,
        cache_create_tokens=usage.cache_write,
    )


def _stdout_session_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = event.get('sessionID') if isinstance(event, dict) else None
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _root_session(
    exports: Sequence[Mapping[str, Any]], named_id: str | None,
) -> Mapping[str, Any] | None:
    if named_id:
        matches = [export for export in exports if _session_info(export).get('id') == named_id]
        rule = f'the id {named_id!r} its stdout names'
    else:
        matches = [export for export in exports if not _session_info(export).get('parentID')]
        rule = 'having no parent (its stdout names no session)'
    if len(matches) == 1:
        return matches[0]
    logger.warning(
        'mimo backend: %d of the stored sessions [%s] match %s; the run needs exactly one '
        'root session, so it fails',
        len(matches), ', '.join(repr(_session_info(export).get('id')) for export in exports), rule,
    )
    return None


def _subtype(
    run: _SubprocessResult,
    root: Mapping[str, Any] | None,
    root_step_count: int,
    terminal_info: Mapping[str, Any],
    max_turns: int,
) -> str:
    if root is None:
        return 'error_no_root_session'
    if root_step_count >= max_turns:
        return 'error_max_turns'
    finished = terminal_info.get('error') is None and terminal_info.get('finish') == 'stop'
    return 'success' if run.returncode == 0 and not run.timed_out and finished else 'error'


def _failure_output(
    run: _SubprocessResult,
    subtype: str,
    text: str,
    terminal_info: Mapping[str, Any],
    max_turns: int,
) -> str:
    if subtype == 'error_no_root_session':
        return f"{_exit_line(run)} and its store holds no session identifiable as the run's root"
    if subtype == 'error_max_turns':
        return f'{_exit_line(run)} at its {max_turns}-step cap' + (f': {text}' if text else '')
    return _error_message(terminal_info.get('error')) or text or _unfinished(run, terminal_info)


def _session_info(export: Mapping[str, Any]) -> Mapping[str, Any]:
    return export.get('info') or {}


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
