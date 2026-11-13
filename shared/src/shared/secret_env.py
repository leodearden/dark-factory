"""Which environment variables carry secrets, and which of them an agent subprocess may inherit.

A variable is secret-bearing when its name starts with the OAuth pool prefix or
ends in ``_API_KEY`` or ``_TOKEN``; the rule reads names only, never values. An
agent child inherits no secret-bearing variable except its own session
credential, ``CLAUDE_CODE_OAUTH_TOKEN``. An agent child also inherits no
endpoint-selection variable (``AGENT_ENDPOINT_VARS``: the endpoint and its
client key) and no Python-environment selection (``shared.python_env``); an
endpoint reaches a child only through its caller's overrides.
``without_agent_endpoint_vars`` is those two scrubs alone, for a child that
needs its inherited credentials. Operator view: ``OPERATIONS.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

from shared.python_env import without_python_env_selection

__all__ = [
    'AGENT_CREDENTIAL_VAR',
    'AGENT_ENDPOINT_VARS',
    'POOL_TOKEN_PREFIX',
    'agent_subprocess_env',
    'is_secret_env_name',
    'secret_env_values',
    'without_agent_endpoint_vars',
]

POOL_TOKEN_PREFIX = 'CLAUDE_OAUTH_TOKEN_'
AGENT_CREDENTIAL_VAR = 'CLAUDE_CODE_OAUTH_TOKEN'
AGENT_ENDPOINT_VARS = frozenset({'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'})

_SECRET_SUFFIXES = ('_API_KEY', '_TOKEN')


def is_secret_env_name(name: str) -> bool:
    return name.startswith(POOL_TOKEN_PREFIX) or name.endswith(_SECRET_SUFFIXES)


def without_agent_endpoint_vars(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in without_python_env_selection(environ).items()
        if name not in AGENT_ENDPOINT_VARS
    }


def agent_subprocess_env(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in without_agent_endpoint_vars(environ).items()
        if name == AGENT_CREDENTIAL_VAR or not is_secret_env_name(name)
    }


def secret_env_values(environ: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in environ.items() if value and is_secret_env_name(name)}
