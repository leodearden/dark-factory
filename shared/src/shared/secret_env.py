"""Which environment variables carry secrets, and which of them an agent subprocess may inherit.

A variable is secret-bearing when its name starts with the OAuth pool prefix or
ends in ``_API_KEY`` or ``_TOKEN``; the rule reads names only, never values. An
agent child inherits no secret-bearing variable except its own session
credential, ``CLAUDE_CODE_OAUTH_TOKEN``. An agent child also inherits no
Python-environment selection (``shared.python_env``). Operator view:
``OPERATIONS.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

from shared.python_env import without_python_env_selection

__all__ = [
    'AGENT_CREDENTIAL_VAR',
    'POOL_TOKEN_PREFIX',
    'agent_subprocess_env',
    'is_secret_env_name',
    'secret_env_values',
]

POOL_TOKEN_PREFIX = 'CLAUDE_OAUTH_TOKEN_'
AGENT_CREDENTIAL_VAR = 'CLAUDE_CODE_OAUTH_TOKEN'

_SECRET_SUFFIXES = ('_API_KEY', '_TOKEN')


def is_secret_env_name(name: str) -> bool:
    return name.startswith(POOL_TOKEN_PREFIX) or name.endswith(_SECRET_SUFFIXES)


def agent_subprocess_env(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in without_python_env_selection(environ).items()
        if name == AGENT_CREDENTIAL_VAR or not is_secret_env_name(name)
    }


def secret_env_values(environ: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in environ.items() if value and is_secret_env_name(name)}
