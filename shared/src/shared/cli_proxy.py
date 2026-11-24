"""The host's CLI proxy (``cliproxyapi.service``) as a ``ClaudeEndpoint``.

The proxy's client file is the one ``~/.local/bin/claude-proxy`` sources: ``export`` lines setting the
two ``shared.secret_env.AGENT_ENDPOINT_VARS``. A host without that file has no proxy.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from shared.claude_endpoint import ClaudeEndpoint
from shared.secret_env import ENDPOINT_AUTH_TOKEN_VAR, ENDPOINT_BASE_URL_VAR

__all__ = ['CLIENT_ENV_PATH', 'ProxyClientFileInvalid', 'proxy_endpoint']

CLIENT_ENV_PATH = Path.home() / '.config' / 'cliproxyapi' / 'client.env'


class ProxyClientFileInvalid(ValueError):
    """The proxy's client file exists but does not set both endpoint variables."""


def proxy_endpoint(client_env: Path = CLIENT_ENV_PATH) -> ClaudeEndpoint | None:
    """The proxy named by *client_env*; None when the file does not exist."""
    if not client_env.is_file():
        return None
    values = dotenv_values(client_env)
    missing = [name for name in (ENDPOINT_BASE_URL_VAR, ENDPOINT_AUTH_TOKEN_VAR) if not values.get(name)]
    if missing:
        raise ProxyClientFileInvalid(f'{client_env} sets no {" and no ".join(missing)}')
    return ClaudeEndpoint(
        base_url=str(values[ENDPOINT_BASE_URL_VAR]), auth_token=str(values[ENDPOINT_AUTH_TOKEN_VAR]),
    )
