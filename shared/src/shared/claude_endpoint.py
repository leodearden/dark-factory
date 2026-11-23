"""An Anthropic-protocol endpoint a ``claude`` child can be sent to in place of a Claude login."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from shared.secret_env import ENDPOINT_AUTH_TOKEN_VAR, ENDPOINT_BASE_URL_VAR

__all__ = ['ClaudeEndpoint']


@dataclass(frozen=True)
class ClaudeEndpoint:
    """The endpoint's http(s) URL and the client key it takes; the repr leaves the key out."""

    base_url: str
    auth_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if urlparse(self.base_url).scheme not in ('http', 'https'):
            raise ValueError(f'an endpoint base_url must be an http(s) URL, got {self.base_url!r}')
        if not self.auth_token:
            raise ValueError('an endpoint needs a non-empty auth_token')

    def env(self) -> dict[str, str]:
        return {ENDPOINT_BASE_URL_VAR: self.base_url, ENDPOINT_AUTH_TOKEN_VAR: self.auth_token}
