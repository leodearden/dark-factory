"""One cheap startup assertion that the configured Z.ai key answers on the GLM
Coding Plan's OpenAI-protocol coding endpoint.

Whoever opens Z.ai eval cells calls ``require_zai_coding_endpoint`` ONCE,
before the first cell (the production caller is a follow-up for ε2, task
5388). ``claude_endpoint_candidates()`` and ``get_config_by_name`` deliberately
never call it: they are offline by-name lookups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from orchestrator.evals.configs import GLM_AUTH_TOKEN_ENV, GLM_FLASH_MODEL, ZAI_CODING_BASE_URL


@dataclass(frozen=True)
class ZaiProbeOutcome:
    """What the coding endpoint answered; ``status_code`` is None when nothing answered."""

    base_url: str
    status_code: int | None
    detail: str

    @property
    def ok(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300


def _probe(client: httpx.Client, auth_token: str, *, timeout: float) -> ZaiProbeOutcome:
    response = client.post(
        f'{ZAI_CODING_BASE_URL}/chat/completions',
        headers={'Authorization': f'Bearer {auth_token}'},
        json={
            'model': GLM_FLASH_MODEL,
            'messages': [{'role': 'user', 'content': 'ping'}],
            'max_tokens': 1,
        },
        timeout=timeout,
    )
    return ZaiProbeOutcome(ZAI_CODING_BASE_URL, response.status_code, f'HTTP {response.status_code}')


def require_zai_coding_endpoint(
    *,
    auth_token: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> ZaiProbeOutcome:
    token = os.environ.get(GLM_AUTH_TOKEN_ENV, '') if auth_token is None else auth_token
    if client is not None:
        return _probe(client, token, timeout=timeout)
    with httpx.Client() as own_client:
        return _probe(own_client, token, timeout=timeout)
