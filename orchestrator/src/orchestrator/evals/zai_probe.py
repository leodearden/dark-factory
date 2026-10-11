"""One cheap startup assertion that the configured Z.ai key answers on the GLM
Coding Plan's OpenAI-protocol coding endpoint.

Whoever opens Z.ai eval cells calls ``require_zai_coding_endpoint`` ONCE,
before the first cell (the production caller is a follow-up for ε2, task
5388). ``claude_endpoint_candidates()`` and ``get_config_by_name`` deliberately
never call it: they are offline by-name lookups.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
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


class ZaiCodingPlanUnavailable(RuntimeError):
    """The Z.ai coding endpoint is unusable with the configured key; ``outcome`` is None
    when no probe was sent because the key is missing.

    Raised once, at startup: without it every GLM eval cell 4xxs individually and reads
    as a model failure rather than as one configuration failure (INV-11).
    """

    def __init__(self, message: str, *, outcome: ZaiProbeOutcome | None) -> None:
        super().__init__(message)
        self.outcome = outcome


def _probe(client: httpx.Client, auth_token: str, *, timeout: float) -> ZaiProbeOutcome:
    try:
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
    except (httpx.HTTPError, OSError) as exc:
        return ZaiProbeOutcome(ZAI_CODING_BASE_URL, None, f'{type(exc).__name__}: {exc}')
    detail = f'HTTP {response.status_code}: {response.text[:200]}'
    return ZaiProbeOutcome(ZAI_CODING_BASE_URL, response.status_code, detail)


def _unavailable_message(reason: str) -> str:
    return (
        f'Z.ai coding endpoint {ZAI_CODING_BASE_URL} is unavailable ({reason}); it requires '
        f'a GLM Coding Plan key in {GLM_AUTH_TOKEN_ENV} (a general API key errors on it).'
    )


def require_zai_coding_endpoint(
    *,
    auth_token: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> ZaiProbeOutcome:
    """Probe once; raise ``ZaiCodingPlanUnavailable`` unless the key gets a 2xx answer."""
    token = os.environ.get(GLM_AUTH_TOKEN_ENV, '') if auth_token is None else auth_token
    if not token:
        reason = f'{GLM_AUTH_TOKEN_ENV} is unset or empty'
        raise ZaiCodingPlanUnavailable(_unavailable_message(reason), outcome=None)
    with httpx.Client() if client is None else nullcontext(client) as session:
        outcome = _probe(session, token, timeout=timeout)
    if not outcome.ok:
        raise ZaiCodingPlanUnavailable(_unavailable_message(outcome.detail), outcome=outcome)
    return outcome
