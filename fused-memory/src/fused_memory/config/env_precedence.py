"""Loud reporting for config-vs-environment OpenAI endpoint disagreements.

Both the graphiti LLM path and both mem0 paths now pass the configured
``providers.openai.api_url`` explicitly, which makes YAML authoritative over
the ambient ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` fallback the openai SDK
and mem0 would otherwise apply. That is the intended behaviour — an endpoint
the operator wrote down should win over an inherited environment variable —
but it is NOT a no-op for one specific existing deployment shape: a site that
routes OpenAI traffic through a gateway by exporting ``OPENAI_BASE_URL``
*without* setting ``OPENAI_API_URL`` used to reach the gateway and now reaches
``api.openai.com`` instead. That is an egress and billing change, so it must
not happen quietly (docs/legibility/design-invariants.md, no-silent-fail-soft).

Placed under ``config/`` rather than ``backends/`` deliberately: it is about
config-vs-environment precedence, it has no third-party imports, and both
``backends/graphiti_client.py`` and ``backends/mem0_client.py`` call it —
routing one through the other would create a needless import edge between the
graphiti and mem0 backends.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The vars the openai SDK and mem0 consult when no base_url is passed.
# OPENAI_API_BASE is mem0's deprecated spelling; both are checked because
# either one silently redirected traffic before this became explicit.
_AMBIENT_BASE_URL_VARS = ('OPENAI_BASE_URL', 'OPENAI_API_BASE')

# One warning per distinct (context, env var, env value, configured value).
# _build_config_dict runs once per project instance, so an unguarded warning
# would repeat for every project on a busy server and read as a new incident
# each time.
_warned: set[tuple[str, str, str, str]] = set()


def warn_if_ambient_base_url_is_overridden(configured_url: str | None, *, context: str) -> None:
    """Warn when an ambient base-URL env var disagrees with the configured one.

    ``context`` names the client being built (e.g. ``'graphiti LLM'``), so the
    log line says which traffic is affected.

    Silent when the vars are unset, when they agree with the config, or when
    nothing was configured — the only reportable case is a genuine
    disagreement, where the config value is about to win and redirect traffic
    away from wherever the environment was pointing it.
    """
    if not configured_url:
        return
    for var in _AMBIENT_BASE_URL_VARS:
        ambient = os.environ.get(var)
        if not ambient or ambient.rstrip('/') == configured_url.rstrip('/'):
            continue
        key = (context, var, ambient, configured_url)
        if key in _warned:
            continue
        _warned.add(key)
        logger.warning(
            f'{context}: {var}={ambient} is set but the configured endpoint '
            f'({configured_url}) takes precedence, so traffic goes to '
            f'{configured_url}. Config is authoritative over the environment '
            f'here. If {ambient} is the intended endpoint (e.g. a gateway or '
            f'proxy), set OPENAI_API_URL={ambient} or write it into the '
            'provider block of the fused-memory config instead.'
        )


def reset_warning_cache() -> None:
    """Clear the warn-once cache. For tests only."""
    _warned.clear()
