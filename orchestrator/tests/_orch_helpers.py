"""Non-fixture test helpers for orchestrator tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountState, UsageGate


def pydantic_spec(model: type[BaseModel]) -> type:
    """Return a proxy class exposing ``model``'s fields for ``MagicMock(spec_set=...)``.

    Pydantic v2 hides field names from ``dir()``, so passing a BaseModel subclass
    directly to ``spec_set=`` would only expose BaseModel/BaseSettings methods.
    The returned proxy has each field name as a class attribute, so MagicMock
    sees them and rejects typos on both get and set.
    """
    return type(
        f'_{model.__name__}Spec',
        (),
        {f: None for f in model.model_fields},
    )


def build_usage_gate(
    account_configs: list[AccountConfig],
    tokens: Sequence[str | None],
    *,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    """Create a UsageGate with tokens pre-injected (no os.environ lookup).

    Bypasses UsageGate.__init__ via __new__ and sets all private attrs directly,
    then injects AccountState entries from the parallel ``tokens`` list rather
    than reading from environment variables.  This is the canonical pattern for
    constructing test gates — both _make_gate (test_usage_gate.py) and
    _make_reify_gate (test_reify_multi_account.py) delegate to this helper.

    Parameters
    ----------
    account_configs:
        List of AccountConfig instances (same shape as UsageCapConfig.accounts).
    tokens:
        Parallel list of OAuth token strings (or None for default-credential
        accounts).  Must be the same length as ``account_configs``.
    wait_for_reset:
        Forwarded to UsageCapConfig.
    session_budget_usd:
        Forwarded to UsageCapConfig.
    probe_interval_secs:
        Forwarded to UsageCapConfig.
    max_probe_interval_secs:
        Forwarded to UsageCapConfig.

    Raises
    ------
    TypeError
        If ``tokens`` is a bare ``str`` instead of a list/tuple.
    ValueError
        If ``account_configs`` and ``tokens`` have different lengths.
    """
    if isinstance(tokens, str):
        raise TypeError(
            f'tokens must be a list/tuple of str|None, not a bare str; '
            f'got {tokens!r}. Did you forget to wrap it in a list?'
        )
    if len(account_configs) != len(tokens):
        raise ValueError(
            f'account_configs and tokens must have the same length; '
            f'got {len(account_configs)} account(s) and {len(tokens)} token(s)'
        )

    config = UsageCapConfig(
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
        accounts=account_configs,
    )

    gate = UsageGate.__new__(UsageGate)
    gate._config = config
    gate._open = asyncio.Event()
    gate._open.set()
    gate._lock = asyncio.Lock()
    gate._cumulative_cost = 0.0
    gate._paused_reason = ''
    gate._pause_started_at = None
    gate._total_pause_secs = 0.0
    gate._cost_store = None
    gate._project_id = None
    gate._run_id = None
    gate._last_account_name = None
    gate._background_tasks = set()
    gate._probe_config_dir = MagicMock()
    gate._run_probe = AsyncMock(return_value=True)
    gate._accounts = [
        AccountState(name=cfg.name, token=tok)
        for cfg, tok in zip(account_configs, tokens, strict=True)
    ]
    return gate
