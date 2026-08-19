"""Shared test kit for the model-scoped usage-cap suites.

Canonical home for the usage-gate test helpers that were previously copied
byte-for-byte across ``test_usage_gate_scope_selection.py`` and
``test_usage_gate_scope_boundary.py`` (make_gate / make_mock_cost_store /
set_scope_cap), plus the full-loop fakes that lived only in the boundary
file (scripted_invoke / cap_result / ok_result). Consumers import these by
bare module name -- ``conftest.py`` prepends ``shared/tests`` to
``sys.path`` (same convention as ``silent_fallthrough_scan.py``). The
leading underscore keeps pytest from collecting this module as a test file.

``make_gate`` here defaults ``wait_for_reset`` to ``False`` (the scope-suite
semantics). ``test_cap_retry.py`` wraps this helper to restore its
historical ``wait_for_reset=True`` default -- see that file.
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import ScopeCap, UsageGate

SCOPE = 'claude-fable-5'


def make_gate(account_names: list[str], *, cost_store=None, **cfg) -> UsageGate:
    """Create a real UsageGate with fake accounts, probe + reset-wait disabled."""
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    cfg.setdefault('wait_for_reset', False)
    config = UsageCapConfig(accounts=acct_cfgs, **cfg)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)
    # Mock _run_probe to prevent real subprocess spawning.
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_mock_cost_store() -> AsyncMock:
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    store.save_invocation = AsyncMock(return_value=None)
    return store


def set_scope_cap(
    acct,
    scope: str = SCOPE,
    *,
    capped: bool = True,
    resets_at: datetime | None = None,
    capped_at: datetime | None = None,
    near_cap: bool = False,
) -> ScopeCap:
    """Directly install a ScopeCap overlay on *acct* (bypassing the β handler)."""
    sc = ScopeCap(
        capped=capped, resets_at=resets_at, near_cap=near_cap, capped_at=capped_at,
    )
    acct.scope_caps[scope] = sc
    return sc


class scripted_invoke:
    """Async fake ``invoke_fn`` yielding each of *results* in call order
    (the last result repeats if called more times than results supplied).

    Tolerates arbitrary ``**kwargs`` (``prompt``/``model``/``oauth_token``/
    ``config_dir``/``backend`` -- everything ``invoke_with_cap_retry``
    forwards to ``invoke_fn``) and records every call's kwargs onto
    ``self.calls`` for assertions (e.g. B8's backend-forwarding check).
    """

    def __init__(self, *results: AgentResult) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx]


def cap_result(stderr: str, output: str = 'partial') -> AgentResult:
    """A capped AgentResult carrying a cap-hit stderr pattern.

    ``success=True`` mirrors ``test_cap_retry.py::test_real_gate_failover``'s
    ``capped`` fixture: cap detection happens via ``slot.detect_cap_hit``,
    which builds its OWN synthetic ``success=False`` result internally
    (``UsageGate.detect_cap_hit``), so the raw result's ``success`` value is
    irrelevant to cap classification here.
    """
    return AgentResult(success=True, output=output, stderr=stderr, cost_usd=0.5)


def ok_result(output: str = 'complete') -> AgentResult:
    """A successful AgentResult (mirrors make_result in test_cap_retry.py)."""
    return AgentResult(success=True, output=output, cost_usd=0.5)
