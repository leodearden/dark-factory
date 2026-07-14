"""Model routing: allowlist + fail-fast validation + per-account availability probe.

Task beta (Phase-1 substrate of plans/adaptive-model-routing-prd.md). This
module is the PRD-named "allowlist home": ``DEFAULT_ALLOWED_MODELS`` is the
source of truth for ``OrchestratorConfig.routing``'s default (see
``config.py``'s ``RoutingConfig`` submodel and its
``_validate_models_in_allowlist`` cross-field validator).

This module will also host the per-account model-availability probe
(``probe_models``) and its rendered artifact format
(``render_probe_artifact``), consumed by the ``orchestrator probe-models`` CLI
subcommand (``orchestrator/cli.py``) — added by later steps in this task's
plan, not yet present here.

Kept import-light at module top (stdlib only) so ``config.py`` can
``from orchestrator.routing import DEFAULT_ALLOWED_MODELS`` with no circular
import; heavier imports (e.g. ``shared.cli_invoke.invoke_claude_agent``) are
deferred to inside the functions that need them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from shared.cli_invoke import AgentResult
    from shared.config_models import AccountConfig

# Claude-backend model aliases admitted by default (task beta). claude-fable-5
# is deliberately NOT included here — task xi admits it to the runtime
# allowlist once probe_models confirms availability across every pool account
# (see FABLE_CANDIDATE_MODEL below).
DEFAULT_ALLOWED_MODELS: tuple[str, ...] = ('haiku', 'sonnet', 'opus')

# Candidate model probed for availability even though it is not yet admitted
# to the runtime allowlist — beta is the G3 gate that produces the
# per-account fable-availability data task xi's admission gate consumes.
FABLE_CANDIDATE_MODEL: str = 'claude-fable-5'

# Default path for the committed probe-models artifact, sibling of
# config/usage-accounts.yaml (the account-pool source of truth).
DEFAULT_PROBE_ARTIFACT_PATH: str = 'config/model-availability.yaml'

# Cheap 1-turn prompt used by probe_models's default invocation -- just
# enough to confirm the model string resolves and the account can complete a
# turn, without incurring meaningful cost.
DEFAULT_PROBE_PROMPT: str = 'Reply with the single word: ok'


def _dedup_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate *items*, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@dataclass
class ProbeReport:
    """Result of a probe_models run: the target model set actually probed,
    plus per-account x per-model availability status (see
    ``classify_probe_outcome`` for the status vocabulary)."""

    models: list[str]
    accounts: dict[str, dict[str, str]]


def classify_probe_outcome(outcome: object) -> str:
    """Map a ``shared.invocation_outcome.InvocationOutcome`` to a probe
    status string: OK->available, ModelNotFound->unavailable,
    AuthFailed->auth_error, CapHit/NearCap->capped, else->error.

    Pure -- reads only *outcome*, performs no I/O.
    """
    from shared.invocation_outcome import OK, AuthFailed, CapHit, ModelNotFound, NearCap

    if isinstance(outcome, OK):
        return 'available'
    if isinstance(outcome, ModelNotFound):
        return 'unavailable'
    if isinstance(outcome, AuthFailed):
        return 'auth_error'
    if isinstance(outcome, (CapHit, NearCap)):
        return 'capped'
    return 'error'


async def probe_models(
    accounts: list[AccountConfig],
    allowed_models: list[str],
    *,
    models: list[str] | None = None,
    invoke_fn: Callable[..., Awaitable[AgentResult]] | None = None,
    token_resolver: Callable[[str], str | None] = os.environ.get,
    prompt: str = DEFAULT_PROBE_PROMPT,
    cwd: Path | None = None,
    max_turns: int = 1,
    budget_usd: float = 0.05,
) -> ProbeReport:
    """Probe every (account, model) pair for availability.

    The target model set defaults to ``dedup(allowed_models +
    [FABLE_CANDIDATE_MODEL])``, order-preserving, so the probe always
    exercises ``claude-fable-5`` even though it is not yet admitted to the
    runtime allowlist (task xi's G3 gate; see this task's plan
    design_decisions) -- pass *models* explicitly to override.

    For each account, the OAuth token is resolved once via
    ``token_resolver(account.oauth_token_env)``. When unresolvable, every
    target model is recorded as ``'no_token'`` for that account and
    *invoke_fn* is never called for it. Otherwise, *invoke_fn* (default
    ``invoke_claude_agent``) is called once per target model with a cheap
    1-turn invocation, and the result is classified via
    ``classify_invocation`` / ``classify_probe_outcome`` into a status
    string.

    *invoke_fn* and *token_resolver* are dependency-injected (mirrors
    ``invoke_with_cap_retry``'s ``invoke_fn=`` seam) so callers can drive
    this network-free in tests.
    """
    from pathlib import Path as _Path

    from shared.cli_invoke import invoke_claude_agent
    from shared.invocation_outcome import classify_invocation

    invoke = invoke_fn or invoke_claude_agent
    target_models = (
        list(models)
        if models is not None
        else _dedup_preserve_order([*allowed_models, FABLE_CANDIDATE_MODEL])
    )
    resolved_cwd = cwd or _Path.cwd()

    report_accounts: dict[str, dict[str, str]] = {}
    for account in accounts:
        token = token_resolver(account.oauth_token_env)
        if not token:
            report_accounts[account.name] = dict.fromkeys(target_models, 'no_token')
            continue

        statuses: dict[str, str] = {}
        for model in target_models:
            result = await invoke(
                prompt=prompt,
                system_prompt='',
                cwd=resolved_cwd,
                model=model,
                max_turns=max_turns,
                max_budget_usd=budget_usd,
                oauth_token=token,
            )
            outcome = classify_invocation(result, strict_confirm=False, backend='claude')
            statuses[model] = classify_probe_outcome(outcome)
        report_accounts[account.name] = statuses

    return ProbeReport(models=target_models, accounts=report_accounts)
