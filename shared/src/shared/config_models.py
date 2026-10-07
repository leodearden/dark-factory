"""Shared configuration models used across dark-factory subsystems."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

__all__ = [
    'AccountConfig',
    'UsageCapConfig',
]


class AccountConfig(BaseModel):
    """A Claude Max account for failover."""

    name: str = Field(description='Human-readable account label')
    oauth_token_env: str = Field(
        description='Env var holding the long-lived OAuth token for this account'
    )


class UsageCapConfig(BaseModel):
    """Usage cap detection and handling."""

    @model_validator(mode='before')
    @classmethod
    def _reject_legacy_pause_threshold(cls, data: object) -> object:
        """Detect the removed ``pause_threshold`` key and raise a clear error.

        The proactive "pause at N% of quota" path that consumed this field was
        removed when the claude.ai usage API became unavailable.  Cap detection
        is now entirely reactive — usage limits surface via stderr pattern
        matching (see UsageGate.check_at_startup).  The field no longer maps to
        any runtime behaviour.

        Without this validator, pydantic's ``extra='ignore'`` default would
        silently drop a ``pause_threshold:`` key from an operator's YAML,
        leaving them with no indication that their tuning had no effect.

        .. note::
            **Breaking change for operators**: any config file (orchestrator YAML,
            fused-memory ``config.yaml``, or a shared accounts file) that still
            contains ``usage_cap.pause_threshold`` will fail to load after this
            validator is present.  Remove the key before deploying.
        """
        if isinstance(data, dict) and 'pause_threshold' in data:
            raise ValueError(
                "UsageCapConfig: 'pause_threshold' is no longer a valid field. "
                "The proactive 'pause at N% of quota' path was removed when the "
                'claude.ai usage API became unavailable; cap detection is now '
                'reactive via stderr pattern matching (see UsageGate.check_at_startup). '
                "Remove 'usage_cap.pause_threshold' from your config — it would "
                "otherwise be silently ignored (extra='ignore')."
            )
        return data

    enabled: bool = Field(default=True)
    session_budget_usd: float | None = Field(default=None)
    wait_for_reset: bool = Field(default=True)
    probe_interval_secs: int = Field(default=300)
    max_probe_interval_secs: int = Field(default=1800)
    auth_reprobe_secs: int = Field(
        default=3600,
        description='Seconds between auth re-probes for auth_failed accounts.',
    )
    auth_reprobe_enabled: bool = Field(
        default=True,
        description=(
            'Re-probe an auth_failed account every auth_reprobe_secs (each '
            'attempt reloads .env with override). False keeps it out for the '
            'life of the gate, for a oneshot that must not reload .env or '
            'spend calls on a rejected token (consumer: scripts/legibility, '
            'task 6042).'
        ),
    )
    accounts: list[AccountConfig] = Field(default_factory=list)
    fallback_to_default_credential: bool = Field(
        default=True,
        description=(
            'When no configured account resolves a token, adopt the operator\'s '
            'own ~/.claude login as a single "default" account. False leaves the '
            'pool empty instead, so a caller never authenticates as that login '
            '(consumer: scripts/legibility, task 6042).'
        ),
    )
    accounts_file: str | None = Field(
        default=None,
        description='Path to shared YAML file with accounts list (overrides inline accounts)',
    )
    scoped_cap_models: list[str] = Field(
        default_factory=lambda: ['claude-fable-5'],
        description=(
            'Models tracked in a separate per-(account, model) cap scope; '
            '[] disables scoping entirely (kill switch).'
        ),
    )

    @model_validator(mode='after')
    def _load_accounts_file(self) -> UsageCapConfig:
        """Load accounts from external file when accounts_file is set."""
        if not self.accounts_file:
            return self
        path = Path(self.accounts_file)
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            import logging

            logging.getLogger(__name__).warning(
                f'accounts_file not found: {path} — using inline accounts'
            )
            return self
        data = yaml.safe_load(path.read_text())
        entries = data.get('accounts', [])
        self.accounts = [AccountConfig(**entry) for entry in entries]
        return self
