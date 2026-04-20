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

    enabled: bool = Field(default=True)
    session_budget_usd: float | None = Field(default=None)
    pause_threshold: float = Field(default=0.96)
    wait_for_reset: bool = Field(default=True)
    probe_interval_secs: int = Field(default=300)
    max_probe_interval_secs: int = Field(default=1800)
    auth_reprobe_secs: int = Field(
        default=3600,
        description='Seconds between auth re-probes for auth_failed accounts.',
    )
    accounts: list[AccountConfig] = Field(default_factory=list)
    accounts_file: str | None = Field(
        default=None,
        description='Path to shared YAML file with accounts list (overrides inline accounts)',
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
