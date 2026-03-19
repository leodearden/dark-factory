"""Shared configuration models used across dark-factory subsystems."""

from pydantic import BaseModel, Field


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
    accounts: list[AccountConfig] = Field(default_factory=list)
