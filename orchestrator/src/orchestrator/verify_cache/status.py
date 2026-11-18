"""What `orchestrator verify-cache status` prints: each role's mode, the
registry's recorded greens by tier, and the kill switch."""

from __future__ import annotations

from datetime import datetime

from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache.registry import PolicyState, RegistryError, RegistrySummary
from orchestrator.verify_evidence import UnitTier


def render_status(modes: VerifyCacheConfig, summary: RegistrySummary | RegistryError, now: datetime) -> list[str]:
    """The report's lines; an unreadable registry claims no counts and no policy."""
    mode = f'mode: task={modes.task_mode} merge={modes.merge_mode}'
    if isinstance(summary, RegistryError):
        return [mode, f'registry: {summary.path} UNREADABLE — {summary.cause}']
    presence = 'present' if summary.exists else 'absent — cold start'
    counts = ' '.join(f'{tier}={summary.entries_by_tier.get(tier, 0)}' for tier in UnitTier)
    return [mode, f'registry: {summary.path} ({presence})', f'entries: {counts}', _policy_line(summary.policy, now)]


def _policy_line(policy: PolicyState, now: datetime) -> str:
    if policy.disabled_until is None or not policy.is_disabled(now):
        return f'policy: enabled (streak {policy.streak})'
    return (
        f'policy: DISABLED until {policy.disabled_until.isoformat()} — {policy.reason} '
        f'(streak {policy.streak}, leaks: {", ".join(policy.leaks)})'
    )
