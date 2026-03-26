"""Usage cap detection, pause gate, and auto-resume for OAuth-based Claude agents.

Supports multiple Claude Max accounts for failover: when one account hits its
usage cap, the gate returns the next available account's token. Only blocks
when *all* accounts are capped.

Works with 1 or N accounts — there is no separate "single-account" path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from shared.config_models import UsageCapConfig

logger = logging.getLogger(__name__)

__all__ = [
    'UsageGate',
    'AccountState',
    'SessionBudgetExhausted',
]

# Patterns that indicate a usage cap has been hit (from Claude Code CLI output)
CAP_HIT_PREFIXES = [
    "You've hit your",
    "You've used",
    "You're out of extra usage",
]
# Secondary confirmation — must also appear in the same text
CAP_CONFIRM_KEYWORDS = ["resets", "usage limit", "upgrade"]

# Patterns for near-cap warnings (pause proactively)
NEAR_CAP_PREFIXES = [
    "You're close to",
    "You're now using extra usage",
]

# Codex (OpenAI) cap-hit patterns
CODEX_CAP_PATTERNS = ['usage limit reached', 'rate limit', 'quota exceeded',
                      'insufficient_quota', 'rate_limit_exceeded']

# Gemini (Google) cap-hit patterns
GEMINI_CAP_PATTERNS = ['quota exceeded', 'rate limit', 'resource exhausted',
                       'RESOURCE_EXHAUSTED', 'quota_exceeded']

CREDENTIALS_PATH = Path.home() / '.claude' / '.credentials.json'
USAGE_API_URL = 'https://claude.ai/api/oauth/usage'

# Consecutive API failures required before tentative uncap (after resets_at)
_TENTATIVE_UNCAP_THRESHOLD = 2


@dataclass
class AccountState:
    """Per-account cap tracking."""

    name: str
    token: str | None          # None = default account (no override)
    capped: bool = False
    resets_at: datetime | None = None
    pause_started_at: datetime | None = None
    resume_task: asyncio.Task | None = field(default=None, repr=False)
    api_fail_count: int = 0    # consecutive _query_usage_api failures


class SessionBudgetExhausted(Exception):
    """Raised when the per-run session budget is exceeded."""

    def __init__(self, cumulative_cost: float):
        self.cumulative_cost = cumulative_cost
        super().__init__(f'Session budget exhausted: ${cumulative_cost:.2f} spent')


class UsageGate:
    """Shared gate that pauses all agent invocations when a usage cap is hit.

    Tracks cap status per account and returns the first available account's
    token from ``before_invoke()``. Only blocks when *all* accounts are
    capped.  Works with 1 or N accounts.
    """

    def __init__(self, config: UsageCapConfig):
        self._config: UsageCapConfig = config
        self._open = asyncio.Event()
        self._open.set()  # start open
        self._lock = asyncio.Lock()
        self._cumulative_cost: float = 0.0
        self._paused_reason: str = ''
        self._pause_started_at: datetime | None = None
        self._total_pause_secs: float = 0.0

        self._accounts: list[AccountState] = self._init_accounts()

    def _init_accounts(self) -> list[AccountState]:
        """Resolve account tokens from env vars.

        If no accounts are configured, falls back to reading the default
        credential from ``~/.claude/.credentials.json``.
        """
        accounts: list[AccountState] = []
        for acct_cfg in self._config.accounts:
            token = os.environ.get(acct_cfg.oauth_token_env)
            if not token:
                logger.warning(
                    f'Account {acct_cfg.name!r}: env var {acct_cfg.oauth_token_env} '
                    f'not set — skipping'
                )
                continue
            accounts.append(AccountState(name=acct_cfg.name, token=token))

        if not accounts:
            token = _read_oauth_token()
            if token:
                accounts.append(AccountState(name='default', token=token))
                logger.info('Single-account mode: using default credential')
            else:
                logger.warning('No accounts configured and no default credential found')

        if accounts:
            logger.info(
                f'Failover: {len(accounts)} account(s) active — '
                + ', '.join(a.name for a in accounts)
            )
        return accounts

    async def check_at_startup(self) -> None:
        """Query /api/oauth/usage for each account. Pause if all are over threshold."""
        for acct in self._accounts:
            usage = await self._query_usage_api(token=acct.token)
            if usage is None:
                logger.info(f'Account {acct.name}: usage API unavailable — assuming OK')
                continue

            for tier_name, tier_data in usage.items():
                if not isinstance(tier_data, dict):
                    continue
                utilization = tier_data.get('utilization', 0)
                resets_at_str = tier_data.get('resets_at')

                logger.info(
                    f'Account {acct.name} tier {tier_name}: {utilization:.0%} utilized'
                    + (f', resets at {resets_at_str}' if resets_at_str else '')
                )

                if utilization >= self._config.pause_threshold:
                    acct.capped = True
                    acct.resets_at = _parse_iso_timestamp(resets_at_str) if resets_at_str else None
                    acct.pause_started_at = datetime.now(UTC)
                    logger.warning(f'Account {acct.name}: capped at startup ({utilization:.0%})')
                    self._start_account_resume_probe(acct)
                    break  # one capped tier is enough

        # If ALL accounts are capped, close the global gate
        if self._accounts and all(a.capped for a in self._accounts):
            self._open.clear()
            self._paused_reason = f'All {len(self._accounts)} account(s) capped at startup'
            self._pause_started_at = datetime.now(UTC)
            logger.warning(f'Usage gate PAUSED: {self._paused_reason}')

    async def before_invoke(self) -> str | None:
        """Block until at least one account is available. Return its OAuth token.

        Returns ``None`` if no accounts are configured (no token override).
        """
        # Session budget check
        if (self._config.session_budget_usd is not None
                and self._cumulative_cost >= self._config.session_budget_usd):
            raise SessionBudgetExhausted(self._cumulative_cost)

        if not self._accounts:
            raise RuntimeError(
                'No OAuth accounts available — configure accounts or provide credentials'
            )

        # Find first non-capped account (works with 1 or N)
        while True:
            for acct in self._accounts:
                if not acct.capped:
                    logger.debug(f'Using account {acct.name}')
                    return acct.token

            # All capped — do a fresh API check before blocking.
            # Background probes may be sleeping through exponential backoff,
            # but one account may have already reset.
            refreshed = await self._refresh_capped_accounts()
            if refreshed:
                continue  # re-check accounts with updated flags

            # Still all capped after fresh check — wait on global gate
            logger.info('All accounts capped — waiting for any to reopen')
            self._open.clear()
            await self._open.wait()

    def detect_cap_hit(
        self,
        stderr: str,
        result_text: str,
        backend: str = 'claude',
        oauth_token: str | None = None,
    ) -> bool:
        """Scan stderr and result text for cap-hit patterns.

        Marks the specific account identified by ``oauth_token`` as capped
        and starts its resume probe. Returns True so the retry loop re-calls
        ``before_invoke()`` to pick the next available account.
        """
        combined = f'{stderr}\n{result_text}'

        # Check backend-specific patterns first
        if backend == 'codex':
            for pattern in CODEX_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    self._handle_cap_detected(
                        f'Codex cap hit: {pattern}', None, oauth_token,
                    )
                    return True
        elif backend == 'gemini':
            for pattern in GEMINI_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    self._handle_cap_detected(
                        f'Gemini cap hit: {pattern}', None, oauth_token,
                    )
                    return True

        for prefixes in (CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES):
            for prefix in prefixes:
                if prefix.lower() in combined.lower():
                    resets_at = _parse_resets_at(combined)
                    reason = _extract_cap_message(combined, prefix) or f'Cap detected: {prefix}'
                    self._handle_cap_detected(reason, resets_at, oauth_token)
                    return True

        return False

    def _handle_cap_detected(
        self,
        reason: str,
        resets_at: datetime | None,
        oauth_token: str | None,
    ) -> None:
        """Mark the matching account as capped."""
        acct = self._find_account_by_token(oauth_token) if oauth_token else None
        if acct is None:
            # Unknown token — try first uncapped account as best guess
            for a in self._accounts:
                if not a.capped:
                    acct = a
                    break
        if acct is None:
            logger.warning(f'Cap detected but no matching account: {reason}')
            return

        acct.capped = True
        acct.resets_at = resets_at
        if acct.pause_started_at is None:
            acct.pause_started_at = datetime.now(UTC)
        logger.warning(f'Account {acct.name} CAPPED: {reason}')
        self._start_account_resume_probe(acct)

        # If all accounts are now capped, close the global gate
        if all(a.capped for a in self._accounts):
            self._open.clear()
            self._paused_reason = f'All accounts capped (last: {reason})'
            if self._pause_started_at is None:
                self._pause_started_at = datetime.now(UTC)
            logger.warning(f'Usage gate PAUSED: {self._paused_reason}')

    def _find_account_by_token(self, token: str) -> AccountState | None:
        for acct in self._accounts:
            if acct.token == token:
                return acct
        return None

    def _start_account_resume_probe(self, acct: AccountState) -> None:
        """Start an async resume probe for a specific account."""
        if not self._config.wait_for_reset:
            return
        try:
            loop = asyncio.get_running_loop()
            if acct.resume_task is None or acct.resume_task.done():
                acct.resume_task = loop.create_task(
                    self._account_resume_probe_loop(acct),
                    name=f'usage-gate-resume-{acct.name}',
                )
        except RuntimeError:
            pass

    async def _refresh_capped_accounts(self) -> bool:
        """Query the usage API for all capped accounts. Return True if any uncapped."""
        any_uncapped = False
        for acct in self._accounts:
            if not acct.capped:
                continue
            usage = await self._query_usage_api(token=acct.token)
            if usage is None:
                acct.api_fail_count += 1
                past_reset = (
                    acct.resets_at is not None
                    and datetime.now(UTC) > acct.resets_at
                )
                if past_reset and acct.api_fail_count >= _TENTATIVE_UNCAP_THRESHOLD:
                    logger.warning(
                        f'Account {acct.name}: usage API unavailable after '
                        f'expected reset — tentatively uncapping'
                    )
                    acct.capped = False
                    acct.pause_started_at = None
                    acct.api_fail_count = 0
                    any_uncapped = True
                else:
                    logger.warning(
                        f'Account {acct.name}: usage API unavailable during '
                        f'refresh — keeping capped (will retry via probe)'
                    )
                continue

            acct.api_fail_count = 0
            still_capped = False
            for _tier_name, tier_data in usage.items():
                if not isinstance(tier_data, dict):
                    continue
                utilization = tier_data.get('utilization', 0)
                if utilization >= self._config.pause_threshold:
                    still_capped = True
                    break

            if not still_capped:
                logger.info(f'Account {acct.name}: cap cleared on fresh check')
                acct.capped = False
                if acct.pause_started_at:
                    pause_duration = (
                        datetime.now(UTC) - acct.pause_started_at
                    ).total_seconds()
                    self._total_pause_secs += pause_duration
                acct.pause_started_at = None
                any_uncapped = True

        if any_uncapped:
            self._open.set()
        return any_uncapped

    def on_agent_complete(self, cost: float) -> None:
        """Accumulate cost for session budget tracking."""
        self._cumulative_cost += cost

    async def _account_resume_probe_loop(self, acct: AccountState) -> None:
        """Per-account resume probe. When the account reopens, mark it available."""
        interval = self._config.probe_interval_secs

        if acct.resets_at:
            wait_secs = max(0, (acct.resets_at - datetime.now(UTC)).total_seconds())
            if wait_secs > 0:
                logger.info(f'Account {acct.name}: sleeping {wait_secs:.0f}s until expected reset')
                try:
                    await asyncio.sleep(wait_secs)
                except asyncio.CancelledError:
                    return

        while True:
            usage = await self._query_usage_api(token=acct.token)
            if usage is not None:
                acct.api_fail_count = 0
                still_capped = False
                for tier_name, tier_data in usage.items():
                    if not isinstance(tier_data, dict):
                        continue
                    utilization = tier_data.get('utilization', 0)
                    if utilization >= self._config.pause_threshold:
                        still_capped = True
                        resets_at_str = tier_data.get('resets_at')
                        if resets_at_str:
                            acct.resets_at = _parse_iso_timestamp(resets_at_str)
                        logger.info(
                            f'Account {acct.name} probe: {tier_name} still at {utilization:.0%}'
                        )
                        break

                if not still_capped:
                    acct.capped = False
                    if acct.pause_started_at:
                        pause_duration = (datetime.now(UTC) - acct.pause_started_at).total_seconds()
                        self._total_pause_secs += pause_duration
                    acct.pause_started_at = None
                    logger.info(f'Account {acct.name} RESUMED')
                    # Reopen global gate so blocked before_invoke() callers wake up
                    self._open.set()
                    return
            else:
                acct.api_fail_count += 1
                past_reset = (
                    acct.resets_at is not None
                    and datetime.now(UTC) > acct.resets_at
                )
                if past_reset and acct.api_fail_count >= _TENTATIVE_UNCAP_THRESHOLD:
                    logger.warning(
                        f'Account {acct.name}: usage API unavailable after '
                        f'expected reset — tentatively uncapping'
                    )
                    acct.capped = False
                    acct.pause_started_at = None
                    acct.api_fail_count = 0
                    self._open.set()
                    return
                logger.warning(
                    f'Account {acct.name}: usage API unavailable — '
                    f'keeping capped (will retry in {interval}s)'
                )

            try:
                logger.info(f'Account {acct.name}: still capped, retrying in {interval}s')
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            interval = min(interval * 2, self._config.max_probe_interval_secs)

    async def _query_usage_api(self, token: str | None = None) -> dict[str, Any] | None:
        """GET /api/oauth/usage with the given OAuth token."""
        if not token:
            logger.debug('No OAuth token found — cannot query usage API')
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    USAGE_API_URL,
                    headers={
                        'Authorization': f'Bearer {token}',
                        'Content-Type': 'application/json',
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f'Usage API returned {resp.status_code}: {resp.text[:200]}')
                return None
        except Exception as e:
            logger.warning(f'Usage API request failed: {e}')
            return None

    async def shutdown(self) -> None:
        """Cancel all resume probe tasks."""
        for acct in self._accounts:
            if acct.resume_task and not acct.resume_task.done():
                acct.resume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await acct.resume_task
                acct.resume_task = None

    @property
    def is_paused(self) -> bool:
        if not self._accounts:
            return False
        return all(a.capped for a in self._accounts)

    @property
    def paused_reason(self) -> str:
        return self._paused_reason

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def total_pause_secs(self) -> float:
        if self._pause_started_at:
            return self._total_pause_secs + (
                datetime.now(UTC) - self._pause_started_at
            ).total_seconds()
        return self._total_pause_secs

    @property
    def active_account_name(self) -> str | None:
        """Name of the first non-capped account, or None."""
        for acct in self._accounts:
            if not acct.capped:
                return acct.name
        return None


# --- Helpers ---


def _read_oauth_token() -> str | None:
    """Read the OAuth access token from ~/.claude/.credentials.json."""
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
        # The credentials file may have different structures
        # Try common patterns
        if isinstance(data, dict):
            # Direct token field
            if 'accessToken' in data:
                return data['accessToken']
            if 'access_token' in data:
                return data['access_token']
            # Nested under a provider key
            for _key, val in data.items():
                if isinstance(val, dict):
                    if 'accessToken' in val:
                        return val['accessToken']
                    if 'access_token' in val:
                        return val['access_token']
        return None
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.debug(f'Cannot read OAuth credentials: {e}')
        return None


def _parse_resets_at(text: str) -> datetime:
    """Parse reset time from cap-hit message text.

    Handles:
    - "resets in 3h" / "resets in 45m" / "resets in 2d"
    - "resets 9pm (Europe/London)" / "resets 3:00 AM (US/Pacific)"
    - Falls back to 1 hour from now
    """
    # Relative: "resets in Xh", "resets in Xm", "resets in Xd"
    m = re.search(r'resets\s+in\s+(\d+)\s*([hmd])', text, re.IGNORECASE)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta = {
            'h': timedelta(hours=amount),
            'm': timedelta(minutes=amount),
            'd': timedelta(days=amount),
        }.get(unit, timedelta(hours=1))
        return datetime.now(UTC) + delta

    # Absolute: "resets Xpm (TZ)" or "resets X:XX AM (TZ)"
    m = re.search(
        r'resets\s+(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            time_str = m.group(1).strip()
            tz_str = m.group(2).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            # Try parsing with different formats
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                return datetime.now(UTC) + timedelta(hours=1)

            now_in_tz = datetime.now(tz)
            target = now_in_tz.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            if target <= now_in_tz:
                target += timedelta(days=1)
            return target.astimezone(UTC)
        except Exception:
            pass

    # Fallback: 1 hour from now
    return datetime.now(UTC) + timedelta(hours=1)


def _parse_iso_timestamp(ts: str) -> datetime | None:
    """Parse an ISO 8601 timestamp string."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def _extract_cap_message(text: str, prefix: str) -> str:
    """Extract the full sentence containing the cap-hit prefix."""
    lower = text.lower()
    idx = lower.find(prefix.lower())
    if idx == -1:
        return ''
    # Find the end of the sentence
    end = text.find('\n', idx)
    if end == -1:
        end = min(idx + 200, len(text))
    return text[idx:end].strip()
