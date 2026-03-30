"""Usage cap detection, pause gate, and auto-resume for OAuth-based Claude agents.

Supports multiple Claude Max accounts for failover: when one account hits its
usage cap, the gate returns the next available account's token. Only blocks
when *all* accounts are capped.

Cap detection is reactive (stderr pattern matching). Resume is timer-based:
when an account is capped, we sleep until the parsed ``resets_at`` time, then
uncap. If the uncap is premature, the retry loop in ``invoke_with_cap_retry``
re-detects the cap on the next invocation.

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
from typing import TYPE_CHECKING

from shared.config_models import UsageCapConfig

if TYPE_CHECKING:
    from shared.cost_store import CostStore

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


@dataclass
class AccountState:
    """Per-account cap tracking."""

    name: str
    token: str | None          # None = default account (no override)
    capped: bool = False
    resets_at: datetime | None = None
    pause_started_at: datetime | None = None
    resume_task: asyncio.Task | None = field(default=None, repr=False)
    probe_count: int = 0


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

    def __init__(self, config: UsageCapConfig, *, cost_store: CostStore | None = None):
        self._config: UsageCapConfig = config
        self._open = asyncio.Event()
        self._open.set()  # start open
        self._lock = asyncio.Lock()
        self._cumulative_cost: float = 0.0
        self._paused_reason: str = ''
        self._pause_started_at: datetime | None = None
        self._total_pause_secs: float = 0.0
        self._cost_store: CostStore | None = cost_store
        self._project_id: str | None = None
        self._run_id: str | None = None
        self._last_account_name: str | None = None

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
        """No-op: pre-existing caps are detected reactively on first invocation.

        The usage API (claude.ai/api/oauth/usage) is no longer available.
        If an account is already capped, the first invocation attempt will
        detect it via stderr pattern matching in ``detect_cap_hit()``.
        """
        logger.info(
            'Usage gate startup: %d account(s) configured — '
            'caps will be detected reactively',
            len(self._accounts),
        )

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

            # All capped — check if any reset times have passed before blocking.
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
        self._fire_cost_event(acct.name, 'cap_hit', json.dumps({'reason': reason}))
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

    async def _write_cost_event(
        self,
        account_name: str,
        event_type: str,
        details: str,
    ) -> None:
        """Write a cost event to CostStore. Silently swallows errors (telemetry only)."""
        if self._cost_store is None:
            return
        try:
            await self._cost_store.save_account_event(
                account_name=account_name,
                event_type=event_type,
                project_id=self._project_id,
                run_id=self._run_id,
                details=details,
                created_at=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:
            logger.warning('CostStore write failed for %s/%s: %s', account_name, event_type, exc)

    def _fire_cost_event(
        self,
        account_name: str,
        event_type: str,
        details: str,
    ) -> None:
        """Fire-and-forget wrapper for _write_cost_event (for use in sync contexts)."""
        if self._cost_store is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._write_cost_event(account_name, event_type, details),
                name=f'cost-event-{event_type}-{account_name}',
            )
        except RuntimeError:
            pass

    async def _refresh_capped_accounts(self) -> bool:
        """Check reset times for all capped accounts. Return True if any uncapped."""
        now = datetime.now(UTC)
        any_uncapped = False
        for acct in self._accounts:
            if not acct.capped:
                continue
            if acct.resets_at is not None and now >= acct.resets_at:
                logger.info(f'Account {acct.name}: reset time passed — uncapping')
                acct.capped = False
                if acct.pause_started_at:
                    self._total_pause_secs += (now - acct.pause_started_at).total_seconds()
                acct.pause_started_at = None
                any_uncapped = True

        if any_uncapped:
            self._open.set()
        return any_uncapped

    def on_agent_complete(self, cost: float) -> None:
        """Accumulate cost for session budget tracking."""
        self._cumulative_cost += cost

    async def _account_resume_probe_loop(self, acct: AccountState) -> None:
        """Sleep for a probe interval, then optimistically uncap the account.

        Uses exponential backoff: ``probe_interval_secs * 2^probe_count``,
        capped at ``max_probe_interval_secs``.  The sleep duration is also
        bounded by the time remaining until ``resets_at``.

        If the uncap is premature, ``invoke_with_cap_retry`` will re-detect
        the cap on the next invocation and ``_handle_cap_detected`` will
        start a new probe with an incremented ``probe_count``.
        """
        target = acct.resets_at
        if target is None:
            target = datetime.now(UTC) + timedelta(hours=1)
            logger.warning(f'Account {acct.name}: no resets_at — defaulting to 1h')

        base = self._config.probe_interval_secs
        ceiling = self._config.max_probe_interval_secs
        interval = min(base * (2 ** acct.probe_count), ceiling)

        remaining = max(0, (target - datetime.now(UTC)).total_seconds())
        sleep_for = min(interval, remaining) if remaining > 0 else 0

        if sleep_for > 0:
            logger.info(
                f'Account {acct.name}: sleeping {sleep_for:.0f}s '
                f'(probe #{acct.probe_count + 1}, resets in {remaining:.0f}s)',
            )
            try:
                await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                return

        # _refresh_capped_accounts may have already uncapped this account
        if not acct.capped:
            return

        past_reset = datetime.now(UTC) >= target
        if past_reset:
            acct.probe_count = 0
        else:
            acct.probe_count += 1

        acct.capped = False
        if acct.pause_started_at:
            self._total_pause_secs += (
                datetime.now(UTC) - acct.pause_started_at
            ).total_seconds()
        acct.pause_started_at = None

        label = 'reset time passed' if past_reset else f'optimistic probe #{acct.probe_count}'
        logger.info(f'Account {acct.name} RESUMED ({label})')
        self._open.set()
        await self._write_cost_event(acct.name, 'resumed', json.dumps({'label': label}))

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

    @property
    def project_id(self) -> str | None:
        """Project identifier set by the harness at run start."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str | None) -> None:
        self._project_id = value

    @property
    def run_id(self) -> str | None:
        """Run identifier set by the harness at run start."""
        return self._run_id

    @run_id.setter
    def run_id(self, value: str | None) -> None:
        self._run_id = value


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


_MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def _parse_resets_at(text: str) -> datetime:
    """Parse reset time from cap-hit message text.

    Handles:
    - "resets in 3h" / "resets in 45m" / "resets in 2d"
    - "resets Mar 30, 6pm (Europe/London)" (date + time + tz)
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

    # Absolute with date: "resets Mar 30, 6pm (Europe/London)"
    m = re.search(
        r'resets\s+([A-Za-z]{3})\s+(\d{1,2}),?\s+'
        r'(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            month_str = m.group(1).lower()
            day = int(m.group(2))
            time_str = m.group(3).strip()
            tz_str = m.group(4).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            month = _MONTH_ABBR.get(month_str)
            if month is None:
                raise ValueError(f'Unknown month: {month_str}')
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f'Cannot parse time: {time_str}')
            now_in_tz = datetime.now(tz)
            year = now_in_tz.year
            target = now_in_tz.replace(
                year=year, month=month, day=day,
                hour=parsed_time.hour, minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            # If target is in the past, assume next year
            if target <= now_in_tz:
                target = target.replace(year=year + 1)
            return target.astimezone(UTC)
        except Exception:
            pass

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
