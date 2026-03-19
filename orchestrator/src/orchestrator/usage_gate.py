"""Usage cap detection, pause gate, and auto-resume for OAuth-based Claude agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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


class SessionBudgetExhausted(Exception):
    """Raised when the per-run session budget is exceeded."""

    def __init__(self, cumulative_cost: float):
        self.cumulative_cost = cumulative_cost
        super().__init__(f'Session budget exhausted: ${cumulative_cost:.2f} spent')


class UsageGate:
    """Shared gate that pauses all agent invocations when a usage cap is hit.

    The gate is an asyncio.Event: set = open (agents can proceed),
    clear = paused (agents block in before_invoke).
    """

    def __init__(self, config):
        from orchestrator.config import UsageCapConfig
        self._config: UsageCapConfig = config
        self._open = asyncio.Event()
        self._open.set()  # start open
        self._lock = asyncio.Lock()
        self._cumulative_cost: float = 0.0
        self._resets_at: datetime | None = None
        self._paused_reason: str = ''
        self._resume_task: asyncio.Task | None = None
        self._pause_started_at: datetime | None = None
        self._total_pause_secs: float = 0.0

    async def check_at_startup(self) -> None:
        """One API call to /api/oauth/usage. If already over threshold, pause immediately."""
        usage = await self._query_usage_api()
        if usage is None:
            logger.info('Usage API unavailable at startup — proceeding without cap check')
            return

        # Check all cap tiers
        for tier_name, tier_data in usage.items():
            if not isinstance(tier_data, dict):
                continue
            utilization = tier_data.get('utilization', 0)
            resets_at_str = tier_data.get('resets_at')

            logger.info(f'Usage tier {tier_name}: {utilization:.0%} utilized'
                        + (f', resets at {resets_at_str}' if resets_at_str else ''))

            if utilization >= self._config.pause_threshold:
                resets_at = _parse_iso_timestamp(resets_at_str) if resets_at_str else None
                await self._pause(
                    f'Startup check: {tier_name} at {utilization:.0%} (threshold {self._config.pause_threshold:.0%})',
                    resets_at,
                )
                return  # one pause is enough

    async def before_invoke(self) -> None:
        """Block until gate is open. Check session budget."""
        await self._open.wait()
        if (self._config.session_budget_usd is not None
                and self._cumulative_cost >= self._config.session_budget_usd):
            raise SessionBudgetExhausted(self._cumulative_cost)

    def detect_cap_hit(self, stderr: str, result_text: str, backend: str = 'claude') -> bool:
        """Scan stderr and result text for cap-hit patterns.

        If detected, closes the gate synchronously and returns True.
        Callers should loop back to ``before_invoke()`` which will block
        until the gate reopens. The resume probe is started asynchronously.
        """
        combined = f'{stderr}\n{result_text}'

        # Check backend-specific patterns first
        if backend == 'codex':
            for pattern in CODEX_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    self._open.clear()
                    self._paused_reason = f'Codex cap hit: {pattern}'
                    if self._pause_started_at is None:
                        self._pause_started_at = datetime.now(UTC)
                    logger.warning(f'Usage gate PAUSED: Codex cap — {pattern}')
                    return True
        elif backend == 'gemini':
            for pattern in GEMINI_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    self._open.clear()
                    self._paused_reason = f'Gemini cap hit: {pattern}'
                    if self._pause_started_at is None:
                        self._pause_started_at = datetime.now(UTC)
                    logger.warning(f'Usage gate PAUSED: Gemini cap — {pattern}')
                    return True

        for prefixes in (CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES):
            for prefix in prefixes:
                if prefix.lower() in combined.lower():
                    resets_at = _parse_resets_at(combined)
                    reason = _extract_cap_message(combined, prefix) or f'Cap detected: {prefix}'
                    # Close gate synchronously (Event.clear is safe without a loop)
                    self._open.clear()
                    self._paused_reason = reason
                    self._resets_at = resets_at
                    if self._pause_started_at is None:
                        self._pause_started_at = datetime.now(UTC)
                    logger.warning(f'Usage gate PAUSED: {reason}')
                    # Start resume probe if configured (requires running loop)
                    if self._config.wait_for_reset:
                        try:
                            loop = asyncio.get_running_loop()
                            if self._resume_task is None or self._resume_task.done():
                                self._resume_task = loop.create_task(
                                    self._resume_probe_loop(),
                                    name='usage-gate-resume-probe',
                                )
                        except RuntimeError:
                            pass  # no running loop — caller will handle
                    return True

        return False

    def on_agent_complete(self, cost: float) -> None:
        """Accumulate cost for session budget tracking."""
        self._cumulative_cost += cost

    async def _pause(self, reason: str, resets_at: datetime | None) -> None:
        """Close gate, optionally start resume probe task."""
        async with self._lock:
            if not self._open.is_set():
                return  # already paused

            self._open.clear()
            self._paused_reason = reason
            self._resets_at = resets_at
            self._pause_started_at = datetime.now(UTC)
            logger.warning(f'Usage gate PAUSED: {reason}')
            if resets_at:
                logger.info(f'Expected reset at: {resets_at.isoformat()}')

            if self._config.wait_for_reset:
                # Cancel any existing probe task
                if self._resume_task and not self._resume_task.done():
                    self._resume_task.cancel()
                self._resume_task = asyncio.create_task(
                    self._resume_probe_loop(),
                    name='usage-gate-resume-probe',
                )

    async def _resume_probe_loop(self) -> None:
        """Sleep until resets_at, call /api/oauth/usage once, reopen if under threshold."""
        interval = self._config.probe_interval_secs

        # If we know when the cap resets, sleep until then first
        if self._resets_at:
            wait_secs = max(0, (self._resets_at - datetime.now(UTC)).total_seconds())
            if wait_secs > 0:
                logger.info(f'Usage gate: sleeping {wait_secs:.0f}s until expected reset')
                try:
                    await asyncio.sleep(wait_secs)
                except asyncio.CancelledError:
                    return

        # Probe loop with exponential backoff
        while True:
            usage = await self._query_usage_api()
            if usage is not None:
                # Check if any tier is still over threshold
                still_capped = False
                for tier_name, tier_data in usage.items():
                    if not isinstance(tier_data, dict):
                        continue
                    utilization = tier_data.get('utilization', 0)
                    if utilization >= self._config.pause_threshold:
                        still_capped = True
                        # Update resets_at from fresh data
                        resets_at_str = tier_data.get('resets_at')
                        if resets_at_str:
                            self._resets_at = _parse_iso_timestamp(resets_at_str)
                        logger.info(f'Usage gate probe: {tier_name} still at {utilization:.0%}')
                        break

                if not still_capped:
                    await self._resume()
                    return
            else:
                # API unavailable — optimistically resume
                logger.warning('Usage API unavailable during probe — resuming optimistically')
                await self._resume()
                return

            # Exponential backoff: interval doubles each time, capped at max
            try:
                logger.info(f'Usage gate: still capped, retrying in {interval}s')
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            interval = min(interval * 2, self._config.max_probe_interval_secs)

    async def _resume(self) -> None:
        """Reopen the gate."""
        async with self._lock:
            if self._pause_started_at:
                pause_duration = (datetime.now(UTC) - self._pause_started_at).total_seconds()
                self._total_pause_secs += pause_duration
            self._open.set()
            self._paused_reason = ''
            self._pause_started_at = None
            logger.info('Usage gate RESUMED')

    async def _query_usage_api(self) -> dict[str, Any] | None:
        """GET /api/oauth/usage with OAuth token from ~/.claude/.credentials.json.

        Returns the usage dict or None on error.
        """
        token = _read_oauth_token()
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
        """Cancel resume probe task if running."""
        if self._resume_task and not self._resume_task.done():
            self._resume_task.cancel()
            try:
                await self._resume_task
            except asyncio.CancelledError:
                pass
            self._resume_task = None

    @property
    def is_paused(self) -> bool:
        return not self._open.is_set()

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


def _parse_resets_at(text: str) -> datetime | None:
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
