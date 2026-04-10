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

from shared.config_dir import TaskConfigDir
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
    "You're out of extra",
]
# Secondary confirmation — must also appear in the same text
CAP_CONFIRM_KEYWORDS = ["resets", "usage limit", "upgrade"]

# Patterns for near-cap warnings (pause proactively)
NEAR_CAP_PREFIXES = [
    "You're close to",
    "You're now using extra",
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
    near_cap: bool = False
    # Probe lifecycle:
    #   probing=True  → freshly uncapped by probe, first task should claim it
    #   probe_in_flight=True → one task is testing, others must wait
    probing: bool = False
    probe_in_flight: bool = False


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
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

        self._probe_config_dir = TaskConfigDir('usage-gate-probe')
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
            async with self._lock:
                for acct in self._accounts:
                    if acct.capped or acct.probe_in_flight:
                        continue
                    if acct.probing:
                        # First task claims the probe slot — others block
                        # until confirm_account_ok() or _handle_cap_detected().
                        acct.probing = False
                        acct.probe_in_flight = True
                        self._open.clear()
                        logger.info(
                            f'Account {acct.name}: probe slot claimed — '
                            f'single task testing',
                        )
                    logger.debug(f'Using account {acct.name}')
                    # Failover detection: emit event if account changed.
                    # Update _last_account_name FIRST to close the race window,
                    # then fire the event non-blocking (fire-and-forget).
                    if (
                        self._last_account_name is not None
                        and self._last_account_name != acct.name
                    ):
                        old_name = self._last_account_name
                        self._last_account_name = acct.name
                        if self._cost_store:
                            self._fire_cost_event(
                                acct.name,
                                'failover',
                                json.dumps({'from': old_name, 'to': acct.name}),
                            )
                    else:
                        self._last_account_name = acct.name
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

        for prefix in CAP_HIT_PREFIXES:
            if prefix.lower() in combined.lower():
                resets_at = _parse_resets_at(combined)
                reason = _extract_cap_message(combined, prefix) or f'Cap detected: {prefix}'
                self._handle_cap_detected(reason, resets_at, oauth_token)
                return True

        for prefix in NEAR_CAP_PREFIXES:
            if prefix.lower() in combined.lower():
                reason = _extract_cap_message(combined, prefix) or f'Near-cap warning: {prefix}'
                self._handle_near_cap_warning(reason, oauth_token)
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
        acct.near_cap = False
        acct.probing = False
        acct.probe_in_flight = False
        acct.resets_at = resets_at
        if acct.pause_started_at is None:
            acct.pause_started_at = datetime.now(UTC)
        logger.warning(f'Account {acct.name} CAPPED: {reason}')
        if self._cost_store:
            self._fire_cost_event(acct.name, 'cap_hit', json.dumps({'reason': reason}))
        self._start_account_resume_probe(acct)

        # If all accounts are now capped, close the global gate
        if all(a.capped for a in self._accounts):
            self._open.clear()
            self._paused_reason = f'All accounts capped (last: {reason})'
            if self._pause_started_at is None:
                self._pause_started_at = datetime.now(UTC)
            logger.warning(f'Usage gate PAUSED: {self._paused_reason}')

    def _handle_near_cap_warning(
        self,
        reason: str,
        oauth_token: str | None,
    ) -> None:
        """Record a near-cap warning without blocking the account."""
        acct = self._find_account_by_token(oauth_token) if oauth_token else None
        if acct is None:
            for a in self._accounts:
                if not a.capped:
                    acct = a
                    break
        if acct is None:
            logger.warning(f'Near-cap warning but no matching account: {reason}')
            return

        acct.near_cap = True
        logger.warning(f'Account {acct.name} NEAR CAP: {reason}')
        if self._cost_store:
            self._fire_cost_event(acct.name, 'near_cap', json.dumps({'reason': reason}))

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
        except RuntimeError:
            logger.warning(
                'No running event loop for cost event %s/%s', event_type, account_name
            )
            return
        coro = self._write_cost_event(account_name, event_type, details)
        try:
            task = loop.create_task(
                coro,
                name=f'cost-event-{event_type}-{account_name}',
            )
        except RuntimeError as exc:
            coro.close()
            logger.warning(
                'Failed to schedule cost event %s/%s: %s', event_type, account_name, exc
            )
            return
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _refresh_capped_accounts(self) -> bool:
        """Check reset times for all capped accounts. Return True if any uncapped."""
        now = datetime.now(UTC)
        any_uncapped = False
        for acct in self._accounts:
            if not acct.capped:
                continue
            if acct.resets_at is not None and now >= acct.resets_at:
                logger.info(f'Account {acct.name}: reset time passed — uncapping (probing)')
                acct.capped = False
                acct.near_cap = False
                acct.probing = True  # gate: one task confirms before opening to all
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
        """Repeatedly probe an account until it uncaps.

        Uses exponential backoff: ``probe_interval_secs * 2^probe_count``,
        capped at ``max_probe_interval_secs``.  The sleep duration is also
        bounded by the time remaining until ``resets_at``.

        Fires a minimal Claude invocation (haiku, 1 turn) to verify the
        account actually has capacity.  Only uncaps the account on success.
        """
        while acct.capped:
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

            acct.probe_count += 1
            logger.info(
                f'Account {acct.name}: firing probe #{acct.probe_count}',
            )

            ok = await self._run_probe(acct)

            if ok:
                acct.capped = False
                acct.probing = True  # gate: let one real task confirm first
                acct.probe_count = 0
                if acct.pause_started_at:
                    self._total_pause_secs += (
                        datetime.now(UTC) - acct.pause_started_at
                    ).total_seconds()
                acct.pause_started_at = None
                logger.info(f'Account {acct.name} RESUMED (probe confirmed)')
                self._open.set()
                if self._cost_store:
                    await self._write_cost_event(
                        acct.name, 'resumed',
                        json.dumps({'label': f'probe #{acct.probe_count} confirmed'}),
                    )
                return
            else:
                logger.info(
                    f'Account {acct.name}: probe #{acct.probe_count} failed — '
                    f'retrying after backoff',
                )

    async def _run_probe(self, acct: AccountState) -> bool:
        """Fire a minimal Claude invocation to test if *acct* has capacity.

        Returns ``True`` if the invocation succeeded (no cap hit), ``False``
        otherwise.  Uses haiku to minimise cost (~$0.001 per probe).
        """
        _PROBE_TIMEOUT = 30

        self._probe_config_dir.write_credentials(acct.token)

        cmd = [
            'claude', '--print', '--output-format', 'json',
            '--model', 'haiku',
            '--max-turns', '1',
            '--max-budget-usd', '0.01',
            '--permission-mode', 'bypassPermissions',
            '--', 'Say ok',
        ]

        env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
        env['CLAUDE_CODE_OAUTH_TOKEN'] = acct.token
        env['CLAUDE_CONFIG_DIR'] = str(self._probe_config_dir.path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_PROBE_TIMEOUT,
            )
        except (TimeoutError, asyncio.CancelledError):
            logger.warning(f'Account {acct.name}: probe timed out / cancelled')
            return False
        except Exception as exc:
            logger.warning(f'Account {acct.name}: probe error: {exc}')
            return False

        combined = (
            (stderr_bytes.decode(errors='replace') if stderr_bytes else '')
            + '\n'
            + (stdout_bytes.decode(errors='replace') if stdout_bytes else '')
        )

        for prefixes in (CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES):
            for prefix in prefixes:
                if prefix.lower() in combined.lower():
                    logger.info(
                        f'Account {acct.name}: probe got cap message: {prefix}',
                    )
                    return False

        if proc.returncode != 0:
            logger.warning(
                f'Account {acct.name}: probe exited {proc.returncode}',
            )
            return False

        logger.info(f'Account {acct.name}: probe succeeded')
        return True

    async def shutdown(self) -> None:
        """Cancel all resume probe tasks and drain in-flight background cost-event tasks."""
        for acct in self._accounts:
            if acct.resume_task and not acct.resume_task.done():
                acct.resume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await acct.resume_task
                acct.resume_task = None

        for task in list(self._background_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._probe_config_dir.cleanup()

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
    def account_count(self) -> int:
        """Number of configured accounts."""
        return len(self._accounts)

    @property
    def active_account_name(self) -> str | None:
        """Name of the first non-capped account, or None."""
        for acct in self._accounts:
            if not acct.capped:
                return acct.name
        return None

    def confirm_account_ok(self, oauth_token: str | None) -> None:
        """Clear the probing gate after a successful invocation.

        Called by ``invoke_with_cap_retry`` when an invocation succeeds
        (no cap detected). Allows other tasks to use this account.
        """
        acct = self._find_account_by_token(oauth_token) if oauth_token else None
        if acct is None:
            return
        # A successful invocation proves the account is healthy — clear stale
        # near_cap regardless of whether a probe cycle was in progress.
        acct.near_cap = False
        if acct.probe_in_flight:
            acct.probe_in_flight = False
            acct.probe_count = 0
            logger.info(f'Account {acct.name}: probe confirmed OK — opening to all tasks')
            self._open.set()

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
